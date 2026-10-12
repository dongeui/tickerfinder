"""한국투자증권 KIS Developers REST 연동 (모의투자 / 실전).

설정은 프로젝트 루트 `.env` (git 제외):
    KIS_MODE=paper            # paper(모의) | real(실전)
    KIS_APP_KEY=...
    KIS_APP_SECRET=...
    KIS_ACCOUNT=12345678-01   # 계좌번호 8자리-상품코드 2자리
    KIS_PHONE=01012345678     # 해외주식 주문 시 CTAC_TLNO (선택)

tr_id 는 공식 예제(koreainvestment/open-trading-api, 2026) 기준. 모의투자는 국내·미국 지정가만 지원.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests

from .universe import DATA_DIR

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT / ".env"
TOKEN_FILE = DATA_DIR / "kis_token.json"
EXCH_FILE = DATA_DIR / "kis_exchange.json"  # 미국 티커 → 거래소 코드 캐시
ORDER_LOG = DATA_DIR / "orders.jsonl"

BASE = {"paper": "https://openapivts.koreainvestment.com:29443", "real": "https://openapi.koreainvestment.com:9443"}
# (실전, 모의)
TR = {
    "kr_buy": ("TTTC0012U", "VTTC0012U"), "kr_sell": ("TTTC0011U", "VTTC0011U"),
    "kr_balance": ("TTTC8434R", "VTTC8434R"), "kr_price": ("FHKST01010100", "FHKST01010100"),
    "us_buy": ("TTTT1002U", "VTTT1002U"), "us_sell": ("TTTT1006U", "VTTT1001U"),
    "us_balance": ("TTTS3012R", "VTTS3012R"), "us_price": ("HHDFS00000300", "HHDFS00000300"),
    "us_psamount": ("TTTS3007R", "VTTS3007R"),
    "kr_trades": ("TTTC8001R", "VTTC8001R"),  # 국내 일별 주문체결 (3개월 이내)
    "us_trades": ("TTTS3035R", "VTTS3035R"),  # 해외 주문체결내역
}
# 주문용 거래소 코드 ↔ 시세용 거래소 코드
EXCG = {"NASD": "NAS", "NYSE": "NYS", "AMEX": "AMS"}
_lock = threading.Lock()


def load_env() -> dict:
    env = {}
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                v = v.split(" #", 1)[0].split("\t#", 1)[0].strip().strip('"').strip("'")  # 줄 끝 주석 제거
                env[k.strip()] = v
    for k in ("KIS_MODE", "KIS_APP_KEY", "KIS_APP_SECRET", "KIS_ACCOUNT", "KIS_PHONE"):
        if os.environ.get(k):
            env[k] = os.environ[k]
    env.setdefault("KIS_MODE", "paper")
    return env


class KIS:
    def __init__(self):
        self.env = load_env()
        self.mode = "real" if self.env.get("KIS_MODE") == "real" else "paper"
        self.base = BASE[self.mode]
        acct = self.env.get("KIS_ACCOUNT", "")
        self.cano, _, self.prdt = acct.partition("-")
        self.prdt = self.prdt or "01"
        self._token: str | None = None
        self._token_exp: datetime | None = None
        self._last_call = 0.0

    @property
    def configured(self) -> bool:
        return bool(self.env.get("KIS_APP_KEY") and self.env.get("KIS_APP_SECRET") and len(self.cano) == 8)

    def tr(self, key: str) -> str:
        return TR[key][0 if self.mode == "real" else 1]

    # ---------- 토큰 ----------
    def token(self) -> str:
        with _lock:
            if self._token and self._token_exp and datetime.now() < self._token_exp - timedelta(minutes=10):
                return self._token
            if TOKEN_FILE.exists():
                try:
                    t = json.loads(TOKEN_FILE.read_text())
                    exp = datetime.fromisoformat(t["exp"])
                    if t.get("mode") == self.mode and t.get("key") == self.env["KIS_APP_KEY"][:8] and datetime.now() < exp - timedelta(minutes=10):
                        self._token, self._token_exp = t["token"], exp
                        return self._token
                except Exception:  # noqa: BLE001
                    pass
            r = requests.post(f"{self.base}/oauth2/tokenP", json={
                "grant_type": "client_credentials", "appkey": self.env["KIS_APP_KEY"], "appsecret": self.env["KIS_APP_SECRET"]}, timeout=20)
            j = r.json()
            if "access_token" not in j:
                raise RuntimeError(f"token failed: {j.get('error_description') or j}")
            self._token = j["access_token"]
            # access_token_token_expired: "2026-10-13 16:30:00"
            try:
                self._token_exp = datetime.strptime(j["access_token_token_expired"], "%Y-%m-%d %H:%M:%S")
            except Exception:  # noqa: BLE001
                self._token_exp = datetime.now() + timedelta(hours=23)
            TOKEN_FILE.write_text(json.dumps({"token": self._token, "exp": self._token_exp.isoformat(), "mode": self.mode,
                                              "key": self.env["KIS_APP_KEY"][:8]}))
            return self._token

    def _headers(self, tr_id: str) -> dict:
        return {"content-type": "application/json; charset=utf-8", "authorization": f"Bearer {self.token()}",
                "appkey": self.env["KIS_APP_KEY"], "appsecret": self.env["KIS_APP_SECRET"], "tr_id": tr_id, "custtype": "P"}

    def _throttle(self):
        # 모의투자는 초당 요청 제한이 빡빡하므로 0.5초 간격
        gap = 0.5 if self.mode == "paper" else 0.25
        wait = self._last_call + gap - time.time()
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.time()

    def get(self, path: str, tr_key: str, params: dict, tr_cont: str = "") -> dict:
        for i in range(3):  # 조회는 "초당 거래건수 초과" 시 잠깐 쉬고 재시도 (주문 POST 는 재시도하지 않음)
            self._throttle()
            h = self._headers(self.tr(tr_key))
            if tr_cont:
                h["tr_cont"] = tr_cont
            r = requests.get(f"{self.base}{path}", headers=h, params=params, timeout=20)
            j = r.json()
            if j.get("rt_cd") in ("0", 0):
                j["_tr_cont"] = r.headers.get("tr_cont", "")
                return j
            msg = str(j.get("msg1") or j)
            if "초당" in msg and i < 2:
                time.sleep(1.0)
                continue
            raise RuntimeError(f"{tr_key}: {msg}")
        return j

    def post(self, path: str, tr_key: str, body: dict) -> dict:
        self._throttle()
        r = requests.post(f"{self.base}{path}", headers=self._headers(self.tr(tr_key)), json=body, timeout=20)
        j = r.json()
        if j.get("rt_cd") not in ("0", 0):
            raise RuntimeError(f"{tr_key}: {j.get('msg1') or j}")
        return j

    # ---------- 시세 ----------
    def price_kr(self, code: str) -> float:
        j = self.get("/uapi/domestic-stock/v1/quotations/inquire-price", "kr_price", {"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": code})
        return float(j["output"]["stck_prpr"])

    def exchange_us(self, symbol: str) -> str:
        """미국 티커의 거래소 코드(NASD/NYSE/AMEX). 시세 조회로 탐색해 캐시."""
        cache = json.loads(EXCH_FILE.read_text()) if EXCH_FILE.exists() else {}
        if symbol in cache:
            return cache[symbol]
        for excg, excd in EXCG.items():
            try:
                j = self.get("/uapi/overseas-price/v1/quotations/price", "us_price", {"AUTH": "", "EXCD": excd, "SYMB": symbol})
                if j.get("output", {}).get("last") not in (None, "", "0"):
                    cache[symbol] = excg
                    EXCH_FILE.write_text(json.dumps(cache))
                    return excg
            except Exception:  # noqa: BLE001
                continue
        raise RuntimeError(f"exchange not found for {symbol}")

    def price_us(self, symbol: str) -> tuple[float, str]:
        excg = self.exchange_us(symbol)
        j = self.get("/uapi/overseas-price/v1/quotations/price", "us_price", {"AUTH": "", "EXCD": EXCG[excg], "SYMB": symbol})
        return float(j["output"]["last"]), excg

    # ---------- 잔고 ----------
    def balance_kr(self) -> dict:
        params = {"CANO": self.cano, "ACNT_PRDT_CD": self.prdt, "AFHR_FLPR_YN": "N", "OFL_YN": "", "INQR_DVSN": "02", "UNPR_DVSN": "01",
                  "FUND_STTL_ICLD_YN": "N", "FNCG_AMT_AUTO_RDPT_YN": "N", "PRCS_DVSN": "00", "CTX_AREA_FK100": "", "CTX_AREA_NK100": ""}
        j = self.get("/uapi/domestic-stock/v1/trading/inquire-balance", "kr_balance", params)
        hold = [{"ticker": h["pdno"], "name": h["prdt_name"], "qty": int(float(h["hldg_qty"])), "avg": float(h["pchs_avg_pric"]),
                 "price": float(h["prpr"]), "value": float(h["evlu_amt"]), "pnl": float(h["evlu_pfls_amt"]), "pnl_pct": float(h["evlu_pfls_rt"]), "region": "KR"}
                for h in j.get("output1", []) if int(float(h.get("hldg_qty", 0))) > 0]
        o2 = (j.get("output2") or [{}])[0]
        return {"holdings": hold, "cash": float(o2.get("dnca_tot_amt", 0)), "eval_total": float(o2.get("tot_evlu_amt", 0)),
                "pnl": float(o2.get("evlu_pfls_smtl_amt", 0)), "currency": "KRW"}

    def balance_us(self) -> dict:
        hold = []
        for excg in ("NASD", "NYSE", "AMEX"):
            try:
                j = self.get("/uapi/overseas-stock/v1/trading/inquire-balance", "us_balance",
                             {"CANO": self.cano, "ACNT_PRDT_CD": self.prdt, "OVRS_EXCG_CD": excg, "TR_CRCY_CD": "USD", "CTX_AREA_FK200": "", "CTX_AREA_NK200": ""})
            except Exception as e:  # noqa: BLE001
                log.warning("us balance %s: %s", excg, e)
                continue
            for h in j.get("output1", []):
                q = int(float(h.get("ovrs_cblc_qty", 0)))
                if q > 0:
                    hold.append({"ticker": h["ovrs_pdno"], "name": h.get("ovrs_item_name", ""), "qty": q, "avg": float(h["pchs_avg_pric"]),
                                 "price": float(h["now_pric2"]), "value": float(h["ovrs_stck_evlu_amt"]), "pnl": float(h["frcr_evlu_pfls_amt"]),
                                 "pnl_pct": float(h["evlu_pfls_rt"]), "region": "US", "excg": excg})
        cash = None
        try:  # 주문가능 달러 (아무 종목으로나 조회)
            j = self.get("/uapi/overseas-stock/v1/trading/inquire-psamount", "us_psamount",
                         {"CANO": self.cano, "ACNT_PRDT_CD": self.prdt, "OVRS_EXCG_CD": "NASD", "OVRS_ORD_UNPR": "1", "ITEM_CD": "AAPL"})
            cash = float(j["output"]["ovrs_ord_psbl_amt"])
        except Exception as e:  # noqa: BLE001
            log.warning("us psamount: %s", e)
        return {"holdings": hold, "cash": cash, "currency": "USD"}

    # ---------- 주문/체결 내역 ----------
    @staticmethod
    def _f(v, default=0.0) -> float:
        try:
            return float(str(v).replace(",", "")) if v not in (None, "") else default
        except ValueError:
            return default

    def trades_kr(self, start: str, end: str) -> list[dict]:
        """국내 주문·체결 내역 (YYYYMMDD ~ YYYYMMDD, 3개월 이내). 미체결 포함."""
        rows, fk, nk, cont = [], "", "", ""
        for _ in range(10):
            j = self.get("/uapi/domestic-stock/v1/trading/inquire-daily-ccld", "kr_trades",
                         {"CANO": self.cano, "ACNT_PRDT_CD": self.prdt, "INQR_STRT_DT": start, "INQR_END_DT": end, "SLL_BUY_DVSN_CD": "00",
                          "INQR_DVSN": "00", "PDNO": "", "CCLD_DVSN": "00", "ORD_GNO_BRNO": "", "ODNO": "", "INQR_DVSN_3": "00", "INQR_DVSN_1": "",
                          "CTX_AREA_FK100": fk, "CTX_AREA_NK100": nk}, tr_cont=cont)
            for h in j.get("output1", []):
                oq, cq = self._f(h.get("ord_qty")), self._f(h.get("tot_ccld_qty"))
                rows.append({"region": "KR", "date": h.get("ord_dt", ""), "time": h.get("ord_tmd", ""), "ticker": h.get("pdno", ""), "name": h.get("prdt_name", ""),
                             "side": "sell" if str(h.get("sll_buy_dvsn_cd")) == "01" else "buy", "ord_qty": oq, "ord_px": self._f(h.get("ord_unpr")),
                             "filled_qty": cq, "filled_px": self._f(h.get("avg_prvs")), "amount": self._f(h.get("tot_ccld_amt")), "remaining": self._f(h.get("rmn_qty")),
                             "cancelled": str(h.get("cncl_yn", "")) == "Y", "rejected": self._f(h.get("rjct_qty")) > 0, "odno": h.get("odno", ""), "orgn_odno": h.get("orgn_odno", ""),
                             "kind": h.get("ord_dvsn_name", ""), "currency": "KRW"})
            nk, fk = j.get("ctx_area_nk100", "").strip(), j.get("ctx_area_fk100", "").strip()
            if j.get("_tr_cont") not in ("F", "M") or not nk:
                break
            cont = "N"
        return rows

    def trades_us(self, start: str, end: str) -> list[dict]:
        """해외(미국) 주문·체결 내역 (YYYYMMDD ~ YYYYMMDD). 미체결 포함."""
        rows, fk, nk, cont = [], "", "", ""
        for _ in range(10):
            j = self.get("/uapi/overseas-stock/v1/trading/inquire-ccnl", "us_trades",
                         {"CANO": self.cano, "ACNT_PRDT_CD": self.prdt, "PDNO": "", "ORD_STRT_DT": start, "ORD_END_DT": end, "SLL_BUY_DVSN": "00",
                          "CCLD_NCCS_DVSN": "00", "OVRS_EXCG_CD": "", "SORT_SQN": "DS", "ORD_DT": "", "ORD_GNO_BRNO": "", "ODNO": "",
                          "CTX_AREA_NK200": nk, "CTX_AREA_FK200": fk}, tr_cont=cont)
            for h in j.get("output", []):
                oq, cq = self._f(h.get("ft_ord_qty")), self._f(h.get("ft_ccld_qty"))
                rows.append({"region": "US", "date": h.get("ord_dt", ""), "time": h.get("ord_tmd", ""), "ticker": h.get("pdno", ""), "name": h.get("prdt_name", ""),
                             "side": "sell" if str(h.get("sll_buy_dvsn_cd")) == "01" else "buy", "ord_qty": oq, "ord_px": self._f(h.get("ft_ord_unpr3")),
                             "filled_qty": cq, "filled_px": self._f(h.get("ft_ccld_unpr3")), "amount": self._f(h.get("ft_ccld_amt3")), "remaining": self._f(h.get("nccs_qty")),
                             "cancelled": str(h.get("rvse_cncl_dvsn", "")) == "02" or "취소" in str(h.get("prcs_stat_name", "")), "rejected": bool(str(h.get("rjct_rson", "")).strip()),
                             "odno": h.get("odno", ""), "orgn_odno": h.get("orgn_odno", ""), "kind": h.get("prcs_stat_name", "") or h.get("rvse_cncl_dvsn_name", ""),
                             "excg": h.get("ovrs_excg_cd", ""), "currency": h.get("tr_crcy_cd", "USD"), "reject_reason": h.get("rjct_rson", "")})
            nk, fk = j.get("ctx_area_nk200", "").strip(), j.get("ctx_area_fk200", "").strip()
            if j.get("_tr_cont") not in ("F", "M") or not nk:
                break
            cont = "N"
        return rows

    # ---------- 주문 ----------
    def order(self, region: str, ticker: str, side: str, qty: int, price: float) -> dict:
        """지정가 주문. side: buy | sell. 반환: 주문번호 등."""
        if qty <= 0:
            raise ValueError("qty must be > 0")
        if region == "KR":
            body = {"CANO": self.cano, "ACNT_PRDT_CD": self.prdt, "PDNO": ticker, "ORD_DVSN": "00",
                    "ORD_QTY": str(int(qty)), "ORD_UNPR": str(int(round(price))), "EXCG_ID_DVSN_CD": "KRX"}
            if side == "sell":
                body["SLL_TYPE"] = "01"
            j = self.post("/uapi/domestic-stock/v1/trading/order-cash", f"kr_{side}", body)
        else:
            excg = self.exchange_us(ticker)
            body = {"CANO": self.cano, "ACNT_PRDT_CD": self.prdt, "OVRS_EXCG_CD": excg, "PDNO": ticker,
                    "ORD_QTY": str(int(qty)), "OVRS_ORD_UNPR": f"{price:.2f}", "CTAC_TLNO": self.env.get("KIS_PHONE", ""),
                    "MGCO_APTM_ODNO": "", "SLL_TYPE": "00" if side == "sell" else "", "ORD_SVR_DVSN_CD": "0", "ORD_DVSN": "00"}
            try:
                j = self.post("/uapi/overseas-stock/v1/trading/order", f"us_{side}", body)
            except requests.RequestException as e:
                rec = {"at": datetime.now().isoformat(timespec="seconds"), "mode": self.mode, "region": region, "ticker": ticker, "side": side,
                       "qty": qty, "price": price, "odno": None, "msg": f"응답 없음({type(e).__name__}) — 접수됐을 수 있으니 거래내역에서 확인 후 재주문"}
                with ORDER_LOG.open("a") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                raise RuntimeError(rec["msg"]) from e
        out = j.get("output", {})
        rec = {"at": datetime.now().isoformat(timespec="seconds"), "mode": self.mode, "region": region, "ticker": ticker, "side": side,
               "qty": qty, "price": price, "odno": out.get("ODNO"), "msg": j.get("msg1"), "fill_status": "accepted" if out.get("ODNO") else "rejected"}
        with ORDER_LOG.open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec


_client: KIS | None = None


def client() -> KIS:
    global _client
    if _client is None or _client.env != load_env():
        _client = KIS()
    return _client


def order_history(limit: int = 100) -> list[dict]:
    if not ORDER_LOG.exists():
        return []
    rows = [json.loads(l) for l in ORDER_LOG.read_text().splitlines() if l.strip()]
    return rows[-limit:][::-1]
