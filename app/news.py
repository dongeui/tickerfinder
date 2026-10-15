"""미국 증시 주요 뉴스 클립 (리포트 탭). 키 없는 공개 RSS 4개를 합쳐 최근 36시간 안의 기사 20개를 고른다.

- CNBC Markets · MarketWatch Top Stories (영문) · Google News 검색(영문: stock market/nasdaq/wall street, 국문: 뉴욕증시/미국 증시/나스닥)
- 제목 정규화로 중복 제거, 피드별로 번갈아 뽑아(round-robin) 한 매체에 쏠리지 않게 한 뒤 최신순.
- 번역·영향 정리: 로컬 `claude -p`(Claude Code CLI, 모델 claude-haiku-5-5) 한 번 호출로 새 기사들을 한꺼번에 처리
  (영문 제목 한글화 — 회사·인명·제품·티커 등 고유명사는 영문 그대로, 기사가 영향을 줄 수 있는 종목·시장 5줄 이내, 무관하면 빈칸).
  CLI 가 없거나 실패하면 구글 번역(gtx)으로 제목만 번역하되 유니버스 회사명·대문자 약어는 자리표로 보호해 원문 유지.
- 링크별 결과는 data/news_cache.json 에 캐시(한 번 처리한 기사는 다시 묻지 않음). 목록은 data/news.json 에 20분 캐시.
- 갱신은 백그라운드 스레드: 캐시가 있으면 바로 이전 목록을 주고(refreshing=true) 끝나면 다음 요청부터 새 목록.
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import threading
import time
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime

from .profile import translate_ko
from .scanner import KST, atomic_write_text
from .universe import DATA_DIR

LLM_MODEL = os.environ.get("NEWS_LLM_MODEL", "claude-haiku-5-5")
REFRESH_HOURS = float(os.environ.get("NEWS_REFRESH_HOURS", "3"))  # 리포트 탭을 안 열어도 주기적으로 수집·기록
BACKFILL_DAYS = int(os.environ.get("NEWS_BACKFILL_DAYS", "30"))    # 로그에 비어 있는 지난 날짜를 Google News 날짜 검색으로 채움
BACKFILL_MIN = 8                                                   # 하루에 이 개수 미만이면 그 날짜를 백필
BACKFILL_PER_DAY = 20
LLM_TIMEOUT = 150
TONES = ("긍정", "우려", "중립", "혼재")  # 기사가 미국 증시에 주는 함의

log = logging.getLogger(__name__)
CACHE = DATA_DIR / "news.json"
LOG = DATA_DIR / "news_log.json"  # 한 번 표시된 기사 누적(링크별): 일별·월별 요약의 원천
KO_CACHE = DATA_DIR / "news_cache.json"
TTL = timedelta(minutes=20)
MAX_AGE = timedelta(hours=36)
LIMIT = 20
POOL = 32  # LLM 에 보낼 후보 수 (상품 홍보성 기사를 걸러낸 뒤 LIMIT 개만 남김)
# 증권사·운용사 상품 기사(ETF/펀드 출시·상장, 순자산·설정액 돌파, 수상·캠페인)는 제외 — 시장 판단에 쓸모없음
PR_PATTERNS = [
    re.compile(r"(ETF|ETN|펀드|상품|리츠|랩|신탁|ELS|채권형|주식형).{0,20}(출시|신규\s*상장|상장(?!사|기업|폐지)|선보|론칭|출범|판매)"),
    re.compile(r"(출시|신규\s*상장|선보|론칭).{0,20}(ETF|ETN|펀드|상품)"),
    re.compile(r"(순자산|설정액|수탁고|운용자산|AUM|잔고|가입자|판매액|누적\s*판매).{0,16}(돌파|넘어|넘겨|달성|육박|돌파)"),
    re.compile(r"(억|조)\s*원?\s*(돌파|달성|넘어|넘겨).{0,30}(ETF|펀드|상품|순자산|설정액)|(ETF|펀드|상품|순자산|설정액).{0,30}(억|조)\s*원?\s*(돌파|달성|넘어|넘겨)"),
    re.compile(r"(자산운용|증권|투자증권|운용사|은행|보험|카드).{0,12}(출시|선정|수상|대상|1위|캠페인|이벤트|프로모션|기념|MOU|협약|업무협약|리뉴얼|오픈)"),
    re.compile(r"(개인|자금|투자자|머니).{0,8}(순매수|유입|몰렸|몰린|몰려).{0,24}(ETF|펀드|상품)|(ETF|펀드|상품).{0,24}(순매수|유입).{0,10}(최대|1위|억|조)"),
    re.compile(r"(ETF|펀드|상품|운용사).{0,12}(최대|1위|최단|최초|점유율)"),
    re.compile(r"(자산운용|운용사|운용,|증권,|KODEX|TIGER|ACE|RISE|SOL|PLUS|KIWOOM|HANARO|1Q|TIME|WON).{0,40}(억|조)\s*원?\s*(돌파|달성|넘어|넘겨|육박)"),
    re.compile(r"\[(여의도\s*단신|단신|포토|부고|인사|신상품|보도자료|이벤트)\]|보도자료|\bsponsored\b|\bpromotion\b", re.I),
    re.compile(r"\b(launch(es|ed|ing)?|debuts?|unveils?|introduc(es|ed))\b.{0,40}\b(ETFs?|funds?|ETNs?)\b|\b(ETFs?|funds?)\b.{0,30}\b(launch(es|ed)?|debuts?)\b", re.I),
    re.compile(r"\b(assets? under management|AUM|inflows?)\b.{0,40}\b(surpass|top|cross|hit|reach)", re.I),
]


def _is_product_pr(title: str, desc: str = "") -> bool:
    t = f"{title} {desc}"
    return any(p.search(t) for p in PR_PATTERNS)
UA = {"User-Agent": "Mozilla/5.0 (Macintosh) tickerfinder/1.0"}
GN_EN = 'stock market OR nasdaq OR "wall street" OR "S&P 500"'
GN_KO = "뉴욕증시 OR 미국 증시 OR 나스닥"


def _gnews_url(query: str, lang: str, after: str | None = None, before: str | None = None) -> str:
    """Google News RSS 검색. after/before(YYYY-MM-DD) 를 주면 그 날짜 범위(백필용), 아니면 최근 1일."""
    q = query + (f" after:{after} before:{before}" if after and before else " when:1d")
    loc = "&hl=en-US&gl=US&ceid=US:en" if lang == "en" else "&hl=ko&gl=KR&ceid=KR:ko"
    return "https://news.google.com/rss/search?q=" + urllib.request.quote(q) + loc


FEEDS = [
    ("CNBC", "https://www.cnbc.com/id/10000664/device/rss/rss.html", "en"),
    ("MarketWatch", "https://feeds.content.dowjones.io/public/rss/mw_topstories", "en"),
    ("Google News", _gnews_url(GN_EN, "en"), "en"),
    ("Google 뉴스", _gnews_url(GN_KO, "ko"), "ko"),
]
_lock = threading.Lock()


def _fetch_feed(name: str, url: str, lang: str) -> list[dict]:
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=12) as r:
            root = ET.fromstring(r.read())
    except Exception as e:  # noqa: BLE001
        log.warning("news feed failed %s: %s", name, e)
        return []
    out = []
    for it in root.iter("item"):
        title = (it.findtext("title") or "").strip()
        link = (it.findtext("link") or "").strip()
        if not title or not link:
            continue
        src_el = it.find("source")
        source = (src_el.text or "").strip() if src_el is not None and src_el.text else name
        if src_el is not None:  # Google News 는 제목 끝에 " - 매체명" 을 붙인다 (source 요소와 표기가 다를 때도 있어 짧은 꼬리는 떼어냄)
            if title.endswith(" - " + source):
                title = title[: -len(source) - 3].strip()
            else:
                title = re.sub(r"\s+-\s+[^-]{1,24}$", "", title).strip()
        try:
            pub = parsedate_to_datetime(it.findtext("pubDate") or "")
            if pub.tzinfo is None:
                pub = pub.replace(tzinfo=timezone.utc)
        except (TypeError, ValueError):
            continue
        desc = re.sub(r"<[^>]+>", "", it.findtext("description") or "").strip()
        out.append({"title": title, "link": link, "source": source, "feed": name, "lang": lang,
                    "published": pub.astimezone(KST).isoformat(timespec="minutes"), "_pub": pub,
                    "desc": desc[:200] if lang == "en" and not desc.startswith("http") else ""})
    return out


def _norm(t: str) -> str:
    return re.sub(r"[^0-9a-z가-힣]", "", t.lower())[:60]


def _shingles(t: str) -> set[str]:
    """비슷한 제목(같은 사건을 다룬 여러 매체) 판정용: 영문은 단어, 국문은 2글자 조각."""
    s = re.sub(r"[^0-9a-z가-힣 ]", " ", t.lower())
    words = [w for w in s.split() if len(w) > 1]
    if re.search(r"[가-힣]", s):
        return {w[i:i + 2] for w in words for i in range(len(w) - 1)} | set(words)
    return set(words)


def _similar(a: set[str], b: set[str], th: float = 0.6) -> bool:
    if not a or not b:
        return False
    return len(a & b) / min(len(a), len(b)) >= th


def _load_ko() -> dict:
    try:
        return json.loads(KO_CACHE.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


_UNI_NAMES: list[str] | None = None


def _uni_names() -> list[str]:
    """유니버스 회사명(짧게 다듬은 것)을 길이 내림차순으로. 구글 번역 fallback 에서 자리표로 보호할 고유명사."""
    global _UNI_NAMES
    if _UNI_NAMES is not None:
        return _UNI_NAMES
    names: set[str] = {"Anthropic", "OpenAI", "SpaceX", "Nvidia", "Alphabet", "Meta", "Tesla", "Apple", "Amazon", "Microsoft", "Broadcom", "Rocket Lab", "CoreWeave", "Palantir", "Robinhood"}
    try:
        uni = json.loads((DATA_DIR / "universe.json").read_text())
        items = uni["items"] if isinstance(uni, dict) and "items" in uni else uni
        for u in items:
            if u.get("region") != "US":
                continue
            n = re.sub(r"\s*(,?\s*Inc\.?|Corp\.?|Corporation|Co\.?|Company|Ltd\.?|plc|PLC|Holdings|Group|Class [A-C]|\(.*?\))\s*$", "", u.get("name", "")).strip()
            if len(n) >= 4 and not n.isupper():
                names.add(n)
    except Exception:  # noqa: BLE001
        pass
    _UNI_NAMES = sorted(names, key=len, reverse=True)
    return _UNI_NAMES


def _gtx_protected(title: str) -> str | None:
    """구글 번역에 보내기 전에 회사명·대문자 약어(티커 등)를 자리표(ZQX0…)로 바꿔 원문 그대로 남긴다."""
    keep: list[str] = []
    def hold(m):
        keep.append(m.group(0)); return f"ZQX{len(keep)-1}"
    t = title
    for n in _uni_names():
        t = re.sub(r"(?<![A-Za-z])" + re.escape(n) + r"(?![a-z])", hold, t)
    t = re.sub(r"\b(?!ZQX\d)[A-Z][A-Z0-9&]{1,6}\b", hold, t)  # NVDA, CPI, AI, S&P … (자리표 자신은 제외)
    ko = translate_ko(t)
    if not ko:
        return None
    for i in range(len(keep) - 1, -1, -1):  # ZQX1 이 ZQX10 에 겹치지 않게 큰 번호부터
        ko = ko.replace(f"ZQX{i}", keep[i]).replace(f"ZQX {i}", keep[i])
    return ko


LLM_CHUNK = 40


def _llm_enrich(items: list[dict]) -> dict[int, dict]:
    """claude -p 로 번역 + 영향 정리 (LLM_CHUNK 개씩 나눠 호출, 번호는 전체 기준). {번호: {title_ko, impact, tone, skip}}. 실패 시 {}."""
    res: dict[int, dict] = {}
    for off in range(0, len(items), LLM_CHUNK):
        part = _llm_enrich_once(items[off:off + LLM_CHUNK])
        res.update({off + i: v for i, v in part.items()})
    return res


def _llm_enrich_once(items: list[dict]) -> dict[int, dict]:
    exe = shutil.which("claude") or os.path.expanduser("~/.local/bin/claude")
    if not os.path.exists(exe) or not items:
        return {}
    lines = []
    for i, it in enumerate(items, 1):
        lines.append(f"{i}. [{it['lang']}] {it['title']}" + (f" — {it['desc']}" if it.get("desc") else "") + f" (출처: {it['source']})")
    prompt = (
        "아래는 미국 증시 관련 뉴스 제목(일부는 요약 포함)이다. 각 항목에 대해 JSON 배열만 출력하라. 다른 말은 쓰지 마라.\n"
        "형식: [{\"i\": 번호, \"title_ko\": \"...\", \"impact\": \"...\", \"tone\": \"긍정|우려|중립|혼재\", \"skip\": false}, ...]\n"
        "규칙:\n"
        "- tone: 이 기사가 미국 주식시장(관련 종목·지수)에 주는 함의. 상승 요인이면 긍정, 하락·위험 요인이면 우려, 방향성이 없거나 정보성이면 중립, 호재·악재가 섞이면 혼재. 네 단어 중 하나만.\n"
        "- skip: 다음이면 true — 증권사·자산운용사·은행의 상품 기사(ETF/펀드/ETN/랩 출시·상장, 순자산·설정액·판매액·자금유입 돌파, 수상·캠페인·이벤트·제휴 홍보, 단신 모음), 광고성·보도자료성 기사, 특정 종목·상품 매수 추천 광고('이 ETF 가 월 $100 를…' 류), 개인 재무상담·상속·부동산·가족 사연 같은 라이프 칼럼, 증시와 무관한 기사. "
        "시장·경제·기업 실적·정책·금리·환율·지수 동향·대형주 뉴스는 false.\n"
        "- title_ko: [en] 항목은 자연스러운 한국어 뉴스 제목으로 번역. 회사명·인명·제품명·티커·기관명 같은 고유명사는 번역/음역하지 말고 영문 원문 그대로 둔다 (예: Anthropic, Rocket Lab, Morgan Stanley, Neutron). "
        "단 나스닥·S&P500·다우·연준·월가·국채 같은 일반 시장 용어는 한국어로 쓴다. [ko] 항목은 원문 제목을 그대로 넣는다.\n"
        "- impact: 이 기사가 어떤 종목(가능하면 티커)·업종·시장(나스닥/S&P500/국내 증시/환율/금리 등)에 어떤 방향으로 영향을 줄 수 있는지 한국어 5줄 이내로, 줄바꿈은 \\n. 제목·요약에 근거한 것만 쓰고 추측을 늘리지 마라. "
        "증시와 뚜렷한 연관이 없는 기사(일반 사회·연예·광고성·개별 상품 홍보 등)는 빈 문자열 \"\" 로 둔다.\n\n" + "\n".join(lines))
    try:
        r = subprocess.run([exe, "-p", "--no-session-persistence", "--tools", "", "--model", LLM_MODEL, "--output-format", "text"],  # --bare 는 로그인 정보를 안 읽어 실패
                           input=prompt, capture_output=True, text=True, timeout=LLM_TIMEOUT, cwd=os.path.expanduser("~"),
                           env={**os.environ, "PATH": os.environ.get("PATH", "") + ":" + os.path.expanduser("~/.local/bin")})
        out = r.stdout.strip()
        if r.returncode != 0 or not out:
            log.warning("news llm failed rc=%s: %s", r.returncode, (r.stderr or out)[:300])
            return {}
        m = re.search(r"\[.*\]", out, re.S)
        arr = json.loads(m.group(0) if m else out)
        res = {}
        for a in arr:
            try:
                tone = str(a.get("tone") or "").strip()
                res[int(a["i"])] = {"title_ko": str(a.get("title_ko") or "").strip(), "impact": str(a.get("impact") or "").strip(), "skip": bool(a.get("skip")),
                                    "tone": tone if tone in TONES else ""}
            except (KeyError, TypeError, ValueError):
                continue
        return res
    except Exception as e:  # noqa: BLE001
        log.warning("news llm failed: %s", e)
        return {}


def _translate(items: list[dict]) -> None:
    """캐시에 없는 기사만 모아 LLM 에 한 번 묻고, 실패분은 구글 번역으로 제목만."""
    ko = _load_ko()
    todo = [it for it in items if not isinstance(ko.get(it["link"]), dict) or "skip" not in ko[it["link"]] or "tone" not in ko[it["link"]]]  # skip/tone 판정이 없는 캐시(옛 것·gtx fallback)는 다시 묻는다
    got = _llm_enrich(todo) if todo else {}
    changed = False
    for i, it in enumerate(todo, 1):
        g = got.get(i)
        if g and (it["lang"] == "ko" or g["title_ko"] or g.get("skip")):  # 제외 판정이면 제목이 비어도 그대로 받아들인다
            ko[it["link"]] = {"title_ko": g["title_ko"] if it["lang"] == "en" else "", "impact": g["impact"], "skip": g.get("skip", False), "tone": g.get("tone", ""), "src": "llm"}
        else:
            t = _gtx_protected(it["title"]) if it["lang"] == "en" else ""
            ko[it["link"]] = {"title_ko": t or "", "impact": "", "src": "gtx" if t else "none"}  # skip 키 없음 → 다음 갱신 때 LLM 에 다시 묻는다
        changed = True
    for it in items:
        c = ko.get(it["link"]) or {}
        tk = c.get("title_ko") or ""
        tk = re.sub(r"^[\(\[]\s*" + re.escape(it["source"]) + r"\s*[\)\]]\s*[:：-]?\s*", "", tk, flags=re.I)  # 모델이 "(매체명) " 을 제목 앞에 붙인 경우
        tk = re.sub(r"^\(출처:[^)]*\)\s*", "", tk)
        it["title_ko"] = tk or None
        it["impact"] = c.get("impact") or ""
        it["enrich"] = c.get("src")
        it["skip"] = bool(c.get("skip"))
        it["tone"] = c.get("tone") or ""
    if changed:
        if len(ko) > 600:  # 오래된 것부터 버림 (dict 는 삽입순)
            ko = dict(list(ko.items())[-400:])
        try:
            atomic_write_text(KO_CACHE, json.dumps(ko, ensure_ascii=False))
        except OSError:
            pass


def _select(items: list[dict], pool: int = POOL, max_age: timedelta | None = MAX_AGE) -> list[dict]:
    now = datetime.now(timezone.utc)
    seen: set[str] = set()
    shingles: list[set[str]] = []
    by_feed: dict[str, list[dict]] = {}
    for it in sorted(items, key=lambda x: x["_pub"], reverse=True):
        if (max_age is not None and now - it["_pub"] > max_age) or _is_product_pr(it["title"], it.get("desc", "")):
            continue
        k = _norm(it["title"])
        if not k or k in seen:
            continue
        sh = _shingles(it["title"])
        if any(_similar(sh, x) for x in shingles):  # 같은 사건을 다룬 다른 매체 기사
            continue
        seen.add(k)
        shingles.append(sh)
        by_feed.setdefault(it["feed"], []).append(it)
    picked: list[dict] = []
    queues = [by_feed[n] for n, _, _ in FEEDS if by_feed.get(n)]
    while len(picked) < pool and any(queues):  # 피드별로 번갈아 최신 기사부터 (LLM 이 홍보성으로 거른 뒤 LIMIT 개)
        for q in queues:
            if q and len(picked) < pool:
                picked.append(q.pop(0))
    picked.sort(key=lambda x: x["_pub"], reverse=True)
    for it in picked:
        it.pop("_pub", None)
    return picked


def load_cached() -> dict | None:
    try:
        return json.loads(CACHE.read_text())
    except (OSError, json.JSONDecodeError):
        return None


_REFRESHING = {"on": False}


def _fresh(c: dict | None) -> bool:
    try:
        return bool(c) and datetime.now(KST) - datetime.fromisoformat(c["fetched_at"]) < TTL
    except (KeyError, TypeError, ValueError):
        return False


def refresh() -> dict:
    """피드 수집 → 선별 → 번역/영향 → data/news.json. 동시 실행은 하나만."""
    with _lock:
        _REFRESHING["on"] = True
        try:
            items: list[dict] = []
            for name, url, lang in FEEDS:
                items += _fetch_feed(name, url, lang)
            cached = load_cached()
            if not items:
                return {**cached, "stale": True} if cached else {"fetched_at": None, "items": [], "error": "뉴스 피드를 받지 못했습니다"}
            picked = _select(items)
            try:
                _translate(picked)
            except Exception as e:  # noqa: BLE001
                log.warning("news translate failed: %s", e)
            skipped = [it["title"] for it in picked if it.get("skip")]
            kept = [it for it in picked if not it.get("skip")]
            try:
                _log_items(kept)  # 일별 요약에는 화면 20개보다 넓게(후보 중 홍보성 제외 전부) 쌓는다
            except Exception as e:  # noqa: BLE001
                log.warning("news log failed: %s", e)
            picked = kept[:LIMIT]
            res = {"fetched_at": datetime.now(KST).isoformat(timespec="seconds"), "items": picked, "skipped": skipped[:20],
                   "feeds": [n for n, _, _ in FEEDS], "n_raw": len(items)}
            try:
                atomic_write_text(CACHE, json.dumps(res, ensure_ascii=False))
            except OSError:
                pass
            return res
        finally:
            _REFRESHING["on"] = False


def get(force: bool = False) -> dict:
    """캐시가 신선하면 그대로. 아니면 캐시가 있을 때는 백그라운드로 갱신을 시작하고 이전 목록을 refreshing=true 로 준다(LLM 호출이 30~60초).
    캐시가 아예 없으면 동기 갱신."""
    cached = load_cached()
    if cached and _fresh(cached) and not force:
        return cached
    if cached:
        if not _REFRESHING["on"]:
            threading.Thread(target=refresh, daemon=True).start()
        return {**cached, "refreshing": True}
    return refresh()


# ---------- 누적 로그 → 일별·월별 요약 ----------
def _load_log() -> dict:
    try:
        d = json.loads(LOG.read_text())
        return d if isinstance(d, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _log_items(items: list[dict]) -> None:
    """표시된 기사를 링크별로 누적. 판정(tone/impact)이 나중에 채워지면 갱신."""
    d = _load_log()
    changed = False
    for it in items:
        cur = d.get(it["link"])
        rec = {"title": it["title"], "title_ko": it.get("title_ko") or "", "source": it["source"], "published": it["published"],
               "tone": it.get("tone") or "", "impact": it.get("impact") or ""}
        if cur != rec:
            d[it["link"]] = rec
            changed = True
    if changed:
        atomic_write_text(LOG, json.dumps(d, ensure_ascii=False))


def _verdict(pos: int, neg: int, n: int) -> str:
    if n == 0:
        return "-"
    if pos >= 2 and pos >= neg * 1.5:
        return "상승 우위"
    if neg >= 2 and neg >= pos * 1.5:
        return "하락 우위"
    return "혼조"


def _agg(recs: list[dict]) -> dict:
    c = {"긍정": 0, "우려": 0, "중립": 0, "혼재": 0}
    for r in recs:
        if r.get("tone") in c:
            c[r["tone"]] += 1
    n = len(recs)
    return {"n": n, "pos": c["긍정"], "neg": c["우려"], "neu": c["중립"], "mix": c["혼재"], "verdict": _verdict(c["긍정"], c["우려"], n),
            "score": round((c["긍정"] - c["우려"]) / n, 2) if n else None}


def daily() -> dict:
    """일별(기사 발행일, KST) 집계와 월별 집계. 각 일자에 그날 기사 목록 포함."""
    d = _load_log()
    by_day: dict[str, list[dict]] = {}
    for link, r in d.items():
        day = (r.get("published") or "")[:10]
        if len(day) == 10:
            by_day.setdefault(day, []).append({**r, "link": link})
    days = []
    for day in sorted(by_day, reverse=True):
        recs = sorted(by_day[day], key=lambda r: r["published"], reverse=True)
        days.append({"date": day, **_agg(recs), "items": [{k: r.get(k) for k in ("title", "title_ko", "source", "tone", "link", "published")} for r in recs]})
    months = []
    for m in sorted({x["date"][:7] for x in days}, reverse=True):
        recs = [r for day, rs in by_day.items() if day[:7] == m for r in rs]
        months.append({"month": m, **_agg(recs), "days": sum(1 for x in days if x["date"][:7] == m)})
    return {"days": days, "months": months, "since": min(by_day) if by_day else None, "total": len(d), "backfill": dict(BF_STATE),
            "refresh_hours": REFRESH_HOURS, "backfill_days": BACKFILL_DAYS}


# ---------- 백필 + 주기 실행 ----------
BF_STATE = {"running": False, "stage": "", "done": 0, "total": 0, "last": None}


def _day_counts() -> dict[str, int]:
    c: dict[str, int] = {}
    for r in _load_log().values():
        d = (r.get("published") or "")[:10]
        if len(d) == 10:
            c[d] = c.get(d, 0) + 1
    return c


def backfill_day(day: str, per_day: int = BACKFILL_PER_DAY) -> int:
    """하루치: Google News 날짜 검색(영문·국문) → 홍보성 제외·중복 제거 → LLM 번역/판정 → 로그. 반환: 기록한 수."""
    nxt = (datetime.fromisoformat(day) + timedelta(days=1)).strftime("%Y-%m-%d")
    items = _fetch_feed("Google News", _gnews_url(GN_EN, "en", day, nxt), "en") + _fetch_feed("Google 뉴스", _gnews_url(GN_KO, "ko", day, nxt), "ko")
    items = [it for it in items if it["published"][:10] == day]  # 구글이 범위를 느슨하게 잡으므로 발행일 재확인
    picked = _select(items, pool=per_day + 8, max_age=None)
    if not picked:
        return 0
    try:
        _translate(picked)
    except Exception as e:  # noqa: BLE001
        log.warning("news backfill translate failed %s: %s", day, e)
    kept = [it for it in picked if not it.get("skip")][:per_day]
    _log_items(kept)
    return len(kept)


def ensure_days(days: int = BACKFILL_DAYS, min_items: int = BACKFILL_MIN) -> dict:
    """최근 days 일 중 로그가 min_items 미만인 날짜(오늘 제외)를 백필. 동시 실행은 하나만."""
    if BF_STATE["running"]:
        return {"skipped": "already running"}
    BF_STATE.update(running=True, stage="확인", done=0, total=0)
    try:
        counts = _day_counts()
        today = datetime.now(KST).date()
        targets = [(today - timedelta(days=i)).isoformat() for i in range(1, days + 1)]
        targets = [d for d in targets if counts.get(d, 0) < min_items]
        BF_STATE.update(total=len(targets))
        filled = {}
        for i, d in enumerate(targets):
            BF_STATE.update(stage=f"백필 {d}", done=i)
            try:
                filled[d] = backfill_day(d)
            except Exception as e:  # noqa: BLE001
                log.warning("news backfill failed %s: %s", d, e)
            time.sleep(1.0)  # Google News 요청 간격
        BF_STATE.update(stage="완료", done=len(targets), last=datetime.now(KST).isoformat(timespec="seconds"))
        log.info("news backfill: %d days, %s items", len(filled), sum(filled.values()))
        return {"days": filled}
    finally:
        BF_STATE["running"] = False


def loop() -> None:
    """서버 기동 2분 뒤 첫 수집·백필, 이후 REFRESH_HOURS 마다 수집, 매일 07:00 이후 첫 회차에 지난 날짜 백필 점검."""
    time.sleep(120)
    last_bf_day = None
    while True:
        try:
            refresh()
        except Exception as e:  # noqa: BLE001
            log.warning("news periodic refresh failed: %s", e)
        now = datetime.now(KST)
        if last_bf_day != now.date() and (last_bf_day is None or now.hour >= 7):
            last_bf_day = now.date()
            try:
                ensure_days()
            except Exception as e:  # noqa: BLE001
                log.warning("news backfill failed: %s", e)
        time.sleep(max(600, REFRESH_HOURS * 3600))


def _sample_weekdays(year: int, month: int, k: int) -> list[str]:
    """그 달의 평일 중 k 개를 고르게 뽑는다 (시장 뉴스는 평일에 몰림)."""
    import calendar
    days = [datetime(year, month, d) for d in range(1, calendar.monthrange(year, month)[1] + 1)]
    wd = [d for d in days if d.weekday() < 5]
    if len(wd) <= k:
        return [d.strftime("%Y-%m-%d") for d in wd]
    idx = [round(i * (len(wd) - 1) / (k - 1)) for i in range(k)] if k > 1 else [0]
    return [wd[i].strftime("%Y-%m-%d") for i in sorted(set(idx))]


def backfill_days_batch(days: list[str], per_day: int) -> dict[str, int]:
    """여러 날짜를 한 번에: 날짜별로 Google News 날짜 검색 → 선별 → 전체를 LLM 에 묶어 보냄(40개씩) → 날짜별 기록."""
    per_day_items: dict[str, list[dict]] = {}
    for day in days:
        nxt = (datetime.fromisoformat(day) + timedelta(days=1)).strftime("%Y-%m-%d")
        items = _fetch_feed("Google News", _gnews_url(GN_EN, "en", day, nxt), "en")
        time.sleep(0.8)
        items += _fetch_feed("Google 뉴스", _gnews_url(GN_KO, "ko", day, nxt), "ko")
        time.sleep(0.8)
        items = [it for it in items if it["published"][:10] == day]
        per_day_items[day] = _select(items, pool=per_day + 6, max_age=None)
    allitems = [it for day in days for it in per_day_items[day]]
    if allitems:
        try:
            _translate(allitems)
        except Exception as e:  # noqa: BLE001
            log.warning("news batch translate failed: %s", e)
    out = {}
    for day in days:
        kept = [it for it in per_day_items[day] if not it.get("skip")][:per_day]
        if kept:
            _log_items(kept)
        out[day] = len(kept)
    return out


def backfill_months(start: str, end: str, days_per_month: int = 8, per_day: int = 10, min_items: int = 4) -> dict:
    """start~end(YYYY-MM) 각 달에서 평일 days_per_month 일을 표본으로 뽑아 백필 (최근 달부터). 이미 min_items 이상 있는 날짜는 건너뜀."""
    while BF_STATE["running"]:  # 일별 백필이 돌고 있으면 끝날 때까지 대기
        time.sleep(15)
    BF_STATE.update(running=True, stage="월별 백필 준비", done=0, total=0)
    try:
        y0, m0 = (int(x) for x in start.split("-")); y1, m1 = (int(x) for x in end.split("-"))
        months = []
        y, m = y1, m1
        while (y, m) >= (y0, m0):
            months.append((y, m)); m -= 1
            if m == 0:
                y, m = y - 1, 12
        BF_STATE.update(total=len(months))
        filled: dict[str, int] = {}
        for i, (y, m) in enumerate(months):
            counts = _day_counts()
            days = [d for d in _sample_weekdays(y, m, days_per_month) if counts.get(d, 0) < min_items and d < datetime.now(KST).strftime("%Y-%m-%d")]
            BF_STATE.update(stage=f"월별 백필 {y}-{m:02d} ({len(days)}일)", done=i)
            if days:
                try:
                    r = backfill_days_batch(days, per_day)
                    filled[f"{y}-{m:02d}"] = sum(r.values())
                except Exception as e:  # noqa: BLE001
                    log.warning("news month backfill failed %s-%02d: %s", y, m, e)
        BF_STATE.update(stage="완료", done=len(months), last=datetime.now(KST).isoformat(timespec="seconds"))
        log.info("news month backfill: %s", filled)
        return {"months": filled}
    finally:
        BF_STATE["running"] = False
