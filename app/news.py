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
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime

from .profile import translate_ko
from .scanner import KST, atomic_write_text
from .universe import DATA_DIR

LLM_MODEL = os.environ.get("NEWS_LLM_MODEL", "claude-haiku-5-5")
LLM_TIMEOUT = 150

log = logging.getLogger(__name__)
CACHE = DATA_DIR / "news.json"
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
FEEDS = [
    ("CNBC", "https://www.cnbc.com/id/10000664/device/rss/rss.html", "en"),
    ("MarketWatch", "https://feeds.content.dowjones.io/public/rss/mw_topstories", "en"),
    ("Google News", "https://news.google.com/rss/search?q=" + urllib.request.quote('stock market OR nasdaq OR "wall street" OR "S&P 500" when:1d') + "&hl=en-US&gl=US&ceid=US:en", "en"),
    ("Google 뉴스", "https://news.google.com/rss/search?q=" + urllib.request.quote("뉴욕증시 OR 미국 증시 OR 나스닥 when:1d") + "&hl=ko&gl=KR&ceid=KR:ko", "ko"),
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
        if src_el is not None and title.endswith(" - " + source):  # Google News 는 제목 끝에 매체명을 붙인다
            title = title[: -len(source) - 3].strip()
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


def _llm_enrich(items: list[dict]) -> dict[int, dict]:
    """claude -p 한 번으로 번역 + 영향 정리. {번호: {title_ko, impact}}. 실패 시 {}."""
    exe = shutil.which("claude") or os.path.expanduser("~/.local/bin/claude")
    if not os.path.exists(exe) or not items:
        return {}
    lines = []
    for i, it in enumerate(items, 1):
        lines.append(f"{i}. [{it['lang']}] ({it['source']}) {it['title']}" + (f" — {it['desc']}" if it.get("desc") else ""))
    prompt = (
        "아래는 미국 증시 관련 뉴스 제목(일부는 요약 포함)이다. 각 항목에 대해 JSON 배열만 출력하라. 다른 말은 쓰지 마라.\n"
        "형식: [{\"i\": 번호, \"title_ko\": \"...\", \"impact\": \"...\", \"skip\": false}, ...]\n"
        "규칙:\n"
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
                res[int(a["i"])] = {"title_ko": str(a.get("title_ko") or "").strip(), "impact": str(a.get("impact") or "").strip(), "skip": bool(a.get("skip"))}
            except (KeyError, TypeError, ValueError):
                continue
        return res
    except Exception as e:  # noqa: BLE001
        log.warning("news llm failed: %s", e)
        return {}


def _translate(items: list[dict]) -> None:
    """캐시에 없는 기사만 모아 LLM 에 한 번 묻고, 실패분은 구글 번역으로 제목만."""
    ko = _load_ko()
    todo = [it for it in items if not isinstance(ko.get(it["link"]), dict) or (ko[it["link"]].get("src") == "llm" and "skip" not in ko[it["link"]])]  # skip 판정이 없는 옛 캐시는 다시 묻는다
    got = _llm_enrich(todo) if todo else {}
    changed = False
    for i, it in enumerate(todo, 1):
        g = got.get(i)
        if g and (it["lang"] == "ko" or g["title_ko"]):
            ko[it["link"]] = {"title_ko": g["title_ko"] if it["lang"] == "en" else "", "impact": g["impact"], "skip": g.get("skip", False), "src": "llm"}
        else:
            t = _gtx_protected(it["title"]) if it["lang"] == "en" else ""
            ko[it["link"]] = {"title_ko": t or "", "impact": "", "src": "gtx" if t else "none"}
        changed = True
    for it in items:
        c = ko.get(it["link"]) or {}
        it["title_ko"] = c.get("title_ko") or None
        it["impact"] = c.get("impact") or ""
        it["enrich"] = c.get("src")
        it["skip"] = bool(c.get("skip"))
    if changed:
        if len(ko) > 600:  # 오래된 것부터 버림 (dict 는 삽입순)
            ko = dict(list(ko.items())[-400:])
        try:
            atomic_write_text(KO_CACHE, json.dumps(ko, ensure_ascii=False))
        except OSError:
            pass


def _select(items: list[dict]) -> list[dict]:
    now = datetime.now(timezone.utc)
    seen: set[str] = set()
    shingles: list[set[str]] = []
    by_feed: dict[str, list[dict]] = {}
    for it in sorted(items, key=lambda x: x["_pub"], reverse=True):
        if now - it["_pub"] > MAX_AGE or _is_product_pr(it["title"], it.get("desc", "")):
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
    while len(picked) < POOL and any(queues):  # 피드별로 번갈아 최신 기사부터 (LLM 이 홍보성으로 거른 뒤 LIMIT 개)
        for q in queues:
            if q and len(picked) < POOL:
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
            picked = [it for it in picked if not it.get("skip")][:LIMIT]
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
