"""미국 증시 주요 뉴스 클립 (리포트 탭). 키 없는 공개 RSS 4개를 합쳐 최근 36시간 안의 기사 20개를 고른다.

- CNBC Markets · MarketWatch Top Stories (영문) · Google News 검색(영문: stock market/nasdaq/wall street, 국문: 뉴욕증시/미국 증시/나스닥)
- 제목 정규화로 중복 제거, 피드별로 번갈아 뽑아(round-robin) 한 매체에 쏠리지 않게 한 뒤 최신순.
- 영문 제목은 profile.translate_ko(구글 번역 gtx) 로 한글화해 링크별로 캐시(data/news_cache.json).
- 결과는 data/news.json 에 20분 캐시. 수집 실패 시 이전 결과를 그대로 준다.
"""
from __future__ import annotations

import json
import logging
import re
import threading
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime

from .profile import translate_ko
from .scanner import KST, atomic_write_text
from .universe import DATA_DIR

log = logging.getLogger(__name__)
CACHE = DATA_DIR / "news.json"
KO_CACHE = DATA_DIR / "news_cache.json"
TTL = timedelta(minutes=20)
MAX_AGE = timedelta(hours=36)
LIMIT = 20
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


def _load_ko() -> dict:
    try:
        return json.loads(KO_CACHE.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _translate(items: list[dict]) -> None:
    ko = _load_ko()
    changed = False
    for it in items:
        if it["lang"] != "en":
            continue
        t = ko.get(it["link"])
        if t is None:
            t = translate_ko(it["title"])
            if t:
                ko[it["link"]] = t
                changed = True
        it["title_ko"] = t
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
    by_feed: dict[str, list[dict]] = {}
    for it in sorted(items, key=lambda x: x["_pub"], reverse=True):
        if now - it["_pub"] > MAX_AGE:
            continue
        k = _norm(it["title"])
        if not k or k in seen:
            continue
        seen.add(k)
        by_feed.setdefault(it["feed"], []).append(it)
    picked: list[dict] = []
    queues = [by_feed[n] for n, _, _ in FEEDS if by_feed.get(n)]
    while len(picked) < LIMIT and any(queues):  # 피드별로 번갈아 최신 기사부터
        for q in queues:
            if q and len(picked) < LIMIT:
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


def get(force: bool = False) -> dict:
    cached = load_cached()
    if cached and not force:
        try:
            if datetime.now(KST) - datetime.fromisoformat(cached["fetched_at"]) < TTL:
                return cached
        except (KeyError, ValueError):
            pass
    with _lock:
        cached2 = load_cached()  # 다른 스레드가 방금 갱신했으면 그것
        if cached2 and not force and cached2 is not cached:
            try:
                if datetime.now(KST) - datetime.fromisoformat(cached2["fetched_at"]) < TTL:
                    return cached2
            except (KeyError, ValueError):
                pass
        items: list[dict] = []
        for name, url, lang in FEEDS:
            items += _fetch_feed(name, url, lang)
        if not items:
            if cached:
                return {**cached, "stale": True}
            return {"fetched_at": None, "items": [], "error": "뉴스 피드를 받지 못했습니다"}
        picked = _select(items)
        try:
            _translate(picked)
        except Exception as e:  # noqa: BLE001
            log.warning("news translate failed: %s", e)
        res = {"fetched_at": datetime.now(KST).isoformat(timespec="seconds"), "items": picked,
               "feeds": [n for n, _, _ in FEEDS], "n_raw": len(items)}
        try:
            atomic_write_text(CACHE, json.dumps(res, ensure_ascii=False))
        except OSError:
            pass
        return res
