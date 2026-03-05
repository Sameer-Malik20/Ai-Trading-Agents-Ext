"""Agent 4: sentiment analysis using Google News RSS, FinBERT, and Groq refinement."""

from __future__ import annotations

import asyncio
import datetime as dt
import difflib
import json
import os
import re
from typing import Any, Dict, List, Optional
from urllib.parse import quote_plus

import feedparser
import httpx
import requests
from zoneinfo import ZoneInfo

from utils.data_fetcher import current_market_session

try:
    from transformers import pipeline  # type: ignore
except Exception:  # pragma: no cover - runtime fallback
    pipeline = None

try:
    import praw  # type: ignore
except Exception:  # pragma: no cover - runtime fallback
    praw = None

IST = ZoneInfo("Asia/Kolkata")

_FINBERT_PIPELINE = None
_FINBERT_LOCK = asyncio.Lock()


class GroqRateLimiter:
    """Async rate limiter ensuring max one Groq call per configured interval."""

    def __init__(self, min_interval_seconds: float = 3.0) -> None:
        self.min_interval_seconds = min_interval_seconds
        self._lock = asyncio.Lock()
        self._last_call: float = 0.0

    async def wait_turn(self) -> None:
        """Wait until call is permitted."""

        async with self._lock:
            now = asyncio.get_running_loop().time()
            gap = now - self._last_call
            if gap < self.min_interval_seconds:
                await asyncio.sleep(self.min_interval_seconds - gap)
            self._last_call = asyncio.get_running_loop().time()


GROQ_LIMITER = GroqRateLimiter(min_interval_seconds=3.0)

COMPANY_FILTERS: Dict[str, Dict[str, List[str]]] = {
    "RELIANCE": {
        "include": ["reliance", "reliance industries", "ril", "nse:reliance", "reliance.ns"],
        "exclude": [
            "reliance power",
            "rpower",
            "reliance infrastructure",
            "reliance infra",
            "reliance communications",
            "rccom",
            "adani",
            "vedanta",
            "ongc",
            "ioc",
            "bpcl",
            "hpcl",
        ],
    },
    "TCS": {
        "include": ["tcs", "tata consultancy", "tata consultancy services", "nse:tcs", "tcs.ns"],
        "exclude": ["tata motors", "tatamotors", "tata steel", "tatasteel", "tata power", "tatapower"],
    },
    "INFY": {
        "include": ["infy", "infosys", "nse:infy", "infosys ltd", "infosys limited"],
        "exclude": [],
    },
}

COMPANY_FULL_NAME: Dict[str, str] = {
    "RELIANCE": "Reliance Industries Limited",
    "TCS": "Tata Consultancy Services",
    "HDFCBANK": "HDFC Bank Limited",
    "INFY": "Infosys Limited",
    "ICICIBANK": "ICICI Bank Limited",
    "SBIN": "State Bank of India",
}

NEWS_SOURCE_PRIORITY: Dict[str, int] = {
    "NSE Corporate Announcements": 1,
    "Economic Times RSS": 2,
    "Google News RSS": 3,
    "Extension": 0,
}


async def run(symbol: str, provided_headlines: Optional[List[str]] = None, lightweight: bool = False) -> Dict[str, Any]:
    """Run full sentiment pipeline with robust fallback."""

    try:
        news_items = collect_news(symbol, limit=5)
        if provided_headlines:
            extension_items = [{"title": h, "description": "", "published": "", "source": "Extension"} for h in provided_headlines]
            news_items = _merge_news_sources(symbol, [extension_items, news_items], limit=5)

        recency = _apply_news_recency_policy(news_items)
        scoring_news = recency["items_for_scoring"]
        display_news = recency["items_for_display"]
        sentiment_flags = list(recency["flags"])
        news_freshness = str(recency["news_freshness"])
        sentiment_confidence = float(recency["sentiment_confidence"])

        finbert_results = await analyze_finbert(scoring_news, lightweight=lightweight)
        finbert_score = _score_from_finbert(finbert_results)

        groq_analysis = await refine_with_groq(
            symbol,
            scoring_news if scoring_news else display_news,
            finbert_score,
            news_freshness=news_freshness,
        )
        social = await social_sentiment(symbol) if not lightweight else {"score": 50.0, "source": "skipped"}

        sentiment_score = _aggregate_sentiment(finbert_score, groq_analysis, social)
        if news_freshness == "STALE":
            sentiment_score = min(sentiment_score, 50.0)
        overall = _band(sentiment_score)

        return {
            "ok": True,
            "sentiment_score": round(float(sentiment_score), 2),
            "news_items": display_news,
            "finbert_results": finbert_results,
            "groq_analysis": groq_analysis,
            "social_score": round(float(social.get("score", 50.0)), 2),
            "overall_sentiment": overall,
            "news_freshness": news_freshness,
            "sentiment_confidence": round(sentiment_confidence, 2),
            "sentiment_flags": sentiment_flags,
            "source": _build_source(finbert_results, groq_analysis),
            "agent_status": "GREEN" if sentiment_score >= 35 else "RED",
            "error": "",
        }
    except Exception as exc:
        return {
            "ok": False,
            "sentiment_score": 50.0,
            "news_items": [],
            "finbert_results": [],
            "groq_analysis": {},
            "social_score": 50.0,
            "overall_sentiment": "NEUTRAL",
            "news_freshness": "STALE",
            "sentiment_confidence": 20.0,
            "sentiment_flags": ["STALE_NEWS_WARNING"],
            "source": "fallback",
            "agent_status": "AMBER",
            "error": f"agent4_failure: {exc}",
        }


def collect_news(symbol: str, limit: int = 10) -> List[Dict[str, str]]:
    """Collect company-specific headlines with priority NSE -> ET -> Google and return top recent set."""

    primary = collect_nse_announcements(symbol, limit=max(limit * 4, 20))
    backup = collect_economic_times_news(symbol, limit=max(limit * 3, 15))
    fallback = collect_google_news(symbol, limit=max(limit * 4, 20))
    return _merge_news_sources(symbol, [primary, backup, fallback], limit=limit)


def collect_nse_announcements(symbol: str, limit: int = 20) -> List[Dict[str, str]]:
    """Collect corporate announcements from NSE APIs."""

    sym = str(symbol or "").strip().upper()
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json,text/plain,*/*",
        "Referer": "https://www.nseindia.com/",
    }
    endpoints = [
        f"https://www.nseindia.com/api/corp-info?symbol={sym}&corpType=announcements&market=eq",
        f"https://www.nseindia.com/api/corporate-announcements?symbol={sym}&index=equities",
    ]

    rows: List[Dict[str, Any]] = []
    session = requests.Session()
    try:
        session.get("https://www.nseindia.com", headers=headers, timeout=10)
        for endpoint in endpoints:
            try:
                response = session.get(endpoint, headers=headers, timeout=10)
                if response.status_code != 200:
                    continue
                payload = response.json()
                data = payload.get("data", []) if isinstance(payload, dict) else payload
                if isinstance(data, list) and data:
                    rows = [row for row in data if isinstance(row, dict)]
                    break
            except Exception:
                continue
    except Exception:
        return []

    items: List[Dict[str, str]] = []
    for row in rows:
        row_symbol = str(row.get("symbol", "")).strip().upper()
        if row_symbol and row_symbol != sym:
            continue

        title = str(row.get("attchmntText") or row.get("subject") or row.get("desc") or "").strip()
        desc = str(row.get("desc") or "").strip()
        if not title:
            continue
        published_dt = _parse_datetime_any(row.get("sort_date") or row.get("an_dt") or row.get("dt"))
        attachment = str(row.get("attchmntFile") or "").strip()
        if attachment:
            desc = f"{desc} | {attachment}".strip(" |")

        items.append(
            {
                "title": title,
                "description": desc,
                "published": published_dt.isoformat() if published_dt else "",
                "source": "NSE Corporate Announcements",
            }
        )

    return _sort_news_by_recency(items)[:limit]


def collect_economic_times_news(symbol: str, limit: int = 15) -> List[Dict[str, str]]:
    """Collect Economic Times market headlines (feedparser-based backup source)."""

    sym = str(symbol or "").strip().upper()
    full_name = COMPANY_FULL_NAME.get(sym, sym)
    slug = _slugify_company_name(full_name)
    urls = [
        f"https://economictimes.indiatimes.com/markets/stocks/news/{sym.lower()}.cms",
        f"https://economictimes.indiatimes.com/markets/stocks/news/{slug}.cms",
    ]

    items: List[Dict[str, str]] = []
    for url in urls:
        try:
            parsed = feedparser.parse(url)
        except Exception:
            continue
        if not getattr(parsed, "entries", None):
            continue
        for entry in parsed.entries[: max(limit * 2, 20)]:
            title = str(entry.get("title", "")).strip()
            if not title:
                continue
            published_dt = _entry_published_datetime(entry) or _parse_datetime_any(entry.get("published"))
            items.append(
                {
                    "title": title,
                    "description": str(entry.get("summary", "")).strip(),
                    "published": published_dt.isoformat() if published_dt else "",
                    "source": "Economic Times RSS",
                }
            )
    return _sort_news_by_recency(items)[:limit]


def collect_google_news(symbol: str, limit: int = 20) -> List[Dict[str, str]]:
    """Collect Google News as strict last-resort source."""

    sym = str(symbol or "").strip().upper()
    company_name = COMPANY_FULL_NAME.get(sym, sym)
    query = quote_plus(f"{company_name} NSE India")
    url = f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"
    parsed = feedparser.parse(url)

    items: List[Dict[str, str]] = []
    for entry in parsed.entries[: max(limit * 3, 20)]:
        title = str(entry.get("title", "")).strip()
        if not title:
            continue
        published_dt = _entry_published_datetime(entry)
        items.append(
            {
                "title": title,
                "description": str(entry.get("summary", "")).strip(),
                "published": published_dt.isoformat() if published_dt else "",
                "source": "Google News RSS",
            }
        )
    return _sort_news_by_recency(items)[:limit]


async def analyze_finbert(news_items: List[Dict[str, str]], lightweight: bool = False) -> List[Dict[str, Any]]:
    """Run local FinBERT headline sentiment if available."""

    if not news_items or pipeline is None:
        return []

    max_items = 3 if lightweight else 10
    headlines = [n["title"] for n in news_items[:max_items] if n.get("title")]
    if not headlines:
        return []

    finbert = await _load_finbert()
    if finbert is None:
        return []

    def _infer() -> List[Dict[str, Any]]:
        outputs = finbert(headlines, truncation=True)
        rows: List[Dict[str, Any]] = []
        for head, out in zip(headlines, outputs):
            label = str(out.get("label", "neutral")).lower()
            score = float(out.get("score", 0.0))
            mapped = _finbert_to_0_100(label, score)
            rows.append({"headline": head, "label": label, "confidence": score, "score_0_100": mapped})
        return rows

    try:
        return await asyncio.to_thread(_infer)
    except Exception:
        return []


async def _load_finbert():
    """Load FinBERT pipeline once, lazily."""

    global _FINBERT_PIPELINE
    if _FINBERT_PIPELINE is not None:
        return _FINBERT_PIPELINE
    async with _FINBERT_LOCK:
        if _FINBERT_PIPELINE is not None:
            return _FINBERT_PIPELINE
        if pipeline is None:
            return None
        try:
            _FINBERT_PIPELINE = await asyncio.to_thread(
                pipeline,
                "sentiment-analysis",
                model="ProsusAI/finbert",
            )
            return _FINBERT_PIPELINE
        except Exception:
            return None


def _score_from_finbert(rows: List[Dict[str, Any]]) -> float:
    """Aggregate FinBERT row scores to a single sentiment score."""

    if not rows:
        return 50.0
    total = sum(float(r.get("score_0_100", 50.0)) for r in rows)
    return max(0.0, min(100.0, total / len(rows)))


async def refine_with_groq(symbol: str, news_items: List[Dict[str, str]], finbert_score: float, news_freshness: str = "MIXED") -> Dict[str, Any]:
    """Ask Groq to refine financial sentiment."""

    if not news_items:
        return {}
    headlines = "\n".join([f"- {n['title']}" for n in news_items[:10]])
    system_prompt = (
        "You are a financial sentiment analyst for Indian stock markets. "
        "Return only JSON."
    )
    stale_note = "Note: news may be stale, be conservative.\n" if str(news_freshness).upper() == "STALE" else ""
    user_prompt = (
        f"Symbol: {symbol}\n"
        f"Baseline sentiment score: {finbert_score:.2f}\n"
        f"News freshness state: {news_freshness}\n"
        f"{stale_note}"
        "Analyze these headlines and return JSON with keys:\n"
        "sentiment_score (0-100, 50=neutral), "
        "key_positive_factors (array), key_negative_factors (array), "
        "market_impact (HIGH|MEDIUM|LOW), recommendation (BUY_SIGNAL|SELL_SIGNAL|NEUTRAL).\n"
        f"Headlines:\n{headlines}"
    )
    return await call_groq_json(system_prompt=system_prompt, user_prompt=user_prompt)


async def social_sentiment(symbol: str) -> Dict[str, Any]:
    """Optional Reddit sentiment using free API if credentials are available."""

    client_id = os.getenv("REDDIT_CLIENT_ID", "").strip()
    client_secret = os.getenv("REDDIT_CLIENT_SECRET", "").strip()
    user_agent = os.getenv("REDDIT_USER_AGENT", "market-agent-sentiment/1.0").strip()
    if not client_id or not client_secret or praw is None:
        return {"score": 50.0, "source": "reddit_unavailable"}

    def _fetch() -> float:
        reddit = praw.Reddit(client_id=client_id, client_secret=client_secret, user_agent=user_agent)
        sub = reddit.subreddit("IndianStockMarket")
        positives = 0
        negatives = 0
        for post in sub.new(limit=40):
            txt = f"{post.title} {post.selftext}".lower()
            if symbol.lower() not in txt:
                continue
            if any(w in txt for w in ("bull", "breakout", "buy", "accumulate", "positive")):
                positives += 1
            if any(w in txt for w in ("bear", "breakdown", "sell", "negative", "crash")):
                negatives += 1
        total = positives + negatives
        if total == 0:
            return 50.0
        score = 50.0 + ((positives - negatives) / total) * 35.0
        return max(0.0, min(100.0, score))

    try:
        score = await asyncio.to_thread(_fetch)
        return {"score": score, "source": "reddit"}
    except Exception:
        return {"score": 50.0, "source": "reddit_error"}


def _aggregate_sentiment(finbert_score: float, groq_analysis: Dict[str, Any], social: Dict[str, Any]) -> float:
    """Weighted aggregate sentiment score."""

    groq_score = float(groq_analysis.get("sentiment_score", finbert_score if groq_analysis else 50.0))
    social_score = float(social.get("score", 50.0))
    return max(0.0, min(100.0, finbert_score * 0.35 + groq_score * 0.55 + social_score * 0.10))


def _apply_news_recency_policy(news_items: List[Dict[str, str]]) -> Dict[str, Any]:
    """Apply recency policy and attach age metadata to each news item."""

    now = dt.datetime.now(IST)
    session = current_market_session(now)
    market_hours = session.status in {"MARKET_OPEN", "OPENING_VOLATILITY"}
    max_age_hours = 6.0 if market_hours else 24.0
    annotated = _annotate_news_age(_sort_news_by_recency(news_items), now)

    fresh = [item for item in annotated if item.get("news_age_hours") is None or float(item.get("news_age_hours", 0.0)) <= max_age_hours]
    stale = [item for item in annotated if item.get("news_age_hours") is not None and float(item.get("news_age_hours", 0.0)) > max_age_hours]

    flags: List[str] = []
    sentiment_confidence = 100.0
    if annotated and not fresh:
        flags.append("STALE_NEWS_WARNING")
        sentiment_confidence -= 30.0
        news_freshness = "STALE"
        return {
            "items_for_scoring": [],
            "items_for_display": annotated[:5],
            "flags": flags,
            "sentiment_confidence": max(0.0, min(100.0, sentiment_confidence)),
            "news_freshness": news_freshness,
        }

    display = fresh[:5] if fresh else annotated[:5]
    unknown_age = sum(1 for item in display if item.get("news_age_hours") is None)
    if not display:
        flags.append("STALE_NEWS_WARNING")
        sentiment_confidence -= 30.0
        news_freshness = "STALE"
    elif stale and fresh:
        news_freshness = "MIXED"
    elif unknown_age > 0:
        news_freshness = "MIXED"
    else:
        news_freshness = "FRESH"

    return {
        "items_for_scoring": fresh[:5],
        "items_for_display": display,
        "flags": flags,
        "sentiment_confidence": max(0.0, min(100.0, sentiment_confidence)),
        "news_freshness": news_freshness,
    }


def _annotate_news_age(news_items: List[Dict[str, str]], now: dt.datetime) -> List[Dict[str, str]]:
    """Attach `news_age_hours` metadata to every item."""

    out: List[Dict[str, str]] = []
    for item in news_items:
        row = dict(item)
        published_dt = _parse_datetime_any(item.get("published"))
        age_hours: float | None = None
        if published_dt is not None:
            age_hours = max(0.0, (now - published_dt).total_seconds() / 3600.0)
        row["news_age_hours"] = round(age_hours, 2) if age_hours is not None else None
        out.append(row)
    return out


def _band(score: float) -> str:
    """Map score to broad sentiment band."""

    if score >= 65:
        return "POSITIVE"
    if score <= 35:
        return "NEGATIVE"
    return "NEUTRAL"


def _build_source(finbert_rows: List[Dict[str, Any]], groq_analysis: Dict[str, Any]) -> str:
    """Human-readable source marker."""

    if finbert_rows and groq_analysis:
        return "finbert+groq"
    if groq_analysis:
        return "groq"
    if finbert_rows:
        return "finbert"
    return "fallback"


def _entry_published_datetime(entry: Any) -> Optional[dt.datetime]:
    """Extract timezone-aware publish datetime from feed entry."""

    parsed = entry.get("published_parsed") or entry.get("updated_parsed")
    if parsed is not None:
        try:
            return dt.datetime(*parsed[:6], tzinfo=dt.timezone.utc).astimezone(IST)
        except Exception:
            pass
    return _parse_datetime_any(entry.get("published") or entry.get("updated"))


def _dedupe_news(items: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Remove duplicate headlines preserving order."""

    seen = set()
    out: List[Dict[str, str]] = []
    for item in items:
        title = item.get("title", "").strip()
        key = title.lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _merge_news_sources(symbol: str, source_lists: List[List[Dict[str, str]]], limit: int) -> List[Dict[str, str]]:
    """Merge prioritized sources, remove fuzzy duplicates, and keep most recent headlines."""

    sym = str(symbol or "").strip().upper()
    merged: List[Dict[str, str]] = []
    for source_items in source_lists:
        filtered = _filter_news_for_symbol(sym, source_items)
        filtered = _sort_news_by_recency(_dedupe_news(filtered))
        for item in filtered:
            if not _is_similar_title(item.get("title", ""), [m.get("title", "") for m in merged]):
                merged.append(item)

    merged = _sort_news_by_recency(merged)
    return merged[: max(1, limit)]


def _sort_news_by_recency(items: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Sort headlines by recency, with source priority as tie-breaker."""

    def _sort_key(item: Dict[str, str]) -> tuple[float, int]:
        published_dt = _parse_datetime_any(item.get("published"))
        ts = published_dt.timestamp() if published_dt else 0.0
        source_rank = NEWS_SOURCE_PRIORITY.get(str(item.get("source", "")).strip(), 99)
        return ts, -source_rank

    return sorted(items, key=_sort_key, reverse=True)


def _is_similar_title(title: str, existing_titles: List[str]) -> bool:
    """Check if a title is near-duplicate of existing list using normalized similarity."""

    norm = _normalize_title(title)
    if not norm:
        return True
    for existing in existing_titles:
        other = _normalize_title(existing)
        if not other:
            continue
        if norm == other:
            return True
        ratio = difflib.SequenceMatcher(a=norm, b=other).ratio()
        if ratio >= 0.9:
            return True
    return False


def _normalize_title(title: str) -> str:
    """Normalize title text for similarity comparisons."""

    text = re.sub(r"[^a-z0-9 ]+", " ", str(title or "").lower())
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _slugify_company_name(name: str) -> str:
    """Create Economic Times style slug from company name."""

    text = re.sub(r"[^a-z0-9 ]+", " ", str(name or "").lower())
    tokens = [t for t in text.split() if t not in {"limited", "ltd", "india"}]
    return "-".join(tokens[:6]) if tokens else "stocks"


def _parse_datetime_any(raw_value: Any) -> Optional[dt.datetime]:
    """Parse multiple timestamp formats to timezone-aware IST datetime."""

    if raw_value is None:
        return None
    if isinstance(raw_value, dt.datetime):
        return raw_value.astimezone(IST) if raw_value.tzinfo else raw_value.replace(tzinfo=IST)

    raw = str(raw_value).strip()
    if not raw:
        return None

    iso_candidate = raw.replace("Z", "+00:00")
    try:
        parsed_iso = dt.datetime.fromisoformat(iso_candidate)
        return parsed_iso.astimezone(IST) if parsed_iso.tzinfo else parsed_iso.replace(tzinfo=IST)
    except Exception:
        pass

    fmts = [
        "%d-%b-%Y %H:%M:%S",
        "%d-%b-%Y %H:%M",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%a, %d %b %Y %H:%M:%S %z",
        "%a, %d %b %Y %H:%M:%S %Z",
    ]
    for fmt in fmts:
        try:
            parsed = dt.datetime.strptime(raw, fmt)
            return parsed.astimezone(IST) if parsed.tzinfo else parsed.replace(tzinfo=IST)
        except Exception:
            continue
    return None


def _filter_news_for_symbol(symbol: str, items: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Keep only headlines directly relevant to the target company/symbol."""

    sym = str(symbol or "").strip().upper()
    rule = COMPANY_FILTERS.get(sym, {})
    include_terms = [t.lower() for t in rule.get("include", [])]
    exclude_terms = [t.lower() for t in rule.get("exclude", [])]
    full_name = COMPANY_FULL_NAME.get(sym, "")
    full_name_terms = [t for t in re.split(r"[^a-z0-9]+", full_name.lower()) if len(t) > 2 and t not in {"limited", "ltd"}]
    include_terms.extend(full_name_terms)
    include_terms = list(dict.fromkeys(include_terms))
    symbol_tokens = [sym.lower(), f"nse:{sym.lower()}", f"bse:{sym.lower()}", f"{sym.lower()}.ns", f"{sym.lower()}.bo"]

    filtered: List[Dict[str, str]] = []
    for item in items:
        title = str(item.get("title", "")).strip()
        desc = str(item.get("description", "")).strip()
        if not title:
            continue
        text = f"{title} {desc}".lower()

        if any(term and term in text for term in exclude_terms):
            continue

        has_symbol_token = any(token in text for token in symbol_tokens)
        has_include_term = any(term in text for term in include_terms) if include_terms else False

        if include_terms:
            if not (has_symbol_token or has_include_term):
                continue
        elif not has_symbol_token:
            continue

        filtered.append(item)
    return filtered


def _finbert_to_0_100(label: str, confidence: float) -> float:
    """Map FinBERT label and confidence to 0-100 scale."""

    conf = max(0.0, min(1.0, float(confidence)))
    low = label.lower()
    if "pos" in low:
        return 50.0 + (conf * 50.0)
    if "neg" in low:
        return 50.0 - (conf * 50.0)
    return 45.0 + (conf * 10.0)


async def call_groq_json(system_prompt: str, user_prompt: str) -> Dict[str, Any]:
    """Call Groq chat completion and parse JSON response."""

    api_key = os.getenv("GROQ_API_KEY", "").strip()
    model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile").strip()
    if not api_key:
        return {}
    await GROQ_LIMITER.wait_turn()
    url = "https://api.groq.com/openai/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        "temperature": 0.1,
        "response_format": {"type": "json_object"},
        "max_tokens": 700,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(url, headers=headers, json=payload)
        if resp.status_code != 200:
            return {}
        content = resp.json()["choices"][0]["message"]["content"]
        return json.loads(content) if isinstance(content, str) else {}
    except Exception:
        return {}


async def call_groq_text(system_prompt: str, user_prompt: str) -> str:
    """Call Groq chat completion returning plain text."""

    api_key = os.getenv("GROQ_API_KEY", "").strip()
    model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile").strip()
    if not api_key:
        return ""
    await GROQ_LIMITER.wait_turn()
    url = "https://api.groq.com/openai/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        "temperature": 0.2,
        "max_tokens": 350,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(url, headers=headers, json=payload)
        if resp.status_code != 200:
            return ""
        return str(resp.json()["choices"][0]["message"]["content"]).strip()
    except Exception:
        return ""


async def call_groq_vision(image_base64_or_data_url: str) -> Dict[str, Any]:
    """Call Groq vision-compatible model for chart interpretation."""

    api_key = os.getenv("GROQ_API_KEY", "").strip()
    model = os.getenv("GROQ_VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct").strip()
    if not api_key:
        return {}
    await GROQ_LIMITER.wait_turn()
    image_url = image_base64_or_data_url
    if not image_url.startswith("data:image"):
        image_url = f"data:image/png;base64,{image_base64_or_data_url}"

    url = "https://api.groq.com/openai/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze this chart and return JSON: patterns (array), trend_direction, key_levels (array)."},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
        "temperature": 0.1,
        "response_format": {"type": "json_object"},
        "max_tokens": 500,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    try:
        async with httpx.AsyncClient(timeout=45.0) as client:
            resp = await client.post(url, headers=headers, json=payload)
        if resp.status_code != 200:
            return {}
        content = resp.json()["choices"][0]["message"]["content"]
        return json.loads(content) if isinstance(content, str) else {}
    except Exception:
        return {}
