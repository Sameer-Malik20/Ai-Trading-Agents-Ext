"""Utilities for market data fetching, caching, symbol normalization, and market timing."""

from __future__ import annotations

import datetime as dt
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yfinance as yf
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")

_CACHE_LOCK = threading.Lock()
_OHLC_CACHE: Dict[Tuple[str, str, str], tuple[float, pd.DataFrame]] = {}
_CACHE_TTL_SECONDS = 60.0

INDEX_SYMBOL_MAP: Dict[str, str] = {
    "NIFTY": "^NSEI",
    "NIFTY50": "^NSEI",
    "NIFTY_50": "^NSEI",
    "NIFTYBANK": "^NSEBANK",
    "BANKNIFTY": "^NSEBANK",
    "INDIAVIX": "^INDIAVIX",
    "VIX": "^INDIAVIX",
}


@dataclass
class MarketSession:
    """Represents the Indian market session classification and advisory flags."""

    status: str
    is_open_for_trading: bool
    warning: str
    current_time_ist: str


def normalize_symbol(raw_symbol: str) -> str:
    """Normalize symbol into NSE/BSE-neutral uppercase equity symbol.

    Examples:
    - ``NSE:RELIANCE`` -> ``RELIANCE``
    - ``reliance.ns`` -> ``RELIANCE``
    - ``BSE:TCS`` -> ``TCS``
    """

    cleaned = (raw_symbol or "").strip().upper()
    if cleaned.startswith("^"):
        return cleaned
    cleaned = cleaned.replace(".NS", "").replace(".BO", "")
    if ":" in cleaned:
        cleaned = cleaned.split(":", 1)[1]
    out = "".join(ch for ch in cleaned if ch.isalnum() or ch in {"-", "_"})
    # TradingView title/url may emit aliases with separators.
    return out.replace("-", "_")


def to_yfinance_symbol(symbol: str) -> str:
    """Convert normalized symbol to yfinance ticker with index support."""

    base = normalize_symbol(symbol)
    if not base:
        return "NIFTYBEES.NS"
    if base.startswith("^"):
        return base
    mapped = INDEX_SYMBOL_MAP.get(base)
    if mapped:
        return mapped
    return f"{base}.NS"


def yfinance_symbol_candidates(symbol: str) -> List[str]:
    """Return candidate yfinance symbols in priority order."""

    base = normalize_symbol(symbol)
    if not base:
        return ["NIFTYBEES.NS"]
    if base.startswith("^"):
        return [base]
    mapped = INDEX_SYMBOL_MAP.get(base)
    if mapped:
        return [mapped]
    return [f"{base}.NS", f"{base}.BO"]


def resolve_yfinance_symbol(symbol: str) -> str:
    """Resolve best yfinance ticker by probing candidates."""

    candidates = yfinance_symbol_candidates(symbol)
    if len(candidates) == 1:
        return candidates[0]

    for candidate in candidates:
        probe = fetch_ohlc(candidate, interval="1d", period="1mo")
        if probe is not None and not probe.empty:
            return candidate
    return candidates[0]


def current_market_session(now_ist: Optional[dt.datetime] = None) -> MarketSession:
    """Classify current Indian market time window."""

    now = now_ist or dt.datetime.now(IST)
    hhmm = now.hour * 100 + now.minute

    if 900 <= hhmm < 915:
        return MarketSession(
            status="PRE_MARKET",
            is_open_for_trading=False,
            warning="Pre-market session (09:00-09:15 IST). Signals may be unstable.",
            current_time_ist=now.isoformat(),
        )
    if 915 <= hhmm < 930:
        return MarketSession(
            status="OPENING_VOLATILITY",
            is_open_for_trading=True,
            warning="Opening volatility window (09:15-09:30 IST). Trade with caution.",
            current_time_ist=now.isoformat(),
        )
    if 930 <= hhmm <= 1530:
        return MarketSession(
            status="MARKET_OPEN",
            is_open_for_trading=True,
            warning="",
            current_time_ist=now.isoformat(),
        )
    return MarketSession(
        status="AFTER_HOURS",
        is_open_for_trading=False,
        warning="After market hours. Using last session data.",
        current_time_ist=now.isoformat(),
    )


def _get_cached_ohlc(symbol: str, interval: str, period: str) -> Optional[pd.DataFrame]:
    """Fetch a cached OHLC dataframe if still fresh."""

    key = (symbol, interval, period)
    with _CACHE_LOCK:
        item = _OHLC_CACHE.get(key)
        if not item:
            return None
        ts, data = item
        if (time.time() - ts) > _CACHE_TTL_SECONDS:
            _OHLC_CACHE.pop(key, None)
            return None
    return data.copy()


def _set_cached_ohlc(symbol: str, interval: str, period: str, df: pd.DataFrame) -> None:
    """Store OHLC dataframe in cache."""

    key = (symbol, interval, period)
    with _CACHE_LOCK:
        _OHLC_CACHE[key] = (time.time(), df.copy())


def fetch_ohlc(symbol: str, interval: str, period: str) -> pd.DataFrame:
    """Fetch OHLCV candles from yfinance with 60-second cache."""

    resolved_interval = "1h" if interval == "60m" else interval

    cached = _get_cached_ohlc(symbol, resolved_interval, period)
    if cached is not None:
        return cached

    df = pd.DataFrame()
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(
            period=period,
            interval=resolved_interval,
            auto_adjust=True,
            actions=False,
            prepost=False,
            repair=False,
        )
    except Exception:
        df = pd.DataFrame()

    if df is None or df.empty:
        try:
            df = yf.download(
                tickers=symbol,
                period=period,
                interval=resolved_interval,
                auto_adjust=True,
                progress=False,
                prepost=False,
                threads=False,
            )
        except Exception:
            df = pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [str(c[0]).lower() for c in df.columns]
    else:
        df.columns = [str(c).lower() for c in df.columns]

    for col in ("open", "high", "low", "close", "volume"):
        if col not in df.columns:
            df[col] = pd.NA
    df = df[["open", "high", "low", "close", "volume"]].copy()
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Major index tickers on Yahoo often have zero/empty intraday volume.
    # Use neutral synthetic volume so downstream validation/indicators remain functional.
    if symbol.startswith("^"):
        positive_vol = (df["volume"].fillna(0.0) > 0.0).sum()
        if int(positive_vol) == 0:
            df["volume"] = 1.0

    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]

    if isinstance(df.index, pd.DatetimeIndex):
        try:
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC").tz_convert(IST)
            else:
                df.index = df.index.tz_convert(IST)
        except Exception:
            pass

    _set_cached_ohlc(symbol, resolved_interval, period, df)
    return df


def fetch_all_timeframes(symbol: str) -> Dict[str, pd.DataFrame]:
    """Fetch required timeframes for the symbol."""

    yf_symbol = resolve_yfinance_symbol(symbol)
    one_hour_period = "60d" if yf_symbol.startswith("^") else "30d"
    return {
        "5m": fetch_ohlc(yf_symbol, interval="5m", period="5d"),
        "15m": fetch_ohlc(yf_symbol, interval="15m", period="5d"),
        "1h": fetch_ohlc(yf_symbol, interval="1h", period=one_hour_period),
        "1d": fetch_ohlc(yf_symbol, interval="1d", period="1y"),
    }


def fetch_nifty_data() -> pd.DataFrame:
    """Fetch Nifty50 data for correlation context."""

    return fetch_ohlc("^NSEI", interval="1d", period="1y")


def fetch_market_overview() -> dict:
    """Fetch Nifty50, BankNifty, and India VIX snapshot."""

    nifty = fetch_live_price("^NSEI")
    banknifty = fetch_live_price("^NSEBANK")
    india_vix = fetch_live_price("^INDIAVIX")
    return {"nifty50": nifty, "banknifty": banknifty, "india_vix": india_vix}


def fetch_live_price(symbol: str) -> dict:
    """Fetch latest price metadata from yfinance fast_info with fallback."""

    ticker = yf.Ticker(symbol)
    try:
        info = ticker.fast_info
        return {
            "symbol": symbol,
            "last_price": float(info.get("lastPrice") or 0.0),
            "day_high": float(info.get("dayHigh") or 0.0),
            "day_low": float(info.get("dayLow") or 0.0),
            "previous_close": float(info.get("previousClose") or 0.0),
            "volume": float(info.get("lastVolume") or 0.0),
            "error": "",
        }
    except Exception as exc:
        try:
            hist = ticker.history(period="2d", interval="1d")
            if hist is not None and not hist.empty:
                row = hist.iloc[-1]
                return {
                    "symbol": symbol,
                    "last_price": float(row.get("Close", 0.0)),
                    "day_high": float(row.get("High", 0.0)),
                    "day_low": float(row.get("Low", 0.0)),
                    "previous_close": float(hist.iloc[-2]["Close"]) if len(hist) > 1 else 0.0,
                    "volume": float(row.get("Volume", 0.0)),
                    "error": f"fast_info_failed: {exc}",
                }
        except Exception:
            pass
        return {
            "symbol": symbol,
            "last_price": 0.0,
            "day_high": 0.0,
            "day_low": 0.0,
            "previous_close": 0.0,
            "volume": 0.0,
            "error": f"live_price_unavailable: {exc}",
        }


def estimate_index_correlation(symbol_df: pd.DataFrame, index_df: pd.DataFrame) -> float:
    """Estimate rolling daily return correlation between symbol and Nifty50."""

    if symbol_df.empty or index_df.empty:
        return 0.0
    try:
        s_ret = symbol_df["close"].pct_change().dropna()
        i_ret = index_df["close"].pct_change().dropna()
        joined = pd.concat([s_ret, i_ret], axis=1, join="inner").dropna().tail(90)
        if joined.empty:
            return 0.0
        corr = joined.iloc[:, 0].corr(joined.iloc[:, 1])
        if pd.isna(corr):
            return 0.0
        return float(corr)
    except Exception:
        return 0.0
