"""NSE options chain analysis using direct NSE website APIs."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests


def analyze_options(symbol: str, spot_price: float) -> Dict[str, Any]:
    """Analyze options chain for PCR, max pain, OI levels, and IV diagnostics.

    Uses direct NSE web APIs with session-cookie flow:
    1) https://www.nseindia.com
    2) https://www.nseindia.com/option-chain
    3) https://www.nseindia.com/api/option-chain-equities?symbol={symbol}
    """

    default = {
        "options_score": 50.0,
        "pcr": 1.0,
        "max_pain": spot_price,
        "key_resistance": spot_price * 1.01 if spot_price else 0.0,
        "key_support": spot_price * 0.99 if spot_price else 0.0,
        "signal": "NEUTRAL",
        "iv_rank": 50.0,
        "iv_state": "NEUTRAL",
        "data_unavailable": True,
        "error": "",
        "raw_summary": {},
    }

    try:
        chain = _fetch_option_chain(symbol)
        if not chain:
            default["error"] = "option_chain_empty"
            return default

        rows = _extract_option_rows(chain)
        if not rows:
            default["error"] = "option_rows_empty"
            return default

        pcr = _compute_pcr(chain, rows)
        support, resistance = _key_oi_levels(rows)
        max_pain = _compute_max_pain(rows)
        iv_rank, iv_state = _iv_rank_and_state(rows)
        signal = _pcr_signal(pcr)
        options_score = _score_options(pcr, spot_price, max_pain, iv_rank)

        return {
            "options_score": round(float(options_score), 2),
            "pcr": round(float(pcr), 4),
            "max_pain": round(float(max_pain), 2),
            "key_resistance": round(float(resistance), 2),
            "key_support": round(float(support), 2),
            "signal": signal,
            "iv_rank": round(float(iv_rank), 2),
            "iv_state": iv_state,
            "data_unavailable": False,
            "error": "",
            "raw_summary": {
                "rows": len(rows),
                "highest_call_oi_strike": resistance,
                "highest_put_oi_strike": support,
            },
        }
    except Exception as exc:
        default["error"] = f"options_analysis_failed: {exc}"
        return default


def _fetch_option_chain(symbol: str) -> Dict[str, Any]:
    """Fetch option chain from NSE with retries and mandatory delay flow."""

    sym = str(symbol or "").strip().upper()
    if not sym:
        return {}

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json,text/plain,*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nseindia.com/option-chain",
        "Connection": "keep-alive",
    }

    landing_url = "https://www.nseindia.com"
    option_chain_page_url = "https://www.nseindia.com/option-chain"
    api_url = f"https://www.nseindia.com/api/option-chain-equities?symbol={sym}"

    for _ in range(3):
        session = requests.Session()
        try:
            session.get(landing_url, headers=headers, timeout=12)
            time.sleep(2.0)

            session.get(option_chain_page_url, headers=headers, timeout=12)
            time.sleep(2.0)

            response = session.get(api_url, headers=headers, timeout=15)
            if response.status_code != 200:
                time.sleep(2.0)
                continue

            payload = response.json()
            if isinstance(payload, dict) and payload:
                return payload
        except Exception:
            time.sleep(2.0)
    return {}


def _extract_option_rows(chain: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract normalized option rows from NSE option-chain response."""

    records = chain.get("records", {}) if isinstance(chain, dict) else {}
    filtered = chain.get("filtered", {}) if isinstance(chain, dict) else {}
    data = records.get("data", []) if isinstance(records, dict) else []
    if not data and isinstance(filtered, dict):
        data = filtered.get("data", []) or []
    rows: List[Dict[str, Any]] = []
    for item in data:
        strike = float(item.get("strikePrice", 0.0) or 0.0)
        ce = item.get("CE", {}) or {}
        pe = item.get("PE", {}) or {}
        rows.append(
            {
                "strike": strike,
                "ce_oi": float(ce.get("openInterest", 0.0) or 0.0),
                "pe_oi": float(pe.get("openInterest", 0.0) or 0.0),
                "ce_change_oi": float(ce.get("changeinOpenInterest", 0.0) or 0.0),
                "pe_change_oi": float(pe.get("changeinOpenInterest", 0.0) or 0.0),
                "ce_iv": float(ce.get("impliedVolatility", 0.0) or 0.0),
                "pe_iv": float(pe.get("impliedVolatility", 0.0) or 0.0),
                "ce_ltp": float(ce.get("lastPrice", 0.0) or 0.0),
                "pe_ltp": float(pe.get("lastPrice", 0.0) or 0.0),
            }
        )
    return [r for r in rows if r["strike"] > 0]


def _compute_pcr(chain: Dict[str, Any], rows: List[Dict[str, Any]]) -> float:
    """Compute Put-Call Ratio from totals, fallback to row sums."""

    records = chain.get("records", {}) if isinstance(chain, dict) else {}
    filtered = chain.get("filtered", {}) if isinstance(chain, dict) else {}

    total_put = _as_float(records.get("totalPutOI"))
    total_call = _as_float(records.get("totalCallOI"))
    if total_put <= 0 or total_call <= 0:
        total_put = _as_float(filtered.get("totalPEOpenInterest"))
        total_call = _as_float(filtered.get("totalCEOpenInterest"))
    if total_put <= 0 or total_call <= 0:
        total_put = float(sum(r["pe_oi"] for r in rows))
        total_call = float(sum(r["ce_oi"] for r in rows))
    if total_call <= 0:
        return 1.0
    return total_put / total_call


def _key_oi_levels(rows: List[Dict[str, Any]]) -> Tuple[float, float]:
    """Find strongest support and resistance by maximum OI."""

    if not rows:
        return 0.0, 0.0
    support_row = max(rows, key=lambda x: x["pe_oi"])
    resistance_row = max(rows, key=lambda x: x["ce_oi"])
    return float(support_row["strike"]), float(resistance_row["strike"])


def _compute_max_pain(rows: List[Dict[str, Any]]) -> float:
    """Approximate max pain by minimum payout objective over strikes."""

    if not rows:
        return 0.0
    strikes = np.array(sorted({r["strike"] for r in rows}), dtype=float)
    losses: Dict[float, float] = {}
    for candidate in strikes:
        total_loss = 0.0
        for r in rows:
            ce_loss = max(0.0, candidate - r["strike"]) * r["ce_oi"]
            pe_loss = max(0.0, r["strike"] - candidate) * r["pe_oi"]
            total_loss += ce_loss + pe_loss
        losses[float(candidate)] = total_loss
    return min(losses, key=losses.get) if losses else float(strikes[len(strikes) // 2])


def _iv_rank_and_state(rows: List[Dict[str, Any]]) -> Tuple[float, str]:
    """Estimate IV rank-like proxy and classify IV state."""

    iv_values: List[float] = []
    for r in rows:
        if r["ce_iv"] > 0:
            iv_values.append(r["ce_iv"])
        if r["pe_iv"] > 0:
            iv_values.append(r["pe_iv"])
    if not iv_values:
        return 50.0, "UNKNOWN"
    avg_iv = float(np.mean(iv_values))
    iv_min = float(np.min(iv_values))
    iv_max = float(np.max(iv_values))
    span = max(iv_max - iv_min, 1e-6)
    iv_rank = max(0.0, min(100.0, ((avg_iv - iv_min) / span) * 100.0))
    if iv_rank > 70:
        state = "HIGH"
    elif iv_rank > 50:
        state = "ELEVATED"
    else:
        state = "NORMAL"
    return iv_rank, state


def _pcr_signal(pcr: float) -> str:
    """Map PCR to directional signal bands."""

    if pcr > 1.3:
        return "STRONGLY_BULLISH"
    if 1.0 <= pcr <= 1.3:
        return "MILD_BULLISH"
    if 0.7 <= pcr < 1.0:
        return "MILD_BEARISH"
    return "STRONGLY_BEARISH"


def _score_options(pcr: float, spot: float, max_pain: float, iv_rank: float) -> float:
    """Compute a normalized options score (0-100)."""

    pcr_component = 50.0
    if pcr > 1.3:
        pcr_component = 75.0
    elif pcr >= 1.0:
        pcr_component = 62.0
    elif pcr >= 0.7:
        pcr_component = 42.0
    else:
        pcr_component = 28.0

    if spot > 0:
        dist = abs(spot - max_pain) / spot
        pain_component = max(20.0, 100.0 - dist * 500.0)
    else:
        pain_component = 50.0

    iv_penalty = max(0.0, iv_rank - 50.0) * 0.4
    score = (pcr_component * 0.55) + (pain_component * 0.45) - iv_penalty
    return max(0.0, min(100.0, score))


def _as_float(value: Any) -> float:
    """Best-effort float parser with 0 fallback."""

    try:
        return float(value or 0.0)
    except Exception:
        return 0.0
