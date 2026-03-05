"""Options chain analyzer with Angel One primary and safe fallbacks."""

from __future__ import annotations

import copy
import json
import logging
import math
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yfinance as yf
from dotenv import load_dotenv

logger = logging.getLogger("market-agent.options")

INDEX_SYMBOLS = {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"}
CACHE_TTL_SECONDS = 300

_OPTIONS_CACHE: Dict[str, Dict[str, Any]] = {}
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")
load_dotenv(BASE_DIR.parent / ".env")


def analyze_options(symbol: str, spot_price: float) -> Dict[str, Any]:
    """Analyze options data with Angel One primary and yfinance equity backup."""

    norm_symbol = str(symbol or "").upper().strip()
    cache_key = f"options_{norm_symbol}"
    cached = _read_cache(cache_key)
    if cached is not None:
        return cached

    errors: List[str] = []

    try:
        records, source_meta = _fetch_from_angelone(norm_symbol, spot_price)
        result = _build_response_from_records(norm_symbol, spot_price, records, source_meta)
        _write_cache(cache_key, result)
        return copy.deepcopy(result)
    except Exception as exc:
        errors.append(f"angelone_failed: {exc}")
        logger.warning("options_angelone_failed symbol=%s error=%s", norm_symbol, exc)

    if norm_symbol not in INDEX_SYMBOLS:
        try:
            records, source_meta = _fetch_from_yfinance(norm_symbol, spot_price)
            result = _build_response_from_records(norm_symbol, spot_price, records, source_meta)
            _write_cache(cache_key, result)
            return copy.deepcopy(result)
        except Exception as exc:
            errors.append(f"yfinance_failed: {exc}")
            logger.warning("options_yfinance_failed symbol=%s error=%s", norm_symbol, exc)
    else:
        errors.append("yfinance_skipped_for_index")

    fallback = _neutral_fallback(spot_price=spot_price, error=" | ".join(errors)[:800])
    _write_cache(cache_key, fallback)
    return copy.deepcopy(fallback)


def _fetch_from_angelone(symbol: str, spot_price: float) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Fetch option-chain data from Angel One SmartAPI."""

    api_key = os.getenv("ANGELONE_API_KEY", "").strip()
    client_id = os.getenv("ANGELONE_CLIENT_ID", "").strip()
    mpin = os.getenv("ANGELONE_MPIN", "").strip()
    totp_key = os.getenv("ANGELONE_TOTP_KEY", "").strip()

    if not all([api_key, client_id, mpin, totp_key]):
        raise RuntimeError("angelone_credentials_missing")

    try:
        import pyotp
        from SmartApi import SmartConnect
        import logzero
    except Exception as exc:
        raise RuntimeError(f"angelone_dependency_missing: {exc}") from exc

    logzero.loglevel(logging.CRITICAL)
    obj = SmartConnect(api_key=api_key)
    try:
        totp = pyotp.TOTP(totp_key).now()
        session = obj.generateSession(client_id, mpin, totp)
        if not isinstance(session, dict) or not session.get("status"):
            reason = ""
            if isinstance(session, dict):
                reason = str(session.get("message") or session.get("errorcode") or "").strip()
            raise RuntimeError(f"angelone_login_failed{(': ' + reason) if reason else ''}")

        refresh_token = str(session.get("data", {}).get("refreshToken", "")).strip()
        if refresh_token:
            try:
                obj.getProfile(refresh_token)
            except Exception as exc:
                logger.warning("angelone_get_profile_failed symbol=%s error=%s", symbol, exc)

        call_errors: List[str] = []

        expiry_probe_params = {"name": symbol, "expirydate": ""}
        expiry_response = _call_angel_option_api(obj=obj, params=expiry_probe_params, call_errors=call_errors)
        print(f"EXPIRY_LIST: {expiry_response}")
        time.sleep(1.0)

        ordered_formats = _build_angel_expiry_candidates(symbol=symbol, expiry_response=expiry_response)

        response: Dict[str, Any] = {}
        working_format = ""
        for fmt in ordered_formats:
            try:
                params = {"name": symbol, "expirydate": fmt}
                response = _call_angel_option_api(obj=obj, params=params, call_errors=call_errors)
                message = str(response.get("message", ""))
                print(f"FORMAT_TRIED: {fmt} → {message}")
                print(f"FULL_RESPONSE: {json.dumps(response, indent=2, default=str)}")
                if response.get("status") is True:
                    working_format = fmt
                    print(f"WORKING_FORMAT: {fmt}")
                    break
            except Exception as exc:
                print(f"FORMAT_ERROR: {fmt} → {exc}")
                call_errors.append(f"format_{fmt}:{exc}")
                continue
            finally:
                time.sleep(1.0)

        if not isinstance(response, dict) or not response.get("status"):
            detail = " | ".join(call_errors) if call_errors else str(response)
            raise RuntimeError(f"angelone_optionchain_failed: {detail[:700]}")

        records = _records_from_angelone_response(response)

        if not records:
            raise RuntimeError("angelone_empty_records")

        source_meta = {
            "source": "ANGELONE",
            "endpoint": symbol,
            "timestamp": working_format or "UNKNOWN",
            "underlying_value": _to_float(spot_price),
        }
        return records, source_meta
    finally:
        try:
            obj.terminateSession(client_id)
        except Exception:
            pass


def _get_equity_token(obj: Any, symbol: str) -> str:
    """Resolve equity token via Angel One search API."""

    try:
        result = obj.searchScrip("NSE", symbol)
        if isinstance(result, dict) and result.get("status") and result.get("data"):
            for item in result.get("data", []):
                token = str(item.get("symboltoken", "")).strip()
                trading_symbol = str(item.get("tradingsymbol", "")).upper()
                if token and symbol in trading_symbol:
                    return token
    except Exception:
        pass
    return ""


def _get_nearest_expiry(obj: Any, symbol: str, exchange: str) -> str:
    """Return nearest Thursday expiry in DDMMMYYYY format."""

    today = datetime.now().date()
    days_ahead = 3 - today.weekday()
    if days_ahead <= 0:
        days_ahead += 7
    expiry = today + timedelta(days=days_ahead)
    return expiry.strftime("%d%b%Y").upper()


def _call_angel_option_api(obj: Any, params: Dict[str, Any], call_errors: List[str]) -> Dict[str, Any]:
    """Call available Angel option chain methods in fallback order."""

    response: Dict[str, Any] = {}
    if hasattr(obj, "optionGreeks"):
        try:
            response = obj.optionGreeks(params)  # type: ignore[attr-defined]
            if isinstance(response, dict):
                return response
        except Exception as exc:
            call_errors.append(f"optionGreeks:{exc}")

    try:
        response = obj.optionGreek(params)
        if isinstance(response, dict):
            return response
    except Exception as exc:
        call_errors.append(f"optionGreek:{exc}")

    if hasattr(obj, "option_chain"):
        try:
            response = obj.option_chain(params)  # type: ignore[attr-defined]
            if isinstance(response, dict):
                return response
        except Exception as exc:
            call_errors.append(f"option_chain:{exc}")

    return response if isinstance(response, dict) else {}


def _extract_expiry_formats(response: Dict[str, Any]) -> List[str]:
    """Extract possible expiry format strings from probe response."""

    if not isinstance(response, dict):
        return []

    found: List[str] = []

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if isinstance(value, (dict, list)):
                    visit(value)
                elif isinstance(value, str) and "expiry" in str(key).lower():
                    txt = value.strip()
                    if txt and txt not in found:
                        found.append(txt)
        elif isinstance(node, list):
            for item in node:
                visit(item)

    visit(response)
    return found


def _build_angel_expiry_candidates(symbol: str, expiry_response: Dict[str, Any]) -> List[str]:
    """Build ordered expiry candidates; prioritize symbol expiry weekday."""

    ordered: List[str] = []

    for candidate in _extract_expiry_formats(expiry_response):
        text = str(candidate or "").strip().upper()
        if text and text not in ordered:
            ordered.append(text)

    today = datetime.now().date()
    # Observed mapping: NIFTY Tuesdays are valid on this account.
    symbol_weekday = {
        "NIFTY": 1,       # Tuesday
        "BANKNIFTY": 2,   # Wednesday
        "FINNIFTY": 1,    # Tuesday
        "MIDCPNIFTY": 0,  # Monday
    }.get(symbol, 3)      # default Thursday for stock options

    # Primary: next 8 occurrences of preferred weekday.
    for i in range(1, 57):
        d = today + timedelta(days=i)
        if d.weekday() == symbol_weekday:
            fmt = d.strftime("%d%b%Y").upper()
            if fmt not in ordered:
                ordered.append(fmt)

    # Fallback: dense next 14 calendar days.
    for i in range(1, 15):
        fmt = (today + timedelta(days=i)).strftime("%d%b%Y").upper()
        if fmt not in ordered:
            ordered.append(fmt)

    return ordered


def _records_from_angelone_response(response: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Normalize Angel One optionGreek response into CE/PE per strike records."""

    by_strike: Dict[float, Dict[str, Any]] = {}
    for item in response.get("data", []) or []:
        strike = _to_float(item.get("strikePrice") or item.get("strike"))
        if strike <= 0:
            continue

        entry = by_strike.setdefault(
            float(strike),
            {
                "strikePrice": float(strike),
                "CE": {"openInterest": 0.0, "impliedVolatility": 0.0},
                "PE": {"openInterest": 0.0, "impliedVolatility": 0.0},
            },
        )

        option_type = str(item.get("optionType") or "").upper().strip()
        oi_val = _to_float(
            item.get("openInterest")
            or item.get("ceOpenInterest")
            or item.get("peOpenInterest")
            or item.get("tradeVolume")
        )
        iv_val = _to_float(
            item.get("impliedVolatility")
            or item.get("ceIV")
            or item.get("peIV")
            or item.get("callIV")
            or item.get("putIV")
        )

        if option_type.startswith("C"):
            entry["CE"]["openInterest"] = oi_val
            entry["CE"]["impliedVolatility"] = iv_val
        elif option_type.startswith("P"):
            entry["PE"]["openInterest"] = oi_val
            entry["PE"]["impliedVolatility"] = iv_val
        else:
            # Unknown option side; keep both untouched.
            continue

    return [by_strike[k] for k in sorted(by_strike.keys())]


def _fetch_from_yfinance(symbol: str, spot_price: float) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Fetch options chain from yfinance and normalize to internal structure."""

    candidates = [f"{symbol}.NS"]
    errors: List[str] = []
    today = datetime.now(timezone.utc).date()
    selected_symbol = ""
    selected_expirations: List[str] = []

    for candidate in candidates:
        try:
            ticker = yf.Ticker(candidate)
            expirations = list(ticker.options or [])
            if expirations:
                selected_symbol = candidate
                selected_expirations = expirations
                break
            errors.append(f"{candidate}:no_expiry_dates_for_{candidate}")
        except Exception as exc:
            errors.append(f"{candidate}:{exc}")
            logger.warning("options_yf_candidate_failed symbol=%s yf_symbol=%s error=%s", symbol, candidate, exc)

    if not selected_symbol:
        raise RuntimeError(" | ".join(errors)[:600] if errors else f"no_yfinance_candidates_for_{symbol}")

    try:
        ticker = yf.Ticker(selected_symbol)
        expirations = selected_expirations
        if not expirations:
            expirations = list(ticker.options or [])
        if not expirations:
            raise RuntimeError(f"no_expiry_dates_for_{selected_symbol}")

        near_month_expiries: List[str] = []
        for expiry in expirations:
            try:
                expiry_date = datetime.strptime(str(expiry), "%Y-%m-%d").date()
            except Exception:
                continue
            if 0 <= (expiry_date - today).days <= 30:
                near_month_expiries.append(str(expiry))
        nearest_expiry = near_month_expiries[0] if near_month_expiries else str(expirations[0])

        chain = ticker.option_chain(nearest_expiry)
        calls = chain.calls
        puts = chain.puts
        if calls.empty or puts.empty:
            raise RuntimeError(f"empty_chain_for_{selected_symbol}")

        call_oi_values = [_to_float(v) for v in calls.get("openInterest", [])]
        put_oi_values = [_to_float(v) for v in puts.get("openInterest", [])]
        oi_multiplier = _oi_multiplier(symbol=symbol, call_oi_values=call_oi_values, put_oi_values=put_oi_values)

        all_strikes = sorted(
            set([_to_float(v) for v in calls["strike"].tolist()] + [_to_float(v) for v in puts["strike"].tolist()])
        )
        records: List[Dict[str, Any]] = []
        for strike in all_strikes:
            if strike <= 0:
                continue

            call_row = calls[calls["strike"] == strike]
            put_row = puts[puts["strike"] == strike]

            ce_oi = _to_float(call_row["openInterest"].iloc[0]) * oi_multiplier if not call_row.empty else 0.0
            pe_oi = _to_float(put_row["openInterest"].iloc[0]) * oi_multiplier if not put_row.empty else 0.0
            ce_iv = _to_float(call_row["impliedVolatility"].iloc[0]) * 100.0 if not call_row.empty else 0.0
            pe_iv = _to_float(put_row["impliedVolatility"].iloc[0]) * 100.0 if not put_row.empty else 0.0

            records.append(
                {
                    "strikePrice": float(strike),
                    "CE": {"openInterest": float(ce_oi), "impliedVolatility": float(ce_iv)},
                    "PE": {"openInterest": float(pe_oi), "impliedVolatility": float(pe_iv)},
                }
            )

        if not records:
            raise RuntimeError(f"no_normalized_records_for_{selected_symbol}")

        total_ce_oi = sum(_to_float(r.get("CE", {}).get("openInterest")) for r in records)
        total_pe_oi = sum(_to_float(r.get("PE", {}).get("openInterest")) for r in records)
        if total_ce_oi == 0.0 and total_pe_oi == 0.0:
            raise RuntimeError("all_oi_zero_likely_index_symbol")

        source_meta = {
            "source": "YFINANCE",
            "endpoint": selected_symbol,
            "timestamp": str(nearest_expiry),
            "underlying_value": _to_float(spot_price),
            "oi_multiplier": oi_multiplier,
        }
        return records, source_meta
    except Exception as exc:
        raise RuntimeError(f"{selected_symbol}:{exc}") from exc


def _oi_multiplier(symbol: str, call_oi_values: List[float], put_oi_values: List[float]) -> float:
    """Scale yfinance OI for index contracts when values appear too small."""

    if str(symbol or "").upper().strip() != "NIFTY":
        return 1.0

    non_zero = sorted([v for v in (call_oi_values + put_oi_values) if v > 0])
    if not non_zero:
        return 1.0
    median_oi = non_zero[len(non_zero) // 2]
    return 50.0 if median_oi < 100.0 else 1.0


def _build_response_from_records(
    symbol: str,
    spot_price: float,
    records: List[Dict[str, Any]],
    source_meta: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute PCR, max pain, key levels, and options score from records."""

    if not records:
        raise RuntimeError("empty_records")

    total_call_oi = sum(_to_float(r.get("CE", {}).get("openInterest")) for r in records)
    total_put_oi = sum(_to_float(r.get("PE", {}).get("openInterest")) for r in records)
    if total_call_oi <= 0:
        raise RuntimeError("invalid_total_call_oi")

    pcr = round(total_put_oi / total_call_oi, 4)
    signal, options_score = _pcr_signal_score(pcr)

    max_pain = _calculate_max_pain(records)
    key_resistance = _key_level(records, side="CE")
    key_support = _key_level(records, side="PE")
    iv_rank, iv_state = _iv_rank_state(records)

    raw_summary = {
        "source": source_meta.get("source", "UNKNOWN"),
        "endpoint": source_meta.get("endpoint", ""),
        "timestamp": source_meta.get("timestamp", ""),
        "underlying_value": _to_float(source_meta.get("underlying_value")),
        "records_count": len(records),
        "total_call_oi": round(total_call_oi, 2),
        "total_put_oi": round(total_put_oi, 2),
        "atm_strike": _nearest_strike(records, spot_price),
    }

    return {
        "options_score": float(options_score),
        "pcr": float(pcr),
        "max_pain": float(max_pain),
        "key_resistance": float(key_resistance),
        "key_support": float(key_support),
        "signal": signal,
        "iv_rank": float(iv_rank),
        "iv_state": iv_state,
        "data_unavailable": False,
        "error": "",
        "raw_summary": raw_summary,
    }


def _pcr_signal_score(pcr: float) -> Tuple[str, float]:
    """Map PCR values to directional signal and score."""

    if pcr >= 1.5:
        return "STRONGLY_BULLISH", 85.0
    if pcr >= 1.2:
        return "BULLISH", 70.0
    if pcr >= 1.0:
        return "MILDLY_BULLISH", 60.0
    if pcr >= 0.8:
        return "MILDLY_BEARISH", 40.0
    if pcr >= 0.5:
        return "BEARISH", 30.0
    return "STRONGLY_BEARISH", 15.0


def _calculate_max_pain(records: List[Dict[str, Any]]) -> float:
    """Return strike where option writers lose minimum payout."""

    strikes = sorted({_to_float(r.get("strikePrice")) for r in records if _to_float(r.get("strikePrice")) > 0})
    if not strikes:
        return 0.0

    min_payout = math.inf
    best_strike = strikes[0]
    for settle in strikes:
        payout = 0.0
        for row in records:
            strike = _to_float(row.get("strikePrice"))
            call_oi = _to_float(row.get("CE", {}).get("openInterest"))
            put_oi = _to_float(row.get("PE", {}).get("openInterest"))
            payout += call_oi * max(0.0, settle - strike)
            payout += put_oi * max(0.0, strike - settle)
        if payout < min_payout:
            min_payout = payout
            best_strike = settle
    return float(best_strike)


def _key_level(records: List[Dict[str, Any]], side: str) -> float:
    """Return strike with highest OI for selected side."""

    max_oi = -1.0
    level = 0.0
    for row in records:
        strike = _to_float(row.get("strikePrice"))
        oi = _to_float(row.get(side, {}).get("openInterest"))
        if oi > max_oi:
            max_oi = oi
            level = strike
    return float(level)


def _nearest_strike(records: List[Dict[str, Any]], spot_price: float) -> float:
    """Return nearest available strike to spot."""

    spot = _to_float(spot_price)
    strikes = [_to_float(r.get("strikePrice")) for r in records if _to_float(r.get("strikePrice")) > 0]
    if not strikes:
        return 0.0
    return float(min(strikes, key=lambda x: abs(x - spot)))


def _iv_rank_state(records: List[Dict[str, Any]]) -> Tuple[float, str]:
    """Estimate IV rank/state from combined chain IV samples."""

    iv_values: List[float] = []
    for row in records:
        ce_iv = _to_float(row.get("CE", {}).get("impliedVolatility"))
        pe_iv = _to_float(row.get("PE", {}).get("impliedVolatility"))
        if ce_iv > 0:
            iv_values.append(ce_iv)
        if pe_iv > 0:
            iv_values.append(pe_iv)

    if len(iv_values) < 3:
        return 50.0, "UNKNOWN"

    iv_values.sort()
    current_iv = sum(iv_values[-5:]) / min(5, len(iv_values))
    min_iv = iv_values[0]
    max_iv = iv_values[-1]
    if max_iv - min_iv <= 1e-9:
        rank = 50.0
    else:
        rank = ((current_iv - min_iv) / (max_iv - min_iv)) * 100.0
    rank = float(max(0.0, min(100.0, rank)))

    if rank >= 75:
        state = "HIGH"
    elif rank <= 25:
        state = "LOW"
    else:
        state = "NORMAL"
    return rank, state


def _read_cache(cache_key: str) -> Optional[Dict[str, Any]]:
    """Read cache if TTL has not expired."""

    item = _OPTIONS_CACHE.get(cache_key)
    if not item:
        return None
    ts = item.get("ts")
    if not isinstance(ts, datetime):
        return None
    if datetime.now(timezone.utc) - ts > timedelta(seconds=CACHE_TTL_SECONDS):
        _OPTIONS_CACHE.pop(cache_key, None)
        return None
    return copy.deepcopy(item.get("data"))


def _write_cache(cache_key: str, data: Dict[str, Any]) -> None:
    """Write options result into in-memory TTL cache."""

    _OPTIONS_CACHE[cache_key] = {"ts": datetime.now(timezone.utc), "data": copy.deepcopy(data)}


def _neutral_fallback(spot_price: float, error: str) -> Dict[str, Any]:
    """Neutral output for degraded mode, never raising pipeline exceptions."""

    spot = _to_float(spot_price)
    return {
        "options_score": 50.0,
        "pcr": 1.0,
        "max_pain": float(spot),
        "key_resistance": float(spot * 1.01 if spot else 0.0),
        "key_support": float(spot * 0.99 if spot else 0.0),
        "signal": "NEUTRAL",
        "iv_rank": 50.0,
        "iv_state": "UNKNOWN",
        "data_unavailable": True,
        "error": error,
        "raw_summary": {},
    }


def _to_float(value: Any) -> float:
    """Convert numeric-like values to float with safe fallback."""

    try:
        if value is None:
            return 0.0
        return float(value)
    except Exception:
        return 0.0


def _utc_now_iso() -> str:
    """UTC ISO8601 timestamp helper."""

    return datetime.now(timezone.utc).isoformat()
