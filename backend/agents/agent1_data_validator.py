"""Agent 1: data fetch and validation for multi-timeframe market context."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import pandas as pd

from utils.data_fetcher import (
    current_market_session,
    estimate_index_correlation,
    fetch_all_timeframes,
    fetch_live_price,
    fetch_nifty_data,
    normalize_symbol,
    resolve_yfinance_symbol,
)


@dataclass
class DataContext:
    """Container for validated market data across timeframes."""

    symbol: str
    yfinance_symbol: str
    timeframes: Dict[str, pd.DataFrame]
    nifty_df: pd.DataFrame
    current_price: float
    market_session: str
    session_warning: str
    data_quality_score: float
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    validation: Dict[str, Any] = field(default_factory=dict)
    nifty_correlation: float = 0.0


def run(symbol: str) -> Dict[str, Any]:
    """Execute agent with robust error handling and degraded fallback."""

    try:
        norm = normalize_symbol(symbol)
        yf_symbol = resolve_yfinance_symbol(norm)

        raw_frames = fetch_all_timeframes(norm)
        frames, prep_notes = _prepare_timeframes_for_validation(raw_frames)
        nifty = fetch_nifty_data()
        live = fetch_live_price(yf_symbol)
        session = current_market_session()

        quality, validation, errors, warnings = _validate_frames(frames, raw_frames)
        warnings.extend(prep_notes)
        corr = estimate_index_correlation(frames.get("1d", pd.DataFrame()), nifty)
        price = float(live.get("last_price") or 0.0)

        ctx = DataContext(
            symbol=norm,
            yfinance_symbol=yf_symbol,
            timeframes=frames,
            nifty_df=nifty,
            current_price=price,
            market_session=session.status,
            session_warning=session.warning,
            data_quality_score=quality,
            errors=errors,
            warnings=warnings,
            validation=validation,
            nifty_correlation=corr,
        )
        return {"ok": True, "data_context": ctx, "agent_output": _context_summary(ctx)}
    except Exception as exc:
        degraded = _build_degraded(symbol, str(exc))
        return {"ok": False, "data_context": degraded, "agent_output": _context_summary(degraded)}


def _prepare_timeframes_for_validation(timeframes: Dict[str, pd.DataFrame]) -> tuple[Dict[str, pd.DataFrame], List[str]]:
    """Normalize candle frames and remove zero-volume rows before validation/analysis."""

    cleaned: Dict[str, pd.DataFrame] = {}
    notes: List[str] = []
    for tf, df in timeframes.items():
        if df is None or df.empty:
            cleaned[tf] = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
            continue

        out = df.copy()
        out = out.rename(columns=str.lower)
        for col in ("open", "high", "low", "close", "volume"):
            if col not in out.columns:
                out[col] = pd.NA
            out[col] = pd.to_numeric(out[col], errors="coerce")
        out = out.dropna(subset=["open", "high", "low", "close"])

        before_count = len(out)
        non_zero_mask = out["volume"].fillna(0.0) > 0.0
        out = out.loc[non_zero_mask].copy()
        dropped = before_count - len(out)
        if dropped > 0:
            notes.append(f"{tf}: dropped {dropped} zero-volume candles before validation")

        out = out.sort_index()
        out = out[~out.index.duplicated(keep="last")]
        cleaned[tf] = out
    return cleaned, notes


def _validate_frames(
    timeframes: Dict[str, pd.DataFrame],
    raw_timeframes: Dict[str, pd.DataFrame] | None = None,
) -> tuple[float, Dict[str, Any], List[str], List[str]]:
    """Validate completeness, gaps, and volume consistency."""

    errors: List[str] = []
    warnings: List[str] = []
    validation: Dict[str, Any] = {"timeframes": {}}
    min_required = 20
    expected_freq = {"5m": 5, "15m": 15, "1h": 60, "1d": 1440}
    completeness_hits = 0
    volume_hits = 0
    gap_penalty = 0.0

    for tf, df in timeframes.items():
        raw_df = (raw_timeframes or {}).get(tf, df)
        non_zero_rows = int((raw_df.get("volume", pd.Series(dtype=float)).fillna(0.0) > 0.0).sum()) if not raw_df.empty else 0
        tf_flags = {
            "candles_ok": False,
            "gaps_ok": True,
            "volume_ok": True,
            "candle_count": int(len(df)),
            "raw_candle_count": int(len(raw_df)),
            "non_zero_volume_rows": non_zero_rows,
        }
        if len(df) >= min_required:
            completeness_hits += 1
            tf_flags["candles_ok"] = True
        else:
            errors.append(f"{tf}: less than {min_required} candles")

        if non_zero_rows >= min_required:
            volume_hits += 1
        else:
            tf_flags["volume_ok"] = False
            warnings.append(f"{tf}: insufficient non-zero volume rows for robust indicators")

        gap_df = raw_df if isinstance(raw_df.index, pd.DatetimeIndex) and len(raw_df.index) > 1 else df
        if isinstance(gap_df.index, pd.DatetimeIndex) and len(gap_df.index) > 1:
            deltas = gap_df.index.to_series().diff().dropna()
            min_gap_seconds = (expected_freq.get(tf, 5) + 30) * 60
            if tf in {"5m", "15m", "1h"}:
                # Ignore expected overnight/weekend closures for intraday bars.
                # Penalize only anomalous in-session style gaps.
                max_penalized_seconds = 8 * 60 * 60
                gaps = deltas[(deltas.dt.total_seconds() > min_gap_seconds) & (deltas.dt.total_seconds() < max_penalized_seconds)]
            else:
                gaps = deltas[deltas.dt.total_seconds() > min_gap_seconds]
            if not gaps.empty:
                tf_flags["gaps_ok"] = False
                gap_penalty += min(20.0, float(len(gaps)))
                warnings.append(f"{tf}: {len(gaps)} large time gaps detected")
        validation["timeframes"][tf] = tf_flags

    completeness_score = (completeness_hits / 4.0) * 45.0
    volume_score = (volume_hits / 4.0) * 35.0
    recency_score = 20.0
    score = max(0.0, min(100.0, completeness_score + volume_score + recency_score - gap_penalty))
    validation["score_components"] = {
        "completeness_score": round(completeness_score, 2),
        "volume_score": round(volume_score, 2),
        "recency_score": recency_score,
        "gap_penalty": round(gap_penalty, 2),
    }
    return round(score, 2), validation, errors, warnings


def _build_degraded(symbol: str, reason: str) -> DataContext:
    """Build minimal degraded data context when all else fails."""

    empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    frames = {"5m": empty.copy(), "15m": empty.copy(), "1h": empty.copy(), "1d": empty.copy()}
    session = current_market_session()
    return DataContext(
        symbol=normalize_symbol(symbol) or "UNKNOWN",
        yfinance_symbol=resolve_yfinance_symbol(symbol),
        timeframes=frames,
        nifty_df=empty.copy(),
        current_price=0.0,
        market_session=session.status,
        session_warning=session.warning,
        data_quality_score=0.0,
        errors=[f"agent1_failure: {reason}"],
        warnings=["Degraded mode active due to data fetch failure."],
        validation={"timeframes": {}, "score_components": {}},
        nifty_correlation=0.0,
    )


def _context_summary(ctx: DataContext) -> Dict[str, Any]:
    """Generate serializable summary for frontend transparency."""

    return {
        "symbol": ctx.symbol,
        "yfinance_symbol": ctx.yfinance_symbol,
        "current_price": round(float(ctx.current_price), 4),
        "market_session": ctx.market_session,
        "session_warning": ctx.session_warning,
        "data_quality_score": round(float(ctx.data_quality_score), 2),
        "errors": ctx.errors,
        "warnings": ctx.warnings,
        "nifty_correlation": round(float(ctx.nifty_correlation), 4),
        "validation": ctx.validation,
        "candle_counts": {k: int(len(v)) for k, v in ctx.timeframes.items()},
    }
