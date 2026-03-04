"""Agent 2: quantitative indicator engine and MTF confluence scoring."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from agents.agent1_data_validator import DataContext
from utils.indicators import (
    apply_indicators,
    compute_pivots,
    detect_market_regime,
    round_number_levels,
    swing_levels,
    timeframe_confluence_score,
)


def run(data_context: DataContext) -> Dict[str, Any]:
    """Run quant analysis across all timeframes."""

    try:
        indicators_by_tf: Dict[str, pd.DataFrame] = {}
        snapshot_by_tf: Dict[str, Dict[str, float]] = {}
        confluence_checks: Dict[str, Dict[str, bool]] = {}
        tf_scores: Dict[str, int] = {}

        for tf, df in data_context.timeframes.items():
            prepared = _prepare_for_indicators(df)
            enriched = apply_indicators(prepared)
            indicators_by_tf[tf] = enriched
            if enriched.empty:
                snapshot_by_tf[tf] = {}
                tf_scores[tf] = 0
                confluence_checks[tf] = {}
                continue
            latest = enriched.iloc[-1]
            score, checks = timeframe_confluence_score(latest)
            tf_scores[tf] = int(score)
            confluence_checks[tf] = checks
            snapshot_by_tf[tf] = _extract_snapshot(enriched)

        mtf_score = int(sum(tf_scores.values()))
        mtf_confluence = max(0.0, min(100.0, (mtf_score / 40.0) * 100.0))
        direction = _direction_from_mtf(tf_scores)

        m15_df = indicators_by_tf.get("15m", pd.DataFrame())
        m15_latest = m15_df.iloc[-1] if not m15_df.empty else pd.Series(dtype=float)
        atr_avg_20 = float(m15_df["atr_14"].tail(20).mean()) if "atr_14" in m15_df else 0.0
        regime = detect_market_regime(m15_latest, atr_avg_20) if not m15_df.empty else "NEUTRAL"

        daily_df = indicators_by_tf.get("1d", pd.DataFrame())
        pivots = {}
        if len(daily_df) >= 2:
            pivots = compute_pivots(daily_df.iloc[-1], daily_df.iloc[-2])
        swings = swing_levels(m15_df if not m15_df.empty else daily_df, lookback=20)
        round_lvls = round_number_levels(float(data_context.current_price))
        h1_df = indicators_by_tf.get("1h", pd.DataFrame())
        patterns = _detect_patterns(m15_df, h1_df)

        adx = _safe_metric(snapshot_by_tf.get("15m", {}).get("adx_14"), default=20.0)
        rsi = _safe_metric(snapshot_by_tf.get("15m", {}).get("rsi_14"), default=50.0)
        vol_ratio = _safe_metric(snapshot_by_tf.get("15m", {}).get("vol_ratio"), default=1.0)
        quant_score = _quant_score(mtf_confluence, adx, rsi, vol_ratio)

        return {
            "ok": True,
            "quant_score": round(quant_score, 2),
            "regime": regime,
            "mtf_score": mtf_score,
            "mtf_confluence": round(mtf_confluence, 2),
            "direction": direction,
            "timeframe_scores": tf_scores,
            "confluence_checks": confluence_checks,
            "indicators": _serialize_snapshot(snapshot_by_tf),
            "patterns": patterns,
            "key_levels": {
                "pivots": pivots,
                "swings": swings,
                "round_levels": round_lvls,
            },
            "atr_current": float(snapshot_by_tf.get("15m", {}).get("atr_14", 0.0)),
            "adx_current": adx,
            "error": "",
        }
    except Exception as exc:
        return _degraded_output(str(exc))


def _prepare_for_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Clean OHLCV input before indicator engine to avoid silent failures."""

    if df is None or df.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    out = df.copy().rename(columns=str.lower)
    for col in ("open", "high", "low", "close", "volume"):
        if col not in out.columns:
            out[col] = pd.NA
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["open", "high", "low", "close"])
    out = out[out["volume"].fillna(0.0) > 0.0].copy()
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out


def _extract_snapshot(enriched: pd.DataFrame) -> Dict[str, float]:
    """Extract last valid indicator snapshot for transparent output."""

    keys = [
        "close",
        "ema_9",
        "ema_21",
        "ema_50",
        "ema_200",
        "macd",
        "macd_signal",
        "macd_hist",
        "adx_14",
        "rsi_14",
        "stochrsi_k",
        "stochrsi_d",
        "roc_10",
        "atr_14",
        "bb_upper",
        "bb_mid",
        "bb_lower",
        "bb_width",
        "kc_upper",
        "kc_mid",
        "kc_lower",
        "vwap",
        "obv",
        "vol_sma_20",
        "vol_ratio",
    ]
    out: Dict[str, float] = {}
    for key in keys:
        try:
            if key not in enriched.columns:
                out[key] = float("nan")
                continue
            val = _last_valid(enriched[key])
            out[key] = round(float(val), 6) if pd.notna(val) else float("nan")
        except Exception:
            out[key] = float("nan")
    return out


def _quant_score(mtf_confluence: float, adx: float, rsi: float, vol_ratio: float) -> float:
    """Derive normalized quant score (0-100) from MTF and core factors."""

    trend_component = min(max((adx / 40.0) * 100.0, 0.0), 100.0)
    rsi_component = 100.0 - min(abs(rsi - 50.0) * 2.0, 100.0)
    volume_component = min(max(vol_ratio * 50.0, 0.0), 100.0)
    return (mtf_confluence * 0.55) + (trend_component * 0.2) + (rsi_component * 0.15) + (volume_component * 0.1)


def _last_valid(series: pd.Series) -> float:
    """Return last non-NaN value from a series."""

    valid = series.dropna()
    if valid.empty:
        return float("nan")
    return float(valid.iloc[-1])


def _safe_metric(value: Any, default: float) -> float:
    """Parse finite float metric with neutral fallback."""

    try:
        parsed = float(value)
        if np.isfinite(parsed):
            return parsed
    except Exception:
        pass
    return float(default)


def _serialize_snapshot(snapshot: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, Any]]:
    """Convert non-finite float values to None for JSON-safe API responses."""

    out: Dict[str, Dict[str, Any]] = {}
    for tf, values in snapshot.items():
        out[tf] = {}
        for key, value in values.items():
            try:
                val = float(value)
                out[tf][key] = round(val, 6) if np.isfinite(val) else None
            except Exception:
                out[tf][key] = None
    return out


def _direction_from_mtf(tf_scores: Dict[str, int]) -> str:
    """Infer directional bias from timeframe confluence map."""

    bull_votes = sum(1 for score in tf_scores.values() if score >= 6)
    bear_votes = sum(1 for score in tf_scores.values() if score <= 4)
    if bull_votes >= 3:
        return "BULLISH"
    if bear_votes >= 3:
        return "BEARISH"
    return "NEUTRAL"


def _degraded_output(reason: str) -> Dict[str, Any]:
    """Fallback output if quant engine fails."""

    return {
        "ok": False,
        "quant_score": 50.0,
        "regime": "NEUTRAL",
        "mtf_score": 20,
        "mtf_confluence": 50.0,
        "direction": "NEUTRAL",
        "timeframe_scores": {"5m": 5, "15m": 5, "1h": 5, "1d": 5},
        "confluence_checks": {},
        "indicators": {},
        "patterns": [{"name": "Quant Degraded", "reason": "Indicator computation failed."}],
        "key_levels": {"pivots": {}, "swings": {}, "round_levels": {}},
        "atr_current": 0.0,
        "adx_current": 0.0,
        "error": f"agent2_failure: {reason}",
    }


def _detect_patterns(m15_df: pd.DataFrame, h1_df: pd.DataFrame) -> List[Dict[str, str]]:
    """Detect candlestick and chart patterns on 15m candles with 1h confirmation."""

    if m15_df is None or m15_df.empty or len(m15_df) < 20:
        return [{"name": "No Strong Pattern", "direction": "NEUTRAL", "confidence": "LOW", "reason": "Insufficient 15m candles for pattern scan."}]

    patterns: List[Dict[str, str]] = []
    c3 = m15_df.tail(3).copy()
    c20 = m15_df.tail(20).copy()
    trend_hint = _short_trend_bias(m15_df.tail(10))
    h1_bias = _timeframe_bias(h1_df)

    # Candlestick patterns over last 3 candles.
    if len(c3) >= 2:
        prev = c3.iloc[-2]
        curr = c3.iloc[-1]
        if _is_bullish_engulfing(prev, curr):
            patterns.append(_build_pattern("Bullish Engulfing", "BULLISH", 2, "Bearish candle followed by full bullish body engulfing previous body.", h1_bias))
        if _is_bearish_engulfing(prev, curr):
            patterns.append(_build_pattern("Bearish Engulfing", "BEARISH", 2, "Bullish candle followed by full bearish body engulfing previous body.", h1_bias))

    for _, candle in c3.iterrows():
        if _is_doji(candle):
            doji_direction = _doji_direction(trend_hint, h1_bias)
            patterns.append(_build_pattern("Doji", doji_direction, 1, "Open and close are nearly equal (<=0.1%), signaling indecision and reversal risk.", h1_bias))
        if _is_hammer(candle):
            patterns.append(_build_pattern("Hammer", "BULLISH", 2, "Lower wick is at least 2x body with small upper wick after local weakness.", h1_bias))
        if _is_shooting_star(candle):
            patterns.append(_build_pattern("Shooting Star", "BEARISH", 2, "Upper wick is at least 2x body with small lower wick after local strength.", h1_bias))

    if len(c3) == 3:
        if _is_morning_star(c3):
            patterns.append(_build_pattern("Morning Star", "BULLISH", 3, "Three-candle bullish reversal: strong down candle, indecision, then bullish recovery.", h1_bias))
        if _is_evening_star(c3):
            patterns.append(_build_pattern("Evening Star", "BEARISH", 3, "Three-candle bearish reversal: strong up candle, indecision, then bearish rejection.", h1_bias))

    # Chart patterns over last 20 candles.
    if _is_double_bottom(c20):
        patterns.append(_build_pattern("Double Bottom", "BULLISH", 3, "Two similar lows formed with neckline breakout on 15m structure.", h1_bias))
    if _is_double_top(c20):
        patterns.append(_build_pattern("Double Top", "BEARISH", 3, "Two similar highs formed with neckline breakdown on 15m structure.", h1_bias))
    if _is_bull_flag(c20):
        patterns.append(_build_pattern("Bull Flag", "BULLISH", 2, "Strong impulse up followed by tight/slight pullback consolidation.", h1_bias))
    if _is_bear_flag(c20):
        patterns.append(_build_pattern("Bear Flag", "BEARISH", 2, "Strong impulse down followed by tight/slight upward consolidation.", h1_bias))
    if _is_hh_hl(c20):
        patterns.append(_build_pattern("Higher Highs Higher Lows", "BULLISH", 2, "Recent swings show sequential higher highs and higher lows.", h1_bias))
    if _is_lh_ll(c20):
        patterns.append(_build_pattern("Lower Highs Lower Lows", "BEARISH", 2, "Recent swings show sequential lower highs and lower lows.", h1_bias))

    deduped = _dedupe_patterns(patterns)
    if deduped:
        return deduped
    return [{"name": "No Strong Pattern", "direction": "NEUTRAL", "confidence": "LOW", "reason": "No high-quality reversal or continuation structure detected on 15m."}]


def _build_pattern(name: str, direction: str, base_strength: int, reason: str, h1_bias: str) -> Dict[str, str]:
    """Build pattern response with 1h confirmation-aware confidence."""

    confirmed = direction == h1_bias and direction in {"BULLISH", "BEARISH"}
    adjusted_strength = base_strength + (1 if confirmed else -1)
    confidence = "HIGH" if adjusted_strength >= 3 else "MEDIUM" if adjusted_strength == 2 else "LOW"
    confirm_note = " 1h confirms direction." if confirmed else " 1h confirmation is weak/mixed."
    return {
        "name": name,
        "direction": direction,
        "confidence": confidence,
        "reason": f"{reason}{confirm_note}",
    }


def _dedupe_patterns(patterns: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Deduplicate patterns and keep the highest confidence instance."""

    rank = {"LOW": 1, "MEDIUM": 2, "HIGH": 3}
    best: Dict[Tuple[str, str], Dict[str, str]] = {}
    order: List[Tuple[str, str]] = []
    for item in patterns:
        key = (item.get("name", ""), item.get("direction", ""))
        if key not in best:
            best[key] = item
            order.append(key)
            continue
        cur = rank.get(item.get("confidence", "LOW"), 1)
        prev = rank.get(best[key].get("confidence", "LOW"), 1)
        if cur > prev:
            best[key] = item
    return [best[k] for k in order]


def _safe_float(value: Any) -> float:
    """Safe float casting."""

    try:
        return float(value)
    except Exception:
        return 0.0


def _candle_parts(candle: pd.Series) -> Tuple[float, float, float, float, float]:
    """Return body, upper wick, lower wick, range and open-close delta."""

    o = _safe_float(candle.get("open"))
    h = _safe_float(candle.get("high"))
    l = _safe_float(candle.get("low"))
    c = _safe_float(candle.get("close"))
    body = abs(c - o)
    upper = max(0.0, h - max(o, c))
    lower = max(0.0, min(o, c) - l)
    rng = max(1e-9, h - l)
    delta = c - o
    return body, upper, lower, rng, delta


def _is_bullish_engulfing(prev: pd.Series, curr: pd.Series) -> bool:
    """Check bullish engulfing on two candles."""

    po, pc = _safe_float(prev.get("open")), _safe_float(prev.get("close"))
    co, cc = _safe_float(curr.get("open")), _safe_float(curr.get("close"))
    prev_bear = pc < po
    curr_bull = cc > co
    prev_body = abs(pc - po)
    curr_body = abs(cc - co)
    engulf = co <= pc and cc >= po
    return prev_bear and curr_bull and curr_body > prev_body and engulf


def _is_bearish_engulfing(prev: pd.Series, curr: pd.Series) -> bool:
    """Check bearish engulfing on two candles."""

    po, pc = _safe_float(prev.get("open")), _safe_float(prev.get("close"))
    co, cc = _safe_float(curr.get("open")), _safe_float(curr.get("close"))
    prev_bull = pc > po
    curr_bear = cc < co
    prev_body = abs(pc - po)
    curr_body = abs(cc - co)
    engulf = co >= pc and cc <= po
    return prev_bull and curr_bear and curr_body > prev_body and engulf


def _is_doji(candle: pd.Series) -> bool:
    """Doji: open-close difference within 0.1% of close."""

    o = _safe_float(candle.get("open"))
    c = _safe_float(candle.get("close"))
    ref = max(abs(c), 1e-9)
    return abs(o - c) / ref <= 0.001


def _is_hammer(candle: pd.Series) -> bool:
    """Hammer: long lower wick, small upper wick, compact body."""

    body, upper, lower, rng, _ = _candle_parts(candle)
    return lower >= (2.0 * max(body, 1e-9)) and upper <= (0.35 * max(body, 1e-9)) and body / rng <= 0.5


def _is_shooting_star(candle: pd.Series) -> bool:
    """Shooting star: long upper wick, small lower wick, compact body."""

    body, upper, lower, rng, _ = _candle_parts(candle)
    return upper >= (2.0 * max(body, 1e-9)) and lower <= (0.35 * max(body, 1e-9)) and body / rng <= 0.5


def _is_morning_star(c3: pd.DataFrame) -> bool:
    """Morning star three-candle bullish reversal."""

    c1, c2, c3c = c3.iloc[0], c3.iloc[1], c3.iloc[2]
    o1, c1c = _safe_float(c1.get("open")), _safe_float(c1.get("close"))
    o2, c2c = _safe_float(c2.get("open")), _safe_float(c2.get("close"))
    o3, c3v = _safe_float(c3c.get("open")), _safe_float(c3c.get("close"))
    body1 = abs(c1c - o1)
    body2 = abs(c2c - o2)
    midpoint1 = (o1 + c1c) / 2.0
    return (c1c < o1) and (body2 <= body1 * 0.6) and (c3v > o3) and (c3v > midpoint1)


def _is_evening_star(c3: pd.DataFrame) -> bool:
    """Evening star three-candle bearish reversal."""

    c1, c2, c3c = c3.iloc[0], c3.iloc[1], c3.iloc[2]
    o1, c1c = _safe_float(c1.get("open")), _safe_float(c1.get("close"))
    o2, c2c = _safe_float(c2.get("open")), _safe_float(c2.get("close"))
    o3, c3v = _safe_float(c3c.get("open")), _safe_float(c3c.get("close"))
    body1 = abs(c1c - o1)
    body2 = abs(c2c - o2)
    midpoint1 = (o1 + c1c) / 2.0
    return (c1c > o1) and (body2 <= body1 * 0.6) and (c3v < o3) and (c3v < midpoint1)


def _is_double_bottom(c20: pd.DataFrame) -> bool:
    """Double bottom with neckline breakout."""

    lows = c20["low"].reset_index(drop=True)
    highs = c20["high"].reset_index(drop=True)
    closes = c20["close"].reset_index(drop=True)
    idx = [i for i in range(1, len(lows) - 1) if lows.iloc[i] <= lows.iloc[i - 1] and lows.iloc[i] <= lows.iloc[i + 1]]
    if len(idx) < 2:
        return False
    i1, i2 = idx[-2], idx[-1]
    if (i2 - i1) < 3:
        return False
    l1, l2 = lows.iloc[i1], lows.iloc[i2]
    if abs(l1 - l2) / max(l1, l2, 1e-9) > 0.008:
        return False
    neckline = float(highs.iloc[i1 : i2 + 1].max())
    return float(closes.iloc[-1]) > neckline * 1.001


def _is_double_top(c20: pd.DataFrame) -> bool:
    """Double top with neckline breakdown."""

    lows = c20["low"].reset_index(drop=True)
    highs = c20["high"].reset_index(drop=True)
    closes = c20["close"].reset_index(drop=True)
    idx = [i for i in range(1, len(highs) - 1) if highs.iloc[i] >= highs.iloc[i - 1] and highs.iloc[i] >= highs.iloc[i + 1]]
    if len(idx) < 2:
        return False
    i1, i2 = idx[-2], idx[-1]
    if (i2 - i1) < 3:
        return False
    h1, h2 = highs.iloc[i1], highs.iloc[i2]
    if abs(h1 - h2) / max(h1, h2, 1e-9) > 0.008:
        return False
    neckline = float(lows.iloc[i1 : i2 + 1].min())
    return float(closes.iloc[-1]) < neckline * 0.999


def _is_bull_flag(c20: pd.DataFrame) -> bool:
    """Bull flag: strong up move then tight consolidation."""

    if len(c20) < 20:
        return False
    part1 = c20.iloc[:8]
    part2 = c20.iloc[8:]
    pole_move = (_safe_float(part1["close"].iloc[-1]) - _safe_float(part1["open"].iloc[0])) / max(_safe_float(part1["open"].iloc[0]), 1e-9)
    flag_range = (_safe_float(part2["high"].max()) - _safe_float(part2["low"].min())) / max(_safe_float(c20["close"].iloc[-1]), 1e-9)
    flag_slope = (_safe_float(part2["close"].iloc[-1]) - _safe_float(part2["close"].iloc[0])) / max(_safe_float(part2["close"].iloc[0]), 1e-9)
    return pole_move >= 0.02 and flag_range <= 0.03 and -0.03 <= flag_slope <= 0.01


def _is_bear_flag(c20: pd.DataFrame) -> bool:
    """Bear flag: strong down move then tight consolidation."""

    if len(c20) < 20:
        return False
    part1 = c20.iloc[:8]
    part2 = c20.iloc[8:]
    pole_move = (_safe_float(part1["close"].iloc[-1]) - _safe_float(part1["open"].iloc[0])) / max(_safe_float(part1["open"].iloc[0]), 1e-9)
    flag_range = (_safe_float(part2["high"].max()) - _safe_float(part2["low"].min())) / max(_safe_float(c20["close"].iloc[-1]), 1e-9)
    flag_slope = (_safe_float(part2["close"].iloc[-1]) - _safe_float(part2["close"].iloc[0])) / max(_safe_float(part2["close"].iloc[0]), 1e-9)
    return pole_move <= -0.02 and flag_range <= 0.03 and -0.01 <= flag_slope <= 0.03


def _is_hh_hl(c20: pd.DataFrame) -> bool:
    """Higher highs and higher lows confirmation."""

    highs = c20["high"].tail(6).to_numpy(dtype=float)
    lows = c20["low"].tail(6).to_numpy(dtype=float)
    if len(highs) < 6 or len(lows) < 6:
        return False
    hh_hits = int((np.diff(highs) > 0).sum())
    hl_hits = int((np.diff(lows) > 0).sum())
    return hh_hits >= 4 and hl_hits >= 4


def _is_lh_ll(c20: pd.DataFrame) -> bool:
    """Lower highs and lower lows confirmation."""

    highs = c20["high"].tail(6).to_numpy(dtype=float)
    lows = c20["low"].tail(6).to_numpy(dtype=float)
    if len(highs) < 6 or len(lows) < 6:
        return False
    lh_hits = int((np.diff(highs) < 0).sum())
    ll_hits = int((np.diff(lows) < 0).sum())
    return lh_hits >= 4 and ll_hits >= 4


def _short_trend_bias(window: pd.DataFrame) -> str:
    """Short trend bias from close slope in 15m candles."""

    if window is None or window.empty or len(window) < 5:
        return "NEUTRAL"
    first = _safe_float(window["close"].iloc[0])
    last = _safe_float(window["close"].iloc[-1])
    move = (last - first) / max(first, 1e-9)
    if move > 0.008:
        return "BULLISH"
    if move < -0.008:
        return "BEARISH"
    return "NEUTRAL"


def _timeframe_bias(h1_df: pd.DataFrame) -> str:
    """1h timeframe bias used as confirmation layer."""

    if h1_df is None or h1_df.empty or len(h1_df) < 10:
        return "NEUTRAL"
    latest = h1_df.iloc[-1]
    close = _safe_float(latest.get("close"))
    ema21 = _safe_float(latest.get("ema_21"))
    ema9 = _safe_float(latest.get("ema_9"))
    macd_hist = _safe_float(latest.get("macd_hist"))
    if close > ema21 and ema9 >= ema21 and macd_hist >= 0 and _is_hh_hl(h1_df.tail(20)):
        return "BULLISH"
    if close < ema21 and ema9 <= ema21 and macd_hist <= 0 and _is_lh_ll(h1_df.tail(20)):
        return "BEARISH"
    return "NEUTRAL"


def _doji_direction(trend_hint: str, h1_bias: str) -> str:
    """Map neutral doji into directional context."""

    if trend_hint == "BEARISH":
        return "BULLISH"
    if trend_hint == "BULLISH":
        return "BEARISH"
    if h1_bias in {"BULLISH", "BEARISH"}:
        return h1_bias
    return "BULLISH"
