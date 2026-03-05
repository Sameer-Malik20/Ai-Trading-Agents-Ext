"""Agent 2: quantitative indicator engine, MTF confluence, and professional pattern library."""

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


CONFIDENCE_WEIGHT = {"HIGH": 10, "MEDIUM": 5, "LOW": 2, "NEUTRAL": 2}


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
        pivots: Dict[str, float] = {}
        if len(daily_df) >= 2:
            pivots = compute_pivots(daily_df.iloc[-1], daily_df.iloc[-2])
        swings = swing_levels(m15_df if not m15_df.empty else daily_df, lookback=20)
        round_lvls = round_number_levels(float(data_context.current_price))

        h1_df = indicators_by_tf.get("1h", pd.DataFrame())
        patterns, pattern_summary = _detect_patterns(m15_df, h1_df)

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
            "pattern_summary": pattern_summary,
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
        "patterns": [
            {
                "name": "No Strong Pattern",
                "direction": "NEUTRAL",
                "confidence": "LOW",
                "timeframe": "15m",
                "confirmed_on_1h": False,
                "reason": "Indicator computation failed.",
                "trade_implication": "Avoid fresh positions until stable signal appears.",
            }
        ],
        "pattern_summary": {
            "bullish_patterns": 0,
            "bearish_patterns": 0,
            "neutral_patterns": 1,
            "strong_bullish_confluence": False,
            "strong_bearish_confluence": False,
            "pattern_conflict": False,
            "conviction_adjustment": 0,
        },
        "key_levels": {"pivots": {}, "swings": {}, "round_levels": {}},
        "atr_current": 0.0,
        "adx_current": 0.0,
        "error": f"agent2_failure: {reason}",
    }


def _detect_patterns(m15_df: pd.DataFrame, h1_df: pd.DataFrame) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Detect complete candlestick, chart, and volume pattern library."""

    if m15_df is None or m15_df.empty or len(m15_df) < 30:
        fallback = [
            _pattern(
                name="No Strong Pattern",
                direction="NEUTRAL",
                confidence="LOW",
                confirmed_on_1h=False,
                reason="Insufficient 15m candles for full pattern scan.",
                trade_implication="Wait for more structure before entering a trade.",
            )
        ]
        return fallback, _pattern_summary(fallback)

    m15 = m15_df.tail(60).copy()
    h1 = h1_df.tail(60).copy() if h1_df is not None else pd.DataFrame()
    h1_bias = _timeframe_bias(h1)

    patterns: List[Dict[str, Any]] = []
    c1 = m15.iloc[-1]
    c2 = m15.iloc[-2]
    trend20 = _trend_context(m15.tail(20))

    # 1) Single-candle patterns.
    if _is_doji(c1):
        patterns.append(
            _pattern(
                "Doji",
                "NEUTRAL",
                "LOW",
                _confirm_direction("NEUTRAL", h1_bias),
                "Open and close are nearly equal, reflecting indecision.",
                "Wait for breakout confirmation before committing.",
            )
        )
    if _is_hammer(c1) and trend20 == "DOWN":
        patterns.append(
            _pattern(
                "Hammer",
                "BULLISH",
                "MEDIUM",
                _confirm_direction("BULLISH", h1_bias),
                "Long lower wick after downtrend shows buyer absorption.",
                "Bias long only above hammer high.",
            )
        )
    if _is_shooting_star(c1) and trend20 == "UP":
        patterns.append(
            _pattern(
                "Shooting Star",
                "BEARISH",
                "MEDIUM",
                _confirm_direction("BEARISH", h1_bias),
                "Long upper wick after uptrend shows rejection at highs.",
                "Bias short only below pattern low.",
            )
        )
    if _is_hammer(c1) and trend20 == "UP":
        patterns.append(
            _pattern(
                "Hanging Man",
                "BEARISH",
                "MEDIUM",
                _confirm_direction("BEARISH", h1_bias),
                "Hammer-like candle after uptrend warns of distribution.",
                "Avoid fresh longs until confirmation.",
            )
        )
    if _is_shooting_star(c1) and trend20 == "DOWN":
        patterns.append(
            _pattern(
                "Inverted Hammer",
                "BULLISH",
                "MEDIUM",
                _confirm_direction("BULLISH", h1_bias),
                "Shooting-star shape after downtrend suggests possible reversal.",
                "Consider long on follow-through candle.",
            )
        )
    if _is_bullish_marubozu(c1):
        patterns.append(
            _pattern(
                "Bullish Marubozu",
                "BULLISH",
                "HIGH",
                _confirm_direction("BULLISH", h1_bias),
                "Large bullish body with tiny wicks indicates strong control.",
                "Momentum continuation favored.",
            )
        )
    if _is_bearish_marubozu(c1):
        patterns.append(
            _pattern(
                "Bearish Marubozu",
                "BEARISH",
                "HIGH",
                _confirm_direction("BEARISH", h1_bias),
                "Large bearish body with tiny wicks indicates aggressive selling.",
                "Momentum downside continuation favored.",
            )
        )
    if _is_spinning_top(c1):
        patterns.append(
            _pattern(
                "Spinning Top",
                "NEUTRAL",
                "LOW",
                _confirm_direction("NEUTRAL", h1_bias),
                "Small body with balanced wicks shows indecision.",
                "Reduce size until direction resolves.",
            )
        )

    # 2) Two-candle patterns.
    if _is_bullish_engulfing(c2, c1):
        patterns.append(_pattern("Bullish Engulfing", "BULLISH", "HIGH", _confirm_direction("BULLISH", h1_bias), "Bearish body is fully engulfed by strong bullish candle.", "Long bias with tight stop below pattern low."))
    if _is_bearish_engulfing(c2, c1):
        patterns.append(_pattern("Bearish Engulfing", "BEARISH", "HIGH", _confirm_direction("BEARISH", h1_bias), "Bullish body is fully engulfed by strong bearish candle.", "Short bias with stop above pattern high."))
    if _is_tweezer_bottom(c2, c1):
        patterns.append(_pattern("Tweezer Bottom", "BULLISH", "MEDIUM", _confirm_direction("BULLISH", h1_bias), "Two candles reject the same low zone.", "Potential reversal if neckline breaks."))
    if _is_tweezer_top(c2, c1):
        patterns.append(_pattern("Tweezer Top", "BEARISH", "MEDIUM", _confirm_direction("BEARISH", h1_bias), "Two candles reject the same high zone.", "Potential reversal if support breaks."))
    if _is_piercing_line(c2, c1):
        patterns.append(_pattern("Piercing Line", "BULLISH", "MEDIUM", _confirm_direction("BULLISH", h1_bias), "Bull candle closes above midpoint of prior bear body.", "Early bullish reversal signal."))
    if _is_dark_cloud_cover(c2, c1):
        patterns.append(_pattern("Dark Cloud Cover", "BEARISH", "MEDIUM", _confirm_direction("BEARISH", h1_bias), "Bear candle closes below midpoint of prior bull body.", "Early bearish reversal signal."))

    # 3) Three-candle patterns.
    last3 = m15.tail(3)
    if _is_morning_star(last3):
        patterns.append(_pattern("Morning Star", "BULLISH", "HIGH", _confirm_direction("BULLISH", h1_bias), "Three-candle bullish reversal with strong recovery candle.", "Long setup on confirmation break."))
    if _is_evening_star(last3):
        patterns.append(_pattern("Evening Star", "BEARISH", "HIGH", _confirm_direction("BEARISH", h1_bias), "Three-candle bearish reversal with strong rejection candle.", "Short setup on confirmation break."))
    if _is_three_white_soldiers(last3):
        patterns.append(_pattern("Three White Soldiers", "BULLISH", "HIGH", _confirm_direction("BULLISH", h1_bias), "Three strong consecutive bullish candles indicate sustained demand.", "Momentum long continuation setup."))
    if _is_three_black_crows(last3):
        patterns.append(_pattern("Three Black Crows", "BEARISH", "HIGH", _confirm_direction("BEARISH", h1_bias), "Three strong consecutive bearish candles indicate sustained supply.", "Momentum short continuation setup."))
    if _is_three_inside_up(last3):
        patterns.append(_pattern("Three Inside Up", "BULLISH", "MEDIUM", _confirm_direction("BULLISH", h1_bias), "Inside candle followed by bullish confirmation breakout.", "Bullish reversal with moderate confidence."))
    if _is_three_inside_down(last3):
        patterns.append(_pattern("Three Inside Down", "BEARISH", "MEDIUM", _confirm_direction("BEARISH", h1_bias), "Inside candle followed by bearish confirmation breakdown.", "Bearish reversal with moderate confidence."))

    # 4) Chart patterns (20-50 candles).
    c20 = m15.tail(20)
    c50 = m15.tail(50)
    if _is_double_bottom(c20):
        patterns.append(_pattern("Double Bottom", "BULLISH", "HIGH", _confirm_direction("BULLISH", h1_bias), "Two comparable lows with neckline reclaim.", "Bullish reversal breakout candidate."))
    if _is_double_top(c20):
        patterns.append(_pattern("Double Top", "BEARISH", "HIGH", _confirm_direction("BEARISH", h1_bias), "Two comparable highs with neckline breakdown.", "Bearish reversal breakdown candidate."))
    if _is_head_shoulders(c50):
        patterns.append(_pattern("Head and Shoulders", "BEARISH", "HIGH", _confirm_direction("BEARISH", h1_bias), "Higher middle peak with weaker right shoulder.", "Bearish reversal bias; watch neckline."))
    if _is_inverse_head_shoulders(c50):
        patterns.append(_pattern("Inverse Head and Shoulders", "BULLISH", "HIGH", _confirm_direction("BULLISH", h1_bias), "Lower middle trough with stronger right shoulder.", "Bullish reversal bias; watch neckline."))
    if _is_bull_flag(c20):
        patterns.append(_pattern("Bull Flag", "BULLISH", "HIGH", _confirm_direction("BULLISH", h1_bias), "Strong impulse up followed by shallow consolidation.", "Trend continuation long setup."))
    if _is_bear_flag(c20):
        patterns.append(_pattern("Bear Flag", "BEARISH", "HIGH", _confirm_direction("BEARISH", h1_bias), "Strong impulse down followed by weak bounce.", "Trend continuation short setup."))
    if _is_ascending_triangle(c50):
        patterns.append(_pattern("Ascending Triangle", "BULLISH", "MEDIUM", _confirm_direction("BULLISH", h1_bias), "Flat resistance with rising lows compressing price.", "Bullish breakout probability elevated."))
    if _is_descending_triangle(c50):
        patterns.append(_pattern("Descending Triangle", "BEARISH", "MEDIUM", _confirm_direction("BEARISH", h1_bias), "Flat support with falling highs compressing price.", "Bearish breakdown probability elevated."))
    sym_direction = _is_symmetrical_triangle(c50)
    if sym_direction:
        patterns.append(_pattern("Symmetrical Triangle", sym_direction, "MEDIUM", _confirm_direction(sym_direction, h1_bias), "Converging highs/lows with breakout-led directional edge.", "Trade only in breakout direction."))
    if _is_rising_wedge(c50):
        patterns.append(_pattern("Rising Wedge", "BEARISH", "MEDIUM", _confirm_direction("BEARISH", h1_bias), "Upward sloping but narrowing structure weakens trend quality.", "Bearish breakdown risk rising."))
    if _is_falling_wedge(c50):
        patterns.append(_pattern("Falling Wedge", "BULLISH", "MEDIUM", _confirm_direction("BULLISH", h1_bias), "Downward sloping but narrowing structure indicates selling exhaustion.", "Bullish breakout risk rising."))
    if _is_cup_handle(c50):
        patterns.append(_pattern("Cup and Handle", "BULLISH", "HIGH", _confirm_direction("BULLISH", h1_bias), "Rounded base followed by controlled pullback handle.", "Bullish continuation breakout setup."))
    if _is_hh_hl(c20):
        patterns.append(_pattern("Higher Highs Higher Lows", "BULLISH", "HIGH", _confirm_direction("BULLISH", h1_bias), "Swing structure confirms persistent uptrend.", "Prefer long pullback entries."))
    if _is_lh_ll(c20):
        patterns.append(_pattern("Lower Highs Lower Lows", "BEARISH", "HIGH", _confirm_direction("BEARISH", h1_bias), "Swing structure confirms persistent downtrend.", "Prefer short bounce entries."))

    # 5) Volume patterns.
    if _is_volume_climax(m15):
        patterns.append(_pattern("Volume Climax", "NEUTRAL", "LOW", _confirm_direction("NEUTRAL", h1_bias), "Extreme volume spike suggests possible blow-off/exhaustion move.", "Protect profit and wait for confirmation."))
    if _is_volume_divergence(m15):
        patterns.append(_pattern("Volume Divergence", "NEUTRAL", "LOW", _confirm_direction("NEUTRAL", h1_bias), "Price trend and volume trend are diverging.", "Caution: trend may weaken soon."))
    breakout_dir = _breakout_with_volume(m15)
    if breakout_dir:
        patterns.append(_pattern("Breakout with Volume", breakout_dir, "HIGH", _confirm_direction(breakout_dir, h1_bias), "Range breakout occurred with volume expansion.", "Follow breakout direction with risk control."))

    deduped = _dedupe_patterns(patterns)
    if not deduped:
        deduped = [
            _pattern(
                "No Strong Pattern",
                "NEUTRAL",
                "LOW",
                False,
                "No meaningful pattern cluster detected on 15m.",
                "Stand aside until pattern clarity improves.",
            )
        ]
    return deduped, _pattern_summary(deduped)


def _pattern(name: str, direction: str, confidence: str, confirmed_on_1h: bool, reason: str, trade_implication: str) -> Dict[str, Any]:
    """Create one normalized pattern object."""

    return {
        "name": name,
        "direction": direction,
        "confidence": confidence,
        "timeframe": "15m",
        "confirmed_on_1h": bool(confirmed_on_1h),
        "reason": reason,
        "trade_implication": trade_implication,
    }


def _pattern_summary(patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build confluence/conflict summary and conviction adjustment."""

    bullish = [p for p in patterns if p.get("direction") == "BULLISH"]
    bearish = [p for p in patterns if p.get("direction") == "BEARISH"]
    neutral = [p for p in patterns if p.get("direction") == "NEUTRAL"]

    adjustment = 0
    for p in bullish:
        adjustment += CONFIDENCE_WEIGHT.get(str(p.get("confidence", "LOW")).upper(), 2)
    for p in bearish:
        adjustment -= CONFIDENCE_WEIGHT.get(str(p.get("confidence", "LOW")).upper(), 2)

    pattern_conflict = len(bullish) > 0 and len(bearish) > 0
    if pattern_conflict:
        adjustment -= 10

    return {
        "bullish_patterns": len(bullish),
        "bearish_patterns": len(bearish),
        "neutral_patterns": len(neutral),
        "strong_bullish_confluence": len(bullish) >= 3,
        "strong_bearish_confluence": len(bearish) >= 3,
        "pattern_conflict": pattern_conflict,
        "conviction_adjustment": int(max(-40, min(40, adjustment))),
    }


def _dedupe_patterns(patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deduplicate patterns and keep strongest confidence per pattern name."""

    rank = {"LOW": 1, "NEUTRAL": 1, "MEDIUM": 2, "HIGH": 3}
    best: Dict[str, Dict[str, Any]] = {}
    for item in patterns:
        key = str(item.get("name", "")).strip()
        if not key:
            continue
        if key not in best:
            best[key] = item
            continue
        cur = rank.get(str(item.get("confidence", "LOW")).upper(), 1)
        prev = rank.get(str(best[key].get("confidence", "LOW")).upper(), 1)
        if cur > prev:
            best[key] = item
    ordered = list(best.values())
    ordered.sort(key=lambda p: (str(p.get("confidence", "LOW")) != "HIGH", p.get("name", "")))
    return ordered[:18]


def _safe_float(value: Any) -> float:
    """Safe float casting."""

    try:
        return float(value)
    except Exception:
        return 0.0


def _candle_parts(candle: pd.Series) -> Tuple[float, float, float, float, float]:
    """Return body, upper wick, lower wick, range and delta."""

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


def _trend_context(window: pd.DataFrame) -> str:
    """Simple trend context for pattern direction gating."""

    if window is None or window.empty or len(window) < 8:
        return "SIDE"
    first = _safe_float(window["close"].iloc[0])
    last = _safe_float(window["close"].iloc[-1])
    change = (last - first) / max(first, 1e-9)
    if change > 0.01:
        return "UP"
    if change < -0.01:
        return "DOWN"
    return "SIDE"


def _confirm_direction(direction: str, h1_bias: str) -> bool:
    """Confirm pattern direction against 1h bias."""

    if direction == "NEUTRAL":
        return h1_bias == "NEUTRAL"
    return direction == h1_bias


def _is_doji(candle: pd.Series) -> bool:
    """Doji threshold at 0.1%."""

    o = _safe_float(candle.get("open"))
    c = _safe_float(candle.get("close"))
    return abs(o - c) / max(abs(c), 1e-9) <= 0.001


def _is_hammer(candle: pd.Series) -> bool:
    body, upper, lower, rng, _ = _candle_parts(candle)
    return lower >= (2.0 * max(body, 1e-9)) and upper <= (0.35 * max(body, 1e-9)) and body / rng <= 0.45


def _is_shooting_star(candle: pd.Series) -> bool:
    body, upper, lower, rng, _ = _candle_parts(candle)
    return upper >= (2.0 * max(body, 1e-9)) and lower <= (0.35 * max(body, 1e-9)) and body / rng <= 0.45


def _is_bullish_marubozu(candle: pd.Series) -> bool:
    body, upper, lower, rng, delta = _candle_parts(candle)
    return delta > 0 and body / rng >= 0.85 and upper / rng <= 0.08 and lower / rng <= 0.08


def _is_bearish_marubozu(candle: pd.Series) -> bool:
    body, upper, lower, rng, delta = _candle_parts(candle)
    return delta < 0 and body / rng >= 0.85 and upper / rng <= 0.08 and lower / rng <= 0.08


def _is_spinning_top(candle: pd.Series) -> bool:
    body, upper, lower, rng, _ = _candle_parts(candle)
    return body / rng <= 0.3 and upper >= body and lower >= body


def _is_bullish_engulfing(prev: pd.Series, curr: pd.Series) -> bool:
    po, pc = _safe_float(prev.get("open")), _safe_float(prev.get("close"))
    co, cc = _safe_float(curr.get("open")), _safe_float(curr.get("close"))
    prev_bear = pc < po
    curr_bull = cc > co
    engulf = co <= pc and cc >= po
    return prev_bear and curr_bull and engulf and abs(cc - co) > abs(pc - po)


def _is_bearish_engulfing(prev: pd.Series, curr: pd.Series) -> bool:
    po, pc = _safe_float(prev.get("open")), _safe_float(prev.get("close"))
    co, cc = _safe_float(curr.get("open")), _safe_float(curr.get("close"))
    prev_bull = pc > po
    curr_bear = cc < co
    engulf = co >= pc and cc <= po
    return prev_bull and curr_bear and engulf and abs(cc - co) > abs(pc - po)


def _is_tweezer_bottom(prev: pd.Series, curr: pd.Series) -> bool:
    p_low = _safe_float(prev.get("low"))
    c_low = _safe_float(curr.get("low"))
    tol = max(p_low, c_low, 1e-9) * 0.0015
    return abs(p_low - c_low) <= tol and _safe_float(curr.get("close")) > _safe_float(curr.get("open"))


def _is_tweezer_top(prev: pd.Series, curr: pd.Series) -> bool:
    p_high = _safe_float(prev.get("high"))
    c_high = _safe_float(curr.get("high"))
    tol = max(p_high, c_high, 1e-9) * 0.0015
    return abs(p_high - c_high) <= tol and _safe_float(curr.get("close")) < _safe_float(curr.get("open"))


def _is_piercing_line(prev: pd.Series, curr: pd.Series) -> bool:
    po, pc = _safe_float(prev.get("open")), _safe_float(prev.get("close"))
    co, cc = _safe_float(curr.get("open")), _safe_float(curr.get("close"))
    midpoint = (po + pc) / 2.0
    return pc < po and cc > co and cc > midpoint and co < pc


def _is_dark_cloud_cover(prev: pd.Series, curr: pd.Series) -> bool:
    po, pc = _safe_float(prev.get("open")), _safe_float(prev.get("close"))
    co, cc = _safe_float(curr.get("open")), _safe_float(curr.get("close"))
    midpoint = (po + pc) / 2.0
    return pc > po and cc < co and cc < midpoint and co > pc


def _is_morning_star(c3: pd.DataFrame) -> bool:
    c1, c2, c3c = c3.iloc[0], c3.iloc[1], c3.iloc[2]
    o1, cl1 = _safe_float(c1.get("open")), _safe_float(c1.get("close"))
    o2, cl2 = _safe_float(c2.get("open")), _safe_float(c2.get("close"))
    o3, cl3 = _safe_float(c3c.get("open")), _safe_float(c3c.get("close"))
    body1 = abs(cl1 - o1)
    body2 = abs(cl2 - o2)
    midpoint = (o1 + cl1) / 2.0
    return cl1 < o1 and body2 <= body1 * 0.6 and cl3 > o3 and cl3 > midpoint


def _is_evening_star(c3: pd.DataFrame) -> bool:
    c1, c2, c3c = c3.iloc[0], c3.iloc[1], c3.iloc[2]
    o1, cl1 = _safe_float(c1.get("open")), _safe_float(c1.get("close"))
    o2, cl2 = _safe_float(c2.get("open")), _safe_float(c2.get("close"))
    o3, cl3 = _safe_float(c3c.get("open")), _safe_float(c3c.get("close"))
    body1 = abs(cl1 - o1)
    body2 = abs(cl2 - o2)
    midpoint = (o1 + cl1) / 2.0
    return cl1 > o1 and body2 <= body1 * 0.6 and cl3 < o3 and cl3 < midpoint


def _is_three_white_soldiers(c3: pd.DataFrame) -> bool:
    arr = c3[["open", "close"]].to_numpy(dtype=float)
    return bool(np.all(arr[:, 1] > arr[:, 0]) and np.all(np.diff(arr[:, 1]) > 0))


def _is_three_black_crows(c3: pd.DataFrame) -> bool:
    arr = c3[["open", "close"]].to_numpy(dtype=float)
    return bool(np.all(arr[:, 1] < arr[:, 0]) and np.all(np.diff(arr[:, 1]) < 0))


def _is_three_inside_up(c3: pd.DataFrame) -> bool:
    first, second, third = c3.iloc[0], c3.iloc[1], c3.iloc[2]
    f_open, f_close = _safe_float(first["open"]), _safe_float(first["close"])
    s_open, s_close = _safe_float(second["open"]), _safe_float(second["close"])
    t_close = _safe_float(third["close"])
    first_bear = f_close < f_open
    inside = min(f_open, f_close) <= s_open <= max(f_open, f_close) and min(f_open, f_close) <= s_close <= max(f_open, f_close)
    return first_bear and inside and t_close > max(f_open, f_close)


def _is_three_inside_down(c3: pd.DataFrame) -> bool:
    first, second, third = c3.iloc[0], c3.iloc[1], c3.iloc[2]
    f_open, f_close = _safe_float(first["open"]), _safe_float(first["close"])
    s_open, s_close = _safe_float(second["open"]), _safe_float(second["close"])
    t_close = _safe_float(third["close"])
    first_bull = f_close > f_open
    inside = min(f_open, f_close) <= s_open <= max(f_open, f_close) and min(f_open, f_close) <= s_close <= max(f_open, f_close)
    return first_bull and inside and t_close < min(f_open, f_close)


def _is_double_bottom(c20: pd.DataFrame) -> bool:
    lows = c20["low"].reset_index(drop=True)
    highs = c20["high"].reset_index(drop=True)
    closes = c20["close"].reset_index(drop=True)
    piv = [i for i in range(1, len(lows) - 1) if lows.iloc[i] <= lows.iloc[i - 1] and lows.iloc[i] <= lows.iloc[i + 1]]
    if len(piv) < 2:
        return False
    i1, i2 = piv[-2], piv[-1]
    if (i2 - i1) < 3:
        return False
    l1, l2 = lows.iloc[i1], lows.iloc[i2]
    if abs(l1 - l2) / max(l1, l2, 1e-9) > 0.009:
        return False
    neckline = float(highs.iloc[i1 : i2 + 1].max())
    return float(closes.iloc[-1]) > neckline * 1.001


def _is_double_top(c20: pd.DataFrame) -> bool:
    lows = c20["low"].reset_index(drop=True)
    highs = c20["high"].reset_index(drop=True)
    closes = c20["close"].reset_index(drop=True)
    piv = [i for i in range(1, len(highs) - 1) if highs.iloc[i] >= highs.iloc[i - 1] and highs.iloc[i] >= highs.iloc[i + 1]]
    if len(piv) < 2:
        return False
    i1, i2 = piv[-2], piv[-1]
    if (i2 - i1) < 3:
        return False
    h1, h2 = highs.iloc[i1], highs.iloc[i2]
    if abs(h1 - h2) / max(h1, h2, 1e-9) > 0.009:
        return False
    neckline = float(lows.iloc[i1 : i2 + 1].min())
    return float(closes.iloc[-1]) < neckline * 0.999


def _is_head_shoulders(c50: pd.DataFrame) -> bool:
    highs = c50["high"].to_numpy(dtype=float)
    if len(highs) < 25:
        return False
    p = _pivot_high_idx(highs)
    if len(p) < 3:
        return False
    ls, hd, rs = p[-3], p[-2], p[-1]
    h_ls, h_hd, h_rs = highs[ls], highs[hd], highs[rs]
    shoulders_close = abs(h_ls - h_rs) / max(h_ls, h_rs, 1e-9) <= 0.015
    return shoulders_close and h_hd > h_ls and h_hd > h_rs


def _is_inverse_head_shoulders(c50: pd.DataFrame) -> bool:
    lows = c50["low"].to_numpy(dtype=float)
    if len(lows) < 25:
        return False
    p = _pivot_low_idx(lows)
    if len(p) < 3:
        return False
    ls, hd, rs = p[-3], p[-2], p[-1]
    l_ls, l_hd, l_rs = lows[ls], lows[hd], lows[rs]
    shoulders_close = abs(l_ls - l_rs) / max(l_ls, l_rs, 1e-9) <= 0.015
    return shoulders_close and l_hd < l_ls and l_hd < l_rs


def _is_bull_flag(c20: pd.DataFrame) -> bool:
    part1 = c20.iloc[:8]
    part2 = c20.iloc[8:]
    pole_move = (_safe_float(part1["close"].iloc[-1]) - _safe_float(part1["open"].iloc[0])) / max(_safe_float(part1["open"].iloc[0]), 1e-9)
    flag_range = (_safe_float(part2["high"].max()) - _safe_float(part2["low"].min())) / max(_safe_float(c20["close"].iloc[-1]), 1e-9)
    flag_slope = (_safe_float(part2["close"].iloc[-1]) - _safe_float(part2["close"].iloc[0])) / max(_safe_float(part2["close"].iloc[0]), 1e-9)
    return pole_move >= 0.02 and flag_range <= 0.035 and -0.03 <= flag_slope <= 0.01


def _is_bear_flag(c20: pd.DataFrame) -> bool:
    part1 = c20.iloc[:8]
    part2 = c20.iloc[8:]
    pole_move = (_safe_float(part1["close"].iloc[-1]) - _safe_float(part1["open"].iloc[0])) / max(_safe_float(part1["open"].iloc[0]), 1e-9)
    flag_range = (_safe_float(part2["high"].max()) - _safe_float(part2["low"].min())) / max(_safe_float(c20["close"].iloc[-1]), 1e-9)
    flag_slope = (_safe_float(part2["close"].iloc[-1]) - _safe_float(part2["close"].iloc[0])) / max(_safe_float(part2["close"].iloc[0]), 1e-9)
    return pole_move <= -0.02 and flag_range <= 0.035 and -0.01 <= flag_slope <= 0.03


def _is_ascending_triangle(c50: pd.DataFrame) -> bool:
    highs = c50["high"].tail(20).to_numpy(dtype=float)
    lows = c50["low"].tail(20).to_numpy(dtype=float)
    if len(highs) < 20:
        return False
    res_band = np.std(highs[-8:]) / max(np.mean(highs[-8:]), 1e-9)
    rising_lows = np.polyfit(np.arange(len(lows)), lows, 1)[0] > 0
    return res_band < 0.005 and rising_lows


def _is_descending_triangle(c50: pd.DataFrame) -> bool:
    highs = c50["high"].tail(20).to_numpy(dtype=float)
    lows = c50["low"].tail(20).to_numpy(dtype=float)
    if len(lows) < 20:
        return False
    sup_band = np.std(lows[-8:]) / max(np.mean(lows[-8:]), 1e-9)
    falling_highs = np.polyfit(np.arange(len(highs)), highs, 1)[0] < 0
    return sup_band < 0.005 and falling_highs


def _is_symmetrical_triangle(c50: pd.DataFrame) -> str | None:
    highs = c50["high"].tail(20).to_numpy(dtype=float)
    lows = c50["low"].tail(20).to_numpy(dtype=float)
    closes = c50["close"].tail(20).to_numpy(dtype=float)
    if len(highs) < 20:
        return None
    high_slope = np.polyfit(np.arange(len(highs)), highs, 1)[0]
    low_slope = np.polyfit(np.arange(len(lows)), lows, 1)[0]
    if not (high_slope < 0 and low_slope > 0):
        return None
    top_line = highs.max()
    bot_line = lows.min()
    if closes[-1] > top_line * 0.998:
        return "BULLISH"
    if closes[-1] < bot_line * 1.002:
        return "BEARISH"
    return "NEUTRAL"


def _is_rising_wedge(c50: pd.DataFrame) -> bool:
    highs = c50["high"].tail(25).to_numpy(dtype=float)
    lows = c50["low"].tail(25).to_numpy(dtype=float)
    if len(highs) < 25:
        return False
    hs = np.polyfit(np.arange(len(highs)), highs, 1)[0]
    ls = np.polyfit(np.arange(len(lows)), lows, 1)[0]
    narrowing = (highs[-1] - lows[-1]) < (highs[0] - lows[0])
    return hs > 0 and ls > 0 and ls > hs and narrowing


def _is_falling_wedge(c50: pd.DataFrame) -> bool:
    highs = c50["high"].tail(25).to_numpy(dtype=float)
    lows = c50["low"].tail(25).to_numpy(dtype=float)
    if len(highs) < 25:
        return False
    hs = np.polyfit(np.arange(len(highs)), highs, 1)[0]
    ls = np.polyfit(np.arange(len(lows)), lows, 1)[0]
    narrowing = (highs[-1] - lows[-1]) < (highs[0] - lows[0])
    return hs < 0 and ls < 0 and hs > ls and narrowing


def _is_cup_handle(c50: pd.DataFrame) -> bool:
    if len(c50) < 35:
        return False
    closes = c50["close"].to_numpy(dtype=float)
    left = closes[:20]
    handle = closes[-10:]
    trough_idx = int(np.argmin(left))
    left_ok = trough_idx > 4 and trough_idx < 16
    rim_close = max(left[0], left[-1])
    cup_depth = (rim_close - left[trough_idx]) / max(rim_close, 1e-9)
    handle_pullback = (max(handle) - min(handle)) / max(max(handle), 1e-9)
    return left_ok and 0.02 <= cup_depth <= 0.2 and handle_pullback <= 0.03 and closes[-1] >= np.percentile(left, 80)


def _is_hh_hl(c20: pd.DataFrame) -> bool:
    highs = c20["high"].tail(6).to_numpy(dtype=float)
    lows = c20["low"].tail(6).to_numpy(dtype=float)
    if len(highs) < 6:
        return False
    return int((np.diff(highs) > 0).sum()) >= 4 and int((np.diff(lows) > 0).sum()) >= 4


def _is_lh_ll(c20: pd.DataFrame) -> bool:
    highs = c20["high"].tail(6).to_numpy(dtype=float)
    lows = c20["low"].tail(6).to_numpy(dtype=float)
    if len(highs) < 6:
        return False
    return int((np.diff(highs) < 0).sum()) >= 4 and int((np.diff(lows) < 0).sum()) >= 4


def _is_volume_climax(df: pd.DataFrame) -> bool:
    if "volume" not in df.columns or len(df) < 20:
        return False
    vol = df["volume"].tail(20)
    return _safe_float(vol.iloc[-1]) > (_safe_float(vol.mean()) * 2.5)


def _is_volume_divergence(df: pd.DataFrame) -> bool:
    if "volume" not in df.columns or len(df) < 12:
        return False
    close = df["close"].tail(12).to_numpy(dtype=float)
    vol = df["volume"].tail(12).to_numpy(dtype=float)
    close_slope = np.polyfit(np.arange(len(close)), close, 1)[0]
    vol_slope = np.polyfit(np.arange(len(vol)), vol, 1)[0]
    return (close_slope > 0 and vol_slope < 0) or (close_slope < 0 and vol_slope > 0)


def _breakout_with_volume(df: pd.DataFrame) -> str | None:
    if "volume" not in df.columns or len(df) < 25:
        return None
    closes = df["close"].tail(21)
    highs = df["high"].tail(21)
    lows = df["low"].tail(21)
    vol = df["volume"].tail(21)
    prior_high = _safe_float(highs.iloc[:-1].max())
    prior_low = _safe_float(lows.iloc[:-1].min())
    last_close = _safe_float(closes.iloc[-1])
    vol_ok = _safe_float(vol.iloc[-1]) >= (_safe_float(vol.iloc[:-1].mean()) * 1.5)
    if not vol_ok:
        return None
    if last_close > prior_high:
        return "BULLISH"
    if last_close < prior_low:
        return "BEARISH"
    return None


def _pivot_high_idx(arr: np.ndarray) -> List[int]:
    idx: List[int] = []
    for i in range(1, len(arr) - 1):
        if arr[i] >= arr[i - 1] and arr[i] >= arr[i + 1]:
            idx.append(i)
    return idx


def _pivot_low_idx(arr: np.ndarray) -> List[int]:
    idx: List[int] = []
    for i in range(1, len(arr) - 1):
        if arr[i] <= arr[i - 1] and arr[i] <= arr[i + 1]:
            idx.append(i)
    return idx


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
