"""Indicator and market-structure helpers using pandas-ta."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    import pandas_ta as ta
except Exception:  # pragma: no cover - runtime fallback
    ta = None


def apply_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Apply required indicator set to OHLCV dataframe."""

    if df.empty:
        return df.copy()

    out = df.copy()
    out = out.rename(columns=str.lower)
    for col in ("open", "high", "low", "close", "volume"):
        if col not in out.columns:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=["open", "high", "low", "close"])
    out = out[out["volume"].fillna(0.0) > 0.0].copy()

    if out.empty:
        return out

    if len(out) < 30:
        for col in (
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
        ):
            out[col] = np.nan
        return out

    for length in (9, 21, 50, 200):
        out[f"ema_{length}"] = _safe_series(lambda: _ema(out["close"], length), out.index)

    macd_line, macd_signal, macd_hist = _safe_tuple_series(lambda: _macd(out["close"]), out.index, 3)
    out["macd"] = macd_line
    out["macd_signal"] = macd_signal
    out["macd_hist"] = macd_hist

    out["adx_14"] = _safe_series(lambda: _adx(out, 14), out.index)
    out["rsi_14"] = _safe_series(lambda: _rsi(out["close"], 14), out.index)
    stoch_k, stoch_d = _safe_tuple_series(lambda: _stochrsi(out["close"], 14, 3, 3), out.index, 2)
    out["stochrsi_k"] = stoch_k
    out["stochrsi_d"] = stoch_d
    out["roc_10"] = _safe_series(lambda: _roc(out["close"], 10), out.index)
    out["atr_14"] = _safe_series(lambda: _atr(out, 14), out.index)

    bb_upper, bb_mid, bb_lower = _safe_tuple_series(lambda: _bbands(out["close"], 20, 2.0), out.index, 3)
    out["bb_upper"] = bb_upper
    out["bb_mid"] = bb_mid
    out["bb_lower"] = bb_lower
    out["bb_width"] = (out["bb_upper"] - out["bb_lower"]) / out["bb_mid"].replace(0, np.nan)

    kc_upper, kc_mid, kc_lower = _safe_tuple_series(lambda: _keltner(out, 20, 1.5), out.index, 3)
    out["kc_upper"] = kc_upper
    out["kc_mid"] = kc_mid
    out["kc_lower"] = kc_lower

    out["vwap"] = _safe_series(lambda: _vwap(out), out.index)
    out["obv"] = _safe_series(lambda: _obv(out["close"], out["volume"]), out.index)
    out["vol_sma_20"] = _safe_series(lambda: out["volume"].rolling(20, min_periods=1).mean(), out.index)
    out["vol_ratio"] = out["volume"] / out["vol_sma_20"].replace(0, np.nan)

    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def timeframe_confluence_score(latest: pd.Series) -> tuple[int, Dict[str, bool]]:
    """Compute weighted confluence score for one timeframe (max 10).

    Weights:
    - +2: price > EMA21
    - +2: RSI > 50
    - +2: price > VWAP
    - +1: MACD histogram > 0
    - +1: ADX > 25
    - +2: EMA9 > EMA21
    """

    checks = {
        "price_above_ema21": _safe_gt(latest.get("close"), latest.get("ema_21")),
        "rsi_above_50": _safe_gt(latest.get("rsi_14"), 50.0),
        "price_above_vwap": _safe_gt(latest.get("close"), latest.get("vwap")),
        "macd_hist_positive": _safe_gt(latest.get("macd_hist"), 0.0),
        "adx_above_25": _safe_gt(latest.get("adx_14"), 25.0),
        "ema9_above_ema21": _safe_gt(latest.get("ema_9"), latest.get("ema_21")),
    }
    score = 0
    score += 2 if checks["price_above_ema21"] else 0
    score += 2 if checks["rsi_above_50"] else 0
    score += 2 if checks["price_above_vwap"] else 0
    score += 1 if checks["macd_hist_positive"] else 0
    score += 1 if checks["adx_above_25"] else 0
    score += 2 if checks["ema9_above_ema21"] else 0
    return int(score), checks


def detect_market_regime(latest: pd.Series, atr_avg_20: float) -> str:
    """Detect market regime based on ADX, ATR expansion, and BB location."""

    adx = float(latest.get("adx_14", 0) or 0)
    close = float(latest.get("close", 0) or 0)
    ema50 = float(latest.get("ema_50", 0) or 0)
    atr = float(latest.get("atr_14", 0) or 0)
    bb_upper = float(latest.get("bb_upper", 0) or 0)
    bb_lower = float(latest.get("bb_lower", 0) or 0)

    if adx > 25 and close > ema50:
        return "TRENDING"
    if atr_avg_20 > 0 and atr > (1.5 * atr_avg_20):
        return "VOLATILE"
    if adx < 20 and bb_lower <= close <= bb_upper:
        return "SIDEWAYS"
    return "NEUTRAL"


def compute_pivots(latest_daily: pd.Series, prev_daily: pd.Series) -> Dict[str, float]:
    """Compute classic and Camarilla pivot levels from previous daily candle."""

    ph = float(prev_daily.get("high", 0.0))
    pl = float(prev_daily.get("low", 0.0))
    pc = float(prev_daily.get("close", 0.0))
    rng = ph - pl

    pp = (ph + pl + pc) / 3.0
    r1 = (2 * pp) - pl
    s1 = (2 * pp) - ph
    r2 = pp + rng
    s2 = pp - rng
    r3 = ph + 2 * (pp - pl)
    s3 = pl - 2 * (ph - pp)

    h4 = pc + rng * 1.1 / 2
    h3 = pc + rng * 1.1 / 4
    l3 = pc - rng * 1.1 / 4
    l4 = pc - rng * 1.1 / 2

    return {
        "classic_pp": _r(pp),
        "classic_r1": _r(r1),
        "classic_r2": _r(r2),
        "classic_r3": _r(r3),
        "classic_s1": _r(s1),
        "classic_s2": _r(s2),
        "classic_s3": _r(s3),
        "camarilla_h4": _r(h4),
        "camarilla_h3": _r(h3),
        "camarilla_l3": _r(l3),
        "camarilla_l4": _r(l4),
        "close": _r(float(latest_daily.get("close", 0.0))),
    }


def swing_levels(df: pd.DataFrame, lookback: int = 20) -> Dict[str, float]:
    """Find recent swing high/low over lookback candles."""

    if df.empty:
        return {"swing_high": 0.0, "swing_low": 0.0}
    tail = df.tail(lookback)
    return {
        "swing_high": _r(float(tail["high"].max())),
        "swing_low": _r(float(tail["low"].min())),
    }


def round_number_levels(price: float) -> Dict[str, float]:
    """Generate nearby round-number levels."""

    if price <= 0:
        return {"round_floor": 0.0, "round_ceil": 0.0}
    step = 100.0 if price > 1000 else 50.0 if price > 300 else 10.0
    floor_lvl = np.floor(price / step) * step
    ceil_lvl = np.ceil(price / step) * step
    return {"round_floor": _r(float(floor_lvl)), "round_ceil": _r(float(ceil_lvl))}


def build_pattern_reasons(indicators_by_tf: Dict[str, Dict[str, float]], regime: str) -> List[Dict[str, str]]:
    """Create pattern-style explanations from indicator states."""

    patterns: List[Dict[str, str]] = []
    m15 = indicators_by_tf.get("15m", {})
    h1 = indicators_by_tf.get("1h", {})

    if m15.get("macd_hist", 0) > 0 and m15.get("rsi_14", 0) > 55:
        patterns.append(
            {
                "name": "Momentum Alignment",
                "reason": "15m MACD histogram positive with RSI above 55.",
            }
        )
    if m15.get("vol_ratio", 1) > 1.2:
        patterns.append(
            {
                "name": "Volume Expansion",
                "reason": "Current volume is above 20-period average.",
            }
        )
    if h1.get("close", 0) > h1.get("ema_50", 0) > h1.get("ema_200", 0):
        patterns.append(
            {
                "name": "Higher-Timeframe Uptrend",
                "reason": "1h close above EMA50 and EMA50 above EMA200.",
            }
        )
    if regime == "VOLATILE":
        patterns.append(
            {
                "name": "Volatility Expansion",
                "reason": "ATR expansion indicates likely stop-hunts and rapid swings.",
            }
        )
    if not patterns:
        patterns.append({"name": "No Strong Pattern", "reason": "Confluence is mixed right now."})
    return patterns


def _safe_bool(value: Any) -> bool:
    """Safely cast indicator check values to bool."""

    try:
        return bool(value)
    except Exception:
        return False


def _safe_gt(left: Any, right: Any) -> bool:
    """Safe numeric greater-than comparison with NaN guard."""

    try:
        lval = float(left)
        rval = float(right)
        if np.isnan(lval) or np.isnan(rval):
            return False
        return lval > rval
    except Exception:
        return False


def _r(value: float) -> float:
    """Round helper."""

    return round(float(value), 4)


def _safe_series(fn, index: pd.Index) -> pd.Series:
    """Execute indicator function and return aligned NaN series on failure."""

    try:
        series = fn()
        if isinstance(series, pd.Series):
            return series.reindex(index)
    except Exception:
        pass
    return pd.Series(np.nan, index=index, dtype=float)


def _safe_tuple_series(fn, index: pd.Index, expected: int) -> Tuple[pd.Series, ...]:
    """Execute tuple-returning indicator function safely."""

    try:
        values = fn()
        if isinstance(values, tuple) and len(values) == expected and all(isinstance(v, pd.Series) for v in values):
            return tuple(v.reindex(index) for v in values)
    except Exception:
        pass
    return tuple(pd.Series(np.nan, index=index, dtype=float) for _ in range(expected))


def _ema(series: pd.Series, length: int) -> pd.Series:
    """EMA with pandas-ta fallback to pandas ewm."""

    if ta is not None:
        try:
            return ta.ema(series, length=length)
        except Exception:
            pass
    return series.ewm(span=length, adjust=False).mean()


def _macd(series: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """MACD calculation with fallback."""

    if ta is not None:
        try:
            macd = ta.macd(series, fast=12, slow=26, signal=9)
            if macd is not None and not macd.empty:
                cols = list(macd.columns)
                return macd[cols[0]], macd[cols[1]], macd[cols[2]]
        except Exception:
            pass
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    line = ema12 - ema26
    signal = line.ewm(span=9, adjust=False).mean()
    return line, signal, line - signal


def _rsi(series: pd.Series, length: int) -> pd.Series:
    """RSI calculation with fallback."""

    if ta is not None:
        try:
            return ta.rsi(series, length=length)
        except Exception:
            pass
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    avg_gain = up.ewm(alpha=1 / length, adjust=False).mean()
    avg_loss = down.ewm(alpha=1 / length, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _stochrsi(series: pd.Series, length: int, k: int, d: int) -> Tuple[pd.Series, pd.Series]:
    """Stochastic RSI with fallback."""

    if ta is not None:
        try:
            stoch = ta.stochrsi(series, length=length, rsi_length=length, k=k, d=d)
            if stoch is not None and not stoch.empty:
                cols = list(stoch.columns)
                return stoch[cols[0]], stoch[cols[1]]
        except Exception:
            pass
    rsi = _rsi(series, length)
    min_rsi = rsi.rolling(length, min_periods=1).min()
    max_rsi = rsi.rolling(length, min_periods=1).max()
    stoch = ((rsi - min_rsi) / (max_rsi - min_rsi).replace(0, np.nan)) * 100
    k_line = stoch.rolling(k, min_periods=1).mean()
    d_line = k_line.rolling(d, min_periods=1).mean()
    return k_line, d_line


def _roc(series: pd.Series, length: int) -> pd.Series:
    """Rate of change with fallback."""

    if ta is not None:
        try:
            return ta.roc(series, length=length)
        except Exception:
            pass
    return (series / series.shift(length) - 1.0) * 100.0


def _atr(df: pd.DataFrame, length: int) -> pd.Series:
    """ATR with fallback."""

    if ta is not None:
        try:
            return ta.atr(df["high"], df["low"], df["close"], length=length)
        except Exception:
            pass
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift(1)).abs()
    lc = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(length, min_periods=1).mean()


def _adx(df: pd.DataFrame, length: int) -> pd.Series:
    """ADX with fallback approximation."""

    if ta is not None:
        try:
            adx = ta.adx(df["high"], df["low"], df["close"], length=length)
            if adx is not None and not adx.empty:
                col = [c for c in adx.columns if "ADX" in c.upper()]
                if col:
                    return adx[col[0]]
        except Exception:
            pass
    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr = _atr(df, 1).replace(0, np.nan)
    plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(length, min_periods=1).sum() / tr.rolling(
        length, min_periods=1
    ).sum()
    minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(length, min_periods=1).sum() / tr.rolling(
        length, min_periods=1
    ).sum()
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100
    return dx.rolling(length, min_periods=1).mean()


def _bbands(series: pd.Series, length: int, std: float) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Bollinger Bands with fallback."""

    if ta is not None:
        try:
            bb = ta.bbands(series, length=length, std=std)
            if bb is not None and not bb.empty:
                cols = list(bb.columns)
                return bb[cols[0]], bb[cols[1]], bb[cols[2]]
        except Exception:
            pass
    mid = series.rolling(length, min_periods=1).mean()
    sigma = series.rolling(length, min_periods=1).std(ddof=0)
    upper = mid + std * sigma
    lower = mid - std * sigma
    return upper, mid, lower


def _keltner(df: pd.DataFrame, length: int, scalar: float) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Keltner channels with fallback."""

    if ta is not None:
        try:
            kc = ta.kc(df["high"], df["low"], df["close"], length=length, scalar=scalar)
            if kc is not None and not kc.empty:
                cols = list(kc.columns)
                return kc[cols[0]], kc[cols[1]], kc[cols[2]]
        except Exception:
            pass
    mid = _ema(df["close"], length)
    atr = _atr(df, length)
    upper = mid + scalar * atr
    lower = mid - scalar * atr
    return upper, mid, lower


def _vwap(df: pd.DataFrame) -> pd.Series:
    """VWAP with daily reset for intraday candles."""

    if not isinstance(df.index, pd.DatetimeIndex):
        tp = (df["high"] + df["low"] + df["close"]) / 3.0
        return (tp * df["volume"]).cumsum() / df["volume"].replace(0, np.nan).cumsum()
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    day_key = df.index.date
    pv = tp * df["volume"]
    return pv.groupby(day_key).cumsum() / df["volume"].replace(0, np.nan).groupby(day_key).cumsum()


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume with fallback."""

    if ta is not None:
        try:
            return ta.obv(close, volume)
        except Exception:
            pass
    direction = np.sign(close.diff().fillna(0))
    return (direction * volume).cumsum()
