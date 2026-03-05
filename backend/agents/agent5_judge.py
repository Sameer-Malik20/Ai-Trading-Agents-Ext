"""Agent 5: final judge combining all agents into trade decision and rationale."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List

from agents.agent1_data_validator import DataContext
from agents.agent4_sentiment import call_groq_text
from utils.data_fetcher import current_market_session, fetch_market_overview
from utils.trade_levels import calculate_position_size, calculate_trade_levels

LEARNING_FILE = Path(__file__).resolve().parents[1] / "learning_insights.json"


async def run(
    data_context: DataContext,
    quant: Dict[str, Any],
    options: Dict[str, Any],
    sentiment: Dict[str, Any],
    capital: float,
) -> Dict[str, Any]:
    """Compute conviction, apply kill switches, conflicts, and produce trade plan."""

    try:
        quant_score = _clamp(float(quant.get("quant_score", 50.0)))
        mtf_confluence = _clamp(float(quant.get("mtf_confluence", 50.0)))
        options_score = _clamp(float(options.get("options_score", 50.0)))
        sentiment_score = _clamp(float(sentiment.get("sentiment_score", 50.0)))
        data_quality = _clamp(float(data_context.data_quality_score))

        conviction = (
            quant_score * 0.35
            + mtf_confluence * 0.25
            + options_score * 0.20
            + sentiment_score * 0.15
            + data_quality * 0.05
        )

        market_context = {"gap_info": data_context.gap_info if isinstance(data_context.gap_info, dict) else {}}
        gap_info = market_context.get("gap_info", {})
        gap_percent = float(gap_info.get("gap_percent", 0.0) or 0.0)
        gap_type = str(gap_info.get("gap_type", "UNKNOWN")).upper()
        gap_gate_recommended = bool(gap_info.get("opening_gap_gate_recommended", True))

        direction = _direction(quant)
        direction_side = _direction_side(direction)
        conviction, gap_reasons = _apply_gap_adjustment(
            conviction=conviction,
            direction=direction_side,
            gap_type=gap_type,
            gap_percent=gap_percent,
        )

        # Pattern-driven conviction tuning.
        pattern_summary = quant.get("pattern_summary", {}) if isinstance(quant.get("pattern_summary"), dict) else {}
        pattern_adjust = float(pattern_summary.get("conviction_adjustment", 0.0) or 0.0)
        conviction = _clamp(conviction + pattern_adjust)

        patterns = quant.get("patterns", []) if isinstance(quant.get("patterns"), list) else []
        conflict = detect_conflicts(
            quant=quant_score,
            options=options_score,
            sentiment=sentiment_score,
            patterns=patterns,
        )
        conviction = _clamp(conviction - float(conflict.get("penalty", 0)))

        # Learning-based adaptive conviction.
        learning_note = ""
        learning_insights = _load_learning_insights()
        if learning_insights and int(learning_insights.get("total_trades", 0)) >= 20:
            adx = float(quant.get("adx_current", 0.0) or 0.0)
            rsi = float(quant.get("indicators", {}).get("15m", {}).get("rsi_14", 50.0) or 50.0)
            vix = float(fetch_market_overview().get("india_vix", {}).get("last_price") or 0.0)
            best_adx_min = float(learning_insights.get("best_adx_min", 25.0) or 25.0)
            best_rsi = float(learning_insights.get("best_rsi_range", 50.0) or 50.0)
            best_vix_max = float(learning_insights.get("best_vix_max", 22.0) or 22.0)
            min_conviction = float(learning_insights.get("recommended_min_conviction", 60.0) or 60.0)

            if adx >= best_adx_min:
                conviction += 3.0
            if abs(rsi - best_rsi) <= 5.0:
                conviction += 3.0
            if vix <= best_vix_max:
                conviction += 2.0
            if conviction < min_conviction:
                conviction -= 2.0
            conviction = _clamp(conviction)

            learning_note = (
                f"Learned from {int(learning_insights['total_trades'])} trades: "
                f"Win rate {float(learning_insights.get('win_rate', 0.0)) * 100:.0f}%"
            )

        setup_signal = _signal_from_conviction(conviction, direction)
        signal = setup_signal

        market_overview = fetch_market_overview()
        india_vix = float(market_overview.get("india_vix", {}).get("last_price") or 0.0)
        adx = float(quant.get("adx_current", 0.0) or 0.0)
        regime = str(quant.get("regime", "NEUTRAL")).upper()
        session = current_market_session()

        kill_switch_status, reasons = _evaluate_kill_switches(
            data_quality=data_quality,
            regime=regime,
            conviction=conviction,
            sentiment_score=sentiment_score,
            signal=signal,
            adx=adx,
            session_status=session.status,
            india_vix=india_vix,
            gap_percent=gap_percent,
            gap_type=gap_type,
            gap_gate_recommended=gap_gate_recommended,
        )
        reasons = gap_reasons + reasons
        if bool(conflict.get("has_conflict")):
            reasons.append(f"Signal conflict: {conflict.get('reason', 'mixed inputs')}")

        if reasons:
            signal = "AVOID"

        atr = _finite_or(float(quant.get("atr_current", 0.0) or 0.0), 0.0)
        ltp = float(data_context.current_price or 0.0)
        levels_signal = _levels_signal(setup_signal=setup_signal, direction=direction, quant=quant)
        trade_levels = calculate_trade_levels(signal=levels_signal, ltp=ltp, atr=atr)

        risk_per_trade = float(os.getenv("RISK_PER_TRADE", "0.01") or 0.01)
        sizing = calculate_position_size(
            capital=capital,
            risk_per_trade=risk_per_trade,
            entry=float(trade_levels["entry"]),
            sl=float(trade_levels["sl"]),
            max_position_pct=0.10,
        )
        if signal == "AVOID":
            sizing["quantity"] = 0.0
            sizing["capital_required"] = 0.0
            sizing["effective_risk"] = 0.0

        setup_quality, trade_recommendation = _setup_quality(conviction=conviction, has_conflict=bool(conflict.get("has_conflict")))

        rationale = await _groq_rationale(
            symbol=data_context.symbol,
            signal=signal,
            conviction=conviction,
            direction=direction,
            regime=regime,
            reasons=reasons,
            quant=quant,
            options=options,
            sentiment=sentiment,
            learning_note=learning_note,
        )

        return {
            "ok": True,
            "signal": signal,
            "setup_signal": setup_signal,
            "direction": direction,
            "conviction": round(conviction, 2),
            "trade_levels": trade_levels,
            "trade_levels_basis_signal": levels_signal,
            "position_size": sizing,
            "risk_reward_ratio": trade_levels["risk_reward_ratio"],
            "no_trade_reasons": reasons,
            "groq_rationale": rationale,
            "kill_switch_status": kill_switch_status,
            "conviction_breakdown": {
                "quant": round(quant_score, 2),
                "mtf": round(mtf_confluence, 2),
                "options": round(options_score, 2),
                "sentiment": round(sentiment_score, 2),
                "data_quality": round(data_quality, 2),
                "pattern_adjustment": round(pattern_adjust, 2),
                "conflict_penalty": round(float(conflict.get("penalty", 0.0)), 2),
            },
            "signal_conflict": conflict,
            "setup_quality": setup_quality,
            "trade_recommendation": trade_recommendation,
            "learning_note": learning_note,
            "market_context": {
                "regime": regime,
                "adx": round(adx, 3),
                "india_vix": round(india_vix, 3),
                "session": session.status,
                "session_warning": session.warning,
                "gap_percent": round(gap_percent, 4),
                "gap_type": gap_type,
                "gap_direction": str(gap_info.get("gap_direction", "FLAT")).upper(),
                "gap_gate_recommended": gap_gate_recommended,
            },
            "agent_status": "GREEN" if signal != "AVOID" else "RED",
            "error": "",
        }
    except Exception as exc:
        return {
            "ok": False,
            "signal": "AVOID",
            "direction": "NEUTRAL",
            "conviction": 40.0,
            "trade_levels": calculate_trade_levels(signal="AVOID", ltp=data_context.current_price, atr=0.0),
            "position_size": {"quantity": 0.0, "risk_budget": 0.0, "capital_limit": 0.0, "capital_required": 0.0, "effective_risk": 0.0},
            "risk_reward_ratio": 0.0,
            "no_trade_reasons": [f"judge_failure: {exc}"],
            "groq_rationale": [
                "Pipeline entered degraded mode.",
                "Signal forced to AVOID to protect capital.",
                "Retry analysis after data stabilizes.",
            ],
            "kill_switch_status": {},
            "signal_conflict": {"has_conflict": False, "bullish_signals": 0, "bearish_signals": 0, "penalty": 0, "reason": "judge_error"},
            "conviction_breakdown": {"quant": 50.0, "mtf": 50.0, "options": 50.0, "sentiment": 50.0, "data_quality": 0.0},
            "setup_quality": "D_GRADE",
            "trade_recommendation": "Poor setup - avoid completely",
            "learning_note": "",
            "market_context": {"regime": "UNKNOWN", "adx": 0.0, "india_vix": 0.0, "session": "UNKNOWN", "session_warning": ""},
            "agent_status": "RED",
            "error": f"agent5_failure: {exc}",
        }


def detect_conflicts(quant: float, options: float, sentiment: float, patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Detect cross-signal disagreement to prevent forced low-quality trades."""

    bullish_count = 0
    bearish_count = 0

    if quant > 60:
        bullish_count += 1
    elif quant < 40:
        bearish_count += 1

    if options > 60:
        bullish_count += 1
    elif options < 40:
        bearish_count += 1

    if sentiment > 60:
        bullish_count += 1
    elif sentiment < 40:
        bearish_count += 1

    for p in patterns:
        if p.get("direction") == "BULLISH":
            bullish_count += 1
        elif p.get("direction") == "BEARISH":
            bearish_count += 1

    conflict = bullish_count > 0 and bearish_count > 0
    return {
        "has_conflict": conflict,
        "bullish_signals": bullish_count,
        "bearish_signals": bearish_count,
        "penalty": 15 if conflict else 0,
        "reason": f"{bullish_count} bullish vs {bearish_count} bearish",
    }


def _load_learning_insights() -> Dict[str, Any]:
    """Load learning insights if available, else return empty dict."""

    try:
        if LEARNING_FILE.exists():
            return json.loads(LEARNING_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return {}


def _setup_quality(conviction: float, has_conflict: bool) -> tuple[str, str]:
    """Classify setup quality into A/B/C/D actionable bucket."""

    if conviction >= 75 and not has_conflict:
        return "A_GRADE", "Strong setup - consider trading"
    if conviction >= 65 and not has_conflict:
        return "B_GRADE", "Decent setup - small position"
    if conviction >= 55:
        return "C_GRADE", "Weak setup - paper trade only"
    return "D_GRADE", "Poor setup - avoid completely"


def _direction(quant: Dict[str, Any]) -> str:
    """Derive direction from MTF and quant bias."""

    direct = str(quant.get("direction", "NEUTRAL")).upper()
    if direct in {"BULLISH", "BEARISH"}:
        return direct
    mtf_score = int(quant.get("mtf_score", 10))
    if mtf_score > 12:
        return "BULLISH"
    if mtf_score < 8:
        return "BEARISH"
    return "NEUTRAL"


def _signal_from_conviction(conviction: float, direction: str) -> str:
    """Map conviction+direction to signal bands."""

    if conviction < 45:
        return "AVOID"
    if conviction < 55:
        return "WEAK_AVOID"
    if direction == "NEUTRAL":
        return "AVOID"
    if conviction > 70:
        return "STRONG_BUY" if direction == "BULLISH" else "STRONG_SELL"
    return "BUY" if direction == "BULLISH" else "SELL"


def _levels_signal(setup_signal: str, direction: str, quant: Dict[str, Any]) -> str:
    """Choose signal basis for ATR trade-level construction."""

    side = str(setup_signal or "").upper()
    if side in {"BUY", "SELL", "STRONG_BUY", "STRONG_SELL"}:
        return side
    if direction == "BULLISH":
        return "BUY"
    if direction == "BEARISH":
        return "SELL"
    inferred = _infer_side_from_quant(quant)
    if inferred:
        return inferred
    return "AVOID"


def _infer_side_from_quant(quant: Dict[str, Any]) -> str | None:
    """Infer side from 15m indicator bias when MTF direction is neutral."""

    indicators = quant.get("indicators", {}).get("15m", {}) if isinstance(quant.get("indicators"), dict) else {}
    macd_hist = _finite_or(float(indicators.get("macd_hist", 0.0) or 0.0), 0.0)
    rsi = _finite_or(float(indicators.get("rsi_14", 50.0) or 50.0), 50.0)
    close = _finite_or(float(indicators.get("close", 0.0) or 0.0), 0.0)
    ema21 = _finite_or(float(indicators.get("ema_21", 0.0) or 0.0), 0.0)
    bull_votes = int(macd_hist > 0.0) + int(rsi >= 50.0) + int(close > ema21 and ema21 > 0.0)
    bear_votes = int(macd_hist < 0.0) + int(rsi < 50.0) + int(close < ema21 and ema21 > 0.0)
    if bull_votes >= 2:
        return "BUY"
    if bear_votes >= 2:
        return "SELL"
    return None


def _evaluate_kill_switches(
    data_quality: float,
    regime: str,
    conviction: float,
    sentiment_score: float,
    signal: str,
    adx: float,
    session_status: str,
    india_vix: float,
    gap_percent: float,
    gap_type: str,
    gap_gate_recommended: bool,
) -> tuple[Dict[str, bool], List[str]]:
    """Apply hard kill-switch conditions and collect no-trade reasons."""

    reasons: List[str] = []
    status = {
        "data_quality_gate": data_quality >= 60,
        "volatility_gate": not (regime == "VOLATILE" and conviction < 75),
        "sentiment_gate": not (sentiment_score < 25 and "BUY" in signal),
        "trend_gate": adx >= 15,
        "opening_window_gate": session_status != "OPENING_VOLATILITY",
        "opening_gap_gate": bool(gap_gate_recommended) and not (session_status == "OPENING_VOLATILITY" and abs(gap_percent) > 0.5),
        "vix_gate": not (india_vix > 25),
    }
    if not gap_gate_recommended:
        status["opening_gap_gate"] = False

    if not status["data_quality_gate"]:
        reasons.append("Data quality below 60.")
    if not status["volatility_gate"]:
        reasons.append("Volatile regime with insufficient conviction.")
    if not status["sentiment_gate"]:
        reasons.append("Sentiment below 25 blocks BUY setups.")
    if not status["trend_gate"]:
        reasons.append("ADX below 15 indicates no-trend market.")
    if not status["opening_window_gate"]:
        reasons.append("Opening volatility window (09:15-09:30 IST).")
    if not status["opening_gap_gate"]:
        reasons.append("Gap open - wait for price stabilization.")
    if not status["vix_gate"]:
        reasons.append("India VIX above 25; high fear market.")
    return status, reasons


def _apply_gap_adjustment(conviction: float, direction: str, gap_type: str, gap_percent: float) -> tuple[float, List[str]]:
    """Apply opening-gap directional penalties before final signal mapping."""

    adjusted = float(conviction)
    reasons: List[str] = []
    dir_buy = str(direction).upper() == "BUY"
    dir_sell = str(direction).upper() == "SELL"

    if gap_type == "STRONG_GAP_DOWN" and dir_buy:
        adjusted -= 25.0
        reasons.append(f"Gap down {gap_percent:.1f}% - avoid long")
    elif gap_type == "GAP_DOWN" and dir_buy:
        adjusted -= 15.0
    elif gap_type == "STRONG_GAP_UP" and dir_sell:
        adjusted -= 25.0
        reasons.append(f"Gap up {gap_percent:.1f}% - avoid short")
    elif gap_type == "GAP_UP" and dir_sell:
        adjusted -= 15.0

    return _clamp(adjusted), reasons


def _direction_side(direction: str) -> str:
    """Map directional bias into BUY/SELL/NEUTRAL side."""

    direct = str(direction).upper()
    if direct == "BULLISH":
        return "BUY"
    if direct == "BEARISH":
        return "SELL"
    return "NEUTRAL"


async def _groq_rationale(
    symbol: str,
    signal: str,
    conviction: float,
    direction: str,
    regime: str,
    reasons: List[str],
    quant: Dict[str, Any],
    options: Dict[str, Any],
    sentiment: Dict[str, Any],
    learning_note: str,
) -> List[str]:
    """Generate concise 3-line rationale from Groq with deterministic fallback."""

    system_prompt = "You are a concise quant trading assistant. Provide exactly 3 short lines."
    user_prompt = (
        f"Symbol: {symbol}\nSignal: {signal}\nConviction: {conviction:.2f}\nDirection: {direction}\n"
        f"Regime: {regime}\nNo-trade reasons: {reasons}\nQuant score: {quant.get('quant_score')}\n"
        f"MTF: {quant.get('mtf_score')}\nOptions score: {options.get('options_score')}\n"
        f"Sentiment score: {sentiment.get('sentiment_score')}\nLearning note: {learning_note}\n"
        "Explain why this trade makes sense RIGHT NOW."
    )
    text = await call_groq_text(system_prompt=system_prompt, user_prompt=user_prompt)
    if text:
        lines = [line.strip("- ").strip() for line in text.splitlines() if line.strip()]
        if lines:
            return lines[:3]
    if reasons:
        return [
            f"{symbol}: Trade blocked due to risk controls.",
            f"Primary blocker: {reasons[0]}",
            "Wait for cleaner trend/sentiment alignment before entry.",
        ]
    return [
        f"{symbol}: {signal} setup with {conviction:.1f} conviction.",
        f"Direction {direction} in {regime} regime with multi-factor confluence.",
        "Use strict SL and position sizing before execution.",
    ]


def _clamp(value: float) -> float:
    """Clamp score values to 0-100."""

    return max(0.0, min(100.0, float(value)))


def _finite_or(value: float, fallback: float) -> float:
    """Return value if finite else fallback."""

    return float(value) if math.isfinite(float(value)) else float(fallback)
