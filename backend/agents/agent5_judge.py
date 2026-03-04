"""Agent 5: final judge combining all agents into trade decision and rationale."""

from __future__ import annotations

import math
import os
from typing import Any, Dict, List

from agents.agent1_data_validator import DataContext
from agents.agent4_sentiment import call_groq_text
from utils.data_fetcher import current_market_session, fetch_market_overview
from utils.trade_levels import calculate_position_size, calculate_trade_levels


async def run(
    data_context: DataContext,
    quant: Dict[str, Any],
    options: Dict[str, Any],
    sentiment: Dict[str, Any],
    capital: float,
) -> Dict[str, Any]:
    """Compute conviction, apply kill switches, and produce trade plan."""

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

        direction = _direction(quant)
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
        )

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
            },
            "market_context": {
                "regime": regime,
                "adx": round(adx, 3),
                "india_vix": round(india_vix, 3),
                "session": session.status,
                "session_warning": session.warning,
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
            "conviction_breakdown": {"quant": 50.0, "mtf": 50.0, "options": 50.0, "sentiment": 50.0, "data_quality": 0.0},
            "market_context": {"regime": "UNKNOWN", "adx": 0.0, "india_vix": 0.0, "session": "UNKNOWN", "session_warning": ""},
            "agent_status": "RED",
            "error": f"agent5_failure: {exc}",
        }


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
) -> tuple[Dict[str, bool], List[str]]:
    """Apply hard kill-switch conditions and collect no-trade reasons."""

    reasons: List[str] = []
    status = {
        "data_quality_gate": data_quality >= 60,
        "volatility_gate": not (regime == "VOLATILE" and conviction < 75),
        "sentiment_gate": not (sentiment_score < 25 and "BUY" in signal),
        "trend_gate": adx >= 15,
        "opening_window_gate": session_status != "OPENING_VOLATILITY",
        "vix_gate": not (india_vix > 25),
    }
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
    if not status["vix_gate"]:
        reasons.append("India VIX above 25; high fear market.")
    return status, reasons


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
) -> List[str]:
    """Generate concise 3-line rationale from Groq with deterministic fallback."""

    system_prompt = "You are a concise quant trading assistant. Provide exactly 3 short lines."
    user_prompt = (
        f"Symbol: {symbol}\nSignal: {signal}\nConviction: {conviction:.2f}\nDirection: {direction}\n"
        f"Regime: {regime}\nNo-trade reasons: {reasons}\nQuant score: {quant.get('quant_score')}\n"
        f"MTF: {quant.get('mtf_score')}\nOptions score: {options.get('options_score')}\n"
        f"Sentiment score: {sentiment.get('sentiment_score')}\n"
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
