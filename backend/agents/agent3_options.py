"""Agent 3: options chain intelligence for PCR, max pain, and OI levels."""

from __future__ import annotations

from typing import Any, Dict

from utils.options_analyzer import analyze_options


def run(symbol: str, current_price: float) -> Dict[str, Any]:
    """Run options analysis safely with neutral degraded fallback."""

    try:
        result = analyze_options(symbol=symbol, spot_price=current_price)
        return {
            "ok": not result.get("data_unavailable", True),
            **result,
            "agent_status": "GREEN" if not result.get("data_unavailable", True) else "AMBER",
        }
    except Exception as exc:
        return {
            "ok": False,
            "options_score": 50.0,
            "pcr": 1.0,
            "max_pain": float(current_price or 0.0),
            "key_resistance": float(current_price or 0.0) * 1.01 if current_price else 0.0,
            "key_support": float(current_price or 0.0) * 0.99 if current_price else 0.0,
            "signal": "NEUTRAL",
            "iv_rank": 50.0,
            "iv_state": "UNKNOWN",
            "data_unavailable": True,
            "error": f"agent3_failure: {exc}",
            "raw_summary": {},
            "agent_status": "RED",
        }
