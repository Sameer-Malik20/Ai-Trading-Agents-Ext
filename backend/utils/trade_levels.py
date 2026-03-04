"""Trade level and risk sizing helpers."""

from __future__ import annotations

import math
from typing import Dict


def calculate_trade_levels(signal: str, ltp: float, atr: float) -> Dict[str, float]:
    """Compute ATR-based entry, SL, and three targets."""

    entry = float(ltp or 0.0)
    atr_value = max(float(atr or 0.0), max(entry * 0.002, 0.05))
    side = signal.upper()

    if side in {"BUY", "STRONG_BUY"}:
        sl = entry - (1.5 * atr_value)
        t1 = entry + (1.5 * atr_value)
        t2 = entry + (3.0 * atr_value)
        t3 = entry + (4.5 * atr_value)
    elif side in {"SELL", "STRONG_SELL"}:
        sl = entry + (1.5 * atr_value)
        t1 = entry - (1.5 * atr_value)
        t2 = entry - (3.0 * atr_value)
        t3 = entry - (4.5 * atr_value)
    else:
        sl = entry
        t1 = entry
        t2 = entry
        t3 = entry

    risk = abs(entry - sl)
    reward = abs(t2 - entry)
    rr = (reward / risk) if risk > 0 else 0.0

    return {
        "entry": round(entry, 4),
        "sl": round(sl, 4),
        "t1": round(t1, 4),
        "t2": round(t2, 4),
        "t3": round(t3, 4),
        "atr_used": round(atr_value, 4),
        "risk_reward_ratio": round(rr, 3),
    }


def calculate_position_size(
    capital: float,
    risk_per_trade: float,
    entry: float,
    sl: float,
    max_position_pct: float = 0.10,
) -> Dict[str, float]:
    """Calculate quantity with risk cap and max capital allocation cap."""

    cap = max(float(capital or 0.0), 0.0)
    risk_fraction = max(min(float(risk_per_trade or 0.0), 0.05), 0.001)
    entry_price = max(float(entry or 0.0), 0.0)
    stop_loss = max(float(sl or 0.0), 0.0)
    per_share_risk = abs(entry_price - stop_loss)

    risk_budget = cap * risk_fraction
    if per_share_risk <= 0 or entry_price <= 0:
        return {
            "quantity": 0.0,
            "risk_budget": round(risk_budget, 2),
            "capital_limit": round(cap * max_position_pct, 2),
            "capital_required": 0.0,
            "effective_risk": 0.0,
        }

    qty_risk = math.floor(risk_budget / per_share_risk)
    cap_limit = cap * max_position_pct
    qty_cap = math.floor(cap_limit / entry_price)
    qty = max(min(qty_risk, qty_cap), 0)
    capital_required = qty * entry_price
    effective_risk = qty * per_share_risk

    return {
        "quantity": float(qty),
        "risk_budget": round(risk_budget, 2),
        "capital_limit": round(cap_limit, 2),
        "capital_required": round(capital_required, 2),
        "effective_risk": round(effective_risk, 2),
    }
