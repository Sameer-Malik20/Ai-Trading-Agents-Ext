"""Self-learning utilities based on completed trade outcomes."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


def avg(values: List[float], default: float = 0.0) -> float:
    """Return mean of list with safe fallback."""

    clean = [float(v) for v in values if v is not None]
    if not clean:
        return float(default)
    return float(sum(clean) / len(clean))


def analyze_performance(trades: List[Dict[str, Any]], output_file: Path) -> Dict[str, Any] | None:
    """Analyze completed trades and persist adaptive learning insights."""

    if len(trades) < 10:
        return None

    wins = [t for t in trades if str(t.get("outcome", "")).startswith("WIN")]
    losses = [t for t in trades if str(t.get("outcome", "")) == "LOSS"]
    total = len(trades)
    accuracy = len(wins) / total if total else 0.0

    insights: Dict[str, Any] = {
        "total_trades": total,
        "win_rate": round(accuracy, 3),
        "loss_rate": round((len(losses) / total) if total else 0.0, 3),
        "avg_conviction_wins": round(avg([float(t.get("conviction", 0.0)) for t in wins], 0.0), 3),
        "avg_conviction_losses": round(avg([float(t.get("conviction", 0.0)) for t in losses], 0.0), 3),
        "best_rsi_range": round(avg([float(t.get("rsi", 50.0)) for t in wins], 50.0), 3),
        "best_adx_min": round(avg([float(t.get("adx", 20.0)) for t in wins], 20.0), 3),
        "best_pcr_range": round(avg([float(t.get("pcr", 1.0)) for t in wins], 1.0), 3),
        "recommended_min_conviction": round(avg([float(t.get("conviction", 60.0)) for t in wins], 60.0) - 5.0, 3),
        "best_vix_max": round(avg([float(t.get("vix", 22.0)) for t in wins], 22.0), 3),
        "last_updated": datetime.now().isoformat(),
    }

    if accuracy < 0.45:
        insights["action"] = "TIGHTEN"
        insights["conviction_adjust"] = 5
        insights["adx_adjust"] = 3
    elif accuracy > 0.70:
        insights["action"] = "RELAX"
        insights["conviction_adjust"] = -2
        insights["adx_adjust"] = 0
    else:
        insights["action"] = "MAINTAIN"
        insights["conviction_adjust"] = 0
        insights["adx_adjust"] = 0

    output_file.write_text(json.dumps(insights, indent=2), encoding="utf-8")
    return insights


def load_learning_insights(path: Path) -> Dict[str, Any]:
    """Load persisted learning insights if available."""

    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return {}


def performance_summary(trades: List[Dict[str, Any]], insights: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Create API-friendly summary of pending/completed outcomes."""

    insights = insights or {}
    completed = [t for t in trades if str(t.get("outcome", "PENDING")) != "PENDING"]
    pending = [t for t in trades if str(t.get("outcome", "PENDING")) == "PENDING"]
    wins = [t for t in completed if str(t.get("outcome", "")).startswith("WIN")]
    losses = [t for t in completed if str(t.get("outcome", "")) == "LOSS"]

    best_symbol = ""
    if wins:
        counts: Dict[str, int] = {}
        for t in wins:
            sym = str(t.get("symbol", ""))
            counts[sym] = counts.get(sym, 0) + 1
        best_symbol = max(counts, key=counts.get)

    recommendation = "Need 20+ trades to learn"
    if insights and int(insights.get("total_trades", 0)) >= 20:
        recommendation = f"Learning active ({float(insights.get('win_rate', 0.0)) * 100:.1f}% win rate)"

    return {
        "total_trades": len(trades),
        "completed_trades": len(completed),
        "win_rate": round((len(wins) / len(completed)) if completed else 0.0, 3),
        "loss_rate": round((len(losses) / len(completed)) if completed else 0.0, 3),
        "pending_trades": len(pending),
        "avg_conviction_wins": round(avg([float(t.get("conviction", 0.0)) for t in wins], 0.0), 3),
        "avg_conviction_losses": round(avg([float(t.get("conviction", 0.0)) for t in losses], 0.0), 3),
        "best_symbol": best_symbol,
        "learning_insights": insights,
        "recommendation": recommendation,
    }
