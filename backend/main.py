"""FastAPI entrypoint for the multi-agent Indian market trading system."""

from __future__ import annotations

import asyncio
import csv
import hashlib
import json
import logging
import os
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from agents import agent1_data_validator, agent2_quant, agent3_options, agent4_sentiment, agent5_judge
from utils.data_fetcher import (
    current_market_session,
    fetch_live_price,
    fetch_market_overview,
    normalize_symbol,
    resolve_yfinance_symbol,
)
from utils.learning_engine import analyze_performance, load_learning_insights, performance_summary

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
load_dotenv(BASE_DIR.parent / ".env")

DEFAULT_CAPITAL = float(os.getenv("DEFAULT_CAPITAL", "100000") or 100000)
RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", "0.01") or 0.01)
LOG_FILE = BASE_DIR / "trading_agent.log"
CSV_LOG_FILE = BASE_DIR / "analysis_history.csv"
PENDING_TRADES_FILE = BASE_DIR / "pending_trades.json"
LEARNING_INSIGHTS_FILE = BASE_DIR / "learning_insights.json"
CSV_COLUMNS = [
    "timestamp",
    "trade_id",
    "symbol",
    "signal",
    "conviction",
    "entry",
    "sl",
    "t1",
    "t2",
    "t3",
    "outcome",
    "detected_patterns",
    "signal_conflict.has_conflict",
    "signal_conflict.bullish_signals",
    "signal_conflict.bearish_signals",
    "setup_quality",
    "trade_recommendation",
    "learning_note",
    "performance.win_rate",
    "rsi",
    "adx",
    "pcr",
    "vix",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"), logging.StreamHandler()],
)
logger = logging.getLogger("market-agent")
_OUTCOME_TASK: Optional[asyncio.Task] = None

app = FastAPI(title="Market Agent API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    """Analyze endpoint request model."""

    symbol: str
    capital: float = DEFAULT_CAPITAL
    news_headlines: List[str] = Field(default_factory=list)


class QuickScanRequest(BaseModel):
    """Quick scan endpoint request model."""

    symbols: List[str]
    capital: float = DEFAULT_CAPITAL


class VisionRequest(BaseModel):
    """Vision endpoint request model."""

    image: str
    symbol: str = "UNKNOWN"


@app.on_event("startup")
async def startup_event() -> None:
    """Start background outcome evaluator loop."""

    global _OUTCOME_TASK
    if _OUTCOME_TASK is None or _OUTCOME_TASK.done():
        _OUTCOME_TASK = asyncio.create_task(_outcome_evaluator_loop())


@app.on_event("shutdown")
async def shutdown_event() -> None:
    """Stop background outcome evaluator loop."""

    global _OUTCOME_TASK
    if _OUTCOME_TASK and not _OUTCOME_TASK.done():
        _OUTCOME_TASK.cancel()
        try:
            await _OUTCOME_TASK
        except asyncio.CancelledError:
            pass
    _OUTCOME_TASK = None


@app.get("/health")
async def health() -> Dict[str, Any]:
    """Health endpoint."""

    return {"ok": True, "service": "market-agent", "risk_per_trade": RISK_PER_TRADE}


@app.get("/performance/summary")
async def performance_summary_endpoint() -> Dict[str, Any]:
    """Return trading performance and learning snapshot."""

    trades = _load_pending_trades()
    insights = load_learning_insights(LEARNING_INSIGHTS_FILE)
    return performance_summary(trades, insights)


@app.post("/analyze")
async def analyze(request: AnalyzeRequest) -> Dict[str, Any]:
    """Run complete 5-agent analysis pipeline for one symbol."""

    try:
        result = await _run_pipeline(symbol=request.symbol, capital=request.capital, headlines=request.news_headlines, lightweight=False)
        trade_id = _register_pending_trade(result)
        if trade_id:
            result["trade_id"] = trade_id
        _append_csv_log(result)
        return result
    except Exception as exc:
        logger.exception("analyze_failed")
        return _degraded_response(symbol=request.symbol, reason=f"analyze_exception: {exc}")


@app.post("/quick-scan")
async def quick_scan(request: QuickScanRequest) -> Dict[str, Any]:
    """Run lightweight scan over multiple symbols and rank by conviction."""

    try:
        if not request.symbols:
            return {"ok": True, "results": [], "ranked": []}

        semaphore = asyncio.Semaphore(4)

        async def _scan_one(sym: str) -> Dict[str, Any]:
            async with semaphore:
                return await _run_pipeline(symbol=sym, capital=request.capital, headlines=[], lightweight=True)

        results = await asyncio.gather(*[_scan_one(sym) for sym in request.symbols], return_exceptions=True)
        clean: List[Dict[str, Any]] = []
        for sym, item in zip(request.symbols, results):
            if isinstance(item, Exception):
                clean.append(_degraded_response(symbol=sym, reason=f"quick_scan_error: {item}"))
            else:
                clean.append(item)

        ranked = sorted(
            [
                {
                    "symbol": r.get("symbol", ""),
                    "signal": r.get("signal", "AVOID"),
                    "conviction": float(r.get("conviction", 0.0)),
                    "regime": r.get("regime", "UNKNOWN"),
                    "degraded": bool(r.get("degraded", False)),
                }
                for r in clean
            ],
            key=lambda x: x["conviction"],
            reverse=True,
        )
        return {"ok": True, "results": clean, "ranked": ranked}
    except Exception as exc:
        logger.exception("quick_scan_failed")
        return {"ok": False, "results": [], "ranked": [], "error": f"quick_scan_exception: {exc}"}


@app.get("/market-overview")
async def market_overview() -> Dict[str, Any]:
    """Return broad market snapshot with sentiment label."""

    try:
        overview = fetch_market_overview()
        nifty = float(overview["nifty50"]["last_price"] or 0.0)
        nifty_prev = float(overview["nifty50"]["previous_close"] or 0.0)
        vix = float(overview["india_vix"]["last_price"] or 0.0)
        pct = ((nifty - nifty_prev) / nifty_prev * 100.0) if nifty_prev > 0 else 0.0

        sentiment = "NEUTRAL"
        if pct > 0.6 and vix < 18:
            sentiment = "BULL"
        elif pct < -0.6 or vix > 24:
            sentiment = "BEAR"

        return {
            "ok": True,
            "market_sentiment": sentiment,
            "nifty50": overview["nifty50"],
            "banknifty": overview["banknifty"],
            "india_vix": overview["india_vix"],
            "market_session": current_market_session().status,
        }
    except Exception as exc:
        logger.exception("market_overview_failed")
        return {"ok": False, "market_sentiment": "NEUTRAL", "error": f"market_overview_exception: {exc}"}


@app.get("/export/analysis-history")
async def export_analysis_history(limit: int = 2000, log_tail: int = 300) -> Dict[str, Any]:
    """Export backend analysis CSV rows plus recent runtime log lines for transparency."""

    try:
        safe_limit = max(1, min(int(limit or 2000), 10000))
        safe_log_tail = max(0, min(int(log_tail or 300), 2000))

        rows: List[Dict[str, Any]] = []
        if CSV_LOG_FILE.exists():
            with CSV_LOG_FILE.open("r", encoding="utf-8", newline="") as fh:
                reader = csv.DictReader(fh)
                rows = [dict(item) for item in reader]
        if safe_limit and len(rows) > safe_limit:
            rows = rows[-safe_limit:]

        recent_log_lines: List[str] = []
        if safe_log_tail > 0 and LOG_FILE.exists():
            with LOG_FILE.open("r", encoding="utf-8", errors="ignore") as fh:
                recent_log_lines = [line.rstrip("\n") for line in fh.readlines()[-safe_log_tail:]]

        return {
            "ok": True,
            "rows": rows,
            "count": len(rows),
            "recent_log_lines": recent_log_lines,
            "log_count": len(recent_log_lines),
        }
    except Exception as exc:
        logger.exception("export_analysis_history_failed")
        return {"ok": False, "rows": [], "count": 0, "recent_log_lines": [], "log_count": 0, "error": f"export_exception: {exc}"}


@app.post("/vision/analyze")
async def vision_analyze(request: VisionRequest) -> Dict[str, Any]:
    """Analyze chart image via Groq vision model with deterministic fallback."""

    try:
        out = await agent4_sentiment.call_groq_vision(request.image)
        if out:
            return {
                "ok": True,
                "symbol": normalize_symbol(request.symbol),
                "patterns": out.get("patterns", []),
                "trend_direction": out.get("trend_direction", "UNKNOWN"),
                "key_levels": out.get("key_levels", []),
                "raw": out,
            }
        return _vision_fallback(request.symbol, request.image, reason="groq_vision_unavailable")
    except Exception as exc:
        logger.exception("vision_failed")
        return _vision_fallback(request.symbol, request.image, reason=f"vision_exception: {exc}")


@app.websocket("/ws/live/{symbol}")
async def ws_live(websocket: WebSocket, symbol: str) -> None:
    """Stream lightweight live updates every 3 seconds."""

    await websocket.accept()
    norm = normalize_symbol(symbol)
    yf_symbol = resolve_yfinance_symbol(norm)
    try:
        while True:
            snap = fetch_live_price(yf_symbol)
            await websocket.send_json(
                {
                    "ok": True,
                    "symbol": norm,
                    "price": snap,
                    "market_session": current_market_session().status,
                }
            )
            await asyncio.sleep(3)
    except WebSocketDisconnect:
        logger.info("ws_disconnect %s", norm)
    except Exception as exc:
        logger.exception("ws_live_failed")
        try:
            await websocket.send_json({"ok": False, "symbol": norm, "error": f"ws_exception: {exc}"})
        except Exception:
            pass
        await websocket.close()


async def _run_pipeline(symbol: str, capital: float, headlines: List[str], lightweight: bool) -> Dict[str, Any]:
    """Run all agents in sequence and merge results into one response."""

    norm_symbol = normalize_symbol(symbol)
    cap = float(capital if capital > 0 else DEFAULT_CAPITAL)

    a1 = agent1_data_validator.run(norm_symbol)
    ctx = a1["data_context"]

    a2 = agent2_quant.run(ctx)
    a3 = agent3_options.run(ctx.symbol, ctx.current_price) if not lightweight else {
        "ok": True,
        "options_score": 50.0,
        "pcr": 1.0,
        "max_pain": ctx.current_price,
        "key_resistance": ctx.current_price * 1.01 if ctx.current_price else 0.0,
        "key_support": ctx.current_price * 0.99 if ctx.current_price else 0.0,
        "signal": "NEUTRAL",
        "iv_rank": 50.0,
        "iv_state": "UNKNOWN",
        "data_unavailable": True,
        "error": "",
        "raw_summary": {},
        "agent_status": "AMBER",
    }
    a4 = await agent4_sentiment.run(ctx.symbol, provided_headlines=headlines, lightweight=lightweight)
    a5 = await agent5_judge.run(data_context=ctx, quant=a2, options=a3, sentiment=a4, capital=cap)

    degraded = not all([a1.get("ok", False), a2.get("ok", False), a4.get("ok", False), a5.get("ok", False)])
    response = {
        "ok": True,
        "degraded": degraded,
        "symbol": ctx.symbol,
        "yfinance_symbol": ctx.yfinance_symbol,
        "current_price": round(float(ctx.current_price), 4),
        "capital": cap,
        "signal": a5["signal"],
        "setup_signal": a5.get("setup_signal", a5["signal"]),
        "direction": a5["direction"],
        "conviction": a5["conviction"],
        "trade_levels": a5["trade_levels"],
        "trade_levels_basis_signal": a5.get("trade_levels_basis_signal", a5["signal"]),
        "position_size": a5["position_size"],
        "risk_reward_ratio": a5["risk_reward_ratio"],
        "regime": a2.get("regime", "NEUTRAL"),
        "mtf_score": a2.get("mtf_score", 10),
        "quant_score": a2.get("quant_score", 50.0),
        "options_score": a3.get("options_score", 50.0),
        "sentiment_score": a4.get("sentiment_score", 50.0),
        "news_freshness": a4.get("news_freshness", "MIXED"),
        "sentiment_confidence": a4.get("sentiment_confidence", 50.0),
        "sentiment_flags": a4.get("sentiment_flags", []),
        "data_quality_score": ctx.data_quality_score,
        "kill_switch_status": a5.get("kill_switch_status", {}),
        "no_trade_reasons": a5.get("no_trade_reasons", []),
        "groq_rationale": a5.get("groq_rationale", []),
        "conviction_breakdown": a5.get("conviction_breakdown", {}),
        "signal_conflict": a5.get("signal_conflict", {}),
        "setup_quality": a5.get("setup_quality", "D_GRADE"),
        "trade_recommendation": a5.get("trade_recommendation", "Poor setup - avoid completely"),
        "learning_note": a5.get("learning_note", ""),
        "market_context": a5.get("market_context", {}),
        "agent_status": {
            "agent1_data_validator": "GREEN" if a1.get("ok") else "RED",
            "agent2_quant": "GREEN" if a2.get("ok") else "RED",
            "agent3_options": a3.get("agent_status", "AMBER"),
            "agent4_sentiment": a4.get("agent_status", "AMBER"),
            "agent5_judge": a5.get("agent_status", "AMBER"),
        },
        "news_feed": a4.get("news_items", []),
        "finbert_results": a4.get("finbert_results", []),
        "sentiment_groq_analysis": a4.get("groq_analysis", {}),
        "social_score": a4.get("social_score", 50.0),
        "detected_patterns": a2.get("patterns", []),
        "key_levels": {
            "quant_levels": a2.get("key_levels", {}),
            "options_support": a3.get("key_support", 0.0),
            "options_resistance": a3.get("key_resistance", 0.0),
            "max_pain": a3.get("max_pain", 0.0),
        },
        "options_data": {
            "pcr": a3.get("pcr", 1.0),
            "signal": a3.get("signal", "NEUTRAL"),
            "iv_rank": a3.get("iv_rank", 50.0),
            "iv_state": a3.get("iv_state", "UNKNOWN"),
            "raw_summary": a3.get("raw_summary", {}),
            "data_unavailable": a3.get("data_unavailable", True),
        },
        "market_research": {
            "nifty_correlation": ctx.nifty_correlation,
            "market_session": ctx.market_session,
            "session_warning": ctx.session_warning,
            "gap_info": ctx.gap_info,
            "quant_summary": a2.get("indicators", {}).get("15m", {}),
            "sentiment_source": a4.get("source", "fallback"),
            "news_freshness": a4.get("news_freshness", "MIXED"),
            "sentiment_confidence": a4.get("sentiment_confidence", 50.0),
        },
        "transparency": {
            "agent1": a1.get("agent_output", {}),
            "agent2_error": a2.get("error", ""),
            "agent3_error": a3.get("error", ""),
            "agent4_error": a4.get("error", ""),
            "agent5_error": a5.get("error", ""),
        },
        "warnings": [w for w in [ctx.session_warning] if w],
        "errors": [e for e in [a2.get("error"), a3.get("error"), a4.get("error"), a5.get("error")] if e],
    }
    perf = performance_summary(_load_pending_trades(), load_learning_insights(LEARNING_INSIGHTS_FILE))
    response["performance"] = {"win_rate": perf.get("win_rate", 0.0)}
    return response


def _load_pending_trades() -> List[Dict[str, Any]]:
    """Load pending/completed trade list from local JSON file."""

    try:
        if not PENDING_TRADES_FILE.exists():
            PENDING_TRADES_FILE.write_text("[]", encoding="utf-8")
            return []
        if PENDING_TRADES_FILE.exists():
            data = PENDING_TRADES_FILE.read_text(encoding="utf-8").strip()
            if data:
                parsed = json.loads(data)
                if isinstance(parsed, list):
                    return parsed
    except Exception:
        logger.exception("load_pending_trades_failed")
    return []


def _save_pending_trades(trades: List[Dict[str, Any]]) -> None:
    """Persist pending/completed trades atomically."""

    try:
        PENDING_TRADES_FILE.write_text(json.dumps(trades, indent=2), encoding="utf-8")
    except Exception:
        logger.exception("save_pending_trades_failed")


def _register_pending_trade(result: Dict[str, Any]) -> str | None:
    """Register BUY/SELL trade into pending tracker for auto outcome evaluation."""

    signal = str(result.get("signal", "AVOID")).upper()
    if signal not in {"BUY", "SELL", "STRONG_BUY", "STRONG_SELL"}:
        return None

    try:
        trade_id = str(uuid.uuid4())
        tl = result.get("trade_levels", {})
        indicators_15m = result.get("market_research", {}).get("quant_summary", {})
        pending = _load_pending_trades()

        trade = {
            "trade_id": trade_id,
            "timestamp": datetime.now().isoformat(),
            "symbol": str(result.get("symbol", "")),
            "signal": "BUY" if "BUY" in signal else "SELL",
            "entry": float(tl.get("entry", 0.0) or 0.0),
            "sl": float(tl.get("sl", 0.0) or 0.0),
            "t1": float(tl.get("t1", 0.0) or 0.0),
            "t2": float(tl.get("t2", 0.0) or 0.0),
            "t3": float(tl.get("t3", 0.0) or 0.0),
            "conviction": float(result.get("conviction", 0.0) or 0.0),
            "rsi": float(indicators_15m.get("rsi_14", 50.0) or 50.0),
            "macd": float(indicators_15m.get("macd", 0.0) or 0.0),
            "adx": float(indicators_15m.get("adx_14", 0.0) or 0.0),
            "pcr": float(result.get("options_data", {}).get("pcr", 1.0) or 1.0),
            "vix": float(result.get("market_context", {}).get("india_vix", 0.0) or 0.0),
            "outcome": "PENDING",
        }
        pending.append(trade)
        _save_pending_trades(pending)
        return trade_id
    except Exception:
        logger.exception("register_pending_trade_failed")
        return None


def _append_csv_log(result: Dict[str, Any], override_outcome: str = "") -> None:
    """Append one analysis/trade row in extended transparent CSV format."""

    try:
        _ensure_csv_schema()
        exists = CSV_LOG_FILE.exists()
        with CSV_LOG_FILE.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
            if not exists:
                writer.writeheader()

            tl = result.get("trade_levels", {})
            patterns = result.get("detected_patterns", [])
            pattern_names = "|".join([str(p.get("name", "")) for p in patterns if isinstance(p, dict) and p.get("name")])
            conflict = result.get("signal_conflict", {})
            quant_summary = result.get("market_research", {}).get("quant_summary", {})
            row = {
                "timestamp": current_market_session().current_time_ist,
                "trade_id": str(result.get("trade_id", "")),
                "symbol": result.get("symbol", ""),
                "signal": result.get("signal", "AVOID"),
                "conviction": result.get("conviction", 0.0),
                "entry": tl.get("entry", 0.0),
                "sl": tl.get("sl", 0.0),
                "t1": tl.get("t1", 0.0),
                "t2": tl.get("t2", 0.0),
                "t3": tl.get("t3", 0.0),
                "outcome": override_outcome or result.get("outcome", ""),
                "detected_patterns": pattern_names,
                "signal_conflict.has_conflict": conflict.get("has_conflict", False),
                "signal_conflict.bullish_signals": conflict.get("bullish_signals", 0),
                "signal_conflict.bearish_signals": conflict.get("bearish_signals", 0),
                "setup_quality": result.get("setup_quality", "D_GRADE"),
                "trade_recommendation": result.get("trade_recommendation", ""),
                "learning_note": result.get("learning_note", ""),
                "performance.win_rate": result.get("performance", {}).get("win_rate", 0.0),
                "rsi": quant_summary.get("rsi_14", 0.0),
                "adx": quant_summary.get("adx_14", 0.0),
                "pcr": result.get("options_data", {}).get("pcr", 1.0),
                "vix": result.get("market_context", {}).get("india_vix", 0.0),
            }
            writer.writerow(row)
    except Exception:
        logger.exception("csv_log_failed")


def _ensure_csv_schema() -> None:
    """Upgrade legacy CSV header to the current transparent schema."""

    try:
        if not CSV_LOG_FILE.exists():
            return
        with CSV_LOG_FILE.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            existing_fields = reader.fieldnames or []
            if existing_fields == CSV_COLUMNS:
                return
            old_rows = [dict(r) for r in reader]

        with CSV_LOG_FILE.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            for old in old_rows:
                writer.writerow(
                    {
                        "timestamp": old.get("timestamp", ""),
                        "trade_id": old.get("trade_id", ""),
                        "symbol": old.get("symbol", ""),
                        "signal": old.get("signal", ""),
                        "conviction": old.get("conviction", ""),
                        "entry": old.get("entry", ""),
                        "sl": old.get("sl", ""),
                        "t1": old.get("t1", ""),
                        "t2": old.get("t2", ""),
                        "t3": old.get("t3", ""),
                        "outcome": old.get("outcome", ""),
                        "detected_patterns": old.get("detected_patterns", ""),
                        "signal_conflict.has_conflict": old.get("signal_conflict.has_conflict", ""),
                        "signal_conflict.bullish_signals": old.get("signal_conflict.bullish_signals", ""),
                        "signal_conflict.bearish_signals": old.get("signal_conflict.bearish_signals", ""),
                        "setup_quality": old.get("setup_quality", ""),
                        "trade_recommendation": old.get("trade_recommendation", ""),
                        "learning_note": old.get("learning_note", ""),
                        "performance.win_rate": old.get("performance.win_rate", ""),
                        "rsi": old.get("rsi", ""),
                        "adx": old.get("adx", ""),
                        "pcr": old.get("pcr", ""),
                        "vix": old.get("vix", ""),
                    }
                )
    except Exception:
        logger.exception("ensure_csv_schema_failed")


def _append_trade_outcome_row(trade: Dict[str, Any]) -> None:
    """Append outcome-update row to CSV to keep audit trail complete."""

    fake_result = {
        "trade_id": trade.get("trade_id", ""),
        "symbol": trade.get("symbol", ""),
        "signal": trade.get("signal", "AVOID"),
        "conviction": trade.get("conviction", 0.0),
        "trade_levels": {
            "entry": trade.get("entry", 0.0),
            "sl": trade.get("sl", 0.0),
            "t1": trade.get("t1", 0.0),
            "t2": trade.get("t2", 0.0),
            "t3": trade.get("t3", 0.0),
        },
        "detected_patterns": [],
        "signal_conflict": {"has_conflict": False, "bullish_signals": 0, "bearish_signals": 0},
        "setup_quality": "",
        "trade_recommendation": "",
        "learning_note": "",
        "performance": {"win_rate": 0.0},
        "market_research": {"quant_summary": {"rsi_14": trade.get("rsi", 0.0), "adx_14": trade.get("adx", 0.0)}},
        "options_data": {"pcr": trade.get("pcr", 1.0)},
        "market_context": {"india_vix": trade.get("vix", 0.0)},
        "outcome": trade.get("outcome", ""),
    }
    _append_csv_log(fake_result, override_outcome=str(trade.get("outcome", "")))


def _evaluate_trade_outcome(trade: Dict[str, Any], current_price: float) -> str:
    """Evaluate current outcome for one pending trade."""

    signal = str(trade.get("signal", "")).upper()
    t1 = float(trade.get("t1", 0.0) or 0.0)
    t2 = float(trade.get("t2", 0.0) or 0.0)
    t3 = float(trade.get("t3", 0.0) or 0.0)
    sl = float(trade.get("sl", 0.0) or 0.0)

    if signal == "BUY":
        if current_price >= t3 > 0:
            return "WIN_T3"
        if current_price >= t2 > 0:
            return "WIN_T2"
        if current_price >= t1 > 0:
            return "WIN_T1"
        if current_price <= sl and sl > 0:
            return "LOSS"
    if signal == "SELL":
        if current_price <= t3 and t3 > 0:
            return "WIN_T3"
        if current_price <= t2 and t2 > 0:
            return "WIN_T2"
        if current_price <= t1 and t1 > 0:
            return "WIN_T1"
        if current_price >= sl and sl > 0:
            return "LOSS"
    return "PENDING"


async def _update_pending_outcomes_once() -> None:
    """Single pass evaluation for pending trades."""

    trades = _load_pending_trades()
    if not trades:
        return

    changed = False
    now = datetime.now()
    for trade in trades:
        if str(trade.get("outcome", "PENDING")) != "PENDING":
            continue

        symbol = normalize_symbol(str(trade.get("symbol", "")))
        yf_symbol = resolve_yfinance_symbol(symbol)
        snap = fetch_live_price(yf_symbol)
        price = float(snap.get("last_price") or 0.0)

        new_outcome = _evaluate_trade_outcome(trade, price)
        if new_outcome == "PENDING":
            try:
                created_at = datetime.fromisoformat(str(trade.get("timestamp")))
            except Exception:
                created_at = now
            if now - created_at > timedelta(hours=6):
                new_outcome = "EXPIRED"

        if new_outcome != "PENDING":
            trade["outcome"] = new_outcome
            trade["updated_at"] = now.isoformat()
            trade["exit_price"] = price
            _append_trade_outcome_row(trade)
            changed = True

    if changed:
        _save_pending_trades(trades)
        completed = [t for t in trades if str(t.get("outcome", "PENDING")) != "PENDING"]
        if len(completed) >= 50 and (len(completed) % 50 == 0):
            insights = analyze_performance(completed, LEARNING_INSIGHTS_FILE)
            if insights:
                logger.info("Learning updated - win rate %.2f%%", float(insights.get("win_rate", 0.0)) * 100.0)


async def _outcome_evaluator_loop() -> None:
    """Run outcome evaluator every 30 minutes in background."""

    while True:
        try:
            await _update_pending_outcomes_once()
        except Exception:
            logger.exception("outcome_evaluator_failed")
        await asyncio.sleep(1800)


def _degraded_response(symbol: str, reason: str) -> Dict[str, Any]:
    """Unified degraded payload to avoid frontend crashes."""

    norm = normalize_symbol(symbol)
    return {
        "ok": True,
        "degraded": True,
        "symbol": norm,
        "signal": "AVOID",
        "setup_signal": "AVOID",
        "direction": "NEUTRAL",
        "conviction": 40.0,
        "trade_levels": {"entry": 0.0, "sl": 0.0, "t1": 0.0, "t2": 0.0, "t3": 0.0, "risk_reward_ratio": 0.0},
        "trade_levels_basis_signal": "AVOID",
        "position_size": {"quantity": 0, "risk_budget": 0.0, "capital_limit": 0.0, "capital_required": 0.0, "effective_risk": 0.0},
        "risk_reward_ratio": 0.0,
        "regime": "UNKNOWN",
        "mtf_score": 10,
        "quant_score": 50.0,
        "options_score": 50.0,
        "sentiment_score": 50.0,
        "news_freshness": "STALE",
        "sentiment_confidence": 20.0,
        "sentiment_flags": ["STALE_NEWS_WARNING"],
        "data_quality_score": 0.0,
        "kill_switch_status": {},
        "no_trade_reasons": [reason],
        "groq_rationale": [
            "System entered degraded mode.",
            "Data unavailable for reliable conviction.",
            "Avoid new positions until service restores.",
        ],
        "signal_conflict": {"has_conflict": False, "bullish_signals": 0, "bearish_signals": 0, "penalty": 0, "reason": "degraded"},
        "setup_quality": "D_GRADE",
        "trade_recommendation": "Poor setup - avoid completely",
        "learning_note": "",
        "conviction_breakdown": {"quant": 50, "mtf": 50, "options": 50, "sentiment": 50, "data_quality": 0},
        "agent_status": {
            "agent1_data_validator": "RED",
            "agent2_quant": "RED",
            "agent3_options": "AMBER",
            "agent4_sentiment": "AMBER",
            "agent5_judge": "RED",
        },
        "news_feed": [],
        "finbert_results": [],
        "sentiment_groq_analysis": {},
        "social_score": 50.0,
        "detected_patterns": [],
        "key_levels": {"quant_levels": {}, "options_support": 0.0, "options_resistance": 0.0, "max_pain": 0.0},
        "options_data": {"pcr": 1.0, "signal": "NEUTRAL", "iv_rank": 50.0, "iv_state": "UNKNOWN", "raw_summary": {}, "data_unavailable": True},
        "market_research": {"nifty_correlation": 0.0, "market_session": current_market_session().status, "session_warning": "", "quant_summary": {}, "sentiment_source": "fallback"},
        "market_context": {"regime": "UNKNOWN", "adx": 0.0, "india_vix": 0.0, "session": current_market_session().status, "session_warning": "", "gap_percent": 0.0, "gap_type": "UNKNOWN", "gap_direction": "FLAT"},
        "transparency": {"error": reason},
        "warnings": [],
        "errors": [reason],
        "performance": {"win_rate": 0.0},
    }


def _vision_fallback(symbol: str, image: str, reason: str) -> Dict[str, Any]:
    """Deterministic fallback for vision endpoint."""

    digest = hashlib.sha256(image.encode("utf-8")).hexdigest()
    bucket = int(digest[:2], 16) % 5
    patterns = [
        ["Double Bottom", "Volume Expansion"],
        ["Bull Flag", "Breakout Retest"],
        ["Bear Flag", "Lower High Rejection"],
        ["Range Compression", "Potential Expansion"],
        ["Triangle Build-up", "Momentum Watch"],
    ][bucket]
    return {
        "ok": True,
        "degraded": True,
        "symbol": normalize_symbol(symbol),
        "patterns": patterns,
        "trend_direction": "UNKNOWN",
        "key_levels": [],
        "raw": {},
        "error": reason,
    }
