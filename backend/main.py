"""FastAPI entrypoint for the multi-agent Indian market trading system."""

from __future__ import annotations

import asyncio
import csv
import hashlib
import logging
import os
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

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
load_dotenv(BASE_DIR.parent / ".env")

DEFAULT_CAPITAL = float(os.getenv("DEFAULT_CAPITAL", "100000") or 100000)
RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", "0.01") or 0.01)
LOG_FILE = BASE_DIR / "trading_agent.log"
CSV_LOG_FILE = BASE_DIR / "analysis_history.csv"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"), logging.StreamHandler()],
)
logger = logging.getLogger("market-agent")

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


@app.get("/health")
async def health() -> Dict[str, Any]:
    """Health endpoint."""

    return {"ok": True, "service": "market-agent", "risk_per_trade": RISK_PER_TRADE}


@app.post("/analyze")
async def analyze(request: AnalyzeRequest) -> Dict[str, Any]:
    """Run complete 5-agent analysis pipeline for one symbol."""

    try:
        result = await _run_pipeline(symbol=request.symbol, capital=request.capital, headlines=request.news_headlines, lightweight=False)
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
        "data_quality_score": ctx.data_quality_score,
        "kill_switch_status": a5.get("kill_switch_status", {}),
        "no_trade_reasons": a5.get("no_trade_reasons", []),
        "groq_rationale": a5.get("groq_rationale", []),
        "conviction_breakdown": a5.get("conviction_breakdown", {}),
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
            "quant_summary": a2.get("indicators", {}).get("15m", {}),
            "sentiment_source": a4.get("source", "fallback"),
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
    return response


def _append_csv_log(result: Dict[str, Any]) -> None:
    """Append analysis row for future backtesting dataset."""

    try:
        exists = CSV_LOG_FILE.exists()
        with CSV_LOG_FILE.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            if not exists:
                writer.writerow(["timestamp", "symbol", "signal", "conviction", "entry", "sl", "t1", "outcome"])
            tl = result.get("trade_levels", {})
            writer.writerow(
                [
                    current_market_session().current_time_ist,
                    result.get("symbol", ""),
                    result.get("signal", "AVOID"),
                    result.get("conviction", 0.0),
                    tl.get("entry", 0.0),
                    tl.get("sl", 0.0),
                    tl.get("t1", 0.0),
                    "",
                ]
            )
    except Exception:
        logger.exception("csv_log_failed")


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
        "data_quality_score": 0.0,
        "kill_switch_status": {},
        "no_trade_reasons": [reason],
        "groq_rationale": [
            "System entered degraded mode.",
            "Data unavailable for reliable conviction.",
            "Avoid new positions until service restores.",
        ],
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
        "transparency": {"error": reason},
        "warnings": [],
        "errors": [reason],
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
