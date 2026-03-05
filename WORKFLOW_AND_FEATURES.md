# Aegis Trading Agent - Complete Workflow and Feature Guide

This document explains exactly how the system works, what updates when, what trader problems it solves, and what it can handle in production use.

## 1) What This System Is

Aegis is a decision-support trading co-pilot for Indian markets (NSE/BSE):

- Chrome Extension (Manifest V3 sidepanel) on chart websites
- FastAPI backend with multi-agent analysis
- Live price stream + scheduled re-analysis
- Transparent output with export options (CSV + styled PDF report)

It is built to reduce low-quality entries and give traders clear, auditable trade plans.

## 2) Where It Works

The extension content script is configured for:

- TradingView (`*://*.tradingview.com/*`)
- Zerodha (`*://*.zerodha.com/*`)
- Upstox (`*://*.upstox.com/*`)

## 3) End-to-End Workflow (From Chart Open to Decision)

1. Trader opens a supported chart page and opens sidepanel.
2. Content script extracts symbol from URL/title and sends it to background worker.
3. Sidepanel asks background for current active symbol.
4. Sidepanel auto-connects live WebSocket (`/ws/live/{symbol}`).
5. Sidepanel runs first full analysis (`/analyze`).
6. Backend runs all analysis agents and returns:
   - Signal (`BUY`/`SELL`/`AVOID`)
   - Conviction score
   - Entry, SL, T1/T2/T3, quantity, R:R
   - Kill switches, no-trade reasons
   - Patterns, levels, options context, sentiment context
7. Sidepanel renders all widgets and starts live-monitoring timers.
8. Ongoing updates happen by timer + price-trigger + WebSocket stream.

## 4) Live Monitoring Update Matrix (Exact)

## 4.1 Continuous WebSocket (every ~3 sec from backend loop)

Updates:

- Live price
- Market session/status tag
- Live tick log used in transparency export

Does not directly run full technical analysis on every tick.

## 4.2 Every 2 Minutes (Technical Refresh)

Trigger: `auto_technical`

Updates:

- Full `/analyze` pipeline is called
- Technicals, patterns, levels, conviction, signal, trade plan refresh
- Kill switches and no-trade reasons refresh

News behavior at 2 min:

- News is not force-refreshed every 2 min
- If 15-min news window not reached, last news/sentiment snapshot is reused in UI for stability and lower noise

## 4.3 Every 15 Minutes (News Refresh)

Trigger: `auto_news`

Updates:

- Full `/analyze` pipeline is called
- News feed and sentiment are force-refreshed
- Signal/conviction and all derived outputs refresh accordingly

## 4.4 Price-Move Trigger

Condition:

- If live price moves significantly from reference price (configured threshold ~0.35%)
- Cooldown applies (~90 seconds) to avoid repeated spam

Action:

- Auto re-analysis trigger (`price_move`)
- Refreshes core analysis sooner than timer schedule

## 4.5 Manual Trigger

When trader clicks `Analyze`:

- Full analysis runs immediately
- News refresh is forced

## 5) Backend Multi-Agent Pipeline

## Agent 1 - Data Validator

- Fetches multi-timeframe OHLC/volume
- Validates data quality, candle sufficiency, integrity checks
- Produces market context used by later agents

## Agent 2 - Quant Engine

- Computes technical indicators and multi-timeframe confluence
- Detects candlestick/chart pattern structures
- Determines regime-style context (trend/volatility/sideways signals)

## Agent 3 - Options Analyzer

- Fetches options-chain context (with safe fallback mode when unavailable)
- Produces PCR, max pain, support/resistance style levels, options score

## Agent 4 - Sentiment Engine

- Collects and filters finance-relevant news
- Runs sentiment scoring (FinBERT + LLM refinement paths)
- Produces sentiment score and rationale context

## Agent 5 - Judge

- Combines agent outputs into final conviction
- Generates final trade intent and direction
- Applies risk logic, kill switches, setup quality
- Produces trader-facing reasons and recommendation

## 6) Trader-Facing UI Features

- Signal Banner: `BUY` / `SELL` / `AVOID`
- Conviction Meter (0-100)
- Trade Card: entry, SL, T1, T2, T3, R:R, quantity
- Conviction Breakdown bars
- Kill-switch status lights
- Agent status LEDs
- Pattern list
- Key levels + options summary
- News list + judge rationale
- Signal change tracker/history
- Live monitoring indicator (blinking green when active)
- Screenshot analysis action (chart image -> vision endpoint)
- CSV transparency export
- Styled PDF summary report with optional appendix expansion

## 7) Export and Transparency Workflow

## CSV Export

Includes flattened and auditable data:

- Latest analysis payload
- Quick scan snapshot
- Vision results
- Signal change log
- Live tick log
- Event/error logs
- Backend analysis-history rows
- Backend runtime log tail

## PDF Export

Human-readable summary:

- Decision snapshot
- Trade plan
- No-trade blockers
- Pattern table
- News snapshot
- Full appendix (hidden by default, expandable)

## 8) API Surface (Operational)

- `GET /health`
- `POST /analyze`
- `POST /quick-scan`
- `GET /market-overview`
- `POST /vision/analyze`
- `GET /export/analysis-history`
- `GET /performance/summary`
- `WS /ws/live/{symbol}`

## 9) What Problems It Solves for Traders

- Reduces impulsive trades with rule-based gating
- Converts scattered signals into one decision view
- Gives ready-to-use levels and position sizing
- Prevents hidden-risk entries via kill switches
- Improves confidence with transparent reasons and logs
- Saves review time with downloadable analysis evidence

## 10) What It Can Handle

- Single symbol deep analysis
- Multi-symbol quick ranking
- Live monitoring with auto-refresh
- Volatility/news-sensitive re-checks
- Degraded/fallback mode when some data source fails
- Continuous logging for later performance review

## 11) Important Practical Limits

- Free market data can be delayed or temporarily inconsistent
- Tick-by-tick parity with paid charting infra is not guaranteed
- AI/sentiment calls add processing latency
- This is a decision-support tool, not guaranteed-profit automation

## 12) Best-Use Operating Playbook

1. Open chart and confirm symbol detected.
2. Run manual `Analyze` once after chart load.
3. Keep sidepanel open for live monitoring.
4. Re-check when signal flips or conviction changes.
5. Use kill-switch/no-trade reasons as hard filters.
6. Export CSV/PDF for journaling and strategy review.

