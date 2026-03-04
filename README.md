# Aegis Quant Co-Pilot

Professional AI Trading Co-Pilot for Indian markets (NSE/BSE), built as:
- Chrome Extension (Manifest V3 Sidepanel)
- Python FastAPI multi-agent backend
- 100% free data stack (no paid market API required)

This project is designed to reduce low-quality trades, improve decision clarity, and give trader-facing transparency before every entry.

## Why Traders Use This

Instead of taking trades on raw indicators, this system combines:
- Multi-timeframe technical confluence
- Institutional-style options context (PCR, max pain, OI levels)
- News and sentiment analysis (NSE announcements + FinBERT + Groq)
- Rule-based risk controls and kill-switches

Result:
- Better filtering of weak setups
- Cleaner entry/SL/targets
- Clear reasons for both trade and no-trade scenarios

## Core Value for Traders

| Trader Problem | What Aegis Does |
|---|---|
| Emotional entries | Gives rule-based signal with conviction score and kill-switch logic |
| Confusion in levels | Auto-generates entry, SL, T1, T2, T3, R:R, quantity |
| News overload | Filters and summarizes relevant news + sentiment impact |
| Hidden risk | Shows no-trade reasons, volatility warnings, sentiment gates |
| Trust issues in black-box AI | Provides transparency panel, agent status LEDs, downloadable CSV evidence |

## Key Features

### 1) Multi-Agent Decision Engine
- Agent 1: Data Validator
- Agent 2: Quant + pattern detection
- Agent 3: Options Analyzer (direct NSE option-chain API flow with fallback)
- Agent 4: Sentiment Analyzer (NSE corporate announcements, Economic Times, Google News, FinBERT, Groq)
- Agent 5: Judge (conviction score + final signal + risk plan)

### 2) Advanced Quant Layer
- EMA, MACD, ADX, RSI, StochRSI, ATR, BB, Keltner, VWAP, OBV
- Multi-timeframe confluence scoring (5m, 15m, 1h, 1d)
- Regime detection: trending, volatile, sideways
- Candlestick + chart patterns on 15m with 1h confirmation:
  - Bullish/Bearish Engulfing, Doji, Hammer, Shooting Star
  - Morning Star, Evening Star
  - Double Bottom/Top, Bull/Bear Flag, HH-HL, LH-LL

### 3) Smart Risk Layer
- ATR-based trade levels (entry/SL/T1/T2/T3)
- Risk-budget based quantity suggestion
- Kill-switches for unstable conditions
- No-trade reasons shown directly in UI

### 4) Professional Sidepanel UI
- Signal banner: BUY/SELL/AVOID
- Conviction meter (0-100)
- Trade card with risk-reward and quantity
- Conviction breakdown bars
- Kill-switch status and agent LEDs
- Pattern list, key levels, options summary
- News feed + rationale from Judge agent
- Screenshot analysis trigger
- Full transparency CSV download

### 5) Live + Scan Workflows
- Single symbol deep analyze
- Quick-scan ranking for multiple symbols
- Market overview endpoint
- WebSocket live price stream

## Supported Platforms

- TradingView
- Zerodha
- Upstox

Content script detects chart symbol, sidepanel requests full analysis from backend, and renders actionable output.

## High-Level Workflow

1. Trader opens chart and sidepanel.
2. Extension detects symbol from URL/title.
3. Backend fetches and validates multi-timeframe market data.
4. Quant, options, sentiment, and judge agents run in sequence.
5. UI displays signal, conviction, risk levels, reasons, and warnings.
6. Trader can export full transparency CSV for audit and journaling.

## Tech Stack

- Backend: FastAPI
- Market Data: yfinance
- Options Data: direct NSE web API session flow
- Quant: `pandas-ta-classic` (import as `pandas_ta`)
- Sentiment: NSE announcements + Economic Times + Google News + local FinBERT + Groq refinement
- LLM: Groq (`llama-3.3-70b-versatile`)
- Frontend: Chrome Extension Manifest V3 sidepanel

## Project Structure

- `backend/main.py`
- `backend/agents/agent1_data_validator.py`
- `backend/agents/agent2_quant.py`
- `backend/agents/agent3_options.py`
- `backend/agents/agent4_sentiment.py`
- `backend/agents/agent5_judge.py`
- `backend/utils/data_fetcher.py`
- `backend/utils/indicators.py`
- `backend/utils/options_analyzer.py`
- `backend/utils/trade_levels.py`
- `backend/requirements.txt`
- `backend/trading_agent.log`
- `backend/analysis_history.csv`
- `extension/manifest.json`
- `extension/background.js`
- `extension/content.js`
- `extension/sidepanel.html`
- `extension/sidepanel.js`
- `extension/sidepanel.css`
- `.env`

## Environment Variables

Use root `.env` or `backend/.env`:

```env
GROQ_API_KEY=your_key_here
GROQ_MODEL=llama-3.3-70b-versatile
GROQ_VISION_MODEL=meta-llama/llama-4-scout-17b-16e-instruct
DEFAULT_CAPITAL=100000
RISK_PER_TRADE=0.01
REDDIT_CLIENT_ID=
REDDIT_CLIENT_SECRET=
REDDIT_USER_AGENT=market-agent/1.0
HF_TOKEN=
```

`HF_TOKEN` optional hai, lekin FinBERT model download rate-limit issues reduce kar deta hai.

## Backend Setup

```powershell
cd "C:\Projects\Market Agent\backend"
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

## Chrome Extension Setup

1. Open `chrome://extensions`.
2. Enable `Developer mode`.
3. Click `Load unpacked`.
4. Select `C:\Projects\Market Agent\extension`.
5. Open TradingView/Zerodha/Upstox chart.
6. Open extension sidepanel.
7. Click `Analyze`.

## API Endpoints

- `GET /health`
- `POST /analyze`
- `POST /quick-scan`
- `GET /market-overview`
- `POST /vision/analyze`
- `GET /export/analysis-history`
- `WS /ws/live/{symbol}`

## What Trader Can Do Faster With This

- Pre-trade checklist auto-validation
- Quick symbol ranking before market opens
- Live signal monitoring with change tracking
- Structured risk planning in seconds
- News + pattern + options context in one screen
- Audit trail export for performance review

## Transparency and Logging

- Runtime log: `backend/trading_agent.log`
- Analysis history: `backend/analysis_history.csv`
- Sidepanel CSV export includes:
  - latest analysis payload
  - signal changes
  - live ticks
  - errors/events
  - backend log tail

## Important Notes

- This is a decision-support system, not guaranteed-profit automation.
- Market data from free providers can sometimes be delayed or temporarily unavailable.
- Always use personal risk limits and broker-side risk controls.
