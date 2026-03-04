const DEFAULT_API_BASE = "http://127.0.0.1:8000";
const tabState = new Map();

chrome.runtime.onInstalled.addListener(async () => {
  await chrome.sidePanel.setPanelBehavior({ openPanelOnActionClick: true });
});

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message?.type === "SYMBOL_DETECTED") {
    if (sender.tab?.id) {
      tabState.set(sender.tab.id, {
        symbol: sanitizeSymbol(message.symbol),
        updatedAt: Date.now()
      });
    }
    sendResponse?.({ ok: true });
    return;
  }

  if (message?.type === "GET_ACTIVE_SYMBOL") {
    handleGetActiveSymbol().then((res) => sendResponse(res));
    return true;
  }

  if (message?.type === "ANALYZE_SYMBOL") {
    handleAnalyze(message.payload)
      .then((res) => sendResponse({ ok: true, result: res }))
      .catch((error) =>
        sendResponse({
          ok: true,
          result: degradedAnalyzeResult(message.payload?.symbol, String(error?.message || error)),
          degraded: true
        })
      );
    return true;
  }

  if (message?.type === "QUICK_SCAN") {
    handleQuickScan(message.payload)
      .then((res) => sendResponse({ ok: true, result: res }))
      .catch((error) =>
        sendResponse({
          ok: true,
          result: { ok: false, results: [], ranked: [], error: String(error?.message || error) }
        })
      );
    return true;
  }

  if (message?.type === "MARKET_OVERVIEW") {
    handleMarketOverview()
      .then((res) => sendResponse({ ok: true, result: res }))
      .catch((error) =>
        sendResponse({
          ok: true,
          result: { ok: false, market_sentiment: "NEUTRAL", error: String(error?.message || error) }
        })
      );
    return true;
  }

  if (message?.type === "SCREENSHOT_ANALYSIS") {
    handleScreenshotAnalysis(message.payload)
      .then((res) => sendResponse({ ok: true, result: res }))
      .catch((error) => sendResponse({ ok: false, error: String(error?.message || error) }));
    return true;
  }
});

async function getApiBase() {
  const stored = await chrome.storage.local.get(["apiBase"]);
  return stored?.apiBase || DEFAULT_API_BASE;
}

async function postJson(path, payload) {
  const base = await getApiBase();
  const response = await fetch(`${base}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload || {})
  });
  if (!response.ok) {
    throw new Error(`${path} failed ${response.status}`);
  }
  return response.json();
}

async function getJson(path) {
  const base = await getApiBase();
  const response = await fetch(`${base}${path}`);
  if (!response.ok) {
    throw new Error(`${path} failed ${response.status}`);
  }
  return response.json();
}

async function handleGetActiveSymbol() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  const fromState = tab?.id ? tabState.get(tab.id)?.symbol : "";
  const fromTab = extractSymbolFromTab(tab);
  const symbol = sanitizeSymbol(fromState || fromTab || "");
  if (tab?.id && symbol) {
    tabState.set(tab.id, { symbol, updatedAt: Date.now() });
  }
  return { ok: true, symbol: symbol || "" };
}

async function handleAnalyze(payload) {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  const symbol = sanitizeSymbol(
    payload?.symbol || (tab?.id ? tabState.get(tab.id)?.symbol : "") || extractSymbolFromTab(tab) || ""
  );
  if (!symbol) {
    return degradedAnalyzeResult("UNKNOWN", "No symbol detected from chart.");
  }
  return postJson("/analyze", {
    symbol,
    capital: Number(payload?.capital || 100000)
  });
}

async function handleQuickScan(payload) {
  const symbols = Array.isArray(payload?.symbols) ? payload.symbols.map((s) => sanitizeSymbol(s)).filter(Boolean) : [];
  return postJson("/quick-scan", {
    symbols,
    capital: Number(payload?.capital || 100000)
  });
}

async function handleMarketOverview() {
  return getJson("/market-overview");
}

async function handleScreenshotAnalysis(payload) {
  const capture = await chrome.tabs.captureVisibleTab(undefined, { format: "png" });
  return postJson("/vision/analyze", {
    image: capture,
    symbol: sanitizeSymbol(payload?.symbol || "")
  });
}

function sanitizeSymbol(input) {
  return String(input || "")
    .toUpperCase()
    .trim()
    .replace(/^NSE[:\-]/, "")
    .replace(/^BSE[:\-]/, "")
    .replace(/[^A-Z0-9_\-.]/g, "")
    .slice(0, 20);
}

function extractSymbolFromTab(tab) {
  if (!tab) return "";
  return sanitizeSymbol(symbolFromUrl(tab.url || "") || symbolFromTitle(tab.title || "") || "");
}

function symbolFromUrl(rawUrl) {
  try {
    const url = new URL(rawUrl);
    const qSymbol = url.searchParams.get("symbol");
    if (qSymbol) return qSymbol;

    const path = url.pathname || "";
    const pathMatch = path.match(/(?:NSE|BSE)[:\-]([A-Z0-9_\-.]+)/i);
    if (pathMatch?.[1]) return pathMatch[1];

    const queryMatch = rawUrl.match(/symbol=([A-Z]+[:\-][A-Z0-9_\-.]+)/i);
    if (queryMatch?.[1]) return queryMatch[1];
  } catch (_error) {
    return "";
  }
  return "";
}

function symbolFromTitle(title) {
  const tvMatch = title.match(/(?:NSE|BSE)[:\-]\s*([A-Z0-9_\-.]+)/i);
  if (tvMatch?.[1]) return tvMatch[1];

  const split = title.split("|")[0]?.trim() || "";
  const clean = split.replace(/[^A-Z0-9_\-. ]/gi, " ").trim();
  const token = clean.split(/\s+/).find((t) => /^[A-Z][A-Z0-9_\-.]{1,14}$/i.test(t));
  return token || "";
}

function degradedAnalyzeResult(symbol, reason) {
  return {
    ok: true,
    degraded: true,
    symbol: sanitizeSymbol(symbol || "UNKNOWN"),
    signal: "AVOID",
    direction: "NEUTRAL",
    conviction: 40,
    current_price: 0,
    trade_levels: { entry: 0, sl: 0, t1: 0, t2: 0, t3: 0, risk_reward_ratio: 0 },
    position_size: { quantity: 0, risk_budget: 0, capital_limit: 0, capital_required: 0, effective_risk: 0 },
    conviction_breakdown: { quant: 50, mtf: 50, options: 50, sentiment: 50, data_quality: 0 },
    kill_switch_status: {},
    no_trade_reasons: [reason || "Backend unavailable"],
    news_feed: [],
    detected_patterns: [],
    key_levels: { quant_levels: {}, options_support: 0, options_resistance: 0, max_pain: 0 },
    options_data: { pcr: 1.0, signal: "NEUTRAL", iv_rank: 50, iv_state: "UNKNOWN", data_unavailable: true },
    groq_rationale: [
      "System is in degraded mode.",
      "Could not fetch complete backend analysis.",
      "Avoid new trades until service restores."
    ],
    errors: [reason || "backend_error"]
  };
}
