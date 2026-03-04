const ui = {
  symbolText: document.getElementById("symbol-text"),
  marketStatus: document.getElementById("market-status"),
  livePrice: document.getElementById("live-price"),
  sessionWarning: document.getElementById("session-warning"),
  signalBanner: document.getElementById("signal-banner"),
  signalChange: document.getElementById("signal-change"),
  lastUpdated: document.getElementById("last-updated"),
  meterProgress: document.getElementById("meter-progress"),
  convictionText: document.getElementById("conviction-text"),
  entry: document.getElementById("entry"),
  sl: document.getElementById("sl"),
  t1: document.getElementById("t1"),
  t2: document.getElementById("t2"),
  t3: document.getElementById("t3"),
  rrr: document.getElementById("rrr"),
  qty: document.getElementById("qty"),
  riskBudget: document.getElementById("risk-budget"),
  breakdownBars: document.getElementById("breakdown-bars"),
  killSwitchList: document.getElementById("kill-switch-list"),
  agentLeds: document.getElementById("agent-leds"),
  noTradeReasons: document.getElementById("no-trade-reasons"),
  patternsList: document.getElementById("patterns-list"),
  levelsList: document.getElementById("levels-list"),
  newsList: document.getElementById("news-list"),
  optionsList: document.getElementById("options-list"),
  rationaleList: document.getElementById("rationale-list"),
  signalHistoryList: document.getElementById("signal-history-list"),
  analyzeBtn: document.getElementById("analyze-btn"),
  quickScanBtn: document.getElementById("quick-scan-btn"),
  screenshotBtn: document.getElementById("screenshot-btn"),
  downloadCsvBtn: document.getElementById("download-csv-btn"),
  scanOutput: document.getElementById("scan-output"),
  loadingSkeleton: document.getElementById("loading-skeleton")
};

const CIRCLE_CIRCUMFERENCE = 327;
let activeSymbol = "";
let ws = null;
let lastAnalysis = null;
let lastVision = null;
let lastQuickScan = null;
let lastSignalSnapshot = null;
const signalChangeLog = [];
const liveTickLog = [];
const eventLog = [];
const errorLog = [];

bootstrap();

async function bootstrap() {
  wireEvents();
  await hydrateActiveSymbol();
  pushEvent("panel_bootstrap", { symbol: activeSymbol || "" });
  renderSignalHistory();
  if (activeSymbol) {
    ui.symbolText.textContent = activeSymbol;
    connectLiveSocket(activeSymbol);
  }
}

function wireEvents() {
  ui.analyzeBtn?.addEventListener("click", onAnalyze);
  ui.quickScanBtn?.addEventListener("click", onQuickScan);
  ui.screenshotBtn?.addEventListener("click", onScreenshot);
  const downloadHandler =
    typeof onDownloadCsv === "function"
      ? onDownloadCsv
      : () => {
          ui.scanOutput.textContent = "CSV export handler not available. Reload extension once.";
        };
  ui.downloadCsvBtn?.addEventListener("click", downloadHandler);
}

async function hydrateActiveSymbol() {
  try {
    const res = await safeRuntimeMessage({ type: "GET_ACTIVE_SYMBOL" });
    activeSymbol = sanitizeSymbol(res?.symbol || "");
  } catch (_error) {
    activeSymbol = "";
  }
}

async function onAnalyze() {
  showLoading(true);
  try {
    await hydrateActiveSymbol();
    if (!activeSymbol) {
      throw new Error("Symbol not detected. Open TradingView chart with NSE/BSE symbol.");
    }
    pushEvent("analyze_request", { symbol: activeSymbol, capital: 100000 });
    const response = await safeRuntimeMessage({
      type: "ANALYZE_SYMBOL",
      payload: { symbol: activeSymbol, capital: 100000 }
    });
    if (!response?.ok || !response?.result) {
      throw new Error(response?.error || "Analyze failed");
    }
    lastAnalysis = response.result;
    pushEvent("analyze_response", {
      symbol: lastAnalysis?.symbol || activeSymbol,
      signal: lastAnalysis?.signal || "",
      conviction: Number(lastAnalysis?.conviction || 0),
      degraded: Boolean(lastAnalysis?.degraded)
    });
    renderAnalysis(lastAnalysis);
    connectLiveSocket(activeSymbol);
  } catch (error) {
    renderError(String(error?.message || error));
  } finally {
    showLoading(false);
  }
}

async function onQuickScan() {
  try {
    await hydrateActiveSymbol();
    const symbols = dedupe([
      activeSymbol,
      "RELIANCE",
      "TCS",
      "INFY",
      "HDFCBANK",
      "ICICIBANK"
    ]).filter(Boolean);
    pushEvent("quick_scan_request", { symbols, capital: 100000 });
    const response = await safeRuntimeMessage({
      type: "QUICK_SCAN",
      payload: { symbols, capital: 100000 }
    });
    if (!response?.ok || !response?.result) {
      throw new Error(response?.error || "Quick scan failed");
    }
    lastQuickScan = response.result;
    pushEvent("quick_scan_response", {
      ranked_count: Array.isArray(lastQuickScan?.ranked) ? lastQuickScan.ranked.length : 0,
      top_symbol: lastQuickScan?.ranked?.[0]?.symbol || "",
      top_signal: lastQuickScan?.ranked?.[0]?.signal || "",
      top_conviction: Number(lastQuickScan?.ranked?.[0]?.conviction || 0)
    });
    const ranked = response.result?.ranked || [];
    if (!ranked.length) {
      ui.scanOutput.textContent = "Quick Scan: no symbols ranked.";
      return;
    }
    ui.scanOutput.textContent = `Top: ${ranked[0].symbol} (${ranked[0].signal}, ${num(ranked[0].conviction, 1)}%)`;
  } catch (error) {
    pushError("quick_scan_error", String(error?.message || error));
    ui.scanOutput.textContent = `Quick Scan error: ${String(error?.message || error)}`;
  }
}

async function onScreenshot() {
  try {
    const symbol = activeSymbol || (lastAnalysis?.symbol || "");
    pushEvent("screenshot_request", { symbol });
    const response = await safeRuntimeMessage({
      type: "SCREENSHOT_ANALYSIS",
      payload: { symbol }
    });
    if (!response?.ok || !response?.result) {
      throw new Error(response?.error || "Screenshot analysis failed");
    }
    lastVision = response.result;
    pushEvent("screenshot_response", {
      symbol: lastVision?.symbol || symbol,
      patterns: Array.isArray(lastVision?.patterns) ? lastVision.patterns : []
    });
    appendVisionPatterns(lastVision?.patterns || []);
  } catch (error) {
    pushError("screenshot_error", String(error?.message || error));
    ui.scanOutput.textContent = `Screenshot error: ${String(error?.message || error)}`;
  }
}

function renderAnalysis(data) {
  const symbol = sanitizeSymbol(data?.symbol || activeSymbol || "");
  const signal = String(data?.signal || "AVOID").toUpperCase();
  const conviction = clamp(Number(data?.conviction || 0), 0, 100);
  const price = Number(data?.current_price || 0);

  ui.symbolText.textContent = symbol || "--";
  ui.livePrice.textContent = num(price, 2);
  ui.sessionWarning.textContent = data?.market_research?.session_warning || "Session stable";

  renderSignal(signal, conviction);
  updateSignalTracker({
    symbol,
    signal,
    direction: String(data?.direction || "NEUTRAL").toUpperCase(),
    conviction,
    price
  });
  renderTrade(data?.trade_levels || {}, data?.position_size || {});
  renderBreakdown(data?.conviction_breakdown || {});
  renderKillSwitch(data?.kill_switch_status || {});
  renderAgentLeds(data?.agent_status || {});
  renderNoTradeReasons(data?.no_trade_reasons || []);
  renderPatterns(data?.detected_patterns || []);
  renderLevels(data?.key_levels || {});
  renderNews(data?.news_feed || []);
  renderOptions(data?.options_data || {});
  renderRationale(data?.groq_rationale || []);
  if (lastVision?.patterns?.length) appendVisionPatterns(lastVision.patterns);

  const marketTag = String(data?.market_research?.market_session || "UNKNOWN").toUpperCase();
  ui.marketStatus.textContent = marketTag;
  ui.marketStatus.className = `pill ${marketClass(marketTag)}`;
}

function renderSignal(signal, conviction) {
  const cls = signal.includes("BUY")
    ? "green"
    : signal.includes("SELL")
      ? "red"
      : "yellow";
  ui.signalBanner.className = `signal ${cls} animate`;
  ui.signalBanner.textContent = signal;
  setTimeout(() => ui.signalBanner.classList.remove("animate"), 250);

  const dash = CIRCLE_CIRCUMFERENCE - (conviction / 100) * CIRCLE_CIRCUMFERENCE;
  ui.meterProgress.style.strokeDashoffset = String(dash);
  ui.meterProgress.style.stroke = cls === "green" ? "#00d4aa" : cls === "red" ? "#ff4757" : "#ffa502";
  ui.convictionText.textContent = `${num(conviction, 0)}%`;
}

function renderTrade(trade, sizing) {
  ui.entry.textContent = num(trade?.entry || 0, 2);
  ui.sl.textContent = num(trade?.sl || 0, 2);
  ui.t1.textContent = num(trade?.t1 || 0, 2);
  ui.t2.textContent = num(trade?.t2 || 0, 2);
  ui.t3.textContent = num(trade?.t3 || 0, 2);
  ui.rrr.textContent = num(trade?.risk_reward_ratio || 0, 2);
  ui.qty.textContent = num(sizing?.quantity || 0, 0);
  ui.riskBudget.textContent = num(sizing?.risk_budget || 0, 2);
}

function renderBreakdown(breakdown) {
  const rows = [
    ["Quant", breakdown?.quant || 0],
    ["MTF", breakdown?.mtf || 0],
    ["Options", breakdown?.options || 0],
    ["Sentiment", breakdown?.sentiment || 0]
  ];
  ui.breakdownBars.innerHTML = "";
  rows.forEach(([name, val]) => {
    const v = clamp(Number(val || 0), 0, 100);
    const row = document.createElement("div");
    row.className = "bar-row";
    row.innerHTML = `
      <span>${name}</span>
      <div class="bar-track"><div class="bar-fill" style="width:${v}%"></div></div>
      <span>${num(v, 0)}%</span>
    `;
    ui.breakdownBars.appendChild(row);
  });
}

function renderKillSwitch(statusMap) {
  ui.killSwitchList.innerHTML = "";
  const rows = Object.entries(statusMap);
  if (!rows.length) {
    ui.killSwitchList.appendChild(makeRow("No kill-switch data", "yellow"));
    return;
  }
  rows.forEach(([key, ok]) => {
    const cls = ok ? "green" : "red";
    const item = document.createElement("div");
    item.className = "status-item";
    item.innerHTML = `<span>${humanizeKey(key)}</span><span class="dot ${cls}"></span>`;
    ui.killSwitchList.appendChild(item);
  });
}

function renderAgentLeds(statusMap) {
  ui.agentLeds.innerHTML = "";
  const rows = Object.entries(statusMap);
  if (!rows.length) {
    ui.agentLeds.appendChild(makeRow("No agent status", "yellow"));
    return;
  }
  rows.forEach(([agent, status]) => {
    const cls = statusToClass(status);
    const item = document.createElement("div");
    item.className = "status-item";
    item.innerHTML = `<span>${humanizeKey(agent)}</span><span class="dot ${cls}"></span>`;
    ui.agentLeds.appendChild(item);
  });
}

function renderNoTradeReasons(reasons) {
  ui.noTradeReasons.innerHTML = "";
  if (!Array.isArray(reasons) || reasons.length === 0) {
    ui.noTradeReasons.appendChild(makeRow("No hard blocker detected.", "green"));
    return;
  }
  reasons.forEach((text) => ui.noTradeReasons.appendChild(makeRow(text, "red")));
}

function renderPatterns(patterns) {
  ui.patternsList.innerHTML = "";
  if (!Array.isArray(patterns) || !patterns.length) {
    ui.patternsList.appendChild(makeRow("No clear pattern.", "yellow"));
    return;
  }
  patterns.forEach((p) => {
    const line = `${p?.name || "Pattern"}: ${p?.reason || p?.why || "--"}`;
    ui.patternsList.appendChild(makeRow(line, "neutral"));
  });
}

function appendVisionPatterns(patterns) {
  if (!Array.isArray(patterns) || !patterns.length) return;
  patterns.forEach((name) => ui.patternsList.appendChild(makeRow(`Vision: ${name}`, "green")));
}

function renderLevels(levels) {
  ui.levelsList.innerHTML = "";
  const q = levels?.quant_levels || {};
  const pivot = q?.pivots || {};
  const swings = q?.swings || {};
  const roundLevels = q?.round_levels || {};
  const rows = [
    `Options Support: ${num(levels?.options_support || 0, 2)}`,
    `Options Resistance: ${num(levels?.options_resistance || 0, 2)}`,
    `Max Pain: ${num(levels?.max_pain || 0, 2)}`,
    `Pivot PP: ${num(pivot?.classic_pp || 0, 2)} | R1: ${num(pivot?.classic_r1 || 0, 2)} | S1: ${num(pivot?.classic_s1 || 0, 2)}`,
    `Swing High: ${num(swings?.swing_high || 0, 2)} | Swing Low: ${num(swings?.swing_low || 0, 2)}`,
    `Round Floor: ${num(roundLevels?.round_floor || 0, 2)} | Round Ceil: ${num(roundLevels?.round_ceil || 0, 2)}`
  ];
  rows.forEach((line) => ui.levelsList.appendChild(makeRow(line, "neutral")));
}

function renderNews(news) {
  ui.newsList.innerHTML = "";
  if (!Array.isArray(news) || !news.length) {
    ui.newsList.appendChild(makeRow("No latest news found in last 24h.", "yellow"));
    return;
  }
  news.slice(0, 10).forEach((n) => {
    const title = n?.title || "--";
    const src = n?.source ? ` [${n.source}]` : "";
    ui.newsList.appendChild(makeRow(`${title}${src}`, "neutral"));
  });
}

function renderOptions(optionsData) {
  ui.optionsList.innerHTML = "";
  const rows = [
    `PCR: ${num(optionsData?.pcr || 1.0, 3)}`,
    `Signal: ${optionsData?.signal || "NEUTRAL"}`,
    `IV Rank: ${num(optionsData?.iv_rank || 50, 2)} (${optionsData?.iv_state || "UNKNOWN"})`,
    optionsData?.data_unavailable ? "Options data unavailable - neutral scoring applied." : "Options data available."
  ];
  rows.forEach((line) => ui.optionsList.appendChild(makeRow(line, optionsData?.data_unavailable ? "yellow" : "neutral")));
}

function renderRationale(lines) {
  ui.rationaleList.innerHTML = "";
  if (!Array.isArray(lines) || !lines.length) {
    ui.rationaleList.appendChild(makeRow("Rationale unavailable.", "yellow"));
    return;
  }
  lines.forEach((line) => ui.rationaleList.appendChild(makeRow(line, "neutral")));
}

function renderError(message) {
  pushError("render_error", message);
  ui.scanOutput.textContent = message;
  ui.noTradeReasons.innerHTML = "";
  ui.noTradeReasons.appendChild(makeRow(message, "red"));
}

function makeRow(text, type) {
  const div = document.createElement("div");
  div.className = `row ${type === "red" ? "red" : ""}`;
  div.textContent = String(text);
  return div;
}

function showLoading(show) {
  ui.loadingSkeleton.classList.toggle("hidden", !show);
}

async function safeRuntimeMessage(message) {
  if (!chrome?.runtime?.id) {
    throw new Error("Extension reloaded. Reopen sidepanel and try again.");
  }
  try {
    return await chrome.runtime.sendMessage(message);
  } catch (error) {
    const raw = String(error?.message || error || "");
    if (raw.includes("Extension context invalidated")) {
      throw new Error("Extension reloaded. Reopen sidepanel and retry.");
    }
    throw error;
  }
}

function updateSignalTracker(snapshot) {
  const ts = nowIso();
  const current = {
    ts,
    symbol: sanitizeSymbol(snapshot?.symbol || activeSymbol || ""),
    signal: String(snapshot?.signal || "AVOID").toUpperCase(),
    direction: String(snapshot?.direction || "NEUTRAL").toUpperCase(),
    conviction: Number(snapshot?.conviction || 0),
    price: Number(snapshot?.price || 0)
  };

  let chipClass = current.signal.includes("BUY") ? "green" : current.signal.includes("SELL") ? "red" : "yellow";
  let message = `${current.signal} | ${current.direction} | Conviction ${num(current.conviction, 1)}%`;

  if (!lastSignalSnapshot) {
    signalChangeLog.push({
      ts: current.ts,
      symbol: current.symbol,
      from_signal: "INIT",
      to_signal: current.signal,
      from_direction: "INIT",
      to_direction: current.direction,
      conviction_delta: num(current.conviction, 2),
      price: num(current.price, 2),
      reason: "first_analysis"
    });
  } else {
    const changedSignal = lastSignalSnapshot.signal !== current.signal;
    const changedDirection = lastSignalSnapshot.direction !== current.direction;
    const convictionDelta = current.conviction - lastSignalSnapshot.conviction;
    if (changedSignal || changedDirection) {
      message = `Change: ${lastSignalSnapshot.signal}/${lastSignalSnapshot.direction} -> ${current.signal}/${current.direction} (${convictionDelta >= 0 ? "+" : ""}${num(convictionDelta, 1)}%)`;
      chipClass = changedSignal
        ? current.signal.includes("BUY")
          ? "green"
          : current.signal.includes("SELL")
            ? "red"
            : "yellow"
        : "yellow";
      signalChangeLog.push({
        ts: current.ts,
        symbol: current.symbol,
        from_signal: lastSignalSnapshot.signal,
        to_signal: current.signal,
        from_direction: lastSignalSnapshot.direction,
        to_direction: current.direction,
        conviction_delta: num(convictionDelta, 2),
        price: num(current.price, 2),
        reason: "signal_or_direction_change"
      });
      pushEvent("signal_change", {
        symbol: current.symbol,
        from_signal: lastSignalSnapshot.signal,
        to_signal: current.signal,
        from_direction: lastSignalSnapshot.direction,
        to_direction: current.direction,
        conviction_delta: convictionDelta
      });
    } else {
      message = `No signal change | ${current.signal} stable (${convictionDelta >= 0 ? "+" : ""}${num(convictionDelta, 1)}%)`;
    }
  }

  lastSignalSnapshot = current;
  ui.signalChange.className = `signal-chip ${chipClass}`;
  ui.signalChange.textContent = message;
  ui.lastUpdated.textContent = `Updated: ${formatLocalTime(ts)}`;
  renderSignalHistory();
}

function renderSignalHistory() {
  ui.signalHistoryList.innerHTML = "";
  if (!signalChangeLog.length) {
    ui.signalHistoryList.appendChild(makeRow("No signal flip detected yet.", "yellow"));
    return;
  }
  signalChangeLog
    .slice(-20)
    .reverse()
    .forEach((item) => {
      const line = `${formatLocalTime(item.ts)} | ${item.from_signal}/${item.from_direction} -> ${item.to_signal}/${item.to_direction} | dConv ${item.conviction_delta}% | Price ${item.price}`;
      ui.signalHistoryList.appendChild(makeRow(line, "neutral"));
    });
}

async function onDownloadCsv() {
  try {
    const backendSnapshot = await fetchBackendTransparency();
    const rows = buildTransparencyRows(backendSnapshot);
    const csvText = rowsToCsv(rows);
    const symbol = sanitizeSymbol(activeSymbol || lastAnalysis?.symbol || "MARKET");
    const stamp = fileTimestamp();
    const filename = `aegis_transparency_${symbol}_${stamp}.csv`;
    downloadTextAsFile(csvText, filename, "text/csv;charset=utf-8");
    ui.scanOutput.textContent = `CSV downloaded: ${filename} (${rows.length} rows)`;
    pushEvent("csv_export", { filename, rows: rows.length });
  } catch (error) {
    pushError("csv_export_error", String(error?.message || error));
    ui.scanOutput.textContent = `CSV export error: ${String(error?.message || error)}`;
  }
}

async function fetchBackendTransparency() {
  try {
    const base = await getApiBase();
    const response = await fetch(`${base}/export/analysis-history`);
    if (!response.ok) throw new Error(`backend export failed ${response.status}`);
    const body = await response.json();
    return {
      rows: Array.isArray(body?.rows) ? body.rows : [],
      recentLogLines: Array.isArray(body?.recent_log_lines) ? body.recent_log_lines : []
    };
  } catch (error) {
    pushError("backend_export_fetch", String(error?.message || error));
    return { rows: [], recentLogLines: [] };
  }
}

async function getApiBase() {
  try {
    const stored = await chrome.storage.local.get(["apiBase"]);
    const base = String(stored?.apiBase || "").trim();
    return base || "http://127.0.0.1:8000";
  } catch (_error) {
    return "http://127.0.0.1:8000";
  }
}

function buildTransparencyRows(backendSnapshot) {
  const now = nowIso();
  const symbol = sanitizeSymbol(activeSymbol || lastAnalysis?.symbol || "");
  const rows = [];
  const add = (section, key, value, json = "", ts = now, sym = symbol) => {
    rows.push({
      section,
      timestamp: ts,
      symbol: sym,
      key,
      value: String(value ?? ""),
      json: json ? String(json) : ""
    });
  };

  add("meta", "exported_at", now);
  add("meta", "active_symbol", symbol || "UNKNOWN");
  add("meta", "current_live_price", ui.livePrice.textContent || "");
  add("meta", "signal_chip", ui.signalChange.textContent || "");

  flattenToRows("analysis", lastAnalysis || {}, add, "", symbol, now);
  flattenToRows("quick_scan", lastQuickScan || {}, add, "", symbol, now);
  flattenToRows("vision", lastVision || {}, add, "", symbol, now);

  signalChangeLog.forEach((item, index) => add("signal_change_log", `row_${index + 1}`, "", safeJson(item), item.ts, item.symbol || symbol));
  liveTickLog.forEach((item, index) => add("live_ticks", `tick_${index + 1}`, item.price, safeJson(item), item.ts, item.symbol || symbol));
  eventLog.forEach((item, index) => add("events", `event_${index + 1}`, item.type, safeJson(item), item.ts, symbol));
  errorLog.forEach((item, index) => add("errors", `error_${index + 1}`, item.message, safeJson(item), item.ts, symbol));

  const backendRows = Array.isArray(backendSnapshot?.rows) ? backendSnapshot.rows : [];
  backendRows.forEach((item, index) => {
    add("backend_analysis_history", `history_${index + 1}`, `${item.signal || ""} | ${item.conviction || ""}`, safeJson(item), item.timestamp || now, item.symbol || symbol);
  });

  const recentLogLines = Array.isArray(backendSnapshot?.recentLogLines) ? backendSnapshot.recentLogLines : [];
  recentLogLines.forEach((line, index) => {
    const ts = extractLogTimestamp(line, now);
    add("backend_runtime_log", `line_${index + 1}`, line, "", ts, symbol);
  });

  return rows;
}

function flattenToRows(section, payload, addFn, prefix = "", rowSymbol = "", rowTimestamp = "") {
  const resolvedSymbol = sanitizeSymbol(rowSymbol || activeSymbol || lastAnalysis?.symbol || "UNKNOWN");
  const resolvedTs = rowTimestamp || nowIso();

  if (payload === null || payload === undefined) {
    addFn(section, prefix || "value", "", "", resolvedTs, resolvedSymbol);
    return;
  }
  if (Array.isArray(payload)) {
    if (!payload.length) {
      addFn(section, prefix || "array", "[]", "", resolvedTs, resolvedSymbol);
      return;
    }
    payload.forEach((item, index) => {
      const p = prefix ? `${prefix}[${index}]` : `[${index}]`;
      const nestedSymbol = inferSymbol(item, resolvedSymbol);
      flattenToRows(section, item, addFn, p, nestedSymbol, resolvedTs);
    });
    return;
  }
  if (typeof payload === "object") {
    const objSymbol = inferSymbol(payload, resolvedSymbol);
    const entries = Object.entries(payload);
    if (!entries.length) {
      addFn(section, prefix || "object", "{}", "", resolvedTs, objSymbol);
      return;
    }
    entries.forEach(([key, value]) => {
      const p = prefix ? `${prefix}.${key}` : key;
      const nestedSymbol = key === "symbol" ? sanitizeSymbol(value || objSymbol) : objSymbol;
      flattenToRows(section, value, addFn, p, nestedSymbol, resolvedTs);
    });
    return;
  }
  addFn(section, prefix || "value", payload, "", resolvedTs, resolvedSymbol);
}

function inferSymbol(payload, fallback = "") {
  if (payload && typeof payload === "object" && typeof payload.symbol === "string") {
    const parsed = sanitizeSymbol(payload.symbol);
    if (parsed) return parsed;
  }
  return sanitizeSymbol(fallback || "");
}

function extractLogTimestamp(line, fallbackIso) {
  const raw = String(line || "");
  const match = raw.match(/^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})/);
  if (!match?.[1]) return fallbackIso;
  const normalized = match[1].replace(",", ".");
  const date = new Date(normalized.replace(" ", "T"));
  return Number.isNaN(date.getTime()) ? fallbackIso : date.toISOString();
}

function rowsToCsv(rows) {
  const header = ["section", "timestamp", "symbol", "key", "value", "json"];
  const lines = [header.join(",")];
  rows.forEach((row) => {
    const line = header.map((key) => csvCell(row[key])).join(",");
    lines.push(line);
  });
  return `${lines.join("\n")}\n`;
}

function csvCell(input) {
  const str = String(input ?? "");
  if (!/[,"\n]/.test(str)) return str;
  return `"${str.replace(/"/g, '""')}"`;
}

function downloadTextAsFile(text, filename, mimeType) {
  const blob = new Blob([text], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function pushEvent(type, payload) {
  eventLog.push({ ts: nowIso(), type, payload: payload || {} });
  if (eventLog.length > 2000) eventLog.shift();
}

function pushError(type, message) {
  errorLog.push({ ts: nowIso(), type, message: String(message || "unknown_error") });
  if (errorLog.length > 500) errorLog.shift();
}

function pushLiveTick(tick) {
  liveTickLog.push(tick);
  if (liveTickLog.length > 1000) liveTickLog.shift();
}

function nowIso() {
  return new Date().toISOString();
}

function formatLocalTime(isoString) {
  const date = new Date(isoString);
  return Number.isNaN(date.getTime()) ? "--" : date.toLocaleString("en-IN");
}

function safeJson(value) {
  try {
    return JSON.stringify(value);
  } catch (_error) {
    return "";
  }
}

function fileTimestamp() {
  const date = new Date();
  const yyyy = date.getFullYear();
  const mm = String(date.getMonth() + 1).padStart(2, "0");
  const dd = String(date.getDate()).padStart(2, "0");
  const hh = String(date.getHours()).padStart(2, "0");
  const mi = String(date.getMinutes()).padStart(2, "0");
  const ss = String(date.getSeconds()).padStart(2, "0");
  return `${yyyy}${mm}${dd}_${hh}${mi}${ss}`;
}

function connectLiveSocket(symbol) {
  try {
    if (ws) ws.close();
    pushEvent("live_socket_connect", { symbol });
    ws = new WebSocket(`ws://127.0.0.1:8000/ws/live/${encodeURIComponent(symbol)}`);
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data || "{}");
      if (!data?.ok) return;
      const price = Number(data?.price?.last_price || 0);
      if (price > 0) ui.livePrice.textContent = num(price, 2);
      pushLiveTick({
        symbol: data?.symbol || symbol,
        price,
        market_session: data?.market_session || "",
        ts: nowIso()
      });
      if (data?.market_session) {
        ui.marketStatus.textContent = data.market_session;
        ui.marketStatus.className = `pill ${marketClass(data.market_session)}`;
      }
    };
    ws.onerror = () => {
      pushError("ws_error", `Live socket error for ${symbol}`);
    };
  } catch (_error) {}
}

function marketClass(status) {
  const s = String(status || "").toUpperCase();
  if (s.includes("OPEN")) return "green";
  if (s.includes("VOLATILITY") || s.includes("PRE")) return "yellow";
  return "red";
}

function statusToClass(status) {
  const s = String(status || "").toUpperCase();
  if (s.includes("GREEN")) return "green";
  if (s.includes("RED")) return "red";
  return "yellow";
}

function humanizeKey(key) {
  return String(key || "")
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
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

function dedupe(items) {
  const seen = new Set();
  const out = [];
  for (const item of items) {
    const k = String(item || "").trim().toUpperCase();
    if (!k || seen.has(k)) continue;
    seen.add(k);
    out.push(k);
  }
  return out;
}

function clamp(value, low, high) {
  return Math.max(low, Math.min(high, value));
}

function num(value, precision = 2) {
  const n = Number(value || 0);
  return Number.isFinite(n) ? n.toFixed(precision) : "--";
}
