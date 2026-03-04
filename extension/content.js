let lastSentSymbol = "";
let pollTimer = null;

bootstrap();

function bootstrap() {
  detectAndSendSymbol();
  pollTimer = setInterval(detectAndSendSymbol, 5000);
  window.addEventListener("beforeunload", stopPolling, { once: true });
}

function detectAndSendSymbol() {
  if (!runtimeAvailable()) {
    stopPolling();
    return;
  }
  const symbol = extractSymbol();
  if (!symbol) return;
  const changed = symbol !== lastSentSymbol;
  lastSentSymbol = symbol;
  try {
    chrome.runtime.sendMessage(
      {
        type: "SYMBOL_DETECTED",
        symbol,
        changed
      },
      () => {
        const err = chrome.runtime?.lastError?.message || "";
        if (err.includes("Extension context invalidated")) {
          stopPolling();
        }
      }
    );
  } catch (_error) {
    stopPolling();
  }
}

function extractSymbol() {
  const fromUrl = symbolFromUrl(location.href);
  if (fromUrl) return fromUrl;

  const fromTitle = symbolFromTitle(document.title || "");
  if (fromTitle) return fromTitle;

  return "";
}

function symbolFromUrl(rawUrl) {
  try {
    const url = new URL(rawUrl);
    const qSymbol = url.searchParams.get("symbol");
    if (qSymbol) return normalizeSymbol(qSymbol);

    const path = url.pathname || "";
    const pathMatch = path.match(/(?:NSE|BSE)[:\-]([A-Z0-9_\-.]+)/i);
    if (pathMatch?.[1]) return normalizeSymbol(pathMatch[1]);

    const queryMatch = rawUrl.match(/symbol=([A-Z]+[:\-][A-Z0-9_\-.]+)/i);
    if (queryMatch?.[1]) return normalizeSymbol(queryMatch[1]);
  } catch (_error) {
    return "";
  }
  return "";
}

function symbolFromTitle(title) {
  const tvMatch = title.match(/(?:NSE|BSE)[:\-]\s*([A-Z0-9_\-.]+)/i);
  if (tvMatch?.[1]) return normalizeSymbol(tvMatch[1]);

  const split = title.split("|")[0]?.trim() || "";
  const clean = split.replace(/[^A-Z0-9_\-. ]/gi, " ").trim();
  const token = clean.split(/\s+/).find((t) => /^[A-Z][A-Z0-9_\-.]{1,14}$/i.test(t));
  return token ? normalizeSymbol(token) : "";
}

function normalizeSymbol(input) {
  const raw = String(input || "").toUpperCase().trim();
  const withoutPrefix = raw.replace(/^NSE[:\-]/, "").replace(/^BSE[:\-]/, "");
  return withoutPrefix.replace(/[^A-Z0-9_\-.]/g, "").slice(0, 20);
}

function runtimeAvailable() {
  return Boolean(chrome?.runtime?.id);
}

function stopPolling() {
  if (!pollTimer) return;
  clearInterval(pollTimer);
  pollTimer = null;
}
