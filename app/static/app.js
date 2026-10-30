// Entry point: theme, connection status, routing between the two views, keyboard shortcuts.

import { initBenchmarks, showBenchmarks } from "./benchmarks.js";
import { $, store } from "./dom.js";
import { initSearch, setMode } from "./search.js";

const THEME_KEY = "retrieval-lab.theme";
const THEMES = ["system", "light", "dark"];
const MODE_KEYS = { 1: "dense", 2: "bm25", 3: "hybrid", 4: "hybrid_rerank" };

/* ---------- theme ---------- */

function applyTheme(theme) {
  const root = document.documentElement;
  if (theme === "system") root.removeAttribute("data-theme");
  else root.dataset.theme = theme;
  $("theme").textContent = theme === "system" ? "Auto" : theme === "light" ? "Light" : "Dark";
  $("theme").setAttribute("aria-label", `Colour theme: ${theme}. Press to change.`);
}

function initTheme() {
  let theme = store.get(THEME_KEY, "system");
  if (!THEMES.includes(theme)) theme = "system";
  applyTheme(theme);
  $("theme").addEventListener("click", () => {
    theme = THEMES[(THEMES.indexOf(theme) + 1) % THEMES.length];
    store.set(THEME_KEY, theme);
    applyTheme(theme);
  });
}

/* ---------- connection status ---------- */

async function loadStatus() {
  const box = $("status");
  try {
    const res = await fetch("/health");
    if (!res.ok) throw new Error(res.status);
    const info = await res.json();
    box.dataset.state = "ok";
    $("status-text").textContent = `${info.chunks.toLocaleString()} passages, ${info.cache} cache`;
  } catch {
    box.dataset.state = "down";
    $("status-text").textContent = "API unreachable";
  }
}

/* ---------- routing ---------- */

function route() {
  const onBenchmarks = location.hash.startsWith("#/benchmarks");
  $("view-search").hidden = onBenchmarks;
  $("view-benchmarks").hidden = !onBenchmarks;
  for (const link of document.querySelectorAll(".nav a")) {
    const active = (link.getAttribute("href") === "#/benchmarks") === onBenchmarks;
    if (active) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  }
  if (onBenchmarks) showBenchmarks();
  window.scrollTo(0, 0);
}

/* ---------- keyboard ---------- */

function initShortcuts() {
  document.addEventListener("keydown", (e) => {
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    if (/^(input|textarea|select)$/i.test(document.activeElement?.tagName ?? "")) return;
    if (e.key === "/") {
      e.preventDefault();
      location.hash = "#/";
      $("q").focus();
      $("q").select();
    } else if (MODE_KEYS[e.key]) {
      setMode(MODE_KEYS[e.key]);
    }
  });
}

initTheme();
initSearch();
initBenchmarks();
initShortcuts();
window.addEventListener("hashchange", route);
route();
loadStatus();
