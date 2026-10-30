// The search view: ask a question, stream the answer, compare the four retrieval modes.

import { $, el, ms, reducedMotion, store, svgEl } from "./dom.js";

const STAGES = {
  dense: { label: "Vector search", color: "var(--dense)" },
  lexical: { label: "Full-text search", color: "var(--lexical)" },
  fusion: { label: "Rank fusion", color: "var(--fusion)" },
  rerank: { label: "Rerank", color: "var(--rerank)" },
};

const CHIP_LABELS = { dense: "meaning", lexical: "keywords" };

const MODES = [
  { id: "dense", name: "Meaning", blurb: "pgvector only", color: "var(--dense)" },
  { id: "bm25", name: "Keywords", blurb: "Postgres full-text only", color: "var(--lexical)" },
  { id: "hybrid", name: "Hybrid", blurb: "both lists, fused", color: "var(--fusion)" },
  { id: "hybrid_rerank", name: "Hybrid + rerank", blurb: "fused, then reordered", color: "var(--rerank)" },
];

const STOPWORDS = new Set(
  "the a an and or of to in on for with by is are was were be been does do did can could how what why which who that this these those from as at it its into than then".split(" "),
);

// Symbols the model tends to emit as LaTeX; swap the common ones for real characters.
const LATEX_SYMBOLS = { "\\ge": "≥", "\\geq": "≥", "\\le": "≤", "\\leq": "≤", "\\approx": "≈", "\\times": "×", "\\pm": "±", "\\%": "%" };

const HISTORY_KEY = "retrieval-lab.history";
const HISTORY_SIZE = 6;

let currentRun = 0; // bumped on every submit so late responses from an older question are ignored
let controller = null;
let lastQuery = "";
let compareFor = ""; // the query the comparison board currently shows

/* ---------- text helpers ---------- */

function searchTerms(query) {
  return [...new Set(query.toLowerCase().match(/[a-z0-9]+/g) ?? [])].filter((w) => w.length > 2 && !STOPWORDS.has(w));
}

/** Text with the query's words wrapped in <mark>, built from nodes so passage text is never parsed as HTML. */
function highlighted(text, terms) {
  const frag = document.createDocumentFragment();
  if (!terms.length) {
    frag.append(text);
    return frag;
  }
  const escaped = terms.map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
  const pattern = new RegExp(`\\b(${escaped.join("|")})`, "gi");
  let last = 0;
  for (const match of text.matchAll(pattern)) {
    frag.append(text.slice(last, match.index), el("mark", {}, match[0]));
    last = match.index + match[0].length;
  }
  frag.append(text.slice(last));
  return frag;
}

function cleanLatex(text) {
  return text.replace(/\$([^$]{1,40})\$/g, (_, inner) =>
    inner.replace(/\\[a-z%]+/gi, (cmd) => LATEX_SYMBOLS[cmd] ?? cmd).replace(/\s+/g, " ").trim(),
  );
}

/** Inline formatting: **bold** and [n] citations that point at the numbered source cards. */
function inline(text) {
  const frag = document.createDocumentFragment();
  // [2] and also [2, 3], which the model writes when several passages back one claim
  for (const part of text.split(/(\*\*[^*]+\*\*|\[\d+(?:\s*,\s*\d+)*\])/)) {
    if (!part) continue;
    const cites = /^\[(\d+(?:\s*,\s*\d+)*)\]$/.exec(part);
    if (part.startsWith("**") && part.endsWith("**") && part.length > 4) {
      frag.append(el("strong", {}, part.slice(2, -2)));
    } else if (cites) {
      for (const n of cites[1].split(/\s*,\s*/)) {
        frag.append(el("button", { type: "button", class: "cite", "data-n": n, "aria-label": `Passage ${n}` }, n));
      }
    } else {
      frag.append(part);
    }
  }
  return frag;
}

const BULLET = /^\s*[-*•]\s+/;

function bulletList(lines) {
  return el("ul", {}, ...lines.map((l) => el("li", {}, inline(l.replace(BULLET, "")))));
}

/** Just enough markdown for what the model returns: paragraphs and bullet lists. */
function renderAnswer(text, streaming) {
  const root = $("answer");
  const nodes = [];
  for (const block of cleanLatex(text).trim().split(/\n{2,}/)) {
    const lines = block.split("\n");
    const firstBullet = lines.findIndex((l) => BULLET.test(l));
    if (firstBullet === 0 && lines.every((l) => BULLET.test(l))) {
      nodes.push(bulletList(lines));
    } else if (firstBullet > 0) {
      // an intro line followed by bullets in the same block
      nodes.push(el("p", {}, inline(lines.slice(0, firstBullet).join(" "))), bulletList(lines.slice(firstBullet)));
    } else {
      nodes.push(el("p", {}, inline(lines.join(" "))));
    }
  }
  root.replaceChildren(...nodes);
  if (streaming) (root.lastElementChild?.lastElementChild ?? root.lastElementChild)?.classList.add("caret");
}

/* ---------- pipeline trace ---------- */

function renderPending() {
  $("trace").replaceChildren(el("p", { class: "trace-note" }, "Searching the corpus…"), el("div", { class: "trace-bar pending" }));
}

function renderCacheHit(similarity, totalMs) {
  const bar = el("div", { class: "trace-bar" }, el("div", { class: "trace-seg", style: "--w:1;--c:var(--cache);--i:0" }));
  const legend = el(
    "ul",
    { class: "trace-legend" },
    el("li", { style: "--c:var(--cache)" }, el("i"), "Semantic cache", el("b", {}, `similarity ${Number(similarity).toFixed(2)}`)),
    el("li", { class: "total" }, `Answered in ${ms(totalMs)}, no retrieval or model call`),
  );
  $("trace").replaceChildren(bar, legend);
}

function renderTrace(timings) {
  const stages = Object.keys(STAGES).filter((name) => name in timings);
  const bar = el("div", { class: "trace-bar", role: "img", "aria-label": "Time spent in each retrieval stage" });
  const legend = el("ul", { class: "trace-legend" });
  stages.forEach((name, i) => {
    const { label, color } = STAGES[name];
    bar.append(el("div", { class: "trace-seg", style: `--w:${Math.max(timings[name], 1)};--c:${color};--i:${i}` }));
    legend.append(el("li", { style: `--c:${color}` }, el("i"), label, el("b", {}, ms(timings[name]))));
  });
  legend.append(el("li", { class: "total" }, `Retrieval ${ms(timings.total)}`));
  $("trace").replaceChildren(bar, legend);
}

/* ---------- sources ---------- */

function sourceCard(hit, i, terms) {
  const chips = Object.entries(hit.ranks ?? {}).map(([name, rank]) => {
    const stage = STAGES[name];
    return el("span", { class: "chip", style: `--c:${stage.color}`, title: `Ranked #${rank} by ${stage.label.toLowerCase()}` }, `${CHIP_LABELS[name]} #${rank}`);
  });
  const text = el("p", { class: "source-text" }, highlighted(hit.text, terms));
  const card = el(
    "li",
    { class: "source", id: `source-${i + 1}` },
    el(
      "div",
      { class: "source-head" },
      el("span", { class: "source-n" }, String(i + 1)),
      el("span", { class: "source-doc" }, `Document ${hit.doc_id}`),
      ...chips,
      el("span", { class: "source-score", title: "Score from the last stage that ran" }, hit.score.toFixed(3)),
    ),
    text,
  );
  if (hit.text.length > 260) {
    const toggle = el("button", { type: "button", class: "source-toggle" }, "Show full passage");
    toggle.addEventListener("click", () => {
      const open = card.classList.toggle("open");
      toggle.textContent = open ? "Show less" : "Show full passage";
    });
    card.append(toggle);
  }
  return card;
}

function renderSources(hits, query) {
  const terms = searchTerms(query);
  $("sources").replaceChildren(...hits.map((hit, i) => sourceCard(hit, i, terms)));
}

function bindCitations() {
  const answer = $("answer");
  const light = (e, on) => {
    const cite = e.target.closest(".cite");
    if (cite) $(`source-${cite.dataset.n}`)?.classList.toggle("lit", on);
  };
  answer.addEventListener("mouseover", (e) => light(e, true));
  answer.addEventListener("mouseout", (e) => light(e, false));
  answer.addEventListener("click", (e) => {
    const cite = e.target.closest(".cite");
    const card = cite && $(`source-${cite.dataset.n}`);
    if (!card) return;
    card.scrollIntoView({ behavior: reducedMotion() ? "auto" : "smooth", block: "center" });
    card.classList.add("lit");
    setTimeout(() => card.classList.remove("lit"), 1600);
  });
}

/* ---------- asking ---------- */

export const selectedMode = () => document.querySelector('input[name="mode"]:checked').value;

export function setMode(id) {
  const input = document.querySelector(`input[name="mode"][value="${id}"]`);
  if (input) input.checked = true;
}

function parseEvent(block) {
  const event = /^event: (.*)$/m.exec(block)?.[1];
  const raw = /^data: (.*)$/m.exec(block)?.[1];
  return event && raw !== undefined ? { event, data: JSON.parse(raw) } : null;
}

async function httpError(res) {
  if (res.status === 429) return "You have reached the limit of requests per minute. Wait a moment and ask again.";
  const body = await res.text().catch(() => "");
  return `The server returned ${res.status}. ${body.slice(0, 160)}`;
}

export async function ask(query) {
  const run = ++currentRun;
  controller?.abort();
  controller = new AbortController();
  lastQuery = query;
  rememberQuery(query);

  $("results").hidden = false;
  showTab("answer");
  renderPending();
  $("sources").replaceChildren();
  $("answer-foot").textContent = "";
  $("answer").replaceChildren(el("p", { class: "waiting" }, "Retrieving passages…"));
  $("go").disabled = true;
  $("results").scrollIntoView({ behavior: reducedMotion() ? "auto" : "smooth", block: "start" });

  const started = performance.now();
  let text = "";
  let frame = 0;
  const paint = (streaming) => {
    cancelAnimationFrame(frame);
    frame = requestAnimationFrame(() => run === currentRun && renderAnswer(text, streaming));
  };

  try {
    const res = await fetch("/ask", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ query, mode: selectedMode(), use_cache: $("use-cache").checked }),
      signal: controller.signal,
    });
    if (!res.ok) throw new Error(await httpError(res));

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let end;
      while ((end = buffer.indexOf("\n\n")) >= 0) {
        const message = parseEvent(buffer.slice(0, end));
        buffer = buffer.slice(end + 2);
        if (!message || run !== currentRun) continue;
        const { event, data } = message;

        if (event === "meta") {
          renderSources(data.sources, query);
          if (data.cache === "hit") renderCacheHit(data.similarity, performance.now() - started);
          else renderTrace(data.timings_ms);
          $("answer").replaceChildren(el("p", { class: "waiting" }, "Writing an answer…"));
        } else if (event === "token") {
          text += data;
          paint(true);
        } else if (event === "error") {
          $("answer").querySelector(".waiting")?.remove();
          $("answer").prepend(el("p", { class: "notice", role: "alert" }, data.message));
        } else if (event === "done") {
          paint(false);
          const first = data.first_token_ms ? `First token after ${ms(data.first_token_ms)}. ` : "";
          $("answer-foot").textContent = `${first}Finished in ${ms(data.total_ms)}.`;
        }
      }
    }
    if (!text && run === currentRun && !$("answer").querySelector(".notice")) {
      $("answer").replaceChildren(el("p", { class: "waiting" }, "No answer text came back. The passages on the right are what was retrieved."));
    }
  } catch (err) {
    if (err.name === "AbortError" || run !== currentRun) return;
    $("trace").replaceChildren();
    $("answer").replaceChildren(el("p", { class: "notice", role: "alert" }, err.message || "Something went wrong. Try again."));
  } finally {
    if (run === currentRun) $("go").disabled = false;
  }
}

/* ---------- compare board ---------- */

export function showTab(name) {
  for (const id of ["answer", "compare"]) {
    const active = id === name;
    $(`tab-${id}`).setAttribute("aria-selected", String(active));
    $(`panel-${id}`).hidden = !active;
  }
  if (name === "compare" && lastQuery && compareFor !== lastQuery) loadCompare(lastQuery);
  if (name === "compare") requestAnimationFrame(drawLines);
}

async function searchMode(mode, query) {
  try {
    const res = await fetch("/search", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ query, mode: mode.id, k: 5 }),
    });
    if (!res.ok) throw new Error(await httpError(res));
    return { mode, ...(await res.json()) };
  } catch (err) {
    return { mode, error: err.message };
  }
}

function column(mode, meta, body) {
  return el(
    "section",
    { class: "col", style: `--c:${mode.color}` },
    el("div", { class: "col-head" }, el("h3", {}, mode.name), el("p", {}, `${mode.blurb}, ${meta}`)),
    body,
  );
}

function cell(hit, i) {
  return el(
    "li",
    { class: "cell", "data-doc": hit.doc_id },
    el("span", { class: "cell-rank" }, `#${i + 1}`),
    el("div", {}, el("div", { class: "cell-doc" }, `Document ${hit.doc_id}`), el("p", { class: "cell-text" }, hit.text)),
  );
}

async function loadCompare(query) {
  compareFor = query;
  const board = $("board");
  $("overlap").replaceChildren();
  board.replaceChildren(...MODES.map((m) => column(m, "searching…", el("p", { class: "col-empty" }, "Searching…"))));

  const results = await Promise.all(MODES.map((m) => searchMode(m, query)));
  if (compareFor !== query) return;

  board.replaceChildren(
    svgEl("svg", { class: "board-lines", "aria-hidden": "true" }),
    ...results.map(({ mode, error, hits, timings_ms }) => {
      if (error) return column(mode, "failed", el("p", { class: "col-empty" }, error));
      const body = hits.length ? el("ol", {}, ...hits.map(cell)) : el("p", { class: "col-empty" }, "No passages matched.");
      return column(mode, ms(timings_ms.total), body);
    }),
  );
  bindBoardHover(board);
  drawLines();
  renderOverlap(results.filter((r) => !r.error));
}

/** Curves between the same document in neighbouring columns. */
function drawLines() {
  const board = $("board");
  const svg = board.querySelector("svg.board-lines");
  if (!svg || $("panel-compare").hidden) return;
  const origin = board.getBoundingClientRect();
  const cols = [...board.querySelectorAll(".col")];
  const paths = [];
  for (let i = 0; i < cols.length - 1; i++) {
    const right = new Map([...cols[i + 1].querySelectorAll(".cell")].map((c) => [c.dataset.doc, c]));
    for (const from of cols[i].querySelectorAll(".cell")) {
      const to = right.get(from.dataset.doc);
      if (!to) continue;
      const a = from.getBoundingClientRect();
      const b = to.getBoundingClientRect();
      const x1 = a.right - origin.left;
      const y1 = a.top + a.height / 2 - origin.top;
      const x2 = b.left - origin.left;
      const y2 = b.top + b.height / 2 - origin.top;
      const mid = (x1 + x2) / 2;
      const path = svgEl("path", { d: `M${x1},${y1} C${mid},${y1} ${mid},${y2} ${x2},${y2}` });
      path.dataset.doc = from.dataset.doc;
      paths.push(path);
    }
  }
  svg.replaceChildren(...paths);
}

function bindBoardHover(board) {
  const set = (doc) => {
    board.classList.toggle("hovering", Boolean(doc));
    board.querySelectorAll("[data-doc]").forEach((n) => n.classList.toggle("on", Boolean(doc) && n.dataset.doc === doc));
  };
  board.addEventListener("mouseover", (e) => set(e.target.closest(".cell")?.dataset.doc ?? null));
  board.addEventListener("mouseleave", () => set(null));
}

/** How many of the top five documents each pair of modes agrees on. */
function renderOverlap(results) {
  if (results.length < 2) return;
  const sets = results.map((r) => new Set(r.hits.map((h) => h.doc_id)));
  const head = el("tr", {}, el("th"), ...results.map((r) => el("th", { scope: "col" }, r.mode.name)));
  const rows = results.map((row, i) =>
    el(
      "tr",
      {},
      el("th", { scope: "row" }, row.mode.name),
      ...results.map((_, j) => {
        const shared = [...sets[i]].filter((d) => sets[j].has(d)).length;
        const share = sets[i].size ? shared / sets[i].size : 0;
        return el("td", { style: `--v:${share.toFixed(2)}`, class: i === j ? "self" : "" }, String(shared));
      }),
    ),
  );
  $("overlap").replaceChildren(
    el("h3", {}, "Agreement between modes"),
    el("p", { class: "compare-note" }, "How many of the five documents two modes have in common. Low numbers mean the modes found different evidence."),
    el("table", { class: "matrix" }, el("thead", {}, head), el("tbody", {}, ...rows)),
  );
}

/* ---------- history and sharing ---------- */

function rememberQuery(query) {
  const list = [query, ...store.get(HISTORY_KEY, []).filter((q) => q !== query)].slice(0, HISTORY_SIZE);
  store.set(HISTORY_KEY, list);
  renderHistory();
  const url = new URL(location.href);
  url.search = new URLSearchParams({ q: query, mode: selectedMode() }).toString();
  history.replaceState(null, "", url);
}

function renderHistory() {
  const list = store.get(HISTORY_KEY, []);
  const box = $("history");
  box.hidden = list.length === 0;
  box.replaceChildren(
    el("span", {}, "Recent"),
    ...list.map((q) => el("button", { type: "button", "data-q": q }, q)),
    el("button", { type: "button", class: "clear", id: "clear-history" }, "Clear"),
  );
}

async function copyLink() {
  const button = $("share");
  try {
    await navigator.clipboard.writeText(location.href);
    button.textContent = "Link copied";
  } catch {
    button.textContent = "Copy it from the address bar";
  }
  setTimeout(() => (button.textContent = "Copy link"), 1800);
}

/* ---------- wiring ---------- */

export function initSearch() {
  $("ask-form").addEventListener("submit", (e) => {
    e.preventDefault();
    const query = $("q").value.trim();
    if (query) ask(query);
  });

  for (const box of [$("examples"), $("history")]) {
    box.addEventListener("click", (e) => {
      const button = e.target.closest("button");
      if (!button) return;
      if (button.id === "clear-history") {
        store.set(HISTORY_KEY, []);
        renderHistory();
        return;
      }
      const query = button.dataset.q ?? button.textContent;
      $("q").value = query;
      ask(query);
    });
  }

  for (const id of ["answer", "compare"]) $(`tab-${id}`).addEventListener("click", () => showTab(id));
  $("share").addEventListener("click", copyLink);
  new ResizeObserver(drawLines).observe($("board"));
  bindCitations();
  renderHistory();

  // a shared link such as /?q=...&mode=hybrid runs straight away
  const params = new URLSearchParams(location.search);
  if (params.get("mode")) setMode(params.get("mode"));
  const shared = params.get("q")?.trim();
  if (shared) {
    $("q").value = shared;
    ask(shared);
  }
}
