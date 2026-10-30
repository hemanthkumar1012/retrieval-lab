// retrieval-lab front end. No framework: one form, one results area, one SSE reader.

const $ = (id) => document.getElementById(id);

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

// Characters the model tends to emit as LaTeX; swap the common ones for real symbols.
const LATEX_SYMBOLS = { "\\ge": "≥", "\\geq": "≥", "\\le": "≤", "\\leq": "≤", "\\approx": "≈", "\\times": "×", "\\pm": "±", "\\%": "%" };

let currentRun = 0; // bumped on every submit so late responses from an older question are ignored
let controller = null;
let lastQuery = "";
let sources = [];

/* ---------- small helpers ---------- */

function el(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (key === "class") node.className = value;
    else if (key === "style") node.setAttribute("style", value);
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value);
  }
  node.append(...children);
  return node;
}

function svgEl(tag, attrs = {}) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
  return node;
}

const ms = (value) => (value >= 1000 ? `${(value / 1000).toFixed(2)} s` : `${Math.round(value)} ms`);

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
  const pattern = new RegExp(`\\b(${terms.map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|")})`, "gi");
  let last = 0;
  for (const match of text.matchAll(pattern)) {
    frag.append(text.slice(last, match.index), el("mark", {}, match[0]));
    last = match.index + match[0].length;
  }
  frag.append(text.slice(last));
  return frag;
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

/* ---------- answer text ---------- */

function cleanLatex(text) {
  return text.replace(/\$([^$]{1,40})\$/g, (_, inner) =>
    inner.replace(/\\[a-z%]+/gi, (cmd) => LATEX_SYMBOLS[cmd] ?? cmd).replace(/\s+/g, " ").trim(),
  );
}

/** Inline formatting: **bold** and [n] citations that point at the numbered source cards. */
function inline(text) {
  const frag = document.createDocumentFragment();
  for (const part of text.split(/(\*\*[^*]+\*\*|\[\d+\])/)) {
    if (!part) continue;
    const cite = /^\[(\d+)\]$/.exec(part);
    if (part.startsWith("**") && part.endsWith("**") && part.length > 4) {
      frag.append(el("strong", {}, part.slice(2, -2)));
    } else if (cite) {
      const n = Number(cite[1]);
      frag.append(
        el("button", { type: "button", class: "cite", "data-n": n, "aria-label": `Passage ${n}` }, String(n)),
      );
    } else {
      frag.append(part);
    }
  }
  return frag;
}

/** Just enough markdown for what the model returns: paragraphs and bullet lists. */
function renderAnswer(text, streaming) {
  const root = $("answer");
  const blocks = cleanLatex(text).trim().split(/\n{2,}/);
  const nodes = [];
  for (const block of blocks) {
    const lines = block.split("\n");
    const isList = lines.every((l) => /^\s*[-*•]\s+/.test(l));
    if (isList) {
      nodes.push(el("ul", {}, ...lines.map((l) => el("li", {}, inline(l.replace(/^\s*[-*•]\s+/, ""))))));
    } else {
      // a list that starts mid-block (intro line, then bullets) is split so it renders as a list
      const firstBullet = lines.findIndex((l) => /^\s*[-*•]\s+/.test(l));
      if (firstBullet > 0) {
        nodes.push(el("p", {}, inline(lines.slice(0, firstBullet).join(" "))));
        nodes.push(el("ul", {}, ...lines.slice(firstBullet).map((l) => el("li", {}, inline(l.replace(/^\s*[-*•]\s+/, ""))))));
      } else {
        nodes.push(el("p", {}, inline(lines.join(" "))));
      }
    }
  }
  root.replaceChildren(...nodes);
  const tail = root.lastElementChild?.lastElementChild ?? root.lastElementChild;
  root.querySelectorAll(".caret").forEach((n) => n.classList.remove("caret"));
  if (streaming && tail) tail.classList.add("caret");
}

/* ---------- sources ---------- */

function renderSources(hits, query) {
  const terms = searchTerms(query);
  const list = $("sources");
  list.replaceChildren(
    ...hits.map((hit, i) => {
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
          text.style.webkitLineClamp = open ? "unset" : "";
          toggle.textContent = open ? "Show less" : "Show full passage";
        });
        card.append(toggle);
      }
      return card;
    }),
  );
}

function lightSource(n, on) {
  $(`source-${n}`)?.classList.toggle("lit", on);
}

function bindCitations() {
  const answer = $("answer");
  answer.addEventListener("mouseover", (e) => {
    const cite = e.target.closest(".cite");
    if (cite) lightSource(cite.dataset.n, true);
  });
  answer.addEventListener("mouseout", (e) => {
    const cite = e.target.closest(".cite");
    if (cite) lightSource(cite.dataset.n, false);
  });
  answer.addEventListener("click", (e) => {
    const cite = e.target.closest(".cite");
    if (!cite) return;
    const card = $(`source-${cite.dataset.n}`);
    card?.scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "center" });
    card?.classList.add("lit");
    setTimeout(() => card?.classList.remove("lit"), 1600);
  });
}

/* ---------- asking ---------- */

function selectedMode() {
  return document.querySelector('input[name="mode"]:checked').value;
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

async function ask(query) {
  const run = ++currentRun;
  controller?.abort();
  controller = new AbortController();
  lastQuery = query;
  sources = [];

  $("results").hidden = false;
  showTab("answer");
  renderPending();
  $("sources").replaceChildren();
  $("answer-foot").textContent = "";
  $("answer").replaceChildren(el("p", { class: "waiting" }, "Retrieving passages…"));
  $("go").disabled = true;
  $("results").scrollIntoView({ behavior: "smooth", block: "start" });

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
          sources = data.sources;
          renderSources(sources, query);
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

let compareFor = ""; // the query the board currently shows

function showTab(name) {
  for (const id of ["answer", "compare"]) {
    const active = id === name;
    $(`tab-${id}`).setAttribute("aria-selected", String(active));
    $(`panel-${id}`).hidden = !active;
  }
  if (name === "compare" && lastQuery && compareFor !== lastQuery) loadCompare(lastQuery);
  if (name === "compare") requestAnimationFrame(drawLines);
}

async function loadCompare(query) {
  compareFor = query;
  const board = $("board");
  board.replaceChildren(
    ...MODES.map((m) => column(m, "searching…", el("p", { class: "col-empty" }, "Searching…"))),
  );

  const results = await Promise.all(MODES.map((m) => searchMode(m, query)));
  if (compareFor !== query) return;

  board.replaceChildren(
    svgEl("svg", { class: "board-lines", "aria-hidden": "true" }),
    ...results.map(({ mode, error, hits, timings_ms }) => {
      if (error) return column(mode, "failed", el("p", { class: "col-empty" }, error));
      const body = hits.length
        ? el("ol", {}, ...hits.map(cell))
        : el("p", { class: "col-empty" }, "No passages matched.");
      return column(mode, ms(timings_ms.total), body);
    }),
  );
  bindBoardHover(board);
  drawLines();
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

/* ---------- wiring ---------- */

$("ask-form").addEventListener("submit", (e) => {
  e.preventDefault();
  const query = $("q").value.trim();
  if (query) ask(query);
});

$("examples").addEventListener("click", (e) => {
  const button = e.target.closest("button");
  if (!button) return;
  $("q").value = button.textContent;
  ask(button.textContent);
});

for (const id of ["answer", "compare"]) $(`tab-${id}`).addEventListener("click", () => showTab(id));

document.addEventListener("keydown", (e) => {
  const typing = /^(input|textarea|select)$/i.test(document.activeElement?.tagName ?? "");
  if (e.key === "/" && !typing && !e.metaKey && !e.ctrlKey) {
    e.preventDefault();
    $("q").focus();
    $("q").select();
  }
});

new ResizeObserver(() => drawLines()).observe($("board"));
bindCitations();
loadStatus();
