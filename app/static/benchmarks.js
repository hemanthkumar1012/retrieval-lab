// The benchmarks view: the measured results from data.js as small interactive charts.
// Everything is plain SVG and CSS, so there is no chart library to load.

import { $, el, reducedMotion, svgEl } from "./dom.js";
import { CACHE, CACHE_DEFAULT, CHUNKING, DENSE_NDCG, METRICS, MODES, PARITY, RUN_NOTE, WEIGHT_SWEEP } from "./data.js";

const fmt3 = (v) => v.toFixed(3);
let drawn = false;

/* ---------- retrieval modes: horizontal bars with a metric switch ---------- */

function renderModeBars(metric) {
  const top = Math.max(...MODES.map((m) => m[metric.key]));
  const rows = MODES.map((m) => {
    const bar = el("div", { class: "bar", style: `--c:${m.color}` });
    // set the width after insertion so the change animates from the previous metric
    requestAnimationFrame(() => (bar.style.width = `${(m[metric.key] / 0.9) * 100}%`));
    return el(
      "li",
      { class: m[metric.key] === top ? "best" : "" },
      el("span", { class: "bar-name" }, m.name),
      el("div", { class: "bar-track" }, bar),
      el("b", { class: "bar-value" }, fmt3(m[metric.key])),
      el("span", { class: "bar-time" }, m.p50 >= 1000 ? `${(m.p50 / 1000).toFixed(1)} s` : `${m.p50} ms`),
    );
  });
  $("mode-bars").replaceChildren(...rows);
}

function initModeBars() {
  const switcher = $("metric-switch");
  METRICS.forEach((metric, i) => {
    const button = el("button", { type: "button", "aria-pressed": String(i === 0) }, metric.label);
    button.addEventListener("click", () => {
      switcher.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", String(b === button)));
      renderModeBars(metric);
    });
    switcher.append(button);
  });
  renderModeBars(METRICS[0]);
}

/* ---------- shared line chart pieces ---------- */

const PAD = { top: 18, right: 22, bottom: 44, left: 52 };
let W = 640;
let H = 280;
let plotW = W - PAD.left - PAD.right;
let plotH = H - PAD.top - PAD.bottom;

/** Charts are drawn at the width they are shown at, so text stays readable on a phone. */
function measure(container) {
  W = Math.max(300, Math.round(container.clientWidth));
  H = Math.round(Math.max(230, Math.min(300, W * 0.45)));
  plotW = W - PAD.left - PAD.right;
  plotH = H - PAD.top - PAD.bottom;
}

function scale(domain, range) {
  const [d0, d1] = domain;
  const [r0, r1] = range;
  return (v) => r0 + ((v - d0) / (d1 - d0)) * (r1 - r0);
}

function pathFrom(points) {
  return points.map(([x, y], i) => `${i ? "L" : "M"}${x.toFixed(1)},${y.toFixed(1)}`).join(" ");
}

function grid(yScale, ticks, format) {
  return svgEl(
    "g",
    { class: "axis" },
    ...ticks.flatMap((t) => [
      svgEl("line", { x1: PAD.left, x2: W - PAD.right, y1: yScale(t), y2: yScale(t) }),
      svgEl("text", { x: PAD.left - 10, y: yScale(t) + 4, "text-anchor": "end" }, format(t)),
    ]),
  );
}

/** A line that draws itself once. Sets the dash length from the real path length. */
function drawLine(path) {
  if (reducedMotion()) return;
  const length = path.getTotalLength();
  path.style.strokeDasharray = String(length);
  path.style.strokeDashoffset = String(length);
  path.getBoundingClientRect(); // commit the start state before transitioning
  path.style.transition = "stroke-dashoffset .9s cubic-bezier(.2,.7,.2,1)";
  path.style.strokeDashoffset = "0";
}

/* ---------- fusion weight sweep ---------- */

function renderWeightChart() {
  const xs = WEIGHT_SWEEP.map((_, i) => PAD.left + (plotW * i) / (WEIGHT_SWEEP.length - 1));
  const y = scale([0.6, 0.72], [PAD.top + plotH, PAD.top]);
  const series = [
    { key: "hybrid", label: "Hybrid", color: "var(--fusion)" },
    { key: "rerank", label: "Hybrid + rerank", color: "var(--rerank)" },
  ];

  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-labelledby": "weight-title" }, svgEl("title", { id: "weight-title" }, "nDCG@10 against the weight of the full-text list"));
  svg.append(grid(y, [0.6, 0.64, 0.68, 0.72], (t) => t.toFixed(2)));

  svg.append(svgEl("line", { class: "baseline", x1: PAD.left, x2: W - PAD.right, y1: y(DENSE_NDCG), y2: y(DENSE_NDCG) }));

  const lines = series.map((s) => {
    const pts = WEIGHT_SWEEP.map((row, i) => [xs[i], y(row[s.key])]);
    const path = svgEl("path", { class: "line", d: pathFrom(pts), style: `stroke:${s.color}` });
    svg.append(path);
    pts.forEach(([cx, cy]) => svg.append(svgEl("circle", { class: "dot", cx, cy, r: 4.5, style: `fill:${s.color}` })));
    return path;
  });

  WEIGHT_SWEEP.forEach((row, i) => svg.append(svgEl("text", { class: "axis-label", x: xs[i], y: H - 16, "text-anchor": "middle" }, String(row.weight))));
  svg.append(svgEl("text", { class: "axis-title", x: PAD.left + plotW / 2, y: H - 1, "text-anchor": "middle" }, "weight of the full-text list in rank fusion"));

  // a hover band per weight, so the readout works with mouse, touch and keyboard focus
  const readout = $("weight-readout");
  const show = (i) => {
    const row = WEIGHT_SWEEP[i];
    readout.textContent = `Weight ${row.weight}: hybrid ${fmt3(row.hybrid)}, hybrid + rerank ${fmt3(row.rerank)}, vector search alone ${fmt3(DENSE_NDCG)}.`;
  };
  WEIGHT_SWEEP.forEach((_, i) => {
    const band = svgEl("rect", { class: "hit", x: xs[i] - plotW / 6, y: PAD.top, width: plotW / 3, height: plotH, tabindex: "0", "aria-label": `Weight ${WEIGHT_SWEEP[i].weight}` });
    band.addEventListener("mouseenter", () => show(i));
    band.addEventListener("focus", () => show(i));
    svg.append(band);
  });
  show(WEIGHT_SWEEP.length - 1);

  $("weight-chart").replaceChildren(svg);
  $("weight-legend").replaceChildren(
    ...series.map((s) => el("li", { style: `--c:${s.color}` }, el("i"), s.label)),
    el("li", { class: "dashed" }, el("i"), "Vector search alone"),
  );
  return lines;
}

/* ---------- semantic cache threshold ---------- */

function renderCacheChart() {
  const xs = CACHE.map((_, i) => PAD.left + (plotW * i) / (CACHE.length - 1));
  const y = scale([0, 100], [PAD.top + plotH, PAD.top]);
  const series = [
    { key: "hit", color: "var(--cache)" },
    { key: "wrong", color: "var(--bad)" },
  ];
  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-labelledby": "cache-title" }, svgEl("title", { id: "cache-title" }, "Cache hits and wrong hits against similarity threshold"));
  svg.append(grid(y, [0, 25, 50, 75, 100], (t) => `${t}%`));

  const lines = series.map((s) => {
    const path = svgEl("path", { class: "line", d: pathFrom(CACHE.map((row, i) => [xs[i], y(row[s.key])])), style: `stroke:${s.color}` });
    svg.append(path);
    return path;
  });
  CACHE.forEach((row, i) => {
    const crowded = W < 480 && i % 2 === 1; // every second label on a narrow screen
    if (!crowded) svg.append(svgEl("text", { class: "axis-label", x: xs[i], y: H - 16, "text-anchor": "middle" }, String(row.threshold)));
  });
  svg.append(svgEl("text", { class: "axis-title", x: PAD.left + plotW / 2, y: H - 1, "text-anchor": "middle" }, "similarity threshold"));

  const marker = svgEl("g", { class: "marker" }, svgEl("line", { y1: PAD.top, y2: PAD.top + plotH }), svgEl("circle", { class: "m-hit", r: 5.5 }), svgEl("circle", { class: "m-wrong", r: 5.5 }));
  svg.append(marker);

  const slider = $("cache-slider");
  slider.max = String(CACHE.length - 1);
  const move = () => {
    const i = Number(slider.value);
    const row = CACHE[i];
    marker.setAttribute("transform", `translate(${xs[i]},0)`);
    marker.querySelector(".m-hit").setAttribute("cy", y(row.hit));
    marker.querySelector(".m-wrong").setAttribute("cy", y(row.wrong));
    $("cache-readout").textContent =
      `At ${row.threshold}${row.threshold === CACHE_DEFAULT ? " (the default)" : ""}, ${row.hit}% of rephrased questions are answered from cache, and ${row.wrong}% of different questions wrongly get a cached answer.`;
  };
  slider.oninput = move;
  if (!drawn) slider.value = String(CACHE.findIndex((r) => r.threshold === CACHE_DEFAULT));
  move();

  $("cache-chart").replaceChildren(svg);
  $("cache-legend").replaceChildren(
    el("li", { style: "--c:var(--cache)" }, el("i"), "Rephrased questions answered from cache"),
    el("li", { style: "--c:var(--bad)" }, el("i"), "Different questions that wrongly hit"),
  );
  return lines;
}

/* ---------- small tables ---------- */

function renderTables() {
  $("parity").replaceChildren(
    el("div", {}, el("b", {}, PARITY.embedderCosine.toFixed(3)), el("span", {}, "mean cosine between the int8 and original embeddings")),
    el("div", {}, el("b", {}, PARITY.rerankerCorrelation.toFixed(3)), el("span", {}, "score correlation of the int8 and original reranker")),
    el("div", {}, el("b", {}, `${PARITY.embedderMb + PARITY.rerankerMb} MB`), el("span", {}, "both models together, which is why no PyTorch is needed")),
  );
  $("chunking").replaceChildren(
    ...CHUNKING.map((c) => el("tr", {}, el("th", { scope: "row" }, c.name), el("td", {}, fmt3(c.dense)), el("td", {}, fmt3(c.rerank)))),
  );
  $("run-note").textContent = `${RUN_NOTE}. Run to run the numbers move by about 0.005.`;
}

/* ---------- entry ---------- */

export function initBenchmarks() {
  initModeBars();
  renderTables();
}

/** Charts are drawn the first time the view is shown, so their line-drawing animation is seen. */
export function showBenchmarks() {
  if (drawn) return;
  renderCharts(true);
  drawn = true;
  let timer = 0;
  addEventListener("resize", () => {
    clearTimeout(timer);
    timer = setTimeout(() => !$("view-benchmarks").hidden && renderCharts(false), 150);
  });
}

function renderCharts(animate) {
  measure($("weight-chart"));
  const lines = [...renderWeightChart(), ...renderCacheChart()];
  if (animate) lines.forEach(drawLine);
}
