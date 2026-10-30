// Complexity Lab client: builds run/measure pickers from /api/catalog, streams from /ws.
"use strict";

const MAX_RUNS = 6;
const MAX_POINTS = 3000;
const $ = (id) => document.getElementById(id);
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const seriesColor = (slot) => css(`--series-${slot + 1}`);

let catalog = null;
let ws = null;
let playing = false;
let runRows = [];      // [{slot, system, pattern}] in the sidebar
let active = null;     // the started session: {runs: [{slot, label, ...}], measures: [...]}
let history = null;    // {t: [], v: {measureKey: [[run0 values], ...]}, w: same shape, true = warming up}
let charts = {};
let dirty = false;

// Viridis, sampled at 9 stops, for continuous states; binary states draw black/white.
const VIRIDIS = ["#440154", "#472d7b", "#3b528b", "#2c728e", "#21918c", "#28ae80", "#5ec962", "#addc30", "#fde725"]
  .map((h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16)));
const LUT = { continuous: new Uint8Array(256 * 3), discrete: new Uint8Array(256 * 3) };
for (let i = 0; i < 256; i++) {
  const x = (i / 255) * (VIRIDIS.length - 1), k = Math.min(Math.floor(x), VIRIDIS.length - 2), f = x - k;
  for (let c = 0; c < 3; c++) {
    LUT.continuous[i * 3 + c] = VIRIDIS[k][c] * (1 - f) + VIRIDIS[k + 1][c] * f;
    LUT.discrete[i * 3 + c] = i > 127 ? 255 : 0;
  }
}

// ---------- sidebar ----------
// Two pick lists, built once: every system · pattern, and every measure. Clicking an item
// toggles it; only classes and border colours change afterwards, so the scroll position
// stays where it is.

function systemByKey(key) { return catalog.systems.find((s) => s.key === key); }

const runKey = (system, pattern) => `${system}:${pattern}`;
const runItems = new Map();      // runKey -> item element
const measureItems = new Map();  // measure key -> item element
let selectedMeasures = new Set();

// An item: a name, an optional muted "(note)" after it, and an optional meta line under it.
function pickItem(title, meta, onClick, note = "") {
  const el = document.createElement("button");
  el.type = "button";
  el.className = "pick";
  el.setAttribute("aria-pressed", "false");
  const name = document.createElement("span");
  name.className = "name"; name.textContent = title;
  if (note) {
    const n = document.createElement("span");
    n.className = "note"; n.textContent = ` (${note})`;
    name.append(n);
  }
  el.append(name);
  if (meta) {
    const sub = document.createElement("span");
    sub.className = "meta"; sub.innerHTML = meta;
    el.append(sub);
  }
  el.onclick = onClick;
  return el;
}

function buildRunList() {
  const box = $("runs");
  for (const sys of catalog.systems) {
    for (const [pattern, full] of Object.entries(sys.patterns)) {
      // "Helicium solidus (rotating)": the note stays in the list, not in run labels
      const [, name, note] = full.match(/^(.*?)(?:\s*\(([^)]*)\))?$/);
      const el = pickItem(`${sys.name} · ${name}`, "", () => toggleRun(sys.key, pattern), note);
      runItems.set(runKey(sys.key, pattern), el);
      box.append(el);
    }
  }
}

function buildMeasureList() {
  const box = $("measures");
  let group = null;
  for (const m of catalog.measures) {
    if (m.group !== group) {  // the catalog lists each group's measures together
      group = m.group;
      const h = document.createElement("div");
      h.className = "pick-group"; h.textContent = group;
      box.append(h);
    }
    const meta = [m.granularity, m.scope, m.temporal].map((t) => `<span class="tag">${t}</span>`).join("") +
      ` ${m.unit}${m.window > 1 ? ` · window ${m.window}` : ""}`;
    const el = pickItem(m.name, meta, () => {
      selectedMeasures.has(m.key) ? selectedMeasures.delete(m.key) : selectedMeasures.add(m.key);
      updateSidebar();
    });
    el.title = m.description;
    measureItems.set(m.key, el);
    box.append(el);
  }
}

function toggleRun(system, pattern) {
  const row = runRows.find((r) => r.system === system && r.pattern === pattern);
  if (row) runRows = runRows.filter((r) => r !== row);
  else if (runRows.length < MAX_RUNS) addRun(system, pattern);
  else setStatus(`up to ${MAX_RUNS} systems: deselect one first`);
  updateSidebar();
}

function addRun(system, pattern) {
  const used = new Set(runRows.map((r) => r.slot));
  const slot = [...Array(MAX_RUNS).keys()].find((s) => !used.has(s));
  runRows.push({ slot, system, pattern });
}

function updateSidebar() {
  const slots = new Map(runRows.map((r) => [runKey(r.system, r.pattern), r.slot]));
  const full = runRows.length >= MAX_RUNS;
  for (const [key, el] of runItems) {
    const on = slots.has(key);
    el.classList.toggle("selected", on);
    el.classList.toggle("full", full && !on);
    el.setAttribute("aria-pressed", on);
    el.style.borderColor = on ? seriesColor(slots.get(key)) : "";
  }
  for (const m of catalog.measures) {
    const el = measureItems.get(m.key), on = selectedMeasures.has(m.key);
    const applies = runRows.length === 0 || runRows.some((r) => m.accepts.includes(r.system));
    el.classList.toggle("selected", on);
    el.classList.toggle("na", !applies);
    el.setAttribute("aria-pressed", on);
    el.title = m.description + (applies ? "" : "\nNot applicable to the selected systems.");
  }
}

// Off by default: LZMA compresses tighter than PNG and H.264 on every run measured so far.
const UNTICKED = new Set(["png", "video_h264"]);

// ---------- session ----------

// Each start gets a new id, echoed by the server on its init and ticks. Ticks of the old
// session can still arrive after a restart (the server may be mid-tick), so messages with
// another id are dropped, and `active` keeps describing the old session until init.
let sessionId = 0;
let pending = null;  // {measures, slots} of the session being started

function start() {
  const size = parseInt($("size").value, 10) || 128;
  if (!runRows.length) return setStatus("select at least one system");
  const measures = catalog.measures.map((m) => m.key).filter((k) => selectedMeasures.has(k));
  if (!measures.length) return setStatus("select at least one measure");
  sessionId++;
  pending = { measures, slots: runRows.map((r) => r.slot) };
  ws.send(JSON.stringify({
    type: "start", id: sessionId, measures,
    runs: runRows.map((r) => ({ system: r.system, pattern: r.pattern, size: [size, size] })),
  }));
}

function onInit(msg) {
  // Labels under the views, in the table and the charts drop "(…)" descriptions.
  const runs = msg.runs.map((r, i) => ({ ...r, label: r.label.replace(/\s*\([^)]*\)/g, ""), slot: pending.slots[i] }));
  active = { runs, measures: pending.measures.filter((k) => runs.some((r) => r.measures.includes(k))) };
  const perRun = () => Object.fromEntries(active.measures.map((k) => [k, active.runs.map(() => [])]));
  history = { t: [], v: perRun(), w: perRun() };
  buildViews();
  buildCharts();
  setPlaying(true);
  $("empty").style.display = "none";
}

// Frames per second actually received, over the last second of ticks.
let tickTimes = [];
function updateFps() {
  const now = performance.now();
  tickTimes = tickTimes.filter((t) => now - t < 1000);
  $("fps-now").textContent = `${playing ? tickTimes.length : 0} FPS`;
}

function onTick(msg) {
  tickTimes.push(performance.now());
  updateFps();
  history.t.push(msg.t);
  msg.runs.forEach((r, i) => {
    for (const k of active.measures) {
      history.v[k][i].push(k in r.values ? r.values[k] : null);
      history.w[k][i].push(r.warming_up.includes(k));
    }
    drawFrame(i, r.frame);
  });
  if (history.t.length > MAX_POINTS) {
    history.t.shift();
    for (const k of active.measures) for (const s of [...history.v[k], ...history.w[k]]) s.shift();
  }
  renderTable(msg);
  dirty = true;
}

function setPlaying(p) {
  playing = p;
  $("play").textContent = p ? "Pause" : "Play";
  $("play").disabled = $("step").disabled = !active;
  $("step").disabled = p || !active;
}

// ---------- views ----------

function buildViews() {
  const box = $("views");
  box.innerHTML = "";
  active.runs.forEach((r, i) => {
    const el = document.createElement("div");
    el.className = "view";
    const c = document.createElement("canvas");
    c.width = r.width; c.height = r.height;
    const scale = Math.max(1, Math.floor(256 / Math.max(r.width, r.height)));
    c.style.width = `${r.width * scale}px`; c.style.height = `${r.height * scale}px`;
    el.innerHTML = `<div class="cap"><span class="swatch" style="background:${seriesColor(r.slot)}"></span>
      <span>${r.label}</span></div>`;
    el.prepend(c);
    box.append(el);
    r.ctx = c.getContext("2d");
    r.image = r.ctx.createImageData(r.width, r.height);
    r.lut = r.n_states === 2 ? LUT.discrete : LUT.continuous;
  });
}

function drawFrame(i, b64) {
  const r = active.runs[i];
  const raw = atob(b64), px = r.image.data, lut = r.lut;
  for (let p = 0; p < raw.length; p++) {
    const v = raw.charCodeAt(p) * 3;
    px[p * 4] = lut[v]; px[p * 4 + 1] = lut[v + 1]; px[p * 4 + 2] = lut[v + 2]; px[p * 4 + 3] = 255;
  }
  r.ctx.putImageData(r.image, 0, 0);
}

function renderTable(msg) {
  const ms = catalog.measures.filter((m) => active.measures.includes(m.key));
  const head = `<tr><th>Measure</th>${active.runs.map((r) =>
    `<th><span class="swatch" style="display:inline-block;background:${seriesColor(r.slot)}"></span> ${r.label}</th>`).join("")}</tr>`;
  const rows = ms.map((m) => `<tr><td>${m.name} <span class="ms">${m.unit}</span></td>` +
    active.runs.map((r, i) => {
      const v = msg.runs[i].values[m.key];
      return v === undefined ? `<td title="not applicable">n/a</td>`
        : msg.runs[i].warming_up.includes(m.key)
        ? `<td class="warming" title="window still filling: ${m.window} frames needed">${v.toFixed(4)}<span class="ms">${msg.runs[i].ms[m.key].toFixed(0)} ms</span></td>`
        : `<td>${v.toFixed(4)}<span class="ms">${msg.runs[i].ms[m.key].toFixed(0)} ms</span></td>`;
    }).join("") + "</tr>").join("");
  $("table-wrap").innerHTML = `<table>${head}${rows}</table>`;
}

// ---------- charts ----------

function niceTicks(lo, hi, n = 4) {
  const span = hi - lo || 1, step0 = span / n, mag = 10 ** Math.floor(Math.log10(step0));
  const step = [1, 2, 2.5, 5, 10].map((s) => s * mag).find((s) => s >= step0);
  const ticks = [];
  for (let v = Math.floor(lo / step) * step; v < hi + step - 1e-12; v += step) ticks.push(+v.toPrecision(12));
  return ticks;
}

function buildCharts() {
  const box = $("charts");
  box.innerHTML = "";
  charts = {};
  for (const key of active.measures) {
    const m = catalog.measures.find((x) => x.key === key);
    const el = document.createElement("div");
    el.className = "chart";
    el.innerHTML = `<h3>${m.name}</h3><span class="unit">${m.unit} vs. time step</span>`;
    el.querySelector("h3").title = m.description;  // as in the measure list
    const c = document.createElement("canvas");
    const tip = document.createElement("div");
    tip.className = "tooltip";
    el.append(c);
    if (active.runs.length > 1) {
      const lg = document.createElement("div");
      lg.className = "legend";
      lg.innerHTML = active.runs.map((r) => `<span><i style="background:${seriesColor(r.slot)}"></i>${r.label}</span>`).join("");
      el.append(lg);
    }
    el.append(tip);
    box.append(el);
    const chart = { key, canvas: c, tip, hover: null };
    c.addEventListener("mousemove", (e) => { chart.hover = e.offsetX; dirty = true; });
    c.addEventListener("mouseleave", () => { chart.hover = null; tip.style.display = "none"; dirty = true; });
    charts[key] = chart;
  }
}

function drawChart(chart) {
  const c = chart.canvas, dpr = window.devicePixelRatio || 1;
  const W = c.clientWidth, H = c.clientHeight;
  if (c.width !== W * dpr || c.height !== H * dpr) { c.width = W * dpr; c.height = H * dpr; }
  const g = c.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, W, H);
  const ts = history.t, series = history.v[chart.key], warming = history.w[chart.key];
  if (!ts.length) return;
  const pad = { l: 46, r: 10, t: 8, b: 22 };
  let lo = Infinity, hi = -Infinity;
  for (const s of series) for (const v of s) if (v !== null) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
  if (!isFinite(lo)) return;
  if (hi - lo < 1e-9) { lo -= 0.05; hi += 0.05; }
  const yt = niceTicks(lo, hi);
  lo = Math.min(lo, yt[0]); hi = Math.max(hi, yt[yt.length - 1]);
  const t0 = ts[0], t1 = Math.max(ts[ts.length - 1], t0 + 1);
  const X = (t) => pad.l + ((t - t0) / (t1 - t0)) * (W - pad.l - pad.r);
  const Y = (v) => H - pad.b - ((v - lo) / (hi - lo)) * (H - pad.t - pad.b);

  g.font = "11px system-ui, sans-serif";
  g.fillStyle = css("--text-muted");
  g.strokeStyle = css("--grid");
  g.lineWidth = 1;
  g.textAlign = "right"; g.textBaseline = "middle";
  for (const v of yt) {
    const y = Math.round(Y(v)) + 0.5;
    g.beginPath(); g.moveTo(pad.l, y); g.lineTo(W - pad.r, y); g.stroke();
    g.fillText(+v.toPrecision(3), pad.l - 6, y);
  }
  g.textAlign = "center"; g.textBaseline = "top";
  for (const t of niceTicks(t0, t1, 5)) if (t >= t0 && t <= t1) g.fillText(t, X(t), H - pad.b + 6);

  g.lineWidth = 2; g.lineJoin = g.lineCap = "round";
  series.forEach((s, i) => {
    // Warm-up values (window still filling) are drawn faint: they cover fewer frames.
    g.strokeStyle = seriesColor(active.runs[i].slot);
    for (const faint of [true, false]) {
      g.globalAlpha = faint ? 0.3 : 1;
      g.beginPath();
      for (let k = 1; k < s.length; k++) {
        if (s[k] === null || s[k - 1] === null || warming[i][k] !== faint) continue;
        g.moveTo(X(ts[k - 1]), Y(s[k - 1])); g.lineTo(X(ts[k]), Y(s[k]));
      }
      g.stroke();
    }
    g.globalAlpha = 1;
  });

  if (chart.hover !== null && chart.hover >= pad.l) {
    const tHover = t0 + ((chart.hover - pad.l) / (W - pad.l - pad.r)) * (t1 - t0);
    let k = 0;
    for (let j = 1; j < ts.length; j++) if (Math.abs(ts[j] - tHover) < Math.abs(ts[k] - tHover)) k = j;
    const x = X(ts[k]);
    g.strokeStyle = css("--text-muted"); g.lineWidth = 1;
    g.beginPath(); g.moveTo(x, pad.t); g.lineTo(x, H - pad.b); g.stroke();
    const rows = [];
    series.forEach((s, i) => {
      if (s[k] === null) return;
      const col = seriesColor(active.runs[i].slot);
      g.fillStyle = col; g.strokeStyle = css("--surface-1"); g.lineWidth = 2;
      g.beginPath(); g.arc(x, Y(s[k]), 4, 0, 2 * Math.PI); g.fill(); g.stroke();
      rows.push(`<div class="row"><span class="swatch" style="background:${col}"></span>${active.runs[i].label}<b>${s[k].toFixed(4)}</b></div>`);
    });
    chart.tip.innerHTML = `<div style="color:var(--text-secondary)">t = ${ts[k]}</div>${rows.join("")}`;
    chart.tip.style.display = "block";
    const box = chart.canvas.getBoundingClientRect(), parent = chart.canvas.parentElement.getBoundingClientRect();
    const left = box.left - parent.left + x + 12;
    chart.tip.style.left = `${Math.min(left, parent.width - chart.tip.offsetWidth - 8)}px`;
    chart.tip.style.top = `${box.top - parent.top + 8}px`;
  }
}

function frameLoop() {
  requestAnimationFrame(frameLoop);  // first, so an error in one draw cannot stop the loop
  if (dirty && history) { dirty = false; for (const ch of Object.values(charts)) drawChart(ch); }
}

// ---------- wiring ----------

function setStatus(s) { $("status").textContent = s; }

function connect() {
  ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
  ws.onopen = () => { setStatus("connected"); sendSpeed(); };
  ws.onclose = () => { setStatus("disconnected — retrying"); setPlaying(false); setTimeout(connect, 1000); };
  ws.onmessage = (e) => {
    const msg = JSON.parse(e.data);
    if ((msg.type === "init" || msg.type === "tick") && msg.id !== sessionId) return;  // stale
    if (msg.type === "init") onInit(msg);
    else if (msg.type === "tick") onTick(msg);
    else if (msg.type === "error") setStatus(msg.message);
  };
}

// The speed sliders pick an index into these.
const STEPS_PER_FRAME = [1, 2, 4, 8, 16, 32, 64];
const MAX_FPS = [1, 2, 4, 8, 15, 30, 60];

function sendSpeed() {
  const steps = STEPS_PER_FRAME[+$("spt").value], fps = MAX_FPS[+$("fps").value];
  $("spt-out").textContent = steps;
  $("fps-out").textContent = fps;
  ws && ws.readyState === 1 && ws.send(JSON.stringify({ type: "speed", steps_per_tick: steps, fps }));
}

function setPanel(open) {
  document.querySelector(".layout").classList.toggle("collapsed", !open);
  const b = $("toggle-panel"), label = open ? "Hide the controls" : "Show the controls";
  b.title = label; b.setAttribute("aria-label", label); b.setAttribute("aria-expanded", open);
  try { localStorage.setItem("panel", open ? "open" : "closed"); } catch {}
  dirty = true;  // charts take the new width on their next draw
}

async function main() {
  catalog = await (await fetch("/api/catalog")).json();
  const dev = catalog.device;
  $("device").innerHTML = dev.backend === "cpu" ? "<b>CPU</b>" : `<b>${dev.backend.toUpperCase()}</b> · ${dev.name}`;
  let panelOpen = true;
  try { panelOpen = localStorage.getItem("panel") !== "closed"; } catch {}
  setPanel(panelOpen);
  $("toggle-panel").onclick = () => setPanel(document.querySelector(".layout").classList.contains("collapsed"));
  addEventListener("resize", () => { dirty = true; });
  for (const [system, pattern] of [
    ["lenia", "paraptera_arcus_labens"], ["lenia", "paraptera_cavus_furiosus"],
    ["lenia", "tetralapillium_metamorpha"], ["baseline", "iid_noise_continuous"],
  ]) addRun(system, pattern);
  selectedMeasures = new Set(catalog.measures.map((m) => m.key).filter((k) => !UNTICKED.has(k)));
  buildRunList();
  buildMeasureList();
  updateSidebar();
  $("clear-runs").onclick = () => { runRows = []; updateSidebar(); };
  $("clear-measures").onclick = () => { selectedMeasures.clear(); updateSidebar(); };
  $("start").onclick = () => { start(); $("start").textContent = "Restart"; };
  $("play").onclick = () => { ws.send(JSON.stringify({ type: playing ? "pause" : "play" })); setPlaying(!playing); };
  $("step").onclick = () => ws.send(JSON.stringify({ type: "step" }));
  $("spt").oninput = sendSpeed;
  $("fps").oninput = sendSpeed;
  connect();
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { updateSidebar(); active && (buildViews(), buildCharts()); dirty = true; });
  requestAnimationFrame(frameLoop);
  setInterval(updateFps, 250);
}

main();
