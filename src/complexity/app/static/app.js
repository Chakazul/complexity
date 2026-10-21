// Complexity Lab client: builds run/measure pickers from /api/catalog, streams from /ws.
"use strict";

const MAX_RUNS = 4;
const MAX_POINTS = 3000;
const $ = (id) => document.getElementById(id);
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const seriesColor = (slot) => css(`--series-${slot + 1}`);

let catalog = null;
let ws = null;
let playing = false;
let runRows = [];      // [{slot, system, pattern}] in the sidebar
let active = null;     // the started session: {runs: [{slot, label, ...}], measures: [...]}
let history = null;    // {t: [], v: {measureKey: [[run0 values], [run1 values], ...]}}
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

function systemByKey(key) { return catalog.systems.find((s) => s.key === key); }

function renderRuns() {
  const box = $("runs");
  box.innerHTML = "";
  for (const row of runRows) {
    const el = document.createElement("div");
    el.className = "run-row";
    const sys = document.createElement("select");
    for (const s of catalog.systems) sys.add(new Option(s.name, s.key, false, s.key === row.system));
    const pat = document.createElement("select");
    for (const [k, name] of Object.entries(systemByKey(row.system).patterns)) pat.add(new Option(name, k, false, k === row.pattern));
    sys.onchange = () => { row.system = sys.value; row.pattern = Object.keys(systemByKey(row.system).patterns)[0]; renderRuns(); };
    pat.onchange = () => { row.pattern = pat.value; };
    const rm = document.createElement("button");
    rm.className = "remove"; rm.textContent = "×"; rm.title = "Remove run";
    rm.disabled = runRows.length === 1;
    rm.onclick = () => { runRows = runRows.filter((r) => r !== row); renderRuns(); };
    const sw = document.createElement("span");
    sw.className = "swatch"; sw.style.background = seriesColor(row.slot);
    el.append(sw, sys, pat, rm);
    box.append(el);
  }
  $("add-run").disabled = runRows.length >= MAX_RUNS;
  renderMeasures();
}

function addRun(system, pattern) {
  const used = new Set(runRows.map((r) => r.slot));
  const slot = [0, 1, 2, 3].find((s) => !used.has(s));
  runRows.push({ slot, system, pattern });
}

function renderMeasures() {
  const box = $("measures");
  const checked = new Set([...box.querySelectorAll("input:checked")].map((i) => i.value));
  const first = box.childElementCount === 0;
  box.innerHTML = "";
  for (const m of catalog.measures) {
    const applies = runRows.some((r) => m.accepts.includes(r.system));
    const el = document.createElement("label");
    el.className = "measure" + (applies ? "" : " na");
    el.title = m.description + (applies ? "" : "\nNot applicable to the selected systems.");
    const cb = document.createElement("input");
    cb.type = "checkbox"; cb.value = m.key;
    cb.checked = first ? true : checked.has(m.key);
    cb.disabled = !applies;
    const name = document.createElement("span");
    name.className = "name"; name.textContent = m.name;
    const meta = document.createElement("span");
    meta.className = "meta";
    meta.innerHTML = [m.granularity, m.scope, m.temporal].map((t) => `<span class="tag">${t}</span>`).join("") +
      ` ${m.unit}${m.window > 1 ? ` · window ${m.window}` : ""}`;
    el.append(cb, name, meta);
    box.append(el);
  }
}

// ---------- session ----------

function start() {
  const size = parseInt($("size").value, 10) || 128;
  const measures = [...$("measures").querySelectorAll("input:checked")].map((i) => i.value);
  if (!measures.length) return setStatus("select at least one measure");
  pendingSlots = runRows.map((r) => r.slot);
  ws.send(JSON.stringify({
    type: "start", measures,
    runs: runRows.map((r) => ({ system: r.system, pattern: r.pattern, size: [size, size] })),
  }));
  active = { measures, runs: [] };
}
let pendingSlots = [];

function onInit(msg) {
  active.runs = msg.runs.map((r, i) => ({ ...r, slot: pendingSlots[i] }));
  active.measures = active.measures.filter((k) => active.runs.some((r) => r.measures.includes(k)));
  history = { t: [], v: Object.fromEntries(active.measures.map((k) => [k, active.runs.map(() => [])])) };
  buildViews();
  buildCharts();
  setPlaying(true);
  $("empty").style.display = "none";
}

function onTick(msg) {
  history.t.push(msg.t);
  msg.runs.forEach((r, i) => {
    for (const k of active.measures) history.v[k][i].push(k in r.values ? r.values[k] : null);
    drawFrame(i, r.frame);
  });
  if (history.t.length > MAX_POINTS) {
    history.t.shift();
    for (const k of active.measures) for (const s of history.v[k]) s.shift();
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
      <span>${r.label}</span><span class="t" id="t${i}">t = 0</span></div>`;
    el.prepend(c);
    box.append(el);
    r.ctx = c.getContext("2d");
    r.image = r.ctx.createImageData(r.width, r.height);
    r.lut = LUT[systemByKey(r.system).state_type] || LUT.continuous;
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
  $(`t${i}`).textContent = `t = ${history.t[history.t.length - 1]}`;
}

function renderTable(msg) {
  const ms = catalog.measures.filter((m) => active.measures.includes(m.key));
  const head = `<tr><th>Measure</th>${active.runs.map((r) =>
    `<th><span class="swatch" style="display:inline-block;background:${seriesColor(r.slot)}"></span> ${r.label}</th>`).join("")}</tr>`;
  const rows = ms.map((m) => `<tr><td>${m.name} <span class="ms">${m.unit}</span></td>` +
    active.runs.map((r, i) => {
      const v = msg.runs[i].values[m.key];
      return v === undefined ? `<td title="not applicable">n/a</td>`
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
  const ts = history.t, series = history.v[chart.key];
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

  g.lineWidth = 2; g.lineJoin = "round";
  series.forEach((s, i) => {
    g.strokeStyle = seriesColor(active.runs[i].slot);
    g.beginPath();
    let pen = false;
    for (let k = 0; k < s.length; k++) {
      if (s[k] === null) { pen = false; continue; }
      pen ? g.lineTo(X(ts[k]), Y(s[k])) : g.moveTo(X(ts[k]), Y(s[k]));
      pen = true;
    }
    g.stroke();
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
  if (dirty && history) { for (const ch of Object.values(charts)) drawChart(ch); dirty = false; }
  requestAnimationFrame(frameLoop);
}

// ---------- wiring ----------

function setStatus(s) { $("status").textContent = s; }

function connect() {
  ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
  ws.onopen = () => { setStatus("connected"); sendSpeed(); };
  ws.onclose = () => { setStatus("disconnected — retrying"); setPlaying(false); setTimeout(connect, 1000); };
  ws.onmessage = (e) => {
    const msg = JSON.parse(e.data);
    if (msg.type === "init") onInit(msg);
    else if (msg.type === "tick") onTick(msg);
    else if (msg.type === "error") setStatus(msg.message);
  };
}

function sendSpeed() {
  $("spt-out").textContent = $("spt").value;
  $("fps-out").textContent = $("fps").value;
  ws && ws.readyState === 1 && ws.send(JSON.stringify({ type: "speed", steps_per_tick: +$("spt").value, fps: +$("fps").value }));
}

async function main() {
  catalog = await (await fetch("/api/catalog")).json();
  addRun("life", "gosper_glider_gun");
  addRun("lenia", "orbium");
  renderRuns();
  $("add-run").onclick = () => { addRun("life", "random"); renderRuns(); };
  $("start").onclick = () => { start(); $("start").textContent = "Restart"; };
  $("play").onclick = () => { ws.send(JSON.stringify({ type: playing ? "pause" : "play" })); setPlaying(!playing); };
  $("step").onclick = () => ws.send(JSON.stringify({ type: "step" }));
  $("spt").oninput = sendSpeed;
  $("fps").oninput = sendSpeed;
  connect();
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { renderRuns(); active && (buildViews(), buildCharts()); dirty = true; });
  requestAnimationFrame(frameLoop);
}

main();
