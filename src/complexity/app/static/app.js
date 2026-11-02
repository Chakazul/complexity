// Complexity Lab client: builds run/measure pickers from /api/catalog, streams from /ws.
"use strict";

const MAX_RUNS = 8;
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
// Normalized versions (MEASURES.md §5.10): on by default, kept per browser, sent at Start.
let normalized = true;
try { normalized = localStorage.getItem("normalized") !== "off"; } catch {}
// Whether measure m is shown normalized: it has a normalized version and the option is on
// (for a running session, as it was started).
const isNormalized = (m, on = active ? active.normalized : normalized) => !!(on && m.ratio);
const unitOf = (m, on) => (isNormalized(m, on) ? m.ratio_unit : m.unit);
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
    const kind = PEAKS[m.peaks_at];
    const tags = [[kind.tag, kind.tag], [m.granularity, "granularity"], [m.scope, "scope"],
      [m.temporal, "temporal"], ...(m.map ? [["heatmap", "heatmap"]] : [])];
    const meta = tags.map(([t, cls]) => `<span class="tag tag-${cls} tag-v-${t}" title="${TAG_HELP[t] || ""}">${t}</span>`).join("") +
      ` <span class="unit-text" data-key="${m.key}">${unitOf(m, normalized)}</span>`;  // parameters are under the ⚙
    const el = pickItem(`${m.icon} ${m.name}`, meta, () => {
      selectedMeasures.has(m.key) ? selectedMeasures.delete(m.key) : selectedMeasures.add(m.key);
      updateSidebar();
    });
    measureItems.set(m.key, el);
    const row = document.createElement("div");
    row.className = "pick-row";
    const info = document.createElement("button");
    info.type = "button"; info.className = "pick-gear pick-info"; info.textContent = "ⓘ";
    info.setAttribute("aria-label", `About ${m.name}`);
    info.onclick = () => openInfo(m);
    row.append(el, info);
    box.append(row);
    if (Object.keys(m.params).length) buildSettings(m, row);
  }
}

// ---------- measure settings ----------
// A ⚙ beside a measure opens its parameters (window, levels, offset, component, ...).
// Changed values are kept per browser and sent with the next Start; the server checks them.

let measureParams = {};  // key -> {param: value}, only values that differ from the default
try { measureParams = JSON.parse(localStorage.getItem("measureParams") || "{}"); } catch {}

const PARAM_HELP = {
  window: "frames in the sliding window", n_levels: "symbols per cell (empty: auto, the system's states or the measure's default)",
  offset: "distance to the neighbour", component: "which part of the decomposition to report",
  environment: "near: other creatures within the rule's reach of the target; all: everything else",
  target: "typical: the creature closest to the median mass; largest: the most massive",
  saddle: "merge neighbouring potential basins whose saddle is at least this fraction of the lower peak",
  threshold: "the level below which the field is background",
};

function buildSettings(m, row) {
  const gear = document.createElement("button");
  gear.type = "button"; gear.className = "pick-gear"; gear.textContent = "⚙";
  gear.title = "Settings"; gear.setAttribute("aria-label", `Settings of ${m.name}`);
  gear.setAttribute("aria-expanded", "false");
  const panel = document.createElement("div");
  panel.className = "pick-settings"; panel.hidden = true;
  for (const [name, def] of Object.entries(m.params)) {
    const label = document.createElement("label");
    label.textContent = name.replace(/_/g, " ");
    label.title = PARAM_HELP[name] || "";
    let input;
    if (m.choices[name]) {
      input = document.createElement("select");
      for (const c of m.choices[name]) input.add(new Option(c.replace(/_/g, " "), c));
    } else if (typeof def === "boolean") {
      input = document.createElement("input"); input.type = "checkbox";
    } else {
      input = document.createElement("input"); input.type = "number";
      input.step = Number.isInteger(def) || def === null ? "1" : "any";
      if (def === null) input.placeholder = "auto";
    }
    const current = measureParams[m.key]?.[name] ?? def;
    if (input.type === "checkbox") input.checked = current; else input.value = current ?? "";
    input.onchange = () => {
      const v = input.type === "checkbox" ? input.checked
        : input.tagName === "SELECT" ? input.value
        : input.value === "" ? null : Number(input.value);
      const p = (measureParams[m.key] ||= {});
      if (v === def) delete p[name]; else p[name] = v;
      if (!Object.keys(p).length) delete measureParams[m.key];
      try { localStorage.setItem("measureParams", JSON.stringify(measureParams)); } catch {}
      gear.classList.toggle("modified", !!measureParams[m.key]);
    };
    label.append(input);
    panel.append(label);
  }
  const note = document.createElement("span");
  note.className = "hint"; note.textContent = "applies at the next Start";
  panel.append(note);
  gear.classList.toggle("modified", !!measureParams[m.key]);
  gear.onclick = () => {
    panel.hidden = !panel.hidden;
    gear.setAttribute("aria-expanded", !panel.hidden);
  };
  row.append(gear);
  row.after(panel);
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
    // say which selected systems a measure skips, and what it needs
    const skips = [...new Set(runRows.filter((r) => !m.accepts.includes(r.system))
      .map((r) => systemByKey(r.system).name))];
    const needs = m.applies_to ? ` It applies to ${m.applies_to}.` : "";
    el.title = m.description + SUBJECT_NOTE(m) + (skips.length
      ? `\n\n${applies ? "Skipped for" : "Not applicable to"} ${skips.join(", ")}.${needs}` : "");
  }
  if (!active) renderMapControls();  // before a start, the menu follows the selection
}

// Where a measure is highest (MEASURES.md §2.1): randomness measures peak at noise, structural ones
// are low for both order and noise. Logical depth's proxy does neither reliably.
const PEAKS = {
  randomness: { tag: "randomness" },
  structure: { tag: "structural" },
  mixed: { tag: "mixed" },
  none: { tag: "statistic" },
};

// Popups for the tags in the measure list (taxonomy from Complexity_Measure.pdf §1.1-1.2).
const TAG_HELP = {
  randomness: "Peaks at randomness: highest for noise, lowest for uniform or still frames. Order → disorder rises steadily (entropy, compression, chaos).",
  structural: "Peaks at structure: low for both uniform frames and noise, highest in between (moving or large-scale structure).",
  mixed: "Peaks neither at randomness nor reliably at structure; read with care.",
  statistic: "A statistic for reference (mass, speed, change rate), not a complexity measure.",
  pattern: "Pattern-level: measures a particular configuration (frames of this run).",
  rule: "Rule-level: measures the rule itself, from runs of its own; constant for a run.",
  system: "System-level: measures a whole system or family of rules.",
  local: "Local: built from small neighbourhoods of cells, then averaged.",
  global: "Global: computed from the whole frame at once.",
  static: "Static: one frame (or the rule) at a time.",
  dynamic: "Dynamic: uses changes over time (a window of frames, or runs of the system).",
  heatmap: "Has a spatial map: choose it under Heatmap (Run section) to see where the value comes from.",
};

// ---------- measure info popup ----------
// ⓘ beside a measure (or a chart's title) opens its description, equation and facts.
// Equations are LaTeX, rendered by KaTeX, loaded from a CDN on the first popup; without
// it (offline) the LaTeX source is shown instead.

const KATEX = "https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/";
let katexLoading = null;
function loadKatex() {
  if (window.katex) return Promise.resolve(true);
  return katexLoading ||= new Promise((resolve) => {
    const css = document.createElement("link");
    css.rel = "stylesheet"; css.href = KATEX + "katex.min.css";
    const js = document.createElement("script");
    js.src = KATEX + "katex.min.js";
    js.onload = () => resolve(true);
    js.onerror = () => { katexLoading = null; resolve(false); };
    document.head.append(css, js);
  });
}

function renderTex(el, tex) {
  el.textContent = tex;  // the source until KaTeX is there
  el.classList.add("tex-source");
  loadKatex().then((ok) => {
    if (!ok || el.dataset.tex !== tex) return;  // offline, or another measure opened since
    katex.render(tex, el, { displayMode: true, throwOnError: false });
    el.classList.remove("tex-source");
  });
  el.dataset.tex = tex;
}

const NORMALIZED_TEX = {
  mass: String.raw`\frac{(f - f_\varnothing)\,HW}{m}`,
  change: String.raw`\frac{(f - f_\varnothing)\,HW}{r}`,
};

function openInfo(m) {
  const kind = PEAKS[m.peaks_at];
  $("info-title").textContent = `${m.icon} ${m.name}`;
  $("info-tags").innerHTML = [[kind.tag, kind.tag], [m.granularity, "granularity"], [m.scope, "scope"],
    [m.temporal, "temporal"], ...(m.map ? [["heatmap", "heatmap"]] : [])]
    .map(([t, cls]) => `<span class="tag tag-${cls} tag-v-${t}" title="${TAG_HELP[t] || ""}">${t}</span>`).join("");
  renderTex($("info-equation"), m.equation);
  $("info-description").textContent = m.description;
  const facts = [
    ["Unit", m.unit],
    ["Peaks", TAG_HELP[kind.tag]],
    ["Level", `${TAG_HELP[m.granularity]} ${TAG_HELP[m.scope]} ${TAG_HELP[m.temporal]}`],
    ["Computed here", SUBJECT_NOTE(m).trim().replace(/^Computed here: /, "")],
    m.window > 1 && ["Window", `${m.window} frames`],
    m.map && ["Heatmap", m.map === "tile" ? "per tile of the frame" : "one value per cell"],
    m.applies_to && ["Applies to", m.applies_to],
    m.reference && ["Reference", m.reference],
  ].filter(Boolean);
  const dl = $("info-facts");
  dl.innerHTML = "";
  for (const [k, v] of facts) {
    const dt = document.createElement("dt"), dd = document.createElement("dd");
    dt.textContent = k; dd.textContent = v;
    dl.append(dt, dd);
  }
  if (m.ratio) {  // the normalized version, with its formula
    const dt = document.createElement("dt"), dd = document.createElement("dd"), eq = document.createElement("div");
    dt.textContent = "Normalized";
    dd.textContent = RATIO_NOTE(m).trim().replace(/^Normalized \(option beside the Measures title\): /, "") +
      " f: the grid average, f∅: its value on empty frames, H×W: the grid, " +
      (m.ratio === "mass" ? "m: the mass." : "r: the change rate Σ|ΔA| per step.");
    eq.className = "info-equation small";
    dd.prepend(eq);
    dl.append(dt, dd);
    renderTex(eq, NORMALIZED_TEX[m.ratio]);
  }
  $("info").showModal();
  $("info").scrollTop = 0;  // showModal focuses Close, which would scroll a tall popup to its end
}

// ---------- tags popup ----------
// Clicking any tag (in the measure list or a measure's popup) lists all tags with their
// meanings, the clicked one highlighted.

const TAG_GROUPS = [
  ["Where it peaks", [["randomness", "randomness"], ["structural", "structural"], ["mixed", "mixed"], ["statistic", "statistic"]]],
  ["Granularity", [["pattern", "granularity"], ["rule", "granularity"], ["system", "granularity"]]],
  ["Scope", [["local", "scope"], ["global", "scope"]]],
  ["Time", [["static", "temporal"], ["dynamic", "temporal"]]],
  ["Map", [["heatmap", "heatmap"]]],
];

function openTags(clicked) {
  const box = $("tags-list");
  box.innerHTML = "";
  for (const [title, tags] of TAG_GROUPS) {
    const h = document.createElement("h3");
    h.className = "tags-group"; h.textContent = title;
    const dl = document.createElement("dl");
    dl.className = "info-facts tags-facts";
    for (const [t, cls] of tags) {
      const dt = document.createElement("dt"), dd = document.createElement("dd");
      dt.innerHTML = `<span class="tag tag-${cls} tag-v-${t}">${t}</span>`;
      dd.textContent = TAG_HELP[t] || "";
      if (t === clicked) { dt.classList.add("current"); dd.classList.add("current"); }
      dl.append(dt, dd);
    }
    box.append(h, dl);
  }
  $("tags").showModal();
  $("tags").scrollTop = 0;
}

// Any tag opens the tags popup instead of toggling the measure it sits in.
function tagClicks(container) {
  container.addEventListener("click", (e) => {
    const tag = e.target.closest(".tag");
    if (!tag) return;
    e.stopPropagation(); e.preventDefault();
    openTags(tag.textContent);
  }, true);  // capture: before the measure's own click handler
}

// Close by the button, Esc (native) or a click on the backdrop.
function closable(dialog, button) {
  button.onclick = () => dialog.close();
  dialog.onclick = (e) => { if (e.target === dialog) dialog.close(); };
}

// The normalized version, for measures that have one (MEASURES.md §5.10).
const RATIO_NOTE = (m) => !m.ratio ? "" : `\n\nNormalized (option beside the Measures title): the grid's total ${
  { mass: "above an empty frame, divided by the mass", change: "per step, divided by the change rate Σ|ΔA| per step" }[m.ratio]
}, in ${m.ratio_unit}.`;

// What this implementation computes, below the description (MEASURES.md §2.3).
const SUBJECT_NOTE = (m) => `\n\nComputed here: a property of ${
  { instance: "one frame", trajectory: "a stretch of the run", rule: "the rule", ensemble: "an ensemble" }[m.subject]
}, estimated from ${{ object: "that frame", run: "this run", ensemble: "an ensemble of runs" }[m.estimated_from]}.`;

// Ticked at start (Bert, 2026-11-02); the others are a click away.
const DEFAULT_MEASURES = new Set([
  "lzma", "video_lzma", "novelty", "statistical_complexity", "transfer_entropy",
  "iti_features", "lyapunov",
]);

// Presets: a few patterns of one system, with noise of its kind and other references, and
// the measures that are informative for it (MEASURES.md §3, §7). Measures a system does not
// accept are skipped as usual; rule-level ones take seconds on the first tick.
const PRESETS = {
  lenia: {
    name: "Lenia", size: 128,
    runs: [["lenia", "orbium"], ["lenia", "gyrorbium_gyrans"], ["lenia", "paraptera_arcus_pedes"],
      ["lenia", "paraptera_cavus_furiosus"], ["lenia", "tetralapillium_metamorpha"],
      ["lenia", "asterium_vagus"], ["lenia", "random"], ["baseline", "iid_noise_continuous"]],
    // continuous: no transient length or λ; compression and novelty for form, SC, TE, FID
    // and ITI for structure, Lyapunov for the chaotic creatures, speed for gliders
    measures: ["lzma", "video_lzma", "novelty", "statistical_complexity", "transfer_entropy",
      "fid", "iti_features", "lyapunov", "mass", "speed"],
  },
  integer_lenia: {
    name: "Integer Lenia", size: 128,
    runs: [["integer_lenia", "min_orbium"], ["integer_lenia", "random"], ["lenia", "orbium"],
      ["baseline", "iid_noise_13"], ["baseline", "frozen_noise_13"]],
    // 13 exact levels: the discrete measures apply too (block entropy, transient length, λ)
    measures: ["lzma", "video_lzma", "block_entropy", "statistical_complexity",
      "transfer_entropy", "iti_features", "lyapunov", "transient_length", "langton_lambda",
      "mass", "speed"],
  },
  lenia_multi: {
    name: "Multi-channel Lenia", size: 256,
    runs: [["lenia_multi", "aquarium"], ["lenia_multi", "aquarium_unit"],
      ["lenia_multi", "aquarium_scene"], ["lenia", "orbium"], ["baseline", "iid_noise_continuous"]],
    // networks of units: per-object views (ITI features), structure (SC, TE), mass for growth
    measures: ["lzma", "video_lzma", "novelty", "statistical_complexity", "transfer_entropy",
      "iti_features", "lyapunov", "mass"],
  },
  life: {
    name: "Game of Life", size: 128,
    runs: [["life", "glider"], ["life", "gosper_glider_gun"], ["life", "galaxy"],
      ["life", "r_pentomino"], ["life", "pufferfish"], ["life", "random"],
      ["baseline", "iid_noise_binary"], ["baseline", "frozen_noise_binary"]],
    // binary: LZC and block entropy are exact; PID and FID reproduce the papers' Life values;
    // transient length (linear) and λ (0.273) are the rule's
    measures: ["lzma", "video_lzma", "lzc", "block_entropy", "statistical_complexity",
      "transfer_entropy", "pid", "fid", "iti_features", "lyapunov", "transient_length",
      "langton_lambda"],
  },
  eca: {
    name: "Elementary CA", size: 128,
    runs: [["eca", "rule110"], ["eca", "rule54"], ["eca", "rule30"], ["eca", "rule184"],
      ["eca", "rule232"], ["eca", "rule160"], ["eca", "rule90_seed"], ["baseline", "iid_noise_1d_binary"]],
    // frames are space-time diagrams, compared with scrolling 1-D noise; image measures and
    // local fields read them; video,
    // speed and change rate see the scroll, so they are left out; the rule-level ones
    // (λ, Z, depth, transient class) separate the Wolfram classes
    measures: ["lzma", "lzc", "block_entropy", "statistical_complexity", "transfer_entropy",
      "fid", "lyapunov", "langton_lambda", "wuensche_z", "thermodynamic_depth",
      "transient_length"],
  },
};

function applyPreset(key) {
  const preset = PRESETS[key];
  if (!preset) return;
  runRows = [];
  for (const [system, pattern] of preset.runs) addRun(system, pattern);
  const known = new Set(catalog.measures.map((m) => m.key));
  selectedMeasures = new Set(preset.measures.filter((k) => known.has(k)));
  $("size").value = String(preset.size);
  updateSidebar();
  setStatus(`preset ${preset.name}: press Start`);
}

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
  // no measures: run the simulations only
  sessionId++;
  pending = { measures, slots: runRows.map((r) => r.slot), normalized };
  ws.send(JSON.stringify({
    type: "start", id: sessionId, measures,
    measure_params: Object.fromEntries(measures.filter((k) => measureParams[k]).map((k) => {
      const known = catalog.measures.find((m) => m.key === k).params;  // drop stale saved names
      return [k, Object.fromEntries(Object.entries(measureParams[k]).filter(([n]) => n in known))];
    })),
    runs: runRows.map((r) => ({ system: r.system, pattern: r.pattern, size: [size, size] })),
    normalized,
  }));
}

// ---------- heatmaps (MEASURES.md §4) ----------
// One measure's spatial map per run, beside its frame: an orange ramp from the surface
// colour at 0. The scale is shared across runs: 0 to the largest value this tick;
// negatives show as 0. (An overlay on the frame was tried and dropped, 2026-10-30.)

let mapKey = "";
try { mapKey = localStorage.getItem("mapKey") || ""; } catch {}

// The maps that are not measures (neighbour sum, objects), then the chosen measures' maps.
function mapMeasures() {
  const chosen = active ? active.measures : catalog.measures.map((m) => m.key).filter((k) => selectedMeasures.has(k));
  return [...catalog.maps.map((m) => ({ ...m, map: "special" })),
    ...catalog.measures.filter((m) => m.map && chosen.includes(m.key))];
}
const mapInfo = (key) => mapMeasures().find((m) => m.key === key) || catalog.measures.find((m) => m.key === key);

function renderMapControls() {
  const sel = $("map-measure"), options = mapMeasures();
  if (mapKey && !options.some((m) => m.key === mapKey)) setMap("");
  sel.innerHTML = `<option value="">Off</option>` +
    options.map((m) => `<option value="${m.key}"${m.key === mapKey ? " selected" : ""}>${m.icon ? m.icon + " " : ""}${m.name}${m.map === "tile" ? " (tiles)" : ""}</option>`).join("");
  $("views").classList.toggle("heat-on", !!mapKey);
  if (!mapKey) $("heat-legend").hidden = true;
}

function setMap(key) {
  mapKey = key;
  try { localStorage.setItem("mapKey", key); } catch {}
  sendMap();
}

function sendMap() {
  ws && ws.readyState === 1 && ws.send(JSON.stringify({ type: "map", measure: mapKey || null }));
}

function heatRamp() {
  const hex = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));
  return { lo: hex(css("--surface-1")), mid: hex(css("--heat-mid")), hi: hex(css("--heat-hi")) };
}

function decodeMap(m) {
  const bin = atob(m.data), bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Float32Array(bytes.buffer);
}

// The objects map: 0 background, 1 the target (one colour), 2, 3, ... the other creatures
// (a muted palette, cycled).
function drawObjects(maps) {
  const hex = (h) => [1, 3, 5].map((i) => parseInt(h.trim().slice(i, i + 2), 16));
  const bg = hex(css("--surface-1")), target = hex(css("--object-target"));
  const others = css("--object-others").split(",").map(hex);
  maps.forEach((m, i) => {
    const r = active.runs[i];
    if (!m) { r.heat.getContext("2d").clearRect(0, 0, r.heat.width, r.heat.height); return; }
    if (r.heat.width !== m.w || r.heat.height !== m.h) { r.heat.width = m.w; r.heat.height = m.h; }
    const g = r.heat.getContext("2d"), img = g.createImageData(m.w, m.h), px = img.data;
    for (let p = 0; p < m.v.length; p++) {
      const v = m.v[p], c = v === 0 ? bg : v === 1 ? target : others[(v - 2) % others.length];
      px[p * 4] = c[0]; px[p * 4 + 1] = c[1]; px[p * 4 + 2] = c[2]; px[p * 4 + 3] = 255;
    }
    g.putImageData(img, 0, 0);
  });
  const legend = $("heat-legend");
  legend.hidden = !maps.some(Boolean);
  const counts = maps.filter(Boolean).map((m) => Math.max(0, ...m.v) - 1 + (m.v.includes(1) ? 1 : 0));
  legend.innerHTML = `<span>Heatmap: <b>Objects</b> (creatures in the neighbour sum K ∗ A)</span>
    <span><i class="swatch-box" style="background:${css("--object-target")}"></i>target</span>
    <span><i class="swatch-box" style="background:${others.length ? css("--object-others").split(",")[0] : ""}"></i>others</span>
    <span>${counts.join(" / ")} objects</span>`;
}

function drawMaps(msg) {
  const maps = msg.runs.map((r) => (r.map ? { ...r.map, v: decodeMap(r.map) } : null));
  if (mapKey === "objects") return drawObjects(maps);
  let hi = 0;
  for (const m of maps) if (m) for (const v of m.v) if (v > hi) hi = v;
  const scale = hi > 0 ? hi : 1, ramp = heatRamp();
  maps.forEach((m, i) => {
    const r = active.runs[i];
    if (!m) {  // the measure does not run here (or has no map for this system)
      r.heat.getContext("2d").clearRect(0, 0, r.heat.width, r.heat.height);
      return;
    }
    const canvas = r.heat;
    if (canvas.width !== m.w || canvas.height !== m.h) { canvas.width = m.w; canvas.height = m.h; }
    const g = canvas.getContext("2d"), img = g.createImageData(m.w, m.h), px = img.data;
    for (let p = 0; p < m.v.length; p++) {
      const t = Math.min(1, Math.max(0, m.v[p] / scale));
      const [a, b, f] = t < 0.5 ? [ramp.lo, ramp.mid, t * 2] : [ramp.mid, ramp.hi, t * 2 - 1];
      for (let c = 0; c < 3; c++) px[p * 4 + c] = a[c] + (b[c] - a[c]) * f;
      px[p * 4 + 3] = 255;
    }
    g.putImageData(img, 0, 0);
  });
  const m = mapInfo(mapKey);
  const legend = $("heat-legend");
  legend.hidden = !maps.some(Boolean);
  if (!legend.hidden) {
    const bar = `linear-gradient(90deg, ${css("--surface-1")}, ${css("--heat-mid")}, ${css("--heat-hi")})`;
    legend.innerHTML = `<span>Heatmap: <b>${m.name}</b>${m.map === "tile" ? " (per tile, above an empty tile)" : ""}</span>
      <span>0</span><span class="bar" style="background:${bar}"></span><span>${+hi.toPrecision(3)} ${m.map_unit || m.unit}</span>`;
  }
}

function onInit(msg) {
  // Labels under the views, in the table and the charts drop "(…)" descriptions, unless
  // that would make two different patterns look alike (Aquarium (soliton), Aquarium (8 units)).
  const short = (l) => l.replace(/\s*\([^)]*\)/g, "");
  const runs = msg.runs.map((r, i) => {
    const clash = msg.runs.some((o) => short(o.label) === short(r.label) && o.label !== r.label);
    return { ...r, label: clash ? r.label : short(r.label), slot: pending.slots[i] };
  });
  active = { runs, normalized: pending.normalized,
    measures: pending.measures.filter((k) => runs.some((r) => r.measures.includes(k))) };
  const perRun = () => Object.fromEntries(active.measures.map((k) => [k, active.runs.map(() => [])]));
  history = { t: [], v: perRun(), w: perRun() };
  buildViews();
  buildCharts();
  renderMapControls();
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
  if (mapKey) drawMaps(msg);
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
    const scale = Math.max(1, Math.floor(256 / Math.max(r.width, r.height)));
    const w = `${r.width * scale}px`, h = `${r.height * scale}px`;
    el.innerHTML = `<div class="view-row"><canvas class="frame"></canvas><canvas class="heat"></canvas></div>
      <div class="cap"><span class="swatch" style="background:${seriesColor(r.slot)}"></span>
      <span>${r.label}</span></div>`;
    const c = el.querySelector("canvas.frame");
    c.width = r.width; c.height = r.height;
    for (const x of el.querySelectorAll("canvas")) { x.style.width = w; x.style.height = h; }
    r.heat = el.querySelector("canvas.heat");
    box.append(el);
    r.ctx = c.getContext("2d");
    r.image = r.ctx.createImageData(r.width, r.height);
    r.lut = r.n_states === 2 ? LUT.discrete : LUT.continuous;
  });
}

function drawFrame(i, b64) {
  const r = active.runs[i];
  const raw = atob(b64), px = r.image.data, lut = r.lut;
  if (r.rgb) {  // a system's own RGB picture (multi-channel Lenia)
    for (let p = 0; p < raw.length / 3; p++) {
      px[p * 4] = raw.charCodeAt(p * 3); px[p * 4 + 1] = raw.charCodeAt(p * 3 + 1);
      px[p * 4 + 2] = raw.charCodeAt(p * 3 + 2); px[p * 4 + 3] = 255;
    }
    r.ctx.putImageData(r.image, 0, 0);
    return;
  }
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
  const rows = ms.map((m) => `<tr><td>${m.name} <span class="ms">${unitOf(m)}</span></td>` +
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
    // rule-level measures are constant over a run: one bar per run instead of a line
    el.innerHTML = `<h3>${m.icon} ${m.name}${isNormalized(m) ? " (normalized)" : ""}</h3><span class="unit">${unitOf(m)}${m.per_rule ? " per run (constant: a property of the rule)" : " vs. time step"}</span>`;
    el.querySelector("h3").onclick = () => openInfo(m);  // the same popup as the list's ⓘ
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
    const chart = { key, canvas: c, tip, hover: null, bars: m.per_rule };
    c.addEventListener("mousemove", (e) => { chart.hover = e.offsetX; dirty = true; });
    c.addEventListener("mouseleave", () => { chart.hover = null; tip.style.display = "none"; dirty = true; });
    charts[key] = chart;
  }
}

// Names of transient length's classes, shown on its bars (unless it reports the exponent).
const BAR_NAMES = { transient_length: ["bounded", "log", "linear", "polynomial", "exponential"] };

function barText(key, v) {
  const names = BAR_NAMES[key];
  if (names && measureParams[key]?.report !== "exponent" && Number.isInteger(v)) return names[v] ?? String(v);
  return String(+v.toPrecision(3));
}

// One bar per run: its latest value. For measures that do not change over a run.
function drawBars(chart, g, W, H) {
  const series = history.v[chart.key];
  const values = series.map((s) => { for (let k = s.length - 1; k >= 0; k--) if (s[k] !== null) return s[k]; return null; });
  const known = values.filter((v) => v !== null);
  if (!known.length) return;
  const pad = { l: 46, r: 10, t: 18, b: 8 };
  let lo = Math.min(0, ...known), hi = Math.max(0, ...known);
  if (hi - lo < 1e-9) hi = lo + 1;
  const yt = niceTicks(lo, hi);
  lo = Math.min(lo, yt[0]); hi = Math.max(hi, yt[yt.length - 1]);
  const Y = (v) => H - pad.b - ((v - lo) / (hi - lo)) * (H - pad.t - pad.b);
  g.font = "11px system-ui, sans-serif";
  g.fillStyle = css("--text-muted"); g.strokeStyle = css("--grid"); g.lineWidth = 1;
  g.textAlign = "right"; g.textBaseline = "middle";
  for (const v of yt) {
    const y = Math.round(Y(v)) + 0.5;
    g.beginPath(); g.moveTo(pad.l, y); g.lineTo(W - pad.r, y); g.stroke();
    g.fillText(+v.toPrecision(3), pad.l - 6, y);
  }
  const slot = (W - pad.l - pad.r) / values.length, bw = Math.min(56, slot * 0.7);
  const hovered = chart.hover === null ? -1 : Math.floor((chart.hover - pad.l) / slot);
  g.textAlign = "center";
  values.forEach((v, i) => {
    const x = pad.l + slot * (i + 0.5);
    if (v === null) {
      g.fillStyle = css("--text-muted"); g.textBaseline = "bottom";
      g.fillText("n/a", x, Y(0) - 2);
      return;
    }
    g.fillStyle = seriesColor(active.runs[i].slot);
    g.globalAlpha = hovered < 0 || hovered === i ? 1 : 0.45;
    const y0 = Y(0), y1 = Y(v);
    g.fillRect(x - bw / 2, Math.min(y0, y1), bw, Math.max(1, Math.abs(y1 - y0)));
    g.globalAlpha = 1;
    g.fillStyle = css("--text-secondary");
    g.textBaseline = v >= 0 ? "bottom" : "top";
    g.fillText(barText(chart.key, v), x, v >= 0 ? y1 - 2 : y1 + 2);
  });
  if (hovered >= 0 && hovered < values.length) {
    const v = values[hovered], col = seriesColor(active.runs[hovered].slot);
    chart.tip.innerHTML = `<div class="row"><span class="swatch" style="background:${col}"></span>${active.runs[hovered].label}<b>${v === null ? "n/a" : barText(chart.key, v)}</b></div>`;
    chart.tip.style.display = "block";
    const box = chart.canvas.getBoundingClientRect(), parent = chart.canvas.parentElement.getBoundingClientRect();
    const left = box.left - parent.left + pad.l + slot * (hovered + 1);
    chart.tip.style.left = `${Math.min(left, parent.width - chart.tip.offsetWidth - 8)}px`;
    chart.tip.style.top = `${box.top - parent.top + 8}px`;
  } else chart.tip.style.display = "none";
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
  if (chart.bars) return drawBars(chart, g, W, H);
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
  ws.onopen = () => { setStatus("connected"); sendSpeed(); sendMap(); };
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
    ["lenia", "orbium"], ["lenia", "paraptera_arcus_pedes"], ["lenia", "paraptera_cavus_furiosus"],
    ["lenia", "tetralapillium_metamorpha"], ["lenia", "florakronium_arcus"],
    ["baseline", "iid_noise_continuous"],
  ]) addRun(system, pattern);
  selectedMeasures = new Set(catalog.measures
    .filter((m) => DEFAULT_MEASURES.has(m.key)).map((m) => m.key));
  buildRunList();
  buildMeasureList();
  updateSidebar();
  $("map-measure").onchange = (e) => { setMap(e.target.value); renderMapControls(); };
  for (const [key, p] of Object.entries(PRESETS)) $("preset").add(new Option(p.name, key));
  $("preset").onchange = (e) => applyPreset(e.target.value);
  $("clear-runs").onclick = () => { runRows = []; updateSidebar(); };
  $("clear-measures").onclick = () => { selectedMeasures.clear(); updateSidebar(); };
  closable($("info"), $("info-close"));
  closable($("tags"), $("tags-close"));
  tagClicks($("measures"));
  tagClicks($("info-tags"));
  $("normalized").checked = normalized;
  $("normalized").onchange = (e) => {
    normalized = e.target.checked;
    try { localStorage.setItem("normalized", normalized ? "on" : "off"); } catch {}
    for (const span of document.querySelectorAll(".unit-text"))
      span.textContent = unitOf(catalog.measures.find((m) => m.key === span.dataset.key), normalized);
  };
  $("select-structural").onclick = () => {  // a preset: the frame measures not peaking at noise
    selectedMeasures = new Set(catalog.measures
      .filter((m) => !["randomness", "none"].includes(m.peaks_at) && !m.per_rule).map((m) => m.key));
    updateSidebar();
  };
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
