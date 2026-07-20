/* ARSI Studio frontend — faithful implementation of the Claude Design mockup
   (ARSI Studio.dc.html), wired to the FastAPI backend of docs/SPEC.md.
   Vanilla JS, full re-render on state change, event delegation via data-act. */
"use strict";

/* ---------------------------------------------------------------- palette */
const C = {
  bg: "oklch(0.15 0.008 250)", bgSide: "oklch(0.12 0.008 250)",
  bgCard: "oklch(0.17 0.01 250)", bgCard2: "oklch(0.16 0.008 250)",
  bgInput: "oklch(0.13 0.008 250)", bgBtn: "oklch(0.2 0.012 250)",
  bd: "oklch(0.28 0.012 250)", bd2: "oklch(0.24 0.012 250)", bd3: "oklch(0.3 0.014 250)",
  bdBtn: "oklch(0.32 0.014 250)",
  fg: "oklch(0.93 0.006 250)", fg2: "oklch(0.7 0.012 250)", fg3: "oklch(0.6 0.012 250)",
  fg4: "oklch(0.55 0.012 250)", fg5: "oklch(0.5 0.012 250)", dim: "oklch(0.45 0.012 250)",
  acc: "oklch(0.72 0.13 225)", accFg: "oklch(0.85 0.1 225)", accDark: "oklch(0.14 0.008 250)",
  accBg: "oklch(0.22 0.04 225)", accBd: "oklch(0.4 0.08 225)", accBd2: "oklch(0.42 0.09 225)",
  accSel: "oklch(0.24 0.06 225)", accSelBd: "oklch(0.5 0.1 225)",
  green: "oklch(0.75 0.16 150)", greenFg: "oklch(0.82 0.11 150)",
  greenBg: "oklch(0.22 0.04 150)", greenBd: "oklch(0.4 0.08 150)",
  red: "oklch(0.75 0.16 22)", redFg: "oklch(0.78 0.16 22)",
  redBg: "oklch(0.24 0.05 22)", redBd: "oklch(0.42 0.1 22)",
  amber: "oklch(0.8 0.09 75)",
  mono: "ui-monospace,monospace",
};
const TYPE_META = {
  object:   { fg: "oklch(0.82 0.11 225)", bg: "oklch(0.24 0.06 225)", bd: "oklch(0.42 0.1 225)" },
  graffiti: { fg: "oklch(0.8 0.13 300)",  bg: "oklch(0.26 0.08 300)", bd: "oklch(0.45 0.12 300)" },
  damage:   { fg: "oklch(0.83 0.13 22)",  bg: "oklch(0.26 0.08 22)",  bd: "oklch(0.45 0.12 22)" },
  litter:   { fg: "oklch(0.86 0.11 75)",  bg: "oklch(0.27 0.07 75)",  bd: "oklch(0.48 0.11 75)" },
  unknown:  { fg: C.fg2, bg: "oklch(0.22 0.012 250)", bd: C.bdBtn },
};

/* ---------------------------------------------------------------- state */
const S = {
  screen: "dashboard",
  theme: localStorage.getItem("arsi-theme") || "dark",
  health: null, models: [], pipelines: [], demo: [], refs: [], masks: [],
  jobs: [], settings: null, storage: null,
  pulling: null, pullPct: 0, pullStatus: "",
  toast: null,
  wiz: {
    step: 1, source: null, demoSel: [],
    video: null, extracting: false, extractMode: "seconds", extractN: 2,
    trimStart: 0, trimEnd: 100, frames: [],
    maskZones: [], draftPts: [], rectStart: null, maskTool: "poly",
    maskPreview: null, maskPreset: "none", zoneSeq: 1,
    pipeline: "vlm_05", model: null, promptPreset: "conservative",
    promptText: "", refPath: null, advOpen: false,
    diff: 40, minArea: 500, maxRegions: 25, retries: 2,
  },
  run: null,           // {jobId,total,processed,anomalous,failed,retried,done,cancelled,thumbs,log,frames}
  res: {
    jobId: null, data: null, filter: "all", sel: 0, split: false,
    compare: false, compareJob: null, compareData: null, hoverV: -1,
    coordSize: null,          // {w,h} of the bbox coordinate space
    compareCoordSize: null,   // same, for the compared job
    playing: false,
  },
  rev: {                      // review/labelling mode over the open job
    on: false, jobId: null, doc: null, metrics: null,
    saved: true, saving: false,
    draw: false, corner: null,          // missed-box two-click draw state
    pending: null, pendingLabel: "", pendingType: "object",
    pendingFrame: null,                 // frame the pending box was drawn on
    propagate: true,                    // copy verdicts to similar boxes on other frames
  },
  labels: null,               // rows of GET /api/reviews (Labels screen)
  lora: { status: null, exporting: false, result: null },
  ollamaTest: null,
};

/* ---------------------------------------------------------------- helpers */
const $app = () => document.getElementById("app");
const esc = (s) => String(s ?? "").replace(/&/g, "&amp;").replace(/</g, "&lt;")
  .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
const hint = (text) => `<span class="hint" data-tip="${esc(text)}">?</span>`;
const fmtGB = (b) => (b / 1e9).toFixed(1);
const fmtEta = (sec) => {
  sec = Math.max(0, Math.round(sec));
  const h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60), s = sec % 60;
  if (h > 0) return `${h}h ${m}m`;
  if (m > 0) return `${m}m ${s}s`;
  return `${s}s`;
};
const fmtDur = (sec) => {   // coarse, for pre-launch estimates (avoids false precision)
  if (sec < 45) return `~${Math.max(1, Math.round(sec))} s`;
  if (sec < 90 * 60) return `~${Math.round(sec / 60)} min`;
  return `~${(sec / 3600).toFixed(1)} h`;
};
const median = (arr) => {
  if (!arr.length) return 0;
  const a = arr.slice().sort((x, y) => x - y);
  return a[Math.floor(a.length / 2)];
};
async function jget(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
  return r.json();
}
async function jpost(url, body, method = "POST") {
  const r = await fetch(url, { method, headers: { "Content-Type": "application/json" },
                               body: body === undefined ? undefined : JSON.stringify(body) });
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
  return r.json().catch(() => ({}));
}
async function sseFetch(url, body, onEvent) {
  const r = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" },
                               body: JSON.stringify(body || {}) });
  if (!r.ok || !r.body) throw new Error("stream failed: " + r.status);
  const reader = r.body.getReader(), dec = new TextDecoder();
  let buf = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let i;
    while ((i = buf.indexOf("\n\n")) >= 0) {
      const chunk = buf.slice(0, i); buf = buf.slice(i + 2);
      const line = chunk.split("\n").find(l => l.startsWith("data: "));
      if (line) onEvent(JSON.parse(line.slice(6)));
    }
  }
}
let toastTimer = null;
function toast(msg) {
  S.toast = msg; render();
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { S.toast = null; render(); }, 2800);
}
function setScreen(s) {
  S.screen = s;
  if (s === "settings") refreshStorage();
  if (s === "labels") refreshLabels();
  if (s === "lora") refreshLora();
  if (s !== "results" && playTimer) {   // leaving results stops playback
    clearInterval(playTimer); playTimer = null; S.res.playing = false;
  }
  if (("#" + s) !== location.hash) history.replaceState(null, "", "#" + s);
  render();
}

/* ---------------------------------------------------------------- boot */
async function boot() {
  render();
  await Promise.allSettled([
    refreshHealth(), refreshModels(),
    jget("/api/pipelines").then(d => { S.pipelines = d.pipelines; initWizardDefaults(); }),
    jget("/api/demo-frames").then(d => { S.demo = d.frames; }),
    jget("/api/references").then(d => { S.refs = d.references; }),
    refreshMasks(), refreshJobs(),
    jget("/api/settings").then(d => { S.settings = d; }),
  ]);
  const hash = location.hash.slice(1);
  if (["dashboard", "wizard", "run", "results", "history", "labels", "lora", "settings"].includes(hash)) {
    if (hash === "results") await openResults(); else S.screen = hash;
    if (hash === "settings") refreshStorage();
    if (hash === "labels") refreshLabels();
    if (hash === "lora") refreshLora();
  }
  render();
  setInterval(refreshHealth, 15000);
}
async function refreshHealth() {
  let h;
  try { h = await jget("/api/health"); } catch { h = { ollama: false, models: [], gpu: false }; }
  const changed = JSON.stringify(h) !== JSON.stringify(S.health);
  S.health = h;
  if (changed) render();   // a silent poll must not disturb open dialogs/drawing
}
async function refreshModels() {
  try { S.models = (await jget("/api/models")).models; } catch { S.models = []; }
}
async function refreshMasks() {
  try { S.masks = (await jget("/api/masks")).masks; } catch { S.masks = []; }
}
async function refreshJobs() {
  try { S.jobs = (await jget("/api/jobs")).jobs; } catch { S.jobs = []; }
}
async function refreshStorage() {
  try { S.storage = await jget("/api/storage"); } catch { S.storage = null; }
  try { S.settings = await jget("/api/settings"); } catch {}
  render();
}
async function refreshLabels() {
  await refreshJobs();
  try { S.labels = (await jget("/api/reviews")).reviews; } catch { S.labels = []; }
  render();
}
async function refreshLora() {
  try { S.lora.status = await jget("/api/lora/status"); } catch { S.lora.status = null; }
  render();
}
function initWizardDefaults() {
  const p = S.pipelines.find(x => x.key === S.wiz.pipeline);
  if (p) {
    const names = Object.keys(p.prompts);
    S.wiz.promptPreset = names.includes("conservative") ? "conservative" : names[0];
    S.wiz.promptText = p.prompts[S.wiz.promptPreset];
    if (!S.wiz.model) S.wiz.model = p.default_model;
  }
  if (!S.wiz.refPath && S.refs.length) {
    const def = S.refs.find(r => r.path.includes("f0227")) || S.refs[0];
    S.wiz.refPath = def.path;
  }
}

/* ---------------------------------------------------------------- actions */
const ACT = {};

/* --- nav + theme --- */
ACT.go = (arg) => { if (arg === "results") openResults(); else setScreen(arg); };
function resetWizardSource() {
  // a fresh analysis starts from a clean source; pipeline/model/prompt/mask
  // choices are kept (they are per-camera, not per-run)
  const w = S.wiz;
  if (w.maskPreview) { URL.revokeObjectURL(w.maskPreview); }
  Object.assign(w, { step: 1, source: null, demoSel: [], video: null,
                     frames: [], extracting: false, uploading: false,
                     maskPreview: null, reuseId: null, framePool: [], frameSel: [] });
  S._videos = null;
}
function maybeResetWizard() {
  // reset ONLY after a finished run (fresh start); plain tab navigation must
  // keep the in-progress wizard exactly as the user left it
  if (S.run && S.run.done) {
    resetWizardSource(); S.run = null;
    clearTimeout(stageTimer); stageTimer = null;
  }
}
ACT.goWizard = () => { maybeResetWizard(); setScreen("wizard"); };
ACT.goWizardDemo = () => { maybeResetWizard(); S.wiz.source = "demo"; S.wiz.step = 1; setScreen("wizard"); };
ACT.toggleTheme = () => {
  S.theme = S.theme === "dark" ? "light" : "dark";
  localStorage.setItem("arsi-theme", S.theme); render();
};
ACT.resumeLast = () => {
  if (S.run) { setScreen("run"); return; }
  const j = S.jobs[0];
  if (j) { openResults(j.job_id); } else { ACT.goWizard(); }
};

/* --- wizard nav --- */
ACT.wizGoto = (n) => { S.wiz.step = +n; render(); };
ACT.wizBack = () => { S.wiz.step = Math.max(1, S.wiz.step - 1); render(); };
ACT.wizNext = async () => {
  const w = S.wiz;
  if (w.step === 1) {
    if (!w.source) { toast("Pick a source first."); return; }
    if (w.source === "demo" && !w.demoSel.length) { toast("Select at least one demo frame."); return; }
    if (w.source === "video" && !w.video) { toast("Upload a video first."); return; }
    if (w.source === "reuse") {
      if (!(w.frameSel || []).length) { toast("Select at least one frame."); return; }
      w.frames = (w.framePool || []).filter(f => w.frameSel.includes(f.path));
    }
    if (w.source === "demo") {
      w.frames = w.demoSel.map(id => {
        const d = S.demo.find(x => x.id === id);
        return { path: d.path, img: d.img, id: d.id };
      });
      const d0 = S.demo.find(x => x.id === w.demoSel[0]);
      if (d0) w.refPath = d0.reference;
    }
  }
  if (w.step === 2 && w.source === "video" && !w.frames.length) {
    await doExtract(); if (!w.frames.length) return;
  }
  w.step = Math.min(5, w.step + 1); render();
};
ACT.setSource = (v) => { S.wiz.source = v; render(); };
ACT.pickDemo = (id) => {
  const sel = S.wiz.demoSel;
  const i = sel.indexOf(id);
  if (i >= 0) sel.splice(i, 1); else sel.push(id);
  S.wiz.source = "demo"; render();
};
ACT.demoAll = () => {
  S.wiz.demoSel = S.demo.map(d => d.id); S.wiz.source = "demo"; render();
};
ACT.demoNone = () => { S.wiz.demoSel = []; render(); };
ACT.pickVideoFile = () => document.getElementById("videoFile").click();
ACT.reuseVideo = async (videoId) => {
  try {
    const d = await jget(`/api/videos/${videoId}/frames`);
    // reuseId is separate from `video` (the uploaded-video state): mixing them
    // broke the "Upload a video" card after picking an extraction
    Object.assign(S.wiz, { source: "reuse", reuseId: videoId, framePool: d.frames,
                           frameSel: d.frames.map(f => f.path),   // all selected by default
                           frames: [] });
    render();
  } catch (e) { toast("Could not load extraction: " + e.message); }
};
ACT.togglePoolFrame = (path) => {
  const sel = S.wiz.frameSel || [];
  const i = sel.indexOf(path);
  if (i >= 0) sel.splice(i, 1); else sel.push(path);
  S.wiz.frameSel = sel; render();
};
ACT.poolAll = () => { S.wiz.frameSel = (S.wiz.framePool || []).map(f => f.path); render(); };
ACT.poolNone = () => { S.wiz.frameSel = []; render(); };

/* --- extraction --- */
/* Changing the trim or the sampling interval invalidates frames already
   extracted. Without this, coming back to step 2 and re-cutting did nothing:
   wizNext only extracts when w.frames is empty, so it kept the old cut, and
   estFrames() kept reporting the old count instead of the new setting. */
function invalidateExtraction() {
  if (S.wiz.source === "video") S.wiz.frames = [];
}
ACT.setExtMode = (m) => { S.wiz.extractMode = m; invalidateExtraction(); render(); };
async function doExtract() {
  const w = S.wiz;
  const dur = w.video.info.duration_s || 0;
  const startS = w.trimStart / 100 * dur, endS = w.trimEnd / 100 * dur;
  const body = { start_s: startS, end_s: endS };
  if (w.extractMode === "seconds") body.every_s = w.extractN; else body.every_n = w.extractN;
  w.extracting = true; render();
  try {
    const d = await jpost(`/api/videos/${w.video.video_id}/extract`, body);
    w.frames = d.frames;
    toast(`${d.frames.length} frames extracted.`);
  } catch (e) { toast("Extraction failed: " + e.message); }
  w.extracting = false; render();
}
function estFrames() {
  const w = S.wiz;
  if (w.frames.length) return w.frames.length;
  if (w.source !== "video" || !w.video) return w.demoSel.length || 0;
  const dur = (w.video.info.duration_s || 0) * (w.trimEnd - w.trimStart) / 100;
  if (w.extractMode === "seconds") return Math.max(1, Math.round(dur / Math.max(0.1, w.extractN)));
  const fps = w.video.info.fps || 25;
  return Math.max(1, Math.round(dur * fps / Math.max(1, w.extractN)));
}

/* --- mask editor --- */
const VBH = 56.25;  // svg viewBox height (16:9 in percent-of-width units)
ACT.setMaskTool = (t) => { S.wiz.maskTool = t; S.wiz.draftPts = []; S.wiz.rectStart = null; render(); };
ACT.maskSvgClick = (_, ev) => {
  const w = S.wiz;
  if (w.maskPreview) return;
  const svg = document.getElementById("maskSvg");
  const r = svg.getBoundingClientRect();
  const x = Math.round(Math.max(0, Math.min(100, (ev.clientX - r.left) / r.width * 100)) * 10) / 10;
  const y = Math.round(Math.max(0, Math.min(VBH, (ev.clientY - r.top) / r.height * VBH)) * 10) / 10;
  if (w.maskTool === "rect") {
    if (!w.rectStart) { w.rectStart = { x, y }; }
    else {
      const a = w.rectStart;
      w.maskZones.push({ id: w.zoneSeq, label: "Zone " + w.zoneSeq, type: "rect",
                         pts: [{ x: a.x, y: a.y }, { x, y: a.y }, { x, y }, { x: a.x, y }] });
      w.zoneSeq++; w.rectStart = null; w.maskPreset = "custom";
    }
    render(); return;
  }
  w.draftPts.push({ x, y }); render();
};
ACT.maskFinish = () => {
  const w = S.wiz;
  if (w.draftPts.length < 3) { toast("A polygon needs at least 3 points."); return; }
  w.maskZones.push({ id: w.zoneSeq, label: "Zone " + w.zoneSeq, type: "poly", pts: w.draftPts });
  w.zoneSeq++; w.draftPts = []; w.maskPreset = "custom"; render();
};
ACT.maskUndo = () => { S.wiz.draftPts.pop(); render(); };
ACT.maskDelete = (id) => {
  S.wiz.maskZones = S.wiz.maskZones.filter(z => z.id !== +id);
  S.wiz.maskPreset = S.wiz.maskZones.length ? "custom" : "none"; render();
};
ACT.maskClear = () => {
  Object.assign(S.wiz, { maskZones: [], draftPts: [], rectStart: null, maskPreset: "none" });
  render();
};
function maskEditorImage() {
  const w = S.wiz;
  const f = w.frames[0];
  if (f) return f;
  const d = S.demo.find(x => x.id === w.demoSel[0]);
  if (d) return { img: d.img, path: d.path };
  const r = S.refs[0];
  return r ? { img: r.img, path: r.path } : null;
}
function zonesToPixels(imgW, imgH) {
  return S.wiz.maskZones.map(z => ({
    id: String(z.id), label: z.label,
    polygon: z.pts.map(p => [Math.round(p.x / 100 * imgW), Math.round(p.y / VBH * imgH)]),
  }));
}
async function maskSpecFromEditor(name) {
  const src = maskEditorImage();
  if (!src) return null;
  const size = await imageSize(src.img);
  return { name, camera: "tram_1762", image_size: [size.w, size.h],
           zones: zonesToPixels(size.w, size.h), image: src.path };
}
const _sizeCache = {};
function imageSize(url) {
  if (_sizeCache[url]) return Promise.resolve(_sizeCache[url]);
  return new Promise((ok, ko) => {
    const im = new Image();
    im.onload = () => { _sizeCache[url] = { w: im.naturalWidth, h: im.naturalHeight }; ok(_sizeCache[url]); };
    im.onerror = ko; im.src = url;
  });
}
ACT.toggleMaskPreview = async () => {
  const w = S.wiz;
  if (w.maskPreview) { URL.revokeObjectURL(w.maskPreview); w.maskPreview = null; render(); return; }
  if (!w.maskZones.length) { toast("No zones to preview."); return; }
  try {
    const spec = await maskSpecFromEditor("preview");
    const r = await fetch("/api/masks/preview", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(spec) });
    if (!r.ok) throw new Error(r.statusText);
    w.maskPreview = URL.createObjectURL(await r.blob()); render();
  } catch (e) { toast("Preview failed: " + e.message); }
};
ACT.saveMaskPreset = async () => {
  const name = prompt("Preset name (per camera):", S.wiz.maskPreset !== "none" &&
                      S.wiz.maskPreset !== "custom" ? S.wiz.maskPreset : "tram_1762_windows");
  if (!name) return;
  try {
    const spec = await maskSpecFromEditor(name.trim().replace(/\s+/g, "_"));
    if (!spec || !spec.zones.length) { toast("Draw at least one zone first."); return; }
    await jpost("/api/masks", spec);
    await refreshMasks();
    S.wiz.maskPreset = spec.name;
    toast(`Mask preset '${spec.name}' saved.`); render();
  } catch (e) { toast("Save failed: " + e.message); }
};
ACT.setMaskPreset = async (name) => {
  const w = S.wiz;
  if (name === "none") { Object.assign(w, { maskPreset: "none", maskZones: [], draftPts: [] }); render(); return; }
  if (name === "custom") { w.maskPreset = "custom"; render(); return; }
  const m = S.masks.find(x => x.name === name);
  if (!m) return;
  const src = maskEditorImage();
  const size = src ? await imageSize(src.img) : { w: m.image_size[0], h: m.image_size[1] };
  const sx = 100 / m.image_size[0] * (m.image_size[0] / size.w) * (size.w / m.image_size[0]);
  w.maskZones = m.zones.map((z, i) => ({
    id: i + 1, label: z.label || "Zone " + (i + 1), type: "poly",
    pts: z.polygon.map(([px, py]) => ({ x: px / m.image_size[0] * 100,
                                        y: py / m.image_size[1] * VBH })),
  }));
  w.zoneSeq = w.maskZones.length + 1; w.maskPreset = name; w.draftPts = []; render();
};

/* --- pipeline step --- */
ACT.setPipeline = (k) => {
  S.wiz.pipeline = k;
  const p = S.pipelines.find(x => x.key === k);
  if (p) {
    const names = Object.keys(p.prompts);
    S.wiz.promptPreset = names[0];
    S.wiz.promptText = p.prompts[names[0]];
    S.wiz.model = p.default_model;
  }
  render();
};
ACT.setModel = (tag) => { S.wiz.model = tag; render(); };
ACT.pullModel = async (tag) => {
  if (S.pulling) return;
  S.pulling = tag; S.pullPct = 0; render();
  try {
    await sseFetch("/api/models/pull", { tag }, (e) => {
      if (e.status === "error") throw new Error(e.error);
      S.pullPct = e.pct || 0; S.pullStatus = e.status || ""; render();
    });
    toast("Model pulled and installed.");
  } catch (e) { toast("Pull failed: " + e.message); }
  S.pulling = null; await refreshModels(); render();
};
ACT.removeModel = async (tag) => {
  if (!confirm(`Remove ${tag} from Ollama?`)) return;
  try { await jpost("/api/models/" + encodeURIComponent(tag), undefined, "DELETE"); }
  catch (e) { toast("Remove failed: " + e.message); }
  await refreshModels(); render();
};
ACT.setPromptPreset = (name) => {
  const w = S.wiz;
  const p = S.pipelines.find(x => x.key === w.pipeline);
  if (name !== "custom" && p && p.prompts[name]) { w.promptPreset = name; w.promptText = p.prompts[name]; }
  else w.promptPreset = "custom";
  render();
};
ACT.resetPrompt = () => {
  const p = S.pipelines.find(x => x.key === S.wiz.pipeline);
  if (!p) return;
  const name = p.prompts[S.wiz.promptPreset] ? S.wiz.promptPreset : Object.keys(p.prompts)[0];
  S.wiz.promptPreset = name; S.wiz.promptText = p.prompts[name]; render();
};
ACT.toggleAdv = () => { S.wiz.advOpen = !S.wiz.advOpen; render(); };
ACT.setRefFromFrame = (path) => { S.wiz.refPath = path; render(); };
ACT.pickRefFile = () => document.getElementById("refFile").click();

/* --- launch + run --- */
function needRef() {
  const p = S.pipelines.find(x => x.key === S.wiz.pipeline);
  return !!(p && p.ref);
}
/* Real seconds-per-frame from past jobs of the same pipeline. The gpu flag is
   unreliable and per-frame cost depends on model/region-count/cache, all
   captured by a real wall_seconds/n_frames. We take the MAX across matching
   jobs: the verdict cache only ever makes a run FASTER, so the slowest
   observation best predicts a fresh (uncached) video. */
function histPerFrame(script, model) {
  let done = S.jobs.filter(j => j.config && j.config.script === script
    && j.summary && j.summary.n_frames > 0 && j.summary.wall_seconds > 0);
  if (!done.length) return null;
  const sameModel = done.filter(j => j.config.model === model);
  if (sameModel.length) done = sameModel;
  const perFrame = Math.max(...done.map(j => j.summary.wall_seconds / j.summary.n_frames));
  return { perFrame, sameModel: sameModel.length > 0 };
}
// rough per-frame fallback (no usable history yet); replaced after run 1
const ROUGH_PER_FRAME = {
  gpu: { vlm_05: 12, vlm_04: 8, _: 3 },
  cpu: { vlm_05: 180, vlm_04: 120, _: 40 },
};
// below this, a crop pipeline's history is cache-flattered, not a real run
const CROP_MIN_CREDIBLE = 4;
function estimate() {
  const frames = estFrames();
  const script = S.wiz.pipeline;
  const cropPipe = script === "vlm_05" || script === "vlm_04";
  const h = histPerFrame(script, S.wiz.model);
  let perFrame, basis, rough;
  if (h && !(cropPipe && h.perFrame < CROP_MIN_CREDIBLE)) {
    perFrame = h.perFrame; rough = false;
    basis = `based on your ${script} history${h.sameModel ? " with this model" : ""} · ${fmtEta(perFrame)}/frame`;
  } else {
    const t = ROUGH_PER_FRAME[(S.health && S.health.gpu) ? "gpu" : "cpu"];
    perFrame = t[script] || t._; rough = true;
    basis = "rough guess — the real speed appears as soon as the run starts";
  }
  return { frames, perFrame, total: frames * perFrame, basis, rough };
}
/* Live ETA: median of the recently measured frame times (drops the first-call
   model-load spike and adapts if the pace changes); seeded by the pre-launch
   estimate until the first frame lands. */
function runPerFrame(run) {
  if (run.times && run.times.length) return median(run.times.slice(-12));
  return run.estPerFrame || 0;
}
function runEta(run) {
  if (run.done) return "done";
  const remaining = run.total - run.processed;
  if (remaining <= 0) return "finishing…";
  const per = runPerFrame(run);
  return per ? fmtEta(remaining * per) : "…";
}
ACT.launchRun = async () => {
  const w = S.wiz;
  let maskName = null;
  if (w.maskZones.length) {
    if (w.maskPreset === "custom" || w.maskPreset === "none") {
      const spec = await maskSpecFromEditor("_job_" + Date.now());
      await jpost("/api/masks", spec);
      await refreshMasks();
      maskName = spec.name;
    } else maskName = w.maskPreset;
  }
  const params = { max_retries: w.retries };
  if (w.pipeline === "vlm_05") Object.assign(params, {
    DIFF_THRESHOLD: +w.diff, MIN_AREA: +w.minArea, MAX_REGIONS: +w.maxRegions });
  const body = {
    script: w.pipeline, model: w.model, frames: w.frames.map(f => f.path),
    prompt: w.promptText, prompt_name: w.promptPreset,
    reference: needRef() ? w.refPath : null, mask: maskName, params,
  };
  try {
    const { job_id } = await jpost("/api/jobs", body);
    S.run = { jobId: job_id, total: w.frames.length, processed: 0, anomalous: 0,
              failed: 0, retried: 0, done: false, cancelled: false,
              thumbs: [], log: [`[start] job ${job_id} · ${w.pipeline} · ${w.model}`],
              frames: w.frames.slice(), script: w.pipeline, model: w.model,
              // reference/mask: the masked copies replace these as soon as the
              // runner reports mask_applied — the screen must show what the VLM saw
              refImg: body.reference ? "/api/media/" + body.reference : null,
              masked: !!maskName, maskName,
              reel: [], live: null, stage: null,   // big "current frame" viewer
              times: [], estPerFrame: estimate().perFrame, t0: Date.now() };
    setScreen("run");
    watchRun(job_id);
  } catch (e) {
    if (String(e.message).includes("not installed")) toast(e.message + " — pull it in step 4.");
    else toast("Launch failed: " + e.message);
  }
};
async function watchRun(jobId) {
  const r = await fetch(`/api/jobs/${jobId}/events`);
  const reader = r.body.getReader(), dec = new TextDecoder();
  let buf = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let i;
    while ((i = buf.indexOf("\n\n")) >= 0) {
      const chunk = buf.slice(0, i); buf = buf.slice(i + 2);
      const line = chunk.split("\n").find(l => l.startsWith("data: "));
      if (line) handleRunEvent(JSON.parse(line.slice(6)));
    }
  }
}
function handleRunEvent(e) {
  const run = S.run;
  if (!run) return;
  // e.img is the image the runner actually fed the VLM (masked when a mask is
  // set); the wizard copy is only a fallback for pre-frame_start events
  const fallbackImg = (i) => (run.frames[i] || {}).img || "";
  if (e.event === "frame_retry") {
    const fid = e.frame ? e.frame.split("/").pop().replace(/\.[^.]+$/, "") : e.index;
    run.retried++; run.log.unshift(`[${fid}] retry: ${e.error}`);
  } else if (e.event === "frame_start") {
    run.live = { index: e.index, frame_id: e.frame_id,
                 img: e.img || fallbackImg(e.index) };
    pumpStage();
  } else if (e.event === "frame_done") {
    run.processed++;
    if (typeof e.seconds === "number") run.times.push(e.seconds);
    let ring = "green";
    if (e.status === "failed") { run.failed++; ring = "grey";
      run.log.unshift(`[${e.frame_id}] FAILED after ${e.attempts} attempts`); }
    else if (e.anomaly) { run.anomalous++; ring = "red";
      run.log.unshift(`[${e.frame_id}] YES anomaly kept · ${e.n_detections} region(s) · ${e.seconds}s`); }
    else run.log.unshift(`[${e.frame_id}] NO clean · ${e.seconds}s`);
    const rec = { index: e.index, frame_id: e.frame_id,
                  img: e.img || fallbackImg(e.index), status: e.status,
                  anomaly: e.anomaly, detections: e.detections || [],
                  seconds: e.seconds, attempts: e.attempts, error: e.error };
    run.thumbs.unshift({ ring, img: rec.img });
    if (run.thumbs.length > 60) run.thumbs.pop();
    if (run.log.length > 120) run.log.pop();
    if (run.live && run.live.index === e.index) run.live = null;
    run.reel.push(rec);
    if (run.reel.length > 40) run.reel.splice(0, run.reel.length - 40);
    pumpStage();
  } else if (e.event === "job_finished" || e.event === "stream_end") {
    run.done = true;
    run.live = null;
    if (e.status === "cancelled") run.cancelled = true;
    if (e.error) run.log.unshift(`[error] ${e.error}`);
    if (e.event === "stream_end")
      run.log.unshift(`[done] ${run.processed} frames · ${run.anomalous} anomalous · ${run.failed} failed`);
    pumpStage();
    refreshJobs();
  } else if (e.event === "mask_applied") {
    run.masked = true;
    if (e.reference_img) run.refImg = e.reference_img;
    run.log.unshift(`[mask] '${e.mask}' applied to ${e.n_images} image(s)`);
  } else if (e.event === "job_started") {
    run.log.unshift(`[job] started · ${e.n_frames} frames · model ${e.model}`);
  }
  if (S.screen === "run") render();
}

/* --- the big "current frame" viewer -------------------------------------
   Analysis never waits on the display: finished frames queue in run.reel and
   the stage walks that queue, holding each verdict a couple of seconds. When
   the queue backs up (cached or fast runs) the hold shortens instead of
   drifting minutes behind the real work. */
const STAGE_HOLD = 2600, STAGE_HOLD_BUSY = 1400, STAGE_HOLD_RUSH = 700;
let stageTimer = null;
function stageHold(run) {
  return run.reel.length >= 4 ? STAGE_HOLD_RUSH
    : run.reel.length >= 2 ? STAGE_HOLD_BUSY : STAGE_HOLD;
}
const stageKey = (s) => s ? `${s.index}:${s.phase}` : "";
function pumpStage() {
  clearTimeout(stageTimer); stageTimer = null;
  const run = S.run;
  if (!run) return;
  const before = stageKey(run.stage);
  // the frame on screen just finished -> reveal its verdict in place
  if (run.stage && run.stage.phase === "processing") {
    const i = run.reel.findIndex(r => r.index === run.stage.index);
    if (i >= 0) run.stage = Object.assign(run.reel.splice(i, 1)[0],
                                          { phase: "result", at: Date.now() });
  }
  // a verdict that has had its time makes way — but only for something real,
  // so the last frame of a run stays up instead of blanking
  if (run.stage && run.stage.phase === "result"
      && Date.now() - run.stage.at >= stageHold(run)
      && (run.reel.length || run.live)) run.stage = null;
  if (!run.stage) {
    if (run.reel.length)
      run.stage = Object.assign(run.reel.shift(), { phase: "result", at: Date.now() });
    else if (run.live)
      run.stage = Object.assign({}, run.live, { phase: "processing", at: Date.now() });
  }
  if (run.stage && run.stage.phase === "result") {
    const left = stageHold(run) - (Date.now() - run.stage.at);
    stageTimer = setTimeout(pumpStage, Math.max(120, left));
  }
  if (run.stage && run.stage.img && !_sizeCache[run.stage.img])
    imageSize(run.stage.img).then(() => { if (S.screen === "run") render(); })
      .catch(() => {});
  if (stageKey(run.stage) !== before && S.screen === "run") render();
}
ACT.cancelRun = async () => {
  if (!S.run) return;
  try { await jpost(`/api/jobs/${S.run.jobId}/cancel`); toast("Cancelling — partial results kept."); }
  catch (e) { toast(e.message); }
};
ACT.viewRunResults = () => openResults(S.run ? S.run.jobId : null);

/* --- results --- */
async function openResults(jobId) {
  await refreshJobs();
  const withFrames = S.jobs.filter(j => j.summary && j.summary.n_frames);
  const id = jobId || S.res.jobId || (withFrames[0] && withFrames[0].job_id);
  if (!id) { S.res.data = null; setScreen("results"); return; }
  try {
    const data = await jget("/api/jobs/" + id);
    if (S.rev.jobId !== id) Object.assign(S.rev, { on: false, jobId: null, doc: null, metrics: null });
    S.res.jobId = id; S.res.data = data; S.res.sel = 0; S.res.hoverV = -1;
    S.res.coordSize = null;
    const first = (data.frames || [])[0];
    const spaceUrl = jobRefImg(data) || (first && first.img);
    if (spaceUrl) imageSize(spaceUrl).then(sz => { S.res.coordSize = sz; render(); });
  } catch (e) { toast("Could not load job: " + e.message); }
  setScreen("results");
}
/* A masked job compared the frames against the MASKED reference — showing the
   untouched one would misrepresent what the pipeline saw. */
function jobRefImg(data) {
  const cfg = (data && data.config) || {};
  return cfg.reference_masked_img || cfg.reference_img || null;
}
ACT.openJob = (jobId) => openResults(jobId);
/* A half-drawn missed box belongs to the frame it was drawn on. Every change of
   frame drops it: it used to be global state, so the dashed outline followed the
   reviewer onto every other frame (only a page reload cleared it) and labelling
   it there filed the box — with the original frame's coordinates — under the
   wrong frame. Every S.res.sel change must go through setSel. */
function revClearPending() {
  Object.assign(S.rev, { pending: null, corner: null, pendingLabel: "",
                         pendingFrame: null });
}
function setSel(i) {
  if (S.res.sel !== i && (S.rev.pending || S.rev.corner)) {
    revClearPending();
    toast("Unfinished box discarded — it belonged to the previous frame.");
  }
  S.res.sel = i; S.res.hoverV = -1;
}
ACT.setFilter = (f) => { S.res.filter = f; setSel(0); render(); };
ACT.selFrame = (i) => { setSel(+i); render(); };

/* play: auto-advance through the filtered frames (wraps; stops on toggle,
   filter click keeps playing, leaving the screen stops it) */
let playTimer = null;
function visibleIndices() {
  const R = S.res, frames = (R.data && R.data.frames) || [];
  const match = (f) => R.filter === "all" ? true
    : R.filter === "anomalous" ? !!f.anomaly
    : R.filter === "failed" ? f.status === "failed"
    : frameTypes(f).includes(R.filter);
  return frames.map((f, i) => i).filter(i => match(frames[i]));
}
function stopPlay() {
  clearInterval(playTimer); playTimer = null;
  if (S.res.playing) { S.res.playing = false; render(); }
}
ACT.togglePlay = () => {
  if (S.res.playing) { stopPlay(); return; }
  S.res.playing = true;
  playTimer = setInterval(() => {
    if (S.screen !== "results" || !S.res.data) { stopPlay(); return; }
    const vis = visibleIndices();
    if (!vis.length) { stopPlay(); return; }
    const pos = vis.indexOf(Math.min(S.res.sel, S.res.data.frames.length - 1));
    setSel(vis[(pos + 1) % vis.length]);
    S._kbNav = true;
    render();
  }, 900);
  render();
};
ACT.toggleSplit = () => { S.res.split = !S.res.split; render(); };
ACT.toggleCompare = async () => {
  S.res.compare = !S.res.compare;
  if (S.res.compare && !S.res.compareJob) {
    const other = S.jobs.find(j => j.job_id !== S.res.jobId && j.summary && j.summary.n_frames);
    if (other) await ACT.setCompareJob(other.job_id); else { toast("No other job to compare with."); S.res.compare = false; }
  }
  render();
};
ACT.setCompareJob = async (jobId) => {
  try {
    const data = await jget("/api/jobs/" + jobId);
    S.res.compareData = data; S.res.compareJob = jobId;
    S.res.compareCoordSize = null;
    const first = (data.frames || [])[0];
    const spaceUrl = jobRefImg(data) || (first && first.img);
    if (spaceUrl) imageSize(spaceUrl).then(sz => { S.res.compareCoordSize = sz; render(); });
  } catch (e) { toast(e.message); }
  render();
};
ACT.exportReport = () => window.open(`/api/jobs/${S.res.jobId}/report.html`, "_blank");
ACT.exportMd = () => window.open(`/api/jobs/${S.res.jobId}/report.md`, "_blank");
ACT.exportJson = () => window.open(`/api/jobs/${S.res.jobId}/results.json`, "_blank");
ACT.exportXlsx = () => { location.href = `/api/jobs/${S.res.jobId}/export.xlsx`; };
ACT.hoverV = (i) => { S.res.hoverV = +i; render(); };
ACT.unhoverV = () => { S.res.hoverV = -1; render(); };
ACT.openReportJob = (jobId) => window.open(`/api/jobs/${jobId}/report.html`, "_blank");

/* --- review / labelling mode --- */
function curFrame() { return S.res.data.frames[Math.min(S.res.sel, S.res.data.frames.length - 1)]; }
function revEntry(frameId) {
  const fr = S.rev.doc.frames;
  return fr[frameId] || (fr[frameId] = { verdicts: {}, missed: [], done: false });
}
function revActive() { return S.rev.on && S.rev.doc && S.rev.jobId === S.res.jobId; }
ACT.toggleReview = async () => {
  if (S.rev.on) { S.rev.on = false; S.rev.draw = false; revClearPending(); render(); return; }
  try {
    const d = await jget(`/api/jobs/${S.res.jobId}/review`);
    Object.assign(S.rev, { on: true, jobId: S.res.jobId, doc: d.review, metrics: d.metrics,
                           saved: true, saving: false, draw: false, corner: null, pending: null });
  } catch (e) { toast("Review unavailable: " + e.message); }
  render();
};
let revSaveTimer = null;
function revTouch() {
  S.rev.saved = false;
  clearTimeout(revSaveTimer);
  revSaveTimer = setTimeout(revSave, 700);
  render();
}
async function revSave() {
  if (!S.rev.doc) return;
  S.rev.saving = true; render();
  try {
    const d = await jpost(`/api/jobs/${S.rev.jobId}/review`, { frames: S.rev.doc.frames }, "PUT");
    S.rev.doc = d.review; S.rev.metrics = d.metrics; S.rev.saved = true;
  } catch (e) { toast("Review save failed: " + e.message); }
  S.rev.saving = false; render();
}
function iouBox(a, b) {
  const ix = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0]));
  const iy = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
  const inter = ix * iy;
  if (!inter) return 0;
  const area = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
  return inter / area;
}
function revPropagate(fromFrame, det, verdict) {
  // The camera is fixed and forgotten objects don't move: the same object
  // shows up as a near-identical box on every frame. Copy the verdict to
  // UNSET verdicts of unconfirmed frames only — never overwrite a judgement.
  if (!det.bbox) return 0;
  let nBoxes = 0;
  for (const f of S.res.data.frames) {
    if (f.frame_id === fromFrame.frame_id || f.status !== "ok") continue;
    const existing = S.rev.doc.frames[f.frame_id];
    if (existing && existing.done) continue;
    f.detections.forEach((d, i) => {
      if (!d.bbox || d.type !== det.type) return;
      if (existing && existing.verdicts[i] !== undefined) return;
      if (iouBox(d.bbox, det.bbox) >= 0.55) {
        revEntry(f.frame_id).verdicts[i] = verdict;
        nBoxes++;
      }
    });
  }
  return nBoxes;
}
function revSetVerdict(i, v) {   // v undefined -> cycle unset -> tp -> fp -> unset
  const f = curFrame(); const e = revEntry(f.frame_id);
  const cur = e.verdicts[i];
  if (v === undefined) v = cur === undefined ? "tp" : cur === "tp" ? "fp" : null;
  if (v === null || cur === v) { delete e.verdicts[i]; v = null; } else e.verdicts[i] = v;
  if (Object.keys(e.verdicts).length < f.detections.length) e.done = false;
  if (v && S.rev.propagate) {
    const n = revPropagate(f, f.detections[i], v);
    if (n) toast(`Verdict propagated to ${n} similar box(es) on other frames.`);
  }
  revTouch();
}
ACT.revTogglePropagate = () => { S.rev.propagate = !S.rev.propagate; render(); };
ACT.revAllTpConfirm = () => {
  const f = curFrame(); const e = revEntry(f.frame_id);
  f.detections.forEach((d, i) => {
    if (e.verdicts[i] === undefined) {
      e.verdicts[i] = "tp";
      if (S.rev.propagate) revPropagate(f, d, "tp");
    }
  });
  ACT.revConfirm();
};
ACT.revCycle = (i) => revSetVerdict(+i);
ACT.revSet = (arg) => { const [i, v] = arg.split(":"); revSetVerdict(+i, v); };
ACT.revToggleDraw = () => {
  S.rev.draw = !S.rev.draw; revClearPending(); render();
};
ACT.revImgClick = (_, ev) => {
  if (!revActive() || !S.rev.draw || S.rev.pending) return;
  const img = document.getElementById("revImg");
  const cs = S.res.coordSize;
  if (!img || !cs) return;
  const r = img.getBoundingClientRect();
  const x = Math.round(Math.max(0, Math.min(1, (ev.clientX - r.left) / r.width)) * cs.w);
  const y = Math.round(Math.max(0, Math.min(1, (ev.clientY - r.top) / r.height)) * cs.h);
  if (!S.rev.corner) { S.rev.corner = { x, y }; render(); return; }
  const a = S.rev.corner;
  const bbox = [Math.min(a.x, x), Math.min(a.y, y), Math.max(a.x, x), Math.max(a.y, y)];
  S.rev.corner = null;
  if (bbox[2] - bbox[0] < 5 || bbox[3] - bbox[1] < 5) { toast("Box too small — click two opposite corners."); render(); return; }
  S.rev.pending = bbox; S.rev.pendingLabel = ""; S.rev.pendingType = "object";
  S.rev.pendingFrame = curFrame().frame_id;
  render();
  setTimeout(() => { const el = document.getElementById("revLabel"); if (el) el.focus(); }, 30);
};
ACT.revAddMissed = () => {
  const el = document.getElementById("revLabel");
  const label = ((el && el.value) || S.rev.pendingLabel || "").trim();
  if (!S.rev.pending) return;
  if (!label) { toast("Give the missed object a label."); return; }
  // file it under the frame it was drawn on, never wherever the user is now
  const e = revEntry(S.rev.pendingFrame || curFrame().frame_id);
  e.missed.push({ bbox: S.rev.pending, label, type: S.rev.pendingType });
  revClearPending(); S.rev.draw = false;
  revTouch();
};
ACT.revCancelMissed = () => { revClearPending(); render(); };
ACT.revDelMissed = (idx) => {
  const e = revEntry(curFrame().frame_id);
  e.missed.splice(+idx, 1);
  revTouch();
};
ACT.revConfirm = () => {
  const f = curFrame(); const e = revEntry(f.frame_id);
  if (Object.keys(e.verdicts).length < f.detections.length) {
    toast("Review every detection first (click its box or TP/FP)."); return;
  }
  e.done = true;
  revTouch();
  // auto-advance to the next unreviewed frame — within the active filter, so
  // the selection never lands on a frame the gallery isn't showing
  const frames = S.res.data.frames;
  const vis = visibleIndices();
  const pos = Math.max(0, vis.indexOf(S.res.sel));
  for (let k = 1; k <= vis.length; k++) {
    const j = vis[(pos + k) % vis.length];
    const fe = S.rev.doc.frames[frames[j].frame_id];
    if (!fe || !fe.done) { setSel(j); break; }
  }
  S._kbNav = true;
  render();
};

/* --- labels + lora screens --- */
ACT.openJobReview = async (jobId) => {
  await openResults(jobId);
  if (!revActive()) await ACT.toggleReview();
  // jump straight to the first unreviewed frame
  if (revActive()) {
    const frames = S.res.data.frames;
    const i = frames.findIndex(f => { const e = S.rev.doc.frames[f.frame_id]; return !e || !e.done; });
    if (i >= 0) setSel(i);
  }
  render();
};
ACT.deleteReviewOf = async (jobId) => {
  if (!confirm(`Delete the whole review of ${jobId}? All TP/FP verdicts and missed boxes of this job are lost.`)) return;
  try {
    await jpost(`/api/jobs/${jobId}/review`, undefined, "DELETE");
    if (S.rev.jobId === jobId) Object.assign(S.rev, { on: false, jobId: null, doc: null, metrics: null });
    toast("Review deleted.");
  } catch (e) { toast(e.message); }
  await refreshLabels();
};
ACT.exportReviewXlsx = (jobId) => { location.href = `/api/jobs/${jobId}/export.xlsx`; };
ACT.loraExport = async () => {
  if (S.lora.exporting) return;
  S.lora.exporting = true; S.lora.result = null; render();
  try {
    S.lora.result = await jpost("/api/lora/export", {});
    toast("Dataset exported.");
  } catch (e) { toast("Export failed: " + e.message); }
  S.lora.exporting = false;
  await refreshLora();
};

/* --- settings --- */
ACT.testOllama = async () => {
  S.ollamaTest = "testing"; render();
  await refreshHealth();
  S.ollamaTest = S.health && S.health.ollama ? "ok" : "fail"; render();
};
ACT.saveOllamaUrl = async (url) => {
  try { await jpost("/api/settings", { ollama_url: url }); S.ollamaTest = null; toast("Ollama URL saved."); }
  catch (e) { toast(e.message); }
};
ACT.deleteVideo = async (videoId) => {
  if (!confirm(`Delete extracted video ${videoId}?\nJobs that used these frames keep their results but lose the frame images in the gallery.`)) return;
  try { await jpost(`/api/videos/${videoId}`, undefined, "DELETE"); toast("Video deleted."); }
  catch (e) { toast(e.message); }
  await refreshStorage();
};
ACT.deleteJob = async (jobId) => {
  if (!confirm(`Delete job ${jobId} and all its results?`)) return;
  try { await jpost(`/api/jobs/${jobId}`, undefined, "DELETE"); toast("Job deleted."); }
  catch (e) { toast(e.message); }
  await Promise.all([refreshStorage(), refreshJobs()]);
  render();
};

/* change-events (inputs) */
const CHANGE = {
  extractN: v => { S.wiz.extractN = Math.max(0.1, parseFloat(v) || 1); invalidateExtraction(); render(); },
  trimStart: v => { S.wiz.trimStart = Math.min(+v, S.wiz.trimEnd - 2); invalidateExtraction(); render(); },
  trimEnd: v => { S.wiz.trimEnd = Math.max(+v, S.wiz.trimStart + 2); invalidateExtraction(); render(); },
  promptText: v => { S.wiz.promptText = v; S.wiz.promptPreset = "custom"; render(); },
  promptPreset: v => ACT.setPromptPreset(v),
  refPath: v => { S.wiz.refPath = v; render(); },
  maskPreset: v => ACT.setMaskPreset(v),
  diff: v => { S.wiz.diff = +v; }, minArea: v => { S.wiz.minArea = +v; },
  maxRegions: v => { S.wiz.maxRegions = +v; }, retries: v => { S.wiz.retries = +v; },
  compareJob: v => ACT.setCompareJob(v),
  ollamaUrl: v => ACT.saveOllamaUrl(v),
  revLabel: v => { S.rev.pendingLabel = v; },
  revType: v => { S.rev.pendingType = v; },
  refFile: async (_, input) => {
    const file = input.files[0];
    if (!file) return;
    const fd = new FormData();
    fd.append("file", file);
    S.wiz.refUploading = true; render();
    try {
      const r = await fetch("/api/references", { method: "POST", body: fd });
      if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
      const ref = await r.json();
      S.refs.push(ref);
      S.wiz.refPath = ref.path;
      toast(`Reference '${ref.name}' uploaded.`);
    } catch (e) { toast("Upload failed: " + e.message); }
    S.wiz.refUploading = false;
    input.value = "";
    render();
  },
  videoFile: async (_, input) => {
    const file = input.files[0];
    if (!file) return;
    const fd = new FormData();
    fd.append("file", file);
    S.wiz.uploading = file.name; render();     // persistent spinner in step 1
    try {
      const r = await fetch("/api/videos", { method: "POST", body: fd });
      if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
      S.wiz.video = await r.json(); S.wiz.frames = [];
      toast(`Video loaded: ${S.wiz.video.info.frame_count} frames, ${S.wiz.video.info.duration_s.toFixed(1)}s.`);
    } catch (e) { toast("Upload failed: " + e.message); }
    S.wiz.uploading = false;
    input.value = "";      // allow re-selecting the same file later
    render();
  },
};

/* ---------------------------------------------------------------- render */
function render() {
  const app = $app();
  if (!app) return;
  // full re-render loses scroll positions: save/restore containers that
  // declare data-scroll="key" (the gallery bug: clicking a frame reset it)
  const scrollPos = {};
  app.querySelectorAll("[data-scroll]").forEach(el => {
    scrollPos[el.dataset.scroll] = { top: el.scrollTop, left: el.scrollLeft };
  });
  // innerHTML destroys the focused field: without this, a render landing while
  // the user types (autosave, an expiring toast) dropped focus to <body> and
  // the next keystrokes were read as review shortcuts — 'a' confirmed a frame
  // with every box marked TP and jumped to the next one.
  const act = document.activeElement;
  const focus = act && act.id && app.contains(act)
    ? { id: act.id, start: act.selectionStart, end: act.selectionEnd } : null;
  app.innerHTML = `
  <div class="arsi-app${S.theme === "light" ? " light" : ""}">
    ${sidebar()}
    <div style="flex:1; display:flex; flex-direction:column; min-width:0;">
      ${topbar()}
      <div style="flex:1; min-height:0; overflow:hidden; position:relative;">
        ${S.screen === "dashboard" ? dashboard() : ""}
        ${S.screen === "wizard" ? wizard() : ""}
        ${S.screen === "run" ? runView() : ""}
        ${S.screen === "results" ? resultsView() : ""}
        ${S.screen === "history" ? historyView() : ""}
        ${S.screen === "labels" ? labelsView() : ""}
        ${S.screen === "lora" ? loraView() : ""}
        ${S.screen === "settings" ? settingsView() : ""}
      </div>
    </div>
    ${S.toast ? `<div style="position:fixed; bottom:22px; left:50%; transform:translateX(-50%); z-index:60; background:${C.bgBtn}; border:1px solid oklch(0.36 0.014 250); color:${C.fg}; font-size:13px; padding:11px 18px; border-radius:10px; box-shadow:0 12px 34px -12px rgba(0,0,0,0.7); display:flex; align-items:center; gap:10px;"><span style="width:7px; height:7px; border-radius:50%; background:${C.acc};"></span>${esc(S.toast)}</div>` : ""}
  </div>`;
  app.querySelectorAll("[data-scroll]").forEach(el => {
    const p = scrollPos[el.dataset.scroll];
    if (p) { el.scrollTop = p.top; el.scrollLeft = p.left; }
  });
  if (focus) {
    const el = document.getElementById(focus.id);
    if (el) {
      el.focus();
      // setSelectionRange throws on input types that have no text selection
      try { el.setSelectionRange(focus.start, focus.end); } catch { /* not a text field */ }
    }
  }
  const gsel = app.querySelector("#gsel");
  if (gsel && S._kbNav) { gsel.scrollIntoView({ block: "nearest" }); S._kbNav = false; }
  hideTip();   // the hovered ? icon may not exist in the new DOM
}

function navItem(key, label, icon, extra = "") {
  const on = S.screen === key || (key === "wizard" && S.screen === "wizard");
  const bg = on ? C.accBg : "transparent", fg = on ? C.accFg : "oklch(0.68 0.012 250)";
  const act = key === "wizard" ? "goWizard" : "go";
  return `<div class="navitem" data-act="${act}" data-arg="${key}" style="display:flex; align-items:center; gap:10px; padding:9px 10px; border-radius:8px; font-size:13px; margin-bottom:2px; cursor:pointer; background:${bg}; color:${fg};">${icon} ${label}${extra}</div>`;
}
const I = {   // 16px stroke icons from the design
  dash: `<svg width="16" height="16" viewBox="0 0 18 18" fill="none"><rect x="1" y="1" width="7" height="7" rx="1" stroke="currentColor" stroke-width="1.4"/><rect x="10" y="1" width="7" height="7" rx="1" stroke="currentColor" stroke-width="1.4"/><rect x="1" y="10" width="7" height="7" rx="1" stroke="currentColor" stroke-width="1.4"/><rect x="10" y="10" width="7" height="7" rx="1" stroke="currentColor" stroke-width="1.4"/></svg>`,
  plus: `<svg width="16" height="16" viewBox="0 0 18 18" fill="none"><rect x="1.5" y="1.5" width="15" height="15" rx="3" stroke="currentColor" stroke-width="1.4"/><line x1="9" y1="5" x2="9" y2="13" stroke="currentColor" stroke-width="1.4"/><line x1="5" y1="9" x2="13" y2="9" stroke="currentColor" stroke-width="1.4"/></svg>`,
  play: `<svg width="16" height="16" viewBox="0 0 18 18" fill="none"><path d="M5 3 L14 9 L5 15 Z" stroke="currentColor" stroke-width="1.4" stroke-linejoin="round"/></svg>`,
  img: `<svg width="16" height="16" viewBox="0 0 18 18" fill="none"><rect x="1.5" y="3" width="15" height="12" rx="1.5" stroke="currentColor" stroke-width="1.4"/><circle cx="6" cy="8" r="1.6" fill="currentColor"/><path d="M2 13 L7 9 L11 12 L16 8" stroke="currentColor" stroke-width="1.4" fill="none"/></svg>`,
  clock: `<svg width="16" height="16" viewBox="0 0 18 18" fill="none"><circle cx="9" cy="9" r="7.5" stroke="currentColor" stroke-width="1.4"/><path d="M9 5 V9 L12 11" stroke="currentColor" stroke-width="1.4" stroke-linecap="round"/></svg>`,
  sliders: `<svg width="16" height="16" viewBox="0 0 18 18" fill="none"><line x1="2" y1="5" x2="16" y2="5" stroke="currentColor" stroke-width="1.4"/><line x1="2" y1="9" x2="16" y2="9" stroke="currentColor" stroke-width="1.4"/><line x1="2" y1="13" x2="16" y2="13" stroke="currentColor" stroke-width="1.4"/><circle cx="12" cy="5" r="2.2" fill="${C.bgSide}" stroke="currentColor" stroke-width="1.4"/><circle cx="6" cy="9" r="2.2" fill="${C.bgSide}" stroke="currentColor" stroke-width="1.4"/><circle cx="13" cy="13" r="2.2" fill="${C.bgSide}" stroke="currentColor" stroke-width="1.4"/></svg>`,
  tag: `<svg width="16" height="16" viewBox="0 0 18 18" fill="none"><path d="M2 2 H8.5 L16 9.5 A1.5 1.5 0 0 1 16 11.6 L11.6 16 A1.5 1.5 0 0 1 9.5 16 L2 8.5 Z" stroke="currentColor" stroke-width="1.4" stroke-linejoin="round"/><circle cx="6" cy="6" r="1.3" fill="currentColor"/></svg>`,
  spark: `<svg width="16" height="16" viewBox="0 0 18 18" fill="none"><path d="M9 1.5 L10.8 7.2 L16.5 9 L10.8 10.8 L9 16.5 L7.2 10.8 L1.5 9 L7.2 7.2 Z" stroke="currentColor" stroke-width="1.4" stroke-linejoin="round"/></svg>`,
};

function sidebar() {
  const h = S.health;
  const running = S.run && !S.run.done ? 1 : 0;
  const online = h && h.ollama;
  const nInstalled = S.models.filter(m => m.installed).length;
  return `
  <div style="width:248px; flex:0 0 248px; background:${C.bgSide}; border-right:1px solid oklch(0.26 0.012 250); display:flex; flex-direction:column;">
    <div style="padding:18px 18px 14px; border-bottom:1px solid oklch(0.22 0.01 250); display:flex; align-items:center; gap:11px;">
      <div style="width:32px; height:32px; border-radius:8px; background:${C.acc}; display:flex; align-items:center; justify-content:center; font-family:${C.mono}; font-weight:800; font-size:15px; color:${C.accDark};">A</div>
      <div>
        <div style="font-size:14px; font-weight:700; letter-spacing:-0.01em;">ARSI Studio</div>
        <div style="font-family:${C.mono}; font-size:10px; color:${C.fg4}; letter-spacing:0.04em;">tram anomaly lab</div>
      </div>
    </div>
    <div style="margin:14px 12px; padding:12px; border-radius:10px; background:${C.bgCard2}; border:1px solid oklch(0.26 0.012 250);">
      <div style="font-size:10px; text-transform:uppercase; letter-spacing:0.09em; color:${C.fg5}; margin-bottom:9px; font-family:${C.mono};">System</div>
      <div style="display:flex; align-items:center; gap:8px; margin-bottom:8px;">
        <span style="width:7px; height:7px; border-radius:50%; background:${online ? C.green : C.red}; box-shadow:0 0 8px ${online ? C.green : C.red};"></span>
        <span style="font-size:12.5px; color:${online ? "oklch(0.85 0.05 150)" : "oklch(0.85 0.06 22)"}; font-weight:500;">Ollama ${online ? "online" : "offline"}</span>
      </div>
      ${h && h.gpu
        ? `<div style="display:flex; align-items:center; gap:8px; margin-bottom:10px; padding:6px 8px; border-radius:7px; background:oklch(0.22 0.05 145); border:1px solid oklch(0.4 0.09 150);"><span style="font-size:12px; color:oklch(0.85 0.08 150); font-weight:600;">GPU detected</span></div>`
        : `<div style="display:flex; align-items:center; gap:8px; margin-bottom:10px; padding:6px 8px; border-radius:7px; background:oklch(0.24 0.05 75); border:1px solid oklch(0.44 0.09 75);"><span style="font-size:12px; color:oklch(0.88 0.1 75); font-weight:600;">CPU only · 2-4 min/call</span></div>`}
      <div style="display:flex; justify-content:space-between; font-size:11.5px; color:${C.fg3};">
        <span>Models</span><span style="font-family:${C.mono}; color:oklch(0.88 0.006 250);">${nInstalled} installed</span>
      </div>
    </div>
    <div style="flex:1; overflow:auto; padding:4px 12px;">
      <div style="font-size:10px; text-transform:uppercase; letter-spacing:0.09em; color:${C.dim}; margin:10px 8px 6px; font-family:${C.mono};">Workspace</div>
      ${navItem("dashboard", "Dashboard", I.dash)}
      ${navItem("wizard", "New analysis", I.plus)}
      ${navItem("run", `Runs <span style="margin-left:auto; font-size:10.5px; font-family:${C.mono}; color:${C.acc}; background:${C.accBg}; padding:1px 6px; border-radius:8px;">${running}</span>`, I.play)}
      <div style="font-size:10px; text-transform:uppercase; letter-spacing:0.09em; color:${C.dim}; margin:16px 8px 6px; font-family:${C.mono};">Library</div>
      ${navItem("results", "Results", I.img)}
      ${navItem("labels", "Labels", I.tag)}
      ${navItem("history", "History", I.clock)}
      <div style="font-size:10px; text-transform:uppercase; letter-spacing:0.09em; color:${C.dim}; margin:16px 8px 6px; font-family:${C.mono};">Training</div>
      ${navItem("lora", "LoRA", I.spark)}
      <div style="font-size:10px; text-transform:uppercase; letter-spacing:0.09em; color:${C.dim}; margin:16px 8px 6px; font-family:${C.mono};">System</div>
      ${navItem("settings", "Settings", I.sliders)}
    </div>
  </div>`;
}

function topbar() {
  const titles = { dashboard: "Dashboard", wizard: "New analysis", run: "Run",
                   results: "Results", history: "History", settings: "Settings",
                   labels: "Labels", lora: "LoRA training data" };
  const themeIcon = S.theme === "dark"
    ? `<svg width="17" height="17" viewBox="0 0 18 18" fill="none"><path d="M15.5 10.5 A6.8 6.8 0 0 1 7.5 2.5 A6.8 6.8 0 1 0 15.5 10.5 Z" stroke="currentColor" stroke-width="1.4" stroke-linejoin="round"/></svg>`
    : `<svg width="17" height="17" viewBox="0 0 18 18" fill="none"><circle cx="9" cy="9" r="3.4" stroke="currentColor" stroke-width="1.4"/><g stroke="currentColor" stroke-width="1.4" stroke-linecap="round"><line x1="9" y1="1.5" x2="9" y2="3"/><line x1="9" y1="15" x2="9" y2="16.5"/><line x1="1.5" y1="9" x2="3" y2="9"/><line x1="15" y1="9" x2="16.5" y2="9"/></g></svg>`;
  let right = "";
  if (S.screen === "dashboard")
    right = `<button data-act="goWizard" style="font-size:13px; font-weight:600; color:${C.accDark}; background:${C.acc}; border:none; padding:9px 16px; border-radius:8px; cursor:pointer;">+ New analysis</button>`;
  if (S.screen === "results" && S.res.data) {
    right = `
      <div style="display:flex; align-items:center; gap:8px; margin-right:6px;">
        <span style="font-size:12.5px; color:oklch(0.65 0.012 250);">Compare configs</span>
        <div data-act="toggleCompare" style="width:38px; height:21px; border-radius:12px; background:${S.res.compare ? "oklch(0.55 0.11 225)" : C.bd}; position:relative; cursor:pointer;">
          <span style="position:absolute; top:2px; left:${S.res.compare ? "19px" : "2px"}; width:17px; height:17px; border-radius:50%; background:oklch(0.96 0 0); transition:left .15s;"></span>
        </div>
      </div>
      <button data-act="exportReport" style="font-size:12.5px; color:oklch(0.85 0.012 250); background:${C.bgBtn}; border:1px solid ${C.bdBtn}; padding:8px 12px; border-radius:8px; cursor:pointer;">Report</button>
      <button data-act="exportJson" style="font-size:12.5px; color:oklch(0.85 0.012 250); background:${C.bgBtn}; border:1px solid ${C.bdBtn}; padding:8px 12px; border-radius:8px; cursor:pointer;">results.json</button>
      <button data-act="exportXlsx" style="font-size:12.5px; color:oklch(0.85 0.012 250); background:${C.bgBtn}; border:1px solid ${C.bdBtn}; padding:8px 12px; border-radius:8px; cursor:pointer;">XLSX</button>`;
  }
  return `
  <div style="height:60px; flex:0 0 60px; border-bottom:1px solid oklch(0.26 0.012 250); display:flex; align-items:center; padding:0 26px; gap:16px;">
    <div style="font-size:17px; font-weight:650; letter-spacing:-0.01em;">${titles[S.screen] || ""}</div>
    <div style="margin-left:auto; display:flex; gap:10px; align-items:center;">
      <button data-act="toggleTheme" title="Switch theme" style="width:34px; height:34px; display:flex; align-items:center; justify-content:center; color:oklch(0.72 0.012 250); background:oklch(0.18 0.01 250); border:1px solid ${C.bd3}; border-radius:8px; cursor:pointer;">${themeIcon}</button>
      ${right}
    </div>
  </div>`;
}

/* ---------------------------------------------------------------- screens */
function statusMeta(st) {
  if (st === "running" || st === "queued")
    return { fg: C.acc, bg: C.accBg, bd: C.accBd, label: st };
  if (st === "failed") return { fg: C.redFg, bg: C.redBg, bd: C.redBd, label: "Failed" };
  if (st === "cancelled") return { fg: C.fg2, bg: "oklch(0.22 0.012 250)", bd: C.bdBtn, label: "Cancelled" };
  return { fg: "oklch(0.82 0.11 150)", bg: C.greenBg, bd: C.greenBd, label: "Complete" };
}
function jobRow(j, cols) {
  const m = statusMeta(j.status);
  const s = j.summary || {};
  const anomC = (s.n_anomalous || 0) > 0 ? C.red : C.fg3;
  return `
  <div class="hoverable" data-act="openJob" data-arg="${esc(j.job_id)}" style="display:grid; grid-template-columns:${cols}; padding:13px 16px; border-top:1px solid oklch(0.24 0.01 250); align-items:center; font-size:13px; cursor:pointer;">
    <div><div style="font-family:${C.mono}; color:oklch(0.9 0.006 250);">${esc(j.job_id)}</div>
      <div style="font-size:11px; color:${C.fg4}; margin-top:2px;">${esc(j.config.script)} · ${esc(j.config.model || "")}</div></div>
    <span style="text-align:right; font-family:${C.mono}; color:${C.fg2};">${s.n_frames ?? j.config.n_frames ?? ""}</span>
    <span style="text-align:right; font-family:${C.mono}; color:${anomC}; font-weight:600;">${s.n_anomalous ?? "—"}</span>
    <span style="display:flex; justify-content:flex-end;"><span style="font-size:11.5px; color:${m.fg}; background:${m.bg}; border:1px solid ${m.bd}; padding:3px 9px; border-radius:12px;">${m.label}</span></span>
  </div>`;
}

function dashboard() {
  const jobs = S.jobs.slice(0, 5);
  const total = S.jobs.length;
  const anom = S.jobs.reduce((a, j) => a + ((j.summary || {}).n_anomalous || 0), 0);
  const card = (act, title, sub, accent) => `
    <div data-act="${act}" style="background:${accent ? `linear-gradient(160deg, oklch(0.24 0.05 225), oklch(0.2 0.02 225))` : C.bgBtn}; border:1px solid ${accent ? C.accBd : C.bd3}; border-radius:11px; padding:16px; display:flex; gap:13px; align-items:center; cursor:pointer;">
      <div style="width:38px; height:38px; flex:0 0 38px; border-radius:9px; background:${accent ? C.acc : "oklch(0.28 0.014 250)"}; display:flex; align-items:center; justify-content:center; color:${accent ? C.accDark : "oklch(0.85 0.006 250)"};">${accent ? I.play : I.dash}</div>
      <div><div style="font-size:14px; font-weight:600;">${title}</div><div style="font-size:11.5px; color:${accent ? "oklch(0.75 0.04 225)" : "oklch(0.62 0.012 250)"};">${sub}</div></div>
    </div>`;
  const stat = (v, label, color = "") => `
    <div style="background:${C.bgCard}; border:1px solid ${C.bd}; border-radius:10px; padding:14px 16px;">
      <div style="font-family:${C.mono}; font-size:24px; font-weight:700; ${color ? "color:" + color : ""}">${v}</div>
      <div style="font-size:11.5px; color:${C.fg3}; margin-top:2px;">${label}</div></div>`;
  return `
  <div data-scroll="page" style="height:100%; overflow:auto; padding:24px 28px;">
    <div style="display:grid; grid-template-columns:repeat(3,1fr); gap:14px; margin-bottom:24px;">
      ${card("goWizard", "Analyze a video", "Upload, extract, run a pipeline", true)}
      ${card("goWizardDemo", `Try demo frames`, `${S.demo.length} bundled cases`, false)}
      ${card("resumeLast", "Resume last", S.run ? `${S.run.jobId} · ${S.run.processed}/${S.run.total}` : (S.jobs[0] ? S.jobs[0].job_id : "no jobs yet"), false)}
    </div>
    <div style="display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin-bottom:24px;">
      ${stat(total, "Jobs total")}
      ${stat(anom, "Anomalous frames", C.red)}
      ${stat(S.jobs.reduce((a, j) => a + ((j.summary || {}).n_frames || 0), 0), "Frames analyzed", C.green)}
      ${stat(S.health && S.health.gpu ? "0.7<span style='font-size:13px; color:" + C.fg3 + "'>s</span>" : "2-4<span style='font-size:13px; color:" + C.fg3 + "'>min</span>", "Per VLM call (" + (S.health && S.health.gpu ? "GPU" : "CPU") + ")")}
    </div>
    <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:10px;">
      <h3 style="margin:0; font-size:13px; font-weight:600; text-transform:uppercase; letter-spacing:0.08em; color:${C.fg3};">Recent jobs</h3>
      <span data-act="go" data-arg="history" style="font-size:12px; color:${C.acc}; cursor:pointer;">View all</span>
    </div>
    <div style="border:1px solid ${C.bd}; border-radius:11px; overflow:hidden; background:${C.bgCard};">
      <div style="display:grid; grid-template-columns:2.4fr 0.8fr 1fr 1.1fr; padding:10px 16px; background:${C.bg}; border-bottom:1px solid ${C.bd}; font-size:11px; text-transform:uppercase; letter-spacing:0.06em; color:${C.fg4}; font-family:${C.mono};">
        <span>Job</span><span style="text-align:right;">Frames</span><span style="text-align:right;">Anomalies</span><span style="text-align:right;">Status</span>
      </div>
      ${jobs.length ? jobs.map(j => jobRow(j, "2.4fr 0.8fr 1fr 1.1fr")).join("") :
        `<div style="padding:32px; text-align:center; color:${C.fg4}; font-size:12.5px;">No jobs yet — run your first analysis.</div>`}
    </div>
  </div>`;
}

/* --------------- wizard --------------- */
function wizard() {
  const w = S.wiz;
  const steps = [["Source", 1], ["Extract", 2], ["Mask", 3], ["Pipeline", 4], ["Review", 5]];
  const stepper = steps.map(([lb, n]) => `
    <div data-act="wizGoto" data-arg="${n}" style="display:flex; align-items:center; gap:9px; cursor:pointer;">
      <span style="width:24px; height:24px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:12px; font-family:${C.mono}; font-weight:600; background:${w.step > n ? C.acc : (w.step === n ? C.accSel : C.bgBtn)}; color:${w.step >= n ? "oklch(0.95 0.03 225)" : C.fg4}; border:1px solid ${w.step >= n ? C.accSelBd : C.bd3};">${n}</span>
      <span style="font-size:13px; color:${w.step >= n ? "oklch(0.9 0.006 250)" : C.fg4};">${lb}</span>
    </div><span style="width:26px; height:1px; background:${C.bd3}; margin:0 4px;"></span>`).join("");
  return `
  <div style="height:100%; display:flex; flex-direction:column;">
    <div style="flex:0 0 auto; padding:16px 28px; border-bottom:1px solid ${C.bd2}; display:flex; align-items:center; gap:6px;">${stepper}</div>
    <div data-scroll="wiz" style="flex:1; min-height:0; overflow:auto; padding:26px 28px 40px;">
      ${w.step === 1 ? wizStep1() : ""}${w.step === 2 ? wizStep2() : ""}${w.step === 3 ? wizStep3() : ""}${w.step === 4 ? wizStep4() : ""}${w.step === 5 ? wizStep5() : ""}
    </div>
    <div style="flex:0 0 auto; padding:14px 28px; border-top:1px solid ${C.bd2}; display:flex; align-items:center; gap:12px; background:oklch(0.13 0.008 250);">
      ${w.step > 1 ? `<button data-act="wizBack" style="font-size:13px; color:oklch(0.82 0.012 250); background:${C.bgBtn}; border:1px solid ${C.bdBtn}; padding:10px 18px; border-radius:9px; cursor:pointer;">Back</button>` : ""}
      <span style="margin-left:auto;"></span>
      ${w.step !== 5 ? `<button data-act="wizNext" style="font-size:13px; font-weight:600; color:${C.accDark}; background:${C.acc}; border:none; padding:10px 22px; border-radius:9px; cursor:pointer;">${w.extracting ? "Extracting…" : "Continue"}</button>` : ""}
    </div>
  </div>`;
}
function wizStep1() {
  const w = S.wiz;
  const cards = [
    { key: "video", title: "Upload a video", desc: "Drag and drop / browse an .mp4 .mkv .avi clip" },
    { key: "demo", title: "Bundled demo frames", desc: `${S.demo.length} real and synthetic labelled cases` },
    { key: "reuse", title: "Reuse an extraction", desc: "Pick frames from a previous video" },
  ].map(c => `
    <div data-act="setSource" data-arg="${c.key}" style="padding:18px; border-radius:11px; cursor:pointer; background:${w.source === c.key ? C.accBg : "oklch(0.18 0.01 250)"}; border:1px solid ${w.source === c.key ? C.accSelBd : C.bd3};">
      <div style="font-size:14.5px; font-weight:600; margin-bottom:4px;">${c.title}</div>
      <div style="font-size:12px; color:oklch(0.65 0.012 250); line-height:1.45;">${c.desc}</div>
    </div>`).join("");
  let body = "";
  if (w.source === "video") {
    body = w.uploading ? `
      <div style="border:1.5px dashed ${C.accBd}; border-radius:12px; padding:44px; text-align:center; background:${C.bgCard2};">
        <div style="width:34px; height:34px; margin:0 auto 14px; border-radius:50%; border:3px solid ${C.bd3}; border-top-color:${C.acc}; animation:arsispin 0.9s linear infinite;"></div>
        <div style="font-size:14px; font-weight:500; margin-bottom:4px;">Uploading &amp; probing <span style="font-family:${C.mono};">${esc(w.uploading)}</span>…</div>
        <div style="font-size:12px; color:${C.fg3};">Large files take a while — the filmstrip appears when it's done.</div>
      </div>` : w.video ? `
      <div style="padding:16px 18px; border-radius:11px; background:${C.accBg}; border:1px solid ${C.accBd};">
        <div style="font-size:13.5px; font-weight:600; margin-bottom:4px;">Video loaded</div>
        <div style="font-family:${C.mono}; font-size:12px; color:oklch(0.8 0.05 225);">${w.video.info.frame_count} frames · ${w.video.info.duration_s.toFixed(1)}s · ${w.video.info.width}×${w.video.info.height} · ${w.video.info.fps.toFixed(1)} fps</div>
        <div style="display:flex; gap:4px; margin-top:12px; border-radius:8px; overflow:hidden;">${(w.video.thumbs || []).map(t => `<img src="${t}" style="flex:1; height:44px; object-fit:cover; min-width:0;">`).join("")}</div>
      </div>` : `
      <div data-act="pickVideoFile" style="border:1.5px dashed oklch(0.36 0.014 250); border-radius:12px; padding:44px; text-align:center; background:${C.bgCard2}; cursor:pointer;">
        <div style="font-size:14px; font-weight:500; margin-bottom:4px;">Drop a video file here</div>
        <div style="font-size:12px; color:${C.fg3};">.mp4 .mkv .avi — or <span style="color:${C.acc};">browse</span></div>
      </div>`;
  } else if (w.source === "reuse") {
    const pool = w.framePool || [], sel = w.frameSel || [];
    const poolGrid = pool.length ? `
      <div style="display:flex; align-items:center; justify-content:space-between; margin:16px 0 10px;">
        <span style="font-size:12px; text-transform:uppercase; letter-spacing:0.08em; color:${C.fg4}; font-family:${C.mono};">Frames · ${sel.length}/${pool.length} selected</span>
        <span><span data-act="poolAll" style="font-size:12px; color:${C.acc}; cursor:pointer; margin-right:12px;">Select all</span><span data-act="poolNone" style="font-size:12px; color:${C.fg3}; cursor:pointer;">Clear</span></span>
      </div>
      <div style="display:grid; grid-template-columns:repeat(auto-fill,minmax(118px,1fr)); gap:8px;">
        ${pool.map(f => {
          const on = sel.includes(f.path);
          return `
          <div data-act="togglePoolFrame" data-arg="${esc(f.path)}" style="border-radius:8px; overflow:hidden; cursor:pointer; position:relative; outline:${on ? "2px solid oklch(0.8 0.13 225)" : "2px solid transparent"}; outline-offset:1px; opacity:${on ? 1 : 0.55};">
            <img src="${f.img}" loading="lazy" style="width:100%; height:64px; object-fit:cover; display:block;">
            ${on ? `<span style="position:absolute; top:4px; right:4px; width:16px; height:16px; border-radius:50%; background:${C.acc}; color:${C.accDark}; font-size:11px; font-weight:800; display:flex; align-items:center; justify-content:center;">✓</span>` : ""}
            <div style="font-family:${C.mono}; font-size:9px; color:${C.fg3}; padding:3px 5px; background:${C.bg}; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">${esc(f.path.split("/").pop())} · ${f.time_s.toFixed(1)}s</div>
          </div>`;
        }).join("")}
      </div>` : "";
    body = `<div style="font-size:12px; text-transform:uppercase; letter-spacing:0.08em; color:${C.fg4}; font-family:${C.mono}; margin-bottom:12px;">Previous extractions</div>
      <div style="display:flex; flex-direction:column; gap:8px;">${
      (S._videos || []).map(v => `
        <div class="hoverable" data-act="reuseVideo" data-arg="${esc(v.video_id)}" style="padding:13px 15px; border-radius:10px; cursor:pointer; background:${w.reuseId === v.video_id ? C.accBg : C.bgCard}; border:1px solid ${w.reuseId === v.video_id ? C.accSelBd : C.bd}; display:flex; justify-content:space-between; font-size:13px;">
          <span style="font-family:${C.mono};">${esc(v.video_id)}</span><span style="color:${C.fg3};">${v.n_frames} frames</span>
        </div>`).join("") || `<div style="color:${C.fg4}; font-size:12.5px;">No previous extraction found.</div>`}</div>
      ${poolGrid}`;
    if (!S._videos) jget("/api/videos").then(d => { S._videos = d.videos; render(); });
  } else {
    body = `
    <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:12px;">
      <span style="font-size:12px; text-transform:uppercase; letter-spacing:0.08em; color:${C.fg4}; font-family:${C.mono};">Bundled demo frames · ${w.demoSel.length} selected</span>
      <span><span data-act="demoAll" style="font-size:12px; color:${C.acc}; cursor:pointer; margin-right:12px;">Select all</span><span data-act="demoNone" style="font-size:12px; color:${C.fg3}; cursor:pointer;">Clear</span></span>
    </div>
    <div style="display:grid; grid-template-columns:repeat(auto-fill,minmax(148px,1fr)); gap:12px;">
      ${S.demo.map(d => {
        const on = w.demoSel.includes(d.id);
        const kind = d.source === "real" ? "real" : "synthetic";
        return `
        <div data-act="pickDemo" data-arg="${esc(d.id)}" style="border-radius:10px; overflow:hidden; cursor:pointer; outline:${on ? "2px solid oklch(0.8 0.13 225)" : "2px solid transparent"}; outline-offset:2px; background:${C.bgCard2}; border:1px solid oklch(0.26 0.012 250);">
          <img src="${d.img}" loading="lazy" style="width:100%; height:88px; object-fit:cover; display:block;">
          <div style="padding:8px 9px;">
            <span style="font-size:9.5px; padding:2px 7px; border-radius:8px; background:${kind === "real" ? "oklch(0.26 0.06 150)" : "oklch(0.26 0.06 300)"}; color:${kind === "real" ? "oklch(0.88 0.09 150)" : "oklch(0.86 0.1 300)"};">${kind}</span>
            ${d.anomaly ? `<span style="font-size:9.5px; padding:2px 7px; border-radius:8px; margin-left:4px; background:oklch(0.3 0.1 22); color:oklch(0.9 0.12 22);">ANOM</span>` : ""}
            <div style="font-size:12px; color:oklch(0.88 0.006 250); margin-top:5px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">${esc(d.label || d.id)}</div>
            <div style="font-family:${C.mono}; font-size:9.5px; color:${C.fg5}; margin-top:2px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">${esc(d.id)}</div>
          </div>
        </div>`;
      }).join("")}
    </div>`;
  }
  return `<div style="max-width:900px;">
    <div style="display:grid; grid-template-columns:repeat(3,1fr); gap:12px; margin-bottom:24px;">${cards}</div>${body}</div>`;
}
function wizStep2() {
  const w = S.wiz;
  if (w.source !== "video") {
    return `<div style="max-width:820px;"><div style="padding:18px 20px; border-radius:11px; background:oklch(0.18 0.01 250); border:1px solid ${C.bd3}; color:oklch(0.72 0.012 250); font-size:13px; line-height:1.5;">
      You picked ${w.source === "reuse" ? "an existing extraction" : "demo frames"}, so there is nothing to extract. The ${w.frames.length || w.demoSel.length} frames are used as-is. Continue to the mask editor.</div></div>`;
  }
  const dur = w.video ? w.video.info.duration_s : 0;
  const seg = (on) => on ? `background:${C.accSel}; color:${C.accFg};` : `background:transparent; color:oklch(0.65 0.012 250);`;
  return `
  <div style="max-width:820px;">
    <div style="font-size:13px; color:${C.fg3}; margin-bottom:16px;">Sampling rate ${hint("How many frames to analyze. 'Every N seconds' keeps 1 frame per N seconds of video (usual choice: 1-3 s — a forgotten object stays visible for many seconds). 'Every N frames' counts in raw video frames (25-30 per second).")}</div>
    <div style="display:inline-flex; padding:3px; border-radius:9px; background:${C.bg}; border:1px solid ${C.bd}; margin-bottom:20px;">
      <div data-act="setExtMode" data-arg="seconds" style="padding:7px 15px; border-radius:7px; font-size:12.5px; cursor:pointer; ${seg(w.extractMode === "seconds")}">Every N seconds</div>
      <div data-act="setExtMode" data-arg="frames" style="padding:7px 15px; border-radius:7px; font-size:12.5px; cursor:pointer; ${seg(w.extractMode === "frames")}">Every N frames</div>
    </div>
    <div style="display:flex; align-items:center; gap:12px; margin-bottom:26px;">
      <span style="font-size:13px; color:oklch(0.72 0.012 250);">N =</span>
      <input type="number" value="${w.extractN}" data-change="extractN" min="0.1" step="any" style="width:80px; padding:8px 10px; border-radius:8px; background:${C.bgCard2}; border:1px solid ${C.bd3}; color:${C.fg}; font-family:${C.mono}; font-size:13px;">
      <span style="font-size:12.5px; color:${C.fg4};">${w.extractMode === "seconds" ? "seconds between sampled frames" : "video frames skipped between samples"}</span>
    </div>
    <div style="font-size:13px; color:${C.fg3}; margin-bottom:10px;">Trim range · ${(w.trimStart / 100 * dur).toFixed(1)}s → ${(w.trimEnd / 100 * dur).toFixed(1)}s</div>
    <div style="position:relative; height:56px; border-radius:9px; overflow:hidden; background:${C.bgSide}; border:1px solid ${C.bd}; display:flex; margin-bottom:6px;">
      ${(w.video.thumbs || []).map(t => `<img src="${t}" style="flex:1; height:100%; object-fit:cover; opacity:0.5; min-width:0; border-right:1px solid oklch(0.1 0.008 250);">`).join("")}
      <div style="position:absolute; top:0; bottom:0; left:0; width:${w.trimStart}%; background:oklch(0.1 0.008 250 / 0.7);"></div>
      <div style="position:absolute; top:0; bottom:0; right:0; width:${100 - w.trimEnd}%; background:oklch(0.1 0.008 250 / 0.7);"></div>
      <div style="position:absolute; top:0; bottom:0; left:${w.trimStart}%; width:${w.trimEnd - w.trimStart}%; border:2px solid ${C.acc}; border-radius:4px; pointer-events:none;"></div>
    </div>
    <div style="display:flex; gap:20px; align-items:center; margin-bottom:24px;">
      <label style="font-size:12px; color:${C.fg3}; display:flex; flex-direction:column; gap:4px; flex:1;">Start <input type="range" min="0" max="98" value="${w.trimStart}" data-change="trimStart" style="width:100%;"></label>
      <label style="font-size:12px; color:${C.fg3}; display:flex; flex-direction:column; gap:4px; flex:1;">End <input type="range" min="2" max="100" value="${w.trimEnd}" data-change="trimEnd" style="width:100%;"></label>
    </div>
    <div style="display:inline-flex; align-items:center; gap:10px; padding:12px 18px; border-radius:10px; background:${C.accBg}; border:1px solid ${C.accBd};">
      <span style="font-family:${C.mono}; font-size:20px; font-weight:700; color:oklch(0.85 0.1 225);">${estFrames()}</span>
      <span style="font-size:13px; color:oklch(0.72 0.05 225);">frames estimated for analysis</span>
    </div>
  </div>`;
}
function wizStep3() {
  const w = S.wiz;
  const src = maskEditorImage();
  const seg = (on) => on ? `background:${C.accSel}; color:${C.accFg};` : `background:transparent; color:oklch(0.65 0.012 250);`;
  const zonesSvg = w.maskZones.map(z => `<polygon points="${z.pts.map(p => p.x + "," + p.y).join(" ")}" fill="${w.maskPreview ? "oklch(0 0 0)" : "oklch(0.72 0.13 225 / 0.22)"}" stroke="${w.maskPreview ? "oklch(0 0 0)" : "oklch(0.8 0.13 225)"}" stroke-width="0.4"></polygon>`).join("");
  const draft = w.draftPts.length ? `
    <polyline points="${w.draftPts.map(p => p.x + "," + p.y).join(" ")}" fill="oklch(0.72 0.13 225 / 0.15)" stroke="oklch(0.85 0.13 225)" stroke-width="0.4" stroke-dasharray="1 0.6"></polyline>
    ${w.draftPts.map(p => `<circle cx="${p.x}" cy="${p.y}" r="0.9" fill="oklch(0.95 0.03 225)" stroke="oklch(0.3 0.1 225)" stroke-width="0.25"></circle>`).join("")}` : "";
  const rectMark = w.rectStart ? `<circle cx="${w.rectStart.x}" cy="${w.rectStart.y}" r="1" fill="oklch(0.95 0.03 225)"></circle>` : "";
  const presetOpts = [...S.masks.map(m => `<option value="${esc(m.name)}" ${w.maskPreset === m.name ? "selected" : ""}>${esc(m.name)} (${m.zones.length} zones)</option>`),
    `<option value="custom" ${w.maskPreset === "custom" ? "selected" : ""}>Custom (unsaved)</option>`,
    `<option value="none" ${w.maskPreset === "none" ? "selected" : ""}>No mask</option>`].join("");
  return `
  <div style="display:flex; gap:20px; align-items:flex-start;">
    <div style="flex:1; min-width:0;">
      <div style="display:flex; align-items:center; gap:10px; margin-bottom:12px;">
        <div style="display:inline-flex; padding:3px; border-radius:9px; background:${C.bg}; border:1px solid ${C.bd};">
          <div data-act="setMaskTool" data-arg="poly" style="padding:6px 13px; border-radius:7px; font-size:12px; cursor:pointer; ${seg(w.maskTool === "poly")}">Polygon</div>
          <div data-act="setMaskTool" data-arg="rect" style="padding:6px 13px; border-radius:7px; font-size:12px; cursor:pointer; ${seg(w.maskTool === "rect")}">Rectangle</div>
        </div>
        ${w.draftPts.length ? `
          <button data-act="maskFinish" style="font-size:12px; font-weight:600; color:${C.accDark}; background:${C.acc}; border:none; padding:7px 13px; border-radius:8px; cursor:pointer;">Close polygon (${w.draftPts.length} pts)</button>
          <button data-act="maskUndo" style="font-size:12px; color:oklch(0.8 0.012 250); background:${C.bgBtn}; border:1px solid ${C.bdBtn}; padding:7px 11px; border-radius:8px; cursor:pointer;">Undo point</button>` : ""}
        <div style="margin-left:auto; display:flex; align-items:center; gap:8px;">
          <span style="font-size:12px; color:${C.fg3};">Preview ${w.maskPreview ? "Masked" : "Original"}</span>
          <div data-act="toggleMaskPreview" style="width:38px; height:21px; border-radius:12px; background:${w.maskPreview ? "oklch(0.55 0.11 225)" : C.bd}; position:relative; cursor:pointer;">
            <span style="position:absolute; top:2px; left:${w.maskPreview ? "19px" : "2px"}; width:17px; height:17px; border-radius:50%; background:oklch(0.96 0 0);"></span>
          </div>
        </div>
      </div>
      <div style="position:relative; border-radius:10px; overflow:hidden; border:1px solid ${C.bd3}; background:#000;">
        ${src ? `<img src="${w.maskPreview || src.img}" style="width:100%; display:block; user-select:none;">` : `<div style="padding:60px; text-align:center; color:${C.fg4};">Pick frames in step 1 first.</div>`}
        <svg id="maskSvg" data-act="maskSvgClick" viewBox="0 0 100 ${VBH}" preserveAspectRatio="none" style="position:absolute; inset:0; width:100%; height:100%; cursor:crosshair;">${w.maskPreview ? "" : zonesSvg}${draft}${rectMark}</svg>
        ${w.maskPreview ? `<div style="position:absolute; top:8px; left:8px; font-size:10px; font-family:${C.mono}; padding:3px 8px; border-radius:6px; background:oklch(0.14 0.008 250 / 0.85); color:oklch(0.75 0.012 250);">PREVIEW · zones blacked out (server-rendered)</div>` : ""}
      </div>
      <div style="font-size:11.5px; color:${C.fg4}; margin-top:8px;">Click to add points; the camera is fixed, so zones you draw once apply to every frame. Typically trace the windows so outside movement never triggers a detection.</div>
    </div>
    <div style="width:260px; flex:0 0 260px;">
      <div style="border:1px solid ${C.bd}; border-radius:11px; overflow:hidden; background:${C.bgCard2}; margin-bottom:14px;">
        <div style="padding:11px 14px; border-bottom:1px solid ${C.bd2}; font-size:12px; font-family:${C.mono}; text-transform:uppercase; letter-spacing:0.07em; color:${C.fg3}; display:flex; justify-content:space-between;">
          <span>Mask zones</span><span style="color:oklch(0.85 0.006 250);">${w.maskZones.length}</span>
        </div>
        ${!w.maskZones.length ? `<div style="padding:24px 16px; text-align:center; font-size:12px; color:${C.fg4}; line-height:1.5;">No zones yet. Draw on the frame, or pick a preset below. "No mask" is valid.</div>` : ""}
        ${w.maskZones.map(z => `
          <div style="display:flex; align-items:center; gap:9px; padding:10px 14px; border-top:1px solid oklch(0.22 0.01 250); font-size:12.5px;">
            <span style="width:10px; height:10px; border-radius:3px; background:oklch(0.72 0.13 225 / 0.5); border:1px solid oklch(0.8 0.13 225);"></span>
            <span style="color:oklch(0.88 0.006 250);">${esc(z.label)}</span>
            <span style="margin-left:auto; font-family:${C.mono}; font-size:10px; color:${C.fg5};">${z.type}</span>
            <span data-act="maskDelete" data-arg="${z.id}" style="cursor:pointer; color:${C.fg3}; font-size:15px; line-height:1;">×</span>
          </div>`).join("")}
      </div>
      <label style="font-size:11.5px; color:${C.fg3}; display:block; margin-bottom:6px;">Camera preset ${hint("Masks are saved per camera: the camera is fixed, so the same window zones apply to every video from it. Save your drawing once, reuse it for every future run.")}</label>
      <select data-change="maskPreset" style="width:100%; padding:9px 11px; border-radius:8px; background:${C.bgCard2}; border:1px solid ${C.bd3}; color:oklch(0.9 0.006 250); font-size:13px; margin-bottom:12px;">${presetOpts}</select>
      <div style="display:flex; gap:8px;">
        <button data-act="saveMaskPreset" style="flex:1; font-size:12px; font-weight:600; color:${C.accDark}; background:${C.acc}; border:none; padding:9px; border-radius:8px; cursor:pointer;">Save preset</button>
        <button data-act="maskClear" style="font-size:12px; color:oklch(0.8 0.012 250); background:${C.bgBtn}; border:1px solid ${C.bdBtn}; padding:9px 12px; border-radius:8px; cursor:pointer;">Clear</button>
      </div>
    </div>
  </div>`;
}
function wizStep4() {
  const w = S.wiz;
  const pipes = S.pipelines.map(p => `
    <div data-act="setPipeline" data-arg="${p.key}" style="display:flex; align-items:center; gap:12px; padding:13px 15px; border-radius:10px; cursor:pointer; background:${w.pipeline === p.key ? C.accBg : "oklch(0.18 0.01 250)"}; border:1px solid ${w.pipeline === p.key ? C.accSelBd : C.bd3};">
      <span style="width:15px; height:15px; border-radius:50%; border:2px solid ${w.pipeline === p.key ? C.acc : "oklch(0.4 0.014 250)"}; background:${w.pipeline === p.key ? C.acc : "transparent"}; flex:0 0 15px;"></span>
      <div style="flex:1;">
        <div style="font-size:13.5px; font-weight:600; color:oklch(0.92 0.006 250);">${p.name}</div>
        <div style="font-size:12px; color:oklch(0.62 0.012 250); margin-top:1px;">${p.desc}</div>
      </div>
      ${p.recommended ? `<span style="font-size:10px; font-weight:600; padding:3px 9px; border-radius:10px; background:oklch(0.26 0.07 150); color:oklch(0.86 0.1 150); border:1px solid oklch(0.42 0.09 150);">RECOMMENDED</span>` : ""}
    </div>`).join("");
  const models = S.models.map(m => {
    const sel = w.model === m.tag;
    const pulling = S.pulling === m.tag;
    return `
    <div style="padding:11px 13px; border-radius:10px; background:${sel ? C.accBg : C.bgCard}; border:1px solid ${sel ? C.accSelBd : C.bd};">
      <div style="display:flex; align-items:center; gap:10px;">
        <div ${m.installed ? `data-act="setModel" data-arg="${esc(m.tag)}"` : ""} style="flex:1; cursor:${m.installed ? "pointer" : "default"};">
          <div style="font-size:13px; font-weight:600; color:${m.installed ? "oklch(0.92 0.006 250)" : "oklch(0.62 0.012 250)"};">${esc(m.name)}</div>
          <div style="font-family:${C.mono}; font-size:10.5px; color:${C.fg4}; margin-top:1px;">${esc(m.tag)}${m.size ? " · " + m.size : ""}${m.note ? " · " + esc(m.note) : ""}</div>
        </div>
        ${m.installed && m.recommended ? `<span style="font-size:9.5px; padding:2px 7px; border-radius:8px; background:oklch(0.26 0.07 150); color:oklch(0.86 0.1 150);">rec</span>` : ""}
        ${!m.installed ? (pulling
          ? `<span style="font-family:${C.mono}; font-size:11px; color:${C.acc};">${Math.round(S.pullPct)}%</span>`
          : `<button data-act="pullModel" data-arg="${esc(m.tag)}" style="font-size:11px; font-weight:600; color:${C.accFg}; background:${C.accBg}; border:1px solid ${C.accBd2}; padding:5px 10px; border-radius:7px; cursor:pointer; white-space:nowrap;">Pull${m.size ? " (" + m.size + ")" : ""}</button>`) : ""}
      </div>
      ${pulling ? `<div style="height:4px; border-radius:3px; background:${C.bd2}; margin-top:9px; overflow:hidden;"><div style="height:100%; width:${Math.round(S.pullPct)}%; background:${C.acc}; border-radius:3px;"></div></div>` : ""}
    </div>`;
  }).join("");
  const p = S.pipelines.find(x => x.key === w.pipeline);
  const promptOpts = p ? Object.keys(p.prompts).map(n =>
    `<option value="${n}" ${w.promptPreset === n ? "selected" : ""}>${n[0].toUpperCase() + n.slice(1)}</option>`).join("") : "";
  const refOpts = S.refs.map(r =>
    `<option value="${esc(r.path)}" ${w.refPath === r.path ? "selected" : ""}>${esc(r.name)}</option>`).join("");
  return `
  <div style="max-width:940px;">
    <div style="font-size:12px; text-transform:uppercase; letter-spacing:0.08em; color:${C.fg4}; font-family:${C.mono}; margin-bottom:10px;">Pipeline</div>
    <div style="display:flex; flex-direction:column; gap:8px; margin-bottom:24px;">${pipes}</div>
    <div style="display:grid; grid-template-columns:1fr 1fr; gap:20px; margin-bottom:24px;">
      <div>
        <div style="font-size:12px; text-transform:uppercase; letter-spacing:0.08em; color:${C.fg4}; font-family:${C.mono}; margin-bottom:10px;">Model</div>
        <div style="display:flex; flex-direction:column; gap:7px;">${models}</div>
      </div>
      <div>
        <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:10px;">
          <span style="font-size:12px; text-transform:uppercase; letter-spacing:0.08em; color:${C.fg4}; font-family:${C.mono};">Prompt ${hint("The instruction sent to the model for every frame/region. Conservative = strict, fewer false alarms; Lenient = flags more, more false alarms. Edit freely — but a changed prompt means cached verdicts no longer apply, so the run recomputes every call.")}</span>
          <select data-change="promptPreset" style="padding:6px 9px; border-radius:7px; background:${C.bgCard2}; border:1px solid ${C.bd3}; color:oklch(0.9 0.006 250); font-size:12px;">
            ${promptOpts}<option value="custom" ${w.promptPreset === "custom" ? "selected" : ""}>Custom</option>
          </select>
        </div>
        <textarea data-change="promptText" style="width:100%; height:186px; padding:11px 12px; border-radius:9px; background:${C.bgInput}; border:1px solid ${C.bd3}; color:oklch(0.85 0.006 250); font-family:${C.mono}; font-size:11.5px; line-height:1.5; resize:none;">${esc(w.promptText)}</textarea>
        <div style="display:flex; align-items:center; justify-content:space-between; margin-top:8px;">
          <span style="font-size:11px; color:oklch(0.62 0.05 75);">Changing the prompt invalidates the verdict cache.</span>
          <button data-act="resetPrompt" style="font-size:11.5px; color:${C.acc}; background:none; border:none; cursor:pointer;">Reset to preset</button>
        </div>
      </div>
    </div>
    ${needRef() ? (() => {
      const refFromFrames = w.frames.some(f => f.path === w.refPath);
      const cur = S.refs.find(x => x.path === w.refPath)
        || (refFromFrames ? w.frames.find(f => f.path === w.refPath) : null);
      const frameStrip = w.frames.length ? `
        <div style="margin-top:10px;">
          <div style="font-size:11px; color:${C.fg4}; margin-bottom:6px;">…or pick a CLEAN frame from this video ${hint("The reference must show the same camera view, EMPTY and undamaged. If a mask is set, it is applied to the reference too, automatically.")}</div>
          <div data-scroll="refstrip" style="display:flex; gap:6px; overflow-x:auto; padding-bottom:4px;">
            ${w.frames.slice(0, 60).map(f => `
              <img data-act="setRefFromFrame" data-arg="${esc(f.path)}" src="${f.img}" title="${esc(f.path.split("/").pop())}"
                   style="width:86px; height:48px; object-fit:cover; border-radius:6px; cursor:pointer; flex:0 0 86px; border:2px solid ${w.refPath === f.path ? "oklch(0.8 0.13 225)" : "transparent"}; opacity:${w.refPath === f.path ? 1 : 0.75};">`).join("")}
          </div>
        </div>` : "";
      return `
    <div style="padding:14px 16px; border-radius:10px; background:oklch(0.18 0.01 250); border:1px solid ${C.bd3}; margin-bottom:16px;">
      <div style="display:flex; align-items:center; gap:14px;">
        ${cur ? `<img src="${cur.img}" style="width:96px; height:54px; object-fit:cover; border-radius:6px; border:1px solid ${C.bd3};">` : ""}
        <div style="flex:1;">
          <div style="font-size:13px; font-weight:600; margin-bottom:3px;">Reference frame <span style="font-size:11px; color:${C.redFg};">required for this pipeline</span> ${hint("The clean 'before' image the pipeline compares against: same fixed camera, empty tram, no anomalies, ideally same lighting session as the inspected frames.")}</div>
          <div style="display:flex; gap:8px;">
            <select data-change="refPath" style="flex:1; padding:8px 10px; border-radius:8px; background:${C.bg}; border:1px solid ${C.bd3}; color:oklch(0.9 0.006 250); font-family:${C.mono}; font-size:12px;">
              ${refOpts}
              ${refFromFrames ? `<option value="${esc(w.refPath)}" selected>[video frame] ${esc(w.refPath.split("/").pop())}</option>` : ""}
            </select>
            <button data-act="pickRefFile" style="font-size:12px; color:${C.accFg}; background:${C.accBg}; border:1px solid ${C.accBd2}; padding:8px 13px; border-radius:8px; cursor:pointer; white-space:nowrap;">${w.refUploading ? "Uploading…" : "Upload"}</button>
          </div>
        </div>
      </div>
      ${frameStrip}
    </div>`; })() : ""}
    <div style="border:1px solid ${C.bd}; border-radius:10px; overflow:hidden; background:${C.bgCard2};">
      <div data-act="toggleAdv" style="padding:12px 15px; font-size:13px; cursor:pointer; display:flex; align-items:center; gap:8px; color:oklch(0.8 0.012 250);"><span style="font-family:${C.mono}; color:${C.fg3};">${w.advOpen ? "▾" : "▸"}</span> Advanced parameters</div>
      ${w.advOpen ? `
      <div style="padding:4px 15px 16px; display:grid; grid-template-columns:repeat(4,1fr); gap:14px;">
        ${[["Diff threshold", "diff", w.diff,
            "vlm_05 only. Minimum pixel brightness difference (0-255) between reference and frame for a pixel to count as changed. Lower = more sensitive but more noise regions sent to the VLM. Benchmark-tuned default: 40."],
           ["Min region area", "minArea", w.minArea,
            "vlm_05 only. Changed regions smaller than this (in pixels) are dropped as specks/noise. Default 500 ≈ a phone seen from the ceiling camera."],
           ["Max regions", "maxRegions", w.maxRegions,
            "vlm_05 only. Budget cap: at most this many regions per frame are sent to the VLM (most salient first). Protects against rush-hour frames exploding the cost. Default 25."],
           ["Retries on fail", "retries", w.retries,
            "All pipelines. How many times a frame is retried when the model's reply is unusable (bad format) or the call times out, before the frame is marked failed and the batch continues. Default 2."]]
          .map(([lb, k, v, tip]) => `<label style="font-size:11.5px; color:${C.fg3}; display:flex; flex-direction:column; gap:5px;"><span>${lb} ${hint(tip)}</span><input type="number" value="${v}" data-change="${k}" style="padding:7px 9px; border-radius:7px; background:${C.bgInput}; border:1px solid ${C.bd3}; color:oklch(0.9 0.006 250); font-family:${C.mono}; font-size:12px;"></label>`).join("")}
      </div>` : ""}
    </div>
  </div>`;
}
function wizStep5() {
  const w = S.wiz;
  const p = S.pipelines.find(x => x.key === w.pipeline) || {};
  const m = S.models.find(x => x.tag === w.model) || { name: w.model };
  const est = estimate();
  const rows = [
    ["Source", w.source === "video" ? "uploaded video" : (w.source === "reuse" ? "reused extraction" : "demo frames")],
    ["Frames", estFrames() + " frames"],
    ["Mask", w.maskPreset === "none" || !w.maskZones.length ? "no mask" : `${w.maskZones.length} zones (${w.maskPreset})`],
    ["Pipeline", p.name || w.pipeline],
    ["Model", m.name || w.model],
    ["Prompt", w.promptPreset],
    ["Reference", needRef() ? (w.refPath || "—").split("/").pop() : "n/a"],
  ].map(([k, v]) => `
    <div style="display:grid; grid-template-columns:180px 1fr; padding:13px 18px; border-top:1px solid oklch(0.22 0.01 250); font-size:13px;">
      <span style="color:oklch(0.58 0.012 250);">${k}</span>
      <span style="color:oklch(0.92 0.006 250); font-family:${C.mono};">${esc(v)}</span>
    </div>`).join("");
  return `
  <div style="max-width:720px;">
    <div style="border:1px solid ${C.bd}; border-radius:12px; overflow:hidden; background:${C.bgCard2}; margin-bottom:20px;">${rows}</div>
    <div style="display:flex; align-items:center; gap:14px; padding:16px 20px; border-radius:12px; background:oklch(0.2 0.03 225); border:1px solid ${C.accBd}; margin-bottom:22px;">
      <svg width="18" height="18" viewBox="0 0 18 18" fill="none" style="flex:0 0 18px;"><circle cx="9" cy="9" r="7.3" stroke="oklch(0.75 0.1 225)" stroke-width="1.4"/><path d="M9 5 V9 L12 11" stroke="oklch(0.75 0.1 225)" stroke-width="1.4" stroke-linecap="round"/></svg>
      <div style="font-size:13.5px; color:oklch(0.85 0.05 225); line-height:1.5;">
        <span style="font-family:${C.mono}; font-weight:700;">${est.frames} frames</span> · estimated <span style="font-weight:700;">${fmtDur(est.total)}</span>
        <div style="font-size:11.5px; color:${est.rough ? C.amber : "oklch(0.68 0.04 225)"}; margin-top:2px;">${est.basis}</div>
      </div>
    </div>
    <button data-act="launchRun" style="width:100%; font-size:16px; font-weight:700; color:${C.accDark}; background:${C.acc}; border:none; padding:16px; border-radius:12px; cursor:pointer; letter-spacing:0.01em;">Launch analysis</button>
  </div>`;
}

/* --------------- run --------------- */
/* Keeps the stage from crowding out the frame strip and the log below it; the
   detection list scrolls inside the panel instead of stretching the card. */
const STAGE_MAX_H = 500;
/* Boxes drawn straight on the staged frame. Detections without a bbox
   (vlm_01/02 answer on the whole frame) are listed in the side panel. */
function stageBoxes(st) {
  const cs = _sizeCache[st.img];
  if (!cs || !st.detections) return "";
  return st.detections.map(d => {
    if (!d.bbox) return "";
    const [x0, y0, x1, y1] = d.bbox;
    return `<div style="position:absolute; left:${x0 / cs.w * 100}%; top:${y0 / cs.h * 100}%; width:${(x1 - x0) / cs.w * 100}%; height:${(y1 - y0) / cs.h * 100}%; border:2px solid oklch(0.8 0.18 60); border-radius:2px; box-shadow:0 0 0 1px oklch(0.15 0.008 250 / 0.7); pointer-events:none; animation:arsislide .3s ease;">
      <span style="position:absolute; top:-19px; left:-2px; font-size:10.5px; font-family:${C.mono}; background:oklch(0.14 0.008 250 / 0.9); color:oklch(0.88 0.13 60); padding:2px 6px; border-radius:4px; white-space:nowrap;">${esc(d.label)}</span></div>`;
  }).join("");
}
function stageCard(run) {
  const st = run.stage;
  const refBlock = !run.refImg ? "" : `
    <div style="margin-top:12px;">
      <div style="font-size:10px; font-family:${C.mono}; text-transform:uppercase; letter-spacing:0.08em; color:${C.fg5}; margin-bottom:6px;">Reference${run.masked ? " · masked" : ""}</div>
      <img src="${run.refImg}" style="width:100%; border-radius:8px; border:1px solid ${C.bd2}; display:block;">
    </div>`;
  if (!st) return `
    <div style="border:1px solid ${C.bd3}; border-radius:12px; background:${C.bgCard2}; padding:18px 20px; margin-bottom:18px; display:flex; align-items:center; justify-content:center; min-height:120px; color:${C.fg3}; font-size:13px;">
      ${run.done ? "Run finished." : "Waiting for the first frame…"}
    </div>`;
  const done = st.phase === "result";
  const dets = st.detections || [];
  const outcome = !done ? ["ANALYSING", C.accFg, C.accBg, C.accBd]
    : st.status === "failed" ? ["FAILED", C.fg2, "oklch(0.22 0.012 250)", C.bdBtn]
    : st.anomaly ? ["ANOMALY", "oklch(0.85 0.06 22)", C.redBg, C.redBd]
    : ["CLEAN", "oklch(0.82 0.06 150)", C.greenBg, C.greenBd];
  const held = Math.min(stageHold(run), Date.now() - (st.at || Date.now()));
  return `
  <div style="border:1px solid ${C.bd3}; border-radius:12px; background:${C.bgCard2}; margin-bottom:18px; overflow:hidden;">
    <div style="display:flex; align-items:center; gap:10px; padding:11px 16px; border-bottom:1px solid ${C.bd2};">
      <span style="font-size:11px; font-family:${C.mono}; text-transform:uppercase; letter-spacing:0.08em; color:${C.fg4};">${done ? "Result" : "Now analysing"}</span>
      <span style="font-family:${C.mono}; font-size:12.5px; color:oklch(0.9 0.006 250);">${esc(st.frame_id || "")}</span>
      <span style="font-size:11px; color:${C.fg4};">frame ${st.index + 1}/${run.total}</span>
      ${run.masked ? `<span style="font-size:10px; font-family:${C.mono}; padding:2px 7px; border-radius:10px; color:${C.accFg}; background:${C.accBg}; border:1px solid ${C.accBd};">masked</span>` : ""}
      ${run.reel.length ? `<span style="margin-left:auto; font-size:10.5px; font-family:${C.mono}; color:${C.fg5};">+${run.reel.length} ahead</span>` : ""}
    </div>
    <div style="display:grid; grid-template-columns:minmax(0,1fr) 240px; gap:16px; padding:16px; align-items:start;">
      <div style="min-width:0; text-align:center;">
        <div style="position:relative; display:inline-block; max-width:100%; border-radius:9px; overflow:hidden; background:oklch(0.1 0.008 250); vertical-align:top;">
          ${st.img ? `<img src="${st.img}" style="display:block; max-width:100%; max-height:${STAGE_MAX_H}px; ${done ? "" : "opacity:0.82;"}">` : ""}
          ${done ? stageBoxes(st) : `
            <div style="position:absolute; inset:0; background:linear-gradient(180deg, transparent 0%, oklch(0.72 0.13 225 / 0.16) 50%, transparent 100%); animation:arsiscan 1.8s ease-in-out infinite;"></div>`}
        </div>
      </div>
      <div style="min-width:0; display:flex; flex-direction:column; gap:10px; max-height:${STAGE_MAX_H}px;">
        <div>
          <div style="display:inline-flex; align-items:center; gap:7px; font-size:12px; font-family:${C.mono}; font-weight:700; letter-spacing:0.05em; padding:6px 11px; border-radius:8px; color:${outcome[1]}; background:${outcome[2]}; border:1px solid ${outcome[3]};">
            ${!done ? `<span style="width:6px; height:6px; border-radius:50%; background:${C.acc}; animation:arsipulse 1.3s infinite;"></span>` : ""}${outcome[0]}
          </div>
          ${done ? `<div style="font-family:${C.mono}; font-size:11px; color:${C.fg4}; margin-top:8px;">${st.seconds != null ? st.seconds + "s" : ""}${st.attempts > 1 ? ` · ${st.attempts} attempts` : ""}${dets.length ? ` · ${dets.length} detection${dets.length > 1 ? "s" : ""}` : ""}</div>` : ""}
          ${done && st.error ? `<div style="font-size:11.5px; color:${C.redFg}; margin-top:8px; line-height:1.45; word-break:break-word;">${esc(st.error)}</div>` : ""}
        </div>
        ${refBlock}
        ${done && dets.length ? `<div data-scroll="stagedets" style="overflow:auto; min-height:0; padding-right:2px;">${dets.map(d => {
          const tm = TYPE_META[d.type] || TYPE_META.unknown;
          return `<div style="padding:8px 9px; border-radius:8px; margin-bottom:5px; background:${C.bgCard}; border:1px solid ${C.bd2};">
            <span style="font-size:9.5px; padding:2px 6px; border-radius:9px; color:${tm.fg}; background:${tm.bg}; border:1px solid ${tm.bd};">${esc(d.type)}</span>
            <div style="font-size:12px; color:oklch(0.9 0.006 250); margin-top:5px; line-height:1.35;">${esc(d.label)}</div>
            ${!d.bbox ? `<div style="font-size:10px; color:${C.fg5}; margin-top:3px;">${d.zone ? esc(d.zone) : "whole frame"}</div>` : ""}
          </div>`; }).join("")}</div>`
        : done && st.status === "ok" && !st.anomaly
          ? `<div style="font-size:12px; color:${C.fg3}; line-height:1.45;">Nothing reported on this frame.</div>` : ""}
      </div>
    </div>
    ${done ? `<div style="height:2px; background:${C.bd2};"><div style="height:100%; background:${C.acc}; width:100%; transform-origin:left; animation:arsihold ${stageHold(run)}ms linear forwards; animation-delay:-${held}ms;"></div></div>` : ""}
  </div>`;
}
function runView() {
  const run = S.run;
  if (!run) return `
    <div data-scroll="page" style="height:100%; overflow:auto; padding:24px 28px;">
      <div style="max-width:520px; margin:80px auto 0; text-align:center;">
        <div style="width:54px; height:54px; margin:0 auto 18px; border-radius:14px; background:${C.bgBtn}; border:1px solid ${C.bd3}; display:flex; align-items:center; justify-content:center; color:${C.fg3};">${I.play}</div>
        <div style="font-size:15px; font-weight:600; margin-bottom:6px;">No active run</div>
        <div style="font-size:13px; color:${C.fg3}; margin-bottom:20px; line-height:1.5;">Configure an analysis and launch it to watch frames stream in here live.</div>
        <button data-act="goWizard" style="font-size:13px; font-weight:600; color:${C.accDark}; background:${C.acc}; border:none; padding:11px 20px; border-radius:9px; cursor:pointer;">Start a new analysis</button>
      </div>
    </div>`;
  const pct = run.total ? Math.round(run.processed / run.total * 100) : 0;
  const eta = runEta(run);
  const per = runPerFrame(run);
  const measured = run.times && run.times.length
    ? `${per < 10 ? per.toFixed(1) : Math.round(per)} s/frame`
    : "";
  const elapsed = run.t0 ? fmtEta((Date.now() - run.t0) / 1000) : "";
  const ringColor = { red: "oklch(0.62 0.17 22)", grey: "oklch(0.4 0.012 250)", green: "oklch(0.6 0.15 150)" };
  return `
  <div data-scroll="page" style="height:100%; overflow:auto; padding:24px 28px;">
    <div style="border:1px solid ${C.bd3}; border-radius:12px; padding:18px 20px; background:${C.bgCard2}; margin-bottom:18px;">
      <div style="display:flex; align-items:baseline; gap:12px; margin-bottom:12px;">
        <span style="font-family:${C.mono}; font-size:15px; color:oklch(0.92 0.006 250); font-weight:600;">${esc(run.jobId)}</span>
        ${!run.done ? `<span style="display:inline-flex; align-items:center; gap:6px; font-size:11.5px; color:${C.acc};"><span style="width:6px; height:6px; border-radius:50%; background:${C.acc}; animation:arsipulse 1.3s infinite;"></span>running</span>`
          : `<span style="font-size:11.5px; color:${run.cancelled ? C.fg2 : "oklch(0.82 0.1 150)"};">${run.cancelled ? "cancelled" : "complete"}</span>`}
        <span style="margin-left:auto; font-family:${C.mono}; font-size:13px; color:${C.fg2};">${run.processed}/${run.total}${measured ? ` · ${measured}` : ""} · ${run.done ? (run.cancelled ? "stopped" : "done") + (elapsed ? " in " + elapsed : "") : "ETA " + eta}</span>
      </div>
      <div style="height:9px; border-radius:6px; background:${C.bg}; overflow:hidden; margin-bottom:16px;">
        <div style="height:100%; width:${pct}%; background:linear-gradient(90deg, oklch(0.6 0.13 225), oklch(0.75 0.13 225)); border-radius:6px; transition:width .2s;"></div>
      </div>
      <div style="display:grid; grid-template-columns:repeat(4,1fr); gap:12px;">
        ${[[run.processed, "processed", ""], [run.anomalous, "anomalous", C.red],
           [run.failed, "failed", C.fg2], [run.retried, "retried", C.fg2]]
          .map(([v, lb, col]) => `<div><div style="font-family:${C.mono}; font-size:22px; font-weight:700; ${col ? "color:" + col : ""}">${v}</div><div style="font-size:11px; color:${C.fg3};">${lb}</div></div>`).join("")}
      </div>
    </div>
    ${stageCard(run)}
    <div style="font-size:12px; text-transform:uppercase; letter-spacing:0.08em; color:${C.fg4}; font-family:${C.mono}; margin-bottom:10px;">Frames${run.masked ? " · masked" : ""}</div>
    <div style="display:flex; gap:8px; flex-wrap:wrap; margin-bottom:20px; min-height:56px;">
      ${run.thumbs.map(t => `
        <div style="width:74px; height:50px; border-radius:7px; overflow:hidden; border:2px solid ${ringColor[t.ring]}; position:relative; animation:arsislide .25s ease;">
          ${t.img ? `<img src="${t.img}" style="width:100%; height:100%; object-fit:cover; display:block;">` : ""}
          ${t.ring === "grey" ? `<span style="position:absolute; inset:0; background:oklch(0.1 0.008 250 / 0.6); display:flex; align-items:center; justify-content:center; font-size:14px;">↻</span>` : ""}
        </div>`).join("")}
    </div>
    <div style="border:1px solid ${C.bd}; border-radius:10px; overflow:hidden; background:oklch(0.1 0.008 250); margin-bottom:20px;">
      <div style="padding:9px 14px; border-bottom:1px solid ${C.bd2}; font-size:11px; font-family:${C.mono}; text-transform:uppercase; letter-spacing:0.07em; color:${C.fg4};">Live log</div>
      <div data-scroll="runlog" style="max-height:180px; overflow:auto; padding:10px 14px; font-family:${C.mono}; font-size:11.5px; line-height:1.7; color:oklch(0.68 0.012 250);">
        ${run.log.map(l => `<div>${esc(l)}</div>`).join("")}
      </div>
    </div>
    <div style="display:flex; gap:12px;">
      ${!run.done ? `<button data-act="cancelRun" style="font-size:13px; color:oklch(0.85 0.1 22); background:oklch(0.2 0.03 22); border:1px solid ${C.redBd}; padding:11px 20px; border-radius:9px; cursor:pointer;">Cancel (keep partial results)</button>` : ""}
      ${run.done ? `<button data-act="viewRunResults" style="font-size:13px; font-weight:600; color:${C.accDark}; background:${C.acc}; border:none; padding:11px 22px; border-radius:9px; cursor:pointer;">Open results</button>
      <button data-act="goWizard" style="font-size:13px; color:oklch(0.82 0.012 250); background:${C.bgBtn}; border:1px solid ${C.bdBtn}; padding:11px 20px; border-radius:9px; cursor:pointer;">New analysis</button>` : ""}
    </div>
  </div>`;
}

/* --------------- results --------------- */
function frameTypes(f) { return Array.from(new Set(f.detections.map(d => d.type))); }
function bboxOverlay(frame, cs, hoverIdx = -1) {
  if (!cs || !frame || !frame.detections.length) return "";
  return frame.detections.map((d, i) => {
    if (!d.bbox) return "";
    const [x0, y0, x1, y1] = d.bbox;
    const hl = hoverIdx === i;
    return `<div style="position:absolute; left:${x0 / cs.w * 100}%; top:${y0 / cs.h * 100}%; width:${(x1 - x0) / cs.w * 100}%; height:${(y1 - y0) / cs.h * 100}%; border:2px solid ${hl ? "oklch(0.95 0.12 90)" : "oklch(0.7 0.18 150)"}; ${hl ? "box-shadow:0 0 0 9999px rgba(8,10,14,0.45);" : ""} border-radius:2px; pointer-events:none;">
      <span style="position:absolute; top:-18px; left:0; font-size:10px; font-family:${C.mono}; background:oklch(0.14 0.008 250 / 0.85); color:oklch(0.85 0.1 150); padding:1px 5px; border-radius:4px; white-space:nowrap;">${esc(d.label)}</span></div>`;
  }).join("");
}
const REV_COL = { tp: "oklch(0.75 0.16 150)", fp: "oklch(0.72 0.17 22)",
                  unset: "oklch(0.85 0.12 90)", fn: "oklch(0.75 0.15 300)" };
function reviewOverlay(frame, cs) {
  if (!cs) return "";
  const e = S.rev.doc.frames[frame.frame_id] || { verdicts: {}, missed: [] };
  const pct = (b) => `left:${b[0] / cs.w * 100}%; top:${b[1] / cs.h * 100}%; width:${(b[2] - b[0]) / cs.w * 100}%; height:${(b[3] - b[1]) / cs.h * 100}%;`;
  const tag = (txt, col) => `<span style="position:absolute; top:-18px; left:0; font-size:10px; font-family:${C.mono}; background:oklch(0.14 0.008 250 / 0.88); color:${col}; padding:1px 5px; border-radius:4px; white-space:nowrap;">${txt}</span>`;
  const boxes = frame.detections.map((d, i) => {
    if (!d.bbox) return "";
    const v = e.verdicts[i];
    const col = v ? REV_COL[v] : REV_COL.unset;
    return `<div data-act="revCycle" data-arg="${i}" data-hover="${i}" title="click: TP → FP → unset" style="position:absolute; ${pct(d.bbox)} border:2.5px ${v ? "solid" : "dashed"} ${col}; border-radius:2px; cursor:pointer; pointer-events:${S.rev.draw ? "none" : "auto"};">
      ${tag((v ? v.toUpperCase() + " · " : "") + esc(d.label), col)}</div>`;
  }).join("");
  const missed = e.missed.map((m) => `
    <div style="position:absolute; ${pct(m.bbox)} border:2.5px solid ${REV_COL.fn}; border-radius:2px; pointer-events:none;">${tag("FN · " + esc(m.label), REV_COL.fn)}</div>`).join("");
  const corner = S.rev.corner ? `<div style="position:absolute; left:calc(${S.rev.corner.x / cs.w * 100}% - 4px); top:calc(${S.rev.corner.y / cs.h * 100}% - 4px); width:9px; height:9px; border-radius:50%; background:${REV_COL.fn}; pointer-events:none;"></div>` : "";
  const pending = S.rev.pending ? `<div style="position:absolute; ${pct(S.rev.pending)} border:2.5px dashed ${REV_COL.fn}; border-radius:2px; pointer-events:none;"></div>` : "";
  return boxes + missed + corner + pending;
}
function reviewSidebar(sel) {
  const e = S.rev.doc.frames[sel.frame_id] || { verdicts: {}, missed: [], done: false };
  const m = S.rev.metrics;
  const nDets = sel.detections.length;
  const nJudged = Object.keys(e.verdicts).length;
  const vbtn = (i, v, cur) => {
    const on = cur === v;
    const col = REV_COL[v];
    return `<button data-act="revSet" data-arg="${i}:${v}" style="font-size:10.5px; font-weight:700; font-family:${C.mono}; padding:4px 9px; border-radius:7px; cursor:pointer; color:${on ? C.accDark : col}; background:${on ? col : "transparent"}; border:1.5px solid ${col};">${v.toUpperCase()}</button>`;
  };
  const detRows = sel.detections.map((d, i) => {
    const tm = TYPE_META[d.type] || TYPE_META.unknown;
    const hl = S.res.hoverV === i;
    return `
    <div data-hover="${i}" style="display:flex; align-items:center; gap:8px; padding:9px 10px; border-radius:9px; margin-bottom:5px; background:${hl ? "oklch(0.2 0.02 225)" : C.bgCard2}; border:1px solid ${hl ? C.accBd : C.bd2};">
      <span style="font-size:10px; padding:2px 7px; border-radius:10px; color:${tm.fg}; background:${tm.bg}; border:1px solid ${tm.bd}; white-space:nowrap;">${d.type}</span>
      <span style="flex:1; font-size:12px; color:oklch(0.9 0.006 250); overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">${esc(d.label)}${d.bbox ? "" : " <i style='color:" + C.fg4 + "'>(no box)</i>"}</span>
      ${vbtn(i, "tp", e.verdicts[i])}${vbtn(i, "fp", e.verdicts[i])}
    </div>`;
  }).join("");
  const missedRows = e.missed.map((mi, i) => `
    <div style="display:flex; align-items:center; gap:8px; padding:8px 10px; border-radius:9px; margin-bottom:5px; background:${C.bgCard2}; border:1px solid oklch(0.4 0.1 300);">
      <span style="font-size:10px; padding:2px 7px; border-radius:10px; color:oklch(0.85 0.12 300); background:oklch(0.24 0.06 300); border:1px solid oklch(0.42 0.1 300);">FN</span>
      <span style="flex:1; font-size:12px; color:oklch(0.9 0.006 250);">${esc(mi.label)} <span style="color:${C.fg4}; font-size:10.5px;">${mi.type}</span></span>
      <button data-act="revDelMissed" data-arg="${i}" title="Remove" style="font-size:12px; color:${C.fg3}; background:transparent; border:none; cursor:pointer;">✕</button>
    </div>`).join("");
  const pendingForm = S.rev.pending ? `
    <div style="padding:10px; border-radius:9px; margin-bottom:8px; background:oklch(0.18 0.03 300); border:1px solid oklch(0.42 0.1 300);">
      <div style="font-size:11px; color:oklch(0.85 0.1 300); margin-bottom:7px; font-family:${C.mono};">missed box [${S.rev.pending.join(", ")}]</div>
      <input id="revLabel" data-change="revLabel" data-live placeholder="label — e.g. phone on seat" value="${esc(S.rev.pendingLabel)}" style="width:100%; box-sizing:border-box; padding:7px 9px; border-radius:7px; background:${C.bgInput}; border:1px solid ${C.bd3}; color:oklch(0.92 0.006 250); font-size:12px; margin-bottom:7px;">
      <div style="display:flex; gap:6px;">
        <select data-change="revType" style="flex:1; padding:6px 8px; border-radius:7px; background:${C.bgInput}; border:1px solid ${C.bd3}; color:oklch(0.9 0.006 250); font-size:11.5px;">
          ${["object", "graffiti", "damage", "litter", "unknown"].map(t => `<option ${t === S.rev.pendingType ? "selected" : ""}>${t}</option>`).join("")}
        </select>
        <button data-act="revAddMissed" style="font-size:11.5px; font-weight:600; color:${C.accDark}; background:oklch(0.75 0.15 300); border:none; padding:6px 13px; border-radius:7px; cursor:pointer;">Add ↵</button>
        <button data-act="revCancelMissed" style="font-size:11.5px; color:${C.fg2}; background:${C.bgBtn}; border:1px solid ${C.bdBtn}; padding:6px 10px; border-radius:7px; cursor:pointer;">Esc</button>
      </div>
    </div>` : "";
  const canConfirm = nJudged >= nDets;
  const stat = (label, v) => `<div style="display:flex; justify-content:space-between; font-size:11.5px; padding:2px 0;"><span style="color:${C.fg3};">${label}</span><span style="font-family:${C.mono}; color:oklch(0.9 0.006 250);">${v ?? "—"}</span></div>`;
  return `
    <div style="padding:14px 16px 10px; border-bottom:1px solid oklch(0.22 0.01 250);">
      <div style="display:flex; align-items:center; gap:8px;">
        <div style="font-size:12px; text-transform:uppercase; letter-spacing:0.08em; color:oklch(0.85 0.12 90); font-family:${C.mono};">Review</div>
        ${e.done ? `<span style="font-size:10px; padding:2px 8px; border-radius:10px; background:${C.greenBg}; border:1px solid ${C.greenBd}; color:${C.greenFg}; font-weight:600;">✓ reviewed</span>` : ""}
        <span style="margin-left:auto; font-size:10.5px; color:${C.fg4};">${S.rev.saving ? "Saving…" : S.rev.saved ? "Saved ✓" : "…"}</span>
      </div>
      <div style="font-size:11px; color:${C.fg4}; margin-top:4px;">Click a box (or TP/FP) to judge · <b>T</b>/<b>F</b> hovered · <b>A</b> all-TP+confirm · <b>M</b> missed box · <b>C</b> confirm</div>
    </div>
    <div style="padding:10px 12px;">
      ${detRows || `<div style="font-size:12px; color:${C.fg4}; padding:6px 4px 10px;">No detections on this frame — confirm it as clean, or add the boxes the model missed.</div>`}
      ${missedRows}${pendingForm}
      <button data-act="revToggleDraw" style="width:100%; margin:4px 0 8px; font-size:12px; padding:8px 0; border-radius:8px; cursor:pointer; color:${S.rev.draw ? C.accDark : "oklch(0.85 0.12 300)"}; background:${S.rev.draw ? "oklch(0.75 0.15 300)" : "oklch(0.2 0.03 300)"}; border:1.5px solid oklch(0.45 0.11 300);">
        ${S.rev.draw ? "Click two corners on the image… (Esc cancels)" : "+ Missed object (M)"}</button>
      ${nDets && nJudged < nDets ? `
      <button data-act="revAllTpConfirm" style="width:100%; margin-bottom:6px; font-size:12.5px; font-weight:600; padding:9px 0; border-radius:8px; cursor:pointer; color:${REV_COL.tp}; background:oklch(0.2 0.04 150); border:1.5px solid ${C.greenBd};">
        ✓✓ All TP + confirm (A)</button>` : ""}
      <button data-act="revConfirm" ${canConfirm ? "" : "disabled"} style="width:100%; font-size:12.5px; font-weight:600; padding:9px 0; border-radius:8px; cursor:${canConfirm ? "pointer" : "not-allowed"}; opacity:${canConfirm ? 1 : 0.45}; color:${C.accDark}; background:${C.green}; border:none;">
        ✓ Confirm frame & next (C)</button>
      ${nDets ? `<div style="font-size:10.5px; color:${C.fg4}; margin-top:5px; text-align:center;">${nJudged}/${nDets} detection(s) judged</div>` : ""}
      <label style="display:flex; align-items:center; gap:7px; margin-top:8px; font-size:11px; color:${C.fg3}; cursor:pointer;">
        <input type="checkbox" data-act="revTogglePropagate" ${S.rev.propagate ? "checked" : ""} style="accent-color:${C.acc};">
        Propagate verdicts to similar boxes ${hint("Fixed camera + static objects: when you judge a box, the same verdict is copied to overlapping boxes (IoU ≥ 0.55, same type) whose verdict is still unset, on frames not yet confirmed. Never overwrites a judgement.")}
      </label>
    </div>
    ${m ? `
    <div style="padding:12px 16px; border-top:1px solid oklch(0.22 0.01 250);">
      <div style="font-size:10.5px; text-transform:uppercase; letter-spacing:0.08em; color:${C.fg5}; margin-bottom:6px; font-family:${C.mono};">Metrics · ${m.progress.n_done}/${m.progress.n_frames} reviewed</div>
      ${stat("Objects — precision", m.objects.precision)}
      ${stat("Objects — recall", m.objects.recall)}
      ${stat("Objects — TP / FP / FN", `${m.objects.tp} / ${m.objects.fp} / ${m.objects.fn}`)}
      ${stat("Frames — accuracy", m.frames.accuracy)}
      ${stat("Frames — F1", m.frames.f1)}
      ${stat("Frames — TP/FP/TN/FN", `${m.frames.TP}/${m.frames.FP}/${m.frames.TN}/${m.frames.FN}`)}
      <div style="font-size:10px; color:${C.fg5}; margin-top:6px;">Any missed object ⇒ frame scores FN (supervisor rule). Export xlsx to get the review sheet.</div>
    </div>` : ""}`;
}
function resultsView() {
  const R = S.res;
  if (!R.data || !R.data.frames || !R.data.frames.length) return `
    <div style="height:100%; display:flex; align-items:center; justify-content:center;">
      <div style="text-align:center; color:${C.fg4}; font-size:13px;">
        <div style="width:40px; height:40px; margin:0 auto 12px; border-radius:10px; border:1.5px dashed oklch(0.34 0.014 250);"></div>
        No completed job yet.${S.jobs.length ? "" : " Run an analysis first."}</div>
    </div>`;
  const data = R.data, frames = data.frames;
  const counts = { all: frames.length, anomalous: frames.filter(f => f.anomaly).length,
                   failed: frames.filter(f => f.status === "failed").length,
                   object: 0, graffiti: 0, damage: 0, litter: 0 };
  frames.forEach(f => frameTypes(f).forEach(t => { if (counts[t] !== undefined) counts[t]++; }));
  const tabs = [["all", "All"], ["anomalous", "Anomalous"], ["failed", "Failed"],
                ["object", "Object"], ["graffiti", "Graffiti"], ["damage", "Damage"], ["litter", "Litter"]]
    .map(([k, lb]) => {
      const on = R.filter === k;
      return `<div data-act="setFilter" data-arg="${k}" style="display:flex; align-items:center; gap:6px; padding:5px 11px; border-radius:8px; font-size:12.5px; cursor:pointer; white-space:nowrap; background:${on ? C.accSel : C.bgCard}; color:${on ? C.accFg : "oklch(0.68 0.012 250)"}; border:1px solid ${on ? C.accBd2 : C.bd};">${lb} <span style="font-family:${C.mono}; opacity:0.8;">${counts[k]}</span></div>`;
    }).join("");
  const match = (f) => R.filter === "all" ? true
    : R.filter === "anomalous" ? !!f.anomaly
    : R.filter === "failed" ? f.status === "failed"
    : frameTypes(f).includes(R.filter);
  const visible = frames.map((f, i) => ({ f, i })).filter(o => match(o.f));
  const selIdx = Math.min(R.sel, frames.length - 1);
  const sel = frames[selIdx];
  const revGal = revActive() ? S.rev.doc.frames : null;
  const gallery = visible.map(({ f, i }) => {
    const ring = f.status === "failed" ? "oklch(0.45 0.012 250)" : f.anomaly ? "oklch(0.6 0.16 22)" : "oklch(0.5 0.1 150)";
    const badge = f.status === "failed" ? ["FAIL", "oklch(0.3 0.012 250)", C.fg2]
      : f.anomaly ? ["ANOM", "oklch(0.3 0.1 22)", "oklch(0.9 0.12 22)"] : ["clean", "oklch(0.26 0.06 150)", "oklch(0.88 0.09 150)"];
    const done = revGal && revGal[f.frame_id] && revGal[f.frame_id].done;
    return `
    <div data-act="selFrame" data-arg="${i}" ${i === selIdx ? 'id="gsel"' : ""} style="margin-bottom:9px; border-radius:9px; overflow:hidden; cursor:pointer; border:2px solid ${ring}; outline:${i === selIdx ? "2px solid oklch(0.8 0.13 225)" : "none"}; outline-offset:1px;">
      <div style="position:relative;">
        <img src="${f.img}" loading="lazy" style="width:100%; height:76px; object-fit:cover; display:block;">
        <span style="position:absolute; top:5px; right:5px; font-size:9px; font-family:${C.mono}; padding:1px 5px; border-radius:6px; background:${badge[1]}; color:${badge[2]};">${badge[0]}</span>
        ${revGal ? `<span style="position:absolute; bottom:4px; right:5px; width:14px; height:14px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:9px; background:${done ? "oklch(0.65 0.13 150)" : "oklch(0.3 0.014 250 / 0.85)"}; color:${done ? "oklch(0.13 0.008 250)" : C.fg3};">${done ? "✓" : "·"}</span>` : ""}
      </div>
      <div style="font-family:${C.mono}; font-size:9.5px; color:${C.fg3}; padding:4px 6px; background:${C.bg}; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">${esc(f.frame_id)}</div>
    </div>`;
  }).join("");
  if (R.compare) return compareView(tabs, gallery, sel, selIdx);
  const outcome = sel.status === "failed" ? ["FAILED", C.fg2, "oklch(0.22 0.012 250)", C.bdBtn]
    : sel.anomaly ? ["ANOMALY", "oklch(0.85 0.05 22)", C.redBg, C.redBd] : ["CLEAN", "oklch(0.82 0.05 150)", C.greenBg, C.greenBd];
  const revOn = revActive();
  const boxes = revOn ? reviewOverlay(sel, R.coordSize) : bboxOverlay(sel, R.coordSize, R.hoverV);
  const imgWrapAttrs = revOn ? `data-act="revImgClick" style="position:relative; ${S.rev.draw ? "cursor:crosshair;" : ""}"` : `style="position:relative;"`;
  const refImg = jobRefImg(data);
  const masked = !!(data.config || {}).mask;
  const timeline = frames.map((f, i) => `
    <div data-act="selFrame" data-arg="${i}" style="flex:1; height:${f.anomaly ? "100%" : "46%"}; background:${f.status === "failed" ? "oklch(0.45 0.03 250)" : f.anomaly ? "oklch(0.62 0.17 22)" : "oklch(0.32 0.02 250)"}; border-radius:1px; cursor:pointer; outline:${i === selIdx ? "1.5px solid oklch(0.85 0.13 225)" : "none"};"></div>`).join("");
  const verdicts = sel.detections.map((d, i) => {
    const tm = TYPE_META[d.type] || TYPE_META.unknown;
    const hl = R.hoverV === i;
    return `
    <div data-hover="${i}" style="display:flex; align-items:center; gap:9px; padding:10px 10px; border-radius:9px; margin-bottom:5px; cursor:default; background:${hl ? "oklch(0.2 0.02 225)" : C.bgCard2}; border:1px solid ${hl ? C.accBd : C.bd2};">
      <span style="font-size:10px; padding:2px 7px; border-radius:10px; color:${tm.fg}; background:${tm.bg}; border:1px solid ${tm.bd}; white-space:nowrap;">${d.type}</span>
      <span style="font-size:12.5px; color:oklch(0.9 0.006 250); overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">${esc(d.label)}</span>
      <span style="margin-left:auto; font-family:${C.mono}; font-size:11px; color:${C.green}; font-weight:700;">YES</span>
    </div>`;
  }).join("");
  const cfg = data.config || {};
  const jobMeta = `${esc(cfg.script || "")} · ${esc(cfg.model || "")} · ${esc(cfg.prompt_name || "")} prompt${cfg.mask ? " · mask" : ""}`;
  return `
  <div style="height:100%; display:flex; flex-direction:column;">
    <div style="flex:0 0 auto; padding:12px 20px; border-bottom:1px solid ${C.bd2}; display:flex; align-items:center; gap:8px; overflow-x:auto;">
      <span style="margin-right:4px; white-space:nowrap;">
        <span style="font-size:11.5px; color:${C.fg5}; font-family:${C.mono};">${esc(R.jobId)}</span>
        <span style="display:block; font-size:10.5px; color:${C.fg3}; font-family:${C.mono}; margin-top:2px;">${jobMeta}</span>
      </span>${tabs}
      <span style="margin-left:auto; display:flex; align-items:center; gap:10px; white-space:nowrap;">
        ${revOn && S.rev.metrics ? `<span style="font-size:11px; font-family:${C.mono}; color:oklch(0.85 0.12 90);">${Object.values(S.rev.doc.frames).filter(e => e.done).length}/${frames.length} reviewed</span>` : ""}
        <button data-act="toggleReview" title="Judge each detection TP/FP and box what the model missed" style="display:flex; align-items:center; gap:7px; font-size:12px; font-weight:600; color:${revOn ? C.accDark : "oklch(0.85 0.12 90)"}; background:${revOn ? "oklch(0.85 0.12 90)" : "oklch(0.24 0.05 90)"}; border:1px solid oklch(0.5 0.09 90); padding:6px 12px; border-radius:8px; cursor:pointer;">
          ${revOn ? "Exit review" : "Review"}</button>
        <button data-act="togglePlay" title="${R.playing ? "Pause" : "Play through frames"}" style="display:flex; align-items:center; gap:7px; font-size:12px; color:${R.playing ? C.accDark : C.accFg}; background:${R.playing ? C.acc : C.accBg}; border:1px solid ${C.accBd2}; padding:6px 12px; border-radius:8px; cursor:pointer;">
          ${R.playing
            ? `<svg width="11" height="11" viewBox="0 0 12 12"><rect x="1" y="1" width="3.5" height="10" rx="1" fill="currentColor"/><rect x="7.5" y="1" width="3.5" height="10" rx="1" fill="currentColor"/></svg>Pause`
            : `<svg width="11" height="11" viewBox="0 0 12 12"><path d="M2 1 L11 6 L2 11 Z" fill="currentColor"/></svg>Play`}
        </button>
        <span style="font-size:10.5px; color:${C.fg5};">←/→ navigate</span>
      </span>
    </div>
    <div style="flex:1; min-height:0; display:flex;">
      <div data-scroll="gallery" style="width:212px; flex:0 0 212px; border-right:1px solid ${C.bd2}; overflow:auto; padding:12px;">
        ${gallery || `<div style="padding:40px 12px; text-align:center; color:${C.fg5}; font-size:12.5px;"><div style="width:40px; height:40px; margin:0 auto 12px; border-radius:10px; border:1.5px dashed oklch(0.34 0.014 250);"></div>No frames match this filter.</div>`}
      </div>
      <div style="flex:1; min-width:0; display:flex; flex-direction:column;">
        <div data-scroll="rescenter" style="flex:1; min-height:0; overflow:auto; padding:16px 18px;">
          <div style="display:flex; align-items:center; gap:10px; margin-bottom:12px;">
            <span style="font-family:${C.mono}; font-size:13px; color:oklch(0.92 0.006 250); font-weight:600;">${esc(sel.frame_id)}</span>
            <span style="font-size:10.5px; padding:2px 8px; border-radius:10px; background:oklch(0.22 0.012 250); border:1px solid ${C.bdBtn}; color:${C.fg2};">${sel.seconds}s · ${sel.attempts} attempt${sel.attempts > 1 ? "s" : ""}</span>
            <span style="font-size:10.5px; padding:2px 8px; border-radius:10px; background:${outcome[2]}; border:1px solid ${outcome[3]}; color:${outcome[1]}; font-weight:600;">${outcome[0]}</span>
            ${refImg ? `
            <div style="margin-left:auto; display:flex; align-items:center; gap:8px;">
              <span style="font-size:12px; color:${C.fg3};">Reference | inspection</span>
              <div data-act="toggleSplit" style="width:38px; height:21px; border-radius:12px; background:${R.split ? "oklch(0.55 0.11 225)" : C.bd}; position:relative; cursor:pointer;">
                <span style="position:absolute; top:2px; left:${R.split ? "19px" : "2px"}; width:17px; height:17px; border-radius:50%; background:oklch(0.96 0 0);"></span>
              </div>
            </div>` : ""}
          </div>
          ${R.split && refImg ? `
          <div style="display:grid; grid-template-columns:1fr 1fr; gap:10px;">
            <div style="border:1px solid ${C.bd}; border-radius:10px; overflow:hidden;">
              <div style="padding:6px 10px; font-size:11px; font-family:${C.mono}; color:${C.fg3}; background:${C.bg}; border-bottom:1px solid ${C.bd2};">REFERENCE · clean${masked ? " · masked" : ""}</div>
              <img src="${refImg}" style="width:100%; display:block;">
            </div>
            <div style="border:1px solid ${C.bd}; border-radius:10px; overflow:hidden;">
              <div style="padding:6px 10px; font-size:11px; font-family:${C.mono}; color:oklch(0.85 0.05 22); background:${C.bg}; border-bottom:1px solid ${C.bd2};">INSPECTION · now${masked ? " · masked" : ""}</div>
              <div ${imgWrapAttrs}><img id="revImg" src="${sel.img}" style="width:100%; display:block;">${boxes}</div>
            </div>
          </div>` : `
          <div style="border:1px solid ${C.bd}; border-radius:10px; overflow:hidden;">
            <div ${imgWrapAttrs}><img id="revImg" src="${sel.img}" style="width:100%; display:block;">${boxes}</div>
          </div>`}
          ${revOn ? `
          <div style="display:flex; gap:14px; margin-top:10px; font-size:11.5px; color:oklch(0.62 0.012 250); flex-wrap:wrap;">
            <span style="display:flex; align-items:center; gap:6px;"><span style="width:11px; height:11px; border:2px dashed ${REV_COL.unset}; border-radius:2px;"></span>to judge (click it)</span>
            <span style="display:flex; align-items:center; gap:6px;"><span style="width:11px; height:11px; border:2px solid ${REV_COL.tp}; border-radius:2px;"></span>TP</span>
            <span style="display:flex; align-items:center; gap:6px;"><span style="width:11px; height:11px; border:2px solid ${REV_COL.fp}; border-radius:2px;"></span>FP</span>
            <span style="display:flex; align-items:center; gap:6px;"><span style="width:11px; height:11px; border:2px solid ${REV_COL.fn}; border-radius:2px;"></span>missed (FN)</span>
          </div>` : `
          <div style="display:flex; gap:16px; margin-top:10px; font-size:11.5px; color:oklch(0.62 0.012 250);">
            <span style="display:flex; align-items:center; gap:6px;"><span style="width:11px; height:11px; border:2px solid oklch(0.7 0.18 150); border-radius:2px;"></span>kept detection</span>
            <span style="display:flex; align-items:center; gap:6px;"><span style="width:11px; height:11px; border:2px solid oklch(0.95 0.12 90); border-radius:2px;"></span>hovered region</span>
          </div>`}
          <div style="margin-top:16px;">
            <div style="font-size:10.5px; text-transform:uppercase; letter-spacing:0.08em; color:${C.fg5}; margin-bottom:6px; font-family:${C.mono};">Timeline · ${frames.length} frames</div>
            <div style="display:flex; gap:2px; align-items:flex-end; height:34px; padding:4px 2px; background:oklch(0.13 0.008 250); border:1px solid ${C.bd2}; border-radius:8px;">${timeline}</div>
          </div>
        </div>
      </div>
      <div data-scroll="resside" style="width:288px; flex:0 0 288px; border-left:1px solid ${C.bd2}; overflow:auto; display:flex; flex-direction:column;">
        ${revOn ? reviewSidebar(sel) : `
        <div style="padding:14px 16px 10px; border-bottom:1px solid oklch(0.22 0.01 250);">
          <div style="font-size:12px; text-transform:uppercase; letter-spacing:0.08em; color:${C.fg3}; font-family:${C.mono};">Regions</div>
          <div style="font-size:11.5px; color:${C.fg4}; margin-top:3px;">${sel.detections.length} kept detection(s) · ${sel.seconds}s${sel.error ? " · " + esc(sel.error) : ""}</div>
        </div>`}
        ${!revOn && !sel.detections.length ? `
        <div style="padding:34px 18px; text-align:center; color:${C.fg4}; font-size:12.5px; line-height:1.5;">
          ${sel.status === "failed"
            ? `<div style="width:38px; height:38px; margin:0 auto 12px; border-radius:50%; background:${C.redBg}; border:1px solid ${C.redBd}; display:flex; align-items:center; justify-content:center; font-size:16px;">✕</div>Frame failed.<br>${esc(sel.error || "")}`
            : `<div style="width:38px; height:38px; margin:0 auto 12px; border-radius:50%; background:oklch(0.22 0.05 150); border:1px solid oklch(0.4 0.08 150); display:flex; align-items:center; justify-content:center;"><svg width="18" height="18" viewBox="0 0 18 18" fill="none"><path d="M4 9.5 L7.5 13 L14 5" stroke="${C.green}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg></div>No anomaly kept.<br>Frame classified clean.`}
        </div>` : ""}
        ${revOn ? "" : `<div style="padding:8px 12px;">${verdicts}</div>`}
        ${sel.raw_response ? `
        <div style="padding:8px 12px 16px;">
          <div style="font-size:10.5px; text-transform:uppercase; letter-spacing:0.08em; color:${C.fg5}; margin-bottom:6px; font-family:${C.mono};">Raw model output</div>
          <div style="font-family:${C.mono}; font-size:10.5px; line-height:1.6; color:${C.fg3}; background:oklch(0.11 0.008 250); border:1px solid ${C.bd2}; border-radius:8px; padding:9px 11px; max-height:180px; overflow:auto; white-space:pre-wrap;">${esc(sel.raw_response.slice(0, 3000))}</div>
        </div>` : ""}
      </div>
    </div>
  </div>`;
}
function compareView(tabs, gallery, sel, selIdx) {
  const R = S.res;
  const other = R.compareData;
  const otherSel = other ? (other.frames || []).find(f => f.frame_id === sel.frame_id) : null;
  const col = (data, frame, dot, coordSize) => {
    const model = data ? (data.config.model || "?") : "—";
    const script = data ? (data.config.script || "") : "";
    const verdictRows = frame ? frame.detections.map(d => {
      const tm = TYPE_META[d.type] || TYPE_META.unknown;
      return `<div style="display:flex; align-items:center; gap:8px; padding:7px 0; border-top:1px solid oklch(0.2 0.01 250); font-size:12.5px;">
        <span style="font-size:10px; padding:2px 7px; border-radius:10px; color:${tm.fg}; background:${tm.bg}; border:1px solid ${tm.bd};">${d.type}</span>
        <span style="color:oklch(0.9 0.006 250);">${esc(d.label)}</span>
        <span style="margin-left:auto; font-family:${C.mono}; color:${C.green}; font-weight:700;">YES</span>
      </div>`;
    }).join("") : "";
    const empty = frame && !frame.detections.length
      ? `<div style="padding:14px 0; font-size:12px; color:${C.fg4};">${frame.anomaly === false ? "clean" : frame.status}</div>` : "";
    return `
    <div style="background:${C.bg}; display:flex; flex-direction:column; overflow:auto;">
      <div style="padding:10px 14px; display:flex; align-items:center; gap:8px; border-bottom:1px solid oklch(0.22 0.01 250);">
        <span style="width:8px; height:8px; border-radius:2px; background:${dot};"></span>
        <span style="font-size:12.5px; font-weight:600;">${esc(model)}</span>
        <span style="font-family:${C.mono}; font-size:10.5px; color:${C.fg4};">${esc(script)}</span>
        <span style="margin-left:auto; font-family:${C.mono}; font-size:11px; color:${C.fg3};">${data ? esc(data.job_id) : ""}</span>
      </div>
      ${frame ? `<div style="position:relative;"><img src="${frame.img}" style="width:100%; display:block;">${bboxOverlay(frame, coordSize)}</div>`
              : `<div style="padding:40px; text-align:center; color:${C.fg4}; font-size:12px;">no matching frame</div>`}
      <div style="padding:10px 14px;">${verdictRows}${empty}</div>
    </div>`;
  };
  const jobOpts = S.jobs.filter(j => j.job_id !== R.jobId && j.summary && j.summary.n_frames)
    .map(j => `<option value="${esc(j.job_id)}" ${R.compareJob === j.job_id ? "selected" : ""}>${esc(j.job_id)}</option>`).join("");
  return `
  <div style="height:100%; display:flex; flex-direction:column;">
    <div style="flex:0 0 auto; padding:12px 20px; border-bottom:1px solid ${C.bd2}; display:flex; align-items:center; gap:8px; overflow-x:auto;">
      <span style="font-size:11.5px; color:${C.fg5}; font-family:${C.mono}; margin-right:4px;">${esc(R.jobId)}</span>${tabs}
      <span style="margin-left:auto; font-size:12px; color:${C.fg3};">vs</span>
      <select data-change="compareJob" style="padding:5px 8px; border-radius:7px; background:${C.bgCard2}; border:1px solid ${C.bd3}; color:oklch(0.9 0.006 250); font-size:12px; font-family:${C.mono};">${jobOpts}</select>
    </div>
    <div style="flex:1; min-height:0; display:flex;">
      <div data-scroll="gallery" style="width:170px; flex:0 0 170px; border-right:1px solid ${C.bd2}; overflow:auto; padding:10px;">${gallery}</div>
      <div style="flex:1; min-width:0; display:grid; grid-template-columns:1fr 1fr; gap:1px; background:${C.bd2};">
        ${col(S.res.data, sel, C.acc, R.coordSize)}
        ${col(other, otherSel, "oklch(0.7 0.15 300)", R.compareCoordSize)}
      </div>
    </div>
  </div>`;
}

/* --------------- history --------------- */
function historyView() {
  const selId = S.res.jobId || (S.jobs[0] && S.jobs[0].job_id);
  const selJob = S.jobs.find(j => j.job_id === selId);
  const s = selJob && selJob.summary;
  return `
  <div data-scroll="page" style="height:100%; overflow:auto; padding:24px 28px;">
    <div style="display:grid; grid-template-columns:1.15fr 1fr; gap:22px; align-items:start;">
      <div>
        <h3 style="margin:0 0 12px; font-size:13px; font-weight:600; text-transform:uppercase; letter-spacing:0.08em; color:${C.fg3};">All jobs</h3>
        <div style="border:1px solid ${C.bd}; border-radius:11px; overflow:hidden; background:${C.bgCard};">
          <div style="display:grid; grid-template-columns:2fr 0.8fr 1fr 1.1fr; padding:10px 15px; background:${C.bg}; border-bottom:1px solid ${C.bd}; font-size:11px; text-transform:uppercase; letter-spacing:0.06em; color:${C.fg4}; font-family:${C.mono};">
            <span>Job</span><span style="text-align:right;">Frames</span><span style="text-align:right;">Anomalies</span><span style="text-align:right;">Status</span>
          </div>
          ${S.jobs.length ? S.jobs.map(j => jobRow(j, "2fr 0.8fr 1fr 1.1fr")).join("")
            : `<div style="padding:32px; text-align:center; color:${C.fg4}; font-size:12.5px;">No jobs yet.</div>`}
        </div>
      </div>
      <div>
        <h3 style="margin:0 0 12px; font-size:13px; font-weight:600; text-transform:uppercase; letter-spacing:0.08em; color:${C.fg3};">Report · ${esc(selId || "—")}</h3>
        ${selJob ? `
        <div style="border:1px solid ${C.bd}; border-radius:12px; overflow:hidden; background:${C.bgCard2};">
          <div style="padding:18px 20px; border-bottom:1px solid ${C.bd2};">
            <div style="font-size:15px; font-weight:650; margin-bottom:3px;">${esc(selJob.config.script)} — ${esc(selJob.config.prompt_name)} prompt</div>
            <div style="font-family:${C.mono}; font-size:11.5px; color:oklch(0.58 0.012 250);">${esc(selJob.config.model || "")} · status ${esc(selJob.status)}</div>
          </div>
          ${s ? `
          <div style="display:grid; grid-template-columns:repeat(3,1fr); gap:1px; background:${C.bd2};">
            <div style="background:${C.bgCard2}; padding:15px 18px;"><div style="font-family:${C.mono}; font-size:22px; font-weight:700;">${s.n_frames}</div><div style="font-size:11px; color:${C.fg3}; margin-top:2px;">frames</div></div>
            <div style="background:${C.bgCard2}; padding:15px 18px;"><div style="font-family:${C.mono}; font-size:22px; font-weight:700; color:${C.red};">${s.n_anomalous}</div><div style="font-size:11px; color:${C.fg3}; margin-top:2px;">anomalous</div></div>
            <div style="background:${C.bgCard2}; padding:15px 18px;"><div style="font-family:${C.mono}; font-size:22px; font-weight:700;">${s.wall_seconds}<span style="font-size:12px; color:${C.fg3};">s</span></div><div style="font-size:11px; color:${C.fg3}; margin-top:2px;">wall clock</div></div>
          </div>
          <div style="padding:18px 20px; border-top:1px solid ${C.bd2};">
            <div style="font-size:11px; text-transform:uppercase; letter-spacing:0.07em; color:${C.fg4}; font-family:${C.mono}; margin-bottom:10px;">Frame counts</div>
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:8px;">
              <div style="padding:10px 13px; border-radius:8px; background:${C.greenBg}; border:1px solid ${C.greenBd}; display:flex; justify-content:space-between;"><span style="font-size:12px; color:oklch(0.8 0.08 150);">Clean</span><span style="font-family:${C.mono}; font-weight:700; color:oklch(0.85 0.1 150);">${s.n_ok - s.n_anomalous}</span></div>
              <div style="padding:10px 13px; border-radius:8px; background:${C.redBg}; border:1px solid ${C.redBd}; display:flex; justify-content:space-between;"><span style="font-size:12px; color:oklch(0.82 0.09 22);">Anomalous</span><span style="font-family:${C.mono}; font-weight:700; color:oklch(0.85 0.11 22);">${s.n_anomalous}</span></div>
              <div style="padding:10px 13px; border-radius:8px; background:oklch(0.2 0.012 250); border:1px solid ${C.bdBtn}; display:flex; justify-content:space-between;"><span style="font-size:12px; color:${C.fg2};">Failed</span><span style="font-family:${C.mono}; font-weight:700;">${s.n_failed}</span></div>
              <div style="padding:10px 13px; border-radius:8px; background:oklch(0.2 0.012 250); border:1px solid ${C.bdBtn}; display:flex; justify-content:space-between;"><span style="font-size:12px; color:${C.fg2};">OK</span><span style="font-family:${C.mono}; font-weight:700;">${s.n_ok}</span></div>
            </div>
          </div>` : `<div style="padding:24px 20px; font-size:12.5px; color:${C.fg4};">Job has not produced results yet.</div>`}
          <div style="padding:14px 20px; border-top:1px solid ${C.bd2}; display:flex; gap:10px;">
            <button data-act="openReportJob" data-arg="${esc(selId)}" style="font-size:12.5px; color:oklch(0.85 0.012 250); background:${C.bgBtn}; border:1px solid ${C.bdBtn}; padding:8px 14px; border-radius:8px; cursor:pointer;">Open full report</button>
            <button data-act="openJob" data-arg="${esc(selId)}" style="font-size:12.5px; font-weight:600; color:${C.accDark}; background:${C.acc}; border:none; padding:8px 14px; border-radius:8px; cursor:pointer;">View frames</button>
          </div>
        </div>` : `<div style="color:${C.fg4}; font-size:12.5px;">Run a job to see its report here.</div>`}
      </div>
    </div>
  </div>`;
}

/* --------------- labels --------------- */
function labelsView() {
  const rows = S.labels;
  if (rows === null) return `<div style="padding:40px; color:${C.fg4}; font-size:13px;">Loading reviews…</div>`;
  const reviewedIds = new Set(rows.map(r => r.job_id));
  const unreviewed = S.jobs.filter(j => j.summary && j.summary.n_frames && !reviewedIds.has(j.job_id));
  const agg = rows.reduce((a, r) => {
    a.frames += r.metrics.progress.n_done;
    a.tp += r.metrics.objects.tp; a.fp += r.metrics.objects.fp; a.fn += r.metrics.objects.fn;
    a.samples += r.export.samples;
    return a;
  }, { frames: 0, tp: 0, fp: 0, fn: 0, samples: 0 });
  const statCard = (v, label, col) => `
    <div style="border:1px solid ${C.bd}; border-radius:12px; background:${C.bgCard2}; padding:14px 18px; flex:1;">
      <div style="font-family:${C.mono}; font-size:22px; font-weight:700; color:${col || "oklch(0.92 0.006 250)"};">${v}</div>
      <div style="font-size:11px; color:${C.fg4}; margin-top:2px;">${label}</div>
    </div>`;
  const btn = (act, arg, label, col, bg, bd) => `<button data-act="${act}" data-arg="${esc(arg)}" style="font-size:11.5px; font-weight:600; color:${col}; background:${bg}; border:1px solid ${bd}; padding:6px 12px; border-radius:7px; cursor:pointer; white-space:nowrap;">${label}</button>`;
  const reviewRows = rows.map(r => {
    const p = r.metrics.progress, o = r.metrics.objects, f = r.metrics.frames;
    const pct = p.n_frames ? Math.round(p.n_done / p.n_frames * 100) : 0;
    return `
    <div style="display:flex; align-items:center; gap:14px; padding:13px 20px; border-top:1px solid oklch(0.22 0.01 250);">
      <div style="flex:1; min-width:0;">
        <div style="font-family:${C.mono}; font-size:12.5px; color:oklch(0.9 0.006 250);">${esc(r.job_id)}</div>
        <div style="font-size:10.5px; color:${C.fg4}; margin-top:2px;">${esc(r.script || "")} · ${esc(r.model || "")} · updated ${esc((r.updated || "").slice(0, 16).replace("T", " "))}</div>
        <div style="display:flex; align-items:center; gap:8px; margin-top:6px;">
          <div style="flex:0 0 140px; height:5px; border-radius:4px; background:${C.bg}; overflow:hidden;"><div style="width:${pct}%; height:100%; background:${pct === 100 ? C.green : "oklch(0.85 0.12 90)"};"></div></div>
          <span style="font-family:${C.mono}; font-size:10.5px; color:${C.fg3};">${p.n_done}/${p.n_frames} frames</span>
        </div>
      </div>
      <div style="text-align:right; font-family:${C.mono}; font-size:11px; line-height:1.7; white-space:nowrap;">
        <div><span style="color:${REV_COL.tp};">${o.tp} TP</span> · <span style="color:${REV_COL.fp};">${o.fp} FP</span> · <span style="color:${REV_COL.fn};">${o.fn} FN</span></div>
        <div style="color:${C.fg4};">frames ${f.TP}/${f.FP}/${f.TN}/${f.FN}${r.export.exportable ? ` · ${r.export.samples} samples` : " · no ref → not exportable"}</div>
      </div>
      <div style="display:flex; gap:6px;">
        ${btn("openJobReview", r.job_id, pct === 100 ? "Open" : "Continue", C.accDark, "oklch(0.85 0.12 90)", "oklch(0.5 0.09 90)")}
        ${btn("exportReviewXlsx", r.job_id, "xlsx", "oklch(0.85 0.012 250)", C.bgBtn, C.bdBtn)}
        ${btn("deleteReviewOf", r.job_id, "Delete", "oklch(0.8 0.09 22)", "oklch(0.19 0.02 22)", "oklch(0.36 0.07 22)")}
      </div>
    </div>`;
  }).join("");
  const startRows = unreviewed.slice(0, 15).map(j => `
    <div style="display:flex; align-items:center; gap:12px; padding:11px 20px; border-top:1px solid oklch(0.22 0.01 250);">
      <div style="flex:1; min-width:0;">
        <div style="font-family:${C.mono}; font-size:12px; color:oklch(0.85 0.006 250);">${esc(j.job_id)}</div>
        <div style="font-size:10.5px; color:${C.fg4}; margin-top:1px;">${esc(j.config.script || "")} · ${esc(j.config.model || "")} · ${j.summary.n_frames} frames · ${j.summary.n_anomalous} anomalous</div>
      </div>
      ${btn("openJobReview", j.job_id, "Start review", C.accFg, C.accBg, C.accBd2)}
    </div>`).join("");
  return `
  <div data-scroll="page" style="height:100%; overflow:auto; padding:24px 28px;">
    <div style="max-width:980px;">
      <div style="display:flex; gap:14px; margin-bottom:18px;">
        ${statCard(rows.length, "reviewed jobs")}
        ${statCard(agg.frames, "frames confirmed")}
        ${statCard(`${agg.tp} / ${agg.fp} / ${agg.fn}`, "object TP / FP / FN")}
        ${statCard(agg.samples, "LoRA samples ready", agg.samples ? "oklch(0.85 0.12 90)" : undefined)}
      </div>
      <div style="border:1px solid ${C.bd}; border-radius:12px; background:${C.bgCard2}; overflow:hidden; margin-bottom:18px;">
        <div style="padding:14px 20px; font-size:14px; font-weight:600;">Reviews
          <span style="font-size:11px; font-weight:400; color:${C.fg4}; margin-left:8px;">labelling happens in Results → Review; this is the overview</span></div>
        ${reviewRows || `<div style="padding:26px 20px; font-size:12.5px; color:${C.fg4};">No review yet. Pick a job below and start judging its detections.</div>`}
      </div>
      <div style="border:1px solid ${C.bd}; border-radius:12px; background:${C.bgCard2}; overflow:hidden;">
        <div style="padding:14px 20px; font-size:14px; font-weight:600;">Jobs without a review</div>
        ${startRows || `<div style="padding:20px; font-size:12.5px; color:${C.fg4};">Every finished job has a review.</div>`}
      </div>
    </div>
  </div>`;
}

/* --------------- lora --------------- */
function loraView() {
  const st = S.lora.status;
  if (!st) return `<div style="padding:40px; color:${C.fg4}; font-size:13px;">Loading dataset status…</div>`;
  const a = st.aggregate;
  const target = st.target_samples || 300;
  const pct = Math.min(100, Math.round(a.samples / target * 100));
  const yesPct = (a.yes + a.no) ? Math.round(a.yes / (a.yes + a.no) * 100) : 0;
  const jobRows = st.per_job.map(j => `
    <div style="display:flex; align-items:center; gap:10px; padding:9px 18px; border-top:1px solid oklch(0.22 0.01 250); font-size:11.5px;">
      <span style="font-family:${C.mono}; flex:1; color:oklch(0.88 0.006 250); overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">${esc(j.job_id)}</span>
      <span style="color:${C.fg4};">${j.n_done}/${j.n_frames} frames</span>
      <span style="font-family:${C.mono}; white-space:nowrap;">${j.exportable
        ? `<span style="color:${REV_COL.tp};">${j.yes} YES</span> · <span style="color:${REV_COL.fp};">${j.no} NO</span>`
        : `<span style="color:${C.fg4};">no reference — skipped</span>`}</span>
    </div>`).join("");
  const le = st.last_export;
  const step = (done, label, sub) => `
    <div style="display:flex; gap:10px; padding:8px 0; align-items:flex-start;">
      <span style="flex:0 0 18px; height:18px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:10px; margin-top:1px; background:${done ? "oklch(0.65 0.13 150)" : C.bgBtn}; color:${done ? "oklch(0.13 0.008 250)" : C.fg3}; border:1px solid ${done ? "oklch(0.5 0.1 150)" : C.bdBtn};">${done ? "✓" : ""}</span>
      <span style="font-size:12.5px; color:${done ? C.fg2 : "oklch(0.88 0.006 250)"};">${label}${sub ? `<span style="display:block; font-size:10.5px; color:${C.fg4}; margin-top:1px;">${sub}</span>` : ""}</span>
    </div>`;
  return `
  <div data-scroll="page" style="height:100%; overflow:auto; padding:24px 28px;">
    <div style="max-width:980px; display:grid; grid-template-columns:1fr 1fr; gap:18px; align-items:start;">
      <div style="border:1px solid ${C.bd}; border-radius:12px; background:${C.bgCard2}; overflow:hidden;">
        <div style="padding:16px 18px 12px;">
          <div style="font-size:14px; font-weight:600; margin-bottom:12px;">Dataset readiness</div>
          <div style="display:flex; align-items:baseline; gap:8px;">
            <span style="font-family:${C.mono}; font-size:30px; font-weight:800; color:${a.samples >= target ? C.green : "oklch(0.85 0.12 90)"};">${a.samples}</span>
            <span style="font-size:12px; color:${C.fg3};">/ ${target} crop samples ${hint("Human-verified crop pairs harvested from Reviews: each judged detection (TP→YES label, FP→NO) and each missed box. 300 is the working lower bound from the CCTV LoRA paper.")}</span>
          </div>
          <div style="height:7px; border-radius:5px; background:${C.bg}; overflow:hidden; margin:10px 0 14px;"><div style="width:${pct}%; height:100%; background:${a.samples >= target ? C.green : "oklch(0.85 0.12 90)"};"></div></div>
          <div style="font-size:11.5px; color:${C.fg3}; margin-bottom:6px;">Class balance — <span style="color:${REV_COL.tp};">${a.yes} YES</span> · <span style="color:${REV_COL.fp};">${a.no} NO</span></div>
          <div style="height:7px; border-radius:5px; overflow:hidden; display:flex; background:${C.bg};">
            <div style="width:${yesPct}%; background:${REV_COL.tp};"></div><div style="flex:1; background:${REV_COL.fp};"></div>
          </div>
          ${st.balance_warning ? `<div style="margin-top:10px; font-size:11.5px; color:oklch(0.88 0.1 75); background:oklch(0.24 0.05 75); border:1px solid oklch(0.44 0.09 75); border-radius:8px; padding:8px 10px;">Balance is skewed — review more frames of the under-represented class (empty cross-session videos are the best NO source) before training.</div>` : ""}
          ${a.skipped_no_bbox ? `<div style="margin-top:8px; font-size:10.5px; color:${C.fg4};">${a.skipped_no_bbox} verdict(s) without a bbox (vlm_01/02) can't become crops.</div>` : ""}
        </div>
        <div style="border-top:1px solid ${C.bd2}; padding:8px 0 2px;">
          <div style="padding:4px 18px; font-size:11px; font-weight:600; color:${C.fg3};">Per reviewed job</div>
          ${jobRows || `<div style="padding:10px 18px 16px; font-size:12px; color:${C.fg4};">Nothing reviewed yet — start in the Labels tab.</div>`}
        </div>
        <div style="border-top:1px solid ${C.bd2}; padding:14px 18px;">
          <button data-act="loraExport" ${a.samples && !S.lora.exporting ? "" : "disabled"} style="font-size:12.5px; font-weight:600; color:${C.accDark}; background:${a.samples ? "oklch(0.85 0.12 90)" : C.bgBtn}; border:none; padding:9px 16px; border-radius:8px; cursor:${a.samples ? "pointer" : "not-allowed"}; opacity:${S.lora.exporting ? 0.6 : 1};">
            ${S.lora.exporting ? "Exporting…" : "Export dataset"}</button>
          <span style="font-size:11px; color:${C.fg4}; margin-left:10px;">→ ${esc(st.dataset_dir)}/ (train/val JSONL + crops)</span>
          ${le ? `<div style="margin-top:10px; font-family:${C.mono}; font-size:11px; color:${C.fg3};">last export: ${le.total} samples (train ${le.train} / val ${le.val}, YES ${le.yes} / NO ${le.no}) · prompt ${esc(le.prompt_sha1 || "")}</div>` : ""}
          ${S.lora.result && S.lora.result.log ? `<pre style="margin-top:8px; font-size:10.5px; color:${C.fg3}; background:oklch(0.11 0.008 250); border:1px solid ${C.bd2}; border-radius:8px; padding:8px 10px; max-height:140px; overflow:auto; white-space:pre-wrap;">${esc(S.lora.result.log)}</pre>` : ""}
        </div>
      </div>
      <div style="border:1px solid ${C.bd}; border-radius:12px; background:${C.bgCard2}; padding:16px 18px;">
        <div style="font-size:14px; font-weight:600; margin-bottom:6px;">Training pipeline</div>
        <div style="font-size:11px; color:${C.fg4}; margin-bottom:8px;">Steps 3–6 run on the RTX workstation — full commands in <span style="font-family:${C.mono};">RUNBOOK_LORA.md</span>, rationale in <span style="font-family:${C.mono};">docs/LORA_PLAN.md</span>.</div>
        ${step(a.samples >= target, `1 · Label ≥ ${target} crops in Review mode`, `${a.samples} ready`)}
        ${step(!!le, "2 · Export the dataset", le ? `done — copy ${esc(st.dataset_dir)}/ to the workstation` : "")}
        ${step(false, "3 · Train — QLoRA rank 8, vision frozen", "llamafactory-cli train tools/lora/qwen3vl_lora.yaml · minutes on the 3080 Ti")}
        ${step(false, "4 · Merge the adapter", "llamafactory-cli export (unquantized base)")}
        ${step(false, "5 · GGUF + official mmproj → serve", "convert_hf_to_gguf.py · try Ollama import, fallback llama-server --mmproj")}
        ${step(false, "6 · Benchmark A/B — go/no-go", "region precision ≥ +0.10, object recall drop ≤ 0.02, YES/NO format intact")}
        <div style="margin-top:10px; font-size:11px; color:${C.fg4}; border-top:1px solid ${C.bd2}; padding-top:10px;">The 29-case benchmark stays out of training — it is the eval set. The exporter's <span style="font-family:${C.mono};">--include-benchmark</span> flag exists only as a deliberate CLI decision.</div>
      </div>
    </div>
  </div>`;
}

/* --------------- settings --------------- */
function settingsView() {
  const st = S.settings || { ollama_url: "http://localhost:11434", storage: {} };
  const testLabel = S.ollamaTest === "ok" ? `Connected · ${(S.health && S.health.models.length) || 0} models`
    : S.ollamaTest === "testing" ? "Testing…" : S.ollamaTest === "fail" ? "Unreachable" : "Test connection";
  const testColor = S.ollamaTest === "ok" ? "oklch(0.8 0.11 150)" : S.ollamaTest === "fail" ? C.redFg : "oklch(0.85 0.012 250)";
  const sto = st.storage || {};
  const total = (sto.models || 0) + (sto.frames || 0) + (sto.results || 0);
  const pct = (x) => total ? Math.round(x / total * 100) : 0;
  const modelRows = S.models.map(m => `
    <div style="display:flex; align-items:center; gap:12px; padding:13px 20px; border-top:1px solid oklch(0.22 0.01 250);">
      <div style="flex:1;">
        <div style="font-size:13px; font-weight:600; color:oklch(0.9 0.006 250);">${esc(m.name)}</div>
        <div style="font-family:${C.mono}; font-size:10.5px; color:${C.fg4}; margin-top:1px;">${esc(m.tag)}${m.size ? " · " + m.size : ""}</div>
      </div>
      <span style="font-size:11.5px; color:${m.installed ? "oklch(0.8 0.1 150)" : C.fg3};">${m.installed ? "Installed" : "Not installed"}</span>
      ${m.installed
        ? `<button data-act="removeModel" data-arg="${esc(m.tag)}" style="font-size:11.5px; color:oklch(0.8 0.09 22); background:oklch(0.19 0.02 22); border:1px solid oklch(0.36 0.07 22); padding:6px 12px; border-radius:7px; cursor:pointer;">Remove</button>`
        : (S.pulling === m.tag
          ? `<span style="font-family:${C.mono}; font-size:11.5px; color:${C.acc}; width:44px; text-align:right;">${Math.round(S.pullPct)}%</span>`
          : `<button data-act="pullModel" data-arg="${esc(m.tag)}" style="font-size:11.5px; font-weight:600; color:${C.accFg}; background:${C.accBg}; border:1px solid ${C.accBd2}; padding:6px 12px; border-radius:7px; cursor:pointer;">Pull</button>`)}
    </div>`).join("");
  return `
  <div data-scroll="page" style="height:100%; overflow:auto; padding:24px 28px;">
    <div style="max-width:760px;">
      <div style="border:1px solid ${C.bd}; border-radius:12px; background:${C.bgCard2}; padding:18px 20px; margin-bottom:18px;">
        <div style="font-size:14px; font-weight:600; margin-bottom:14px;">Ollama connection</div>
        <div style="display:flex; gap:10px; align-items:flex-end;">
          <label style="flex:1; font-size:11.5px; color:${C.fg3}; display:flex; flex-direction:column; gap:6px;">Server URL
            <input value="${esc(st.ollama_url)}" data-change="ollamaUrl" style="padding:9px 11px; border-radius:8px; background:${C.bgInput}; border:1px solid ${C.bd3}; color:oklch(0.9 0.006 250); font-family:${C.mono}; font-size:12.5px;">
          </label>
          <button data-act="testOllama" style="font-size:12.5px; color:${testColor}; background:${C.bgBtn}; border:1px solid oklch(0.34 0.014 250); padding:9px 15px; border-radius:8px; cursor:pointer; white-space:nowrap;">${testLabel}</button>
        </div>
      </div>
      <div style="border:1px solid ${C.bd}; border-radius:12px; background:${C.bgCard2}; overflow:hidden; margin-bottom:18px;">
        <div style="padding:16px 20px; border-bottom:1px solid ${C.bd2}; font-size:14px; font-weight:600;">Model manager</div>
        ${modelRows}
      </div>
      <div style="display:grid; grid-template-columns:1fr 1fr; gap:18px;">
        <div style="border:1px solid ${C.bd}; border-radius:12px; background:${C.bgCard2}; padding:18px 20px;">
          <div style="font-size:14px; font-weight:600; margin-bottom:14px;">Defaults</div>
          <div style="display:flex; flex-direction:column; gap:12px; font-size:12.5px; color:${C.fg2};">
            <div style="display:flex; justify-content:space-between;"><span>Script</span><span style="font-family:${C.mono}; color:oklch(0.9 0.006 250);">vlm_05</span></div>
            <div style="display:flex; justify-content:space-between;"><span>Prompt</span><span style="font-family:${C.mono}; color:oklch(0.9 0.006 250);">Conservative</span></div>
            <div style="display:flex; justify-content:space-between;"><span>Retries</span><span style="font-family:${C.mono}; color:oklch(0.9 0.006 250);">2</span></div>
          </div>
        </div>
        <div style="border:1px solid ${C.bd}; border-radius:12px; background:${C.bgCard2}; padding:18px 20px;">
          <div style="font-size:14px; font-weight:600; margin-bottom:14px;">Storage</div>
          <div style="font-family:${C.mono}; font-size:22px; font-weight:700; margin-bottom:4px;">${fmtGB(total)} <span style="font-size:13px; color:${C.fg3};">GB</span></div>
          <div style="font-size:11.5px; color:oklch(0.58 0.012 250); margin-bottom:12px;">${st.n_jobs || 0} jobs · models + frames + results</div>
          <div style="height:7px; border-radius:5px; overflow:hidden; display:flex; background:${C.bg};">
            <div style="width:${pct(sto.models || 0)}%; background:oklch(0.6 0.11 225);"></div>
            <div style="width:${pct(sto.frames || 0)}%; background:oklch(0.6 0.11 150);"></div>
            <div style="width:${pct(sto.results || 0)}%; background:oklch(0.6 0.1 75);"></div>
          </div>
          <div style="display:flex; gap:14px; margin-top:10px; font-size:10.5px; color:oklch(0.58 0.012 250);">
            <span>Models ${fmtGB(sto.models || 0)}</span><span>Frames ${fmtGB(sto.frames || 0)}</span><span>Results ${fmtGB(sto.results || 0)}</span>
          </div>
        </div>
      </div>
      ${storageCleanupCard()}
    </div>
  </div>`;
}

function storageCleanupCard() {
  const st = S.storage;
  if (!st) return `<div style="margin-top:18px; font-size:12px; color:${C.fg4};">Loading storage details…</div>`;
  const fmtMB = (b) => b >= 1e9 ? (b / 1e9).toFixed(2) + " GB" : Math.max(1, Math.round(b / 1e6)) + " MB";
  const delBtn = (act, arg, disabled, why) => disabled
    ? `<span class="hint" data-tip="${esc(why)}" style="font-size:11px; color:${C.fg4};">in use</span>`
    : `<button data-act="${act}" data-arg="${esc(arg)}" style="font-size:11.5px; color:oklch(0.8 0.09 22); background:oklch(0.19 0.02 22); border:1px solid oklch(0.36 0.07 22); padding:5px 11px; border-radius:7px; cursor:pointer;">Delete</button>`;
  const row = (main, sub, right) => `
    <div style="display:flex; align-items:center; gap:12px; padding:10px 20px; border-top:1px solid oklch(0.22 0.01 250);">
      <div style="flex:1; min-width:0;">
        <div style="font-family:${C.mono}; font-size:12px; color:oklch(0.9 0.006 250); overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">${main}</div>
        <div style="font-size:10.5px; color:${C.fg4}; margin-top:1px;">${sub}</div>
      </div>${right}
    </div>`;
  const vids = st.videos.map(v => row(
    esc(v.video_id), `${v.n_frames} frames · ${fmtMB(v.bytes)}`,
    delBtn("deleteVideo", v.video_id, v.in_use, "a queued or running job uses these frames")))
    .join("") || `<div style="padding:12px 20px; font-size:12px; color:${C.fg4};">No extracted videos.</div>`;
  const busy = (s) => s === "running" || s === "queued";
  const jobs = st.jobs.map(j => row(
    esc(j.job_id), `${esc(j.script)} · ${esc(j.model)} · ${esc(j.status)} · ${fmtMB(j.bytes)}`,
    delBtn("deleteJob", j.job_id, busy(j.status), "cancel the job first")))
    .join("") || `<div style="padding:12px 20px; font-size:12px; color:${C.fg4};">No job results.</div>`;
  return `
      <div style="border:1px solid ${C.bd}; border-radius:12px; background:${C.bgCard2}; overflow:hidden; margin-top:18px;">
        <div style="padding:16px 20px; border-bottom:1px solid ${C.bd2};">
          <span style="font-size:14px; font-weight:600;">Storage cleanup</span>
          <span style="font-size:11.5px; color:${C.fg4}; margin-left:10px;">verdict cache ${fmtMB(st.cache_bytes)} — kept (re-runs on identical inputs are free thanks to it)</span>
        </div>
        <div style="display:grid; grid-template-columns:1fr 1fr;">
          <div style="border-right:1px solid ${C.bd2};">
            <div style="padding:10px 20px; font-size:12px; font-weight:600; color:${C.fg2};">Extracted videos</div>
            <div style="max-height:300px; overflow:auto;">${vids}</div>
          </div>
          <div>
            <div style="padding:10px 20px; font-size:12px; font-weight:600; color:${C.fg2};">Job results</div>
            <div style="max-height:300px; overflow:auto;">${jobs}</div>
          </div>
        </div>
      </div>`;
}

/* ---------------------------------------------------------------- events */
document.addEventListener("click", (ev) => {
  const el = ev.target.closest("[data-act]");
  if (!el) return;
  const fn = ACT[el.dataset.act];
  if (fn) fn(el.dataset.arg, ev);
});
document.addEventListener("change", (ev) => {
  const el = ev.target.closest("[data-change]");
  if (!el) return;
  const fn = CHANGE[el.dataset.change];
  if (fn) fn(el.value, el);
});
/* Text fields marked data-live mirror every keystroke into the state. 'change'
   alone only fires on blur, so a re-render mid-typing re-rendered the field
   from stale state and silently wiped what had been typed. */
document.addEventListener("input", (ev) => {
  const el = ev.target.closest("[data-live][data-change]");
  if (!el) return;
  const fn = CHANGE[el.dataset.change];
  if (fn) fn(el.value, el);
});
document.addEventListener("keydown", (ev) => {
  if (S.screen !== "results" || !S.res.data) return;
  const t = ev.target;
  if (t && t.id === "revLabel" && ev.key === "Enter") {
    S.rev.pendingLabel = t.value; ACT.revAddMissed(); return;
  }
  if (t && ("value" in t) && /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName)) return;
  if (revActive()) {
    const key = ev.key.toLowerCase();
    if (key === "escape" && (S.rev.pending || S.rev.draw)) {
      ev.preventDefault(); revClearPending(); S.rev.draw = false; render(); return;
    }
    // A pending missed box means the reviewer is writing its label. Letters are
    // label text, never shortcuts — whatever happened to the focus.
    if (S.rev.pending) return;
    if (key === "c") { ev.preventDefault(); ACT.revConfirm(); return; }
    if (key === "a") { ev.preventDefault(); ACT.revAllTpConfirm(); return; }
    if (key === "m") { ev.preventDefault(); ACT.revToggleDraw(); return; }
    if (key === "t" || key === "f") {
      const f = curFrame();
      const i = S.res.hoverV >= 0 ? S.res.hoverV : (f.detections.length === 1 ? 0 : -1);
      if (i >= 0) { ev.preventDefault(); revSetVerdict(i, key === "t" ? "tp" : "fp"); }
      return;
    }
  }
  if (!["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"].includes(ev.key)) return;
  ev.preventDefault();
  const R = S.res, frames = R.data.frames;
  const match = (f) => R.filter === "all" ? true
    : R.filter === "anomalous" ? !!f.anomaly
    : R.filter === "failed" ? f.status === "failed"
    : frameTypes(f).includes(R.filter);
  const visible = frames.map((f, i) => i).filter(i => match(frames[i]));
  if (!visible.length) return;
  const pos = Math.max(0, visible.indexOf(Math.min(R.sel, frames.length - 1)));
  const dir = (ev.key === "ArrowRight" || ev.key === "ArrowDown") ? 1 : -1;
  const next = visible[Math.min(visible.length - 1, Math.max(0, pos + dir))];
  S._kbNav = true;
  ACT.selFrame(next);
});
document.addEventListener("mouseover", (ev) => {
  const el = ev.target.closest("[data-hover]");
  if (el && S.res.hoverV !== +el.dataset.hover) ACT.hoverV(el.dataset.hover);
  const h = ev.target.closest(".hint");
  if (h) showTip(h);
});
document.addEventListener("mouseout", (ev) => {
  const el = ev.target.closest("[data-hover]");
  if (el && !ev.relatedTarget?.closest?.("[data-hover]")) ACT.unhoverV();
  if (ev.target.closest(".hint")) hideTip();
});
function showTip(el) {
  let t = document.getElementById("arsi-tip");
  if (!t) { t = document.createElement("div"); t.id = "arsi-tip"; document.body.appendChild(t); }
  t.textContent = el.dataset.tip || "";
  t.style.visibility = "hidden"; t.style.display = "block";
  const r = el.getBoundingClientRect(), tw = t.offsetWidth, th = t.offsetHeight;
  const x = Math.min(Math.max(8, r.left + r.width / 2 - tw / 2), innerWidth - tw - 8);
  const y = r.top - th - 9 >= 8 ? r.top - th - 9 : r.bottom + 9;
  t.style.left = x + "px"; t.style.top = y + "px"; t.style.visibility = "visible";
}
function hideTip() {
  const t = document.getElementById("arsi-tip");
  if (t) t.style.display = "none";
}
document.addEventListener("scroll", hideTip, true);

boot();
