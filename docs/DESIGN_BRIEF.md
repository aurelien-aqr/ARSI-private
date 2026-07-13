# ARSI Studio — design brief (paste this into Claude Design)

You are designing **ARSI Studio**, a local web app for a public-transport
operator's R&D team. It analyses tram CCTV footage with local vision-language
models (via Ollama) to find anomalies: forgotten objects (bags, phones,
wallets), graffiti/tags, vandalism (slashed seats), litter. Users are
engineers/researchers, not the general public. Everything runs on one
workstation; there is no cloud, no login.

Language: **English**. Tone: technical, calm, trustworthy — a lab instrument,
not a consumer app. Dark-mode friendly (CCTV review rooms are dim). Dense but
readable; keyboard-friendly galleries.

## Screens to design

1. **Home / dashboard**
   - Health strip: Ollama status, GPU vs CPU badge (CPU shows a warning:
     "VLM calls take 2–4 min each on CPU"), models installed count.
   - Quick-start cards: "Analyze a video", "Try demo frames", "Resume last".
   - Recent jobs table (name, pipeline, frames, anomalies found, status).

2. **New analysis (wizard, 3–4 steps)**
   - Step 1 source: upload video (drag-drop) OR pick from bundled demo
     frames (thumbnail grid with labels like "real: forgotten phone",
     "synthetic: graffiti") OR reuse a previous extraction.
   - Step 2 extraction (video only): every-N-frames / every-N-seconds,
     trim range with a filmstrip preview, estimated frame count.
   - Step 3 pipeline: script selector (5 pipelines, one-line description
     each, "vlm_05 reference-diff — recommended" highlighted); model
     dropdown where NOT-installed models show a "Pull (4.7 GB)" button with
     progress; prompt preset dropdown (Conservative / Lenient / Custom)
     opening an editable text area; reference-frame picker (required for
     pipelines 02/05); advanced params accordion (thresholds, retries).
   - Step 4 review + big Launch button, with a time estimate
     ("~120 VLM calls ≈ 2 min on GPU / ~5 h on CPU").

3. **Run view (live)**
   - Prominent progress bar with ETA + counters: processed / anomalous /
     failed / retried.
   - Growing thumbnail strip: each finished frame slides in, green ring =
     clean, red ring = anomaly, grey = failed (with retry icon).
   - Collapsible live log (monospace tail).
   - Cancel button (keeps partial results).

4. **Results view** (the heart of the app)
   - Left: filterable frame gallery (all / anomalous / failed / by type).
   - Center: selected frame LARGE with bounding-box overlays + labels
     (e.g. "phone on seat"); toggle side-by-side reference|inspection view.
   - Bottom: video-timeline strip — one tick per frame, red ticks where
     anomalies were found; click to jump.
   - Right: per-region verdict list (label, type chip, YES/NO, seconds).
   - **Compare mode**: same frames, two configs (e.g. two models) as two
     synchronized columns with their two verdict sets.
   - Export bar: Report (HTML/MD), results.json, XLSX.

5. **History & report viewer** — jobs table + rendered report page
   (summary numbers, confusion-style counts, per-frame table).

6. **Settings** — Ollama URL + test button, model manager (installed list,
   sizes, pull/remove), default script/model/prompt, storage usage.

## Components that need care
- Model selector with installed/not-installed state and inline pull progress.
- Prompt editor: preset dropdown + textarea + "reset to preset" + a note
  that changing the prompt invalidates the verdict cache.
- Error surfaces: banner (Ollama down), toast (frame failed, retrying),
  inline card (model missing → pull CTA). Failures must be visible but
  never modal-blocking a running batch.
- Empty states for every screen (no jobs yet, no models installed).

## Real data to design with (uploaded alongside this brief)
- Annotated CCTV frames with boxes (green/red/blue) — use these as the
  gallery/detail imagery, they are the real output.
- A sample results JSON and a sample markdown report — use their real
  field names and realistic numbers (e.g. "29 cases, F1 0.919, 0.7 s/call").
- Real label strings for verdict lists: "phone on seat", "graffiti tag
  'XRP' on panel", "torn seat cushion", "black backpack on floor".

## Constraints for the exported code
- Single-page app talking to a local FastAPI backend (REST + SSE for
  progress). No cloud fonts/CDNs — fully offline. No heavy framework
  requirements; plain React or vanilla is fine. Standalone HTML export
  must be wirable to `fetch('/api/...')` endpoints listed in docs/SPEC.md.
