# ARSI Studio — application spec (v0)

Web application around the existing vlm_01–05 scripts: load a tram CCTV video,
extract frames, run the pipeline of your choice (script × model × prompt),
watch progress live, browse boxed results, export a report. Local-only
(Ollama), English UI, must run on the RTX 3080 Ti workstation and degrade
gracefully on CPU (with honest time estimates).

Status: DRAFT — amend freely; this file is the contract every coding session
reads first.

## Architecture (3 layers, same repo)

```
arsi_core/      pure-Python engine, no UI, fully testable headless
app/backend/    FastAPI: jobs, SSE progress, model manager, static frontend
app/frontend/   SPA (vanilla JS or light React, no heavy build step)
```

The UI never imports vlm_0x directly; only `arsi_core` does.

## arsi_core

### Video → frames
- Input: uploaded video file (mp4/avi) or an existing frames directory.
- Params: `every_n` frames OR `every_s` seconds, `start`/`end` trim, output
  size.
- Output: `data/app/videos/<video_id>/frames/f%04d.jpg` + `meta.json`
  (fps, duration, frame count, extraction params).

### Masking (user-drawn, camera-wide)
Black out zones that must not influence detection (windows: exterior
movement + light changes). The camera is fixed, so a mask is drawn ONCE on
any frame and applies to every frame of that camera/video.

- Mask spec = named JSON preset: `data/app/masks/<name>.json` →
  `{name, camera, image_size: [w, h], zones: [{id, label, polygon: [[x,y],…]}]}`.
  Polygons (not just rectangles) in reference pixel space — the existing
  `data/masked/` frames follow exactly this convention (pure-black window
  contours) but were produced outside the repo; the app reproduces and
  replaces that external step.
- `apply_mask(image, mask) -> image`: fill polygons with black; scales
  zones if the frame size differs from `image_size` (same aspect).
- **Applied identically to the reference AND every inspection frame, at
  pipeline input** (not at extraction — raw frames stay reusable with a
  different mask). If only one side were masked, the diff pipelines
  (02/05) would detect the mask itself as change.
- vlm_05 verdict-cache keys must include a hash of the active mask
  (a mask change invalidates verdicts, same as a prompt change).
- Job config gets `mask: <name>|null`; report + export record it.

### Localizer vs judge (vlm_05)
vlm_05 is TWO stages that fail for different reasons, so both are chosen
independently: the **localizer** proposes candidate regions and the **judge**
(model + prompt) answers YES/NO on each crop.

- Registry: `arsi_core/localizers.py` — `photo` (shipped pixel diff, the
  DEFAULT), `photo+dino` (pixel diff + DINOv2 feature gate, recommended),
  `dino` (DINOv2 features alone, AnomalyDINO-style). All share vlm_05's
  `(regions, info)` contract with bboxes in reference pixel space and reuse
  its post-processing (person veto, salience cap, merge).
- `JobConfig.localizer` (None → `localizers.DEFAULT`) is validated and
  warmed up ONCE before the frame loop, so an unavailable backbone fails the
  job with one message instead of N frame errors. It is recorded in
  `public_dict()`, results.json, report.md/html and the xlsx.
- Every proposed region is recorded in `FrameResult.candidates`
  (`{bbox, area, channel, label, verdict, outcome, dropped_by}`) with
  `outcome = kept | rejected | filtered`, plus per-frame counts in
  `FrameResult.localization`. `rejected` = the judge answered NO; `filtered` =
  the judge answered YES and one of OUR post-filters overrode it. The
  post-filters are only evaluated on a YES, so a judge rejection is never
  attributed to a filter. Regions dropped BEFORE the judge (person veto,
  salience cap) never reach `candidates` and appear only as counts. The Results
  screen draws the dropped ones dashed under the normal overlay ("Show
  candidates"), which is what answers "was it localized at all?".
- The verdict-cache key deliberately does NOT include the localizer: a box is
  a box, so identical coordinates from two localizers share one verdict. This
  is what makes a localizer A/B nearly free — the gate produces a strict
  subset of the pixel diff's boxes (locked by
  `tests/test_localizers.py`).
- **Camera identity**: a mask is only valid for the viewpoint it was drawn on,
  so `camera` is seeded from the uploaded file name (`1760-cam05.mp4` →
  `1760-cam05`, `camera_slug()`), stored in `videos/<id>/origin.json`, and
  editable in the wizard. Never hardcode it — a 20-camera tram needs 20 masks.
- **LabelMe interop** (the team's annotation tool): same polygons, different
  keys (`shapes[].points` vs `zones[].polygon`, both image pixels).
  `MaskSpec.from_labelme`/`to_labelme` + `POST /api/masks/labelme` (import,
  returns zones for the editor without saving) and
  `GET /api/masks/<name>/labelme` (export). Rectangles/circles are expanded to
  polygons; open shapes (line, linestrip, point) enclose no area and are
  skipped, reported via `labelme_skipped()`. Integral coordinates stay ints so
  a round-trip through LabelMe keeps the mask hash — otherwise it would
  invalidate every cached verdict for nothing. This is the *occlusion* mask,
  not the Task-3 zones-of-interest below.

### Pipeline adapters
One uniform interface over the five scripts:

```python
run_frame(script: str,          # "vlm_01" .. "vlm_05"
          image: Path,
          reference: Path|None, # required for 02/05, optional 03 zones
          model: str,
          prompt: str,
          params: dict)         # script-specific knobs (thresholds, etc.)
    -> FrameResult
```

- vlm_01 single-image, vlm_02 reference compare, vlm_03 bbox output,
  vlm_04 YOLO-World hybrid, vlm_05 reference-diff (regions + crop judge).
- Adapters call refactored functions, not subprocesses; the scripts keep
  their CLI behaviour (import-safe refactor, defaults unchanged).
- vlm_05 reuses the benchmark's verdict cache (same key scheme) so repeated
  runs are near-free.

### FrameResult schema (the single JSON contract, UI + report + export)

```json
{
  "frame_id": "f0037",
  "image": "path.jpg",
  "status": "ok | failed | skipped",
  "attempts": 1,
  "seconds": 1.2,
  "anomaly": true,
  "detections": [
    {"bbox": [x1, y1, x2, y2], "label": "phone on seat",
     "type": "object | graffiti | damage | litter | unknown",
     "channel": "base | second | edge", "score_hint": null}
  ],
  "raw_response": "...",
  "error": null
}
```

Job result = `{job_id, config, started, finished, frames: [FrameResult],
summary: {n_frames, n_anomalous, n_failed, wall_seconds, s_per_call}}`.

### Error taxonomy (behaviour is part of the spec)
| error | behaviour |
|---|---|
| Ollama unreachable | job refuses to start; health banner in UI |
| model not installed | 409 + UI offers "Pull now" (streamed progress) |
| VLM reply unparseable (bad JSON / no YES-NO) | retry ≤ `max_retries` (default 2) with a format-reminder suffix, then `status=failed`, continue |
| frame decode error | `status=failed`, continue |
| per-call timeout (default 120 s) | retry once, then failed, continue |
| user cancel | job → `cancelled`, partial results kept |

Every failure is logged (structured JSONL per job under
`data/app/jobs/<job_id>/job.log`) and visible in the UI, never silent.

## Backend (FastAPI)

- `GET  /api/health` → `{ollama, gpu, cpu_only_warning, version}`
- `GET  /api/models` → installed + curated-recommended list (with the
  benchmark verdicts: GLM = alarm champion, qwen3.5 = inventory, etc.)
- `POST /api/models/pull {name}` → SSE progress
- `POST /api/videos` (upload) / `POST /api/videos/{id}/extract {params}`
- `GET  /api/demo-frames` → curated anomaly frames shipped with the repo
  (benchmark ground-truth cases, grouped real/gpt/variant/clean)
- `GET/POST/PUT/DELETE /api/masks` (named presets);
  `POST /api/masks/preview {frame, zones}` → masked image for live preview;
  `POST /api/masks/labelme {labelme, name?, camera?}` → zones + `skipped[]`;
  `GET /api/masks/{name}/labelme` → LabelMe download
- `GET  /api/localizers` → `{localizers[], default}`; each row carries
  `{key, name, summary, measured, recommended, available,
  unavailable_reason, first_use_download_mb}`. Availability is resolved
  server-side (DINOv2 needs torch) so the UI can grey out what this machine
  cannot run.
- `POST /api/jobs {script, model, prompt, frames[], reference?, mask?,
  localizer?, params, mode}` → `{job_id}`; mode = single | batch | compare
  (two configs, same frames). `localizer` applies to vlm_05 only; unknown
  key → 400, unavailable on this machine → 409 (same shape as a missing
  model).
- `GET  /api/jobs/{id}` state; `GET /api/jobs/{id}/events` SSE progress
  (per-frame: index, status, thumbnail ready)
- `POST /api/jobs/{id}/cancel` → sets a flag; the runner reads it between
  frames and between REGIONS inside a frame, so a job stops after the VLM
  call in flight rather than after the whole frame. Partial results are kept:
  the interrupted frame is stored with `status: "cancelled"` and the regions
  judged so far.
- `GET  /api/jobs` history; `GET /api/jobs/{id}/report.{md,html}`;
  `GET /api/jobs/{id}/export.xlsx` (rows in the ARSI_results_EN format)
- Jobs run in a worker thread queue (one VLM job at a time — Ollama is the
  bottleneck); state machine `queued → running → completed|failed|cancelled`.
  `results.json` is rewritten atomically after EVERY frame with
  `status: "running"`, so a crash keeps the finished frames. A saved job that
  still says running/queued with no live worker is reported as
  `interrupted` — never as a phantom `running`.

## Frontend screens (see docs/DESIGN_BRIEF.md for the visual spec)

1. **Home / dashboard** — health status, quick-start cards, recent jobs.
2. **New analysis wizard** — source (upload video | demo frames | previous
   extraction) → extraction params → mask step (pick a saved mask, draw a
   new one on any frame, or none) → pipeline config (script, model with
   installed-badge + pull button, prompt preset dropdown + editable text,
   advanced params) → review + launch.
3. **Run view** — progress bar + ETA, per-frame counters (done/anomalous/
   failed), live log tail, growing thumbnail strip, cancel button.
4. **Results view** — gallery of frames with box overlays; detail view with
   side-by-side reference|inspection and per-region verdicts; filters
   (anomalous only / failed / by type); video-timeline strip with flagged
   frames marked; compare mode = two result columns on the same frames.
5. **History & reports** — job table, report viewer, export buttons.
6. **Settings** — Ollama URL, defaults, model manager, data folder sizes.

## Non-goals (v1)
- No auth, no multi-user, no docker (plain `uvicorn` on localhost).
- No temporal/video persistence logic (future lever, design for it: job
  results keep frame ordering + timestamps).
- Zone editor (Task-3z LabelMe) = v2; v1 only displays zones if provided.

## Milestones
1. `arsi_core` + tiny CLI (`python -m arsi_core run ...`) + unit tests
   (parsers, error paths with a fake Ollama), integration smoke on 2 frames.
2. FastAPI backend + SSE, exercised by `curl`/httpx tests.
3. Claude Design mockup validated → frontend implemented against the API.
4. Polish: exports, compare mode, health/CPU warnings, failure-injection
   test pass (model absent, bad JSON, corrupt frame).

Each milestone ends runnable; commit per slice; `/code-review` before merge.
