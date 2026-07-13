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
- Params: `every_n` frames OR `every_s` seconds, `start`/`end` trim, optional
  privacy mask (reuse the existing masked-frame convention), output size.
- Output: `data/app/videos/<video_id>/frames/f%04d.jpg` + `meta.json`
  (fps, duration, frame count, extraction params).

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
- `POST /api/jobs {script, model, prompt, frames[], reference?, params, mode}`
  → `{job_id}`; mode = single | batch | compare (two configs, same frames)
- `GET  /api/jobs/{id}` state; `GET /api/jobs/{id}/events` SSE progress
  (per-frame: index, status, thumbnail ready)
- `POST /api/jobs/{id}/cancel`
- `GET  /api/jobs` history; `GET /api/jobs/{id}/report.{md,html}`;
  `GET /api/jobs/{id}/export.xlsx` (rows in the ARSI_results_EN format)
- Jobs run in a worker thread queue (one VLM job at a time — Ollama is the
  bottleneck); state machine `queued → running → completed|failed|cancelled`.

## Frontend screens (see docs/DESIGN_BRIEF.md for the visual spec)

1. **Home / dashboard** — health status, quick-start cards, recent jobs.
2. **New analysis wizard** — source (upload video | demo frames | previous
   extraction) → extraction params → pipeline config (script, model with
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
