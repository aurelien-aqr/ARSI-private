# vlm_05 anomaly-detection benchmark

A reproducible benchmark for `vlm_05_reference_diff.py` (reference-diff + VLM
anomaly detection in a fixed-camera tram).

## What it measures

Primary metric — **frame-level binary anomaly detection**. Each labelled image is
diffed against its clean reference; the VLM classifies every change region in
`filter` mode; the frame is **flagged** if ≥ 1 region survives.

| | predicted anomaly | predicted clean |
|---|---|---|
| actual anomaly | TP | FN |
| actual clean   | FP | TN |

Reported: **accuracy, precision, recall (sensitivity), specificity, F1**, the
confusion matrix, and a **per-anomaly-type recall** breakdown
(object / graffiti / damage / litter).

## Files

- `ground_truth.json` — the labelled dataset (hand-labelled here). Each case =
  an inspection image, the reference to diff it against, `has_anomaly`, the
  anomaly `types`, and a human note. **24 cases: 17 anomalous, 7 clean.**
  - Anomalous: 6 real CCTV frames (forgotten objects) + 10 AI-generated frames
    (objects, graffiti, seat damage, litter, one crowd scene) + 1 second
    synthetic scene ("variant").
  - Clean (false-positive tests): reference-vs-itself (×2), a clean AI frame,
    and 4 empty same-session frames camera-aligned to the reference.
- `run_benchmark.py` — runs the pipeline on every case and writes the report.
- `report.md` — human-readable results (regenerated after every case).
- `results.json` — machine-readable per-case results.
- `cache.json` — per-region VLM verdicts (see "Resumable" below).
- `annotated/<id>.jpg` — each frame with kept regions boxed + VLM label.

## How to run

```bash
# from the repository root, with the venv active and `ollama serve` running:
python benchmark/run_benchmark.py
```

The benchmark inherits the model, prompt and thresholds from
`vlm_05_reference_diff.py`, so it always scores the *current* configuration of the
script under test. Cases are processed cheapest-first (fewest regions), so the
clean/negative cases and the confusion matrix populate first.

## Resumable (important on CPU)

Each VLM verdict is cached in `cache.json`, keyed by (image, reference, region
box, model, prompt). The VLM runs at ~1–2 s/region on the target GPU but
~3–4 min/region on CPU, so a full run is minutes on GPU and several hours on CPU.
If interrupted, **just re-run the same command** — cached regions are skipped and
it continues. Changing the model or the prompt invalidates the cache
automatically (the key includes both), so re-running after editing the prompt
re-evaluates only what changed.

## Current results & the prompt A/B (do on GPU)

The committed `report.md` / `results.json` are the completed **CPU** run, which used
**`PROMPT_LENIENT`** (the original prompt) with all current post-filters. Headline:
frame-level F1 **0.941** (1 FP, 1 FN); object-level recall **0.911** (41/45) —
object 0.94, damage 1.00, litter 1.00, graffiti 0.67; region precision **0.744**.

`vlm_05_reference_diff.py` now DEFAULTS to the conservative **`PROMPT`**, written to
cut the residual false positives (hallucinated objects/graffiti on empty seats, a
person's own clothing, metal scratches read as graffiti). It has **not** been scored
yet: a prompt change invalidates the whole cache → a full re-run, which is hours on
CPU but minutes on GPU. To evaluate it **on GPU**:

```bash
cp benchmark/report.md benchmark/report_lenient_baseline.md   # keep the baseline
ollama ps                                    # confirm PROCESSOR shows GPU
python benchmark/run_benchmark.py            # re-scores all 24 cases with PROMPT
# then diff report.md against report_lenient_baseline.md
```

To A/B, swap the default at the bottom of the prompt block in
`vlm_05_reference_diff.py` (`PROMPT = ...` vs `PROMPT = PROMPT_LENIENT`). Two
precision guards run regardless of prompt, both whole-word matched:
`is_non_anomaly()` (person / "disappeared" labels) and `is_implausible()` (unnamed
YES, or a small-object name on a region > SMALL_OBJECT_MAX_AREA).

## Extending the dataset

Add entries to `ground_truth.json`. `image` and the reference are repo-relative
paths; `reference` is a key into the `references` map. Images of a different size
than the reference are uniformly resized onto it (fine as long as they share the
camera framing / aspect ratio). To add clean negatives from a new camera, pick
empty frames from the same recording session as that camera's reference.
