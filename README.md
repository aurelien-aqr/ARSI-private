# ARSI - anomaly detection in tram interiors

Find **graffiti**, **vandalism**, **damage** and **forgotten objects** in tram
CCTV footage, on a fixed camera, **fully locally**: the vision-language model runs
on your own machine through [Ollama](https://ollama.com).

**ARSI Studio** is the local web app over **`arsi_core`**, the engine.

---

## First run, after cloning

```bash
bash setup.sh                     # venv + libraries + Ollama + the judge model (~8 GB)
venv/bin/pip install torch torchvision    # optional: unlocks the dino localizers
```

Ollama's installer registers a systemd service, so the server is up from now on -
you never start it by hand. `setup.sh` is idempotent and safe to re-run.

The reference frames the benchmark points at travel with the repo; the bulk
footage does not (`data/videos`, `data/raw`, `data/masked`, `weights/`).

## Every run after that

From the repository root, one command:

```bash
venv/bin/python -m uvicorn app.backend.main:app --port 8321
```

Then open <http://localhost:8321>: upload a video, mask the windows once (the
camera is fixed), pick pipeline × localizer × judge × prompt, watch the run,
browse the boxed results, export a report.

---

## How it works

`vlm_05` is **two stages that fail for different reasons**, so both are picked
independently:

```
reference  ─┐
            ├─►  LOCALIZER  ─►  candidate regions  ─►  JUDGE (VLM, one crop at a time)  ─►  boxes
inspection ─┘     (no VLM)                                YES / NO
```

Asking a VLM about a whole tram frame does not work - it invents anomalies out of
worn seats and posters. Asking it about one crop against the same crop of a clean
reference does.

**Localizers** (`arsi_core/localizers.py`), picked per run in the Studio:

- `photo` - photometric diff against the reference.
- `photo+dino` - the diff proposes, DINOv2 features veto.
- `dino` - AnomalyDINO patch features, no pixel comparison.
- `dino+dinomaly` - `dino`, plus a per-camera Dinomaly model as a second veto.

**Judge**: `haervwe/GLM-4.6V-Flash-9B` × the `conservative` prompt - the default
everywhere, and what `setup.sh` installs.

Which localizer and which judge to use, and what each was measured on:
[`docs/DECISIONS.md`](docs/DECISIONS.md).

---

## Repo

```
arsi_core/    engine: video, masking, localizers, adapters, runner, scoring
app/          FastAPI backend (jobs + SSE) + the SPA
benchmark/    the ground truth, the runs, the verdict cache
docs/         SPEC (contracts), DECISIONS (verdicts), and the measured notes
tools/        localizer research, Dinomaly training, doc builders, exporters
vlm_01..05    the standalone scripts, still runnable from the repo root
```

`venv/bin/python -m pytest tests/` - 186 tests, no GPU, no Ollama, no network.
Headless engine CLI: `python -m arsi_core run --script vlm_05 --localizer dino …`.

Target hardware: Ubuntu x86_64 + RTX 3080 Ti (12 GB). CPU works and the app warns
you how slow it will be. Judge fine-tuning: `RUNBOOK_LORA.md`.

---

## When something breaks

| symptom | fix |
|---|---|
| `could not reach the Ollama server` | `systemctl status ollama`, or `ollama serve` by hand |
| a localizer is greyed out | needs torch, or a per-camera checkpoint for `dino+dinomaly` |
| `dino+dinomaly` behaves like `dino` | no checkpoint for that camera - it degrades on purpose |
| out of memory | an 8-9B model needs ~7-8 GB free |
| `ModuleNotFoundError` | you ran `python`, not `venv/bin/python` |

---

*ARSI - VŠB-TUO FEI.*
