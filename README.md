# ARSI-VLM — Tram Interior Anomaly Detection

Detect **graffiti**, **vandalism**, and **forgotten objects** in tram interiors
using a local **vision-language model (VLM)** — Qwen2.5-VL served through
[Ollama](https://ollama.com). Runs fully locally: no cloud, no API key.

---

## Target hardware

| Item  | Value                                                  |
|-------|--------------------------------------------------------|
| OS    | Ubuntu (x86_64)                                        |
| GPU   | NVIDIA RTX 3080 Ti, 12 GB VRAM (CPU-only works, slow)  |
| Model | `qwen3-vl:8b-instruct` default; override with `--model`|

> `NUM_CTX` / `NUM_PREDICT` / `TEMPERATURE` are shared runtime settings tuned
> for 8-9B models on the target GPU. The model itself is a per-run choice:
> every script accepts `--model <ollama-name>` (see also `bench_grid.py` for
> sweeping several models). GPU-day procedure: **RUNBOOK_GPU.md**.

---

## Quick start

```bash
git clone https://github.com/mpyt/ARSI-vlm.git
cd ARSI-vlm
bash setup.sh                 # venv + libraries + Ollama + model download (~6 GB)
source venv/bin/activate      # do this in every new terminal
```

Then put your images here:

```
data/reference/   clean reference image   (e.g. tram_1762_reference.jpg)
data/raw/         frames to inspect        (tram_1762_v1_f0001.jpg)
data/masked/      masked frames            (tram_1762_v1_f0001_masked.jpg)
```

And run a script:

```bash
python vlm_01_single_image.py        # analyse a single image
python vlm_02_reference_compare.py   # compare against a clean reference
python vlm_03_bounding_box.py        # draw bounding boxes -> results/
```

> Run the scripts **from the repository root**. Paths inside each script are
> anchored to the repo root, so `data/...` and `results/...` always resolve.
>
> If you see "could not reach the Ollama server", open a second terminal, run
> `ollama serve`, then try again.

---

## The three scripts

| Script | Input | Output |
|--------|-------|--------|
| `vlm_01_single_image.py`      | one image | structured text report |
| `vlm_02_reference_compare.py` | reference + masked inspection image | structured text (differences only) |
| `vlm_03_bounding_box.py`      | one image | JSON detections + annotated image in `results/` |
| `vlm_04_hybrid_detect.py`     | reference + inspection image | confirmed forgotten objects: JSON + annotated image in `results/` |
| `vlm_05_reference_diff.py`    | reference + inspection image | abandoned objects via change detection: JSON + annotated image in `results/` |

`vlm_04` is a **hybrid** POC for forgotten personal objects (phone, wallet, bag):
an open-vocabulary detector (**YOLO-World**) localizes candidate objects, an
optional **reference filter** keeps only what is NEW versus a clean reference,
and the local **VLM confirms/labels** each surviving crop.

`vlm_05` takes a different route that works when the camera is **fixed**: it
**diffs** the inspection frame against the clean reference to localize whatever
changed (YOLO-World misses tiny objects like a wallet on the floor; the diff
does not), then the **VLM classifies** each changed region as an anomaly or not
(person / reflection / lighting are rejected). Localization is multi-channel
(base photometric diff + a bounded low-threshold channel for low-contrast
objects + an added-edge channel for faint graffiti) with a YOLOv8n **person
veto** — design rationale and measured numbers live in the USER CONFIG comments.

Benchmarking lives in `benchmark/` (frame- and object-level metrics against a
29-case hand-labelled ground truth, resumable VLM cache, localizer-only eval)
and `bench_grid.py` (model × task × image sweep that fills the ARSI results
spreadsheet). See `benchmark/README.md` and `RUNBOOK_GPU.md`.

Structured text format:

```
GRAFFITI: yes/no - note
VANDALISM: yes/no - note
FORGOTTEN OBJECT: yes/no - note
DESCRIPTION: ...
SEVERITY: 1-5
```

`vlm_03` returns JSON (it carries normalized 0–1 bounding boxes) and colours each
box by severity (green = low → red = high).

---

## What you may change

Every script has a clearly marked block:

```python
# =====================================================
#  USER CONFIG  ---  the ONLY part you are meant to edit
# =====================================================
```

Inside it you may freely change the **image paths** and the **`PROMPT`**.
**Do not touch** the `HARDWARE-LOCK` block.

---

## Folder structure

```
ARSI-vlm/
├── setup.sh
├── requirements.txt
├── README.md
├── .gitignore
├── vlm_01_single_image.py
├── vlm_02_reference_compare.py
├── vlm_03_bounding_box.py
├── vlm_04_hybrid_detect.py
├── data/
│   ├── reference/    clean reference images
│   ├── raw/          raw frames to inspect
│   └── masked/       masked frames (windows blacked out)
└── results/          annotated output images
```

---

## Troubleshooting

| Symptom                                  | Fix                                                        |
|------------------------------------------|------------------------------------------------------------|
| `could not reach the Ollama server`      | Run `ollama serve` in another terminal.                    |
| `model 'qwen2.5vl:7b' is not installed`  | Run `ollama pull qwen2.5vl:7b`.                            |
| `image not found: ...`                   | Upload images into `data/...` and edit the path in USER CONFIG. |
| Out-of-memory (OOM) on the GPU           | Close other GPU programs; the 7B model needs ~7–8 GB free. |
| `ModuleNotFoundError: ollama` / `PIL`    | You forgot `source venv/bin/activate`.                     |

---

*ARSI — VŠB-TUO FEI. Local inference with Ollama + Qwen2.5-VL.*
