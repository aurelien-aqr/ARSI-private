# GPU day runbook — ARSI-vlm

> **STATUS 2026-07-13: steps 1–5 are DONE.** Benchmarked: qwen3-vl (both
> prompts), qwen3.5:9b (both prompts), InternVL3_5, GLM-4.6V-Flash-9B,
> minicpm-v4.6 (disqualified — see report note). The spreadsheet grid was
> filled by hand instead of bench_grid.py (11 models in ARSI_results_EN.xlsx).
> Results + verdicts: benchmark/README.md "GPU results" section.
> Remaining GPU work = step 6 (optional) and future precision levers.

Everything below was prepared and smoke-tested on CPU (2026-07-12). On the
RTX 3080 Ti workstation each step is minutes, not hours. Do them in order —
each produces a result the next one uses.

## 0) Setup (~15 min, mostly downloads)

```bash
git clone git@github.com:aurelien-aqr/ARSI-private.git ARSI-vlm && cd ARSI-vlm
bash setup.sh                    # venv + libs + ollama + qwen2.5vl:7b
source venv/bin/activate

# data: reference/, anomalies/ and the benchmark's specific masked negatives
# travel with the repo. The BULK frames (data/raw, data/masked, data/videos)
# do not; if the workstation doesn't already have them, copy once:
#   rsync -av laptop:~/Documents/vsb/ARSI-vlm/data/ data/

# models for the grid + classifier sweep (~45 GB total, trim as needed):
for m in qwen3-vl:8b-instruct qwen3.5:9b blaifa/InternVL3_5:8b \
         openbmb/minicpm-v4.6 haervwe/GLM-4.6V-Flash-9B \
         hf.co/mradermacher/Cosmos-Reason2-8B-GGUF:Q4_K_M \
         llama3.2-vision:11b; do ollama pull $m; done

# CONFIRM THE GPU IS ACTUALLY USED (the whole point):
nvidia-smi                                   # driver present
ollama run qwen3-vl:8b-instruct "hi" && ollama ps
# -> PROCESSOR column must say "100% GPU". If it says CPU, fix before anything.
```

## 1) Sanity (~2 min, no VLM)

```bash
python benchmark/eval_localization.py --variants shipped --quiet
# expected: instance recall 45/45 (GT box of gpt_03 fixed 2026-07-12)
```

## 2) Benchmark: conservative prompt × new localizer (~20-40 min)

The CPU run of 2026-07-12 scored the conservative PROMPT on the OLD 24-case
GT + single-channel localizer (see benchmark/report_conservative_cpu.md if
present, else the repo's committed report.md). Now score the CURRENT stack —
29 cases, multi-channel localizer, person filter:

```bash
python benchmark/run_benchmark.py
# base-channel region boxes are unchanged -> most verdicts come from cache;
# only new-channel boxes and the 5 new cross-session negatives hit the VLM.
cp benchmark/report.md benchmark/report_conservative_multichannel.md
```

Read: frame F1, object recall (lenient + strict IoU), region precision,
per-source table (real vs gpt), and FP counts on the 5 cross-session
negatives (neg_v2/v3/v4_*) — that last number is the deployment story.

## 3) Prompt A/B (~30 min)

```bash
# swap the default prompt: in vlm_05_reference_diff.py set  PROMPT = PROMPT_LENIENT
python benchmark/run_benchmark.py          # full re-run (new fingerprint)
cp benchmark/report.md benchmark/report_lenient_multichannel.md
# swap back to the conservative PROMPT afterwards.
```

Decision to make: does the conservative prompt cut region-FPs without hurting
graffiti/damage recall? Keep whichever wins; note both in the internship report.

## 4) Classifier model sweep on vlm_05 (~30 min/model)

For each candidate, change `MODEL_NAME` in vlm_05_reference_diff.py, re-run
the benchmark, save the report (the cache keys include the model, so nothing
is lost between sweeps):

```bash
for M in qwen3.5:9b blaifa/InternVL3_5:8b haervwe/GLM-4.6V-Flash-9B \
         openbmb/minicpm-v4.6; do
  sed -i "s|^MODEL_NAME = .*|MODEL_NAME = \"$M\"|" vlm_05_reference_diff.py
  python benchmark/run_benchmark.py
  cp benchmark/report.md "benchmark/report_$(echo $M | tr '/:' '__').md"
done
sed -i 's|^MODEL_NAME = .*|MODEL_NAME = "qwen3-vl:8b-instruct"|' vlm_05_reference_diff.py
```

Note: qwen3.5:9b has NO grounding but vlm_05 never asks for coordinates —
it may well beat qwen3-vl as the crop judge (it was the best whole-frame
model in the manual grid).

## 5) Spreadsheet grid: 8 models × 4 tasks (~1-2 h)

```bash
python bench_grid.py --dry-run          # 112 calls planned
python bench_grid.py                    # writes results/grid_results.csv as it goes
python bench_grid.py --to-xlsx ../ARSI_results_EN.xlsx   # or export at the end
```

Resumable: re-running skips rows already in the CSV. Inference times are now
REAL (GPU) — this fills the spreadsheet's time column at last. Correctness /
Rating stay empty: judge them against the images by hand (that column is the
supervisor's protocol).

Task `3z` is the ORIGINAL study's Task 3 (zones of interest given TO the
model, zones from benchmark/zones_tram_1762.json) — clarify with the
supervisor which interpretation he wants in the sheet; the grid records both
(`3` = model outputs boxes, `3z` = zones as input).

## 6) If time remains

- **Cascade probe**: score openbmb/minicpm-v4.6 as classifier (step 4). If its
  FN rate on cached crops is ~0, it becomes the cheap screener in front of
  qwen3-vl (big deployment speedup).
- **Temporal prototype**: the 4 videos in data/videos are the input for
  persistence logic ("forgotten = present N frames with no person nearby") —
  design notes in the internship journal 2026-07-12.
- **AD-Copilot-Thinking** (jiang-cc/AD-Copilot-Thinking, transformers, not
  Ollama): the closest research system to vlm_05 (visual in-context
  comparison); worth one afternoon of comparison.

## Known facts to not rediscover

- gpt_03's faint XRP tag IS catchable — the multi-channel localizer boxes it at
  [1332,536,1412,628] and every judge names it "XRP graffiti". The earlier
  "unreachable" claim came from a misplaced GT box (it covered the ventilation
  grille 200 px to the left; fixed 2026-07-12). Expected localization recall
  is now **45/45**.
- Lowering the BASE threshold below ~30 MERGES busy frames into mega-blobs
  that MAX_AREA then deletes (real_f0112 4/4 -> 0/4 at thr 25). Extra recall
  must come from the bounded ADD channels, never from the base threshold.
- Cross-session empty frames produce 15-37 candidate regions each (exposure
  drift + onboard-display content). The VLM is what keeps specificity; the
  structural fix (rolling / per-lighting reference bank) is future work.
- The reference frame itself contains an equipment bag at the very bottom
  edge and the onboard display is ON: benign diffs there are expected.
