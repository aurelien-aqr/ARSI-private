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
# expected: instance recall 45/45, strict IoU>=0.3 37/45, biggest box 734,400 px
#           (GT box of gpt_03 fixed 2026-07-12; merge shipped 2026-07-21)
# The strict column and the box size are the mega-blob canary — a region merge
# that chains neighbours together still scores 45/45 lenient. Never accept a
# localizer change on the lenient number alone.
```

## 1b) ANSWERED 2026-07-30 — the merge does NOT fix the 5 FN, but keep it anyway

Run on the RTX (GLM judge, conservative prompt, 29 cases; merge-ON 2.0 min,
merge-OFF 0.3 min from cache). Reports: `report_merge_on.md` / `report_merge_off.md`.

| | merge OFF | merge ON | Δ |
|---|---|---|---|
| frame F1 | 1.000 | 1.000 | — (saturated) |
| instance recall | 40/45 = 0.889 | 40/45 = 0.889 | **0** |
| strict IoU>=0.3 | 33/45 = 0.733 | 33/45 = 0.733 | **0** |
| kept boxes | 86 | 74 | −12 |
| FP boxes | 29 | 20 | **−9 (−31%)** |
| region precision | 0.663 | **0.730** | **+0.067** |

**Answer: no.** Recall did not move at lenient OR strict IoU, per-type identical.
Scope it honestly: merge-ON cost only **68 fresh calls**, not the ~559 estimated —
651 regions become 559 but 491 keep identical coordinates and came from cache. So
~14% of regions were re-judged and none of the 5 misses flipped. Conclusive for
this merge at these settings; not a general refutation of fragmentation. A more
aggressive merge cannot settle it (fill 0.25 chains into a full-frame box, strict
IoU 37/45 → 22/45).

**Keep the merge regardless** — it satisfies the pre-registered rule (region
precision above 0.663 with recall intact). It stays shipped as `MERGE_REGIONS = True`.
Per-case FP gains: gpt_07 4→1, gpt_02 4→2, real_f0053 3→2, real_f0100 1→0,
real_f0112 1→0, gpt_11 6→5.

**Validation worth noting:** merge-OFF reproduced the published `report.md`
numbers exactly (0.889 / 0.663 / 29 FP of 86), so this was a genuinely controlled
A/B and the reported figures are reproducible.

**Where the 5 FN actually are:** all 5 are on `real` frames and all are type
`object` (graffiti 6/6, damage 4/4, litter 2/2 are perfect). Localization has all
45. So they are VLM rejections on real footage specifically. It is not a crowding
or cap effect either: real_f0112 scores 4/4 with 77 raw regions — the busiest frame
in the set — while real_f0037 misses one with only 20. Next lever is neither the
merge nor MAX_REGIONS.

> **Runbook bug found while analysing this:** the block below copied only
> `report.md`, so the merge-OFF run silently overwrote `results.json` and the
> merge-ON per-case internals (`localizer.merged_away`, per-region detail) were
> lost. Copy `results.json` alongside each report — fixed below.

```bash
# A/B, GLM judge, conservative prompt. ~560 fresh calls ≈ 7 min at 0.7 s/call.

# ---- STEP 0, MANDATORY: run_benchmark.py has NO --model flag. It reads
# m.MODEL_NAME from the module, and that default is qwen3-vl. The baseline being
# compared against (report.md: object recall 0.889, region precision 0.663) is
# GLM's, so the judge MUST be switched or the A/B compares nothing.
sed -i 's|^MODEL_NAME = .*|MODEL_NAME = "haervwe/GLM-4.6V-Flash-9B:latest"|' \
    vlm_05_reference_diff.py
grep -n '^MODEL_NAME' vlm_05_reference_diff.py     # verify before spending GPU time
# PROMPT already ships as the conservative one — leave it alone. Editing it
# changes the cache fingerprint and turns the merge-OFF baseline into a full
# 651-call re-run.

# ---- merge ON (the experiment): ~559 fresh calls
python benchmark/run_benchmark.py
cp benchmark/report.md    benchmark/report_merge_on.md
cp benchmark/results.json benchmark/results_merge_on.json   # per-case internals

# ---- merge OFF (the baseline): 651 GLM verdicts are already in cache -> ~0 calls
sed -i 's/^MERGE_REGIONS  = True/MERGE_REGIONS  = False/' vlm_05_reference_diff.py
python benchmark/run_benchmark.py
cp benchmark/report.md    benchmark/report_merge_off.md
cp benchmark/results.json benchmark/results_merge_off.json

# ---- restore both constants
sed -i 's/^MERGE_REGIONS  = False/MERGE_REGIONS  = True/' vlm_05_reference_diff.py
sed -i 's|^MODEL_NAME = .*|MODEL_NAME = "qwen3-vl:8b-instruct"|' \
    vlm_05_reference_diff.py
```

Sanity check while it runs: the merge-OFF run should report **0 fresh calls**. If it
starts making calls, the fingerprint moved (prompt or model edited by mistake) and the
two halves are no longer comparable — stop and fix rather than let it finish.

Copy back to the laptop afterwards: `benchmark/report_merge_{on,off}.md`,
`benchmark/results.json` and `benchmark/cache.json`. **Copy the cache, and do not let
the copy-back overwrite `benchmark/README.md`, `ground_truth.json` or
`run_benchmark.py`** — that happened on 2026-07-13 and silently restored stale
versions.

Compare the OBJECT-level block, not the frame-level one — frame-level is already
1.000 on these 29 cases and cannot move. Keep the merge if instance recall rises
above 40/45 or region precision above 0.663 with recall intact; revert it if
recall drops. Either way the answer belongs in benchmark/README.md.

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
