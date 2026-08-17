# vlm_05 reference-diff — anomaly detection benchmark

> **DISQUALIFIED — do not read the scores below as a model ranking.**
> minicpm-v4.6 ignores the "Reply with YES or NO" format on side-by-side
> crops (its replies lead with the object, so the parser scores every verdict
> NO → the all-FN table below), and — worse — its raw replies claim an object
> appeared on **198 of the 199 crops from CLEAN frames** ("bag visible on
> right, absent left." on empty seats, often the same sentence verbatim).
> Fixing the parser would only flip it to flagging 100 % of regions.
> It is unusable as a crop judge via Ollama; note that it DOES answer sensibly
> on whole frames (see the Task-1/2 rows in ARSI_results_EN.xlsx), so the
> failure is specific to the side-by-side crop protocol. Diagnosed 2026-07-13
> from `cache.json`; no GPU re-run needed.

**Status:** COMPLETE  
**Model:** `openbmb/minicpm-v4.6:latest` (Ollama)  
**Decision rule:** frame flagged if the VLM keeps ≥1 region (`filter` mode) after dropping person/"disappeared" labels and de-duplicating overlapping boxes.  
**Diff / region params:** DIFF_THRESHOLD=40, BLUR_RADIUS=3, MIN_AREA=500, MAX_AREA=400000, MAX_REGIONS=25.  
**Wall-clock:** 0.2 min (CPU-only Ollama).

## Prompt

```
You are shown the SAME spot in a tram twice, side by side:
- LEFT half  = the clean reference (normal, empty, undamaged tram).
- RIGHT half = the same spot now.

Report ONLY a clear new ANOMALY that is present on the RIGHT and absent on the
LEFT. An anomaly is one of:
- a forgotten object (bag, backpack, phone, wallet, bottle, package),
- litter left behind (can, bottle, wrapper, paper, food),
- graffiti: painted letters, a tag or a scribble on a surface,
- damage / vandalism: a torn or slashed seat, a broken part.

Answer NO (it is NOT an anomaly) if ANY of these hold:
- the RIGHT shows the SAME empty seat, floor, wall, pole or panel as the LEFT and
  only the brightness, shadow, reflection or colour differs,
- the difference is a person, or anything a person is wearing, holding or sitting
  on,
- it is a scratch, glare or reflection on metal or glass (these are NOT graffiti).

Be conservative: answer YES only if you can clearly SEE and NAME a specific new
object, marking or damage. If you are unsure, answer NO.

Reply with YES or NO, then name what appeared in 2-4 words.
```

## 1) Frame-level (binary: is the frame anomalous?)

- Cases: **29**  (TP=0, FP=0, TN=12, FN=17)
- **Accuracy** 0.414 · **Precision** 0.000 · **Recall** 0.000 · **Specificity** 1.000 · **F1** 0.000

| | predicted anomaly | predicted clean |
|---|---|---|
| **actual anomaly** | TP = 0 | FN = 17 |
| **actual clean**   | FP = 0 | TN = 12 |

## 2) Object-level (did we box each real anomaly?)

- Instances detected: **0 / 45** → **object recall 0.000** (strict IoU≥0.3: 0 / 45 = 0.000)
- False-positive regions (kept boxes matching no real anomaly): **0** of 0 kept → region precision 0.000
- All VLM verdicts served from cache (0 new calls).

| type | instances detected | recall |
|---|---|---|
| object | 0 / 33 | 0.00 |
| graffiti | 0 / 6 | 0.00 |
| damage | 0 / 4 | 0.00 |
| litter | 0 / 2 | 0.00 |

| source | cases | instances detected | FP regions |
|---|---|---|---|
| gpt | 11 | 0 / 20 | 0 |
| real | 15 | 0 / 21 | 0 |
| self | 2 | 0 / 0 | 0 |
| variant | 1 | 0 / 4 | 0 |

## Per-case results

| id | truth | frame | instances hit | FP boxes | VLM kept-labels |
|---|---|---|---|---|---|
| gpt_01_suitcase | anomaly | **FN** | 0/1 | 0 | — |
| gpt_02_multi | anomaly | **FN** | 0/4 | 0 | — |
| gpt_03_faint_tag | anomaly | **FN** | 0/1 | 0 | — |
| gpt_04_graffiti | anomaly | **FN** | 0/1 | 0 | — |
| gpt_05_slash | anomaly | **FN** | 0/1 | 0 | — |
| gpt_07_multi | anomaly | **FN** | 0/4 | 0 | — |
| gpt_08_phone_tag | anomaly | **FN** | 0/2 | 0 | — |
| gpt_09_litter | anomaly | **FN** | 0/1 | 0 | — |
| gpt_10_litter | anomaly | **FN** | 0/1 | 0 | — |
| gpt_11_crowd | anomaly | **FN** | 0/4 | 0 | — |
| real_f0037 | anomaly | **FN** | 0/4 | 0 | — |
| real_f0053 | anomaly | **FN** | 0/4 | 0 | — |
| real_f0100 | anomaly | **FN** | 0/4 | 0 | — |
| real_f0112 | anomaly | **FN** | 0/4 | 0 | — |
| real_f0205 | anomaly | **FN** | 0/2 | 0 | — |
| real_f0219 | anomaly | **FN** | 0/3 | 0 | — |
| variant_01 | anomaly | **FN** | 0/4 | 0 | — |
| neg_gpt_06_clean | clean | **TN** | — | 0 | — |
| neg_real_ref_self | clean | **TN** | — | 0 | — |
| neg_v1_f0151 | clean | **TN** | — | 0 | — |
| neg_v1_f0181 | clean | **TN** | — | 0 | — |
| neg_v1_f0211 | clean | **TN** | — | 0 | — |
| neg_v1_f0241 | clean | **TN** | — | 0 | — |
| neg_v2_f0001_person | clean | **TN** | — | 0 | — |
| neg_v3_f0001 | clean | **TN** | — | 0 | — |
| neg_v4_f0004 | clean | **TN** | — | 0 | — |
| neg_v4_f0016 | clean | **TN** | — | 0 | — |
| neg_v4_f0022 | clean | **TN** | — | 0 | — |
| neg_variant_ref_self | clean | **TN** | — | 0 | — |

Annotated images: `benchmark/annotated/<id>.jpg` (blue = ground-truth boxes, green = correct detections, red = false-positive boxes). Raw results: `benchmark/results.json`.
