# vlm_05 reference-diff — anomaly detection benchmark

**Status:** COMPLETE  
**Model:** `blaifa/InternVL3_5:8b` (Ollama)  
**Decision rule:** frame flagged if the VLM keeps ≥1 region (`filter` mode) after dropping person/"disappeared" labels and de-duplicating overlapping boxes.  
**Diff / region params:** DIFF_THRESHOLD=40, BLUR_RADIUS=3, MIN_AREA=500, MAX_AREA=400000, MAX_REGIONS=25.  
**Wall-clock:** 0.2 min.

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

- Cases: **29**  (TP=16, FP=0, TN=12, FN=1)
- **Accuracy** 0.966 · **Precision** 1.000 · **Recall** 0.941 · **Specificity** 1.000 · **F1** 0.970

| | predicted anomaly | predicted clean |
|---|---|---|
| **actual anomaly** | TP = 16 | FN = 1 |
| **actual clean**   | FP = 0 | TN = 12 |

## 2) Object-level (did we box each real anomaly?)

- Instances detected: **33 / 45** → **object recall 0.733** (strict IoU≥0.3: 26 / 45 = 0.578)
- False-positive regions (kept boxes matching no real anomaly): **13** of 54 kept → region precision 0.759
- All VLM verdicts served from cache (0 new calls).

| type | instances detected | recall |
|---|---|---|
| object | 21 / 33 | 0.64 |
| graffiti | 6 / 6 | 1.00 |
| damage | 4 / 4 | 1.00 |
| litter | 2 / 2 | 1.00 |

| source | cases | instances detected | FP regions |
|---|---|---|---|
| gpt | 11 | 20 / 20 | 9 |
| real | 15 | 9 / 21 | 0 |
| self | 2 | 0 / 0 | 0 |
| variant | 1 | 4 / 4 | 4 |

## Per-case results

| id | truth | frame | instances hit | FP boxes | VLM kept-labels |
|---|---|---|---|---|---|
| gpt_01_suitcase | anomaly | **TP** | 1/1 | 0 | suitcase, suitcase |
| gpt_02_multi | anomaly | **TP** | 4/4 | 0 | backpack., paper bag., smartphone, bottle., brown paper bag. |
| gpt_03_faint_tag | anomaly | **TP** | 1/1 | 0 | graffiti. |
| gpt_04_graffiti | anomaly | **TP** | 1/1 | 0 | graffiti on wall. |
| gpt_05_slash | anomaly | **TP** | 1/1 | 1 | damaged seat., hole in seat. |
| gpt_07_multi | anomaly | **TP** | 4/4 | 1 | bottle on floor., backpack., graffiti on wall., graffiti., torn seat cover. |
| gpt_08_phone_tag | anomaly | **TP** | 2/2 | 1 | graffiti., phone., graffiti appears on the right side of the image. The graffiti is visible on the metal surface near the bottom right corner of the tram's interior panel. This marking differs from the clean and undamaged state seen on the left half of the image. |
| gpt_09_litter | anomaly | **TP** | 1/1 | 0 | bottle., bottle., object on floor. |
| gpt_10_litter | anomaly | **TP** | 1/1 | 1 | bottle on floor., litter left behind. |
| gpt_11_crowd | anomaly | **TP** | 4/4 | 5 | backpack., backpack., bottle on floor., graffiti, bottle on floor., graffiti., graffiti on pole., graffiti., graffiti., torn seat cover. |
| real_f0037 | anomaly | **TP** | 2/4 | 0 | backpack., wallet. |
| real_f0053 | anomaly | **TP** | 2/4 | 0 | backpack., wallet. |
| real_f0100 | anomaly | **TP** | 2/4 | 0 | backpack., wallet. |
| real_f0112 | anomaly | **TP** | 2/4 | 0 | backpack., backpack. |
| real_f0219 | anomaly | **TP** | 1/3 | 0 | jacket, jacket. |
| variant_01 | anomaly | **TP** | 4/4 | 4 | bottle on floor., backpack., backpack., graffiti appears on the right half of the image. The graffiti reads "Kepp" and is present on the wall where there was no such marking on the left half., torn seat cover., bottle on floor., bottle on right., backpack., bottle, backpack. |
| real_f0205 | anomaly | **FN** | 0/2 | 0 | — |
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
