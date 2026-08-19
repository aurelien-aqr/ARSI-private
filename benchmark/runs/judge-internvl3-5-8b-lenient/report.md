# vlm_05 reference-diff - anomaly detection benchmark

**Status:** COMPLETE  
**Model:** `blaifa/InternVL3_5:8b` (Ollama)  
**Decision rule:** frame flagged if the VLM keeps ≥1 region (`filter` mode) after dropping person/"disappeared" labels and de-duplicating overlapping boxes.  
**Diff / region params:** DIFF_THRESHOLD=40, BLUR_RADIUS=3, MIN_AREA=500, MAX_AREA=400000, MAX_REGIONS=25.  
**Wall-clock:** 10.6 min.

## Prompt

```
You are shown the SAME spot in a tram twice, side by side:
- LEFT half  = the clean reference (normal, undamaged tram).
- RIGHT half = the same spot now.
Did an ANOMALY appear on the right that is NOT on the left? An anomaly is any of:
- a forgotten object (bag, backpack, phone, wallet, bottle, package),
- graffiti or a tag / scribble drawn on a surface,
- damage or vandalism (a torn or slashed seat, a broken part).
Answer NO if the only difference is a person or body part, a shadow, a
reflection, or a lighting/exposure change.
Reply with YES or NO, then name what appeared in 2-4 words.
```

## 1) Frame-level (binary: is the frame anomalous?)

- Cases: **68**  (TP=27, FP=0, TN=37, FN=4)
- **Accuracy** 0.941 · **Precision** 1.000 · **Recall** 0.871 · **Specificity** 1.000 · **F1** 0.931

| | predicted anomaly | predicted clean |
|---|---|---|
| **actual anomaly** | TP = 27 | FN = 4 |
| **actual clean**   | FP = 0 | TN = 37 |

## 2) Object-level (did we box each real anomaly?)

- Instances detected: **43 / 73** → **object recall 0.589** (strict IoU≥0.3: 36 / 73 = 0.493)
- False-positive regions (kept boxes matching no real anomaly): **4** of 51 kept → region precision 0.922
- Uncached VLM calls this run: 654, mean 0.8 s/call

| type | instances detected | recall |
|---|---|---|
| object | 26 / 53 | 0.49 |
| graffiti | 6 / 7 | 0.86 |
| damage | 6 / 6 | 1.00 |
| litter | 5 / 7 | 0.71 |

| source | cases | instances detected | FP regions |
|---|---|---|---|
| gpt | 11 | 18 / 20 | 3 |
| real | 54 | 19 / 46 | 1 |
| self | 2 | 0 / 0 | 0 |
| variant | 1 | 6 / 7 | 0 |

## Per-case results

| id | truth | frame | instances hit | FP boxes | VLM kept-labels |
|---|---|---|---|---|---|
| 39T_cam52_084637 | anomaly | **TP** | 1/3 | 0 | backpack. |
| 39T_cam53_083517 | anomaly | **TP** | 1/2 | 0 | yellow cloth. |
| 39T_cam53_084021 | anomaly | **TP** | 1/2 | 0 | yellow cloth. |
| 39T_cam53_084637 | anomaly | **TP** | 1/1 | 0 | green object. |
| 39T_cam53_085124 | anomaly | **TP** | 1/1 | 0 | pink item. |
| 39T_cam54_084021 | anomaly | **TP** | 1/1 | 0 | forgotten object. |
| 39T_cam54_084637 | anomaly | **TP** | 2/2 | 0 | forgotten object., laptop. |
| 39T_cam54_085124 | anomaly | **TP** | 2/3 | 1 | forgotten object., forgotten object., bottle on seat. |
| 39T_cam55_083517 | anomaly | **TP** | 1/2 | 0 | forgotten object. |
| 39T_cam55_084021 | anomaly | **TP** | 1/2 | 0 | forgotten object. |
| 39T_cam55_084637 | anomaly | **TP** | 1/2 | 0 | forgotten object. |
| gpt_01_suitcase | anomaly | **TP** | 1/1 | 0 | suitcase. |
| gpt_02_multi | anomaly | **TP** | 4/4 | 0 | backpack and paper bag., backpack., bottle. |
| gpt_03_faint_tag | anomaly | **TP** | 1/1 | 0 | graffiti or tag. |
| gpt_04_graffiti | anomaly | **TP** | 1/1 | 0 | graffiti., graffiti., graffiti appears on the right half of the image. The graffiti includes scribbles and tags that are not present on the left half. |
| gpt_05_slash | anomaly | **TP** | 1/1 | 0 | damage on seat. |
| gpt_07_multi | anomaly | **TP** | 4/4 | 1 | backpack and bottle., graffiti., damage on seat., bottle., bottle on floor. |
| gpt_08_phone_tag | anomaly | **TP** | 2/2 | 0 | graffiti., phone. |
| gpt_09_litter | anomaly | **TP** | 1/1 | 0 | two cans appear on the right half that are not present on the left half., graffiti or tag., bottle(s), bottle and pieces., damage or vandalism. |
| gpt_11_crowd | anomaly | **TP** | 3/4 | 2 | backpack and bottle., graffiti., graffiti on seat., bottle., graffiti |
| real_f0037 | anomaly | **TP** | 1/4 | 0 | backpack. |
| real_f0053 | anomaly | **TP** | 1/4 | 0 | backpack. |
| real_f0100 | anomaly | **TP** | 1/4 | 0 | backpack. |
| real_f0112 | anomaly | **TP** | 1/4 | 0 | backpack. |
| real_f0205 | anomaly | **TP** | 1/2 | 0 | jacket. |
| real_f0219 | anomaly | **TP** | 1/2 | 0 | jacket on seat. |
| variant_01 | anomaly | **TP** | 6/7 | 0 | backpack., bottle on floor., damage occurred on the right half where there is a torn seat cushion., graffiti., graffiti on seat. |
| 39T_cam52_085124 | anomaly | **FN** | 0/2 | 0 | - |
| 39T_cam54_083517 | anomaly | **FN** | 0/1 | 0 | - |
| 39T_cam55_085124 | anomaly | **FN** | 0/2 | 0 | - |
| gpt_10_litter | anomaly | **FN** | 0/1 | 0 | - |
| 1760_cam04_t120 | clean | **TN** | - | 0 | - |
| 1760_cam04_t220 | clean | **TN** | - | 0 | - |
| 1760_cam04_t320 | clean | **TN** | - | 0 | - |
| 1760_cam04_t420 | clean | **TN** | - | 0 | - |
| 1760_cam04_t520 | clean | **TN** | - | 0 | - |
| 1760_cam04_t570 | clean | **TN** | - | 0 | - |
| 1760_cam06_t120 | clean | **TN** | - | 0 | - |
| 1760_cam06_t220 | clean | **TN** | - | 0 | - |
| 1760_cam06_t320 | clean | **TN** | - | 0 | - |
| 1760_cam06_t420 | clean | **TN** | - | 0 | - |
| 1760_cam06_t520 | clean | **TN** | - | 0 | - |
| 1760_cam06_t570 | clean | **TN** | - | 0 | - |
| 1760_cam13_t120 | clean | **TN** | - | 0 | - |
| 1760_cam13_t220 | clean | **TN** | - | 0 | - |
| 1760_cam13_t320 | clean | **TN** | - | 0 | - |
| 1760_cam13_t420 | clean | **TN** | - | 0 | - |
| 1760_cam13_t520 | clean | **TN** | - | 0 | - |
| 1760_cam13_t570 | clean | **TN** | - | 0 | - |
| 39T_cam52_ref_t120_clean | clean | **TN** | - | 0 | - |
| 39T_cam53_085954_clean | clean | **TN** | - | 0 | - |
| 39T_cam53_ref_t120_clean | clean | **TN** | - | 0 | - |
| 39T_cam54_085954_clean | clean | **TN** | - | 0 | - |
| 39T_cam54_ref_t120_clean | clean | **TN** | - | 0 | - |
| 39T_cam55_085954_clean | clean | **TN** | - | 0 | - |
| 39T_cam55_ref_t120_clean | clean | **TN** | - | 0 | - |
| neg_gpt_06_clean | clean | **TN** | - | 0 | - |
| neg_real_ref_self | clean | **TN** | - | 0 | - |
| neg_v1_f0151 | clean | **TN** | - | 0 | - |
| neg_v1_f0181 | clean | **TN** | - | 0 | - |
| neg_v1_f0211 | clean | **TN** | - | 0 | - |
| neg_v1_f0241 | clean | **TN** | - | 0 | - |
| neg_v2_f0001_person | clean | **TN** | - | 0 | - |
| neg_v3_f0001 | clean | **TN** | - | 0 | - |
| neg_v4_f0004 | clean | **TN** | - | 0 | - |
| neg_v4_f0016 | clean | **TN** | - | 0 | - |
| neg_v4_f0022 | clean | **TN** | - | 0 | - |
| neg_variant_ref_self | clean | **TN** | - | 0 | - |

Annotated images: `benchmark/runs/cli-latest/annotated/<id>.jpg` (blue = ground-truth boxes, green = correct detections, red = false-positive boxes). Raw results: `benchmark/runs/judge-internvl3-5-8b-lenient/results.json`.
