# vlm_05 reference-diff - anomaly detection benchmark

**Status:** COMPLETE  
**Model:** `blaifa/InternVL3_5:8b` (Ollama)  
**Decision rule:** frame flagged if the VLM keeps ≥1 region (`filter` mode) after dropping person/"disappeared" labels and de-duplicating overlapping boxes.  
**Diff / region params:** DIFF_THRESHOLD=40, BLUR_RADIUS=3, MIN_AREA=500, MAX_AREA=400000, MAX_REGIONS=25.  
**Wall-clock:** 10.0 min.

## Prompt

```
You are shown the SAME spot in a tram twice, side by side:
- LEFT half  = the clean reference (normal, empty, undamaged tram).
- RIGHT half = the same spot now.

Report a new ANOMALY that is present on the RIGHT and absent on the LEFT. An
anomaly is one of:
- a forgotten object (bag, backpack, phone, wallet, bottle, package),
- litter left behind (can, bottle, wrapper, paper, food),
- graffiti: painted letters, a tag or a scribble on a surface,
- damage / vandalism: a torn or slashed seat, a broken part.

The object may be SMALL, partly hidden behind a seat or a pole, or lying flat on
the floor. Size does not matter: a phone or a bottle counts exactly as much as a
suitcase. Judge only this crop, on its own.

Answer NO (it is NOT an anomaly) if ANY of these hold:
- the RIGHT shows the SAME empty seat, floor, wall, pole or panel as the LEFT and
  only the brightness, shadow, reflection or colour differs,
- the difference is a person, or anything a person is wearing, holding or sitting
  on,
- it is a scratch, glare or reflection on metal or glass (these are NOT graffiti).

Reply with YES or NO, then name what appeared in 2-4 words.
```

## 1) Frame-level (binary: is the frame anomalous?)

- Cases: **68**  (TP=30, FP=0, TN=37, FN=1)
- **Accuracy** 0.985 · **Precision** 1.000 · **Recall** 0.968 · **Specificity** 1.000 · **F1** 0.984

| | predicted anomaly | predicted clean |
|---|---|---|
| **actual anomaly** | TP = 30 | FN = 1 |
| **actual clean**   | FP = 0 | TN = 37 |

## 2) Object-level (did we box each real anomaly?)

- Instances detected: **48 / 73** → **object recall 0.658** (strict IoU≥0.3: 41 / 73 = 0.562)
- False-positive regions (kept boxes matching no real anomaly): **5** of 57 kept → region precision 0.912
- Uncached VLM calls this run: 599, mean 0.8 s/call

| type | instances detected | recall |
|---|---|---|
| object | 31 / 53 | 0.58 |
| graffiti | 6 / 7 | 0.86 |
| damage | 5 / 6 | 0.83 |
| litter | 6 / 7 | 0.86 |

| source | cases | instances detected | FP regions |
|---|---|---|---|
| gpt | 11 | 19 / 20 | 3 |
| real | 54 | 23 / 46 | 2 |
| self | 2 | 0 / 0 | 0 |
| variant | 1 | 6 / 7 | 0 |

## Per-case results

| id | truth | frame | instances hit | FP boxes | VLM kept-labels |
|---|---|---|---|---|---|
| 39T_cam52_084637 | anomaly | **TP** | 1/3 | 0 | backpack. |
| 39T_cam52_085124 | anomaly | **TP** | 1/2 | 0 | black object. |
| 39T_cam53_083517 | anomaly | **TP** | 1/2 | 0 | yellow cloth. |
| 39T_cam53_084021 | anomaly | **TP** | 1/2 | 0 | yellow cloth. |
| 39T_cam53_084637 | anomaly | **TP** | 1/1 | 0 | green plant. |
| 39T_cam53_085124 | anomaly | **TP** | 1/1 | 0 | pink item. |
| 39T_cam54_084021 | anomaly | **TP** | 1/1 | 0 | litter left behind. |
| 39T_cam54_084637 | anomaly | **TP** | 2/2 | 0 | forgotten object., laptop |
| 39T_cam54_085124 | anomaly | **TP** | 2/3 | 1 | litter left behind., litter left behind., bottle. |
| 39T_cam55_083517 | anomaly | **TP** | 1/2 | 0 | forgotten object. |
| 39T_cam55_084021 | anomaly | **TP** | 2/2 | 0 | forgotten object., laptop. |
| 39T_cam55_084637 | anomaly | **TP** | 2/2 | 0 | forgotten object., laptop |
| 39T_cam55_085124 | anomaly | **TP** | 1/2 | 0 | green bottle. |
| gpt_01_suitcase | anomaly | **TP** | 1/1 | 0 | suitcase. |
| gpt_02_multi | anomaly | **TP** | 4/4 | 0 | backpack., backpack, bottle. |
| gpt_03_faint_tag | anomaly | **TP** | 1/1 | 0 | graffiti. The right half shows the letters "XRP" written on the wall, which is absent on the left half. |
| gpt_04_graffiti | anomaly | **TP** | 1/1 | 0 | graffiti., graffiti., graffiti. |
| gpt_05_slash | anomaly | **TP** | 1/1 | 0 | damage / vandalism. |
| gpt_07_multi | anomaly | **TP** | 4/4 | 1 | backpack, graffiti., damage / vandalism., bottle., bottle. |
| gpt_08_phone_tag | anomaly | **TP** | 2/2 | 0 | graffiti., phone. |
| gpt_09_litter | anomaly | **TP** | 1/1 | 0 | litter left behind., litter left behind., bottle., bottle., litter left behind. |
| gpt_10_litter | anomaly | **TP** | 1/1 | 0 | litter left behind |
| gpt_11_crowd | anomaly | **TP** | 3/4 | 2 | backpack., backpack., graffiti., bottle., graffiti |
| real_f0037 | anomaly | **TP** | 1/4 | 0 | backpack. |
| real_f0053 | anomaly | **TP** | 1/4 | 0 | backpack. |
| real_f0100 | anomaly | **TP** | 1/4 | 0 | backpack |
| real_f0112 | anomaly | **TP** | 1/4 | 0 | backpack. |
| real_f0205 | anomaly | **TP** | 1/2 | 0 | jacket. |
| real_f0219 | anomaly | **TP** | 1/2 | 1 | jacket., jacket. |
| variant_01 | anomaly | **TP** | 6/7 | 0 | backpack., bottle., damage / vandalism: torn seat cushion., graffiti., torn seat cover. |
| 39T_cam54_083517 | anomaly | **FN** | 0/1 | 0 | - |
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

Annotated images: `benchmark/runs/cli-latest/annotated/<id>.jpg` (blue = ground-truth boxes, green = correct detections, red = false-positive boxes). Raw results: `benchmark/runs/judge-internvl3-5-8b-balanced/results.json`.
