# vlm_05 reference-diff - anomaly detection benchmark

**Status:** COMPLETE  
**Model:** `blaifa/InternVL3_5:8b` (Ollama)  
**Decision rule:** frame flagged if the VLM keeps ≥1 region (`filter` mode) after dropping person/"disappeared" labels and de-duplicating overlapping boxes.  
**Diff / region params:** DIFF_THRESHOLD=40, BLUR_RADIUS=3, MIN_AREA=500, MAX_AREA=400000, MAX_REGIONS=25.  
**Wall-clock:** 10.4 min.

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

- Cases: **68**  (TP=25, FP=0, TN=37, FN=6)
- **Accuracy** 0.912 · **Precision** 1.000 · **Recall** 0.806 · **Specificity** 1.000 · **F1** 0.893

| | predicted anomaly | predicted clean |
|---|---|---|
| **actual anomaly** | TP = 25 | FN = 6 |
| **actual clean**   | FP = 0 | TN = 37 |

## 2) Object-level (did we box each real anomaly?)

- Instances detected: **39 / 73** → **object recall 0.534** (strict IoU≥0.3: 34 / 73 = 0.466)
- False-positive regions (kept boxes matching no real anomaly): **3** of 45 kept → region precision 0.933
- Uncached VLM calls this run: 654, mean 0.8 s/call

| type | instances detected | recall |
|---|---|---|
| object | 24 / 53 | 0.45 |
| graffiti | 6 / 7 | 0.86 |
| damage | 4 / 6 | 0.67 |
| litter | 5 / 7 | 0.71 |

| source | cases | instances detected | FP regions |
|---|---|---|---|
| gpt | 11 | 18 / 20 | 2 |
| real | 54 | 16 / 46 | 1 |
| self | 2 | 0 / 0 | 0 |
| variant | 1 | 5 / 7 | 0 |

## Per-case results

| id | truth | frame | instances hit | FP boxes | VLM kept-labels |
|---|---|---|---|---|---|
| 39T_cam52_084637 | anomaly | **TP** | 1/3 | 0 | backpack. |
| 39T_cam53_083517 | anomaly | **TP** | 1/2 | 0 | yellow cloth. |
| 39T_cam53_084021 | anomaly | **TP** | 1/2 | 0 | yellow cloth. |
| 39T_cam53_084637 | anomaly | **TP** | 1/1 | 0 | green plant. |
| 39T_cam54_084021 | anomaly | **TP** | 1/1 | 0 | litter left behind. |
| 39T_cam54_084637 | anomaly | **TP** | 1/2 | 0 | bag |
| 39T_cam54_085124 | anomaly | **TP** | 2/3 | 1 | litter left behind., trash bag., bottle on seat. |
| 39T_cam55_083517 | anomaly | **TP** | 1/2 | 0 | bag on floor. |
| 39T_cam55_084021 | anomaly | **TP** | 1/2 | 0 | bag |
| 39T_cam55_084637 | anomaly | **TP** | 1/2 | 0 | bag on seat. |
| gpt_01_suitcase | anomaly | **TP** | 1/1 | 0 | suitcase |
| gpt_02_multi | anomaly | **TP** | 4/4 | 0 | backpack., backpack., bottle. |
| gpt_03_faint_tag | anomaly | **TP** | 1/1 | 0 | graffiti. |
| gpt_04_graffiti | anomaly | **TP** | 1/1 | 0 | graffiti on wall., graffiti appears on the right half of the image. The graffiti consists of colorful painted letters and scribbles on the surface where the tram panel is located. This is clearly different from the left half, which shows a clean reference without any such markings., graffiti appears on the right half of the image. The graffiti includes painted letters and scribbles on the wall behind the seat. This is absent on the left half, which shows a clean reference without any markings. |
| gpt_05_slash | anomaly | **TP** | 1/1 | 0 | damaged seat cushion. |
| gpt_07_multi | anomaly | **TP** | 4/4 | 0 | backpack., graffiti appears on the right side of the image. The graffiti reads "HOBGOBLIN" written on the wall near the window. This is absent on the left half, which shows a clean reference without any markings or scribbles., torn seat cover., bottle. |
| gpt_08_phone_tag | anomaly | **TP** | 2/2 | 0 | graffiti., phone |
| gpt_09_litter | anomaly | **TP** | 1/1 | 0 | cans, two cans., bottle, litter left behind. |
| gpt_11_crowd | anomaly | **TP** | 3/4 | 2 | backpack., backpack., graffiti appears on the right half of the image. The graffiti reads "HOP8" on the wall near the window., bottle., graffiti. |
| real_f0037 | anomaly | **TP** | 1/4 | 0 | backpack. |
| real_f0053 | anomaly | **TP** | 1/4 | 0 | backpack. |
| real_f0100 | anomaly | **TP** | 1/4 | 0 | backpack. |
| real_f0112 | anomaly | **TP** | 1/4 | 0 | backpack. |
| real_f0219 | anomaly | **TP** | 1/2 | 0 | jacket. |
| variant_01 | anomaly | **TP** | 5/7 | 0 | backpack., bottle on floor., torn seat cushion., graffiti appears on the right half of the image. The graffiti is visible as black markings on the white surface above the yellow object. This is absent on the left half, which shows a clean reference without any such markings. |
| 39T_cam52_085124 | anomaly | **FN** | 0/2 | 0 | - |
| 39T_cam53_085124 | anomaly | **FN** | 0/1 | 0 | - |
| 39T_cam54_083517 | anomaly | **FN** | 0/1 | 0 | - |
| 39T_cam55_085124 | anomaly | **FN** | 0/2 | 0 | - |
| gpt_10_litter | anomaly | **FN** | 0/1 | 0 | - |
| real_f0205 | anomaly | **FN** | 0/2 | 0 | - |
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

Annotated images: `benchmark/runs/cli-latest/annotated/<id>.jpg` (blue = ground-truth boxes, green = correct detections, red = false-positive boxes). Raw results: `benchmark/runs/judge-internvl3-5-8b-conservative/results.json`.
