# vlm_05 reference-diff - anomaly detection benchmark

**Status:** COMPLETE  
**Model:** `haervwe/GLM-4.6V-Flash-9B:latest` (Ollama)  
**Decision rule:** frame flagged if the VLM keeps ≥1 region (`filter` mode) after dropping person/"disappeared" labels and de-duplicating overlapping boxes.  
**Diff / region params:** DIFF_THRESHOLD=40, BLUR_RADIUS=3, MIN_AREA=500, MAX_AREA=400000, MAX_REGIONS=25.  
**Wall-clock:** 13.4 min.

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

- Cases: **68**  (TP=30, FP=0, TN=37, FN=1)
- **Accuracy** 0.985 · **Precision** 1.000 · **Recall** 0.968 · **Specificity** 1.000 · **F1** 0.984

| | predicted anomaly | predicted clean |
|---|---|---|
| **actual anomaly** | TP = 30 | FN = 1 |
| **actual clean**   | FP = 0 | TN = 37 |

## 2) Object-level (did we box each real anomaly?)

- Instances detected: **55 / 73** → **object recall 0.753** (strict IoU≥0.3: 46 / 73 = 0.630)
- False-positive regions (kept boxes matching no real anomaly): **7** of 67 kept → region precision 0.896
- Uncached VLM calls this run: 1228, mean 0.6 s/call

| type | instances detected | recall |
|---|---|---|
| object | 37 / 53 | 0.70 |
| graffiti | 6 / 7 | 0.86 |
| damage | 6 / 6 | 1.00 |
| litter | 6 / 7 | 0.86 |

| source | cases | instances detected | FP regions |
|---|---|---|---|
| gpt | 11 | 20 / 20 | 4 |
| real | 54 | 29 / 46 | 3 |
| self | 2 | 0 / 0 | 0 |
| variant | 1 | 6 / 7 | 0 |

## Per-case results

| id | truth | frame | instances hit | FP boxes | VLM kept-labels |
|---|---|---|---|---|---|
| 39T_cam52_084637 | anomaly | **TP** | 1/3 | 0 | a purple bag. |
| 39T_cam52_085124 | anomaly | **TP** | 1/2 | 2 | wallet, wallet, phone |
| 39T_cam53_083517 | anomaly | **TP** | 1/2 | 0 | yellow bag |
| 39T_cam53_084021 | anomaly | **TP** | 1/2 | 0 | yellow bag |
| 39T_cam53_084637 | anomaly | **TP** | 1/1 | 0 | bottle |
| 39T_cam53_085124 | anomaly | **TP** | 1/1 | 0 | pink item (cloth/hat) |
| 39T_cam54_084021 | anomaly | **TP** | 1/1 | 0 | plastic bag |
| 39T_cam54_084637 | anomaly | **TP** | 2/2 | 0 | cloth on seat, laptop |
| 39T_cam54_085124 | anomaly | **TP** | 3/3 | 1 | crumpled bag, a plastic bag with items, green bottle, laptop |
| 39T_cam55_083517 | anomaly | **TP** | 2/2 | 0 | a crumpled bag, laptop |
| 39T_cam55_084021 | anomaly | **TP** | 2/2 | 0 | plastic bag, a book |
| 39T_cam55_084637 | anomaly | **TP** | 2/2 | 0 | cloth on seat, laptop |
| 39T_cam55_085124 | anomaly | **TP** | 2/2 | 0 | bottle, a green bottle |
| gpt_01_suitcase | anomaly | **TP** | 1/1 | 0 | black suitcase |
| gpt_02_multi | anomaly | **TP** | 4/4 | 1 | backpack, black backpack, plastic bottle, black backpack |
| gpt_03_faint_tag | anomaly | **TP** | 1/1 | 0 | "graffiti 'XRP' on panel" |
| gpt_04_graffiti | anomaly | **TP** | 1/1 | 0 | colorful graffiti on the wall, graffiti, graffiti, purple graffiti scribble |
| gpt_05_slash | anomaly | **TP** | 1/1 | 0 | torn seat cushion |
| gpt_07_multi | anomaly | **TP** | 4/4 | 1 | black backpack, graffiti "HOB88", torn seat cushion, plastic bottle, a plastic bottle |
| gpt_08_phone_tag | anomaly | **TP** | 2/2 | 0 | graffiti, wallet |
| gpt_09_litter | anomaly | **TP** | 1/1 | 0 | litter (cans and paper), three small rectangular objects (litter)., two cans, a can and some wrappers, paper wrapper |
| gpt_10_litter | anomaly | **TP** | 1/1 | 0 | litter (can, papers) |
| gpt_11_crowd | anomaly | **TP** | 4/4 | 2 | black backpack, plastic bottle, graffiti "HOBBO", torn seat cushion, plastic bottle, graffiti "Hax" |
| real_f0037 | anomaly | **TP** | 1/4 | 0 | backpack |
| real_f0053 | anomaly | **TP** | 1/4 | 0 | black backpack |
| real_f0100 | anomaly | **TP** | 2/4 | 0 | backpack, phone |
| real_f0112 | anomaly | **TP** | 2/4 | 0 | black backpack, phone |
| real_f0205 | anomaly | **TP** | 1/2 | 0 | jacket on seat |
| real_f0219 | anomaly | **TP** | 2/2 | 0 | jacket on seat, backpack |
| variant_01 | anomaly | **TP** | 6/7 | 0 | backpack, bottle, torn seat, graffiti "KEPP", torn seat |
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

Annotated images: `benchmark/runs/cli-latest/annotated/<id>.jpg` (blue = ground-truth boxes, green = correct detections, red = false-positive boxes). Raw results: `benchmark/runs/loc-dino4008-all/results.json`.
