# vlm_05 reference-diff - anomaly detection benchmark

**Status:** COMPLETE  
**Model:** `haervwe/GLM-4.6V-Flash-9B:latest` (Ollama)  
**Decision rule:** frame flagged if the VLM keeps ≥1 region (`filter` mode) after dropping person/"disappeared" labels and de-duplicating overlapping boxes.  
**Diff / region params:** DIFF_THRESHOLD=40, BLUR_RADIUS=3, MIN_AREA=500, MAX_AREA=400000, MAX_REGIONS=25.  
**Wall-clock:** 8.0 min.

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

- Cases: **68**  (TP=30, FP=5, TN=32, FN=1)
- **Accuracy** 0.912 · **Precision** 0.857 · **Recall** 0.968 · **Specificity** 0.865 · **F1** 0.909

| | predicted anomaly | predicted clean |
|---|---|---|
| **actual anomaly** | TP = 30 | FN = 1 |
| **actual clean**   | FP = 5 | TN = 32 |

## 2) Object-level (did we box each real anomaly?)

- Instances detected: **56 / 73** → **object recall 0.767** (strict IoU≥0.3: 48 / 73 = 0.658)
- False-positive regions (kept boxes matching no real anomaly): **25** of 89 kept → region precision 0.719
- Uncached VLM calls this run: 654, mean 0.6 s/call

| type | instances detected | recall |
|---|---|---|
| object | 38 / 53 | 0.72 |
| graffiti | 6 / 7 | 0.86 |
| damage | 6 / 6 | 1.00 |
| litter | 6 / 7 | 0.86 |

| source | cases | instances detected | FP regions |
|---|---|---|---|
| gpt | 11 | 18 / 20 | 5 |
| real | 54 | 32 / 46 | 19 |
| self | 2 | 0 / 0 | 0 |
| variant | 1 | 6 / 7 | 1 |

## Per-case results

| id | truth | frame | instances hit | FP boxes | VLM kept-labels |
|---|---|---|---|---|---|
| 3333_cam52_084637\* | anomaly | **TP** | 1/3 | 1 | a purple bag, a torn seat |
| 3333_cam52_085124 | anomaly | **TP** | 1/2 | 4 | wallet, a forgotten object, a red object (bag), phone, handbag |
| 3333_cam53_083517 | anomaly | **TP** | 1/2 | 1 | yellow bag, yellow package |
| 3333_cam53_084021 | anomaly | **TP** | 2/2 | 1 | yellow bag, green object, a forgotten object (bag) |
| 3333_cam53_084637 | anomaly | **TP** | 1/1 | 0 | bottle |
| 3333_cam53_085124 | anomaly | **TP** | 1/1 | 0 | pink item |
| 3333_cam54_084021 | anomaly | **TP** | 1/1 | 0 | plastic bag |
| 3333_cam54_084637 | anomaly | **TP** | 2/2 | 0 | cloth on seat, laptop |
| 3333_cam54_085124 | anomaly | **TP** | 3/3 | 1 | a crumpled bag, a forgotten bag, bottle, laptop |
| 3333_cam55_083517 | anomaly | **TP** | 2/2 | 0 | a forgotten bag, a laptop |
| 3333_cam55_084021 | anomaly | **TP** | 2/2 | 1 | a forgotten bag, a forgotten object, scribbles |
| 3333_cam55_084637 | anomaly | **TP** | 2/2 | 2 | forgotten object, laptop, a forgotten object, bag |
| 3333_cam55_085124 | anomaly | **TP** | 2/2 | 0 | forgotten objects, forgotten objects (bottle and laptop) |
| gpt_01_suitcase | anomaly | **TP** | 1/1 | 0 | suitcase |
| gpt_02_multi | anomaly | **TP** | 2/4 | 1 | black backpack, bottle, backpack |
| gpt_03_faint_tag | anomaly | **TP** | 1/1 | 0 | scribble |
| gpt_04_graffiti | anomaly | **TP** | 1/1 | 1 | graffiti, graffiti, graffiti, blue graffiti, graffiti, green cord/line |
| gpt_05_slash | anomaly | **TP** | 1/1 | 0 | torn seat |
| gpt_07_multi | anomaly | **TP** | 4/4 | 1 | backpack and bottle, graffiti, torn seat, plastic bottle, a bottle and spill |
| gpt_08_phone_tag | anomaly | **TP** | 2/2 | 0 | graffiti, wallet |
| gpt_09_litter | anomaly | **TP** | 1/1 | 0 | cans and trash, three small rectangular objects, two cans, a can and papers, a small object (like a crumpled piece of paper) |
| gpt_10_litter | anomaly | **TP** | 1/1 | 0 | trash on floor |
| gpt_11_crowd | anomaly | **TP** | 4/4 | 2 | backpack and bottle, bottle, graffiti, torn seat, plastic bottle, graffiti |
| real_f0037 | anomaly | **TP** | 2/4 | 0 | backpack, backpack, black bag |
| real_f0053 | anomaly | **TP** | 1/4 | 0 | black backpack |
| real_f0100 | anomaly | **TP** | 2/4 | 1 | backpack, phone, phone |
| real_f0112 | anomaly | **TP** | 3/4 | 0 | black backpack, backpack, phone |
| real_f0205 | anomaly | **TP** | 1/2 | 0 | jacket on seat |
| real_f0219 | anomaly | **TP** | 2/2 | 0 | jacket on seat, backpack |
| variant_01 | anomaly | **TP** | 6/7 | 1 | backpack, bottle, torn seat, graffiti, a torn seat, a torn seat |
| 3333_cam54_083517 | anomaly | **FN** | 0/1 | 0 | - |
| 1760_cam13_t420 | clean | **FP** | - | 2 | vent cover, a white object on the floor |
| 1760_cam13_t570 | clean | **FP** | - | 1 | yellow text on right not left |
| 3333_cam54_ref_t120_clean | clean | **FP** | - | 2 | two small white objects (bottles), white object |
| neg_v4_f0016 | clean | **FP** | - | 1 | backpack |
| neg_v4_f0022 | clean | **FP** | - | 1 | small box on floor |
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
| 1760_cam13_t520 | clean | **TN** | - | 0 | - |
| 3333_cam52_ref_t120_clean | clean | **TN** | - | 0 | - |
| 3333_cam53_085954_clean | clean | **TN** | - | 0 | - |
| 3333_cam53_ref_t120_clean | clean | **TN** | - | 0 | - |
| 3333_cam54_085954_clean | clean | **TN** | - | 0 | - |
| 3333_cam55_085954_clean | clean | **TN** | - | 0 | - |
| 3333_cam55_ref_t120_clean | clean | **TN** | - | 0 | - |
| neg_gpt_06_clean | clean | **TN** | - | 0 | - |
| neg_real_ref_self | clean | **TN** | - | 0 | - |
| neg_v1_f0151 | clean | **TN** | - | 0 | - |
| neg_v1_f0181 | clean | **TN** | - | 0 | - |
| neg_v1_f0211 | clean | **TN** | - | 0 | - |
| neg_v1_f0241 | clean | **TN** | - | 0 | - |
| neg_v2_f0001_person | clean | **TN** | - | 0 | - |
| neg_v3_f0001 | clean | **TN** | - | 0 | - |
| neg_v4_f0004 | clean | **TN** | - | 0 | - |
| neg_variant_ref_self | clean | **TN** | - | 0 | - |

Annotated images: `benchmark/runs/cli-latest/annotated/<id>.jpg` (blue = ground-truth boxes, green = correct detections, red = false-positive boxes). Raw results: `benchmark/runs/judge-glm-46v-flash-9b-latest-lenient/results.json`.

---

\* **3333** is a placeholder, not the tram's real fleet number - the vehicle number of the 2026-08-11 capture is unknown. It was called 39T before, but 39T is the Škoda type, which tram 1760 shares.
