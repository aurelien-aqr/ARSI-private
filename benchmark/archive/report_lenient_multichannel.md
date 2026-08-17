# vlm_05 reference-diff — anomaly detection benchmark

**Status:** COMPLETE  
**Model:** `qwen3-vl:8b-instruct` (Ollama)  
**Decision rule:** frame flagged if the VLM keeps ≥1 region (`filter` mode) after dropping person/"disappeared" labels and de-duplicating overlapping boxes.  
**Diff / region params:** DIFF_THRESHOLD=40, BLUR_RADIUS=3, MIN_AREA=500, MAX_AREA=400000, MAX_REGIONS=25.  
**Wall-clock:** 0.2 min.

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

- Cases: **29**  (TP=17, FP=5, TN=7, FN=0)
- **Accuracy** 0.828 · **Precision** 0.773 · **Recall** 1.000 · **Specificity** 0.583 · **F1** 0.872

| | predicted anomaly | predicted clean |
|---|---|---|
| **actual anomaly** | TP = 17 | FN = 0 |
| **actual clean**   | FP = 5 | TN = 7 |

## 2) Object-level (did we box each real anomaly?)

- Instances detected: **44 / 45** → **object recall 0.978** (strict IoU≥0.3: 37 / 45 = 0.822)
- False-positive regions (kept boxes matching no real anomaly): **55** of 118 kept → region precision 0.534
- All VLM verdicts served from cache (0 new calls).

| type | instances detected | recall |
|---|---|---|
| object | 32 / 33 | 0.97 |
| graffiti | 6 / 6 | 1.00 |
| damage | 4 / 4 | 1.00 |
| litter | 2 / 2 | 1.00 |

| source | cases | instances detected | FP regions |
|---|---|---|---|
| gpt | 11 | 20 / 20 | 28 |
| real | 15 | 20 / 21 | 21 |
| self | 2 | 0 / 0 | 0 |
| variant | 1 | 4 / 4 | 6 |

## Per-case results

| id | truth | frame | instances hit | FP boxes | VLM kept-labels |
|---|---|---|---|---|---|
| gpt_01_suitcase | anomaly | **TP** | 1/1 | 1 | black suitcase appeared., black suitcase appeared, suitcase appears., black suitcase appears., Black suitcase appears |
| gpt_02_multi | anomaly | **TP** | 4/4 | 6 | black backpack appeared, brown paper bag, phone and paper bag, bottle on seat, water bottle appeared, brown paper bag, black bag appears, water bottle appeared, black backpack strap, backpack appears, phone on surface |
| gpt_03_faint_tag | anomaly | **TP** | 1/1 | 0 | XRP graffiti tag |
| gpt_04_graffiti | anomaly | **TP** | 1/1 | 1 | graffiti on wall, graffiti on window, colorful abstract pattern, graffiti or tag |
| gpt_05_slash | anomaly | **TP** | 1/1 | 2 | a small white patch on the blue seat., damage to seat cushion., damage to window surface. |
| gpt_07_multi | anomaly | **TP** | 4/4 | 5 | bottle on floor, backpack and bottle, backpack and water bottle, graffiti tag appears, small metallic object, black bag on seat, torn seat cushion., graffiti tag appears, black backpack appears, damage to seat surface |
| gpt_08_phone_tag | anomaly | **TP** | 2/2 | 2 | graffiti tag appears, phone on seat, graffiti tag on right side, phone on seat |
| gpt_09_litter | anomaly | **TP** | 1/1 | 1 | a can appeared., battery on floor, SIM CARD ON FLOOR, small white object, small object on surface, graffiti tag on vent |
| gpt_10_litter | anomaly | **TP** | 1/1 | 1 | a can appeared., trash on floor, rolled-up banknote |
| gpt_11_crowd | anomaly | **TP** | 4/4 | 8 | backpack on seat, bottle on floor, backpack and bottle, graffiti on wall, backpack on seat, graffiti tag appears, bottle on floor, graffiti tag on wall, damage on seat cushion, dark stain on floor, graffiti tag on wall, torn blue seat cushion, graffiti tag on surface, black jacket appears, damage on seat surface |
| real_f0037 | anomaly | **TP** | 4/4 | 1 | backpack appeared, backpack appeared, phone on seat, wallet on seat, Dark object near railing |
| real_f0053 | anomaly | **TP** | 4/4 | 3 | backpack appeared, black backpack appeared., black bag appears, wallet on seat, phone on seat, wallet on seat, phone on seat |
| real_f0100 | anomaly | **TP** | 4/4 | 2 | backpack appears., backpack on floor, wallet on seat, phone on seat, phone on seat, phone on seat |
| real_f0112 | anomaly | **TP** | 4/4 | 2 | backpack on floor, black backpack appeared., backpack appears., phone appeared, phone on seat, phone on seat |
| real_f0205 | anomaly | **TP** | 2/2 | 0 | jacket on seat, backpack appears, backpack handle visible |
| real_f0219 | anomaly | **TP** | 2/3 | 1 | jacket draped over seat, jacket draped over seat, black backpack appeared., black bag appears |
| variant_01 | anomaly | **TP** | 4/4 | 6 | bottle on floor, backpack on seat, backpack on seat, graffiti tag "KLEP", tear in seat fabric, bottle on floor, bottle on railing, backpack on seat, bottle on floor, hole in seat fabric, damage on seat, backpack appears |
| neg_gpt_06_clean | clean | **FP** | — | 1 | dark stain on seat |
| neg_v1_f0181 | clean | **FP** | — | 1 | graffiti tag appears |
| neg_v3_f0001 | clean | **FP** | — | 5 | backpack appeared, fur on seat edge, animal fur appears., Backpack handle visible, graffiti on seat back |
| neg_v4_f0016 | clean | **FP** | — | 3 | black bag on seat, black bag on seat, graffiti tag appears. |
| neg_v4_f0022 | clean | **FP** | — | 3 | backpack on right side., backpack appears on right., graffiti on seat back |
| neg_real_ref_self | clean | **TN** | — | 0 | — |
| neg_v1_f0151 | clean | **TN** | — | 0 | — |
| neg_v1_f0211 | clean | **TN** | — | 0 | — |
| neg_v1_f0241 | clean | **TN** | — | 0 | — |
| neg_v2_f0001_person | clean | **TN** | — | 0 | — |
| neg_v4_f0004 | clean | **TN** | — | 0 | — |
| neg_variant_ref_self | clean | **TN** | — | 0 | — |

Annotated images: `benchmark/annotated/<id>.jpg` (blue = ground-truth boxes, green = correct detections, red = false-positive boxes). Raw results: `benchmark/results.json`.
