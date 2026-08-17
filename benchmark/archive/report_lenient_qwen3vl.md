# vlm_05 reference-diff — anomaly detection benchmark

**Status:** COMPLETE  
**Model:** `qwen3-vl:8b-instruct` (Ollama)  
**Decision rule:** frame flagged if the VLM keeps ≥1 region (`filter` mode) after dropping person/"disappeared" labels and de-duplicating overlapping boxes.  
**Diff / region params:** DIFF_THRESHOLD=40, BLUR_RADIUS=3, MIN_AREA=500, MAX_AREA=400000, MAX_REGIONS=25.  
**Wall-clock:** 0.1 min (CPU-only Ollama).

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

- Cases: **24**  (TP=16, FP=1, TN=6, FN=1)
- **Accuracy** 0.917 · **Precision** 0.941 · **Recall** 0.941 · **Specificity** 0.857 · **F1** 0.941

| | predicted anomaly | predicted clean |
|---|---|---|
| **actual anomaly** | TP = 16 | FN = 1 |
| **actual clean**   | FP = 1 | TN = 6 |

## 2) Object-level (did we box each real anomaly?)

- Instances detected: **41 / 45** → **object recall 0.911**
- False-positive regions (kept boxes matching no real anomaly): **20** of 78 kept → region precision 0.744

| type | instances detected | recall |
|---|---|---|
| object | 31 / 33 | 0.94 |
| graffiti | 4 / 6 | 0.67 |
| damage | 4 / 4 | 1.00 |
| litter | 2 / 2 | 1.00 |

## Per-case results

| id | truth | frame | instances hit | FP boxes | VLM kept-labels |
|---|---|---|---|---|---|
| gpt_01_suitcase | anomaly | **TP** | 1/1 | 0 | black suitcase appeared., black suitcase appeared, suitcase appears., black suitcase appears. |
| gpt_02_multi | anomaly | **TP** | 4/4 | 4 | black backpack appeared, phone and paper bag, water bottle appeared, brown paper bag, black bag appears, black backpack strap, backpack appears, phone on surface |
| gpt_04_graffiti | anomaly | **TP** | 1/1 | 1 | graffiti on wall, graffiti on window, colorful abstract pattern, graffiti or tag |
| gpt_05_slash | anomaly | **TP** | 1/1 | 0 | damage to seat cushion. |
| gpt_07_multi | anomaly | **TP** | 3/4 | 2 | backpack and bottle, graffiti tag appears, black bag on seat, graffiti tag appears, black backpack appears, damage to seat surface |
| gpt_08_phone_tag | anomaly | **TP** | 1/2 | 1 | phone on seat, phone on seat |
| gpt_09_litter | anomaly | **TP** | 1/1 | 1 | battery on floor, SIM CARD ON FLOOR, small white object, small object on surface, graffiti tag on vent |
| gpt_10_litter | anomaly | **TP** | 1/1 | 1 | a can appeared., trash on floor, rolled-up banknote |
| gpt_11_crowd | anomaly | **TP** | 3/4 | 5 | backpack on seat, backpack and bottle, graffiti on wall, graffiti tag appears, bottle on floor, damage on seat cushion, graffiti tag on wall, torn blue seat cushion, graffiti tag on surface, black jacket appears, damage on seat surface |
| real_f0037 | anomaly | **TP** | 4/4 | 0 | backpack appeared, backpack appeared, phone on seat, wallet on seat |
| real_f0053 | anomaly | **TP** | 4/4 | 1 | backpack appeared, black backpack appeared., wallet on seat, phone on seat, wallet on seat |
| real_f0100 | anomaly | **TP** | 4/4 | 1 | backpack appears., backpack on floor, wallet on seat, phone on seat, phone on seat |
| real_f0112 | anomaly | **TP** | 4/4 | 1 | backpack on floor, black backpack appeared., backpack appears., phone appeared, phone on seat |
| real_f0205 | anomaly | **TP** | 2/2 | 0 | jacket on seat, backpack appears, backpack handle visible |
| real_f0219 | anomaly | **TP** | 3/3 | 1 | jacket draped over seat, jacket draped over seat, black backpack appeared., black bag appears, phone appears on right. |
| variant_01 | anomaly | **TP** | 4/4 | 0 | backpack on seat, graffiti tag "KLEP", tear in seat fabric, backpack on seat, bottle on floor, backpack appears |
| gpt_03_faint_tag | anomaly | **FN** | 0/1 | 0 | — |
| neg_v1_f0181 | clean | **FP** | — | 1 | graffiti tag appears |
| neg_gpt_06_clean | clean | **TN** | — | 0 | — |
| neg_real_ref_self | clean | **TN** | — | 0 | — |
| neg_v1_f0151 | clean | **TN** | — | 0 | — |
| neg_v1_f0211 | clean | **TN** | — | 0 | — |
| neg_v1_f0241 | clean | **TN** | — | 0 | — |
| neg_variant_ref_self | clean | **TN** | — | 0 | — |

Annotated images: `benchmark/annotated/<id>.jpg` (blue = ground-truth boxes, green = correct detections, red = false-positive boxes). Raw results: `benchmark/results.json`.
