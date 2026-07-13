# vlm_05 reference-diff — anomaly detection benchmark

**Status:** COMPLETE  
**Model:** `qwen3.5:9b` (Ollama)  
**Decision rule:** frame flagged if the VLM keeps ≥1 region (`filter` mode) after dropping person/"disappeared" labels and de-duplicating overlapping boxes.  
**Diff / region params:** DIFF_THRESHOLD=40, BLUR_RADIUS=3, MIN_AREA=500, MAX_AREA=400000, MAX_REGIONS=25.  
**Wall-clock:** 0.3 min (cache-only re-score with the corrected gpt_03 GT box; original GPU run 2026-07-13: 6.6 min, 651 fresh calls, mean 0.6 s/call).

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
- False-positive regions (kept boxes matching no real anomaly): **65** of 129 kept → region precision 0.496
- All VLM verdicts served from cache (0 new calls).

| type | instances detected | recall |
|---|---|---|
| object | 32 / 33 | 0.97 |
| graffiti | 6 / 6 | 1.00 |
| damage | 4 / 4 | 1.00 |
| litter | 2 / 2 | 1.00 |

| source | cases | instances detected | FP regions |
|---|---|---|---|
| gpt | 11 | 20 / 20 | 34 |
| real | 15 | 20 / 21 | 25 |
| self | 2 | 0 / 0 | 0 |
| variant | 1 | 4 / 4 | 6 |

## Per-case results

| id | truth | frame | instances hit | FP boxes | VLM kept-labels |
|---|---|---|---|---|---|
| gpt_01_suitcase | anomaly | **TP** | 1/1 | 1 | Black suitcase on floor., Black suitcase appeared., Suitcase appeared on the right., Suitcase on floor., Damaged suitcase corner., Suitcase on right side. |
| gpt_02_multi | anomaly | **TP** | 4/4 | 7 | black backpack on seat, Forgotten package on floor., Brown paper bag appeared., Forgotten phone and bag., Plastic bottle on seat., Plastic bottle on seat., Brown paper bag., Black bag on seat., Plastic bottle on seat., Black bag on seat., Bag on seat., Phone on seat. |
| gpt_03_faint_tag | anomaly | **TP** | 1/1 | 0 | Graffiti tag "XRP" on panel. |
| gpt_04_graffiti | anomaly | **TP** | 1/1 | 2 | graffiti on the wall, Graffiti on window., Graffiti on window., Graffiti on wall., Red graffiti tag on frame. |
| gpt_05_slash | anomaly | **TP** | 1/1 | 2 | Torn seat cushion., Torn seat cushion., Torn blue surface patch. |
| gpt_07_multi | anomaly | **TP** | 4/4 | 6 | backpack, bottle, torn seat, backpack and water bottle, backpack, torn seat, bottle, Black bag on seat., Graffiti tag "HOBES" on panel., Bottle on floor., Black bag on seat., Torn seat cushion., Graffiti tag "Hep*" on surface., Black strap on seat., Torn seat fabric. |
| gpt_08_phone_tag | anomaly | **TP** | 2/2 | 2 | Graffiti tag on wall., Forgotten phone on seat., Graffiti tag on wall., Phone on seat. |
| gpt_09_litter | anomaly | **TP** | 1/1 | 0 | Can left on floor., Green can on floor., Small rectangular object on floor., Small object on floor., Debris on surface. |
| gpt_10_litter | anomaly | **TP** | 1/1 | 1 | Trash can and crumpled paper., Trash on floor (can, wrapper)., Debris and trash on floor. |
| gpt_11_crowd | anomaly | **TP** | 4/4 | 11 | backpack, torn seat, bottle, water bottle on floor, black backpack on seat, graffiti and torn seat, backpack, bottle, torn seat, Black bag on seat., Graffiti tag on surface., forgotten bottle on floor, Graffiti tag on wall., torn seat cushion, Graffiti tag on wall., Torn seat cushion., Graffiti tag on surface., Black object (possibly bag or clothing)., Graffiti on vent cover., Black object on seat., Black jacket on seat., Torn seat fabric exposed foam., Torn seat fabric. |
| real_f0037 | anomaly | **TP** | 4/4 | 1 | Black bag on seat., black backpack on seat, Black bag on seat., Phone on seat., Wallet on seat. |
| real_f0053 | anomaly | **TP** | 4/4 | 6 | Black bag on seat., Black bag on seat., Black backpack on floor., Black bag on floor., Wallet on seat., Phone on seat., Wallet on seat., Phone on seat., Digital display panel., Wallet on seat. |
| real_f0100 | anomaly | **TP** | 4/4 | 4 | Black bag on seat., Backpack on seat., slashed seat cushion, Black backpack on floor., Wallet on seat., Black object on seat., Torn seat fabric., Torn seat fabric. |
| real_f0112 | anomaly | **TP** | 4/4 | 4 | black backpack on floor, black bag on seat, black backpack on floor, Torn seat cushion edge., black phone on seat, black phone on seat, Black object on seat., forgotten phone on seat |
| real_f0205 | anomaly | **TP** | 2/2 | 1 | Grey jacket on seat, backpack on seat, forgotten object (bag) |
| real_f0219 | anomaly | **TP** | 2/3 | 3 | forgotten jacket on seat, forgotten jacket, Bag on seat., Black backpack on seat., Green strap or cord., Black bag on seat. |
| variant_01 | anomaly | **TP** | 4/4 | 6 | forgotten bottle on floor, Backpack and stain on seat., Backpack on seat., Graffiti tag "KEEP" drawn on surface., Torn seat fabric., Bottle on floor., bottle on floor, Backpack on seat., Plastic bottle on floor., Torn seat cushion., Torn seat cushion., Backpack appeared on right. |
| neg_gpt_06_clean | clean | **FP** | — | 2 | Graffiti on seat cushion., Broken part on right side. |
| neg_v2_f0001_person | clean | **FP** | — | 1 | Black panel with text. |
| neg_v3_f0001 | clean | **FP** | — | 2 | Black backpack on seat., Backpack on seat. |
| neg_v4_f0004 | clean | **FP** | — | 1 | Torn seat cushion. |
| neg_v4_f0022 | clean | **FP** | — | 2 | Torn seat cushion., Torn seat fabric. |
| neg_real_ref_self | clean | **TN** | — | 0 | — |
| neg_v1_f0151 | clean | **TN** | — | 0 | — |
| neg_v1_f0181 | clean | **TN** | — | 0 | — |
| neg_v1_f0211 | clean | **TN** | — | 0 | — |
| neg_v1_f0241 | clean | **TN** | — | 0 | — |
| neg_v4_f0016 | clean | **TN** | — | 0 | — |
| neg_variant_ref_self | clean | **TN** | — | 0 | — |

Annotated images: `benchmark/annotated/<id>.jpg` (blue = ground-truth boxes, green = correct detections, red = false-positive boxes). Raw results: `benchmark/results.json`.
