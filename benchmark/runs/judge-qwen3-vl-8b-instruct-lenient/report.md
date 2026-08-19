# vlm_05 reference-diff - anomaly detection benchmark

**Status:** COMPLETE  
**Model:** `qwen3-vl:8b-instruct` (Ollama)  
**Decision rule:** frame flagged if the VLM keeps ≥1 region (`filter` mode) after dropping person/"disappeared" labels and de-duplicating overlapping boxes.  
**Diff / region params:** DIFF_THRESHOLD=40, BLUR_RADIUS=3, MIN_AREA=500, MAX_AREA=400000, MAX_REGIONS=25.  
**Wall-clock:** 13.6 min.

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

- Cases: **68**  (TP=30, FP=26, TN=11, FN=1)
- **Accuracy** 0.603 · **Precision** 0.536 · **Recall** 0.968 · **Specificity** 0.297 · **F1** 0.690

| | predicted anomaly | predicted clean |
|---|---|---|
| **actual anomaly** | TP = 30 | FN = 1 |
| **actual clean**   | FP = 26 | TN = 11 |

## 2) Object-level (did we box each real anomaly?)

- Instances detected: **63 / 73** → **object recall 0.863** (strict IoU≥0.3: 53 / 73 = 0.726)
- False-positive regions (kept boxes matching no real anomaly): **118** of 189 kept → region precision 0.376
- Uncached VLM calls this run: 654, mean 1.1 s/call

| type | instances detected | recall |
|---|---|---|
| object | 45 / 53 | 0.85 |
| graffiti | 6 / 7 | 0.86 |
| damage | 6 / 6 | 1.00 |
| litter | 6 / 7 | 0.86 |

| source | cases | instances detected | FP regions |
|---|---|---|---|
| gpt | 11 | 20 / 20 | 10 |
| real | 54 | 37 / 46 | 106 |
| self | 2 | 0 / 0 | 0 |
| variant | 1 | 6 / 7 | 2 |

## Per-case results

| id | truth | frame | instances hit | FP boxes | VLM kept-labels |
|---|---|---|---|---|---|
| 39T_cam52_084637 | anomaly | **TP** | 3/3 | 4 | purple bag appears., bright reflection on right side seat., White cat sitting on seat, a red bag appears., white substance on seat, small green object, white dog sitting on seats |
| 39T_cam52_085124 | anomaly | **TP** | 1/2 | 6 | phone on seat, wallet on seat, a handbag appears., shadow on seats, wallet on seat, Shadow on seat surface, phone on seat back. |
| 39T_cam53_083517 | anomaly | **TP** | 1/2 | 1 | yellow bag on seat, yellow bag appears |
| 39T_cam53_084021 | anomaly | **TP** | 2/2 | 2 | yellow cloth on seat, colorful items on seat, green bottle appears, a yellow bag appears. |
| 39T_cam53_084637 | anomaly | **TP** | 1/1 | 1 | green bottle on seat, graffiti tag on right side. |
| 39T_cam53_085124 | anomaly | **TP** | 1/1 | 1 | white smudge-like substance, pink strap hanging |
| 39T_cam54_084021 | anomaly | **TP** | 1/1 | 0 | plastic bag on floor |
| 39T_cam54_084637 | anomaly | **TP** | 2/2 | 3 | crumpled bag on seat, laptop on seat, white circular mark, liquid droplets on surface., liquid droplets on surface. |
| 39T_cam54_085124 | anomaly | **TP** | 3/3 | 2 | crumpled trash bag appeared., bottle and trash bag, a blurry reflection appears., bottle on seat, a phone appeared. |
| 39T_cam55_083517 | anomaly | **TP** | 2/2 | 1 | crumpled plastic bag, laptop on seat, light streak on surface |
| 39T_cam55_084021 | anomaly | **TP** | 2/2 | 1 | crumpled plastic bag, laptop on seat, 褶皱的白色布料 |
| 39T_cam55_084637 | anomaly | **TP** | 2/2 | 2 | crumpled trash on seat, laptop on seat, white cord/strap appears, a blurry object appears. |
| 39T_cam55_085124 | anomaly | **TP** | 2/2 | 2 | bottle and notebook, laptop and bottle, a hanging bag appears., graffiti or tag |
| gpt_01_suitcase | anomaly | **TP** | 1/1 | 0 | black suitcase appeared. |
| gpt_02_multi | anomaly | **TP** | 4/4 | 2 | backpack and paper bag, black backpack appeared., water bottle on seat, backpack on seat, graffiti on surface |
| gpt_03_faint_tag | anomaly | **TP** | 1/1 | 0 | graffiti tag "XRP" |
| gpt_04_graffiti | anomaly | **TP** | 1/1 | 2 | graffiti on wall, graffiti tag on wall, graffiti tag on wall, graffiti on right side, graffiti on window, green strap appears, graffiti tag on window, purple object appears |
| gpt_05_slash | anomaly | **TP** | 1/1 | 0 | damage on seat cushion |
| gpt_07_multi | anomaly | **TP** | 4/4 | 2 | backpack and bottle, graffiti tag appears, seat torn and damaged, plastic bottle on floor, bottle and wet stain, graffiti on panel. |
| gpt_08_phone_tag | anomaly | **TP** | 2/2 | 0 | graffiti tag appears, phone on seat |
| gpt_09_litter | anomaly | **TP** | 1/1 | 0 | two cans and litter, two small rectangular objects, two cans appeared., trash on floor, small piece of debris |
| gpt_10_litter | anomaly | **TP** | 1/1 | 0 | trash on floor |
| gpt_11_crowd | anomaly | **TP** | 4/4 | 2 | black backpack on seat, backpack on seat, graffiti tag appears, seat torn and damaged, bottle on floor, graffiti tag on surface |
| real_f0037 | anomaly | **TP** | 2/4 | 0 | backpack appeared, backpack appeared, backpack appeared |
| real_f0053 | anomaly | **TP** | 2/4 | 1 | black backpack appeared, backpack appears., blue light glow |
| real_f0100 | anomaly | **TP** | 3/4 | 1 | backpack on floor, backpack appears, a black strap-like object appears on the right side of the seat, which is not present on the left. This could be part of a bag or accessory that was forgotten or placed there., phone appears on right. |
| real_f0112 | anomaly | **TP** | 3/4 | 2 | backpack on floor, backpack appeared., phone on rail, greenish stain on metal bar., blue light reflection |
| real_f0205 | anomaly | **TP** | 2/2 | 3 | jacket on seat, backpacks appear., green foliage visible, a lock icon appears., purple light glow |
| real_f0219 | anomaly | **TP** | 2/2 | 3 | jacket on seat, black backpack appeared, black backpack appears., phone appears on right., blue light glow |
| variant_01 | anomaly | **TP** | 6/7 | 2 | backpack on seat, bottle on floor, tear in seat fabric, graffiti on wall, damage on seat., damage on seat cushion., small yellow object |
| 39T_cam54_083517 | anomaly | **FN** | 0/1 | 0 | - |
| 1760_cam04_t220 | clean | **FP** | - | 2 | white cat sitting, white plastic bag |
| 1760_cam04_t320 | clean | **FP** | - | 2 | sunlight pattern changed, orange smoke/smoke cloud |
| 1760_cam04_t420 | clean | **FP** | - | 4 | sunlight reflection, bright light reflection, a reflection appears., white plastic bag |
| 1760_cam04_t520 | clean | **FP** | - | 1 | sunlight on seats |
| 1760_cam06_t120 | clean | **FP** | - | 1 | white object inside window |
| 1760_cam06_t320 | clean | **FP** | - | 3 | smoke or steam cloud, bright reflection on glass., smoke or fire visible |
| 1760_cam06_t420 | clean | **FP** | - | 5 | green tinted glass panel, graffiti tag on right side, graffiti tag on surface, green sticker on glass., graffiti tag on right side |
| 1760_cam06_t520 | clean | **FP** | - | 2 | light reflection on floor, light reflection on seat |
| 1760_cam06_t570 | clean | **FP** | - | 1 | paper debris on floor |
| 1760_cam13_t120 | clean | **FP** | - | 1 | a tear or rip appears. |
| 1760_cam13_t220 | clean | **FP** | - | 1 | sunlight reflection |
| 1760_cam13_t320 | clean | **FP** | - | 7 | smoke or haze appears., yellowish light reflection, yellowish substance on seat., a yellow light reflection., white plastic object, white object appears, yellow object appears. |
| 1760_cam13_t420 | clean | **FP** | - | 12 | bright sunlight reflection, white object on seat, graffiti on wall, yellow object appears., yellow light reflection, ventilation grille appears., light reflection on seat, yellow seat appears., white object on seat, a yellow object appears., light reflection on seat, white ball appears. |
| 1760_cam13_t520 | clean | **FP** | - | 1 | a torn seat cover. |
| 1760_cam13_t570 | clean | **FP** | - | 1 | green graffiti tag |
| 39T_cam52_ref_t120_clean | clean | **FP** | - | 3 | bright light reflection, black rectangular patch, Graffiti-like scribble on right side |
| 39T_cam53_085954_clean | clean | **FP** | - | 4 | Dark object on surface, shadows cast by vertical bars., metal pipe attached to wall, white object appears |
| 39T_cam54_ref_t120_clean | clean | **FP** | - | 3 | black box-like object, white cloth on floor, graffiti or tag |
| 39T_cam55_085954_clean | clean | **FP** | - | 4 | reflection on seats, paper on seat., red mark on surface, graffiti tag on wall |
| neg_gpt_06_clean | clean | **FP** | - | 2 | faint scuff mark, dark stain on seat |
| neg_v1_f0151 | clean | **FP** | - | 1 | blue light reflection |
| neg_v2_f0001_person | clean | **FP** | - | 1 | laptop on seat |
| neg_v3_f0001 | clean | **FP** | - | 1 | backpack appeared. |
| neg_v4_f0004 | clean | **FP** | - | 1 | dark object appears |
| neg_v4_f0016 | clean | **FP** | - | 2 | black bag appears, phone screen lit up |
| neg_v4_f0022 | clean | **FP** | - | 6 | black backpack appears., small box on floor, pinkish stain on wood., a box appears., graffiti tag appears., Small rusted metal fragment |
| 1760_cam04_t120 | clean | **TN** | - | 0 | - |
| 1760_cam04_t570 | clean | **TN** | - | 0 | - |
| 1760_cam06_t220 | clean | **TN** | - | 0 | - |
| 39T_cam53_ref_t120_clean | clean | **TN** | - | 0 | - |
| 39T_cam54_085954_clean | clean | **TN** | - | 0 | - |
| 39T_cam55_ref_t120_clean | clean | **TN** | - | 0 | - |
| neg_real_ref_self | clean | **TN** | - | 0 | - |
| neg_v1_f0181 | clean | **TN** | - | 0 | - |
| neg_v1_f0211 | clean | **TN** | - | 0 | - |
| neg_v1_f0241 | clean | **TN** | - | 0 | - |
| neg_variant_ref_self | clean | **TN** | - | 0 | - |

Annotated images: `benchmark/runs/cli-latest/annotated/<id>.jpg` (blue = ground-truth boxes, green = correct detections, red = false-positive boxes). Raw results: `benchmark/runs/judge-qwen3-vl-8b-instruct-lenient/results.json`.
