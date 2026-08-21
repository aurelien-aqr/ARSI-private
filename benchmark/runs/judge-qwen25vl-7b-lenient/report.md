# vlm_05 reference-diff - anomaly detection benchmark

**Status:** COMPLETE  
**Model:** `qwen2.5vl:7b` (Ollama)  
**Decision rule:** frame flagged if the VLM keeps ≥1 region (`filter` mode) after dropping person/"disappeared" labels and de-duplicating overlapping boxes.  
**Diff / region params:** DIFF_THRESHOLD=40, BLUR_RADIUS=3, MIN_AREA=500, MAX_AREA=400000, MAX_REGIONS=25.  
**Wall-clock:** 12.0 min.

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

- Cases: **68**  (TP=30, FP=34, TN=3, FN=1)
- **Accuracy** 0.485 · **Precision** 0.469 · **Recall** 0.968 · **Specificity** 0.081 · **F1** 0.632

| | predicted anomaly | predicted clean |
|---|---|---|
| **actual anomaly** | TP = 30 | FN = 1 |
| **actual clean**   | FP = 34 | TN = 3 |

## 2) Object-level (did we box each real anomaly?)

- Instances detected: **63 / 73** → **object recall 0.863** (strict IoU≥0.3: 53 / 73 = 0.726)
- False-positive regions (kept boxes matching no real anomaly): **508** of 579 kept → region precision 0.123
- Uncached VLM calls this run: 654, mean 0.9 s/call

| type | instances detected | recall |
|---|---|---|
| object | 45 / 53 | 0.85 |
| graffiti | 6 / 7 | 0.86 |
| damage | 6 / 6 | 1.00 |
| litter | 6 / 7 | 0.86 |

| source | cases | instances detected | FP regions |
|---|---|---|---|
| gpt | 11 | 18 / 20 | 80 |
| real | 54 | 39 / 46 | 418 |
| self | 2 | 0 / 0 | 0 |
| variant | 1 | 6 / 7 | 10 |

## Per-case results

| id | truth | frame | instances hit | FP boxes | VLM kept-labels |
|---|---|---|---|---|---|
| 3333_cam52_084637\* | anomaly | **TP** | 3/3 | 9 | shadow., Bag on seat., shadow., Reflection of light., small red mark., Seat damage., Seat damage., Reflection of light., torn seat., A torn seat., Paper bag on seat., damage or vandalism. |
| 3333_cam52_085124 | anomaly | **TP** | 2/2 | 15 | Small black object., Phone on seat., Small black object., damage or vandalism., shadow., Wallet on floor., Handbag on seat., Bag on floor., Blood stain., Red sticker., Shadow., Seat damage., a small object appears., Shadow., Shadow., A small scratch appears on the right side of the image., A white object appears. |
| 3333_cam53_083517 | anomaly | **TP** | 1/2 | 2 | Yellow cloth on seat., Trash can., white powder. |
| 3333_cam53_084021 | anomaly | **TP** | 2/2 | 4 | Shadow., Yellow cloth on seat., scratch mark., Bag on seat., Green object attached., Trash bag. |
| 3333_cam53_084637 | anomaly | **TP** | 1/1 | 2 | Green object on seat., Reflection., shadow. |
| 3333_cam53_085124 | anomaly | **TP** | 1/1 | 3 | white object., pink bag., Forgotten object., Reflection changed. |
| 3333_cam54_084021 | anomaly | **TP** | 1/1 | 5 | plastic bag., graffiti tag scribble., Shadow., torn seat., damage or vandalism., damage or vandalism. |
| 3333_cam54_084637 | anomaly | **TP** | 2/2 | 13 | shadow., Shadow., Reflection of light., Cloth on seat., Laptop on seat., Seat damage., Shadow present., reflection., Seat damage., shadow., Yellow object., Reflection of light., An object appears., Yellow object., shadow. |
| 3333_cam54_085124 | anomaly | **TP** | 3/3 | 9 | A discarded cloth., Bag on floor., Reflection of dog., Broken glass., shadow., Shadow., Reflection of light., Reflection of light., Flame damage., Reflection of light., Reflection of light., graffiti tag. |
| 3333_cam55_083517 | anomaly | **TP** | 2/2 | 9 | Purple plastic bag., damage or vandalism., Folded seat back., damage or vandalism., Reflection of light., damage or vandalism., Scratch on metal., Scratch on wood., damage or vandalism., Seat damage., A torn or slashed seat. |
| 3333_cam55_084021 | anomaly | **TP** | 2/2 | 8 | Paper bag., Seat cushion appears damaged., Reflection of light., Scratch on surface., Seat damage., damage or vandalism., Reflection of light., A bottle., damage or vandalism., An object appears. |
| 3333_cam55_084637 | anomaly | **TP** | 2/2 | 6 | Seat damage., Bag on floor., Laptop on seat., Mask on seat., A small object appears., Bag on floor., damage or vandalism., A torn seat. |
| 3333_cam55_085124 | anomaly | **TP** | 2/2 | 9 | reflection., Reflection of bottle., Bottle and book on seat., Paper bag., Scratch on metal., shadow., A bag appears., shadow., graffiti tag., Shadow., Reflection of light. |
| gpt_01_suitcase | anomaly | **TP** | 1/1 | 5 | Luggage appears., scratch on wood., damage or vandalism., Shadow., damage or vandalism., damage or vandalism. |
| gpt_02_multi | anomaly | **TP** | 2/4 | 7 | Backpack on seat., Bottle on seat., Backpack., Scribble drawn on surface., Seat damage., Scratch on window., poster on window., reflection., A small black mark appears on the right side that isn't present on the left. It looks like a scratch or scuff mark. |
| gpt_03_faint_tag | anomaly | **TP** | 1/1 | 10 | Scribble drawn on surface., A small object appears on the right side., damage or vandalism., torn seat., damage or vandalism., shadow., Damage to window frame., damage or vandalism., damage or vandalism., Seat damage., a small white object. |
| gpt_04_graffiti | anomaly | **TP** | 1/1 | 5 | Graffiti on wall., Graffiti and tag., Graffiti., Graffiti tag scribble., damage or vandalism., Graffiti., Green strap., Blood stain., scratch mark., Package on floor., damage or vandalism. |
| gpt_05_slash | anomaly | **TP** | 1/1 | 10 | Damage to seat., damage or vandalism., shadow., scribble drawn on surface., a torn seat., torn seat., torn seat., torn seat., Shadow., A small black mark appears on the right side that isn't present on the left. It looks like a tag or scribble drawn on the surface., bottle. |
| gpt_07_multi | anomaly | **TP** | 4/4 | 7 | Backpack and bottle., bag on floor., Graffiti tag., damage to seat., plastic bottle., Bottle on floor., phone., a small scratch appears on the right side of the window frame., A small object on floor., damage or vandalism., graffiti. |
| gpt_08_phone_tag | anomaly | **TP** | 2/2 | 8 | Scribble drawn on surface., Black box object., damage or vandalism., torn sign., damage or vandalism., graffiti tag scribble., torn material., Seat damage., damage or vandalism., reflection of object. |
| gpt_09_litter | anomaly | **TP** | 1/1 | 6 | can., Forgotten objects., Can of soda., Canopy damage., Small object on floor., damage or vandalism., damage or vandalism., torn seat., damage or vandalism., damage or vandalism., shadow. |
| gpt_10_litter | anomaly | **TP** | 1/1 | 7 | Canopy and can., torn sign., Scratch mark., bag on floor., damage or vandalism., scratch., damage or vandalism., reflection of object. |
| gpt_11_crowd | anomaly | **TP** | 4/4 | 7 | Backpack and bottle on floor., Black backpack and bottle., Graffiti tag scribble., bag on floor., Scratch on seat., Bottle on floor., torn seat., Graffiti tag scribble., damage or vandalism., damage or vandalism., graffiti tag scribble. |
| real_f0037 | anomaly | **TP** | 3/4 | 8 | Seat damage., Backpack., Seat damage., damage or vandalism., damage or vandalism., damage or vandalism., metal scratch., scratch mark., a small mark appears on the right side that isn't present on the left side. It looks like a scratch or scuff mark., scratch mark., scribble drawn on surface. |
| real_f0053 | anomaly | **TP** | 3/4 | 8 | backpack., Backpack., Bag on seat., Digital display., scribble drawn on surface., scratch mark., Shadow., Reflection of object., light strip., Scratch on metal., scratch mark. |
| real_f0100 | anomaly | **TP** | 2/4 | 9 | Backpack., damage or vandalism., damage or vandalism., Graffiti tag scribble., Seat damage., Scratch on metal frame., damage or vandalism., graffiti tag scribble., phone on seat., damage or vandalism., scratch mark. |
| real_f0112 | anomaly | **TP** | 3/4 | 9 | bag on floor., damage or vandalism., Black backpack., Bag on seat., Black object., Scratch on surface., Package on floor., reflection., damage or vandalism., Green sticker., A blue light reflection., a shadow. |
| real_f0205 | anomaly | **TP** | 2/2 | 11 | Clothing item on seat., backpack., Seat damage., damage or vandalism., debris., A small mark appears on the right side that isn't present on the left. It looks like a scratch or scuff mark., Green sticker., scratch mark., reflection., Seat damage., graffiti tag., blue light reflection., A bottle. |
| real_f0219 | anomaly | **TP** | 2/2 | 11 | Clothing on seat., backpack., Bag on floor., Scratch on metal., package., A blue light anomaly., Seat damage., Seat damage., scratch mark., damage or vandalism., Shadow., phone on seat., graffiti. |
| variant_01 | anomaly | **TP** | 6/7 | 10 | backpack., Bottle on floor., torn seat., Graffiti., torn seat., Scratch on seat., sticker on window., a torn seat., shadow., Small orange object., damage or vandalism., shadow., Shadow., scratch mark., a shadow. |
| 3333_cam54_083517 | anomaly | **FN** | 0/1 | 0 | - |
| 1760_cam04_t120 | clean | **FP** | - | 5 | scratch mark., Scratch on surface., a small white object appears on the right side that is not present on the left side. It looks like a piece of debris or trash., Shadow., shadow. |
| 1760_cam04_t220 | clean | **FP** | - | 10 | Shadow on right., Seat damage., Red light illuminated., Shadow present., Shadow on floor., Reflection of light., Shadow., Shadow., Reflection of light., Shadow. |
| 1760_cam04_t320 | clean | **FP** | - | 8 | Reflection., shadow., Fur on floor., Seat damage., Shadow of dog., A small object appears on the right side., small metal object., Shadow. |
| 1760_cam04_t420 | clean | **FP** | - | 10 | sunlight reflection., Reflection on window., Shadow on right., Reflection., light reflection., Reflection of light., Bag on floor., Seat damage., Shadow., damage or vandalism. |
| 1760_cam04_t520 | clean | **FP** | - | 6 | Shadow present., Seat damage., shadow., A small object appears on the right side., A small object appears on the right side., Seat damage. |
| 1760_cam04_t570 | clean | **FP** | - | 5 | damage or vandalism., A small object appears on the right side., Reflection of light., scratch mark., Black scribble. |
| 1760_cam06_t120 | clean | **FP** | - | 4 | reflection., Seat damage., damage or vandalism., Scratch mark. |
| 1760_cam06_t220 | clean | **FP** | - | 4 | Reflection of light., Seat damage., damage or vandalism., graffiti tag. |
| 1760_cam06_t320 | clean | **FP** | - | 3 | Reflection of light., Reflection of light., Seat damage. |
| 1760_cam06_t420 | clean | **FP** | - | 10 | Reflection of light., Reflection of light., Reflection of light., Reflection of green object., Shadow on floor., Reflection of light., Green bag., Graffiti tag drawn on surface., Scratch on surface., Shadow. |
| 1760_cam06_t520 | clean | **FP** | - | 4 | reflection., damage or vandalism., Seat damage., a phone. |
| 1760_cam06_t570 | clean | **FP** | - | 5 | Reflection of object on floor., Reflection of light., Reflection of light., graffiti tag scribble., damage or vandalism. |
| 1760_cam13_t120 | clean | **FP** | - | 1 | torn seat. |
| 1760_cam13_t220 | clean | **FP** | - | 6 | Seat damage., Reflection of light., phone., Reflection of light., Bag on seat., damage or vandalism. |
| 1760_cam13_t320 | clean | **FP** | - | 14 | Seat damage., damage or vandalism., Smudge on surface., Reflection of light., Reflection., Seat damage., Seat damage., Seat damage., Seat damage., Snow accumulation., Shadow on right., Seat damage., Seat damage., reflection. |
| 1760_cam13_t420 | clean | **FP** | - | 12 | reflection., Seat damage., Reflection of light., damage or vandalism., yellow object., damage or vandalism., Vent cover., Seat damage., Paper bag., Paper on seat., Paper bag on floor., Seat damage. |
| 1760_cam13_t520 | clean | **FP** | - | 5 | Seat damage., Reflection of light., damage or vandalism., A phone appears., Scratch on metal. |
| 1760_cam13_t570 | clean | **FP** | - | 5 | Seat damage., Seat damage., Yellow scribble., reflection of light fixture., A torn seat appears on the right side. |
| 3333_cam52_ref_t120_clean | clean | **FP** | - | 20 | scratch mark., Bottle on floor., A small object appears on the right side., torn seat., Seat damage., Seat damage., Seat damage., graffiti tag scribble., damage or vandalism., Seat damage., damage or vandalism., damage or vandalism., Seat damage., a damaged seat., Seat damage., Seat damage., Scratch on metal., damage or vandalism., a small object., Damage to surface. |
| 3333_cam53_085954_clean | clean | **FP** | - | 6 | shadow., shadow., graffiti., reflection., cable., scribble drawn on surface. |
| 3333_cam54_085954_clean | clean | **FP** | - | 14 | shadow., shadow., Shadow of object., Shadow present., Shadow., Shadow., Shadow., shadow., Seat damage., Reflection of light., Shadow., Shadow present., Reflection of object., Shadow. |
| 3333_cam54_ref_t120_clean | clean | **FP** | - | 8 | Shadow., Shadow of object., Paper bag., Shadow., bottle., scratch mark., Scribble drawn on surface., Shadow. |
| 3333_cam55_085954_clean | clean | **FP** | - | 16 | Seat damage., damage or vandalism., Black object., reflection., damage or vandalism., a forgotten object., A small red mark appears on the right side., scratch mark., Graffiti tag scribble., scratch on wood., reflection., Shadow., Scratch on seat., damage or vandalism., A small object appears on the right side., A small white object appears on the right side that isn't present on the left. It looks like a piece of paper or a small card. |
| 3333_cam55_ref_t120_clean | clean | **FP** | - | 9 | Seat damage., reflection., damage or vandalism., scratch mark., Shadow., damage or vandalism., Reflection of object., A torn seat., damage or vandalism. |
| neg_gpt_06_clean | clean | **FP** | - | 8 | Scribble drawn on surface., damage or vandalism., torn seat., torn seat., Shadow., damage or vandalism., A torn or slashed seat., graffiti. |
| neg_v1_f0151 | clean | **FP** | - | 9 | blue light reflection., damage or vandalism., damage or vandalism., damage or vandalism., graffiti., damage or vandalism., a small white object appears on the right side that is not present on the left side. It looks like a piece of paper or a small card., a dark object., damage or vandalism. |
| neg_v1_f0181 | clean | **FP** | - | 4 | scratch mark., Reflection of light., graffiti., damage or vandalism. |
| neg_v1_f0211 | clean | **FP** | - | 6 | damage or vandalism., damage or vandalism., scratch mark., damage or vandalism., scratch on surface., scratch mark. |
| neg_v1_f0241 | clean | **FP** | - | 5 | Scratch on window., damage or vandalism., damage or vandalism., scratch mark., A small object appears on the right side that isn't present on the left. It looks like a bottle or can. |
| neg_v2_f0001_person | clean | **FP** | - | 5 | package., Scratch on wood., a small object appears., Shadow., Blue mark on glass. |
| neg_v3_f0001 | clean | **FP** | - | 9 | bag on floor., damage or vandalism., a small white object appears on the right side that is not present on the left side. It looks like a piece of paper or a small card., scratch mark., graffiti tag scribble., scratch mark., graffiti tag scribble., A bottle appears on the right side., graffiti. |
| neg_v4_f0004 | clean | **FP** | - | 16 | damage or vandalism., Red scribble., Shadow., Shadow., shadow., graffiti tag scribble., Shadow., graffiti tag., A small object appears., Damage to seat., damage or vandalism., damage or vandalism., damage or vandalism., damage or vandalism., Scratch on window frame., reflection. |
| neg_v4_f0016 | clean | **FP** | - | 15 | Shadow., Reflection of window., a small object., a torn seat., scratch mark., Shadow., Reflection of light., Scratch on metal., damage or vandalism., a small scratch., damage or vandalism., Seat damage., dirt buildup., reflection., graffiti tag scribble. |
| neg_v4_f0022 | clean | **FP** | - | 9 | torn seat., Scratch on seat., Shadow., Seat damage., Reflection on surface., Scratch mark., a small object., scratch mark., small mark on metal. |
| 3333_cam53_ref_t120_clean | clean | **TN** | - | 0 | - |
| neg_real_ref_self | clean | **TN** | - | 0 | - |
| neg_variant_ref_self | clean | **TN** | - | 0 | - |

Annotated images: `benchmark/runs/cli-latest/annotated/<id>.jpg` (blue = ground-truth boxes, green = correct detections, red = false-positive boxes). Raw results: `benchmark/runs/judge-qwen25vl-7b-lenient/results.json`.

---

\* **3333** is a placeholder, not the tram's real fleet number - the vehicle number of the 2026-08-11 capture is unknown. It was called 39T before, but 39T is the Škoda type, which tram 1760 shares.
