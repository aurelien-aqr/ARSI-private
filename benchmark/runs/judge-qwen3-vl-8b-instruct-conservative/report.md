# vlm_05 reference-diff - anomaly detection benchmark

**Status:** COMPLETE  
**Model:** `qwen3-vl:8b-instruct` (Ollama)  
**Decision rule:** frame flagged if the VLM keeps ≥1 region (`filter` mode) after dropping person/"disappeared" labels and de-duplicating overlapping boxes.  
**Diff / region params:** DIFF_THRESHOLD=40, BLUR_RADIUS=3, MIN_AREA=500, MAX_AREA=400000, MAX_REGIONS=25.  
**Wall-clock:** 14.1 min.

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

- Cases: **68**  (TP=30, FP=32, TN=5, FN=1)
- **Accuracy** 0.515 · **Precision** 0.484 · **Recall** 0.968 · **Specificity** 0.135 · **F1** 0.645

| | predicted anomaly | predicted clean |
|---|---|---|
| **actual anomaly** | TP = 30 | FN = 1 |
| **actual clean**   | FP = 32 | TN = 5 |

## 2) Object-level (did we box each real anomaly?)

- Instances detected: **63 / 73** → **object recall 0.863** (strict IoU≥0.3: 53 / 73 = 0.726)
- False-positive regions (kept boxes matching no real anomaly): **188** of 259 kept → region precision 0.274
- Uncached VLM calls this run: 654, mean 1.1 s/call

| type | instances detected | recall |
|---|---|---|
| object | 45 / 53 | 0.85 |
| graffiti | 6 / 7 | 0.86 |
| damage | 6 / 6 | 1.00 |
| litter | 6 / 7 | 0.86 |

| source | cases | instances detected | FP regions |
|---|---|---|---|
| gpt | 11 | 20 / 20 | 15 |
| real | 54 | 37 / 46 | 171 |
| self | 2 | 0 / 0 | 0 |
| variant | 1 | 6 / 7 | 2 |

## Per-case results

| id | truth | frame | instances hit | FP boxes | VLM kept-labels |
|---|---|---|---|---|---|
| 39T_cam52_084637 | anomaly | **TP** | 3/3 | 3 | Purple bag on seat, White plastic bag on seat, White cat sitting on seat, Red bag on seat, Green object on seat, White dog sitting on seat |
| 39T_cam52_085124 | anomaly | **TP** | 1/2 | 7 | Black wallet on seat, wallet on seat, Red bag on right side, White stripe on floor, wallet on seat, Phone on seat back, Small black object attached, white object on seat |
| 39T_cam53_083517 | anomaly | **TP** | 1/2 | 1 | yellow bag on seat, Yellow bag on seat |
| 39T_cam53_084021 | anomaly | **TP** | 2/2 | 2 | Yellow cloth on seat, Colorful items on seat, green bottle attached, Yellow bag on seat |
| 39T_cam53_084637 | anomaly | **TP** | 1/1 | 2 | green bottle on seat, White arrow sticker, Shadowy figure silhouette |
| 39T_cam53_085124 | anomaly | **TP** | 1/1 | 2 | White smudge on floor, pink strap hanging, White shadowy shape on floor |
| 39T_cam54_084021 | anomaly | **TP** | 1/1 | 0 | plastic bag on floor |
| 39T_cam54_084637 | anomaly | **TP** | 2/2 | 4 | crumpled plastic bag, laptop on seat, white circular mark, Wooden panel slightly tilted, Black circular hole above sticker, Yellow liquid stain |
| 39T_cam54_085124 | anomaly | **TP** | 3/3 | 2 | crumpled trash bag on floor, Litter on floor: crumpled bag, White smudge on floor, bottle on seat, Metallic box on seat |
| 39T_cam55_083517 | anomaly | **TP** | 2/2 | 2 | crumpled plastic bag, laptop on seat, Black object under seat, Yellow stripe on wall |
| 39T_cam55_084021 | anomaly | **TP** | 2/2 | 4 | crumpled plastic bag, laptop on seat, Light reflection streak, Black shadow on floor, white cloth draped over seat, Black mark on right side |
| 39T_cam55_084637 | anomaly | **TP** | 2/2 | 3 | pink seat cover added, crumpled white cloth on seat, laptop on seat, white cord on seat, White object partially visible |
| 39T_cam55_085124 | anomaly | **TP** | 2/2 | 3 | bottle on seat, laptop on seat, Black circular object attached, Towel hanging down, metallic stripe added |
| gpt_01_suitcase | anomaly | **TP** | 1/1 | 0 | black suitcase on floor |
| gpt_02_multi | anomaly | **TP** | 4/4 | 3 | brown paper bag, backpack on seat, bottle on seat, black backpack appears, backpack on seat, graffiti on panel |
| gpt_03_faint_tag | anomaly | **TP** | 1/1 | 0 | Graffiti XRP on wall |
| gpt_04_graffiti | anomaly | **TP** | 1/1 | 2 | graffiti on wall, graffiti on wall, graffiti on wall, graffiti on wall behind, graffiti on wall, green strap visible, green tape on window, Purple object visible |
| gpt_05_slash | anomaly | **TP** | 1/1 | 0 | torn seat cushion |
| gpt_07_multi | anomaly | **TP** | 4/4 | 3 | black backpack on seat, black backpack appears, graffiti on panel, torn seat cushion, bottle on floor, bottle on floor, graffiti on panel |
| gpt_08_phone_tag | anomaly | **TP** | 2/2 | 1 | graffiti on wall, phone on seat, scribble on panel |
| gpt_09_litter | anomaly | **TP** | 1/1 | 0 | litter on floor, two rectangular metal plates, two cans on floor, litter left behind, Small piece of litter |
| gpt_10_litter | anomaly | **TP** | 1/1 | 0 | litter on floor |
| gpt_11_crowd | anomaly | **TP** | 4/4 | 4 | black backpack on seat, black backpack on seat, graffiti on panel, black backpack appears, torn seat cushion, bottle on floor, graffiti on surface, Graffiti letters on panel |
| real_f0037 | anomaly | **TP** | 2/4 | 3 | Black backpack appears, backpack on seat, backpack on seat, red light beam visible, backpack on seat, two small holes visible |
| real_f0053 | anomaly | **TP** | 2/4 | 2 | Black backpack visible, black backpack on floor, backpack on seat, Blue light glow visible |
| real_f0100 | anomaly | **TP** | 3/4 | 3 | Black backpack visible, Black backpack on floor, backpack on seat, Black strap on seat, Phone on seat armrest, White sticker on metal panel |
| real_f0112 | anomaly | **TP** | 3/4 | 3 | black backpack visible, backpack on floor, backpack on seat, phone on seat, green moss on metal bar, Blue light reflection |
| real_f0205 | anomaly | **TP** | 2/2 | 3 | Grey jacket on seat, backpack on seat, green leaf visible, Digital display showing lock icon, Purple light reflection |
| real_f0219 | anomaly | **TP** | 2/2 | 4 | Grey jacket on seat, black backpack appears, black backpack appears, Blue backpack visible, Phone on seat back, Blue light reflection |
| variant_01 | anomaly | **TP** | 6/7 | 2 | backpack on seat, bottle on floor, torn seat cushion, graffiti on wall, torn seat cushion, torn blue seat cushion, Small yellow object attached |
| 39T_cam54_083517 | anomaly | **FN** | 0/1 | 0 | - |
| 1760_cam04_t120 | clean | **FP** | - | 2 | Red object attached to frame, Small white mark on surface |
| 1760_cam04_t220 | clean | **FP** | - | 5 | Yellow handrail handle visible, white cat sitting, white plastic bag, Yellow handle grip attached, white plastic bag |
| 1760_cam04_t320 | clean | **FP** | - | 2 | Sunlight patch on floor, Yellowish haze/smoke cloud |
| 1760_cam04_t420 | clean | **FP** | - | 7 | Sunlight reflection on seat, Dark stain on floor, Red button pressed down, Bright light reflection, Blue sticker on wall, Yellow stripe on floor, white plastic bag on seat |
| 1760_cam04_t520 | clean | **FP** | - | 3 | sunlight on seat back, Blue seat has crease, Blue stripe on seat |
| 1760_cam04_t570 | clean | **FP** | - | 2 | pink object visible, Red object visible |
| 1760_cam06_t120 | clean | **FP** | - | 1 | white object visible |
| 1760_cam06_t220 | clean | **FP** | - | 1 | Sunlight casting shadows |
| 1760_cam06_t320 | clean | **FP** | - | 4 | smoke cloud appears, Blurred bright rectangular light reflection, White smoke or haze, Glossy reflective surface |
| 1760_cam06_t420 | clean | **FP** | - | 8 | green tinted panel, graffiti on panel, Dark shadowy figure standing, graffiti letters appear, Small white object on floor, green sticker on seat, graffiti letters appear, Two white metal panels |
| 1760_cam06_t520 | clean | **FP** | - | 2 | white stripe on floor, White streak on seat back |
| 1760_cam06_t570 | clean | **FP** | - | 2 | White streak on floor, white paper on floor |
| 1760_cam13_t120 | clean | **FP** | - | 1 | Torn seat fabric edge |
| 1760_cam13_t220 | clean | **FP** | - | 4 | White cloth draped over seat, Sunlight reflection on seat, White vertical stripe on right side, Yellow cloth draped over pole |
| 1760_cam13_t320 | clean | **FP** | - | 8 | Yellowish substance on ceiling, Yellowish substance on seat, White object under seat, White plastic object on seat, white substance on seat, white ball on floor, Small object on floor, Yellow object on seat |
| 1760_cam13_t420 | clean | **FP** | - | 12 | White patch on seat, white object on seat, White object on floor, Yellow object under seat, Yellow paper on seat, metal vent grille, Yellow light reflection, yellow seat visible, White plastic bag visible, White box under seat, White sheet on seat, white ball visible |
| 1760_cam13_t520 | clean | **FP** | - | 1 | Black strap on pole |
| 1760_cam13_t570 | clean | **FP** | - | 3 | metal pole bent down, green sticker on right side, Dark vertical mark on right side |
| 39T_cam52_ref_t120_clean | clean | **FP** | - | 4 | White light reflection on floor, Black rectangular panel, Dark rectangular mark on right side, White speckled line pattern |
| 39T_cam53_085954_clean | clean | **FP** | - | 5 | Dark shadowy object on floor, Dark object on floor, shadow of vertical bars, metal pipe attached to wall, white object on floor |
| 39T_cam54_085954_clean | clean | **FP** | - | 1 | Long shadow cast across floor |
| 39T_cam54_ref_t120_clean | clean | **FP** | - | 4 | White plastic bottle cap, black box under seat, white cloth on floor, Small dark spots on surface |
| 39T_cam55_085954_clean | clean | **FP** | - | 8 | Black circular mark on white panel, white paper on seat, Brownish discoloration on wood panels, white paper on seat, Brown stain on wood panel, Red mark on seat edge, Green sticker on metal edge, White object on right side |
| 39T_cam55_ref_t120_clean | clean | **FP** | - | 2 | Brown stain on seat back, Reddish stain on seat back |
| neg_gpt_06_clean | clean | **FP** | - | 2 | Dark stain on floor, Dark stain on seat cushion |
| neg_v1_f0151 | clean | **FP** | - | 2 | blue light reflection, purple light reflection |
| neg_v1_f0181 | clean | **FP** | - | 1 | red light beam visible |
| neg_v2_f0001_person | clean | **FP** | - | 1 | laptop on seat |
| neg_v3_f0001 | clean | **FP** | - | 4 | backpack on seat, Black backpack appears, Blue stripe on rail, Purple light reflection |
| neg_v4_f0004 | clean | **FP** | - | 2 | black bag on seat, Black rectangular device mounted |
| neg_v4_f0016 | clean | **FP** | - | 6 | black bag appears, Black backpack visible, vertical crease on right panel, Blue light reflection, Water puddle on floor, Phone screen lit up |
| neg_v4_f0022 | clean | **FP** | - | 10 | black backpack appears, Screen displaying image, Small white box on floor, Brown stain on seat back, pink sticky note, metallic box-like object, Small white mark on metal surface, Small white object on floor, red light streak, Small white object on right side |
| 39T_cam53_ref_t120_clean | clean | **TN** | - | 0 | - |
| neg_real_ref_self | clean | **TN** | - | 0 | - |
| neg_v1_f0211 | clean | **TN** | - | 0 | - |
| neg_v1_f0241 | clean | **TN** | - | 0 | - |
| neg_variant_ref_self | clean | **TN** | - | 0 | - |

Annotated images: `benchmark/runs/cli-latest/annotated/<id>.jpg` (blue = ground-truth boxes, green = correct detections, red = false-positive boxes). Raw results: `benchmark/runs/judge-qwen3-vl-8b-instruct-conservative/results.json`.
