# vlm_05 reference-diff - anomaly detection benchmark

**Status:** COMPLETE  
**Model:** `qwen3-vl:8b-instruct` (Ollama)  
**Decision rule:** frame flagged if the VLM keeps ≥1 region (`filter` mode) after dropping person/"disappeared" labels and de-duplicating overlapping boxes.  
**Diff / region params:** DIFF_THRESHOLD=40, BLUR_RADIUS=3, MIN_AREA=500, MAX_AREA=400000, MAX_REGIONS=25.  
**Wall-clock:** 13.2 min.

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

- Cases: **68**  (TP=30, FP=30, TN=7, FN=1)
- **Accuracy** 0.544 · **Precision** 0.500 · **Recall** 0.968 · **Specificity** 0.189 · **F1** 0.659

| | predicted anomaly | predicted clean |
|---|---|---|
| **actual anomaly** | TP = 30 | FN = 1 |
| **actual clean**   | FP = 30 | TN = 7 |

## 2) Object-level (did we box each real anomaly?)

- Instances detected: **63 / 73** → **object recall 0.863** (strict IoU≥0.3: 53 / 73 = 0.726)
- False-positive regions (kept boxes matching no real anomaly): **194** of 265 kept → region precision 0.268
- Uncached VLM calls this run: 599, mean 1.1 s/call

| type | instances detected | recall |
|---|---|---|
| object | 45 / 53 | 0.85 |
| graffiti | 6 / 7 | 0.86 |
| damage | 6 / 6 | 1.00 |
| litter | 6 / 7 | 0.86 |

| source | cases | instances detected | FP regions |
|---|---|---|---|
| gpt | 11 | 20 / 20 | 17 |
| real | 54 | 37 / 46 | 175 |
| self | 2 | 0 / 0 | 0 |
| variant | 1 | 6 / 7 | 2 |

## Per-case results

| id | truth | frame | instances hit | FP boxes | VLM kept-labels |
|---|---|---|---|---|---|
| 39T_cam52_084637 | anomaly | **TP** | 3/3 | 4 | Sunlight reflection on seat, purple bag hanging from pole, White cat sitting behind pole, Brown bag behind seat, White smudge on seat, Small object on seat, White fluffy animal-like object |
| 39T_cam52_085124 | anomaly | **TP** | 1/2 | 8 | Black phone on seat, wallet left behind, Red object near blue seats, White stripe on floor, wallet on seat, White object on seat, Phone hanging from seat, Small dark object attached, white object attached |
| 39T_cam53_083517 | anomaly | **TP** | 1/2 | 1 | yellow cloth on seat, Yellow bag on floor |
| 39T_cam53_084021 | anomaly | **TP** | 2/2 | 2 | yellow cloth on seat, Colorful items on seat, green bottle attached to handle, Yellow bag on seat |
| 39T_cam53_084637 | anomaly | **TP** | 1/1 | 2 | green bottle on seat, White plastic bag visible, Shadowy figure silhouette |
| 39T_cam53_085124 | anomaly | **TP** | 1/1 | 4 | White smudge on floor, Pink strap hanging down, Blurry shadowy figure, White cloth on floor, White paper towel roll |
| 39T_cam54_084021 | anomaly | **TP** | 1/1 | 2 | plastic bag on floor, White powdery substance, Steps visible on right side |
| 39T_cam54_084637 | anomaly | **TP** | 2/2 | 4 | crumpled plastic bag, laptop on seat, Green sticker on pole, White circular mark on wall, Wooden panel slightly darker, Yellowish liquid stain |
| 39T_cam54_085124 | anomaly | **TP** | 3/3 | 3 | crumpled trash bag, litter left behind, blurry shadowy figure, bottle lying between seats, Yellowish vertical smear, Metallic object on seat |
| 39T_cam55_083517 | anomaly | **TP** | 2/2 | 3 | crumpled plastic bag, laptop leaning against seat, Black shadow on floor, Light streak on wall, Brown seat cover torn |
| 39T_cam55_084021 | anomaly | **TP** | 2/2 | 4 | crumpled plastic bag, laptop resting on seat, bright light streak, White fabric draped over seat, Small white object near top right, White smear on right side |
| 39T_cam55_084637 | anomaly | **TP** | 2/2 | 3 | pinkish patch on seat back, crumpled trash bag, laptop on seat, white cord dangling from seat, Blurry white object |
| 39T_cam55_085124 | anomaly | **TP** | 2/2 | 3 | bottle and notebook, laptop and bottle, Black circular object attached, Towel hanging down, Metallic reflective stripe |
| gpt_01_suitcase | anomaly | **TP** | 1/1 | 0 | black suitcase on floor |
| gpt_02_multi | anomaly | **TP** | 4/4 | 4 | brown paper bag, Black backpack on seat, bottle on seat, Black backpack appears, Black backpack on seat, Graffiti: MD written on panel, Torn seat fabric visible |
| gpt_03_faint_tag | anomaly | **TP** | 1/1 | 0 | Graffiti: XRP written |
| gpt_04_graffiti | anomaly | **TP** | 1/1 | 3 | graffiti on wall, graffiti tags sprayed, graffiti tag on wall, graffiti tag visible, graffiti purple paint, green strap hanging down, Green tape on window frame, Two small metallic bolts, Purple object behind pole |
| gpt_05_slash | anomaly | **TP** | 1/1 | 0 | damaged seat cushion |
| gpt_07_multi | anomaly | **TP** | 4/4 | 3 | Black backpack on seat, Black backpack visible, graffiti on panel, damaged seat cushion, bottle lying on floor, bottle and wet spot, graffiti: "MD" written on panel |
| gpt_08_phone_tag | anomaly | **TP** | 2/2 | 1 | graffiti on wall, phone left behind, Handwritten letters on panel |
| gpt_09_litter | anomaly | **TP** | 1/1 | 0 | litter left behind, Two small rectangular objects, two cans on floor, litter left behind, small piece of litter |
| gpt_10_litter | anomaly | **TP** | 1/1 | 0 | litter left behind |
| gpt_11_crowd | anomaly | **TP** | 4/4 | 4 | bottle on floor, bottle on floor, graffiti on panel, Black backpack present, damaged seat cushion, bottle lying on floor, graffiti tag on surface, black jacket visible |
| real_f0037 | anomaly | **TP** | 2/4 | 2 | backpack on seat, Backpack on seat, backpack appears on right, two small holes visible, Small white dot near edge |
| real_f0053 | anomaly | **TP** | 2/4 | 2 | Black backpack visible, Black backpack on floor, Black backpack visible, Blue light glow visible |
| real_f0100 | anomaly | **TP** | 3/4 | 3 | Black bag under seat, Black backpack on floor, backpack on seat, Black strap-like object, Phone with lock screen visible, Small dark object near bolt |
| real_f0112 | anomaly | **TP** | 3/4 | 4 | Black backpack visible, Black backpack on floor, backpack behind seat, phone lying on floor, Graffiti tag on panel, greenish patch on seat, Blue light reflection |
| real_f0205 | anomaly | **TP** | 2/2 | 4 | jacket draped over seat, backpacks appear, green leaf visible, digital display showing lock icon, Black cylindrical object, Purple light glow |
| real_f0219 | anomaly | **TP** | 2/2 | 3 | Grey jacket draped over seat, Black backpack appears, black backpack appears, Black bag under seat, Blue glowing light streak |
| variant_01 | anomaly | **TP** | 6/7 | 2 | backpack on seat, bottle lying on floor, torn seat cushion, graffiti written on wall, torn seat cushion, damaged seat cushion, small yellow object |
| 39T_cam54_083517 | anomaly | **FN** | 0/1 | 0 | - |
| 1760_cam04_t120 | clean | **FP** | - | 3 | Red emergency handle, Small white reflective object, Red object attached to wall |
| 1760_cam04_t220 | clean | **FP** | - | 3 | white object near seat, white plastic bag visible, white object near window |
| 1760_cam04_t320 | clean | **FP** | - | 3 | sunlight pattern change, White spray paint mark, orange smoke cloud |
| 1760_cam04_t420 | clean | **FP** | - | 5 | Sunlight stripe on floor, Dark circular mark on floor, Small transparent plastic bag, Yellow stripe on floor, white object under seat |
| 1760_cam04_t520 | clean | **FP** | - | 3 | Sunlight casting bright shadow, Blue seat cushion crease, Blue seat cushion crease |
| 1760_cam04_t570 | clean | **FP** | - | 2 | Sunlight casting bright patch, Pink object behind pole |
| 1760_cam06_t120 | clean | **FP** | - | 1 | white object under seat |
| 1760_cam06_t220 | clean | **FP** | - | 1 | Small dark object near top right corner |
| 1760_cam06_t320 | clean | **FP** | - | 3 | smoke or vapor cloud, blurred bright rectangle, Blurry human silhouette |
| 1760_cam06_t420 | clean | **FP** | - | 7 | Greenish tinted panel, graffiti letters visible, White lettering graffiti, Small metallic object, Green seat cover visible, graffiti letters visible, Two white rectangular panels |
| 1760_cam06_t520 | clean | **FP** | - | 2 | long white object lying on floor, light reflection on seat back |
| 1760_cam06_t570 | clean | **FP** | - | 2 | shadow cast by pole, white paper triangle |
| 1760_cam13_t120 | clean | **FP** | - | 1 | Torn seat edge visible |
| 1760_cam13_t220 | clean | **FP** | - | 3 | White cloth draped over seat, Sunlight shadow cast, Yellow cloth draped over pole |
| 1760_cam13_t320 | clean | **FP** | - | 7 | Yellowish substance on ceiling, yellowish substance on seat, White object under seat, White plastic object under pole, white substance on floor, white object behind pole, Yellow object under seat |
| 1760_cam13_t420 | clean | **FP** | - | 12 | bright yellow light patch, white object under seat, Sunlight pattern difference, White rectangular panel visible, Yellow object under seat, Yellowish wooden panel, Metal vent grille added, Yellow seat cover visible, White object under seat, White box-like object under seat, White object under seat, white ball visible |
| 1760_cam13_t520 | clean | **FP** | - | 2 | Black strap hanging down, Human silhouette visible |
| 1760_cam13_t570 | clean | **FP** | - | 3 | Faint figure visible behind panel, Metal pole bent downward, Green sticker on right side |
| 39T_cam52_ref_t120_clean | clean | **FP** | - | 5 | White light reflection, Black rectangular panel, Dark rectangular patch, Dark shape behind panel, White specks along edge |
| 39T_cam53_085954_clean | clean | **FP** | - | 4 | Dark object lying flat, Shadow of vertical bars, metal pole with bent section, White object on floor |
| 39T_cam54_085954_clean | clean | **FP** | - | 4 | Shadow of hanging object, Long shadow cast across floor, White sticker on wall, Shadow cast by unseen object |
| 39T_cam54_ref_t120_clean | clean | **FP** | - | 6 | white plastic bottle, Black box-like object, white cloth on floor, White curved object visible, Small white object near rail, Small dark spots on surface |
| 39T_cam55_085954_clean | clean | **FP** | - | 7 | black circular object, white paper on seat, White paper on seat, Brownish stain on wood panel, Red mark on seat edge, Small green object visible, White object protruding from right side of the tram's roof structure — possibly a damaged or added component. |
| neg_gpt_06_clean | clean | **FP** | - | 2 | Dark stain on floor, dark stain on seat |
| neg_v1_f0151 | clean | **FP** | - | 2 | Blue light reflection, Purple light glow |
| neg_v2_f0001_person | clean | **FP** | - | 1 | laptop computer |
| neg_v3_f0001 | clean | **FP** | - | 3 | Black backpack appears, Black backpack visible, Blue curved object visible |
| neg_v4_f0004 | clean | **FP** | - | 4 | dark object under seat, dark object under seat, Framed picture slightly blurred, Darkened area with no detail |
| neg_v4_f0016 | clean | **FP** | - | 6 | black bag appears, Black backpack visible, vertical crease / wrinkle, Blue light glow visible, Small white object behind seat, phone screen lit up |
| neg_v4_f0022 | clean | **FP** | - | 9 | Black backpack visible, screen displaying image, Small box-like object, Brown stain on seat back, Pinkish paper stuck to seat, Metallic box-like object, Small white speck on surface, Small dark object on floor, Small metallic object |
| 39T_cam53_ref_t120_clean | clean | **TN** | - | 0 | - |
| 39T_cam55_ref_t120_clean | clean | **TN** | - | 0 | - |
| neg_real_ref_self | clean | **TN** | - | 0 | - |
| neg_v1_f0181 | clean | **TN** | - | 0 | - |
| neg_v1_f0211 | clean | **TN** | - | 0 | - |
| neg_v1_f0241 | clean | **TN** | - | 0 | - |
| neg_variant_ref_self | clean | **TN** | - | 0 | - |

Annotated images: `benchmark/runs/cli-latest/annotated/<id>.jpg` (blue = ground-truth boxes, green = correct detections, red = false-positive boxes). Raw results: `benchmark/runs/judge-qwen3-vl-8b-instruct-balanced/results.json`.
