# vlm_05 reference-diff - anomaly detection benchmark

**Status:** COMPLETE  
**Model:** `qwen2.5vl:7b` (Ollama)  
**Decision rule:** frame flagged if the VLM keeps ≥1 region (`filter` mode) after dropping person/"disappeared" labels and de-duplicating overlapping boxes.  
**Diff / region params:** DIFF_THRESHOLD=40, BLUR_RADIUS=3, MIN_AREA=500, MAX_AREA=400000, MAX_REGIONS=25.  
**Wall-clock:** 12.1 min.

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

- Instances detected: **63 / 73** → **object recall 0.863** (strict IoU≥0.3: 52 / 73 = 0.712)
- False-positive regions (kept boxes matching no real anomaly): **246** of 316 kept → region precision 0.222
- Uncached VLM calls this run: 654, mean 0.9 s/call

| type | instances detected | recall |
|---|---|---|
| object | 45 / 53 | 0.85 |
| graffiti | 6 / 7 | 0.86 |
| damage | 6 / 6 | 1.00 |
| litter | 6 / 7 | 0.86 |

| source | cases | instances detected | FP regions |
|---|---|---|---|
| gpt | 11 | 20 / 20 | 43 |
| real | 54 | 37 / 46 | 199 |
| self | 2 | 0 / 0 | 0 |
| variant | 1 | 6 / 7 | 4 |

## Per-case results

| id | truth | frame | instances hit | FP boxes | VLM kept-labels |
|---|---|---|---|---|---|
| 39T_cam52_084637 | anomaly | **TP** | 3/3 | 5 | shadow on seat., Purple bag on seat., Seat stain., small red mark., Seat tear., broken seat., Graffiti., dog. |
| 39T_cam52_085124 | anomaly | **TP** | 2/2 | 12 | small black object on seat back., black wallet., small black object., Damaged seat, Pink object., Wallet on seat., pink object., Bag on seat., blood stain., Red sticker on window., small white object., small black mark., Graffiti on wall., Graffiti. |
| 39T_cam53_083517 | anomaly | **TP** | 1/2 | 2 | Yellow jacket., Wrapper., white powder cloud. |
| 39T_cam53_084021 | anomaly | **TP** | 2/2 | 2 | Yellow cloth on seat., Yellow bag., green bottle., Wrapper on seat. |
| 39T_cam53_084637 | anomaly | **TP** | 1/1 | 0 | Green wrapper. |
| 39T_cam53_085124 | anomaly | **TP** | 1/1 | 3 | paper wrapper., pink bag., Two white objects on floor., black object on floor. |
| 39T_cam54_084021 | anomaly | **TP** | 1/1 | 2 | plastic bag., Litter left behind (can)., litter left behind. |
| 39T_cam54_084637 | anomaly | **TP** | 2/2 | 5 | cloth on seat., Laptop., Seat damage., Seat stain., wrapper paper., Yellow object., Yellow object. |
| 39T_cam54_085124 | anomaly | **TP** | 3/3 | 3 | discarded cloth., Plastic bag on floor., green wrapper., torn seat., Flame., torn seat. |
| 39T_cam55_083517 | anomaly | **TP** | 2/2 | 4 | Purple plastic bag., Folded seat back., Yellow line on floor., small hole., Seat tear., Graffiti. |
| 39T_cam55_084021 | anomaly | **TP** | 2/2 | 3 | plastic bag., Seat cushion., wrapper paper., Paper wrapper., Marked lines on roof. |
| 39T_cam55_084637 | anomaly | **TP** | 2/2 | 3 | pink seat cover., cloth bag., Laptop on seat., Bag on floor., Graffiti. |
| 39T_cam55_085124 | anomaly | **TP** | 2/2 | 4 | Green bottle., Green bottle on seat., paper wrapper., paper bag., A small green object appears on the right side of the image. It looks like a piece of paper or a wrapper., scratch on metal. |
| gpt_01_suitcase | anomaly | **TP** | 1/1 | 4 | black suitcase., torn poster., sticker on window., scratch on metal., black object on seat. |
| gpt_02_multi | anomaly | **TP** | 4/4 | 5 | Backpack on seat., backpack., Water bottle on seat., Black backpack, bottle., scratch on metal., torn seat., Canister on floor. |
| gpt_03_faint_tag | anomaly | **TP** | 1/1 | 4 | XRP graffiti., torn seat., green line., torn seat panel., small black object. |
| gpt_04_graffiti | anomaly | **TP** | 1/1 | 3 | Graffiti on wall., Graffiti on wall, Graffiti on wall, Graffiti., Purple paint mark., Green strap., Blood stain., scratch on wood., Paper wrapper. |
| gpt_05_slash | anomaly | **TP** | 1/1 | 5 | Scratch on seat., torn seat., torn seat panel., sticker on window., black object., Graffiti on wall. |
| gpt_07_multi | anomaly | **TP** | 4/4 | 3 | Backpack on seat, black bag on floor., Graffiti: Hobag., Seat damage., plastic bottle., Water bottle on floor., bottle wrapper. |
| gpt_08_phone_tag | anomaly | **TP** | 2/2 | 5 | Graffiti., small black box., green line., torn seat panel., handlebar., Graffiti., paper wrapper. |
| gpt_09_litter | anomaly | **TP** | 1/1 | 1 | Can of soda., two small rectangular objects., Can., Battery wrapper., small rectangular object., blue object. |
| gpt_10_litter | anomaly | **TP** | 1/1 | 3 | can on floor, black bag., scratch on metal., paper wrapper. |
| gpt_11_crowd | anomaly | **TP** | 4/4 | 5 | Backpack on seat., Black backpack on seat., Graffiti: Hob*8, Scratch on seat., Plastic bottle on floor., Seat torn., Graffiti., torn panel., graffiti. |
| real_f0037 | anomaly | **TP** | 3/4 | 1 | Small rectangular object., backpack., black object on seat., scratch on metal. |
| real_f0053 | anomaly | **TP** | 3/4 | 2 | black bag., backpack., backpack., Digital display., blue light strip. |
| real_f0100 | anomaly | **TP** | 2/4 | 3 | backpack., graffiti., Phone on seat., phone on seat., Graffiti. |
| real_f0112 | anomaly | **TP** | 3/4 | 4 | purple sticker., backpack., backpack., Black object., A small white mark appears on the seat backrest of the right image that is not present in the left image., Green wrapper., blue light reflection. |
| real_f0205 | anomaly | **TP** | 1/2 | 6 | backpack., Green wrapper., scratch on metal., phone on seat., black object on seat., Blue light reflection., graffiti. |
| real_f0219 | anomaly | **TP** | 1/2 | 10 | backpack., backpack., scratch on metal., paper wrapper., blue light strip., black cylindrical object., broken seat., broken window., Purple object on floor., phone on seat., Blue light reflection. |
| variant_01 | anomaly | **TP** | 6/7 | 4 | backpack., Plastic bottle on floor., Torn seat material., Graffiti: "KEPP"., torn seat., Scratch on seat., sticker on wall., sticker on wall., Small yellow object. |
| 39T_cam54_083517 | anomaly | **FN** | 0/1 | 0 | - |
| 1760_cam04_t120 | clean | **FP** | - | 4 | scratch on metal., small white object., Can., red tag. |
| 1760_cam04_t220 | clean | **FP** | - | 4 | Seat torn on right side., red light., Graffiti., Black mark on pole. |
| 1760_cam04_t320 | clean | **FP** | - | 4 | Yellow object attached to seat., yellow stain., small black mark., small metal object. |
| 1760_cam04_t420 | clean | **FP** | - | 7 | blue seat cover., scratch on metal., paper wrapper., reflection water., bottle., Paper wrapper., Graffiti. |
| 1760_cam04_t520 | clean | **FP** | - | 2 | Graffiti., Graffiti on wall. |
| 1760_cam04_t570 | clean | **FP** | - | 1 | paper wrapper. |
| 1760_cam06_t120 | clean | **FP** | - | 1 | Graffiti. |
| 1760_cam06_t220 | clean | **FP** | - | 2 | torn seat panel., scratch on metal. |
| 1760_cam06_t320 | clean | **FP** | - | 4 | Paper bag., Reflection glare., Graffiti., Seat torn. |
| 1760_cam06_t420 | clean | **FP** | - | 9 | Reflection of light., shadow on floor., Reflection of green object., Sticker on window., Sticker on seat., green bag., Graffiti., torn seat panel., Paper wrapper. |
| 1760_cam06_t520 | clean | **FP** | - | 1 | Seat damage. |
| 1760_cam06_t570 | clean | **FP** | - | 2 | Paper wrapper., Seat tear. |
| 1760_cam13_t120 | clean | **FP** | - | 1 | torn seat. |
| 1760_cam13_t220 | clean | **FP** | - | 3 | shadow cast on seat., A small white object appears on the right side near the top of the image. It looks like a piece of paper or a small card., yellow jacket. |
| 1760_cam13_t320 | clean | **FP** | - | 9 | smoke., Yellow object on floor., yellow stain., Seat damage., Paper wrapper., broken panel., snow accumulation., black object on seat., Stained seat area. |
| 1760_cam13_t420 | clean | **FP** | - | 10 | Paper wrapper., Seat torn., Yellow object on seat., Paper wrapper., Vent cover., yellow seat cover., Paper wrapper on floor., Paper wrapper., Paper wrapper., Seat tear. |
| 1760_cam13_t520 | clean | **FP** | - | 2 | torn seat., Damage to seat |
| 1760_cam13_t570 | clean | **FP** | - | 3 | torn seat., Yellow text on metal., Canister hanging. |
| 39T_cam52_ref_t120_clean | clean | **FP** | - | 8 | torn seat., bottle., Graffiti., torn seat., Graffiti on wall., blue object on seat., A torn seat panel., black mark on seat. |
| 39T_cam53_085954_clean | clean | **FP** | - | 2 | Graffiti: painted letters, wire. |
| 39T_cam54_085954_clean | clean | **FP** | - | 4 | Seat cushion., torn seat., seat torn., shadow cast on wall. |
| 39T_cam54_ref_t120_clean | clean | **FP** | - | 5 | Paper wrapper., litter left behind., Graffiti on wall, small scratch on metal., Graffiti. |
| 39T_cam55_085954_clean | clean | **FP** | - | 4 | black sticker., paper wrapper., Pink object., small green object. |
| 39T_cam55_ref_t120_clean | clean | **FP** | - | 2 | small tear on seat., broken window. |
| neg_gpt_06_clean | clean | **FP** | - | 5 | Graffiti., torn seat., torn seat., sticker on window., torn seat. |
| neg_v1_f0151 | clean | **FP** | - | 2 | Graffiti: painted letters "CAD", Black object on seat. |
| neg_v1_f0181 | clean | **FP** | - | 2 | scratch on metal., Graffiti on the wall. |
| neg_v2_f0001_person | clean | **FP** | - | 3 | Orange cloth draped over seat., Damaged seat panel., Blue mark on pole. |
| neg_v3_f0001 | clean | **FP** | - | 5 | Backpack on seat, black bag., scratch on metal., bottle., graffiti. |
| neg_v4_f0004 | clean | **FP** | - | 7 | dark smudge on seat., Seat torn., broken window pane., Litter left behind., bottle., A torn seat appears on the right side., phone on seat. |
| neg_v4_f0016 | clean | **FP** | - | 7 | black bag., scratch on metal., red circle sign., broken window., water stain., debris., sticker on door. |
| neg_v4_f0022 | clean | **FP** | - | 5 | torn seat panel., black object on seat., Graffiti on wall., A small white object appears on the floor near the bottom right of the image on the right side. It is not present in the left image., small white mark. |
| 39T_cam53_ref_t120_clean | clean | **TN** | - | 0 | - |
| neg_real_ref_self | clean | **TN** | - | 0 | - |
| neg_v1_f0211 | clean | **TN** | - | 0 | - |
| neg_v1_f0241 | clean | **TN** | - | 0 | - |
| neg_variant_ref_self | clean | **TN** | - | 0 | - |

Annotated images: `benchmark/runs/cli-latest/annotated/<id>.jpg` (blue = ground-truth boxes, green = correct detections, red = false-positive boxes). Raw results: `benchmark/runs/judge-qwen25vl-7b-conservative/results.json`.
