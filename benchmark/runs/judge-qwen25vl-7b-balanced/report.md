# vlm_05 reference-diff - anomaly detection benchmark

**Status:** COMPLETE  
**Model:** `qwen2.5vl:7b` (Ollama)  
**Decision rule:** frame flagged if the VLM keeps ≥1 region (`filter` mode) after dropping person/"disappeared" labels and de-duplicating overlapping boxes.  
**Diff / region params:** DIFF_THRESHOLD=40, BLUR_RADIUS=3, MIN_AREA=500, MAX_AREA=400000, MAX_REGIONS=25.  
**Wall-clock:** 11.4 min.

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

- Cases: **68**  (TP=30, FP=31, TN=6, FN=1)
- **Accuracy** 0.529 · **Precision** 0.492 · **Recall** 0.968 · **Specificity** 0.162 · **F1** 0.652

| | predicted anomaly | predicted clean |
|---|---|---|
| **actual anomaly** | TP = 30 | FN = 1 |
| **actual clean**   | FP = 31 | TN = 6 |

## 2) Object-level (did we box each real anomaly?)

- Instances detected: **64 / 73** → **object recall 0.877** (strict IoU≥0.3: 53 / 73 = 0.726)
- False-positive regions (kept boxes matching no real anomaly): **284** of 355 kept → region precision 0.200
- Uncached VLM calls this run: 599, mean 0.9 s/call

| type | instances detected | recall |
|---|---|---|
| object | 46 / 53 | 0.87 |
| graffiti | 6 / 7 | 0.86 |
| damage | 6 / 6 | 1.00 |
| litter | 6 / 7 | 0.86 |

| source | cases | instances detected | FP regions |
|---|---|---|---|
| gpt | 11 | 20 / 20 | 41 |
| real | 54 | 38 / 46 | 238 |
| self | 2 | 0 / 0 | 0 |
| variant | 1 | 6 / 7 | 5 |

## Per-case results

| id | truth | frame | instances hit | FP boxes | VLM kept-labels |
|---|---|---|---|---|---|
| 3333_cam52_084637\* | anomaly | **TP** | 3/3 | 3 | Handbag., small red object., small object under seat., small object under seat., small object under seat., dog. |
| 3333_cam52_085124 | anomaly | **TP** | 2/2 | 10 | Small black object., Phone., Small black object., Wallet., Handbag hanging., small object under seat., Blood stain., Red sticker or tag., small object under seat., Paper wrapper., small object under seat., small object under seat. |
| 3333_cam53_083517 | anomaly | **TP** | 1/2 | 2 | Yellow cloth item., Wrapper., paper wrapper. |
| 3333_cam53_084021 | anomaly | **TP** | 2/2 | 3 | can., yellow cloth., yellow object., Green bottle., Wrapper. |
| 3333_cam53_084637 | anomaly | **TP** | 1/1 | 1 | Wrapper., small object under seat. |
| 3333_cam53_085124 | anomaly | **TP** | 1/1 | 4 | paper wrapper., Pink bag hanging., white object., small object under seat., small object under seat. |
| 3333_cam54_084021 | anomaly | **TP** | 1/1 | 2 | Trash bag., Paper wrapper., litter left behind: can. |
| 3333_cam54_084637 | anomaly | **TP** | 2/2 | 7 | Cloth draped seat, phone., A small object on floor., a small object under seat., can., torn seat., small object under seat., Yellow object., yellow object. |
| 3333_cam54_085124 | anomaly | **TP** | 3/3 | 4 | discarded cloth., discarded cloth., Paper wrapper., green wrapper., Paper wrapper., small object under seat., flame. |
| 3333_cam55_083517 | anomaly | **TP** | 2/2 | 5 | Purple plastic bag., Paper wrapper., Small box-like object., Paper wrapper., small object under seat., A small object on the floor., small object under seat. |
| 3333_cam55_084021 | anomaly | **TP** | 2/2 | 5 | paper bag., Small object under seat., small black object., Small object under seat., paper wrapper., bottle., Small object under seat. |
| 3333_cam55_084637 | anomaly | **TP** | 2/2 | 1 | Clothing item on seat., Laptop placed on seat., Bag. |
| 3333_cam55_085124 | anomaly | **TP** | 2/2 | 5 | Green bottle., Green bottle and book., Paper wrapper., paper wrapper., small object under seat., can., small object under seat. |
| gpt_01_suitcase | anomaly | **TP** | 1/1 | 2 | black suitcase., small sticker., Small black object. |
| gpt_02_multi | anomaly | **TP** | 4/4 | 5 | Backpack and bottle present., Black backpack, Water bottle, Black backpack, small object under seat., Graffiti: painted letters., A small object on floor., A small black object appears on the right side of the image. It looks like a phone or a small electronic device. |
| gpt_03_faint_tag | anomaly | **TP** | 1/1 | 4 | XRP graffiti., torn seat., small object under seat., Small black object., small object under seat. |
| gpt_04_graffiti | anomaly | **TP** | 1/1 | 4 | Graffiti: painted letters, tag, scribble., Graffiti: painted letters., Graffiti: painted letters, Graffiti: painted letters., Purple paint splatter., Green strap visible., Blood stain., small object under seat., Paper wrapper., small object under seat. |
| gpt_05_slash | anomaly | **TP** | 1/1 | 6 | Scratch on seat., A small object appears on the right side of the seat. It looks like a phone or wallet., torn seat., A small object on floor., Paper wrapper., small black object., bottle. |
| gpt_07_multi | anomaly | **TP** | 4/4 | 3 | Black backpack and plastic bottle., small black object., Graffiti: Hobag, Damage / vandalism: torn seat., plastic bottle., Bottle and stain., can. |
| gpt_08_phone_tag | anomaly | **TP** | 2/2 | 3 | Graffiti: "REK", small black object., small green object., A torn seat panel., Seat number 17. |
| gpt_09_litter | anomaly | **TP** | 1/1 | 3 | Litter left behind: can, Two small rectangular objects., can., Litter left behind: can, Small object on floor., paper wrapper., Paper wrapper., bottle. |
| gpt_10_litter | anomaly | **TP** | 1/1 | 2 | Litter left behind: can, litter left behind: can., A small black object appears. |
| gpt_11_crowd | anomaly | **TP** | 4/4 | 5 | Black backpack and bottle., Black backpack and bottle., Graffiti: Hobeg, Scratch on seat., plastic bottle., Graffiti: painted letters., backpack., small object under seat., paper wrapper. |
| real_f0037 | anomaly | **TP** | 2/4 | 4 | Small rectangular object., backpack., A small object on the floor., small object under seat., small black object., small object under seat. |
| real_f0053 | anomaly | **TP** | 3/4 | 3 | backpack., A small object on the floor near seat., Digital display., Graffiti: scribble mark., small object., graffiti: painted letters. |
| real_f0100 | anomaly | **TP** | 2/4 | 5 | small object under seat., Backpack., torn seat., phone., Graffiti: painted letters., phone., litter left behind: can. |
| real_f0112 | anomaly | **TP** | 3/4 | 5 | small black object., Black backpack., A small object on seat., Black object., small object under seat., Paper wrapper on floor., Graffiti: painted letters, Graffiti: blue scribble. |
| real_f0205 | anomaly | **TP** | 2/2 | 7 | Clothing item left behind., backpack., litter left behind: paper wrapper., Green wrapper., phone., small object under seat., small object under seat., A small object resembling a bottle., can. |
| real_f0219 | anomaly | **TP** | 2/2 | 7 | Clothing item left behind., backpack., backpack., small object under seat., Black cylindrical object., torn seat., Graffiti: painted letters., Paper wrapper., small object under seat. |
| variant_01 | anomaly | **TP** | 6/7 | 5 | Backpack on seat, Plastic bottle lying down., Torn seat material., Graffiti: "KEPP", Graffiti: bird drawing, small blue object., Small object under seat., can., water stain., A small object appears on the right side of the image that is not present on the left side. It looks like a piece of debris or trash. |
| 3333_cam54_083517 | anomaly | **FN** | 0/1 | 0 | - |
| 1760_cam04_t120 | clean | **FP** | - | 5 | small white mark., small object under seat., small object., Paper wrapper., Sticker on wall. |
| 1760_cam04_t220 | clean | **FP** | - | 6 | small object under seat., Red button anomaly., Graffiti: painted letters., Cable connected to device., Graffiti: painted letters., small object under seat. |
| 1760_cam04_t320 | clean | **FP** | - | 4 | Yellow device attached to seat., yellow cloud., Small object under seat., small object. |
| 1760_cam04_t420 | clean | **FP** | - | 5 | A small white object appears on the right side of the image. It looks like a piece of paper or a sticker., small object under seat., small object under seat., A small object on the floor., A small black object appears on the floor near the pole. |
| 1760_cam04_t520 | clean | **FP** | - | 1 | Red object attached to pole. |
| 1760_cam06_t120 | clean | **FP** | - | 2 | baby carriage icon., small object under seat. |
| 1760_cam06_t220 | clean | **FP** | - | 4 | shadow of object., torn seat., small object under seat., Graffiti: painted letters |
| 1760_cam06_t320 | clean | **FP** | - | 4 | Paper bag., Paper wrapper., Paper wrapper., a small object under seat. |
| 1760_cam06_t420 | clean | **FP** | - | 9 | Paper wrapper., Reflection of light., Sticker or decal., Graffiti: painted letters, small object under seat., Green bag., Graffiti: painted letters., small object under seat., paper wrapper. |
| 1760_cam06_t520 | clean | **FP** | - | 2 | A small object on the floor., small object under seat. |
| 1760_cam06_t570 | clean | **FP** | - | 4 | Paper wrapper., Paper wrapper, Paper wrapper., A small object under seat. |
| 1760_cam13_t120 | clean | **FP** | - | 1 | small object under seat. |
| 1760_cam13_t220 | clean | **FP** | - | 3 | A small object on the floor., yellow cloth., small object under seat. |
| 1760_cam13_t320 | clean | **FP** | - | 11 | a small object under seat., Paper wrapper., paper wrapper., Yellow object., Reflection of light., Graffiti: yellow scribble., Paper wrapper., small object under seat., Snow accumulation., small object under seat., Graffiti: painted letters. |
| 1760_cam13_t420 | clean | **FP** | - | 9 | seat damage., small yellow object., Vent cover., Paper wrapper., Paper wrapper., paper wrapper., Paper wrapper., paper wrapper., A small object under seat. |
| 1760_cam13_t520 | clean | **FP** | - | 3 | small object under seat., bottle., phone. |
| 1760_cam13_t570 | clean | **FP** | - | 2 | bottle., Graffiti: yellow letters |
| 3333_cam52_ref_t120_clean | clean | **FP** | - | 10 | phone., Small object under seat., A small object on the floor., A small object under seat., small object under seat., Paper wrapper., Paper wrapper., bottle., A small object on floor., Graffiti: painted letters |
| 3333_cam53_085954_clean | clean | **FP** | - | 7 | small object under seat., footprints., bottle., Paper wrapper., cable., small object under seat., small object on floor. |
| 3333_cam54_085954_clean | clean | **FP** | - | 7 | small object under seat., Shadow of object., small object under seat., litter left behind: can., Reflection of light., A small object on the floor., can. |
| 3333_cam54_ref_t120_clean | clean | **FP** | - | 6 | shadow of pole., Black object under seat., Paper wrapper., A small object on the floor., small object under seat., Graffiti: scribble on seat. |
| 3333_cam55_085954_clean | clean | **FP** | - | 10 | A small black object appears on the right side of the image, which is not present on the left side. It looks like a piece of debris or trash., small object under seat., small black object., paper wrapper., A small object on the floor., can., Paper wrapper., Small object under seat., small object under seat., small object under seat. |
| 3333_cam55_ref_t120_clean | clean | **FP** | - | 6 | small object under seat., Paper wrapper., can., A small object appears on the right side of the seat., small object under seat., Paper wrapper. |
| neg_gpt_06_clean | clean | **FP** | - | 4 | small object under seat., small dark mark., torn seat., paper wrapper. |
| neg_v1_f0151 | clean | **FP** | - | 3 | blue light anomaly., Graffiti: painted letters., small object under seat. |
| neg_v1_f0241 | clean | **FP** | - | 3 | A small object on the floor near pole., small object under seat., bottle. |
| neg_v2_f0001_person | clean | **FP** | - | 2 | orange cloth draped over seat., Blue object on floor. |
| neg_v3_f0001 | clean | **FP** | - | 5 | Black bag on floor., Black bag., small object under seat., Graffiti: painted letters, red object. |
| neg_v4_f0004 | clean | **FP** | - | 8 | can., Red scribble., small object under seat., A small object on the floor., Paper wrapper., Graffiti: painted letters., torn seat., A small object on the floor near seat. |
| neg_v4_f0016 | clean | **FP** | - | 9 | black bag., A small black object., small red object., small object under seat., a small object under seat., can., Graffiti: scribble on seat, phone., small object under seat. |
| neg_v4_f0022 | clean | **FP** | - | 4 | small object under seat., torn seat., Small object, possibly bottle., small object under seat. |
| 1760_cam04_t570 | clean | **TN** | - | 0 | - |
| 3333_cam53_ref_t120_clean | clean | **TN** | - | 0 | - |
| neg_real_ref_self | clean | **TN** | - | 0 | - |
| neg_v1_f0181 | clean | **TN** | - | 0 | - |
| neg_v1_f0211 | clean | **TN** | - | 0 | - |
| neg_variant_ref_self | clean | **TN** | - | 0 | - |

Annotated images: `benchmark/runs/cli-latest/annotated/<id>.jpg` (blue = ground-truth boxes, green = correct detections, red = false-positive boxes). Raw results: `benchmark/runs/judge-qwen25vl-7b-balanced/results.json`.

---

\* **3333** is a placeholder, not the tram's real fleet number - the vehicle number of the 2026-08-11 capture is unknown. It was called 39T before, but 39T is the Škoda type, which tram 1760 shares.
