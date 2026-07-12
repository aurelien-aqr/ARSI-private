# vlm_05 reference-diff — anomaly detection benchmark

**Status:** COMPLETE  
**Model:** `qwen3-vl:8b-instruct` (Ollama)  
**Decision rule:** frame flagged if the VLM keeps ≥1 region (`filter` mode) after dropping person/"disappeared" labels and de-duplicating overlapping boxes.  
**Diff / region params:** DIFF_THRESHOLD=40, BLUR_RADIUS=3, MIN_AREA=500, MAX_AREA=400000, MAX_REGIONS=25.  
**Wall-clock:** 0.2 min.

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

- Cases: **29**  (TP=17, FP=7, TN=5, FN=0)
- **Accuracy** 0.759 · **Precision** 0.708 · **Recall** 1.000 · **Specificity** 0.417 · **F1** 0.829

| | predicted anomaly | predicted clean |
|---|---|---|
| **actual anomaly** | TP = 17 | FN = 0 |
| **actual clean**   | FP = 7 | TN = 5 |

## 2) Object-level (did we box each real anomaly?)

- Instances detected: **44 / 45** → **object recall 0.978** (strict IoU≥0.3: 37 / 45 = 0.822)
- False-positive regions (kept boxes matching no real anomaly): **118** of 183 kept → region precision 0.355
- All VLM verdicts served from cache (0 new calls).

| type | instances detected | recall |
|---|---|---|
| object | 32 / 33 | 0.97 |
| graffiti | 6 / 6 | 1.00 |
| damage | 4 / 4 | 1.00 |
| litter | 2 / 2 | 1.00 |

| source | cases | instances detected | FP regions |
|---|---|---|---|
| gpt | 11 | 20 / 20 | 53 |
| real | 15 | 20 / 21 | 57 |
| self | 2 | 0 / 0 | 0 |
| variant | 1 | 4 / 4 | 8 |

## Per-case results

| id | truth | frame | instances hit | FP boxes | VLM kept-labels |
|---|---|---|---|---|---|
| gpt_01_suitcase | anomaly | **TP** | 1/1 | 3 | black suitcase, black suitcase, black suitcase with extended handle, black suitcase appears, metallic object corner, Blue seat cushion slightly shifted, Black suitcase corner, Black suitcase corner |
| gpt_02_multi | anomaly | **TP** | 4/4 | 8 | backpack on seat, brown paper bag, brown paper bag, phone on seat, bottle on seat, bottle on seat, Black cable snake-like, brown paper bag, black bag on seat, bottle on seat, Black backpack strap, Black bag on seat, phone on seat |
| gpt_03_faint_tag | anomaly | **TP** | 1/1 | 4 | XRP graffiti on panel, Black vertical stripe pattern, Blue light glow on right side, Small yellow dot on seat back, Black cable snake-like |
| gpt_04_graffiti | anomaly | **TP** | 1/1 | 5 | graffiti on wall, graffiti on window, Colorful abstract pattern, Snake on right side, Black vertical stripe pattern, Blue seat cushion slightly shifted, Red mark on seat back, Vertical black streaks on wall |
| gpt_05_slash | anomaly | **TP** | 1/1 | 3 | White patch on seat, torn seat cushion, Bird feather on window, Purple light glow |
| gpt_07_multi | anomaly | **TP** | 4/4 | 10 | bottle on floor, backpack on seat, bottle on floor, black backpack appears, graffiti on panel, Small metallic object near right edge, Blue seat cushion visible, black bag on seat, torn seat cushion, graffiti on panel, Black cable snake-like, Blue seat cushion slightly lower, black backpack on seat, torn seat cushion, Blue seat cushion slightly shifted |
| gpt_08_phone_tag | anomaly | **TP** | 2/2 | 3 | graffiti on wall, Black phone on seat, graffiti on the wall, Blue seat cushion slightly shifted, Black phone on seat |
| gpt_09_litter | anomaly | **TP** | 1/1 | 1 | Cylindrical metal can, Cylindrical green can, SIM card on floor, Small white rectangular object, Small stone on floor, Small white sticker on vent |
| gpt_10_litter | anomaly | **TP** | 1/1 | 2 | a fallen soda can, Two cans and a crumpled wrapper, Black cable snake-like, Rolled-up banknote on floor |
| gpt_11_crowd | anomaly | **TP** | 4/4 | 12 | black backpack on seat, bottle on floor, bottle on floor, Graffiti on wall panel, bottle on floor, graffiti on panel, bottle on floor, graffiti on wall, Torn seat cushion with yellow patch, Dark stain on floor, graffiti on wall, Black cable visible on right, torn seat cushion, graffiti on surface, black object partially visible, Black object on seat, black jacket draped over seat, beige cloth hanging down, torn seat surface, green bag visible |
| real_f0037 | anomaly | **TP** | 4/4 | 2 | black backpack appears, backpack on seat, backpack on seat, phone on seat, wallet on seat, Blue bag on floor |
| real_f0053 | anomaly | **TP** | 4/4 | 7 | black backpack on seat, backpack on seat, backpack on floor, black bag on floor, wallet on seat, Digital display device, wallet on seat, Small dark object under seat, Small white mark on metal bar, Brown rectangular mark on blue surface, Black wallet on seat |
| real_f0100 | anomaly | **TP** | 4/4 | 4 | backpack on seat, Black bag on seat, Black backpack on floor, wallet on seat, Black object on seat, Phone on seat, Small green object under seat, Phone on seat |
| real_f0112 | anomaly | **TP** | 4/4 | 10 | Black backpack on seat, backpack on floor, Black backpack on seat, black backpack on floor, backpack on right seat, Black bag on seat, green object behind seat, Black phone on seat, phone on seat, phone on seat, Two small metal rivets, Purple sticker on metal frame, phone on seat, Black vertical stripe pattern |
| real_f0205 | anomaly | **TP** | 2/2 | 3 | Grey jacket on seat, backpack on seat, new sticker on frame, backpack appears on right, Purple object under seat, Black vertical stripe on right side of pole. |
| real_f0219 | anomaly | **TP** | 2/3 | 6 | jacket draped over seat, jacket draped over seat, black backpack appears, black bag on seat, black backpack handle, Black chain-like object, Black bag under seat, Metallic bar slightly bent, Red object stuck in gap |
| variant_01 | anomaly | **TP** | 4/4 | 8 | bottle on floor, backpack on seat, backpack on seat, graffiti on wall, torn blue seat cover, bottle on floor, bottle on railing, backpack on seat, bottle on floor, White plastic bag visible, torn seat cushion, torn seat cushion, backpack hanging on seat, Blue seat torn fabric |
| neg_gpt_06_clean | clean | **FP** | — | 2 | dark stain on seat, Blue seat cushion slightly shifted |
| neg_v1_f0181 | clean | **FP** | — | 1 | Black circular mark on surface |
| neg_v2_f0001_person | clean | **FP** | — | 3 | laptop screen visible, Red circular sticker added, Red circular sticker added |
| neg_v3_f0001 | clean | **FP** | — | 6 | black backpack appears, Furry animal on seat, Dark fur on right side, Dark furry animal present, black backpack handle, Black bag handle visible |
| neg_v4_f0004 | clean | **FP** | — | 1 | Black object on railing |
| neg_v4_f0016 | clean | **FP** | — | 8 | Black bag on seat, Black bag on seat, Small white object on right seat, Black cable snagged on pole, small dark object under seat, graffiti MD written, Black object hanging down, Black chain-like object |
| neg_v4_f0022 | clean | **FP** | — | 6 | Black bag on seat, Small white box on floor, Black bag on seat, Screen displaying image, White sticker on seat back, Small black object on panel |
| neg_real_ref_self | clean | **TN** | — | 0 | — |
| neg_v1_f0151 | clean | **TN** | — | 0 | — |
| neg_v1_f0211 | clean | **TN** | — | 0 | — |
| neg_v1_f0241 | clean | **TN** | — | 0 | — |
| neg_variant_ref_self | clean | **TN** | — | 0 | — |

Annotated images: `benchmark/annotated/<id>.jpg` (blue = ground-truth boxes, green = correct detections, red = false-positive boxes). Raw results: `benchmark/results.json`.
