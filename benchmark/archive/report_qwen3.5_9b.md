# vlm_05 reference-diff — anomaly detection benchmark

**Status:** COMPLETE  
**Model:** `qwen3.5:9b` (Ollama)  
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

- Cases: **29**  (TP=17, FP=3, TN=9, FN=0)
- **Accuracy** 0.897 · **Precision** 0.850 · **Recall** 1.000 · **Specificity** 0.750 · **F1** 0.919

| | predicted anomaly | predicted clean |
|---|---|---|
| **actual anomaly** | TP = 17 | FN = 0 |
| **actual clean**   | FP = 3 | TN = 9 |

## 2) Object-level (did we box each real anomaly?)

- Instances detected: **44 / 45** → **object recall 0.978** (strict IoU≥0.3: 37 / 45 = 0.822)
- False-positive regions (kept boxes matching no real anomaly): **51** of 113 kept → region precision 0.549
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
| real | 15 | 20 / 21 | 17 |
| self | 2 | 0 / 0 | 0 |
| variant | 1 | 4 / 4 | 6 |

## Per-case results

| id | truth | frame | instances hit | FP boxes | VLM kept-labels |
|---|---|---|---|---|---|
| gpt_01_suitcase | anomaly | **TP** | 1/1 | 0 | black suitcase on floor, black suitcase on floor, black suitcase on floor, black suitcase on floor |
| gpt_02_multi | anomaly | **TP** | 4/4 | 7 | black backpack on seat, brown paper bag on floor, brown paper bag on floor, brown bag on floor, plastic bottle on seat cushion, plastic water bottle on seat, brown paper bag on floor, black bag on seat, plastic bottle on seat, black bag on seat, black bag on seat, phone on blue surface |
| gpt_03_faint_tag | anomaly | **TP** | 1/1 | 0 | graffiti "XRP" on panel |
| gpt_04_graffiti | anomaly | **TP** | 1/1 | 1 | graffiti on wall panel, blue graffiti tag on wall, graffiti scribbles on wall panel |
| gpt_05_slash | anomaly | **TP** | 1/1 | 2 | tear on blue seat cushion, torn seat cushion hole, torn blue panel surface |
| gpt_07_multi | anomaly | **TP** | 4/4 | 6 | backpack on seat, bottle on floor, black backpack on seat, backpack on seat, bottle on floor, black bag on seat, graffiti tag on panel, small object on floor right side, black bag on seat, torn seat cushion exposed foam, graffiti tag on panel surface, black strap on seat cushion, torn seat fabric exposed foam |
| gpt_08_phone_tag | anomaly | **TP** | 2/2 | 2 | graffiti tag on wall panel, black phone on seat, graffiti tag on wall panel, black phone on seat |
| gpt_09_litter | anomaly | **TP** | 1/1 | 0 | littered aluminum can on floor, green soda can on floor, small rectangular object on floor, paper wrapper on floor, small rock on floor |
| gpt_10_litter | anomaly | **TP** | 1/1 | 1 | littered can and wrapper on floor, Litter on floor (can + wrapper), littered wrapper and tube |
| gpt_11_crowd | anomaly | **TP** | 4/4 | 8 | black backpack on seat, backpack on seat, bottle on floor, black backpack on seat, torn seat cushion; graffiti on wall, black backpack on seat, black bag on seat, graffiti tag on panel, plastic bottle on floor, graffiti tag on wall panel, torn seat cushion exposed foam, black graffiti tag on wall panel, torn seat cushion exposed foam, graffiti tag on panel surface, black object (possibly bag) on seat edge, torn seat fabric exposing foam underneath |
| real_f0037 | anomaly | **TP** | 4/4 | 1 | black bag on seat, black backpack on seat, black bag on seat, phone on seat cushion, black wallet on seat |
| real_f0053 | anomaly | **TP** | 4/4 | 5 | black bag on seat, black bag on seat, black backpack on floor, black bag on floor, black wallet on seat, phone on seat cushion, black wallet on seat, black object on seat (phone?), forgotten wallet on seat |
| real_f0100 | anomaly | **TP** | 4/4 | 2 | black bag on seat, black bag on seat, black backpack on floor, wallet on seat cushion, black object on seat, torn seat fabric on right side |
| real_f0112 | anomaly | **TP** | 4/4 | 2 | black backpack on floor, black backpack on floor, black strap on left seat cushion, black phone on seat backrest, black phone on floor, black strap on seat edge |
| real_f0205 | anomaly | **TP** | 2/2 | 1 | gray jacket on seat, black backpack on seat, grey bag on seat, black backpack on seat |
| real_f0219 | anomaly | **TP** | 2/3 | 2 | forgotten jacket on seat, gray jacket on seat, black backpack on seat, black bag on seat, black backpack on seat |
| variant_01 | anomaly | **TP** | 4/4 | 6 | plastic bottle on floor, backpack on seat (left side), gray backpack on seat, graffiti tag "KEEP" on wall, torn seat fabric exposed foam, bottle on the floor, plastic bottle on floor, backpack on blue seat cushion, plastic bottle on floor, torn seat cushion, torn seat cushion hole, gray backpack on seat |
| neg_gpt_06_clean | clean | **FP** | — | 1 | faint dark stain on seat cushion |
| neg_v3_f0001 | clean | **FP** | — | 3 | black backpack on seat, black bag on seat, backpack on seat |
| neg_v4_f0022 | clean | **FP** | — | 1 | graffiti on seat backrest |
| neg_real_ref_self | clean | **TN** | — | 0 | — |
| neg_v1_f0151 | clean | **TN** | — | 0 | — |
| neg_v1_f0181 | clean | **TN** | — | 0 | — |
| neg_v1_f0211 | clean | **TN** | — | 0 | — |
| neg_v1_f0241 | clean | **TN** | — | 0 | — |
| neg_v2_f0001_person | clean | **TN** | — | 0 | — |
| neg_v4_f0004 | clean | **TN** | — | 0 | — |
| neg_v4_f0016 | clean | **TN** | — | 0 | — |
| neg_variant_ref_self | clean | **TN** | — | 0 | — |

Annotated images: `benchmark/annotated/<id>.jpg` (blue = ground-truth boxes, green = correct detections, red = false-positive boxes). Raw results: `benchmark/results.json`.
