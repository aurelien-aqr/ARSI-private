# vlm_05 reference-diff - anomaly detection benchmark

**Status:** COMPLETE  
**Model:** `haervwe/GLM-4.6V-Flash-9B:latest` (Ollama)  
**Decision rule:** frame flagged if the VLM keeps ≥1 region (`filter` mode) after dropping person/"disappeared" labels and de-duplicating overlapping boxes.  
**Diff / region params:** DIFF_THRESHOLD=40, BLUR_RADIUS=3, MIN_AREA=500, MAX_AREA=400000, MAX_REGIONS=25.  
**Wall-clock:** 6.7 min.

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

- Cases: **68**  (TP=27, FP=0, TN=37, FN=4)
- **Accuracy** 0.941 · **Precision** 1.000 · **Recall** 0.871 · **Specificity** 1.000 · **F1** 0.931

| | predicted anomaly | predicted clean |
|---|---|---|
| **actual anomaly** | TP = 27 | FN = 4 |
| **actual clean**   | FP = 0 | TN = 37 |

## 2) Object-level (did we box each real anomaly?)

- Instances detected: **55 / 73** → **object recall 0.753** (strict IoU≥0.3: 39 / 73 = 0.534)
- False-positive regions (kept boxes matching no real anomaly): **30** of 98 kept → region precision 0.694
- Uncached VLM calls this run: 601, mean 0.6 s/call

| type | instances detected | recall |
|---|---|---|
| object | 37 / 53 | 0.70 |
| graffiti | 6 / 7 | 0.86 |
| damage | 6 / 6 | 1.00 |
| litter | 6 / 7 | 0.86 |

| source | cases | instances detected | FP regions |
|---|---|---|---|
| gpt | 11 | 20 / 20 | 10 |
| real | 54 | 29 / 46 | 16 |
| self | 2 | 0 / 0 | 0 |
| variant | 1 | 6 / 7 | 4 |

## Per-case results

| id | truth | frame | instances hit | FP boxes | VLM kept-labels |
|---|---|---|---|---|---|
| 3333_cam52_085124\* | anomaly | **TP** | 0/2 | 1 | phone |
| 3333_cam53_083517 | anomaly | **TP** | 1/2 | 1 | yellow bag on seat, a yellow bag on the seat |
| 3333_cam53_084021 | anomaly | **TP** | 1/2 | 1 | yellow item on seat, yellow bag on seat |
| 3333_cam54_084021 | anomaly | **TP** | 1/1 | 0 | a crumpled paper wrapper, plastic bag |
| 3333_cam54_084637 | anomaly | **TP** | 2/2 | 3 | cloth on seat, cloth on seat, laptop, laptop, a plastic bag |
| 3333_cam54_085124 | anomaly | **TP** | 1/3 | 3 | a crumpled bag with green items, litter (trash bag), green bottle, a bag and food wrapper |
| 3333_cam55_083517 | anomaly | **TP** | 2/2 | 0 | laptop on left seat, laptop |
| 3333_cam55_084021 | anomaly | **TP** | 2/2 | 0 | a crumpled paper wrapper |
| 3333_cam55_084637 | anomaly | **TP** | 2/2 | 2 | laptop and cloth, cloth on seat, laptop, laptop |
| 3333_cam55_085124 | anomaly | **TP** | 1/2 | 0 | a green bottle |
| gpt_01_suitcase | anomaly | **TP** | 1/1 | 0 | black suitcase, black suitcase, suitcase, black suitcase |
| gpt_02_multi | anomaly | **TP** | 4/4 | 2 | black backpack, brown paper bag, phone, bottle, plastic bottle, brown paper bag, plastic bottle |
| gpt_03_faint_tag | anomaly | **TP** | 1/1 | 0 | graffiti (letters XRP) |
| gpt_04_graffiti | anomaly | **TP** | 1/1 | 0 | graffiti |
| gpt_05_slash | anomaly | **TP** | 1/1 | 0 | torn seat |
| gpt_07_multi | anomaly | **TP** | 4/4 | 1 | black backpack, black backpack, graffiti "HOBBO", torn seat, torn seat |
| gpt_08_phone_tag | anomaly | **TP** | 2/2 | 1 | graffiti "ZONK", a black wallet/package, graffiti "Zebr" |
| gpt_09_litter | anomaly | **TP** | 1/1 | 0 | a can, a can, a small white wrapper, wrapper |
| gpt_10_litter | anomaly | **TP** | 1/1 | 1 | a can, litter (can, wrapper, paper), two pieces of litter (a tube and a crumpled item). |
| gpt_11_crowd | anomaly | **TP** | 4/4 | 5 | water bottle, black backpack, black backpack, torn seat, black backpack, graffiti "Hob88", plastic bottle, graffiti, torn seat, graffiti "HOPE", torn seat |
| real_f0037 | anomaly | **TP** | 3/4 | 0 | backpack, phone, wallet |
| real_f0053 | anomaly | **TP** | 3/4 | 2 | black backpack, black backpack, wallet, phone, wallet |
| real_f0100 | anomaly | **TP** | 3/4 | 0 | black backpack, wallet, a black wallet |
| real_f0112 | anomaly | **TP** | 4/4 | 0 | black backpack, black backpack, phone, phone |
| real_f0205 | anomaly | **TP** | 1/2 | 1 | jacket on seat, jacket on seat |
| real_f0219 | anomaly | **TP** | 2/2 | 2 | jacket on seat, clothes on seat, backpack, backpack on seat, backpack |
| variant_01 | anomaly | **TP** | 6/7 | 4 | bottle, backpack, backpack, graffiti "keep", torn seat fabric, bottle, bottle, backpack, plastic bottle, torn seat, torn seat, backpack |
| 3333_cam52_084637 | anomaly | **FN** | 0/3 | 0 | - |
| 3333_cam53_084637 | anomaly | **FN** | 0/1 | 0 | - |
| 3333_cam53_085124 | anomaly | **FN** | 0/1 | 0 | - |
| 3333_cam54_083517 | anomaly | **FN** | 0/1 | 0 | - |
| 1760_cam04_t120 | clean | **TN** | - | 0 | - |
| 1760_cam04_t220 | clean | **TN** | - | 0 | - |
| 1760_cam04_t320 | clean | **TN** | - | 0 | - |
| 1760_cam04_t420 | clean | **TN** | - | 0 | - |
| 1760_cam04_t520 | clean | **TN** | - | 0 | - |
| 1760_cam04_t570 | clean | **TN** | - | 0 | - |
| 1760_cam06_t120 | clean | **TN** | - | 0 | - |
| 1760_cam06_t220 | clean | **TN** | - | 0 | - |
| 1760_cam06_t320 | clean | **TN** | - | 0 | - |
| 1760_cam06_t420 | clean | **TN** | - | 0 | - |
| 1760_cam06_t520 | clean | **TN** | - | 0 | - |
| 1760_cam06_t570 | clean | **TN** | - | 0 | - |
| 1760_cam13_t120 | clean | **TN** | - | 0 | - |
| 1760_cam13_t220 | clean | **TN** | - | 0 | - |
| 1760_cam13_t320 | clean | **TN** | - | 0 | - |
| 1760_cam13_t420 | clean | **TN** | - | 0 | - |
| 1760_cam13_t520 | clean | **TN** | - | 0 | - |
| 1760_cam13_t570 | clean | **TN** | - | 0 | - |
| 3333_cam52_ref_t120_clean | clean | **TN** | - | 0 | - |
| 3333_cam53_085954_clean | clean | **TN** | - | 0 | - |
| 3333_cam53_ref_t120_clean | clean | **TN** | - | 0 | - |
| 3333_cam54_085954_clean | clean | **TN** | - | 0 | - |
| 3333_cam54_ref_t120_clean | clean | **TN** | - | 0 | - |
| 3333_cam55_085954_clean | clean | **TN** | - | 0 | - |
| 3333_cam55_ref_t120_clean | clean | **TN** | - | 0 | - |
| neg_gpt_06_clean | clean | **TN** | - | 0 | - |
| neg_real_ref_self | clean | **TN** | - | 0 | - |
| neg_v1_f0151 | clean | **TN** | - | 0 | - |
| neg_v1_f0181 | clean | **TN** | - | 0 | - |
| neg_v1_f0211 | clean | **TN** | - | 0 | - |
| neg_v1_f0241 | clean | **TN** | - | 0 | - |
| neg_v2_f0001_person | clean | **TN** | - | 0 | - |
| neg_v3_f0001 | clean | **TN** | - | 0 | - |
| neg_v4_f0004 | clean | **TN** | - | 0 | - |
| neg_v4_f0016 | clean | **TN** | - | 0 | - |
| neg_v4_f0022 | clean | **TN** | - | 0 | - |
| neg_variant_ref_self | clean | **TN** | - | 0 | - |

Annotated images: `benchmark/runs/cli-latest/annotated/<id>.jpg` (blue = ground-truth boxes, green = correct detections, red = false-positive boxes). Raw results: `benchmark/runs/loc-shipped-all/results.json`.

---

\* **3333** is a placeholder, not the tram's real fleet number - the vehicle number of the 2026-08-11 capture is unknown. It was called 39T before, but 39T is the Škoda type, which tram 1760 shares.
