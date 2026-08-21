# Benchmark run - ground_truth · localize

**Run:** `20260817-161119-loc-5282`  
**Status:** completed (21/21 cases)  
**Dataset:** `ground_truth` (digest `1930a2996b23`)  
**Localizer:** `photo` - no VLM call.  
**Wall-clock:** 0.1 min.

## Localization only (upper bound on end-to-end recall)

- Instances localized: **17 / 24** → recall 0.708 (strict IoU≥0.3: 5 / 24 = 0.208)
- Regions proposed: **266** (166 on anomaly frames, 100 on 7 clean frames)
- Biggest box: 911,360 px (the blob canary - a frame-sized box hits every instance leniently while boxing nothing)

| type | localized | recall |
|---|---|---|
| litter | 4 / 5 | 0.80 |
| object | 13 / 19 | 0.68 |

## Per-case

| id | truth | regions | instances localized | biggest box |
|---|---|---|---|---|
| 3333_cam52_084637\* | anomaly | 11 | 2/2 | 558,480 |
| 3333_cam52_085124 | anomaly | 18 | 1/1 | 495,216 |
| 3333_cam53_083517 | anomaly | 11 | 1/2 | 143,488 |
| 3333_cam53_084021 | anomaly | 13 | 1/2 | 144,768 |
| 3333_cam53_084637 | anomaly | 6 | 1/1 | 613,440 |
| 3333_cam53_085124 | anomaly | 15 | 0/1 | 119,888 |
| 3333_cam54_083517 | anomaly | 10 | 0/1 | 130,720 |
| 3333_cam54_084021 | anomaly | 11 | 1/1 | 840,960 |
| 3333_cam54_084637 | anomaly | 18 | 2/2 | 252,880 |
| 3333_cam54_085124 | anomaly | 12 | 1/3 | 102,272 |
| 3333_cam55_083517 | anomaly | 8 | 2/2 | 911,360 |
| 3333_cam55_084021 | anomaly | 6 | 2/2 | 911,360 |
| 3333_cam55_084637 | anomaly | 17 | 2/2 | 117,952 |
| 3333_cam55_085124 | anomaly | 10 | 1/2 | 86,640 |
| 3333_cam53_085954_clean | clean | 29 | - | 276,336 |
| 3333_cam54_085954_clean | clean | 15 | - | 419,136 |
| 3333_cam55_085954_clean | clean | 15 | - | 207,264 |
| 3333_cam52_ref_t120_clean | clean | 3 | - | 792,000 |
| 3333_cam53_ref_t120_clean | clean | 13 | - | 264,576 |
| 3333_cam54_ref_t120_clean | clean | 10 | - | 228,288 |
| 3333_cam55_ref_t120_clean | clean | 15 | - | 622,080 |

---

\* **3333** is a placeholder, not the tram's real fleet number - the vehicle number of the 2026-08-11 capture is unknown. It was called 39T before, but 39T is the Škoda type, which tram 1760 shares.
