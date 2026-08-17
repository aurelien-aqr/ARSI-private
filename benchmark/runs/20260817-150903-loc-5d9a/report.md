# Benchmark run — 39T · localize

**Run:** `20260817-150903-loc-5d9a`  
**Status:** completed (21/21 cases)  
**Dataset:** `39T` (digest `d31a057897ee`)  
**Localizer:** `photo` — no VLM call.  
**Wall-clock:** 0.1 min.

## Localization only (upper bound on end-to-end recall)

- Instances localized: **17 / 24** → recall 0.708 (strict IoU≥0.3: 5 / 24 = 0.208)
- Regions proposed: **266** (166 on anomaly frames, 100 on 7 clean frames)
- Biggest box: 911,360 px (the blob canary — a frame-sized box hits every instance leniently while boxing nothing)

| type | localized | recall |
|---|---|---|
| litter | 4 / 5 | 0.80 |
| object | 13 / 19 | 0.68 |

## Per-case

| id | truth | regions | instances localized | biggest box |
|---|---|---|---|---|
| 39T_cam52_084637 | anomaly | 11 | 2/2 | 558,480 |
| 39T_cam52_085124 | anomaly | 18 | 1/1 | 495,216 |
| 39T_cam53_083517 | anomaly | 11 | 1/2 | 143,488 |
| 39T_cam53_084021 | anomaly | 13 | 1/2 | 144,768 |
| 39T_cam53_084637 | anomaly | 6 | 1/1 | 613,440 |
| 39T_cam53_085124 | anomaly | 15 | 0/1 | 119,888 |
| 39T_cam54_083517 | anomaly | 10 | 0/1 | 130,720 |
| 39T_cam54_084021 | anomaly | 11 | 1/1 | 840,960 |
| 39T_cam54_084637 | anomaly | 18 | 2/2 | 252,880 |
| 39T_cam54_085124 | anomaly | 12 | 1/3 | 102,272 |
| 39T_cam55_083517 | anomaly | 8 | 2/2 | 911,360 |
| 39T_cam55_084021 | anomaly | 6 | 2/2 | 911,360 |
| 39T_cam55_084637 | anomaly | 17 | 2/2 | 117,952 |
| 39T_cam55_085124 | anomaly | 10 | 1/2 | 86,640 |
| 39T_cam53_085954_clean | clean | 29 | — | 276,336 |
| 39T_cam54_085954_clean | clean | 15 | — | 419,136 |
| 39T_cam55_085954_clean | clean | 15 | — | 207,264 |
| 39T_cam52_ref_t120_clean | clean | 3 | — | 792,000 |
| 39T_cam53_ref_t120_clean | clean | 13 | — | 264,576 |
| 39T_cam54_ref_t120_clean | clean | 10 | — | 228,288 |
| 39T_cam55_ref_t120_clean | clean | 15 | — | 622,080 |
