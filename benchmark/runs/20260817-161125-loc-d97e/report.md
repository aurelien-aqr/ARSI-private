# Benchmark run - ground_truth · localize

**Run:** `20260817-161125-loc-d97e`  
**Status:** completed (50/50 cases)  
**Dataset:** `ground_truth` (digest `1930a2996b23`)  
**Localizer:** `photo` - no VLM call.  
**Wall-clock:** 0.3 min.

## Localization only (upper bound on end-to-end recall)

- Instances localized: **62 / 69** → recall 0.899 (strict IoU≥0.3: 42 / 69 = 0.609)
- Regions proposed: **825** (555 on anomaly frames, 270 on 19 clean frames)
- Biggest box: 911,360 px (the blob canary - a frame-sized box hits every instance leniently while boxing nothing)

| type | localized | recall |
|---|---|---|
| damage | 4 / 4 | 1.00 |
| graffiti | 6 / 6 | 1.00 |
| litter | 6 / 7 | 0.86 |
| object | 46 / 52 | 0.88 |

## Per-case

| id | truth | regions | instances localized | biggest box |
|---|---|---|---|---|
| real_f0037 | anomaly | 18 | 4/4 | 34,592 |
| real_f0053 | anomaly | 33 | 4/4 | 37,376 |
| real_f0100 | anomaly | 31 | 4/4 | 25,296 |
| real_f0112 | anomaly | 34 | 4/4 | 204,240 |
| real_f0205 | anomaly | 32 | 2/2 | 299,008 |
| real_f0219 | anomaly | 34 | 3/3 | 87,040 |
| gpt_01_suitcase | anomaly | 15 | 1/1 | 31,824 |
| gpt_02_multi | anomaly | 21 | 4/4 | 68,608 |
| gpt_03_faint_tag | anomaly | 16 | 1/1 | 91,200 |
| gpt_04_graffiti | anomaly | 18 | 1/1 | 734,400 |
| gpt_05_slash | anomaly | 16 | 1/1 | 98,400 |
| gpt_07_multi | anomaly | 19 | 4/4 | 220,224 |
| gpt_08_phone_tag | anomaly | 17 | 2/2 | 11,968 |
| gpt_09_litter | anomaly | 17 | 1/1 | 6,384 |
| gpt_10_litter | anomaly | 17 | 1/1 | 5,440 |
| gpt_11_crowd | anomaly | 27 | 4/4 | 209,664 |
| variant_01 | anomaly | 24 | 4/4 | 54,896 |
| neg_real_ref_self | clean | 0 | - | 0 |
| neg_variant_ref_self | clean | 0 | - | 0 |
| neg_gpt_06_clean | clean | 17 | - | 79,744 |
| neg_v1_f0151 | clean | 12 | - | 16,128 |
| neg_v1_f0181 | clean | 8 | - | 26,496 |
| neg_v1_f0211 | clean | 6 | - | 18,352 |
| neg_v1_f0241 | clean | 8 | - | 16,576 |
| neg_v2_f0001_person | clean | 14 | - | 27,216 |
| neg_v3_f0001 | clean | 27 | - | 24,544 |
| neg_v4_f0004 | clean | 18 | - | 31,104 |
| neg_v4_f0016 | clean | 29 | - | 561,904 |
| neg_v4_f0022 | clean | 31 | - | 463,104 |
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
| 39T_cam53_085954_clean | clean | 29 | - | 276,336 |
| 39T_cam54_085954_clean | clean | 15 | - | 419,136 |
| 39T_cam55_085954_clean | clean | 15 | - | 207,264 |
| 39T_cam52_ref_t120_clean | clean | 3 | - | 792,000 |
| 39T_cam53_ref_t120_clean | clean | 13 | - | 264,576 |
| 39T_cam54_ref_t120_clean | clean | 10 | - | 228,288 |
| 39T_cam55_ref_t120_clean | clean | 15 | - | 622,080 |
