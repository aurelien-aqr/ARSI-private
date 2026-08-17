# Benchmark run - ground_truth · localize

**Run:** `20260817-153920-loc-b46f`  
**Status:** completed (29/29 cases)  
**Dataset:** `ground_truth` (digest `1930a2996b23`)  
**Localizer:** `photo` - no VLM call.  
**Wall-clock:** 0.2 min.

## Localization only (upper bound on end-to-end recall)

- Instances localized: **45 / 45** → recall 1.000 (strict IoU≥0.3: 37 / 45 = 0.822)
- Regions proposed: **559** (389 on anomaly frames, 170 on 12 clean frames)
- Biggest box: 734,400 px (the blob canary - a frame-sized box hits every instance leniently while boxing nothing)

| type | localized | recall |
|---|---|---|
| damage | 4 / 4 | 1.00 |
| graffiti | 6 / 6 | 1.00 |
| litter | 2 / 2 | 1.00 |
| object | 33 / 33 | 1.00 |

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
