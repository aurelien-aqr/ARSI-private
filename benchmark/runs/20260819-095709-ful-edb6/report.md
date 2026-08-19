# Benchmark run - 1760_negatives · full

**Run:** `20260819-095709-ful-edb6`  
**Status:** completed (18/18 cases)  
**Dataset:** `1760_negatives` (digest `6f96fd019855`)  
**Pipeline:** `vlm_05` · localizer `photo` · judge `haervwe/GLM-4.6V-Flash-9B:latest` · prompt `conservative`  
**Wall-clock:** 2.3 min.

## 1) Frame-level (binary: is the frame anomalous?)

- Cases: **18**  (TP=0, FP=0, TN=18, FN=0)
- **Accuracy** 1.000 · **Precision** 0.000 · **Recall** 0.000 · **Specificity** 1.000 · **F1** 0.000

| | predicted anomaly | predicted clean |
|---|---|---|
| **actual anomaly** | TP = 0 | FN = 0 |
| **actual clean**   | FP = 0 | TN = 18 |

## 2) Object-level (did we box each real anomaly?)

- Instances detected: **0 / 0** → **object recall 0.000** (strict IoU≥0.3: 0 / 0 = 0.000)
- False-positive regions (kept boxes matching no real anomaly): **0** of 0 kept → region precision 0.000
- Regions sent to the judge: 158 - **158 fresh VLM call(s)**, 0 served from cache.

| source | cases | instances detected | FP regions |
|---|---|---|---|
| real | 18 | 0 / 0 | 0 |

## Per-case results

| id | truth | frame | instances hit | FP boxes | kept labels |
|---|---|---|---|---|---|
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
