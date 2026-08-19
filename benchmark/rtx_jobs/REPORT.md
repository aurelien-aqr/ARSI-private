# RTX Studio jobs — harvested comparison

Every job run on the 3080 Ti workstation, aggregated from the untracked
`data/app/jobs/` tree. `flag` = frames carrying at least one detection.
`(cache)` marks a replay that made no fresh VLM call (median frame < 1 s).

## 1762-2  STAGED: planted phone/wallet/backpack (73 f)

45 jobs

| pipeline | model | prompt | ref | mask | frames | flag | det | s/frame |
|---|---|---|---|---|---:|---:|---:|---:|
| vlm_01 | minicpm-v4.6:latest | default | - | yes | 72 | 72 | 280 | 1.48 |
| vlm_01 | GLM-4.6V-Flash-9B | default | - | yes | 72 | 72 | 155 | 3.65 |
| vlm_01 | GLM-4.6V-Flash-9B | default | - | yes | 72 | 72 | 145 | 3.64 |
| vlm_01 | qwen3.5:9b | default | - | yes | 72 | 72 | 134 | 3.61 |
| vlm_01 | qwen3-vl:8b-instruct | default | - | yes | 72 | 72 | 124 | 3.17 |
| vlm_01 | InternVL3_5:8b | default | - | yes | 72 | 72 | 115 | 2.97 |
| vlm_01 | Cosmos-Reason2-8B-GGUF:Q4_ | default | - | yes | 72 | 72 | 81 | 2.72 |
| vlm_01 | GLM-4.6V-Flash-9B | default | - | - | 72 | 28 | 56 | 7.05 |
| vlm_02 | GLM-4.6V-Flash-9B | default | yes | yes | 72 | 72 | 202 | 3.63 |
| vlm_02 | GLM-4.6V-Flash-9B | default | yes | yes | 72 | 72 | 197 | 3.57 |
| vlm_02 | minicpm-v4.6:latest | default | yes | yes | 72 | 72 | 149 | 2.15 |
| vlm_02 | InternVL3_5:8b | default | yes | yes | 72 | 72 | 122 | 3.9 |
| vlm_02 | qwen3-vl:8b-instruct | default | yes | yes | 72 | 72 | 72 | 1.16 |
| vlm_02 | qwen3.5:9b | default | yes | yes | 72 | 72 | 72 | 1.78 |
| vlm_02 | Cosmos-Reason2-8B-GGUF:Q4_ | default | yes | yes | 72 | 72 | 72 | 1.11 |
| vlm_02 | GLM-4.6V-Flash-9B | default | yes | yes | 72 | 0 | 0 | 1.46 |
| vlm_03 | minicpm-v4.6:latest | default | - | yes | 72 | 63 | 175 | 1.6 |
| vlm_03 | InternVL3_5:8b | default | - | yes | 72 | 72 | 164 | 2.95 |
| vlm_03 | qwen3-vl:8b-instruct | default | - | yes | 72 | 67 | 119 | 2.87 |
| vlm_03 | GLM-4.6V-Flash-9B | default | - | - | 72 | 69 | 117 | 2.78 |
| vlm_03 | GLM-4.6V-Flash-9B | default | - | yes | 72 | 66 | 111 | 2.69 |
| vlm_03 | Cosmos-Reason2-8B-GGUF:Q4_ | default | - | yes | 72 | 72 | 110 | 2.79 |
| vlm_03 | GLM-4.6V-Flash-9B | default | - | yes | 72 | 61 | 106 | 2.72 |
| vlm_03 | qwen3.5:9b | default | - | yes | 72 | 2 | 2 | 3.18 |
| vlm_04 | minicpm-v4.6:latest | default | - | yes | 72 | 72 | 411 | 2.96 |
| vlm_04 | GLM-4.6V-Flash-9B | default | - | yes | 72 | 72 | 396 | 2.75 |
| vlm_04 | GLM-4.6V-Flash-9B | default | - | yes | 72 | 72 | 379 | 2.35 |
| vlm_04 | GLM-4.6V-Flash-9B | default | - | yes | 72 | 72 | 379 | 2.35 |
| vlm_04 | qwen3.5:9b | default | - | yes | 72 | 72 | 323 | 3.81 |
| vlm_04 | qwen3-vl:8b-instruct | default | - | yes | 72 | 72 | 321 | 6.42 |
| vlm_04 | InternVL3_5:8b | default | - | yes | 72 | 69 | 166 | 2.47 |
| vlm_04 | Cosmos-Reason2-8B-GGUF:Q4_ | default | - | yes | 72 | 68 | 145 | 6.28 |
| vlm_05 | qwen3-vl:8b-instruct | conservative | yes | - | 72 | 72 | 1031 | 44.09 |
| vlm_05 | qwen3-vl:8b-instruct | conservative | yes | yes | 72 | 72 | 617 | 40.44 |
| vlm_05 | Cosmos-Reason2-8B-GGUF:Q4_ | conservative | yes | - | 72 | 67 | 341 | 24.49 |
| vlm_05 | qwen3.5:9b | conservative | yes | yes | 72 | 69 | 330 | 27.68 |
| vlm_05 | qwen3.5:9b *(cache)* | conservative | yes | yes | 72 | 69 | 330 | 0.45 |
| vlm_05 | qwen3.5:9b *(cache)* | conservative | yes | yes | 72 | 69 | 330 | 0.45 |
| vlm_05 | GLM-4.6V-Flash-9B | conservative | yes | - | 72 | 65 | 228 | 22.51 |
| vlm_05 | GLM-4.6V-Flash-9B | conservative | yes | yes | 72 | 66 | 228 | 21.13 |
| vlm_05 | GLM-4.6V-Flash-9B *(cache)* | conservative | yes | yes | 72 | 66 | 228 | 0.49 |
| vlm_05 | GLM-4.6V-Flash-9B | conservative | yes | yes | 72 | 66 | 226 | 22.06 |
| vlm_05 | GLM-4.6V-Flash-9B *(cache)* | conservative | yes | yes | 72 | 66 | 226 | 0.45 |
| vlm_05 | InternVL3_5:8b | conservative | yes | yes | 72 | 62 | 120 | 28.59 |
| vlm_05 | minicpm-v4.6:latest | conservative | yes | yes | 72 | 0 | 0 | 25.97 |

## ?

4 jobs

| pipeline | model | prompt | ref | mask | frames | flag | det | s/frame |
|---|---|---|---|---|---:|---:|---:|---:|
| vlm_05 | GLM-4.6V-Flash-9B | conservative | yes | - | 1 | 1 | 13 | 26.96 |
| vlm_05 | GLM-4.6V-Flash-9B | conservative | yes | yes | 1 | 1 | 12 | 27.4 |
| vlm_05 | GLM-4.6V-Flash-9B | conservative | yes | - | 1 | 1 | 12 | 15.85 |
| vlm_05 | GLM-4.6V-Flash-9B | conservative | yes | - | 1 | 1 | 9 | 21.29 |

## 1762-1  empty tram (123 f)

2 jobs

| pipeline | model | prompt | ref | mask | frames | flag | det | s/frame |
|---|---|---|---|---|---:|---:|---:|---:|
| vlm_03 | GLM-4.6V-Flash-9B | default | - | yes | 122 | 122 | 122 | 2.44 |
| vlm_05 | GLM-4.6V-Flash-9B | conservative | yes | yes | 122 | 0 | 0 | 16.31 |

## 1762-4  short clip (12 f)

2 jobs

| pipeline | model | prompt | ref | mask | frames | flag | det | s/frame |
|---|---|---|---|---|---:|---:|---:|---:|
| vlm_01 | GLM-4.6V-Flash-9B | default | - | - | 11 | 1 | 2 | 6.91 |
| vlm_05 | GLM-4.6V-Flash-9B | conservative | yes | yes | 3 | 0 | 0 | 10.35 |

## 1762-3  tram with clothes on a seat (176 f)

1 jobs

| pipeline | model | prompt | ref | mask | frames | flag | det | s/frame |
|---|---|---|---|---|---:|---:|---:|---:|
| vlm_03 | GLM-4.6V-Flash-9B | default | - | - | 175 | 175 | 209 | 2.48 |

## 1762-3  same clip, 73-frame extraction

1 jobs

| pipeline | model | prompt | ref | mask | frames | flag | det | s/frame |
|---|---|---|---|---|---:|---:|---:|---:|
| vlm_05 | GLM-4.6V-Flash-9B | conservative | yes | yes | 72 | 24 | 48 | 11.88 |

