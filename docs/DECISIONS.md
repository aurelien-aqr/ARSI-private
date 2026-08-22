# What we tried, and what we actually use

One line per settled question, so nobody has to read a benchmark report to find
out whether an approach is usable. **Newest first.** Details always live
somewhere else - this page only says the verdict and where to look.

| Question | Verdict | Measured on | Details |
|---|---|---|---|
| Does a Dinomaly checkpoint work on another camera? | **No, and not for the reason a threshold could fix.** A foreign checkpoint scores CLEAN frames 2.13x higher than the right one (median 0.163 vs 0.077) and its anomaly-minus-clean margin collapses to +0.0009 (own: +0.017 to +0.061). It is blind, not offset. Same-session, different-view transfer inside the 3333 family fails too, so it is the VIEWPOINT, not the lighting. The "one checkpoint per camera" cost in the rows below is therefore a requirement, not a choice | 8 checkpoints x 5 testable cameras, image-level AUROC, 2026-08-21 | `docs/paper/results.md` §1; `tools/cross_camera.py`; `docs/dino_models/cross_camera.json` |
| Dinomaly2 (arXiv 2510.17611), the authors' own answer to that objection | **No, do not ship - but report it.** Ported from their preview code, 8 checkpoints, everything else held fixed. Transfer still sits at chance (~0.45 pooled) and tram_1762 regresses 0.920 -> 0.801. The 16-checkpoint ablation attributes it: **context-aware recentering is the ONLY element that touches the cross-camera margin** (collapse 43x without it, i.e. v1's 41x, against 10x with it) and it is also what costs the 0.10 on tram_1762; the other three elements are what lift 3333-cam53 from 0.750 to 1.000. A trade, not an upgrade - but CAR is the one lever to pull if a single fleet checkpoint is ever needed | idem + `DINOMALY2_CAR=0` ablation, 2026-08-21 | `docs/paper/results.md` §2 and §2b; `tools/dinomaly2.py`, `tools/dinomaly2_train.py` |
| What do the standard baselines score on our cameras? | **0.835-0.920 on the only column with real statistics, and 1.000 on three of four 3333 views - frame-level detection is not our hard problem.** PaDiM 0.915, PatchCore 0.841, Dinomaly 0.886, AnomalyDINO 0.835 on tram_1762 (176 pairs), all untuned. The paper's contribution must sit at region and verdict level, not frame level. Bonus: our simplified Dinomaly re-implementation matches the anomalib reference on 4/5 cameras and beats it on 1762 (0.920 vs 0.886) | anomalib 2.6 reference implementations, same split as above, 2026-08-21 | `docs/paper/results.md` §3; `tools/anomalib_baseline.py`; `docs/dino_models/anomalib_baseline.json` |
| Judge model × prompt, swept | **No change. GLM-4.6V-Flash-9B × conservative stays**, and 12 arms failed to beat it at usable specificity. Read any recall number next to the arm's YES rate: the labels imply ~11 % of crops are anomalies, and the arms posting 63-64/73 answer YES to 29-88 % of them | 12 arms (4 models × 3 prompts) on 68 cases, boxes fixed at `ddgate0.05`, 2026-08-19 | Studio → Notes → *VLM benchmark*; `docs/vlm_benchmark/` |
| Qwen2.5-VL-7B / Qwen3-VL-8B as crop judges | **No.** Frame specificity 0.081-0.297 — they fire on nearly every clean tram. `qwen3-vl:8b-instruct` was the `MODEL_NAME` default of the vlm_0x scripts until 2026-08-20; every script now defaults to GLM | idem | idem |
| The `balanced` prompt (caution clauses dropped, NO conditions kept) | **Keep it available, do not ship it.** It works where a model under-calls: InternVL3.5 39 → 48 instances at specificity 1.000 unchanged. On GLM it moves nothing — a prompt can rescue an under-calling model, not push a calibrated one past its ceiling | idem | `tools/judge_prompts.py` |
| Does a better box become a better verdict? | **Partly - and not the part you would guess.** Object recall is 55/73 under every localizer; better boxes buy frame recall (0.871 → 0.968), region precision (0.694 → 0.896) and correct boxing (0.534 → 0.630), at frame specificity 1.000 throughout | 68 cases end to end, GLM-4.6V-Flash-9B × conservative, 2026-08-19 | `benchmark/README.md` § "Does a better box…"; `tools/rescore_localizer.py` |
| `recommended` badge in the Studio picker | **Moved `photo+dino` → `dino`** on the end-to-end numbers. Trade: +69 % VLM calls for +0.097 frame recall and +0.123 region precision. Revert if judge cost, not miss rate, binds | idem | `arsi_core/localizers.py` docstring |
| Best measured localizer | **AnomalyDINO proposes, Dinomaly vetoes** (`ddgate0.05`): 58/73 strict for 654 regions, against 42/73 / 1192 for the pixel diff. On 1762, where the nominal model is properly held out, −43 % regions at unchanged boxes. End to end it matches `dino` exactly for 47 % fewer calls — but it is not a registry entry, it needs a checkpoint per camera | 68-case localization benchmark, 3 trams / 9 views, 0 VLM calls, 2026-08-19 | Studio → Notes → *Dinomaly & AnomalyDINO*; `docs/dino_models/` |
| Best localizer **without any training** | **AnomalyDINO standalone**, and this REVERSES the 2026-08-12 row below. 58/73 strict against the pixel diff's 42/73; on 3333\*, 19/26 against 5/26. Costs no fewer VLM calls; `dpgate10` gives −16 % of them for free | idem | idem |
| Dinomaly, re-opened now that training is cheap | **Yes, but only as a veto over feature-proposed boxes.** Its August rejection stands for the two roles it was tested in (proposer 48/73, gate over pixel boxes 42/73). Cost objection dead: 56 min → 1 min 42 per camera on the RTX 3080 Ti, fleet ~45 min | idem | idem; GPU support in `tools/dinomaly.py` |
| Gating (any veto) as a way to fix boxes | **No - structural.** Every variant that gates the pixel diff scores exactly 42/73, identical to the ungated pixel diff. A veto deletes regions; it never redraws one, so what decides box quality is the PROPOSER | idem | idem |
| Dinomaly (CVPR 2025), reference-free | **No.** Coded and measured, deliberately not wired into the app | 29-case localization benchmark, 0 VLM calls, 2026-08-16 | `benchmark/README.md` § Dinomaly; code in `tools/dinomaly*.py` |
| AnomalyDINO as a **veto** over the pixel-diff boxes | **Yes on 1762** - it is the `photo+dino` localizer in the Studio picker. Still the right pick there, but it is a cost optimisation, not a quality one | same benchmark, 2026-08-12 | `benchmark/README.md` § DINOv2 feature gate |
| ~~AnomalyDINO as a **replacement** for the pixel diff~~ | ~~**No.** Its boxes are quantised to the patch grid: strict IoU 32/45 vs 37/45~~ **SUPERSEDED 2026-08-19** - true on 1762, false across the fleet; see the top row | same benchmark, 2026-08-12 | idem |
| anomalib as a framework | Not adopted. Worth it only to produce a standard baseline table (PatchCore / EfficientAD, I-AUROC / PRO) for the paper | - | `benchmark/README.md` |
| Merging neighbouring regions before the judge | **Yes, kept.** Region precision 0.663 → 0.730 at identical recall | 29 cases with the VLM, 2026-07-30 | `benchmark/README.md` § Merge A/B |
| Judge model | **GLM-4.6V-Flash-9B × conservative.** Frame F1 1.000 | 29 cases, GPU, 2026-07-12/13 | `benchmark/README.md` § GPU results |
| InternVL3.5-8B / minicpm-v4.6 as judges | No. InternVL systematically rejects real phones; minicpm ignores the output format and hallucinates objects on 198 of 199 clean crops | idem | idem |

## What these verdicts do NOT cover

Every row above was measured on **one camera of tram 1762**, from the July
footage: 29 cases, 45 instance boxes, two reference frames. That was the only
labelled data available when they were decided.

Since then we have the **1760 and 3333 multi-camera captures** - other trams,
other angles, 26 masked cameras.

**Four of those cameras are now labelled** and part of the benchmark:
21 cases / 24 instances on tram 3333 in `benchmark/datasets/ground_truth.json`
(built 2026-08-16, labelled by Claude from the footage). Their first measurement
already moves
the ground under the table above: the shipped localizer scores **17/24
instances and 5/24 strict IoU** there, against 45/45 and 37/45 on 1762, and the
DINOv2 gate cuts 8 % of regions instead of 57 %. Read every row above as "true
on the 1762 camera", not "true".

The remaining 22 cameras are still unmeasured, so:

- a verdict may hold on 1762 and fail on a camera that films through a window,
  or on a night session;
- the **Dinomaly rejection is the most fragile one**. Its main cost in our
  reasoning is "one model to train per camera", and its actual selling point is a
  single multi-class model covering all of them. A multi-camera protocol is
  exactly the setting where that argument could flip;
- what is missing is not code, it is **labels**: clean and anomalous frames per
  camera, with instance boxes, in `benchmark/datasets/ground_truth.json`. Since
  2026-08-17 the Studio's **Benchmark** screen browses and corrects those labels
  directly, and launches a scored run on any subset × localizer × judge × prompt.
  The Labels screen is a different thing: TP/FP review of a job's own output, for
  the LoRA dataset.

Read every row above as measured on the 1762 camera. The benchmark now covers 5
cameras, so a re-run of these questions over the whole thing is the way to find
out which verdicts survive.

## Leads never measured

Three ideas that were queued for a GPU day and never scored. Recorded here so
they are re-decided deliberately rather than re-invented:

- **Cascade**: score `openbmb/minicpm-v4.6` as a cheap screener in front of the
  real judge. If its false-negative rate on cached crops is ~0, it cuts most of
  the judge cost. (It was disqualified as a *judge* - it hallucinates objects on
  198 of 199 clean crops - which says nothing about its recall.)
- **Temporal persistence**: "forgotten = present in N frames with no person
  nearby". The videos in `data/videos` are the input; nothing in the pipeline
  looks across frames today.
- **AD-Copilot-Thinking** (`jiang-cc/AD-Copilot-Thinking`, transformers, not
  Ollama): the closest published system to vlm_05's visual in-context
  comparison. Worth one afternoon of comparison.

---

\* **3333** is a placeholder, not the tram's real fleet number - the vehicle number of the 2026-08-11 capture is unknown. It was called 39T before, but 39T is the Škoda type, which tram 1760 shares.
