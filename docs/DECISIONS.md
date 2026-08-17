# What we tried, and what we actually use

One line per settled question, so nobody has to read a benchmark report to find
out whether an approach is usable. **Newest first.** Details always live
somewhere else — this page only says the verdict and where to look.

| Question | Verdict | Measured on | Details |
|---|---|---|---|
| Dinomaly (CVPR 2025), reference-free | **No.** Coded and measured, deliberately not wired into the app | 29-case localization benchmark, 0 VLM calls, 2026-08-16 | `benchmark/README.md` § Dinomaly; code in `tools/dinomaly*.py` |
| AnomalyDINO as a **veto** over the pixel-diff boxes | **Yes — usable now**, it is the `photo+dino` localizer in the Studio picker | same benchmark, 2026-08-12 | `benchmark/README.md` § DINOv2 feature gate |
| AnomalyDINO as a **replacement** for the pixel diff | **No.** Its boxes are quantised to the patch grid: strict IoU 32/45 vs 37/45 | same benchmark, 2026-08-12 | idem |
| anomalib as a framework | Not adopted. Worth it only to produce a standard baseline table (PatchCore / EfficientAD, I-AUROC / PRO) for the paper | — | `benchmark/README.md` |
| Merging neighbouring regions before the judge | **Yes, kept.** Region precision 0.663 → 0.730 at identical recall | 29 cases with the VLM, 2026-07-30 | `benchmark/README.md` § Merge A/B |
| Judge model | **GLM-4.6V-Flash-9B × conservative.** Frame F1 1.000 | 29 cases, GPU, 2026-07-12/13 | `benchmark/README.md` § GPU results |
| InternVL3.5-8B / minicpm-v4.6 as judges | No. InternVL systematically rejects real phones; minicpm ignores the output format and hallucinates objects on 198 of 199 clean crops | idem | idem |

## What these verdicts do NOT cover

Every row above was measured on **one camera of tram 1762**, from the July
footage: 29 cases, 45 instance boxes, two reference frames. That was the only
labelled data available when they were decided.

Since then we have the **1760 and 39T multi-camera captures** — other trams,
other angles, 26 masked cameras.

**A second protocol now exists** for 4 of those cameras:
`benchmark/datasets/39T.json`, 21 cases / 24 instances on tram 39T (built
2026-08-16, labelled by Claude, unreviewed). Its first measurement already moves
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
  camera, with instance boxes, in the `benchmark/datasets/` format. Since
  2026-08-17 the Studio's **Benchmark** screen browses and corrects those labels
  directly (and counts how many cases a human has confirmed — 39T is still 0/21),
  and launches a scored run on any dataset × localizer × judge × prompt. The
  Labels screen is a different thing: TP/FP review of a job's own output, for the
  LoRA dataset.

Treat this page as valid *for the 1762 benchmark* until that second protocol
exists.
