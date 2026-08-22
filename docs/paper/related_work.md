# Related work / state of the art

Working document for the ARSI paper. Built 2026-08-21 with the scite MCP server
(smart-citation index over ~210 M papers), plus direct reads of the two PDFs
already in the repo root.

**Rules this file follows.**

1. Nothing is cited that was not retrieved through scite or read locally. Every
   entry carries a DOI I actually saw in a search result.
2. Every reference is attached to **a claim we want to make**, not to a topic.
   A reference that supports no claim of ours does not belong in the paper.
3. Where I have a verbatim passage supporting the claim, it is quoted in the
   ledger at the end. Where I only have the abstract, the ledger says so.
4. Anything taken from our own earlier notes and *not* re-verified this session
   is marked **[UNVERIFIED]**. Do not put those in the bibliography until they
   have been checked.

The organising principle is our own pipeline — propose regions, then judge them —
because that is also the axis on which our results are interesting:

```
frame ──▶ localizer (proposes regions) ──▶ crop ──▶ VLM judge (verdict) ──▶ alarm
          §2, §3, §7                                §4, §5, §6
```

---

## 1. What problem the literature thinks it is solving

Three communities touch our task and none of them owns it.

- **Industrial visual anomaly detection (VAD)** works on objects photographed
  under controlled conditions, one product class at a time, and asks *does this
  instance deviate from the nominal distribution*. Benchmarks: MVTec AD, VisA.
- **Video anomaly detection in surveillance** works on unconstrained scenes and
  asks *does something abnormal happen in this clip*, usually with video-level
  labels only. Benchmark: UCF-Crime.
- **Abandoned-object detection (AOD)** works on fixed cameras in public spaces
  and asks *did an object appear and stay*. Benchmarks: PETS2006, i-LIDS AVSS
  2007, ABODA.

Our setting — a fixed camera inside a tram, a clean reference frame available for
that exact camera, anomalies that are *persistent appearance changes* (graffiti,
forgotten bag, damaged seat, stain) rather than events — sits between the three.
That gap is the paper's first justification, and it is documented in
`docs/PUBLIC_DATASETS.md`: no public dataset combines in-vehicle, fixed camera,
and ground truth for graffiti or abandoned objects.

---

## 2. Anomaly detection without language

### 2.1 The reference-free industrial line

The dominant paradigm is *fit the nominal distribution, flag the outliers*.
MVTec AD (Bergmann et al., CVPR 2019, `10.1109/cvpr.2019.00982`) defined the
benchmark; PatchCore (Roth et al., CVPR 2022, `10.1109/cvpr52688.2022.01392`,
arXiv `10.48550/arxiv.2106.08265`) set the memory-bank recipe that most later
work compares against, reporting 99.1 % image-level AUROC on MVTec AD;
EfficientAD (Batzner et al., WACV 2024, `10.1109/wacv57701.2024.00020`) pushed
the same family to millisecond latencies. anomalib (Akçay et al., ICIP 2022,
`10.1109/icip46576.2022.9897283`) is the reference implementation library.

**Why it matters to us.** `docs/DECISIONS.md` records that anomalib was *not
adopted*, and that it would be "worth it only to produce a standard baseline
table (PatchCore / EfficientAD, I-AUROC / PRO) for the paper". That judgement is
correct and it is the paper's main missing experiment: this is the family a
reviewer will ask us to compare against, and the comparison is cheap because
anomalib implements both.

**Claim we can support:** *the standard industrial paradigm assumes a nominal
training set per class and reports image-level AUROC on object-centric
benchmarks; neither the assumption nor the metric transfers unchanged to a fixed
CCTV camera where "nominal" is one photograph of an empty tram.*

### 2.2 Multi-class, and the reason we re-opened Dinomaly

Dinomaly (Guo et al., CVPR 2025, `10.1109/cvpr52734.2025.01900`; arXiv
`10.48550/arxiv.2405.14325`) is the current strong answer to the *multi-class*
setting — one model for all classes rather than one model per class. Verbatim
from the abstract: "builds a unified model for multi-class images […] achieves
impressive image-level AUROC of 99.6 %, 98.7 %, and 89.3 %" on MVTec-AD, VisA and
Real-IAD, "not only superior to state-of-the-art multi-class UAD methods, but
also achieves the most advanced class-separated UAD records".

**Why it matters to us — and this is the sharpest link in the whole survey.**
Our recorded objection to Dinomaly is cost: "one model to train per camera".
Dinomaly's *entire selling point* is that this is not required. `DECISIONS.md`
already flags this as "the most fragile" of our verdicts, and a multi-camera
protocol is exactly the setting where it could flip. The literature does not
contradict our measurement; it contradicts our *reason*. That is worth saying
explicitly in the paper — it is honest and it is the kind of thing reviewers
reward.

It gets worse for our objection, in a useful way. There is now a **successor**:
"One Dinomaly2 Detect Them All: A Unified Framework for Full-Spectrum
Unsupervised Anomaly Detection", `10.48550/arxiv.2510.17611` (preprint; states
verbatim "A preliminary version of this work has been published in CVPR2025
(Dinomaly)"). Full-spectrum unification across modalities and domains is exactly
the axis our fleet objection lives on. Our rejection was measured against the
CVPR 2025 version, which is legitimate and must be *stated as such* in the paper.

**Citation tracing (2.2 bis).** A scite sweep of everything citing Dinomaly or
AnomalyDINO since 2025 (45 papers) answers the question raised in an earlier
draft of §11 — *has anyone already reported the transfer failure we saw on camera
3333?*

- **No one reports it.** What the citing literature does instead is *carry these
  methods to new domains and benchmark them there*: AD4AD for autonomous driving
  (`10.48550/arxiv.2604.15291`), HiMatch-AD for medical imaging
  (`10.48550/arxiv.2606.22556`), OASIC repurposing AnomalyDINO for occlusion
  segmentation (`10.48550/arxiv.2604.04012`), Tiny-Dinomaly for edge/continual
  settings (`10.48550/arxiv.2604.06435`). All preprints, but they establish the
  genre: *take industrial AD to a domain it was not designed for and report what
  breaks.* That is our paper's Section 4, and it is a recognised contribution
  type. Cite one or two as precedent for the protocol, not for results.
- **One critical statement worth quoting**, from RAID
  (`10.48550/arxiv.2602.19611`, preprint): foundation-model AD approaches
  including "WinCLIP, AnomalyDINO, and Dinomaly […] still suffer from unreliable
  feature matching arising from imperfect reconstructions or suboptimal template
  retrieval, and limited few-shot generalization". That is the closest published
  statement to what our 3333 numbers show.

So the claim is available and, as far as this sweep goes, unclaimed: *nobody has
reported how these detectors degrade across cameras of one fixed-camera fleet.*

**And as of 2026-08-21 we have measured it** — see `docs/paper/results.md` §1 and
§2. A foreign checkpoint is blind rather than miscalibrated (clean-frame scores
inflate 2.13×, the anomaly margin collapses to zero), the failure is driven by
viewpoint rather than lighting, and Dinomaly2's context-aware recentering halves
the gap without closing it. That is the empirical content behind claim 1 of §9.

### 2.3 Vision-only foundation features

DINOv2 (Oquab et al., arXiv `10.48550/arxiv.2304.07193`, 2 673 citing
statements) is the feature extractor. AnomalyDINO (Damm, Laszkiewicz, Lederer &
Fischer; WACV 2025 `10.1109/wacv61041.2025.00136`, arXiv
`10.48550/arxiv.2405.14529`) turns it into a few-shot anomaly detector.
Verbatim from the abstract: "explores whether high-quality visual features alone
are sufficient to rival existing state-of-the-art vision-language models. We
affirm this […] The approach is methodologically simple and training-free […]
pushing the one-shot performance on MVTec-AD from an AUROC of 93.1 % to 96.6 %".

**Why it matters to us.** AnomalyDINO is our best *training-free* localizer
(58/73 strict against 42/73 for the pixel diff, 2026-08-19). Their claim —
vision-only features rival vision-language models, training-free — is the same
claim our localization benchmark makes independently, on a completely different
domain. That is a supporting citation in the strong sense: same finding, other
data.

### 2.4 Change detection, which is what our pixel diff really is

CDnet 2014 (Wang, Jodoin, Porikli et al., CVPRW 2014,
`10.1109/cvprw.2014.126`, 691 citing statements) is the reference benchmark for
per-pixel change detection from static cameras, and its
`intermittentObjectMotion` category is defined as "videos containing background
objects moving away, abandoned objects and objects stopping for a short while".

**Why it matters to us.** `vlm_05_reference_diff.py` is a change detector with a
VLM behind it. CDnet lets us evaluate the proposal stage *without the VLM in the
loop* — which is precisely the ablation our own results say is the decisive one
(see §5.2 of this file). The caveat in `PUBLIC_DATASETS.md` stands and must be
restated in the paper: CDnet ground truth marks motion, not abnormality, so the
claim it licenses is "our change-detection stage reaches F1 = x on
intermittentObjectMotion", never "our anomaly detector reaches F1 = x".

---

## 3. Abandoned objects: the closest existing task

Luna, SanMiguel, Ortego et al., "Abandoned Object Detection in
Video-Surveillance: Survey and Comparison", *Sensors* 18(12):4290, 2018,
`10.3390/s18124290` — OA, 20 citing statements, no editorial notices. This is the
survey that unifies the evaluation protocol across AVSS AB 2007, PETS2006,
PETS2007, ABODA and VISOR. Adopting their protocol is much stronger than
inventing ours, and the PDF is already in the repo (`sensors-18-04290-v2.pdf`).

What the field looks like **eight years later** is the finding worth reporting.
A scite sweep of `"abandoned object detection" AND ("video surveillance" OR
"public transport" OR "left luggage")`, 2018→now, 47 results, returns almost
exclusively background-subtraction and tracking pipelines:

- Park, Park & Joo, "Detection of Abandoned and Stolen Objects Based on Dual
  Background Model and Mask R-CNN", *IEEE Access* 8:80010-80019, 2020,
  `10.1109/access.2020.2990618` — dual background model for candidate stationary
  objects, Mask R-CNN for the semantic decision. Their own related-work passage
  states the recurring failure mode: earlier dual-background work "did not solve
  the problem of tracking long-term abandonment and illumination change issue due
  to the limitation of the background subtraction".
- Park & Park, "Robust Detection of Abandoned Object for Smart Video
  Surveillance in Illumination Changes", *Sensors* 19(23):5114, 2019,
  `10.3390/s19235114`.
- Xu, Hu & Zhu, "RAOD: A Benchmark for Road Abandoned Object Detection From Video
  Surveillance", *IEEE Access* 12:123985-123994, 2024,
  `10.1109/access.2024.3407955`.
- Yilmazer & Karaköse, *Applied Sciences* 15(5):2774, 2025,
  `10.3390/app15052774` (DeepSORT-based).

**Claim we can support, and it is a good one:** *abandoned-object detection is
still overwhelmingly solved by background modelling plus a closed-vocabulary
classifier; the semantic stage, when present, is a detector trained on a fixed
label set (Mask R-CNN), not an open-vocabulary reasoner. Illumination change is
the acknowledged, unsolved failure mode of that family.* Both halves of that
sentence are exactly what our architecture attacks: an open-vocabulary judge, and
a masking step for the windows.

---

## 4. Language-guided anomaly detection

### 4.1 CLIP-based zero-shot AD

WinCLIP (Jeong, Zou, Kim et al., CVPR 2023, `10.1109/cvpr52729.2023.01878`, 280
citing statements) opened the line; AnomalyCLIP (Zhou, Pang, Tian et al., arXiv
`10.48550/arxiv.2310.18961`, 74 citing statements) made the prompts
object-agnostic. The line is extremely active — a scite sweep restricted to
2023+ returns 54 papers, nearly all from 2026, with the same structure
(FP-CLIP `10.1016/j.dsp.2025.105798`, DyC-CLIP `10.1016/j.patcog.2026.113215`,
PromptMoE AAAI `10.1609/aaai.v40i11.37842`, VisualAD
`10.48550/arxiv.2603.07952`, and a dozen more). One recent survey passage found
in full text lists nineteen of them by name in a single sentence.

**How to use this in the paper.** Not as a baseline — these methods segment
defects on object-centric industrial images and would need re-training or at
least re-prompting for tram interiors. Use it as the *contrast*: this branch puts
language in the **scoring function** (text embeddings as anomaly prototypes),
whereas we put language in the **verdict** (a generative model answering a
yes/no question about a crop). Those are different architectures with different
failure modes, and ours is the one that produces a human-readable justification.

### 4.2 The LVLM as a detector

AnomalyGPT (Gu, Zhu, Zhu et al., AAAI 38(3):1932-1940, 2024,
`10.1609/aaai.v38i3.27963`; arXiv `10.48550/arxiv.2308.15366`) is the closest
architectural relative. Verbatim from the abstract: LVLMs "lack specific domain
knowledge and have a weaker understanding of localized details within objects,
which hinders their effectiveness in the Industrial Anomaly Detection task", and
"most existing IAD methods only provide anomaly scores and necessitate the manual
setting of thresholds". Their answer is an image decoder plus a prompt learner
fine-tuned on simulated anomalies; 86.1 % accuracy, 94.1 % image AUC one-shot on
MVTec-AD.

**Two claims of ours it licenses.**

1. *LVLMs are weak on localized detail.* This is the published diagnosis of
   exactly what our benchmark measures on tram 3333, where region recall collapses
   because the boxes are small.
2. *Score-plus-threshold pipelines need manual calibration; a language verdict
   does not.* That is our design argument for a yes/no judge, stated by someone
   else first.

### 4.3 The taxonomy to borrow

Ren, Tang, Jia et al., "Foundation Models for Anomaly Detection: Vision and
Challenges", *AI Magazine* 46(4), 2025, `10.1002/aaai.70045` — first
comprehensive survey of FM-based anomaly detection, classifying foundation models
by **the role they play in the pipeline: encoders, detectors, or interpreters**.

This taxonomy is a gift for our §Method, because our system uses all three and
the paper can say so in one sentence: DINOv2 as *encoder*, Dinomaly /
AnomalyDINO as *detector*, GLM-4.6V as *interpreter*. The same survey states
that LVLM-based approaches are "computationally expensive, heavily sensitive to
prompt design" — which is the published motivation for our 12-arm prompt × model
sweep.

---

## 5. Video anomaly detection in surveillance

### 5.1 The weakly supervised mainstream

Sultani, Chen & Shah, "Real-World Anomaly Detection in Surveillance Videos",
CVPR 2018, `10.1109/cvpr.2018.00678` (1 632 citing statements, of which **3
contrasting** — worth reading before citing) introduced UCF-Crime: 1 900
untrimmed videos, 128 hours, 13 anomaly classes, video-level labels, and the
multiple-instance-learning (MIL) formulation that the field still uses. The
recent CLIP-flavoured continuation is well represented in *Journal of Imaging*
itself: WSVAD-CLIP (Li, Lu & Du, 11(10):354, 2025, `10.3390/jimaging11100354`)
and Si, Dong & Yang (12(6):249, 2026, `10.3390/jimaging12060249`).

**Why this is not our baseline, and say it explicitly.** Every UCF-Crime video
comes from a different scene and a different camera; there is no clean footage of
that same camera to subtract from. Our differencing stage has nothing to
subtract. `PUBLIC_DATASETS.md` already draws the right conclusion — treat
UCF-Crime as a separate *compatibility* experiment with the evaluation unit
deliberately adapted, not as the main table.

### 5.2 The paper we are actually in dialogue with

Borodin, Kondrashov, Vasiliev, Gladkova, Larina, Gorodnichev & Mkrtchian,
"Benchmarking Compact VLMs for Clip-Level Surveillance Anomaly Detection Under
Weak Supervision", *Journal of Imaging* 11(11):400, 2025,
`10.3390/jimaging11110400`. OA. (Repo copy: `CCTV_VLM_paper.pdf`; that PDF is
arXiv:2603.13306v1.)

Their conclusion, verbatim from §5 of the PDF:

> "Parameter-efficient fine-tuning emerges as the critical enabler for compact
> vision-language models in video anomaly detection, turning training-free
> variants into reliable clip-level detectors while retaining competitive
> per-clip latency within the shared protocol. **After adaptation, concise
> instruction or zero-shot prompting remains sufficient, and prompt sensitivity
> is notably reduced across models and settings**, yielding consistent behavior
> under the same evaluation conditions."

And from their introduction:

> "There is limited evidence on when parameter-efficient fine-tuning (PEFT), such
> as Low-Rank Adaptation (LoRA), materially improves small VLMs for this task
> compared with training-free use, so explicitly testing both regimes is
> necessary."

**This is the single most useful citation in the file, for two reasons.**

*First*, it frames our prompt sweep as the complementary half of their result.
They measure prompt sensitivity **after** adaptation and find it reduced. We
measured 12 arms (4 models × 3 prompts) **without** adaptation, on 68 cases, and
found that a prompt can rescue a model that under-calls (InternVL3.5, 39 → 48
instances at unchanged specificity) but cannot push a calibrated one past its
ceiling. Those two findings are consistent and, together, they say something
neither says alone: *prompt engineering is a fix for miscalibration, not a source
of capability — and PEFT is what removes the miscalibration.*

*Second*, it is independent support for our `RUNBOOK_LORA.md` / `LORA_PLAN.md`
direction, from a peer-reviewed paper on the same task family. Our LoRA work
stops being "an idea we had" and becomes "the intervention the literature
identifies as decisive, tested in a setting they did not cover".

### 5.3 The three training-free VLM pipelines — our closest competitors

These are the systems Borodin benchmarks against, and the ones that could
pre-empt a novelty claim for a localize-then-judge design. All three now
resolved, and all three read.

**LAVAD** — Zanella, Menapace, Mancini et al., "Harnessing Large Language Models
for Training-free Video Anomaly Detection", CVPR 2024, pp. 18527-18536,
`10.1109/cvpr52733.2024.01753` (arXiv `10.48550/arxiv.2404.01014`). Verbatim:
"We leverage VLM-based captioning models to generate textual descriptions **for
each frame** of any test video. With the textual scene description, we then
devise a prompting mechanism to unlock the capability of LLMs in terms of
temporal aggregation and anomaly score estimation". Evaluated on UCF-Crime and
XD-Violence, outperforming unsupervised and one-class methods with no training.

**VERA** — Ye, Liu, He et al., "VERA: Explainable Video Anomaly Detection via
Verbalized Learning of Vision-Language Models", CVPR 2025, pp. 8679-8688,
`10.1109/cvpr52734.2025.00811` (arXiv `10.48550/arxiv.2412.01095`). Verbatim:
VERA "automatically decomposes the complex reasoning required for VAD into
reflections on simpler, more focused guiding questions capturing distinct
abnormal patterns. It treats these reflective questions as **learnable
parameters** and optimizes them through data-driven verbal interactions between
learner and optimizer VLMs". Output is segment-level scores refined to
frame-level.

**AnomalyRuler** — Yang, Lee, Dariush et al., "Follow the Rules: Reasoning for
Video Anomaly Detection with Large Language Models", ECCV 2024, pp. 304-322,
`10.1007/978-3-031-73004-7_18` (arXiv `10.48550/arxiv.2407.10299`). Verbatim:
two stages, "In the induction stage, the LLM is fed with few-shot normal
reference samples and then summarizes these normal patterns to induce a set of
rules for detecting anomalies. The deduction stage follows the induced rules to
spot anomalous frames in test videos."

**What this buys us — read carefully, it is the novelty argument.**

1. **All three operate on frames, not on proposed regions.** LAVAD captions each
   whole frame; VERA scores segments then frames. Their unit of decision is
   temporal (which frame/segment is anomalous), never spatial (which region of
   this frame). Our pipeline's unit of decision is a **crop**, and our headline
   result is a statement about *which stage of that spatial pipeline binds*. That
   question cannot even be posed in their formulation. *(Caveat: AnomalyRuler's
   perception front-end is paywalled and I could not verify whether it detects
   objects before prompting. Read the ECCV PDF before asserting "all three" in
   print — say "LAVAD and VERA" until then.)*
2. **VERA is the direct precedent for our prompt sweep, and it beat us to the
   idea in a stronger form.** They treat the question text as a learnable
   parameter and optimise it; we hand-wrote three prompts and swept them. We must
   cite VERA and position honestly: our result is not "prompts matter" (they
   showed that), it is *prompts cannot lift a calibrated model past its ceiling,
   only rescue a miscalibrated one* — a limit statement, which is compatible with
   VERA and with Borodin's post-PEFT insensitivity.
3. **They all use the reference set as language, we use it as pixels.**
   AnomalyRuler induces textual rules from few normal shots; we subtract a clean
   reference frame of the same camera. Ours is only possible because the camera
   is fixed — which is the deployment fact that defines our setting.

---

## 6. Can the judge be trusted? (our sharpest result needs this section)

Our benchmark's central caution is that recall must be read next to the YES
rate: the labels imply ~11 % of crops are anomalies, and the arms that post
63-64/73 answer YES to 29-88 % of them. That is not a curiosity, it is a known
and named pathology.

**POPE.** Li, Du, Zhou et al., "Evaluating Object Hallucination in Large
Vision-Language Models", EMNLP 2023, pp. 292-305,
`10.18653/v1/2023.emnlp-main.20` (52 citing statements). Verbatim: they propose
"Polling-based Object Probing Evaluation (POPE) […] The basic idea is to convert
the evaluation of hallucination into a binary classification task by prompting
LVLMs with simple Yes-or-No short questions about the probing objects (e.g., Is
there a car in the image?)" and report that evaluated LVLMs "mostly suffer from
severe object hallucination issues", with objects that "frequently appear in the
visual instructions or co-occur with the image objects" being especially prone.

This is *our exact prompt format*, evaluated as a benchmark, two years earlier.
It gives us the vocabulary and the metric to describe what we saw in one line:
minicpm-v4.6 hallucinating objects on 198 of 199 clean crops is object
hallucination under yes/no probing, and the right way to report it is a YES-rate
alongside recall — which is what our benchmark already does.

**The affirmative-bias literature** is younger and mostly preprints. Two found:
"System-Mediated Attention Imbalances Make Vision-Language Models Say Yes"
(`10.48550/arxiv.2601.12430`) and "Looked but didn't see: inattentional blindness
and yes-bias confabulation in vision-language models"
(`10.64898/2026.06.16.26355792`, openRxiv **preprint** — cite with the usual
caution or not at all).

**LLM-as-a-judge.** Our GLM judge is a judge in the technical sense, and the
methodological literature is now substantial (5 745 results on a
reliability/bias query). The anchor is the peer-reviewed survey: Gu, Jiang et
al., "A survey on LLM-as-a-judge", *The Innovation*, 2026,
`10.1016/j.xinn.2025.101253` (23 citing statements). Also relevant to how we
*report*: Lee, Zeng et al., "How to Correctly Report LLM-as-a-Judge Evaluations"
(`10.48550/arxiv.2511.21140`, preprint, 9 citing statements).

**Claim:** *treating a generative VLM as a binary judge inherits a documented
affirmative bias; any recall figure reported without the accompanying positive
rate is uninterpretable.* We can state that as a methodological contribution
because our sweep demonstrates it on a deployment task.

---

## 7. Proposing the regions the judge will look at

Two families are relevant to the localizer, and we already use one of them.

**Open-vocabulary detection.** Grounding DINO (Liu, Zeng, Ren et al., arXiv
`10.48550/arxiv.2303.05499`, 463 citing statements; ECCV 2024
`10.1007/978-3-031-72970-6_3`) and YOLO-World (Cheng, Song, Ge et al., CVPR 2024
`10.1109/cvpr52733.2024.01599`, arXiv `10.48550/arxiv.2401.17270`). The repo
ships `yolov8x-worldv2.pt`, so YOLO-World is a *used* component and must be cited
in Methods, not just Related Work.

**Tiling for small objects.** Akyön, Altinuç & Temizel, "Slicing Aided Hyper
Inference and Fine-Tuning for Small Object Detection", ICIP 2022,
`10.1109/icip46576.2022.9897990` (213 citing statements; arXiv
`10.48550/arxiv.2202.06934`) is the canonical citation, with a substantial
follow-up literature (e.g. adaptive SAHI for remote sensing,
`10.3390/rs15051249`; slice-aided defect detection on wind-turbine blades,
`10.3390/machines11100953`).

**Why it matters to us.** Our tiling experiment on camera 3333 moved region
recall 0.500 → 0.615 with zero false alarms, and broke specificity on 1762
(1.000 → 0.750). SAHI is the prior art for "slice the image so the small thing
occupies enough pixels", and the wind-turbine and remote-sensing follow-ups are
the precedent for "it is a per-domain option, not a universal default". Our
per-camera result is a *contribution to that specific conversation* rather than
an unexplained inconsistency — that reframing is worth a paragraph.

Also worth a look before the Methods section is finalised, both found in the VLM
sweep and both very close to our design:

- "BEM: Training-Free Background Embedding Memory for False-Positive Suppression
  in Real-Time Fixed-Background Cameras", `10.48550/arxiv.2604.11714` — same
  premise as ours (fixed background, reference memory, false-positive
  suppression). **Read this one before writing Methods.** Preprint.
- "GridVAD: Open-Set Video Anomaly Detection via Spatial Reasoning over
  Stratified Frame Grids", `10.48550/arxiv.2603.25467` — grid-based spatial
  reasoning for VAD, i.e. our tiling idea in the video-anomaly branch. Preprint.
- "Self-Improving Small Object Grounding in LVLMs", `10.48550/arxiv.2606.01612` —
  directly on the "the box is too small for the VLM" failure. Preprint.

---

## 8. In-vehicle CCTV

Thin, and that thinness is part of our motivation. A scite sweep of
in-vehicle / public-transport CCTV crossed with anomaly, vandalism, graffiti or
abandoned objects (2019→now, 3 496 hits, top 25 read) returns **no work on
persistent appearance change inside a vehicle**. What it does return:

- **Ciampi, Foszner, Messina et al., "Bus Violence: An Open Benchmark for Video
  Violence Detection on Public Transport", *Sensors* 22(21):8345, 2022,
  `10.3390/s22218345`** (OA, 9 citing statements). Now verified — note the exact
  title differs from the one in our older notes. Our objection stands and is
  worth stating in one line: the clips are 0.6-1.9 s, and its "NoViolence" label
  asserts only the absence of a violent action, not the absence of a forgotten
  bag, a stain or graffiti. It is therefore **not** 700 verified negatives for our
  task. Cite it as *the* in-vehicle CCTV benchmark and as evidence that the
  in-vehicle literature is about events, not about persistent change.
- **Caetano, Carvalho & Cardoso, "Deep Anomaly Detection for In-Vehicle
  Monitoring — An Application-Oriented Review", *Applied Sciences*
  12(19):10011, 2022, `10.3390/app121910011`** (OA, 11 citing statements). The
  survey that covers our deployment domain. Read it before writing the
  introduction; it is the fastest way to find out whether anyone has framed the
  in-vehicle problem the way we do.
- **Duong, Le et al., "Deep Learning-Based Anomaly Detection in Video
  Surveillance: A Survey", *Sensors* 23(11):5024, 2023, `10.3390/s23115024`**
  (20 citing statements) — the general survey to cite for the field's shape.
- **SanMiguel & Martínez, "Robust Unattended and Stolen Object Detection by
  Fusing Simple Algorithms", AVSS 2008, pp. 18-25, `10.1109/avss.2008.16`** — the
  same group as the Luna survey, and the classical anchor for the AOD lineage of
  §3.
- Marginal but real: "Towards a Forensic Event Ontology to Assist Video
  Surveillance-based Vandalism Detection", `10.48550/arxiv.1903.09012` — one of
  very few hits that names vandalism as the target.

**Claim we can now support with a sweep behind it:** *in-vehicle CCTV analysis is
studied as event detection — violence, aggression — on short clips; no published
benchmark or method addresses persistent appearance change (graffiti, damage,
forgotten objects) inside a vehicle from a fixed camera.*

Still cited through the Luna protocol rather than directly, and still not
individually verified: **i-LIDS AVSS 2007**, **PETS2006**, **ABODA**. That is
acceptable — Luna et al. is the citation that carries the protocol — but do not
attribute per-dataset numbers to them without checking.

---

## 9. Where ARSI actually stands — the four positioning sentences

Draft these into the Introduction; each is backed above.

1. **The task is between three literatures.** Persistent appearance change on a
   fixed in-vehicle camera with a per-camera clean reference is neither
   object-centric industrial AD (§2.1), nor clip-level event VAD (§5.1), nor
   classical AOD (§3) — and no public dataset combines the three conditions.
2. **The AOD family it most resembles is still pre-foundation-model.** Background
   subtraction plus a closed-vocabulary classifier, with illumination change as
   the acknowledged failure mode (§3). We replace the classifier with an
   open-vocabulary judge and the illumination problem with an explicit mask.
3. **The proposer decides, not the judge.** Our end-to-end result — object recall
   pinned at 55/73 under every localizer while box quality moves frame recall
   0.871 → 0.968 and region precision 0.694 → 0.896 — is a statement about
   pipeline design that the VLM-anomaly literature has not made, and §5.3 now
   says *why*: LAVAD and VERA decide **temporally** (which frame or segment is
   anomalous) over whole frames, so a question about the spatial proposal stage
   cannot be posed in their formulation at all. This is the paper's strongest
   original claim, and it is now defensible against the three closest systems.
4. **Prompting is calibration, not capability.** Our 12-arm sweep (§6) and
   Borodin's post-PEFT prompt-insensitivity (§5.2) are two halves of one finding.
   Stating it jointly is a real contribution and it justifies the LoRA work as
   the next step rather than an afterthought.

---

## 10. Citation ledger

Every row was retrieved this session. "Evidence" = what I actually saw.
V = verbatim passage seen · A = abstract only · M = metadata only.

| # | Reference | DOI | Supports which claim of ours | Ev |
|---|---|---|---|---|
| 1 | Bergmann et al., MVTec AD, CVPR 2019 | `10.1109/cvpr.2019.00982` | the industrial benchmark our task is *not* | M |
| 2 | Roth et al., PatchCore, CVPR 2022 | `10.1109/cvpr52688.2022.01392` | baseline family a reviewer will demand | V |
| 3 | Batzner et al., EfficientAD, WACV 2024 | `10.1109/wacv57701.2024.00020` | idem, low-latency end | M |
| 4 | Akçay et al., anomalib, ICIP 2022 | `10.1109/icip46576.2022.9897283` | how we would produce that baseline table | M |
| 5 | Guo et al., Dinomaly, CVPR 2025 | `10.1109/cvpr52734.2025.01900` | multi-class UAD; contradicts our *reason* for rejecting it | V |
| 6 | Oquab et al., DINOv2, 2023 | `10.48550/arxiv.2304.07193` | encoder used by our best training-free localizer | M |
| 7 | Damm et al., AnomalyDINO, WACV 2025 | `10.1109/wacv61041.2025.00136` | vision-only ≥ vision-language, training-free — same finding as ours | V |
| 8 | Wang et al., CDnet 2014, CVPRW | `10.1109/cvprw.2014.126` | evaluate the proposal stage without the VLM | V |
| 9 | Luna et al., AOD survey, Sensors 2018 | `10.3390/s18124290` | the evaluation protocol we should adopt | A |
| 10 | Park et al., dual background + Mask R-CNN, 2020 | `10.1109/access.2020.2990618` | AOD is still background subtraction; illumination unsolved | V |
| 11 | Park & Park, Sensors 2019 | `10.3390/s19235114` | idem, illumination robustness | M |
| 12 | Xu et al., RAOD benchmark, 2024 | `10.1109/access.2024.3407955` | AOD benchmarks stay outdoor/road | M |
| 13 | Jeong et al., WinCLIP, CVPR 2023 | `10.1109/cvpr52729.2023.01878` | language in the scoring function (contrast to ours) | M |
| 14 | Zhou et al., AnomalyCLIP, 2023 | `10.48550/arxiv.2310.18961` | idem, object-agnostic prompts | M |
| 15 | Gu et al., AnomalyGPT, AAAI 2024 | `10.1609/aaai.v38i3.27963` | LVLMs weak on localized detail; threshold-free verdicts | V |
| 16 | Ren et al., FM for AD survey, AI Mag. 2025 | `10.1002/aaai.70045` | encoder/detector/interpreter taxonomy; prompt sensitivity | V |
| 17 | Sultani et al., UCF-Crime, CVPR 2018 | `10.1109/cvpr.2018.00678` | the VAD mainstream and why it is not our benchmark | V |
| 18 | Li et al., WSVAD-CLIP, J. Imaging 2025 | `10.3390/jimaging11100354` | current CLIP-flavoured WSVAD | A |
| 19 | **Borodin et al., compact VLMs, J. Imaging 2025** | `10.3390/jimaging11110400` | PEFT is the enabler; prompt sensitivity drops after adaptation | V |
| 20 | Li et al., POPE, EMNLP 2023 | `10.18653/v1/2023.emnlp-main.20` | yes/no probing; severe object hallucination | V |
| 21 | Gu, Jiang et al., LLM-as-a-judge survey, 2026 | `10.1016/j.xinn.2025.101253` | judge reliability as a named methodological problem | M |
| 22 | Liu et al., Grounding DINO | `10.48550/arxiv.2303.05499` | open-vocabulary proposal | V |
| 23 | Cheng et al., YOLO-World, CVPR 2024 | `10.1109/cvpr52733.2024.01599` | **component we actually ship** | M |
| 24 | Akyön et al., SAHI, ICIP 2022 | `10.1109/icip46576.2022.9897990` | prior art for our tiling result | M |
| 25 | Zanella et al., LAVAD, CVPR 2024 | `10.1109/cvpr52733.2024.01753` | closest competitor; captions **whole frames**, no spatial proposal | V |
| 26 | Ye et al., VERA, CVPR 2025 | `10.1109/cvpr52734.2025.00811` | prompts as learnable parameters; segment/frame-level decisions | V |
| 27 | Yang et al., AnomalyRuler, ECCV 2024 | `10.1007/978-3-031-73004-7_18` | few-normal-shot rule induction; reference set used as *language* | V |
| 28 | Ciampi et al., Bus Violence, Sensors 2022 | `10.3390/s22218345` | the in-vehicle CCTV benchmark, and why it is not ours | M |
| 29 | Caetano et al., in-vehicle AD review, 2022 | `10.3390/app121910011` | survey of our deployment domain | M |
| 30 | Duong et al., VAD survey, Sensors 2023 | `10.3390/s23115024` | general shape of the field | M |
| 31 | SanMiguel & Martínez, AVSS 2008 | `10.1109/avss.2008.16` | classical anchor of the AOD lineage | A |
| 32 | Guo et al., Dinomaly2, 2025 (preprint) | `10.48550/arxiv.2510.17611` | full-spectrum successor; sharpens §2.2 | V |

No editorial notices (retraction / concern / correction) were reported by scite
for any of rows 1-32. Rows 25-32 were added after re-authorising scite; the
`references.bib` next to this file now holds 35 Crossref/DataCite entries
(some papers appear twice, arXiv + proceedings).

---

## 11. What is still missing

Ordered by how much it costs us to leave undone.

1. ~~**The anomalib baseline table.**~~ **Done, 2026-08-21** —
   `docs/paper/results.md` §3. PaDiM, PatchCore, Dinomaly and AnomalyDINO
   (anomalib 2.6 reference implementations) on our cameras. It also settled
   something the file did not anticipate: at frame level the task is close to
   solved by an untuned 2021 baseline, so the paper's claims have to be made at
   region and verdict level. What is *still* missing is the public-dataset half —
   CDnet / the Luna protocol per `docs/PUBLIC_DATASETS.md` — so that the method is
   not only ever evaluated on data we built.
2. ~~**Read BEM.**~~ **Done, 2026-08-22.** Park, Lee & Lim,
   `10.48550/arxiv.2604.11714`. It is a *veto*, not a proposer: a training-free,
   weight-frozen module bolted onto a **pretrained detector at inference**, which
   estimates a background embedding from recent unlabelled frames of a fixed
   camera, keeps a prototype memory, and **re-scores the detector's logits** with
   an inverse-similarity penalty — "reducing false positives while maintaining
   recall". Its premise is ours (on a fixed camera the quasi-static background is
   a stable, label-free prior) and its *position* is ours: after the proposal
   stage, suppress-only. **It cannot threaten our priority — it corroborates half
   our claim from a different task.** Two clean separations to state: its prior
   is a single *global frame* embedding, not a per-position reference; and it
   re-scores a **closed-vocabulary** detector, so it can never box something the
   detector did not already propose. Written into §7 of the paper's related work.
3. ~~**AnomalyRuler's perception front-end.**~~ **Done, 2026-08-22** — the ECCV
   chapter is paywalled but the arXiv preprint (`2407.10299`) is open. §3.1
   verbatim: the visual perception module *"utilizes a VLM to convert video
   frames into text descriptions"*, prompt `pv` = *"What are people doing? What
   are in the images other than people?"*. That is a question about the **frame**,
   not about a region. **The claim can now be written as "all three".**
4. ~~**Read Caetano et al.**~~ **Done, 2026-08-22** (`10.3390/app121910011`,
   Appl. Sci., open access). It is far more useful than expected — four quotable
   facts, all in our favour:
   - in-vehicle monitoring is *"a relevant research opportunity that has been
     overlooked in the accessible literature"*, and previous surveys ignore
     **moving backgrounds and frequent illumination changes** specifically;
   - work in this domain is *"still fully dependent on the availability of
     private datasets"* — **this defends our dataset limitation with a citation**
     instead of an apology;
   - the operational priority is *"the identification of unattended objects […]
     as the main challenge in providing safety in public transport"*, after
     PREVENT CSA — our exact task, named as the priority;
   - and their abandonment heuristic is **our person veto**, stated as a design
     principle: an object is abandoned *"if its presence is not expected without
     a person present in the frame"*.
   Their proposed remedy for the data scarcity is **synthetic augmentation**,
   which is the family our 11 inpainted cases belong to — so those stop being a
   weakness to excuse and become a documented method choice.

Resolved since the first draft: LAVAD / VERA / AnomalyRuler are identified, read
and positioned (§5.3); the in-vehicle sweep is done (§8) and Bus Violence
verified; the Dinomaly / AnomalyDINO citation trace is done (§2.2 bis) and found
nobody reporting our cross-camera transfer failure. **As of 2026-08-22 nothing
on this list is open** — items 2–4 above were the last three, and all are now
read and written into the manuscript. The bib carries 41 entries, 38 of them
cited.

---

*Method note. scite's index does cover the CV literature well — arXiv DOIs
(`10.48550/arxiv.*`) and CVPR/WACV/ICIP proceedings all resolve. Its relevance
ranking, however, is weak on loose queries and heavily biased toward 2026
papers; every foundational reference in this file had to be pulled by exact
title. Treat it as a verification and citation-tracing tool, not as a discovery
engine.*
