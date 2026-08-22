# The paper — outline, claim, and section budget

Written 2026-08-22. This file is the map: what the paper argues, in what order,
which measurement carries which paragraph, and what is still missing. Read it
before touching any other file in `docs/paper/`.

---

## 0. The one decision everything else follows from

**Headline claim: the proposer decides, not the judge.**

Everything measurable in this project points at the same thing. On 68 cases,
73 instances, three trams and nine views, swapping the region-proposal stage
under a *frozen* vision-language judge moves frame recall 0.871 → 0.968 and
region precision 0.694 → 0.896 — while object recall does not move at all
(55/73 in every arm). Meanwhile a twelve-arm sweep over four VLMs and three
prompts fails to beat the shipped judge at usable specificity.

So the paper is not "we built a tram anomaly detector". That paper is not
publishable: §3 of `results.md` shows an untuned 2021 baseline (PaDiM) reaches
0.915 I-AUROC on our main camera, so frame-level detection is close to solved
and claiming it as a contribution would be wrong. The paper is:

> In a reference-based VLM anomaly pipeline, the accuracy ceiling is set by the
> spatial proposal stage, not by the language model that judges it — and the
> proposal stage should compare *features*, not *pixels*.

That claim is **falsifiable, cheap to test, and not made anywhere in the
literature we surveyed**: LAVAD (CVPR 2024) and VERA (CVPR 2025) both decide
*temporally* over whole frames, so the question cannot even be posed in their
formulation (`related_work.md` §5.3). That is the priority argument, and it
survived the check.

### What this decision costs

Two things that feel like results have to be **demoted to supporting evidence**,
not abandoned:

- the system working end to end (it becomes the setup, not the finding);
- the judge sweep (it becomes the *control* that keeps the headline honest —
  without it, a reviewer says "you just picked a bad judge").

And one thing gets promoted: the localizer comparison table, currently buried in
a docstring in `arsi_core/localizers.py`, becomes Table 1.

---

## 0 bis. The manuscript itself

`docs/paper/latex/` — build with `make`, read `latex/README.md` first.

Content is separated from publisher formatting on purpose: `sections/*.tex` is
portable and must never contain a template macro, while `main.tex`,
`preamble.tex` and `metadata.tex` are the three files a conversion replaces.
`README.md` there carries the step-by-step for MDPI and for IEEE, and the only
section-file edit either conversion needs (`table` → `table*` for the three
wide tables in two-column layouts).

Structural and stylistic conventions were taken from four papers by Pavol
Partila and collaborators, chiefly Golej et al., *People Detection Using
Artificial Intelligence with Panchromatic Satellite Images*, Appl. Sci. 2024,
14(18):8555 — an MDPI experimental computer-vision paper with the section
hierarchy, table style, and back-matter statements this manuscript follows.
Two conventions were adopted from it directly: numbered sections with a
Materials-and-Methods spine, and stating sample sizes before any metric that
rests on them. This confirms the venue reasoning below from a second direction
— that group already publishes in exactly this family.

## 1. Venue, and why

**Target: MDPI *Journal of Imaging*, special issue on video surveillance or
anomaly detection. Fallback: MDPI *Sensors*.**

Not fashion — this is literally where the conversation is happening. The paper
we are most directly in dialogue with (Borodin et al., compact VLMs for video
anomaly, `10.3390/jimaging11110400`) is in *J. Imaging*. The abandoned-object
evaluation protocol we adopt (Luna et al., `10.3390/s18124290`) is *Sensors*.
The in-vehicle review (Caetano et al., `10.3390/app121910011`) is *Applied
Sciences*. The dual-background AOD baseline, the in-vehicle CCTV benchmark
(Bus Violence) — all MDPI. Five of our thirty-five references are in the family.

Three practical reasons on top of the fit:

1. **The structure is prescribed and forgiving.** Introduction / Related Work /
   Materials and Methods / Results / Discussion / Conclusions. For a first
   paper that is a scaffold, not a constraint — a CVPR-style paper has to *earn*
   its structure with a novelty leap, which we do not have and do not need.
2. **Thorough evaluation is rewarded there.** Our strength is that we measured
   six things properly, including two that came out negative. A CV conference
   would read that as a thin contribution; this venue reads it as rigour.
3. **No private-dataset veto.** Our main data cannot be released. Conference
   reviewers increasingly treat that as fatal; the CDnet 2014 half of the
   evaluation (§4 of `results.md`) is what makes it acceptable anywhere, and it
   matters more here than the private half.

**Article processing charges are covered** (confirmed 2026-08-22), so the APC
is not a constraint on the choice and IEEE Access is equally open as a
fallback — Pavol and collaborators publish there too.

Nothing in the manuscript's section files changes with the venue. Only the
title block, the preamble and the column count do, which is exactly why the
LaTeX is split the way `latex/README.md` describes.

---

## 2. Title and the four sentences of the abstract

**Working title.**

> Who Proposes, Decides: Region Proposal Governs Reference-Based Anomaly
> Detection with Vision-Language Judges in Tram CCTV

Alternative if the venue wants the application first:

> Feature-Based Region Proposal Sets the Accuracy Ceiling of Vision-Language
> Anomaly Detection in Fixed-Camera Public Transport

**Abstract, four sentences, in this order** (write it LAST, when Table 1 is
frozen):

1. *Problem.* Persistent appearance change on a fixed in-vehicle camera —
   forgotten objects, litter, graffiti, damage — is a task that sits between
   industrial anomaly detection, clip-level video anomaly detection, and
   classical abandoned-object detection, and no public benchmark combines its
   three conditions.
2. *What we do.* We build a fully local, training-free-at-inference pipeline
   that proposes candidate regions against a per-camera clean reference and
   asks a 9B vision-language model a yes/no question about each one, and we
   sweep the two stages independently.
3. *The finding.* Under a frozen judge, replacing the photometric proposer with
   a DINOv2 feature proposer lifts frame F1 from 0.931 to 0.984 and region
   precision from 0.694 to 0.896 at unchanged object recall, while twelve
   model × prompt arms fail to move the judge past its own ceiling — the
   proposal stage, not the language model, is the binding constraint.
4. *External validity.* On CDnet 2014 the same proposal stage recalls 77–100 %
   of annotated foreground boxes across seven perturbation categories, and we
   report where it fails.

---

## 3. Section-by-section plan

Word counts are for a ~8000-word journal paper. Halve them for a workshop paper
and drop §6 into §5.

**As of 2026-08-22 the manuscript is drafted end to end** in
`docs/paper/latex/` (publisher-neutral LaTeX, ~9 400 words of body text,
22 pages at 11pt single-column, builds clean with `make`). The table below is
now a status map rather than a to-do list; what remains is listed in §5.

| § | Section file | Words | Status | Evidence it carries |
|---|---|---|---|---|
| 1 | `01_introduction.tex` | 900 | **drafted** | the 4 positioning sentences, `related_work.md` §9 |
| 2 | `02_related_work.tex` | 1100 | **drafted** (compressed from 627 lines) | `related_work.md`, 32 traced refs |
| 3 | `03_methodology.tex` | 1900 | **drafted** | `method.md` |
| 4 | `04_experimental_setup.tex` | 1000 | **drafted** | data, metrics, protocols |
| 5 | `05_results.tex` | 3100 | **drafted** | `results.md` §0–§5, 6 tables, 2 figures |
| 6 | `06_discussion.tex` | 1000 | **drafted** | limits, threats to validity |
| 7 | `07_conclusion.tex` | 400 | **drafted** | — |

Note the structure changed once the model papers were read: Experimental Setup
is now its own section, split out of Materials and Methods. The reason is that
a reviewer checking whether a number is *fair* reads a different section from
one checking what was *built*, and Golej et al. keep them together only because
MDPI's template pushes them together. Splitting converts cleanly in both
directions; merging back is one file concatenation.

### §1 Introduction — the funnel

Four paragraphs, one per positioning sentence of `related_work.md` §9, in that
order. Paragraph 3 is the contribution paragraph and must state the headline
claim as a *finding*, not as a design choice. End with an explicit
contributions list of four bullets:

1. A reference-based, fully local VLM pipeline for in-vehicle CCTV anomaly
   detection, with the proposal stage and the judge as independently
   selectable components.
2. The controlled two-stage sweep that isolates them, showing the proposal
   stage sets the ceiling.
3. Evidence that prompt and model selection are *calibration* and are exhausted
   at this scale — jointly with Borodin et al.'s post-PEFT prompt insensitivity.
4. An evaluation of the proposal stage on public data (CDnet 2014, seven
   categories) so the method is not only ever evaluated on data we built.

### §2 Related work — compress, do not rewrite

`related_work.md` is 627 lines because it is a working file with a citation
ledger and verbatim evidence. The paper needs ~1600 words. Keep the axis
(propose → judge), keep §2.2/§2.3 (Dinomaly, AnomalyDINO), §3 (AOD), §5.3 (the
three training-free VLM pipelines — this is the priority argument and must not
be trimmed), §6 (POPE / judge reliability). Compress §2.1, §4, §5.1 to one
paragraph each. Drop the ledger and §11; they stay in the repo.

### §3 Materials and methods — see `method.md`

Written. It has to answer, in order: what the data is, what a reference is, how
regions are proposed (three families), what the judge is asked, what the
post-filters do, and what is measured. The reproducibility paragraph at the end
is not optional at this venue.

### §4 Results — the order is the argument

This ordering is deliberate: each subsection removes an objection to the one
before it.

| 4.x | Content | Source | Objection it kills |
|---|---|---|---|
| 4.1 | Dataset and protocol | `method.md` §5 | — |
| 4.2 | **Table 1** — four localizers, frozen judge | `results.md` §0 | *the* result |
| 4.3 | **Table 2** — 12-arm judge sweep | `results.md` §0b | "you picked a weak judge" |
| 4.4 | **Table 3** — anomalib baselines | `results.md` §3 | "is any of this needed?" |
| 4.5 | **Table 4** — CDnet, 7 categories | `results.md` §4 | "private data only" |
| 4.6 | **Table 5–6** — cross-camera, Dinomaly2 | `results.md` §1, §2, §2b | "why not just train per camera?" |
| 4.7 | Qualitative: ABODA | `results.md` §5 | "does it generalise in class?" |

### §5 Discussion — write the limits before a reviewer does

Five paragraphs, and every one of them is already known:

1. **Object recall is stuck at 55/73 and better boxes do not fix it.** This is
   our own headline's boundary: the proposer sets the *ceiling*, and 18 misses
   sit at the judge. Say it, and name the next lever (LoRA, `docs/LORA_PLAN.md`).
2. **The private dataset.** 68 cases, 31 anomalous, one deployment. CDnet is
   the mitigation, not a fix.
3. **Per-camera cost.** A foreign Dinomaly checkpoint is blind, not
   miscalibrated (2.13× score ratio, 41× collapse in separation), so the veto
   arm needs one checkpoint per camera. That is a deployment cost, and CAR
   (Dinomaly2) halves the gap without closing it.
4. **The prompt result is a *limit*, not a law.** VERA treats prompts as
   learnable parameters and gains from it. State ours as "exhausted at this
   scale, for this lot of models, without adaptation" — never as "prompts do
   not matter". This is the single sentence most likely to get us a hostile
   review if phrased loosely.
5. **The proposal stage assumes a valid reference.** CDnet `tramstop` 0.138 and
   `winterDriveway` 0.143 are exactly the failure mode: when the reference goes
   stale, the stage collapses. In a tram, that is the end-of-service refresh.

### §6 Conclusions

Three sentences plus future work. Future work is LoRA (`docs/LORA_PLAN.md`) and
the Luna event-level protocol once annotations are obtained.

---

## 4. Figures — four, and the first one is not optional

| Fig | Content | Status |
|---|---|---|
| 1 | Pipeline diagram: reference → mask → proposer (3 families) → person veto → merge → crop pair → judge → verdict | **to make** |
| 2 | Side-by-side failure: pixel diff's 98.9 %-of-frame box on 3333 vs AnomalyDINO's box on the same frame | **done** — `docs/paper/figures/fig2_proposers.jpg`, `tools/figure_proposer.py` |
| 3 | Qualitative ABODA: the bag boxed alone once people leave | **done** — in the manuscript as `fig4_aboda.jpg` |
| 4 | CDnet per-category bar chart (mask F1 and boxes recall) | data exists, **optional** — Table 6 already carries it, and a bar chart of seven numbers earns its space only if the page budget allows |

Figure 2 is the one that makes the headline claim *visible* in one look, and it
is rendered: on `3333_cam55_083517` the pixel diff scores IoU 0.042 on the
laptop and 0.016 on the floor bag — whose only photometric cover is the
98.9 %-of-frame box — against 0.555 and 0.408 for the feature proposer. Both
panels carry the per-instance IoU and the strict-rule verdict, so the figure is
checkable without the caption.

---

## 5. What is missing, in the order it blocks writing

1. **Michal Plytik's department and e-mail**, the only hole left in
   `metadata.tex`. Author order is settled: Alquier, Plytik, Partila — first
   author did the work, last author supervised, which is the convention this
   group follows (Golej et al. is exactly that shape).
2. ~~Figure 1.~~ **Drawn 2026-08-22**, in TikZ, `figures/fig1_pipeline.tex` —
   it encodes the design claim visually: one dashed box is swappable, everything
   after it is dotted and held fixed.
3. **The back-matter TODOs** in `backmatter.tex`: contribution initials, the
   grant number, and whether an ethics statement is required for operational
   CCTV footage containing identifiable passengers. That last one is a real
   question for the supervisor, not boilerplate.
4. ~~The three reading debts.~~ **Closed 2026-08-22.** BEM turned out to be a
   veto, not a proposer, so it corroborates us rather than threatening priority;
   AnomalyRuler's perception module was read from the arXiv preprint and *does*
   consume whole frames, so the claim is now "all three"; and Caetano et al.
   supplies four quotable facts that make our dataset limitation citable rather
   than apologetic. See `related_work.md` §11.
5. **The Luna event-level protocol.** Blocked on annotations we do not have.
   Stays in the Discussion as declared future work — do not let it hold the
   paper.

Nothing on this list requires GPU time. The measurement phase is over.

---

## 6. How to work on this if you have not written a paper before

Three habits, in decreasing order of how much they will save you:

**Write §3 and §4 first, §1 and §2 last.** Methods and results are the parts
where you already know more than any reader. Introductions are written
backwards, once you know exactly what you are introducing. Almost every first
paper is written in the wrong order and rewritten twice because of it.

**One claim per paragraph, and the evidence in the same paragraph.** If a
paragraph makes a claim whose number lives in a different section, a reviewer
will not go find it. Our claims are all already attached to a measurement —
`results.md` and the ledger in `related_work.md` §10 exist so that this is a
lookup, not a memory exercise.

**Never write a number you cannot re-derive.** Every figure in `results.md`
comes with the script that produced it. If a number ends up in the paper with
no script behind it, that is the one a reviewer will ask about.

And one thing to expect rather than fear: a negative result that was measured
properly (our Dinomaly2 port, our cross-camera transfer) is worth more in this
kind of paper than a positive result that was not. Both are already written up
as verdicts in `docs/DECISIONS.md`, which is unusual and good.
