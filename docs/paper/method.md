# Materials and methods

Draft of §3. Every constant below is the shipped value, read out of the code
rather than remembered: `vlm_05_reference_diff.py`, `tools/dino_localizer.py`,
`tools/dinomaly.py`, `arsi_core/localizers.py`. Where a value was chosen by
measurement, the measurement is named.

---

## 3.1 Deployment and task

The system runs on the fixed interior cameras of a tram. Each camera observes a
rigid scene — seats, floor, poles, panels — from an unchanging viewpoint, at
1280×720, and the vehicle is emptied and inspected at the end of each service.
The task is to flag **persistent appearance change**: an object left behind, a
piece of litter, a graffiti tag, or damage to a fitting. People are explicitly
not anomalies; neither is anything a person is wearing, holding, or sitting on.

This framing has one consequence that shapes the whole method. Because the
viewpoint is fixed and a clean state exists, the detector never has to learn
what a tram looks like in general — it only has to notice that *this* camera's
scene differs from *this* camera's clean reference. The method is therefore
**reference-based** throughout, and every stage compares an inspection frame to
a nominal frame of the same view.

The deployment constraint is that everything runs locally on a single
workstation (NVIDIA RTX 3080 Ti, 12 GB). No frame leaves the machine. This
rules out hosted vision-language APIs and caps the judge at roughly 9B
parameters in 4-bit, which is why model selection is reported as a measurement
(§4.3) and not assumed.

## 3.2 The reference and the occlusion mask

A **reference** is one manually chosen frame of a camera in its clean state.
Nine references cover three trams and nine views.

Windows are the dominant nuisance: exterior motion and daylight change produce
large, fast, entirely uninteresting differences. They are removed by a
**camera-wide occlusion mask** — polygons drawn once per view and filled with
black. The mask is applied identically to the reference and to every inspection
frame at pipeline input, never at frame extraction. Masking only one side would
make the mask itself the largest detected change. Masked pixels are excluded
from every statistic downstream: the photometric channels ignore them
(`BLACK_LEVEL = 12`), and the feature-based proposers drop the patches that
fall inside them before computing any scale estimate.

## 3.3 Region proposal

The pipeline's first stage proposes candidate rectangles; the second judges
them. They fail for different reasons and are improved by different work, so
they are implemented as independently selectable components
(`arsi_core/localizers.py`) that share one contract — a list of boxes in
reference pixel space — and one set of post-filters (§3.4). Swapping the
proposer therefore changes which regions reach the judge, and nothing else.
This is what makes the controlled comparison of §4.2 possible.

Three families are evaluated.

### 3.3.1 Photometric difference (`photo`)

The shipped detector, and the classical choice. Both frames are converted to
grayscale, blurred (Gaussian, radius 3) and differenced; the difference map is
thresholded at `DIFF_THRESHOLD = 40`, downscaled by 4, dilated by 2, and its
connected components become boxes with `MIN_AREA = 500` and
`MAX_AREA = 400 000` px.

A single global threshold cannot both catch faint anomalies and keep the region
count bounded: lowering the base threshold to 25 merges busy frames into single
giant blobs that the area gate then deletes. The base detector is therefore left
untouched and two channels only **add** candidates, as a region-list union (never
a mask union, which would merge the boxes back together):

- a second photometric pass at threshold 30, contributing at most 8 boxes that
  overlap no base region, ranked by salience — this recovers low-contrast solid
  objects such as a dark bottle on a dark floor;
- an **added-edge-energy** channel, `relu(|∇ inspection| − |∇ reference|)`,
  restricted to positions where the reference is locally flat
  (`|∇ ref| < 6.0`), blurred and thresholded at 1.5, contributing at most 4
  boxes. Sensor and JPEG noise is symmetric between the two frames and cancels
  in this difference, whereas a faint tag adds one-sided stroke edges on a flat
  panel. This is what reaches faint graffiti that no global photometric
  threshold can reach without flooding.

On the single-camera benchmark the two channels lift localization recall from
41/45 to 45/45 instances at +52 % candidate regions on anomalous frames.

### 3.3.2 DINOv2 patch features (`dino`)

An AnomalyDINO-style proposer specialised for a fixed camera. Both frames are
resized to 1120 px wide and encoded by a frozen `dinov2_vits14_reg` backbone
(22 M parameters), giving an 80×45 patch grid at 24 px per patch.

Where the original formulation builds one memory bank over all nominal patches —
appropriate for MVTec-style object images, which are centred but not aligned —
our camera is fixed, so a patch is compared **only to nominal patches within a
±1-patch neighbourhood at the same grid position**. That is a far tighter null
hypothesis: a seat cushion is compared to that seat, not to any seat in the
image. The ±1 slack absorbs sub-patch jitter.

Each patch receives the cosine distance to its best nominal match. A patch is
flagged when it exceeds **both** a robust z-score of 4.0 (median and MAD over
valid patches) **and** an absolute cosine-distance floor of 0.08. Both are
needed, and this was measured: the per-frame MAD varies fivefold across our
cases (0.006 on a same-session pair, 0.030 on a frame containing a person), so a
single z-threshold means a 0.043 absolute cut on one frame and 0.225 on another.
On a genuinely near-identical pair the MAD collapses and the z amplifies feature
jitter into 80+ regions; the floor removes those (such frames sit at p99 ≈ 0.07)
while the z continues to handle cross-session pairs, where the whole distance
map shifts up. For scale: true anomalous patches reach 0.37–0.69, and the
faintest true instance we measured peaks at 0.126.

The resulting binary map is passed through the *same* `find_regions` routine as
the photometric channels, so box geometry is produced identically in both
families.

### 3.3.3 Reconstruction veto (`dino+dinomaly`)

An optional third arm in which the feature proposer draws the boxes and a
per-camera model trained on that camera's nominal footage deletes the ones it
can reconstruct. We re-implemented Dinomaly (CVPR 2025) — a reconstruction
model on frozen DINOv2 features with a noisy bottleneck, linear attention, and
a loose reconstruction constraint — and apply it as a veto at a threshold of
0.05 on the reconstruction error inside each proposed box.

A veto can only delete boxes, so this arm's accuracy ceiling is exactly that of
the proposer it filters. It is included because reaching that ceiling at half
the cost is itself informative (§4.2), and because it requires a checkpoint per
camera, which §4.6 quantifies as a genuine deployment cost.

### 3.3.4 Gated variants

For completeness the sweep also includes `photo+dino`, in which the photometric
diff proposes and DINOv2 features veto. It is reported because it isolates the
proposer/veto distinction: it shares the photometric proposer's boxes exactly.

## 3.4 Shared post-processing

Every proposer feeds the same four steps, in this order. Holding them fixed is
what makes §4.2 a controlled comparison rather than a comparison of pipelines.

1. **Merge.** One real object rarely produces one connected component: a
   backpack and its strap, a phone and the sliver of seat edge beside it, land
   separately, and the extra channels then add their own boxes on top. Boxes
   whose gap is below 24 px are merged when the union box is at least 50 % filled
   by its parts; a union emptier than that is treated as two objects and left
   apart. Without this step the same object reached the judge as two to five
   independent fragments.
2. **Person veto.** YOLOv8-nano runs once per inspection frame (~20 ms on GPU),
   and candidate regions whose intersection-over-own-area with a person box
   reaches 0.6 are dropped before the judge. This is what separates "jacket worn
   by a passenger" from "jacket forgotten on a seat", which no label blacklist
   can do, since a forgotten jacket is a real anomaly. Verified to lose zero
   ground-truth instances. If the detector is unavailable the filter degrades to
   a warning and keeps every region.
3. **Salience cap.** At most 25 regions per frame, ranked by salience, bounding
   worst-case judge cost.
4. **Crop rendering.** Each surviving box is rendered as a side-by-side pair —
   reference on the left, inspection on the right — with 40 px plus 75 % of the
   region size as context, and each half upscaled to at least 320 px on its
   shorter side. The upscale is not cosmetic: a phone on a seat occupies ~50 px,
   and without it the judge dismisses small dark objects as reflections.

Regions dropped at steps 2 and 3 never reach the judge and are recorded as
counts only; every region that does reach it is recorded with its verdict and
outcome, which is what allows the localization-only and end-to-end views of the
same run to be reconciled.

## 3.5 The judge

The judge is a local vision-language model served by Ollama, asked a **yes/no
question about one crop pair at a time**. It never sees the whole frame, and it
is never asked to localize anything — the box is already drawn. This is a
deliberate restriction of the language model to the one operation the
hallucination literature finds it most reliable at (a binary probe, cf. POPE),
and it is what makes the two stages separable.

The shipped judge is **GLM-4.6V-Flash-9B** at temperature 0.1, 4096 context,
512 predicted tokens. The prompt states the four anomaly classes (forgotten
object, litter, graffiti, damage), states three explicit NO conditions
(brightness/shadow/reflection change only; a person or anything they wear, hold
or sit on; scratches and glare on metal or glass), and instructs the model to
answer YES only if it can name a specific new object, marking, or damage. The
reply is `YES`/`NO` followed by a two-to-four-word name.

The named answer is used: three post-filters override a YES when the named
label is a person or body part, when it reports a *disappearance* rather than an
appearance, and when a small-object word ("phone", "wallet", "key") is attached
to a region larger than 6000 px. These fire only on a YES, so a rejection is
never misattributed to a filter.

Frame verdict: a frame is flagged when at least one region survives.

## 3.6 Data

Sixty-eight labelled cases over three trams and nine camera views: 31 anomalous
and 37 clean, carrying 73 ground-truth instances. By type: 53 forgotten objects,
7 graffiti, 7 litter, 6 damage. By provenance: 54 frames from real cameras, 11
AI-inpainted anomalies composited onto real scenes of the same cameras, 2
self-staged, 1 viewpoint variant. Every case names the reference it is compared
against, and instance boxes are in that reference's pixel space.

The inpainted cases are labelled as such and reported separately where it
matters; they exist because several anomaly types (damage, graffiti) are rare
enough in a working tram that waiting for them was not an option within the
study period. People are never instances: a kept box on a person counts as a
false positive.

## 3.7 Metrics

Three levels, because the paper's claim is precisely that they move
independently.

- **Frame level** (binary): accuracy, precision, recall, specificity, F1 over
  the 68 cases.
- **Object level**: a ground-truth instance counts as detected if any kept
  region overlaps it under a lenient rule (IoU > 0.1 or centre containment); a
  strict count at IoU ≥ 0.3 is reported alongside, and the gap between the two
  is the box-quality signal. Kept regions matching no instance are
  false-positive regions; **region precision** is the fraction of kept regions
  that match one.
- **Image-level AUROC** for the comparisons against industrial anomaly-detection
  baselines (§4.4, §4.6), defined as
  `P(score(anomalous) > score(clean))` with
  `score(image) = mean of the top 1 % of patch anomaly scores`, masked patches
  excluded. This is the standard metric of that literature and is used so our
  numbers can sit in the same table as published ones.

One reporting rule is applied throughout: **any recall is read next to the
judge's YES rate.** With 73 instances among 654 proposed regions, a calibrated
judge answers YES about 11 % of the time. A judge answering YES to 88 % of crops
also recovers most instances, and that is non-rejection, not detection.

## 3.8 Public-data protocol

To evaluate the proposal stage on data we did not build, we run it on **CDnet
2014** across seven categories. The reference is the per-pixel median of up to
50 frames preceding each sequence's `temporalROI`, which reproduces our clean
reference by construction. Official label semantics are respected (static /
hard shadow / outside-ROI / unknown / motion), the spatial ROI is applied, and
the person veto is **disabled** — on CDnet the annotated foreground *is* people,
so leaving it on would measure the veto rather than the proposer.

Two views of each configuration are reported: the per-pixel change mask, and the
rasterised proposal boxes, which is what actually reaches the judge. Both are
scored with the seven official CDnet metrics.

## 3.9 Reproducibility

All experiments run on one NVIDIA RTX 3080 Ti (12 GB). Models: DINOv2 ViT-S/14
with registers (frozen), YOLOv8-nano, GLM-4.6V-Flash-9B in 4-bit via Ollama, and
our Dinomaly / Dinomaly2 re-implementations. Baselines use anomalib 2.6
reference implementations.

Per-region judge verdicts are cached on a key covering image, reference, box,
model, prompt and occlusion mask, so a re-run reproduces a previous run exactly
and a changed mask or prompt correctly invalidates it. The verdict cache key
deliberately excludes the proposer: a box is a box, so identical coordinates
from two proposers share one verdict — which is also what makes the
proposer sweep of §4.2 cheap.

The code, the ground truth, the per-run reports and the decision log are in the
project repository. The tram footage itself cannot be released; the CDnet 2014
evaluation (§4.5) exists so that the proposal stage's behaviour is verifiable on
public data.
