# Public datasets for publishing the reference-diff + VLM approach

The goal of this document is to settle the
question: which public benchmarks do we evaluate on, so that a paper about
`vlm_05_reference_diff.py` is not rejected on the grounds that it was only ever
tested on our own data from DPO tram.

## The problem

What ARSI does is specific. Fixed CCTV cameras inside tram interiors, looking for
changes that persist: graffiti, vandalism, forgotten objects. The method is to
subtract an inspection frame from a clean reference frame of the same camera, and
hand each region that differs to a vision-language model for judgement.

No public dataset covers that combination. In-vehicle, fixed camera, and ground
truth for graffiti or abandoned objects simply do not exist together in anything
published.

So the plan is not to go looking for one dataset that does everything. The
pipeline is really three separate problems stacked on top of each other: detecting
that something in the image changed, deciding whether that change is persistent
and abnormal, and interpreting what it actually is. Each of those gets evaluated
on the benchmark where it is meaningful, and the Ostrava tram data becomes the
deployment section rather than the only evidence in the paper.

I don't think having our own dataset is not what gets a paper rejected. Having our own dataset
and nothing else is.

## Where the compact-VLM paper stands

Borodin et al., *Benchmarking Compact VLMs for Clip-Level Surveillance Anomaly
Detection Under Weak Supervision*, J. Imaging 11(11):400, November 2025.  Everything they report is on
UCF-Crime dataset.

UCF-Crime is an anomaly dataset: 1900 untrimmed videos, 128 hours, 13
anomaly classes, with labels given at the level of the whole video rather than
per frame (this is what "weak supervision" means here). The reason it cannot
carry our main result is not that the anomalies are the wrong kind. It is that
every video comes from a different scene and a different camera, and there is no
clean footage of that same camera to use as a reference. Our differencing stage
has nothing to subtract from.

## The evaluation plan

| Question we want to answer                                                                         | Benchmark | What we measure |
|----------------------------------------------------------------------------------------------------|---|---|
| Does it detect changes correctly?                                                                  | CDnet 2014 | precision, recall, F1 and false-positive rate, computed per pixel |
| How badly do camera shake, shadows, changing light and moving backgrounds hurt it?                 | CDnet 2014, broken down by category | how much each category degrades the score |
| Do persistent changes actually turn into abandoned-object detections?                              | i-LIDS AVSS 2007, PETS2006 and PETS2007, ABODA, using the Luna et al. protocol | precision, recall and F1 at the level of events rather than pixels |
| How does our VLM stage compare with recent work?                                                   | UCF-Crime, following the Borodin protocol | accuracy, precision, recall, F1, ROC-AUC, latency |
| Does the system meet the actual tram requirement? Graffiti, damaged seats, stains, phones, bottles | our Ostrava data | precision, recall and F1 per class, plus ablations |

Read as a paper outline, that becomes:

```
1. CDnet 2014   evaluate the change-proposal mechanism against
                established change-detection methods
2. PETS / AVSS  show that persistent changes translate into real
                abandoned-object detections
3. UCF-Crime    compare the VLM stage against recent compact-VLM work
4. Ostrava      evaluate the complete system in the deployment domain,
                including anomaly classes no public benchmark covers
```

## Notes on each dataset

### CDnet 2014

Fifty-three sequences across 11 categories, all from static cameras, with ground
truth given for every pixel of every frame. The category called
`intermittentObjectMotion` is defined by its authors as "videos containing
background objects moving away, abandoned objects and objects stopping for a
short while and then moving away", which is our case almost word for word. Its
six sequences:

| Sequence | Frames |
|---|---|
| abandonedBox | 4 500 |
| parking | 2 500 |
| streetLight | 3 200 |
| sofa | 2 750 |
| tramstop | 3 200 |
| winterDriveway | 2 500 |


The other categories let us stress the differencing stage exactly where we
already suspect it is weak: `shadow`, `dynamicBackground`, `cameraJitter`,
`badWeather`, `nightVideos` and `lowFramerate`.

Now the caveat. **CDnet is not an anomaly-detection benchmark.**
Its ground truth marks each pixel as one of: static (0), hard shadow (50),
outside the region of interest (85), unknown motion such as motion blur or
semi-transparency (170), or moving foreground (255). Nothing in there knows that
a bottle has been forgotten, that graffiti is a problem, or that a seat has been
slashed. So the claim we are entitled to make is:

> Correct: "our change-detection stage reaches F1 = 0.84 on
> `intermittentObjectMotion`"
>
> Not correct: "our anomaly detector reaches F1 = 0.84"

Within that limit it is very defensible, and it buys us the one ablation we are
currently missing. **CDnet lets us evaluate `vlm_05_reference_diff.py` without
the VLM in the loop.** When the full pipeline gets something wrong, we will
finally be able to say whether the differencing stage failed to propose the
region, or whether it proposed it correctly and the VLM misjudged it.
`LORA_PLAN.md` already claims localization is not the bottleneck, based on 45 out
of 45 ground-truth objects found on our own data; CDnet is what lets us make that
claim on data someone else built.

- Paper: <https://openaccess.thecvf.com/content_cvpr_workshops_2014/W12/papers/Wang_CDnet_2014_An_2014_CVPR_paper.pdf>
- Site: <https://changedetection.net/dataset2014/> (there is also a Kaggle mirror)

### Abandoned objects: use the Luna protocol

**i-LIDS AVSS 2007** has an explicit Abandoned Baggage challenge filmed in a
railway station. Its official definition is useful to us on its own: the owner
must have left the area and not come back for 60 seconds, and the object types
covered are bottle, can, suitcase, newspaper, paper, rucksack and sports bag.
That is close to what ARSI means by a forgotten object, persistence criterion
included.

**PETS2006** is seven scenarios of abandoned luggage in a station, filmed
simultaneously from four viewpoints at 25 frames per second, with ground truth
and increasing levels of difficulty.

**ABODA** is 11 CCTV sequences of unattended objects, indoors and outdoors, day
and night, with strong lighting changes. Semantically it is exactly our class,
but 11 sequences is too little to carry a main results table, and the licence is
not clearly stated anywhere. Use it for qualitative checks.

The useful find here is a paper rather than a dataset. Luna et al., "Abandoned
Object Detection in Video-Surveillance: Survey and Comparison", Sensors
18(12):4290, 2018, builds a *common evaluation protocol* across AVSS AB 2007,
PETS2006, PETS2007, ABODA and VISOR. It unifies the temporal annotations for
abandoned objects and provides baseline results for each stage of the usual
pipeline (Mixture of Gaussians, K-nearest neighbours, PAWCS, and others, all
background-subtraction methods). Adopting it is much stronger than assembling
our own evaluation.

The original hosting for AVSS 2007
is unreliable, but mirrors exist and the authors' own package is reportedly still
available.

- <https://mdpi.com/1424-8220/18/12/4290/htm>
- AVSS 2007 challenge definition: <https://www.eecs.qmul.ac.uk/~andrea/dwnld/avss2007/>

### Bus Violence: useful, but only if we describe it accurately

Ciampi et al., Sensors 22(21):8345, 2022, from CNR-ISTI in Pisa. It contains 1400
clips, 700 with a violent action and 700 without, filmed by three cameras inside
a moving bus: two in the corners at 960×540 and one fisheye in the middle at
1280×960, all at 25 frames per second. The whole thing is about 449 MB, hosted on
Zenodo as record 7044203.

Two facts kill the obvious framing of it:

1. **The clips are very short: between 16 and 48 frames at 25 frames per second,
   so 0.6 to 1.9 seconds each.** This is not a test of stability over ten minutes
   of a tram moving from sunlight into shade, and it must not be described as
   one.
2. **"NoViolence" does not mean "normal" in our sense.** The label only asserts
   that no violent action takes place. It says nothing about whether a bag was
   left behind, whether there is a stain, whether something is damaged, or
   whether there is graffiti on the wall. Those 700 clips are not 700 verified
   negatives for our task and cannot be counted as such.

- <https://pmc.ncbi.nlm.nih.gov/articles/PMC9658862/> and <https://zenodo.org/records/7044203>


### UCF-Crime: a separate compatibility experiment

This is not our main benchmark. To end up in the same table as the compact
vision-language models, it is not enough to use the same dataset; the whole
protocol has to be reproduced. Verified from the paper:

```
Split       the official one, no filtering, no separate validation subset
Training    1610 clips (800 normal, 810 abnormal)
Test        290 clips  (150 normal, 140 abnormal)
Sampling    videos under 60 s:   0.5 frames per second across the whole video
            videos 60 s or more: 32 frames spread evenly across the timeline
            no audio, no optical flow
Decision    one binary label per clip, inherited from the source video;
            no temporal localization. Output constrained to 0 or 1, except
            in chain-of-thought mode where the reasoning precedes the digit
Models      2-4B: Gemma-3-4B, InternVL3-2B, Qwen2.5-VL-3B
            7-8B: InternVL3-8B, Qwen2.5-VL-7B, Video-LLaVA-7B
Metrics     accuracy, precision, recall, F1, ROC-AUC, plus wall-clock
            latency per clip with 95% confidence intervals
```

ARSI works frame by frame, differencing then cropping then querying the model, so
comparing our F1 to theirs without matching the unit of evaluation would be
indefensible. Better to make it its own experiment, something like "compatibility
with an established VLM anomaly-detection protocol", where ARSI is deliberately
adapted to their evaluation unit. That gives a strong ablation:

| Method | Acc | Precision | Recall | F1 | ROC-AUC | Latency |
|---|---|---|---|---|---|---|
| Qwen2.5-VL, from Borodin et al. | | | | | | |
| InternVL3, from Borodin et al. | | | | | | |
| **ARSI, VLM only** | | | | | | |
| **ARSI, differencing + VLM** | | | | | | |

The question it answers is how much the change-detection front end really adds
over the model on its own. Note that on UCF-Crime the differencing stage is
degraded or switched off entirely, because there is no reference frame to
subtract. Say so explicitly. "Our pipeline reduces to X when no reference is
available" reads as a characterisation of the method, not as a weakness.

The same two-by-two comparison (off-the-shelf versus fine-tuned, full frame
versus differencing) appears in `LORA_PLAN.md` under "An open research question".
The fine-tuning work is what produces the cells this table needs.


## Gaps worth knowing about

**Graffiti has no serious academic benchmark.** There is a roughly 707-image
annotated set on Roboflow and a handful of YOLOv8 conference papers, but nothing
citable at the level of a vision venue. UCF-Crime does have a `Vandalism` class
whose definition mentions graffiti and defacement, but it frames the problem
differently in kind:

```
UCF-Crime   a person commits vandalism        →  an event in a video
ARSI        clean wall → wall with graffiti   →  a persistent change of state
```

So it is not a substitute for the Ostrava data. That gap is an argument *for* our
dataset rather than a weakness in it, and it should be stated explicitly in the
paper.

**If the framing later shifts** from producing a detection score to having the
model explain the anomaly, the relevant benchmarks are UCA (23 542 sentence-level
annotations over 1854 UCF-Crime videos), CUVA and its successor ECVA, HAWK,
VAU-Bench and Holmes-VAU. Given how the pipeline currently works, differencing
then cropping then judging the crop pair, the fixed-camera detection axis is the
better fit.

## Metric conventions to respect

Each community reports different numbers, and using the wrong ones makes results
look non-comparable even when they are not.

- Change detection (CDnet 2014): F-measure per category, alongside the usual
  table of recall, precision, false-positive rate, false-negative rate and
  percentage of wrong classifications.
- Abandoned-object detection (AVSS, PETS, ABODA): precision, recall and F1 at the
  level of events, following the Luna et al. protocol.
- Clip-level vision-language models (UCF-Crime, Borodin protocol): accuracy,
  precision, recall, F1 and ROC-AUC, plus latency per clip with 95% confidence
  intervals.
- Semi-supervised fixed-camera anomaly detection, should we ever add a classic
  row such as ShanghaiTech: area under the ROC curve computed per frame.
