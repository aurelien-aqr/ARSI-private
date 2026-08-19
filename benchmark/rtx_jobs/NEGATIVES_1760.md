# 1760 clean-frame negatives — specificity holds on a third tram

## The dataset

18 clean frames from tram DPO-1760 (04/08/2026, 09:00-09:10, a moving run of the
line), three interior cameras (cam04, cam13, cam06), one reference per camera at
t=70 s and six inspection frames spread over the ten minutes — so every case
pairs the same fixed camera under a visibly different sun. Frames were checked by
eye on a 12-frame strip per camera; cam04 t=20 s was rejected (a passenger is on
the phone in it). No anomalous case, on purpose: this measures specificity only.

Built by `tools/build_1760_benchmark.py` through the hand-drawn LabelMe masks
in `data/masks_labelme/1760/` (~26 polygons per camera, the same source and
format `build_39T_benchmark.py` reads), merged into `ground_truth`
(50 -> 68 cases, 6 -> 9 references). Why it exists: specificity rested on 12
clean 1762 frames and 7 clean 39T ones. **The negatives went from 19 to 37.**

## Result — vlm_05 x GLM-4.6V-Flash-9B:latest x conservative x photo

**18/18 clean, zero false alarms**, 367 regions proposed and every one judged NO.
The result held across three states of these frames — unmasked, coarsely masked,
and correctly masked — so specificity here is not an artefact of the masking.

Consolidated over the 68 cases: frame TP 27 / FP 0 / TN 37 / FN 4, accuracy
0.941, precision 1.000, recall 0.871, **specificity 1.000**, F1 0.931; objects
55/73 = 0.753, strict 0.534, region precision 0.694 (object metrics unchanged —
the 1760 cases carry no instance). Per family: 1762 17/0/12/0, 39T 10/0/7/4,
1760 0/0/18/0.

## Three mistakes made building this, and what they cost

**The frames were first written unmasked**, then through the coarse 5-zone masks
under `data/app/masks/`. The reference source is `data/masks_labelme/1760/`,
which `build_39T_benchmark.py` already pointed at — copying that builder
literally would have avoided both rounds. The protocol is stated in
`bench_runs`: "the benchmark images are already masked, so no mask is applied
here", and it is checkable in one line — pure-black coverage is 27.6 % on the
1762 reference, 18-36 % on the 39T cameras, 29-34 % on these. The first version
had 0.3 %.

**That produced a false finding, now retracted.** On the unmasked frames, six of
eighteen collapsed into a single whole-frame box and the cause looked like merge
chaining clearing `MERGE_MIN_FILL = 0.50` by four points. Re-measured on the
correctly masked frames (localization only, 0 VLM calls):

| config | regions | boxes > 50 % | boxes > 90 % | biggest |
|---|---:|---:|---:|---:|
| merge OFF | 424 | 3 | 1 | 889,920 |
| **fill 0.50 (shipped)** | 367 | 4 | 1 | 889,920 |
| fill 0.75 | 423 | 3 | 1 | 889,920 |

**The biggest box is identical in all three, merge OFF included.** The merge does
not create it. `MERGE_MIN_FILL` is not a per-camera parameter; that claim came
from the unmasked frames and is withdrawn. The shipped 0.50 earns its place here
too — 367 regions against 424, for one extra oversized box.

**The verdict cache is keyed on file NAMES**, not content
(`ref_name|img_name|bbox|model|prompt`). Rewriting the frames in place would have
scored the new pixels against verdicts earned on the old ones, silently. Those
entries were purged by hand both times. `VerdictCache.drop_changed()` now does it
automatically: one sha1 per input image, an existing key scheme left untouched,
and a `cache_invalidated` event in the job log.

## What survives

Oversized regions on moving trams are real, they come from the photometric diff
itself, and **proper masking does not remove them**: 4 of these 18 frames still
carry a box covering more than half the frame and one covers 97 % of it, against
911,360 px (99 %) on 39T where the merge A/B had already shown the same
independence. On those frames the judge is shown a crop in which nothing small
can be seen — the failure `TILING.md` measures and partly repairs.

That also trims the "nothing tuned on 1762 transfers" line to what is actually
measured: the DINOv2 gate (8 % of regions removed on 39T against 57 % on 1762)
and tiling (buys recall on 39T, costs specificity on 1762). Two settings, not
three — the merge transfers fine.
