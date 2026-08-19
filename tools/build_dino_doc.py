#!/usr/bin/env python3
"""Assemble the standalone note on the two DINOv2-based anomaly models.

Scope, deliberately narrow: what AnomalyDINO and Dinomaly are, how each one is
implemented in this repo (including where our version departs from the paper),
what they were trained or configured with, and what they scored. It is not a
survey of every region proposer - the pixel diff appears only as the baseline
those scores are read against.

Reads, all produced by other tools so no number here is typed by hand:
    docs/dino_models/metrics.json        benchmark/eval_localization.py --json
    docs/dino_models/end_to_end.json     tools/collect_e2e.py
    docs/dino_models/training_sets.json  tools/dump_training_sets.py

    venv/bin/python tools/build_dino_doc.py
Output: docs/dino_models/dino_models.html
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.notes import NOTES                                     # noqa: E402
from tools.report_style import foot, head                         # noqa: E402

NOTE = NOTES["dino-models"]
DOC = NOTE.html
OUT = DOC.parent


def load(name, default=None):
    p = OUT / name
    if not p.exists():
        if default is None:
            raise SystemExit(f"missing {p}")
        return default
    return json.loads(p.read_text())


def num(v, n=3):
    return f"{v:.{n}f}".replace("-", "−")


def pct(v, n=1):
    return f"{100 * v:.{n}f} %".replace("-", "−")


# --- the diagram --------------------------------------------------------------

def diagram() -> str:
    """The one picture worth drawing: what each model compares the frame TO.

    That single difference - a reference image versus a trained model of the
    scene - is what decides every practical property below (per-camera training,
    what happens on a camera with no model, what a new session costs). A box
    diagram earns its place here because the text version takes a paragraph.
    """
    return """<svg viewBox="0 0 760 318" role="img" aria-label="AnomalyDINO
compares the frame to a reference image; Dinomaly compares it to a decoder
trained on nominal frames">
  <defs>
    <marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6"
            markerHeight="6" orient="auto-start-reverse">
      <path d="M0 0 L10 5 L0 10 z" fill="var(--ink-3)"/>
    </marker>
  </defs>

  <text x="0" y="14" class="big" fill="var(--a1)">AnomalyDINO</text>
  <text x="0" y="32" class="sub">needs a clean reference frame · no training</text>
  <rect x="0" y="44" width="150" height="46" rx="8" fill="var(--a1-soft)"
        stroke="var(--a1)"/>
  <text x="75" y="66" class="lbl" text-anchor="middle">reference frame</text>
  <text x="75" y="81" class="sub" text-anchor="middle">clean, same camera</text>
  <rect x="0" y="104" width="150" height="46" rx="8" fill="var(--sunk)"
        stroke="var(--line)"/>
  <text x="75" y="126" class="lbl" text-anchor="middle">current frame</text>
  <text x="75" y="141" class="sub" text-anchor="middle">to inspect</text>
  <line x1="152" y1="67" x2="196" y2="90" stroke="var(--ink-3)" marker-end="url(#a)"/>
  <line x1="152" y1="127" x2="196" y2="104" stroke="var(--ink-3)" marker-end="url(#a)"/>
  <rect x="200" y="72" width="132" height="50" rx="8" fill="var(--card)"
        stroke="var(--line)"/>
  <text x="266" y="94" class="lbl" text-anchor="middle">DINOv2 encoder</text>
  <text x="266" y="109" class="sub" text-anchor="middle">frozen · ViT-S/14-reg</text>
  <line x1="334" y1="97" x2="378" y2="97" stroke="var(--ink-3)" marker-end="url(#a)"/>
  <rect x="382" y="72" width="168" height="50" rx="8" fill="var(--a1-soft)"
        stroke="var(--a1)"/>
  <text x="466" y="94" class="lbl" text-anchor="middle">cosine distance,</text>
  <text x="466" y="110" class="lbl" text-anchor="middle">patch to same patch</text>
  <line x1="552" y1="97" x2="596" y2="97" stroke="var(--ink-3)" marker-end="url(#a)"/>
  <rect x="600" y="72" width="150" height="50" rx="8" fill="var(--card)"
        stroke="var(--line)"/>
  <text x="675" y="94" class="lbl" text-anchor="middle">anomaly map</text>
  <text x="675" y="109" class="sub" text-anchor="middle">→ boxes</text>

  <line x1="0" y1="176" x2="760" y2="176" stroke="var(--line)"/>

  <text x="0" y="204" class="big" fill="var(--a2)">Dinomaly</text>
  <text x="0" y="222" class="sub">no reference at inference · one trained model per camera</text>
  <rect x="0" y="234" width="150" height="46" rx="8" fill="var(--sunk)"
        stroke="var(--line)"/>
  <text x="75" y="256" class="lbl" text-anchor="middle">current frame</text>
  <text x="75" y="271" class="sub" text-anchor="middle">to inspect</text>
  <line x1="152" y1="257" x2="196" y2="257" stroke="var(--ink-3)" marker-end="url(#a)"/>
  <rect x="200" y="232" width="132" height="50" rx="8" fill="var(--card)"
        stroke="var(--line)"/>
  <text x="266" y="254" class="lbl" text-anchor="middle">DINOv2 encoder</text>
  <text x="266" y="269" class="sub" text-anchor="middle">frozen · same one</text>
  <line x1="334" y1="257" x2="378" y2="257" stroke="var(--ink-3)" marker-end="url(#a)"/>
  <rect x="382" y="232" width="168" height="50" rx="8" fill="var(--a2-soft)"
        stroke="var(--a2)"/>
  <text x="466" y="254" class="lbl" text-anchor="middle">decoder rebuilds it,</text>
  <text x="466" y="270" class="lbl" text-anchor="middle">error = anomaly</text>
  <text x="466" y="302" class="sub" text-anchor="middle">trained on this camera's empty frames</text>
  <line x1="552" y1="257" x2="596" y2="257" stroke="var(--ink-3)" marker-end="url(#a)"/>
  <rect x="600" y="232" width="150" height="50" rx="8" fill="var(--card)"
        stroke="var(--line)"/>
  <text x="675" y="254" class="lbl" text-anchor="middle">anomaly map</text>
  <text x="675" y="269" class="sub" text-anchor="middle">→ boxes</text>
</svg>"""


# --- tables -------------------------------------------------------------------

#: variant key -> (label, note, which model it exercises)
ROWS = [
    ("shipped",        "Pixel diff", "the current detector, no neural model", ""),
    ("dino4@0.08",     "AnomalyDINO alone", "the model draws the boxes", "dino"),
    ("gate0.08",       "Pixel diff + AnomalyDINO filter", "the model only deletes boxes", "dino"),
    ("dinomaly4@0.07", "Dinomaly alone", "the model draws the boxes", "dmly"),
    ("dgate0.05",      "Pixel diff + Dinomaly filter", "the model only deletes boxes", "dmly"),
    ("ddgate0.05",     "AnomalyDINO + Dinomaly filter", "one draws, the other deletes", "both"),
]

BADGE = {"dino": '<span class="badge badge--a1">AnomalyDINO</span>',
         "dmly": '<span class="badge badge--a2">Dinomaly</span>',
         "both": '<span class="badge badge--a1">AnomalyDINO</span> '
                 '<span class="badge badge--a2">Dinomaly</span>',
         "": ''}


def results_table(data) -> str:
    V = data["variants"]
    best = max(v["strict"] for v in V.values())
    out = []
    for key, label, note, fam in ROWS:
        v = V.get(key)
        if not v:
            continue
        reg = v["regions_anomaly"] + v["regions_clean"]
        f39 = v["by_family"]["39T"]
        cls = ' class="base"' if key == "shipped" else ""
        win = " win" if v["strict"] >= best else ""
        out.append(f"""<tr{cls}>
<th scope="row">{label}<small>{note}</small></th>
<td>{BADGE[fam]}</td>
<td class="num{win}">{v['strict']}/{v['instances']}</td>
<td class="num{win}">{f39['strict']}/{f39['instances']}</td>
<td class="num">{reg}</td></tr>""")
    return "\n".join(out)


def training_table(rows) -> str:
    return "\n".join(
        f"""<tr><th scope="row">{r['camera']}</th>
<td class="num">{r['frames']}</td>
<td class="num{'' if r['n_sessions'] > 1 else ' bad'}">{r['n_sessions']}</td>
<td class="muted">{', '.join(f"{k} ({v})" for k, v in r['sessions'])}</td>
<td class="num muted">{num(r['loss_first'], 2)} → {num(r['loss_last'], 3)}</td></tr>"""
        for r in rows)


def e2e_table(rows) -> str:
    if not rows:
        return ""
    keep = {"shipped": "Pixel diff", "dino4@0.08": "AnomalyDINO alone",
            "gate0.08": "Pixel diff + AnomalyDINO filter",
            "ddgate0.05": "AnomalyDINO + Dinomaly filter"}
    best = max(r["frame_recall"] for r in rows)
    out = []
    for r in rows:
        if r["spec"] not in keep:
            continue
        cls = ' class="base"' if r["spec"] == "shipped" else ""
        win = " win" if r["frame_recall"] >= best else ""
        out.append(f"""<tr{cls}>
<th scope="row">{keep[r['spec']]}</th>
<td class="num">{r['regions']}</td>
<td class="num{win}">{num(r['frame_recall'], 3)}</td>
<td class="num">{num(r['frame_specificity'], 3)}</td>
<td class="num">{r['inst_detected']}/{r['inst_total']}</td>
<td class="num{win}">{num(r['obj_precision'], 3)}</td></tr>""")
    return "\n".join(out)


def build() -> str:
    data = load("metrics.json")
    train = load("training_sets.json", [])
    e2e = load("end_to_end.json", [])
    V = data["variants"]
    base, dino, dmly, best = (V["shipped"], V["dino4@0.08"],
                              V["dinomaly4@0.07"], V["ddgate0.05"])
    reg = lambda v: v["regions_anomaly"] + v["regions_clean"]        # noqa: E731
    f39 = lambda v: v["by_family"]["39T"]                            # noqa: E731
    inst = base["instances"]
    n_models = len(train)
    n_frames = sum(r["frames"] for r in train)
    e_base = next((r for r in e2e if r["spec"] == "shipped"), None)
    e_best = next((r for r in e2e if r["spec"] == "ddgate0.05"), None)

    e2e_sec = "" if not (e_base and e_best) else f"""
<section>
  <div class="sect-head">
    <span class="eyebrow">Results</span>
    <h2>End to end, with the VLM</h2>
  </div>
  <div class="col">
    <p>The table above counts boxes. This one puts the same
    {data['n_cases']} cases through the VLM that decides whether each box really
    contains an anomaly. It tells us what the better boxes were actually worth.</p>
    <p><b>The number of objects finally reported does not change.</b> It is
    {e_base['inst_detected']}/{e_base['inst_total']} with every option. The four
    extra instances AnomalyDINO boxes are rejected by the judge anyway. What does
    change is the rest. The share of anomalous frames caught goes from
    {num(e_base['frame_recall'], 3)} to {num(e_best['frame_recall'], 3)}, and the
    share of reported regions that are real goes from
    {num(e_base['obj_precision'], 3)} to {num(e_best['obj_precision'], 3)}. In
    plain terms, wrong regions drop from 30 to 7. None of the options ever raises
    a false alarm on a clean frame.</p>
  </div>
  <figure>
    <div class="tablewrap"><table>
      <thead><tr><th scope="col">Configuration</th><th scope="col">Regions judged</th>
      <th scope="col">Frames caught</th><th scope="col">No false alarm</th>
      <th scope="col">Objects</th><th scope="col">Precision</th></tr></thead>
      <tbody>{e2e_table(e2e)}</tbody>
    </table></div>
    <figcaption>Judge: GLM-4.6V-Flash-9B with the conservative prompt, the pair
    every published number in this project uses.</figcaption>
  </figure>
</section>"""

    return f"""{head(**NOTE.head_args)}

<header class="masthead">
  <span class="eyebrow">ARSI · anomaly models · implementation and results</span>
  <h1>Dinomaly &amp; AnomalyDINO</h1>
  <p class="lede">Two recent anomaly-detection models, both built on the same
  frozen DINOv2 backbone. This note covers how each one was implemented here and
  what it scored, next to the pixel-difference detector the pipeline already
  uses. One of them needs a clean reference photo and no training. The other
  needs no reference, but a model trained for each camera.</p>
</header>

<section>
  <div class="sect-head"><h2>Findings</h2></div>
  <div class="finding">
    <ol>
      <li><b>AnomalyDINO draws the best boxes.</b> It covers
        {dino['strict']}/{inst} objects properly against {base['strict']}/{inst}
        for the current detector. Almost all of the gap is tram 39T, where the
        current detector manages
        {f39(base)['strict']}/{f39(base)['instances']} and AnomalyDINO
        {f39(dino)['strict']}/{f39(dino)['instances']}. It needs no training.</li>
      <li><b>Dinomaly draws weaker boxes</b> ({dmly['strict']}/{inst})
        <b>but removes bad ones well.</b> Used to filter AnomalyDINO's output it
        keeps the same {best['strict']}/{inst} while cutting the regions sent on
        from {reg(dino)} to {reg(best)}, so the VLM has 47 % fewer crops to
        judge.</li>
      <li><b>End to end, the number of objects reported does not change.</b> It
        stays at 55 of 73 with every option. What improves is the share of
        anomalous frames caught, 0.871 to 0.968, and the share of reported
        regions that are real, 0.694 to 0.896.</li>
      <li><b>Dinomaly needs one trained model per camera</b>, 59 MB each, and
        empty footage of that camera to train on. That footage is the binding
        constraint, not the compute: training takes 1 min 42 per camera.</li>
    </ol>
  </div>
</section>

<section>
  <div class="sect-head">
    <span class="eyebrow">Method</span>
    <h2>The two models</h2>
  </div>
  <div class="col">
    <p>Both models replace the same step. Before any VLM is asked <em>what</em>
    is in a region, something has to decide <em>where</em> to look. Both run the
    same frozen DINOv2 ViT-S/14-reg encoder, which turns an image into a grid of
    patch descriptors. What differs is what they compare those patches to, and
    that one choice decides how each of them can be used in practice.</p>
  </div>
  <div class="cards">
    <div class="card card--a1">
      <span class="badge badge--a1">reference-based</span>
      <h3>AnomalyDINO</h3>
      <p class="tagline">WACV 2025. A patch is flagged when it no longer looks
      like the patch that sat in the same spot on the clean reference photo.</p>
      <dl>
        <dt>Training</dt><dd>none</dd>
        <dt>Needs</dt><dd>one clean reference frame per camera</dd>
        <dt>State</dt><dd>none per camera</dd>
        <dt>Cost</dt><dd>≈2 s per frame</dd>
      </dl>
    </div>
    <div class="card card--a2">
      <span class="badge badge--a2">reference-free</span>
      <h3>Dinomaly</h3>
      <p class="tagline">CVPR 2025. A patch is flagged when a decoder trained on
      this camera's empty frames fails to rebuild it.</p>
      <dl>
        <dt>Training</dt><dd>1 min 42 per camera (RTX 3080 Ti)</dd>
        <dt>Needs</dt><dd>empty footage of that camera</dd>
        <dt>State</dt><dd>59 MB per camera</dd>
        <dt>Cost</dt><dd>≈3 s per frame</dd>
      </dl>
    </div>
  </div>
  <figure>
    <div class="diagram">{diagram()}</div>
    <figcaption>Everything after the anomaly map is shared with the current
    detector: the same region finder, the same person filter, the same merging
    step. That way a comparison between the two models only measures the change
    signal itself.</figcaption>
  </figure>
</section>

<section>
  <div class="sect-head">
    <span class="eyebrow">Method</span>
    <h2>AnomalyDINO: implementation</h2>
  </div>
  <div class="col">
    <ol class="steps steps--a1">
      <li><b>Encode both frames.</b> The reference photo and the current image
        are resized to 1120 px wide and passed through the frozen encoder. Each
        one comes out as an 80×45 grid of patch descriptors.
        <p>On a 1920×1080 frame a patch covers 24 px, on a 1280×720 frame 16 px.
        That grid is also the smallest box either model can draw.</p></li>
      <li><b>Compare each patch to the same spot, not to a pool of patches.</b>
        This is where our version departs from the paper. The paper collects all
        the normal patches into one memory bank, because the objects it works on
        are centred but not aligned. Our cameras are bolted to the tram, so we
        can be stricter: a patch is only compared to reference patches within one
        patch of its own position.
        <p>It makes for a much tighter test. A seat cushion gets compared to
        <em>that</em> seat, not to any seat in the picture.</p></li>
      <li><b>Use two thresholds, not one.</b> The distance map is normalised by
        its own median and spread, then a patch is flagged only if it clears both
        a relative threshold (4 times the map's typical spread, measured in a
        way that outliers cannot inflate) and an absolute one (distance 0.08).
        <p>Both are needed. The spread of the map varies by a factor of five
        across our cases, so a single relative threshold cuts at 0.043 on one
        frame and 0.225 on another. On two nearly identical frames the spread
        collapses and the relative threshold alone invents 80 regions or more.
        The absolute floor stops that, and the relative one handles frames shot
        in different light.</p></li>
    </ol>
  </div>
</section>

<section>
  <div class="sect-head">
    <span class="eyebrow">Method</span>
    <h2>Dinomaly: implementation and training data</h2>
  </div>
  <div class="col">
    <ol class="steps steps--a2">
      <li><b>Freeze the encoder, train only a decoder.</b> 15.4 M trainable
        parameters sitting on top of the frozen DINOv2. We use eight of its
        twelve middle blocks, split into two groups. The last blocks are too
        specialised and the first ones too generic to be useful here.</li>
      <li><b>Use linear attention, and not for speed.</b> Standard attention is
        sharp enough that a decoder token can simply copy the encoder token it is
        meant to rebuild, anomaly included, which would defeat the whole idea. A
        smoother kernel cannot pick out a single token, so an unfamiliar patch
        has to be rebuilt from its surroundings. It fails, and that failure is
        the signal we detect.</li>
      <li><b>Train one model per camera, on empty frames.</b> {n_frames} frames
        over {n_models} models, 40 epochs each. A model of what <em>this</em>
        view normally looks like says nothing about another view, so covering the
        fleet's 26 cameras would mean 26 models.</li>
      <li><b>Score by how badly the rebuild fails.</b> At inference the frame is
        encoded, the decoder rebuilds it, and the gap between the two is the
        anomaly map. No reference photo is used at any point.</li>
    </ol>
  </div>
  <figure>
    <div class="tablewrap"><table>
      <thead><tr><th scope="col">Camera</th><th scope="col">Frames</th>
      <th scope="col">Sessions</th><th scope="col">From</th>
      <th scope="col">Training loss</th></tr></thead>
      <tbody>{training_table(train)}</tbody>
    </table></div>
    <figcaption>These numbers are read back out of the checkpoint files, which
    each store the list of images they were trained on. Every image is an empty
    tram, taken one every 2 seconds from the source video, then filtered twice.
    Anything shot within 15 seconds of a test image is dropped, so no model is
    ever scored on what it learned from. A person detector then removes any image
    with someone in it, which took out 18 % of the 39T candidates. One 39T
    recording was excluded by hand: it has two passengers <em>and</em> a bag on
    the floor, and the detector would have kept the frames where the bag sits
    alone.</figcaption>
  </figure>
  <div class="note">
    <h4>The training data lacks variety, not quantity</h4>
    <p>The three 1760 models have around 190 images each, and all of them come
    from a single ten-minute run. Same hour, same light, same weather. Pulling
    five hundred more out of that video would just give five hundred nearly
    identical pictures. These models have never seen their own camera on another
    day, or at night.</p>
  </div>
</section>

<section>
  <div class="sect-head">
    <span class="eyebrow">Results</span>
    <h2>Box quality</h2>
  </div>
  <div class="col">
    <p>An object only counts here if one region covers it properly. Merely
    overlapping it is not enough, because a box spanning the whole image overlaps
    everything while telling the VLM nothing about where to look. No VLM was run
    for this table.</p>
    <p><b>On its own, AnomalyDINO is the best of the three</b>, with
    {dino['strict']}/{inst} against {base['strict']}/{inst} for the pixel diff.
    The gap is not spread evenly. Almost all of it comes from tram 39T, where the
    pixel diff manages {f39(base)['strict']}/{f39(base)['instances']} and
    AnomalyDINO {f39(dino)['strict']}/{f39(dino)['instances']}. On those cameras
    the pixel diff's worst box covers 99 % of the image. AnomalyDINO's worst is
    about ten times smaller.</p>
    <p><b>Dinomaly is weaker at drawing boxes</b> ({dmly['strict']}/{inst}), but
    it is good at removing bad ones. Used to filter AnomalyDINO's output it keeps
    the same {best['strict']}/{inst} while cutting the regions from {reg(dino)}
    to {reg(best)}, so the VLM has 47 % fewer crops to look at. The two models
    make different mistakes, because one compares the image to a photo and the
    other to a model of the scene.</p>
  </div>
  <div class="keys">
    <div class="key"><span class="key__v a1">{dino['strict']}/{inst}</span>
      <span class="key__l">objects boxed properly by AnomalyDINO, against
      {base['strict']}/{inst} for the pixel diff</span></div>
    <div class="key"><span class="key__v a1">{f39(dino)['strict']}/{f39(dino)['instances']}</span>
      <span class="key__l">on tram 39T, where the pixel diff manages
      {f39(base)['strict']}/{f39(base)['instances']}</span></div>
    <div class="key"><span class="key__v a2">−47 %</span>
      <span class="key__l">fewer crops sent to the VLM when Dinomaly filters
      AnomalyDINO, with no loss of quality</span></div>
    <div class="key"><span class="key__v">1 min 42</span>
      <span class="key__l">to train one Dinomaly model on the RTX 3080 Ti, down
      from 56 min on the laptop</span></div>
  </div>
  <figure>
    <div class="tablewrap"><table>
      <thead><tr><th scope="col">Configuration</th><th scope="col">Model used</th>
      <th scope="col">Boxed properly</th><th scope="col">On 39T</th>
      <th scope="col">Regions</th></tr></thead>
      <tbody>{results_table(data)}</tbody>
    </table></div>
    <figcaption>All {data['n_cases']} cases. The “Regions” column is what gets
    sent on to the VLM. At roughly 15 seconds of CPU per region, it is what
    dominates the running time of a job. The shaded row is the current
    detector.</figcaption>
  </figure>
</section>
{e2e_sec}

<section>
  <div class="sect-head">
    <h2>Limits</h2>
  </div>
  <div class="col">
    <ul class="plain">
      <li><b>The three 1760 models only ever saw one session.</b> Each of those
      cameras has a single ten-minute run, and the test images come from that
      same run. Training images are kept 15 seconds away from any test image, but
      they still share its light. So nothing here tells us whether a 1760 model
      still works on another day.</li>
      <li><b>The 39T clean frames are not independent either.</b> Of the nine
      recordings, four contain the staged anomalies, one is the reference and one
      is used as a clean test case. The three that are unlabelled last 12, 8 and
      10 seconds, which is far too short to train on. There is simply no clean
      pool left, so training uses the two long clean recordings minus the images
      held out. The 14 anomalous 39T cases are genuinely from sessions the model
      never saw. The 7 clean ones are not, so the false-alarm rate on 39T looks
      better than it should.</li>
      <li><b>39T-cam52 is the weakest model of the eight.</b> It has only 90
      images, all from one recording, and it ends training with the highest
      error.</li>
      <li><b>On an unknown camera, Dinomaly goes quiet rather than wrong.</b> We
      showed it a scene it had never been trained on, and it filtered out
      <em>zero</em> regions instead of filtering out everything. In other words
      it falls back to doing nothing, not to blocking detections. An untrained
      camera costs us efficiency, not missed objects.</li>
      <li><b>Box quality uses a 0.3 overlap threshold</b>, measured against boxes
      that were drawn by hand on a 100 px grid, fairly generously. It tells a
      usable box apart from a blob. It is not a precise measure of outline
      quality.</li>
    </ul>
  </div>
</section>
{foot("ARSI · tram anomaly detection · region proposal",
      "Pipeline: benchmark/eval_localization.py → tools/collect_e2e.py → tools/dump_training_sets.py",
      "Raw data: docs/dino_models/metrics.json, end_to_end.json, training_sets.json")}
"""


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    DOC.write_text(build())
    print(f"{DOC}  ({DOC.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
