"""Assemble the standalone HTML report on the 1760 -> 3333 framing offset.

Every figure is re-encoded and inlined as a data: URI, so the document opens on
its own with no image folder alongside it.

Output: docs/camera_alignment/camera_framing_drift_1760_3333.html
"""

from __future__ import annotations

import base64
import json
import math
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.notes import NOTES                                     # noqa: E402
from tools.report_style import TRAM_FOOT, foot, head, tram_star                         # noqa: E402

NOTE = NOTES["camera-alignment"]
DOC = NOTE.html
OUT = DOC.parent

ROWS = json.loads((OUT / "metrics.json").read_text())
TOL = json.loads((OUT / "tolerance.json").read_text())


def embed(path: Path, width: int, quality: int = 72) -> str:
    img = cv2.imread(str(path))
    h, w = img.shape[:2]
    if w > width:
        img = cv2.resize(img, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    assert ok, path
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def med(key):
    vals = sorted(r[key] for r in ROWS if r.get(key) is not None)
    return vals[len(vals) // 2]


def num(v, n=1, signed=False):
    """Number with a true minus sign, and an explicit plus where it matters."""
    return f"{v:{'+' if signed else ''}.{n}f}".replace("-", "−")


# --------------------------------------------------------------------------- #
def build() -> str:
    n = len(ROWS)
    # focal length behind the pixel -> degree conversion, from the assumed field
    focal_px = (ROWS[0]["size"][0] / 2) / math.tan(math.radians(100.0 / 2))
    worst = max(ROWS, key=lambda r: r["shift_px"])
    best = min(ROWS, key=lambda r: r["shift_px"])
    max_rot = max(ROWS, key=lambda r: abs(r["rot_deg"]))
    rots = sorted(abs(r["rot_deg"]) for r in ROWS)
    med_abs_rot = rots[len(rots) // 2]

    fig_summary = embed(OUT / "_offset_summary.jpg", 900, 80)
    fig_sheet = embed(OUT / "_contact_sheet.jpg", 1500, 72)
    fig_tol = embed(OUT / "_tolerance.jpg", 1100, 82)

    # three teaching cases, from the most striking to the most subtle
    cases = [
        ("3333-cam55", "The worst of the rear block",
         "143 px of offset, almost all of it pan. The copied mask leaves 14 % of the frame "
         "as unmasked glass and blacks out just as much useful interior."),
        ("3333-cam52", "The largest gap measured",
         "164 px on the diagonal, and the only pair where the common field drops below 80 %. "
         "Its 3333 mask was also cut much more finely, 52 zones against 20."),
        ("3333-cam50", "The one with the most roll",
         "Only 51 px of offset, but 5.4° of roll: the camera points the right way and sits "
         "turned on its own axis. Its mask happens to survive the copy at IoU 0.96, though a "
         "small offset is no guarantee of that. cam54 is closer still, at 46 px, and its copied "
         "mask only reaches 0.70."),
    ]

    def table_rows():
        out = []
        for r in ROWS:
            sev = "crit" if r["shift_px"] > 120 else ("warn" if r["shift_px"] > 80 else "ok")
            out.append(f"""<tr>
<th scope="row">{r['cam_1760'].replace('1760-', '')} <span class="arrow">→</span> {r['cam_3333'].replace('3333-', '')}</th>
<td class="num">{num(r['dx'], 0, True)}</td><td class="num">{num(r['dy'], 0, True)}</td>
<td class="num"><span class="bar bar--{sev}" style="--v:{min(r['shift_px'] / 170, 1):.3f}"></span>{r['shift_px']:.0f}</td>
<td class="num">{num(r['total_deg'], 1)}</td>
<td class="num">{num(r['rot_deg'], 2, True)}</td>
<td class="num">{num(r['scale'], 3)}</td>
<td class="num">{num(r['fov_overlap'] * 100, 0)}</td>
<td class="num">{num(r['iou_copied'], 2)}</td>
<td class="num strong">{num(r['iou_registered'], 2)}</td>
<td class="num muted">{num(r['ncc_identity'], 2)} → {num(r['ncc'], 2)}</td>
</tr>""")
        return "\n".join(out)

    def tol_rows():
        out = []
        for c in TOL:
            if c["shift_px"] not in (0, 5, 10, 20, 40, 60, 80, 100):
                continue
            out.append(f"""<tr><td class="num">{c['shift_px']}</td><td class="num">{num(c['deg'], 1)}</td>
<td class="num">{num(c['iou_median'], 2)}</td><td class="num">{num(c['iou_p10'], 2)}</td>
<td class="num">{num(c['glass_exposed_pct_median'], 1)} %</td></tr>""")
        return "\n".join(out)

    def gallery():
        out = []
        for r in ROWS:
            cam = r["cam_3333"]
            sev = "crit" if r["shift_px"] > 120 else ("warn" if r["shift_px"] > 80 else "ok")
            out.append(f"""<article class="cam" id="{cam}">
  <header class="cam__head">
    <h3>{r['cam_1760']} <span class="arrow">→</span> {cam}</h3>
    <span class="pill pill--{sev}">{num(r['shift_px'], 0)} px · {num(r['total_deg'], 1)}°</span>
  </header>
  <dl class="cam__stats">
    <div><dt>pan</dt><dd>{num(r['pan_deg'], 1, True)}°</dd></div>
    <div><dt>tilt</dt><dd>{num(r['tilt_deg'], 1, True)}°</dd></div>
    <div><dt>roll</dt><dd>{num(r['rot_deg'], 2, True)}°</dd></div>
    <div><dt>common field</dt><dd>{num(r['fov_overlap'] * 100, 0)} %</dd></div>
    <div><dt>IoU, mask copied</dt><dd>{num(r['iou_copied'], 2)}</dd></div>
    <div><dt>IoU, aligned</dt><dd>{num(r['iou_registered'], 2)}</dd></div>
  </dl>
  <figure><img src="{embed(OUT / f'{cam}_align.jpg', 1400, 72)}" alt="1760 and 3333 framings of {cam}, overlay and residual after alignment" loading="lazy">
    <figcaption>Top: the two median backgrounds. Bottom: the raw overlay, then the residual after alignment.</figcaption></figure>
  <figure><img src="{embed(OUT / f'{cam}_masks.jpg', 1180, 72)}" alt="Masks for {cam}: copied, aligned, redrawn" loading="lazy">
    <figcaption>Glass left visible by the copy: {num(r['glass_exposed_copied_pct'], 1)} % of the frame · interior wrongly masked: {num(r['interior_masked_copied_pct'], 1)} %.</figcaption></figure>
</article>""")
        return "\n".join(out)

    def case_blocks():
        out = []
        for cam, title, text in cases:
            r = next(x for x in ROWS if x["cam_3333"] == cam)
            out.append(f"""<div class="case">
  <h3>{title} <span class="case__cam">{cam}</span></h3>
  <p>{text}</p>
  <figure><img src="{embed(OUT / f'{cam}_masks.jpg', 1280, 78)}" alt="Masks overlaid on {cam}">
    <figcaption>Magenta: the 1760 mask dropped in as-is. Yellow: the same mask, aligned automatically.
    Green: the 3333 mask redrawn by hand. IoU {num(r['iou_copied'], 2)} → {num(r['iou_registered'], 2)}.</figcaption></figure>
</div>""")
        return "\n".join(out)

    flow_cam = "3333-cam52"
    fig_flow = embed(OUT / f"{flow_cam}_flow.jpg", 1280, 78)

    return f"""{head(**NOTE.head_args)}

<header class="masthead">
  <span class="eyebrow">ARSI · onboard cameras · framing drift</span>
  <h1>The framing drift issue</h1>
  <p class="lede">Tram {tram_star()} carries the same {n} interior cameras as tram 1760, mounted in the same
  places. None of them points quite where its counterpart points. The median gap is
  <strong>{num(med('shift_px'), 0)} pixels</strong>, about <strong>{num(med('total_deg'), 1)}°</strong>
  of aim, and that is enough that every mask had to be redrawn.</p>
  <div class="meta">
    <span>Trams 1760 &amp; 3333</span>
    <span>{n} camera pairs</span>
    <span>1280 × 720</span>
  </div>
</header>

<div class="keys">
  <div class="key"><span class="key__v">{num(med('shift_px'), 0)} px</span>
    <span class="key__l">median offset at the image centre</span>
    <span class="key__n">min {num(best['shift_px'], 0)} · max {num(worst['shift_px'], 0)}</span></div>
  <div class="key"><span class="key__v">{num(med('total_deg'), 1)}°</span>
    <span class="key__l">equivalent camera re-aim</span>
    <span class="key__n">up to {num(worst['total_deg'], 1)}° on {worst['cam_3333'].replace('3333-', '')}</span></div>
  <div class="key"><span class="key__v">{num(max_rot['rot_deg'], 1, True)}°</span>
    <span class="key__l">largest roll, camera turned on its own axis</span>
    <span class="key__n">{max_rot['cam_3333'].replace('3333-', '')} · median {num(med_abs_rot, 1)}°</span></div>
  <div class="key"><span class="key__v">{num(med('iou_copied'), 2)}</span>
    <span class="key__l">overlap of a 1760 mask dropped in as-is</span>
    <span class="key__n">down to {num(min(r['iou_copied'] for r in ROWS), 2)}</span></div>
  <div class="key"><span class="key__v">10 px</span>
    <span class="key__l">mounting tolerance to aim for if a mask is to be reused</span>
    <span class="key__n">1.1°, seven times tighter than what we found</span></div>
</div>

<section>
  <div class="sect-head">
    <span class="eyebrow">The problem</span>
    <h2>What went wrong</h2>
  </div>
  <div class="col">
    <p>The exclusion masks (windows, glazed doors, portholes) were drawn once for each of the
    {n} interior cameras of tram 1760. Tram 3333 carries the same camera set, in the same
    positions, in a tram of the same type, so on paper the masks should transfer straight
    across. Not one of them landed correctly, and all {n} had to be redrawn.</p>
    <p>The cause is geometric. Each camera was aimed by hand when it was fitted, and nobody
    aims twice the same way. The gaps run from {num(best['shift_px'], 0)} to
    {num(worst['shift_px'], 0)} pixels and point in every direction, so there is no single
    tram-wide offset to correct. There are {n} independent aiming errors.</p>
  </div>
  <figure>
    <img src="{fig_summary}" alt="Polar map of the offsets: one arrow per camera, scattered in every direction">
    <figcaption>One arrow per camera, starting at the image centre. The directions are scattered,
    so there is no systematic bias to take out globally. Red marks the framings past 100 px.</figcaption>
  </figure>
  <figure>
    <img src="{fig_sheet}" alt="Contact sheet of the 15 pairs overlaid in magenta and green">
    <figcaption>All {n} pairs overlaid. Wherever the magenta and the green split apart, the two
    cameras are not looking at the same thing.</figcaption>
    <div class="legend"><span class="sw m">1760</span><span class="sw v">3333</span><span class="sw" style="color:var(--ink-3)">grey = the two coincide</span></div>
  </figure>
</section>

<section>
  <div class="sect-head">
    <span class="eyebrow">Method</span>
    <h2>How the offset is measured</h2>
  </div>
  <div class="col">
  <ol class="steps">
    <li><div><b>Take one clean image per camera.</b>
      <p>A single still per camera with nobody in it, on both trams.</p></div></li>
    <li><div><b>Find the same details in both images.</b>
      <p>A few hundred small recognisable spots are located in the 1760 image, a screw head, the
      corner of a panel, the edge of a seat, then looked for again in the 3333 image
      (<code>cv2.SIFT</code> keypoints, paired by Lowe's ratio test). Every spot found twice gives
      one arrow: where the detail used to be, and where it is now.</p></div></li>
    <li><div><b>Find the one camera movement that explains every arrow.</b>
      <p>A single arrow proves nothing, since a mistaken match makes a wrong arrow. So we look for
      the movement that accounts for as many arrows as possible at once and discard those that
      disagree with it (RANSAC and MAGSAC, fitting a similarity, an affine and a homography). A
      fourth method skips the details entirely and slides the two whole images over each other
      (phase correlation), so its answer owes nothing to the other three.</p>
      <p>Whichever of the four makes the two images overlap best is kept, overlap being scored by
      correlating their edge maps. The four agree to within 3 px on 13 of the {n} pairs, which is
      the margin of error on the result.</p></div></li>
    <li><div><b>Turn it into numbers.</b>
      <p>How far the scene has slipped inside the frame, in pixels, how far the image has rotated,
      and the equivalent turn of the camera in degrees.</p></div></li>
  </ol>
  </div>
</section>

<section>
  <div class="sect-head">
    <span class="eyebrow">Diagnosis</span>
    <h2>The cameras turned in place</h2>
  </div>
  <div class="col">
    <p>This matters for what comes next. If the cameras had been fixed in different places, no
    aiming instruction could fix the gap. Three things say they were not.</p>
    <ul class="plain">
      <li><b>The displacement field is uniform.</b> On cam52, all 104 matched points move by
      143 to 184 px in the same direction, whether they sit on a seat in the foreground or at
      the far end of the carriage. A displaced mount would produce parallax instead, with near
      objects moving far more than distant ones.</li>
      <li><b>A homography fits.</b> The median residual of the matched points stays under
      2.5 px everywhere, and a homography describes a rotation about the optical centre.</li>
      <li><b>The scale factor is 1.</b> Median {num(med('scale'), 3)} across the {n} pairs, so the
      camera is neither closer to nor further from the scene and nobody touched the zoom.</li>
    </ul>
    <p>These are <em>aiming</em> errors, which means a precise enough orientation procedure can
    get rid of them.</p>
  </div>
  <figure>
    <img src="{fig_flow}" alt="Displacement field of the matched points on cam52, arrows parallel and of constant length">
    <figcaption>{flow_cam}, displacement of every matched structural point. The arrows stay parallel
    and the same length from foreground to background, which is what a pivot looks like. A moved
    mount would show parallax.</figcaption>
  </figure>
</section>

<section>
  <div class="sect-head">
    <span class="eyebrow">Measurements</span>
    <h2>Camera by camera</h2>
  </div>
  <div class="tablewrap full">
    <table>
      <caption>Transformation from the 1760 framing to the 3333 framing. Degrees assume a 100°
      horizontal field; at 90° or 110° they move by about ±10 %. The pixels are the measurement.</caption>
      <thead><tr>
        <th scope="col">pair</th>
        <th scope="col">dx<br>px</th><th scope="col">dy<br>px</th>
        <th scope="col">gap<br>px</th><th scope="col">aim<br>deg</th>
        <th scope="col">roll<br>deg</th><th scope="col">scale</th>
        <th scope="col">common<br>field %</th>
        <th scope="col">IoU<br>copied</th><th scope="col">IoU<br>aligned</th>
        <th scope="col">correlation</th>
      </tr></thead>
      <tbody>{table_rows()}</tbody>
    </table>
  </div>
  <div class="col">
    <p><b>dx</b> and <b>dy</b>: how far the scene has moved within the frame, at the centre.
    A positive dx means the scene drifts to the right. <b>Gap</b>: the two as one distance,
    √(dx² + dy²), how far a centre point travels between the two framings. It has no direction,
    which is why the cameras are ranked by it. <b>Aim</b>: the same gap as a rotation,
    arctan(gap / f) with f ≈ {focal_px:.0f} px from the assumed field; it leaves out the roll,
    which moves nothing at the centre. <b>Roll</b>: rotation of the camera about its optical axis. <b>Common field</b>: share of the 1760 frame still visible in 3333.
    <b>Correlation</b>: how well the edges of the two images line up, with no correction applied
    → with the measured one applied. 0 means unrelated, 1 identical. The jump is what shows the
    transformation is real rather than noise.</p>
    <p><b>IoU</b>, intersection over union, is how the two masks are compared: the area they
    share divided by the area they cover between them. 1 means they coincide exactly, 0 that they
    never touch. It punishes a shift twice over, once by losing intersection and once by growing
    the union, so two identical squares offset by half their width already score 0.33 rather than
    0.50. <em>Copied</em> is the 1760 mask dropped straight onto 3333, <em>aligned</em> the same
    mask moved by the measured transformation. Being symmetric, IoU says how far apart two masks
    are but not in which direction, which is why the exposed glass and the wrongly masked
    interior are reported separately below.</p>
  </div>
</section>

<section>
  <div class="sect-head">
    <span class="eyebrow">Consequence</span>
    <h2>What the offset does to the masks</h2>
  </div>
  <div class="col">
    <p>A misplaced mask fails in two ways at once. Where it no longer covers the glass, the
    outside world scrolls through the inspected area and raises alerts for things that are not
    in the tram. Where it spills onto the interior, it blacks out real fittings, and anything
    sitting there goes unseen.</p>
    <ul class="plain">
      <li>Copying the 1760 masks as-is leaves <b>{num(med('glass_exposed_copied_pct'), 1)} % of the
      frame</b> as unmasked glass, and blacks out <b>{num(med('interior_masked_copied_pct'), 1)} %</b>
      of useful interior by mistake (medians; up to 17 % and 16 % on the worst cameras).</li>
      <li>Aligning those same masks automatically drops the wrongly masked interior to
      <b>{num(med('interior_masked_registered_pct'), 1)} %</b> and lifts the median IoU from
      {num(med('iou_copied'), 2)} to {num(med('iou_registered'), 2)}, so most of the geometric
      error is recoverable after the fact.</li>
      <li>The exposed glass does not improve with alignment. The hand-drawn 3333 masks cover
      zones the 1760 mask never had, so that part of the gap is an annotation difference rather
      than a geometric one. It shows plainly on cam52, which went from 20 zones to 52.</li>
    </ul>
  </div>
  {case_blocks()}
</section>

<section>
  <div class="sect-head">
    <span class="eyebrow">Tolerance</span>
    <h2>How much offset can a mask absorb?</h2>
  </div>
  <div class="col">
    <p>Rather than pick a threshold out of the air, each of the {n} 3333 masks was shifted by
    <em>k</em> pixels in eight directions to find where it stops being usable.</p>
  </div>
  <figure>
    <img src="{fig_tol}" alt="Curve of IoU against applied offset, falling from 1.0 to 0.45">
    <figcaption>The decay is fast, because these masks are cut into many small zones and small
    zones are very sensitive to displacement. At the offset actually seen between the two trams,
    the median IoU has already fallen below 0.60.</figcaption>
  </figure>
  <div class="tablewrap">
    <table>
      <caption>Effect of a pure shift on the {n} 3333 masks, 8 directions.</caption>
      <thead><tr><th scope="col">shift px</th><th scope="col">deg</th><th scope="col">median IoU</th>
      <th scope="col">10th pct IoU</th><th scope="col">glass exposed</th></tr></thead>
      <tbody>{tol_rows()}</tbody>
    </table>
  </div>
  <div class="col note">
    <h3>The mounting target</h3>
    <p><b>Aim for 5 px (0.5°), never exceed 10 px (1.1°).</b> At 5 px the mask keeps an IoU of
    0.95 and transfers with no retouching. At 10 px it still holds at 0.91, which calls for a
    check but not a recut. Past 20 px it has to be redrawn by hand. The median gap measured
    between 1760 and 3333, {num(med('shift_px'), 0)} px, is seven times that limit.</p>
  </div>
</section>

<section>
  <div class="sect-head">
    <span class="eyebrow">Recommendations</span>
    <h2>Aiming the cameras on the next tram</h2>
  </div>
  <div class="col">
    <p>The target is roughly 10 px from the reference framing, about 1° of aim, which is too fine
    to judge by eye. What makes the difference is not really being careful, it is having
    something on screen to compare against while you work.</p>
    <p>So bring the picture of each camera from the previous tram and keep it beside the live
    stream. Better still, superimpose the two instead of putting them side by side. A small
    difference leaps out on an overlay and stays invisible when the images merely sit next to
    each other, which is how 70 px went unnoticed on the 3333.</p>
    <p>While aiming, watch the roll. Getting left/right and up/down correct and leaving the
    camera tilted on its own axis is the easy mistake to make, and cam50 shows it costs as much
    as a bad aim. Line up on fixed details spread around the frame, a door pillar, a seat leg,
    the corner of a panel, rather than on the middle alone, and ignore anything seen through a
    window since it moves with the tram.</p>
    <p>Last, measure before leaving the tram. A short clip per camera is enough to check the gap
    and re-aim whatever is off. Without a number you always think you aimed well.</p>
  </div>
</section>

<section>
  <div class="sect-head">
    <span class="eyebrow">Caveats</span>
    <h2>What the measurement doesn't cover</h2>
  </div>
  <div class="col">
    <ul class="plain">
      <li>The degrees rest on an assumed field. Everything is computed for a 100° horizontal field, which is consistent with the
      barrel distortion visible in the images. At 90° or 110° the angles move by about 10 %. The
      pixels are measured directly and do not depend on this.</li>
      <li>The two trams are not identical down to the detail.</li>
    </ul>
  </div>
</section>

<section>
  <div class="sect-head">
    <span class="eyebrow">Detail</span>
    <h2>All {n} pairs</h2>
  </div>
  <div class="legend">
    <span class="sw m">1760 mask copied as-is</span>
    <span class="sw j">the same, aligned automatically</span>
    <span class="sw v">3333 mask redrawn by hand</span>
  </div>
  <div class="cams full">{gallery()}</div>
</section>

{foot("ARSI · tram anomaly detection · framing drift",
      "Pipeline: tools/build_background_frames.py → tools/camera_alignment_report.py → tools/mask_shift_tolerance.py",
      "Raw data: docs/camera_alignment/metrics.json and tolerance.json",
      TRAM_FOOT)}
"""


if __name__ == "__main__":
    DOC.write_text(build(), encoding="utf-8")
    print(f"{DOC}  ({DOC.stat().st_size / 1e6:.1f} MB)")
