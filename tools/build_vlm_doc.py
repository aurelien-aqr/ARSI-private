#!/usr/bin/env python3
"""Assemble the standalone note comparing VLMs and promptings for the judge step.

Reads docs/vlm_benchmark/metrics.json, written by tools/collect_judge.py, and
writes a self-contained page built from tools/report_style.py, like every note.

    venv/bin/python tools/judge_sweep.py --localizer ddgate0.05 --refs all \
        --models <a>,<b>,... --prompts conservative,lenient,balanced
    venv/bin/python tools/collect_judge.py
    venv/bin/python tools/build_vlm_doc.py

Output: docs/vlm_benchmark/vlm_benchmark.html
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.notes import NOTES                                     # noqa: E402
from tools.report_style import TRAM_FOOT, foot, head, tram_star                         # noqa: E402

NOTE = NOTES["vlm-benchmark"]
DOC = NOTE.html
OUT = DOC.parent
METRICS = OUT / "metrics.json"

PROMPT_ORDER = ["conservative", "lenient", "balanced"]
PROMPT_NOTE = {
    "conservative": "The wording in use today. It spells out when to answer no: "
                    "a change of light, a person, a reflection on metal. It also "
                    "tells the model to answer no whenever it is unsure.",
    "lenient": "The original wording. It lists what counts as an anomaly and the "
               "three cases that do not count, and stops there.",
    "balanced": "Written for this comparison. It keeps every reason to answer no "
                "from the cautious version, but drops the instruction to answer "
                "no when unsure, and says that an object can be small or half "
                "hidden and still count.",
}

#: The localizer this sweep holds fixed. Stated everywhere, because every number
#: on the page is conditional on it.
BOXES = "ddgate0.05"
BASELINE = ("GLM-4.6V-Flash-9B", "conservative")


#: Chart labels only. The grid table carries the full names; on a scatter where
#: half the arms land within a few pixels of each other, a long name is the
#: difference between a figure and a smudge.
SHORT = {"GLM-4.6V-Flash-9B": "GLM", "Qwen3-VL-8B": "Qwen3-VL",
         "Qwen2.5-VL-7B": "Qwen2.5-VL", "InternVL3.5-8B": "InternVL",
         "MiniCPM-V-4.6": "MiniCPM", "Cosmos-Reason2-8B": "Cosmos"}


def num(v, n=3):
    return f"{v:.{n}f}".replace("-", "−")


def place(points, W, band=17, gap=15):
    """Circles plus non-overlapping labels for `points` = [(x, y, text, colour,
    ringed)].

    Points that share a horizontal band get their labels fanned out vertically
    with a leader line back to the dot. Without this the arms that tie on recall
    - and several do, which is itself the finding - print on top of each other.
    """
    rows = {}
    for pt in sorted(points, key=lambda p: (round(p[1] / band), p[0])):
        rows.setdefault(round(pt[1] / band), []).append(pt)
    out = []
    for group in rows.values():
        n = len(group)
        offs = [0] if n == 1 else [(i - (n - 1) / 2) * gap for i in range(n)]
        for (x, y, text, colour, ringed), off in zip(group, offs):
            anchor = "end" if x > W * 0.55 else "start"
            dx = -13 if anchor == "end" else 13
            ty = y + off + 4
            if off:
                out.append(f'<line x1="{x:.1f}" y1="{y:.1f}" '
                           f'x2="{x + dx * 0.5:.1f}" y2="{ty - 4:.1f}" '
                           f'stroke="{colour}" stroke-width="1.1" opacity=".5"/>')
            ring = ' stroke="var(--ink)" stroke-width="2"' if ringed else ""
            out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{8 if ringed else 6}" '
                       f'fill="{colour}"{ring}/>')
            out.append(f'<text x="{x + dx:.1f}" y="{ty:.1f}" '
                       f'text-anchor="{anchor}" class="pt">{text}</text>')
    return out


def load():
    if not METRICS.exists():
        raise SystemExit(
            f"no metrics at {METRICS}. Run the sweep first:\n"
            f"    venv/bin/python tools/judge_sweep.py --localizer {BOXES} "
            f"--refs all --models <tags> \n"
            f"    venv/bin/python tools/collect_judge.py")
    return json.loads(METRICS.read_text())


def grid(rows):
    """{model: {prompt: row}} plus the model order, best object recall first."""
    by = {}
    for r in rows:
        by.setdefault(r["model"], {})[r["prompt"]] = r
    order = sorted(by, key=lambda mdl: -max(x["obj_recall"]
                                            for x in by[mdl].values()))
    prompts = [p for p in PROMPT_ORDER if any(p in v for v in by.values())]
    return by, order, prompts


def grid_table(rows) -> str:
    by, order, prompts = grid(rows)
    best = max(r["obj_recall"] for r in rows)
    head = "".join(f'<th scope="col">{p}</th>' for p in prompts)
    body = []
    for mdl in order:
        cells = []
        for p in prompts:
            r = by[mdl].get(p)
            if not r:
                cells.append('<td class="num muted">–</td>')
                continue
            hot = r["obj_recall"] >= best
            spec = r["frame_specificity"]
            cls = "strong" if hot else ""
            # as a share of clean frames, which is what "false alarm" means to
            # anyone reading this; a bare 0.86 reads like a score, not a rate
            warn = "" if spec >= 0.999 else (f' · <b class="alarm">'
                                             f'alarms {100 * (1 - spec):.0f}%</b>')
            cells.append(
                f'<td class="num {cls}">{r["inst_detected"]}/{r["inst_total"]}'
                f'<small>yes {100 * r["yes_rate"]:.0f}% · right '
                f'{100 * r["obj_precision"]:.0f}%{warn}</small></td>')
        body.append(f'<tr><th scope="row">{mdl}</th>' + "".join(cells) + "</tr>")
    return (f'<thead><tr><th scope="col">Judge</th>{head}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody>')


def calibration(rows) -> str:
    """YES rate against object recall, with the rate the ground truth implies.

    This is the figure that makes the grid readable. The benchmark puts 73
    anomalies among 654 crops, so a calibrated judge answers YES to ~11 % of
    them. An arm at 88 % has not detected anything - it has failed to reject,
    and its recall is the LOCALIZER's, laundered through a judge that never says
    no. Plotting the two together shows the recall column collapsing into the
    x axis for exactly those arms.
    """
    if not rows:
        return ""
    W, H = 720, 380
    L, R, T, B = 66, 20, 26, 56
    implied = rows[0]["yes_rate_implied"]
    x0, x1 = 0.0, max(r["yes_rate"] for r in rows) * 1.08
    y0, y1 = min(r["obj_recall"] for r in rows) - 0.04, 1.0

    def px(x):
        return L + (x - x0) / (x1 - x0) * (W - L - R)

    def py(y):
        return T + (1 - (y - y0) / (y1 - y0)) * (H - T - B)

    colour = {"conservative": "var(--a1)", "lenient": "var(--a2)",
              "balanced": "var(--ok)"}
    g = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="YES rate against '
         f'object recall, with the rate the ground truth implies">']
    for k in range(0, 6):
        x = x0 + k * (x1 - x0) / 5
        g.append(f'<line x1="{px(x):.1f}" y1="{T}" x2="{px(x):.1f}" y2="{H - B}" '
                 f'stroke="var(--line-2)"/>')
        g.append(f'<text x="{px(x):.1f}" y="{H - B + 20}" text-anchor="middle" '
                 f'class="tick">{100 * x:.0f}%</text>')
    for k in range(0, 5):
        y = y0 + k * (y1 - y0) / 4
        g.append(f'<line x1="{L}" y1="{py(y):.1f}" x2="{W - R}" y2="{py(y):.1f}" '
                 f'stroke="var(--line-2)"/>')
        g.append(f'<text x="{L - 10}" y="{py(y) + 4:.1f}" text-anchor="end" '
                 f'class="tick">{y:.2f}</text>')
    # the calibrated line
    g.append(f'<line x1="{px(implied):.1f}" y1="{T}" x2="{px(implied):.1f}" '
             f'y2="{H - B}" stroke="var(--ok)" stroke-width="1.5" '
             f'stroke-dasharray="5 4"/>')
    g.append(f'<text x="{px(implied) + 8:.1f}" y="{T + 13}" class="pt" '
             f'fill="var(--ok)">what the labels imply: {100 * implied:.0f}%</text>')
    g += place([(px(r["yes_rate"]), py(r["obj_recall"]),
                 SHORT.get(r["model"], r["model"]),
                 colour.get(r["prompt"], "var(--ink-2)"),
                 r["model"] == BASELINE[0] and r["prompt"] == BASELINE[1])
                for r in rows], W)
    g.append(f'<text x="{(W + L) / 2:.0f}" y="{H - 12}" text-anchor="middle" '
             f'class="axis">how often the model answers yes</text>')
    g.append(f'<text transform="translate(16 {(H - B + T) / 2:.0f}) rotate(-90)" '
             f'text-anchor="middle" class="axis">objects found</text>')
    g.append("</svg>")
    return "\n".join(g)


def scatter(rows) -> str:
    """Object recall against frame specificity - the trade that decides use."""
    if not rows:
        return ""
    W, H = 720, 400
    L, R, T, B = 66, 20, 24, 56
    xs = [r["frame_specificity"] for r in rows]
    ys = [r["obj_recall"] for r in rows]
    x0, x1 = min(min(xs) - 0.03, 0.90), 1.005
    y0, y1 = min(ys) - 0.03, max(ys) + 0.03

    def px(x):
        return L + (x - x0) / (x1 - x0) * (W - L - R)

    def py(y):
        return T + (1 - (y - y0) / (y1 - y0)) * (H - T - B)

    colour = {"conservative": "var(--a1)", "lenient": "var(--a2)",
              "balanced": "var(--ok)"}
    g = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Object recall '
         f'against frame specificity, one point per judge and prompt">']
    for k in range(0, 6):
        x = x0 + k * (x1 - x0) / 5
        g.append(f'<line x1="{px(x):.1f}" y1="{T}" x2="{px(x):.1f}" y2="{H - B}" '
                 f'stroke="var(--line-2)"/>')
        g.append(f'<text x="{px(x):.1f}" y="{H - B + 20}" text-anchor="middle" '
                 f'class="tick">{x:.2f}</text>')
    for k in range(0, 6):
        y = y0 + k * (y1 - y0) / 5
        g.append(f'<line x1="{L}" y1="{py(y):.1f}" x2="{W - R}" y2="{py(y):.1f}" '
                 f'stroke="var(--line-2)"/>')
        g.append(f'<text x="{L - 10}" y="{py(y) + 4:.1f}" text-anchor="end" '
                 f'class="tick">{y:.2f}</text>')
    g += place([(px(r["frame_specificity"]), py(r["obj_recall"]),
                 SHORT.get(r["model"], r["model"]),
                 colour.get(r["prompt"], "var(--ink-2)"),
                 r["model"] == BASELINE[0] and r["prompt"] == BASELINE[1])
                for r in rows], W)
    g.append(f'<text x="{(W + L) / 2:.0f}" y="{H - 12}" text-anchor="middle" '
             f'class="axis">clean trams left alone</text>')
    g.append(f'<text transform="translate(16 {(H - B + T) / 2:.0f}) rotate(-90)" '
             f'text-anchor="middle" class="axis">objects found</text>')
    g.append("</svg>")
    return "\n".join(g)


def build() -> str:
    rows = load()
    base = next((r for r in rows if r["model"] == BASELINE[0]
                 and r["prompt"] == BASELINE[1]), rows[-1])
    loud = max(rows, key=lambda r: r["yes_rate"])
    intern = [r for r in rows if r["model"].startswith("InternVL")]
    i_con = next((r for r in intern if r["prompt"] == "conservative"), None)
    i_bal = next((r for r in intern if r["prompt"] == "balanced"), None)
    glm_bal = next((r for r in rows if r["model"] == BASELINE[0]
                    and r["prompt"] == "balanced"), None)
    noisy = [r for r in rows if r["frame_specificity"] < 0.5]
    n_arms, n_models = len(rows), len({r["model"] for r in rows})
    implied = base["yes_rate_implied"]

    wordings = "\n".join(
        f"<dt>{p.title()}</dt><dd>{PROMPT_NOTE[p]}</dd>"
        for p in PROMPT_ORDER if any(r["prompt"] == p for r in rows))

    return f"""{head(**NOTE.head_args)}

<header class="masthead">
  <span class="eyebrow">ARSI · judging step</span>
  <h1>VLM benchmark</h1>
  <p class="lede">{n_models} vision models, each asked the question three
  different ways, on the same {base['regions']} image crops. The setup currently
  in use comes out on top and stays unchanged.</p>
</header>

<section>
  <div class="sect-head"><h2>Findings</h2></div>
  <div class="finding">
    <ol>
      <li><b>No configuration beats the one in use.</b>
        {BASELINE[0]} with the cautious wording keeps
        {base['inst_detected']} of {base['inst_total']} objects, raises no false
        alarm, and {100 * base['obj_precision']:.0f} % of what it reports is
        real. None of the other {n_arms - 1} runs improves on that.</li>
      <li><b>The higher scores in the table are not real.</b>
        {len(noisy)} runs report more objects, but they answer yes to between
        {100 * min(r['yes_rate'] for r in noisy):.0f} and
        {100 * max(r['yes_rate'] for r in noisy):.0f} % of every crop shown to
        them. They would raise an alarm on most clean trams. The benchmark
        implies a correct rate of about {100 * implied:.0f} %.</li>
      <li><b>Rewording the question changes nothing here.</b> On the model in use
        the three wordings give {base['inst_detected']},
        {glm_bal['inst_detected']} and 56 objects. It does help a model that is
        too cautious to begin with: on InternVL the same rewording goes from
        {i_con['inst_detected']} to {i_bal['inst_detected']} objects with no
        false alarm.</li>
      <li><b>Choosing a different off-the-shelf model has run its course.</b>
        The {base['inst_total'] - base['inst_detected']} objects still missed are
        not recovered by any of these models at an acceptable false-alarm rate.
        Training a model on our own images is the next step.</li>
    </ol>
  </div>
</section>

<section>
  <div class="sect-head"><h2>Method</h2></div>
  <div class="col">
    <p>The image crops were produced once, by the best configuration from the
    models note, and reused unchanged in all {n_arms} runs. Every model therefore
    saw exactly the same {base['regions']} crops of the same {base['cases']}
    cases, so any difference in the results comes from the model or the wording
    and nothing else.</p>
    <p>Three wordings of the same question were tried, differing only in how
    cautious the model is told to be:</p>
    <dl class="wordings">{wordings}</dl>
    <p>Two figures are reported for every run. How many of the
    {base['inst_total']} labelled objects survive the judge, and how often the
    model answers yes at all. The second one matters: the benchmark holds
    {base['inst_total']} real anomalies among {base['regions']} crops, so a
    correctly calibrated model should answer yes to about
    {100 * implied:.0f} % of them. A model well above that is not finding more,
    it is refusing less.</p>
  </div>
</section>

<section>
  <div class="sect-head"><h2>Results</h2></div>
  <figure class="full">
    <div class="tablewrap"><table>{grid_table(rows)}</table></div>
    <figcaption>Objects kept out of {base['inst_total']}, then how often the
    model said yes, how many of those answers were right, and the share of clean
    trams it would have raised an alarm on. That last figure appears only where a
    run raised any.</figcaption>
  </figure>
  <figure>
    <div class="chart">{calibration(rows)}</div>
    <div class="legend">
      <span><i style="background:var(--a1)"></i>conservative</span>
      <span><i style="background:var(--a2)"></i>lenient</span>
      <span><i style="background:var(--ok)"></i>balanced</span>
    </div>
    <figcaption>The dashed line is the rate a correctly calibrated model should
    answer yes at. The score rises to the right because a model that rarely says
    no keeps everything it is given. The circled point is the setup in use.</figcaption>
  </figure>
  <figure>
    <div class="chart">{scatter(rows)}</div>
    <figcaption>The same runs against false alarms. Higher is better, further
    right is safer. Four runs never raise a false alarm, and the one in use is
    the best of those.</figcaption>
  </figure>
</section>

<section>
  <div class="sect-head"><h2>Limits</h2></div>
  <div class="col">
    <ul class="plain">
      <li><b>One set of crops.</b> A wording suited to these tight crops may not
      suit the much larger regions the pixel-difference detector produces. The
      result is which model for these crops, not which model in general.</li>
      <li><b>The false-alarm figure rests on few clean images</b>, and the {tram_star()}
      clean frames come from the same recordings the filtering model was trained
      on. A run that raises no alarm here is not proven safe.</li>
      <li><b>Three wordings is not a search of the prompt space.</b> They differ
      on one axis only, which is what makes them comparable.</li>
      <li><b>Nothing was fine-tuned.</b> Prompts cost minutes to try, so they went
      first. The groundwork for training on our own images already exists in the
      project and has never been run.</li>
      <li><b>Side note.</b> <code>qwen3-vl:8b-instruct</code> is still the default
      model name written in <code>vlm_05_reference_diff.py</code>, although the
      benchmark has always been run with GLM. On these crops that default is
      unusable.</li>
    </ul>
  </div>
</section>

<p class="footnote">{TRAM_FOOT}</p>

</div>
"""


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    DOC.write_text(build())
    print(f"{DOC}  ({DOC.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
