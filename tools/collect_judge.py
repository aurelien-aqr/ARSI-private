#!/usr/bin/env python3
"""Collect the judge-sweep arms into docs/vlm_benchmark/metrics.json.

Scoring is re-derived with run_benchmark.metrics, never re-implemented, so this
table cannot drift from the per-arm reports it summarises.

    venv/bin/python tools/collect_judge.py
"""

import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "benchmark"))

import run_benchmark as rb                                       # noqa: E402

OUT = REPO_ROOT / "docs" / "vlm_benchmark" / "metrics.json"
PROMPTS = ("conservative", "lenient", "balanced")

#: Ollama tags are long and the report needs a short name. Anything not listed
#: falls back to the directory slug, so an unknown model still appears.
PRETTY = {
    "glm-46v-flash-9b-latest": "GLM-4.6V-Flash-9B",
    "qwen3-vl-8b-instruct": "Qwen3-VL-8B",
    "qwen25vl-7b": "Qwen2.5-VL-7B",
    "internvl3-5-8b": "InternVL3.5-8B",
    "minicpm-v46-latest": "MiniCPM-V-4.6",
    "cosmos-reason2-8b-gguf-q4-k-m": "Cosmos-Reason2-8B",
}


def expected_cases():
    """How many cases a complete arm must have scored.

    run_benchmark writes results.json after EVERY case, so a sweep that is still
    running leaves a perfectly parseable partial file behind. Scoring one would
    silently put an arm with 30 of 73 instances into the table as if it had lost
    the rest, which is the worst kind of wrong number: plausible.
    """
    import json as _json
    gt = _json.loads((REPO_ROOT / "benchmark" / "datasets"
                      / "ground_truth.json").read_text())
    return len(gt["cases"])


def main():
    want = expected_cases()
    rows, partial = [], []
    for d in sorted((REPO_ROOT / "benchmark" / "runs").glob("judge-*")):
        res = d / "results.json"
        if not res.exists():
            continue
        mt = re.match(r"judge-(.+)-(" + "|".join(PROMPTS) + r")$", d.name)
        if not mt:
            print(f"  skipping {d.name}: not a judge arm name")
            continue
        mslug, prompt = mt.group(1), mt.group(2)
        data = json.loads(res.read_text())
        if len(data["rows"]) != want:
            partial.append((d.name, len(data["rows"])))
            continue
        frame, obj = rb.metrics(data["rows"])
        n_regions = sum(r["n_regions"] for r in data["rows"])
        rows.append({
            "model": PRETTY.get(mslug, mslug), "model_tag": data.get("model", ""),
            "prompt": prompt, "cases": frame["n"],
            "regions": n_regions,
            "kept": obj["kept_total"],
            # THE metric that makes this grid readable. The ground truth puts
            # inst_total anomalies among n_regions crops, so a calibrated judge
            # says YES to about inst_total/n_regions of them. An arm at 88 % is
            # not detecting anything - it is failing to reject, and its recall
            # is just the localizer's, laundered through a judge that never says
            # no. Ordering by this explains the whole table.
            "yes_rate": obj["kept_total"] / n_regions if n_regions else 0.0,
            "yes_rate_implied": obj["inst_total"] / n_regions if n_regions else 0.0,
            "frame_f1": frame["f1"], "frame_recall": frame["recall"],
            "frame_specificity": frame["specificity"],
            "obj_recall": obj["recall"], "obj_recall_strict": obj["recall_strict"],
            "obj_precision": obj["region_precision"],
            "inst_detected": obj["inst_detected"], "inst_total": obj["inst_total"],
            "per_type": obj["per_type"],
        })
    for name, n in partial:
        print(f"  skipping {name}: {n}/{want} cases, still running")
    if not rows:
        raise SystemExit("no COMPLETE judge-* arms under benchmark/runs/")
    # object recall first - the metric this sweep exists to move - then
    # specificity, because a recall gain paid for in false alarms is not a gain
    rows.sort(key=lambda r: (-r["obj_recall"], -r["frame_specificity"]))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=2))
    print(f"wrote {OUT}  ({len(rows)} arms)\n")
    implied = rows[0]["yes_rate_implied"]
    print(f"ground truth implies a YES rate of {100 * implied:.1f} % "
          f"({rows[0]['inst_total']} instances in {rows[0]['regions']} crops)\n")
    print(f"{'model':<20} {'prompt':<13} {'YES%':>6} {'objRec':>7} {'objStrict':>9} "
          f"{'objPrec':>8} {'frSpec':>7} {'frF1':>6}")
    for r in sorted(rows, key=lambda r: r["yes_rate"]):
        print(f"{r['model']:<20} {r['prompt']:<13} {100 * r['yes_rate']:>5.1f}% "
              f"{r['inst_detected']:>3}/{r['inst_total']:<3} "
              f"{r['obj_recall_strict']:>9.3f} {r['obj_precision']:>8.3f} "
              f"{r['frame_specificity']:>7.3f} {r['frame_f1']:>6.3f}")


if __name__ == "__main__":
    main()
