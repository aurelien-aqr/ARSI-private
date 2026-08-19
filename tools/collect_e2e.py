#!/usr/bin/env python3
"""Collect the end-to-end runs of tools/rescore_localizer.py into one file the
report reads: docs/dino_models/end_to_end.json.

Each arm is a directory under benchmark/runs/ holding the results.json that
run_benchmark wrote. Scoring is re-derived here with run_benchmark.metrics, not
re-implemented, so the table cannot drift from the reports themselves.

    venv/bin/python tools/collect_e2e.py \
        shipped:"Pixel diff" gate0.08:"Pixel diff + AnomalyDINO gate" \
        dino4@0.08:"AnomalyDINO alone" ddgate0.05:"AnomalyDINO + Dinomaly veto"
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "benchmark"))

import run_benchmark as rb                                       # noqa: E402

OUT = REPO_ROOT / "docs" / "dino_models" / "end_to_end.json"


def slug(spec: str) -> str:
    return spec.replace("@", "").replace("+", "_").replace(".", "")


def main():
    args = sys.argv[1:]
    if not args:
        raise SystemExit(__doc__)
    rows = []
    for arg in args:
        spec, _, label = arg.partition(":")
        path = rb.BENCH_DIR / "runs" / f"loc-{slug(spec)}-all" / "results.json"
        if not path.exists():
            raise SystemExit(f"missing {path} - run tools/rescore_localizer.py "
                             f"--localizer {spec} --refs all first")
        data = json.loads(path.read_text())
        frame, obj = rb.metrics(data["rows"])
        rows.append({
            "spec": spec, "label": label or spec,
            "cases": frame["n"],
            "regions": sum(r["n_regions"] for r in data["rows"]),
            "frame_f1": frame["f1"],
            "frame_recall": frame["recall"],
            "frame_specificity": frame["specificity"],
            "obj_recall": obj["recall"],
            "obj_recall_strict": obj["recall_strict"],
            "obj_precision": obj["region_precision"],
            "inst_detected": obj["inst_detected"],
            "inst_total": obj["inst_total"],
            "per_source": obj["per_source"],
        })
    OUT.write_text(json.dumps(rows, indent=2))
    print(f"wrote {OUT}  ({len(rows)} arms)")
    w = max(len(r["label"]) for r in rows)
    print(f"{'arm':<{w}} {'regions':>8} {'frameF1':>8} {'objRec':>7} {'objPrec':>8}")
    for r in rows:
        print(f"{r['label']:<{w}} {r['regions']:>8} {r['frame_f1']:>8.3f} "
              f"{r['obj_recall']:>7.3f} {r['obj_precision']:>8.3f}")


if __name__ == "__main__":
    main()
