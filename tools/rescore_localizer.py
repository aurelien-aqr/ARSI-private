#!/usr/bin/env python3
# =============================================================================
#  ARSI-VLM - tools/rescore_localizer.py
#  End-to-end A/B of ANY localizer spec, through the real VLM judge.
#
#  WHY. The 2026-08-19 comparison (docs/dino_models/) is localization
#  only: it shows that AnomalyDINO boxes an instance properly where the pixel
#  diff draws a frame-sized blob, but it makes 0 VLM calls, so "a better box
#  becomes a better verdict" stayed an assumption. This runs the judge.
#
#  It generalises tools/rescore_gate.py, which could only do the DINOv2 gate and
#  only for free: a gate's survivors keep their exact bbox, so their cache key is
#  unchanged. A spec whose PROPOSER differs draws new boxes, so its verdicts are
#  genuinely new VLM calls - that is the cost of the question, not a bug, and the
#  miss count is reported rather than guarded against.
#
#  Scoring is NOT reimplemented: run_benchmark.run_case / metrics / write_report
#  are reused with m.localize monkeypatched and the outputs redirected, so
#  `--localizer shipped --refs real,variant` must reproduce the shipped report.
#
#      python tools/rescore_localizer.py --localizer shipped
#      python tools/rescore_localizer.py --localizer ddgate0.05 --refs all
#      -> benchmark/runs/loc-<spec>-<refs>/{report.md,results.json}
# =============================================================================

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "benchmark"))

import vlm_05_reference_diff as m                                # noqa: E402
import run_benchmark as rb                                       # noqa: E402
import tools.localizer_specs as specs                            # noqa: E402
import tools.judge_prompts as prompts                            # noqa: E402

MISSES = []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--localizer", required=True,
                    help="spec understood by tools/localizer_specs.propose, "
                         "e.g. shipped, gate0.08, dino4@0.08, ddgate0.05")
    ap.add_argument("--refs", default="real,variant",
                    help="comma list of benchmark reference keys, or 'all'. The "
                         "default is what run_benchmark has always scored; 'all' "
                         "adds the 1760 and 3333 cameras")
    ap.add_argument("--model", default="haervwe/GLM-4.6V-Flash-9B:latest")
    ap.add_argument("--prompt", default="conservative",
                    choices=sorted(prompts.PROMPTS),
                    help="the judge prompt; changing it changes the cache "
                         "fingerprint, so every arm is a fresh set of calls")
    ap.add_argument("--tag", default="", help="suffix for the output directory")
    ap.add_argument("--out", default="", help="explicit output directory name "
                    "under benchmark/runs/ (the judge sweep names its own arms)")
    args = ap.parse_args()

    m.MODEL_NAME = args.model
    m.PROMPT = prompts.get(args.prompt)

    gt = rb.load_json(rb.GT_PATH, None)
    refs = (tuple(gt["references"]) if args.refs == "all"
            else tuple(r.strip() for r in args.refs.split(",")))
    unknown = [r for r in refs if r not in gt["references"]]
    if unknown:
        raise SystemExit(f"unknown reference key(s): {unknown}. "
                         f"Known: {sorted(gt['references'])}")
    rb.GT_REFS = refs

    slug = args.localizer.replace("@", "").replace("+", "_").replace(".", "")
    scope = "all" if args.refs == "all" else "default"
    name = args.out or f"loc-{slug}-{scope}{args.tag}"
    out = rb.BENCH_DIR / "runs" / name
    out.mkdir(parents=True, exist_ok=True)
    rb.REPORT = out / "report.md"
    rb.RESULTS = out / "results.json"

    #: bbox -> camera, so the Dinomaly specs pick the right checkpoint. The
    #: monkeypatched localize() only receives paths, so the mapping is built
    #: from the ground truth once and looked up by inspection-image path.
    cam_by_image = {}
    for c in gt["cases"]:
        if c.get("reference") in refs:
            cam_by_image[str(rb.resolve(c["image"]))] = specs.camera_of(c)

    def localize(ref_path, img_path):
        camera = cam_by_image.get(str(img_path), "tram_1762")
        return specs.propose(args.localizer, ref_path, img_path, camera)

    m.localize = localize

    orig = rb.classify_cached

    def counted(image, reference, region, cache, fp, img_key, ref_key):
        key = f"{ref_key}|{img_key}|{region['bbox']}|{m.MODEL_NAME}|{fp}"
        if key not in cache:
            MISSES.append(key)
        return orig(image, reference, region, cache, fp, img_key, ref_key)

    rb.classify_cached = counted

    print(f"localizer={args.localizer}  refs={len(refs)} "
          f"({', '.join(refs)})\nmodel={args.model}  -> {out}")
    t0 = time.time()
    rb.main()
    print(f"\n{len(MISSES)} fresh VLM calls, {time.time() - t0:.0f}s total")
    print(f"report: {rb.REPORT}")


if __name__ == "__main__":
    main()
