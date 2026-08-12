#!/usr/bin/env python3
# =============================================================================
#  ARSI-VLM - tools/rescore_gate.py
#  End-to-end A/B of the DINOv2 feature gate, at ZERO fresh VLM calls.
#
#  The gate (tools/dino_localizer.localize_gated) only ever DROPS regions the
#  shipped localizer produced - the survivors keep their exact bbox, so their
#  cache key (ref|img|bbox|model|fingerprint) is unchanged and every verdict is
#  already in benchmark/cache.json from the 2026-07-30 GLM x conservative run.
#  That makes the full object-level comparison measurable on this laptop.
#
#  Scoring is NOT reimplemented: run_benchmark.run_case/metrics/write_report are
#  reused as-is, with m.localize monkeypatched and the output paths redirected,
#  so the control arm (--gate -1, which filters nothing) must reproduce
#  benchmark/report.md to the digit.
#
#      python tools/rescore_gate.py --gate -1     # control = shipped
#      python tools/rescore_gate.py --gate 0.08
#      -> benchmark/report_gate<X>.md / results_gate<X>.json
# =============================================================================

import sys, argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "benchmark"))

import vlm_05_reference_diff as m
import run_benchmark as rb
import tools.dino_localizer as dl

MISSES = []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", type=float, default=0.08,
                    help="min DINOv2 patch distance inside a region (-1 = keep all)")
    ap.add_argument("--stat", default="max", choices=["max", "mean"])
    ap.add_argument("--model", default="haervwe/GLM-4.6V-Flash-9B:latest",
                    help="must match a cached (model, prompt) pair to stay free")
    args = ap.parse_args()

    m.MODEL_NAME = args.model
    tag = "shipped" if args.gate < 0 else f"{args.gate:g}"
    rb.REPORT = rb.BENCH_DIR / f"report_gate{tag}.md"
    rb.RESULTS = rb.BENCH_DIR / f"results_gate{tag}.json"

    def gated(ref_path, img_path):
        regions, info = dl.localize_gated(ref_path, img_path,
                                          gate=args.gate, stat=args.stat)
        info["total"] = len(regions)
        return regions, info

    m.localize = gated

    # Loud guard: this run must not make a single fresh VLM call. A miss would
    # silently score the region NO on CPU (2-4 min/call) instead.
    orig = rb.classify_cached

    def guarded(image, reference, region, cache, fp, img_key, ref_key):
        key = f"{ref_key}|{img_key}|{region['bbox']}|{m.MODEL_NAME}|{fp}"
        if key not in cache:
            MISSES.append(key)
        return orig(image, reference, region, cache, fp, img_key, ref_key)

    rb.classify_cached = guarded

    print(f"gate={args.gate} stat={args.stat} model={args.model}")
    print(f"fingerprint={rb.prompt_fingerprint()}  -> {rb.REPORT.name}")
    rb.main()

    if MISSES:
        print(f"\n!!! {len(MISSES)} UNCACHED regions - results are NOT a pure "
              f"re-score. First: {MISSES[0]}")
    else:
        print("\nAll verdicts served from cache (0 fresh VLM calls).")


if __name__ == "__main__":
    main()
