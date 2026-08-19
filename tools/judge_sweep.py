#!/usr/bin/env python3
"""Sweep the JUDGE (model x prompt) over a FIXED set of proposed boxes.

WHY. As of 2026-08-19 the localizer is no longer the constraint: it proposes
69/73 instances and the judge keeps 55/73, so the judge loses three and a half
times more than the proposer. The loss is entirely on `object` (37/53) while
damage is 6/6. Fixing the box set is what makes this a judge experiment - every
arm sees the identical crops, so any difference is the model or the wording.

Arms run as subprocesses: each one loads a different model into Ollama, and a
crashed or absent model must cost one arm, not the sweep.

    python tools/judge_sweep.py --localizer ddgate0.05 \
        --models haervwe/GLM-4.6V-Flash-9B:latest,qwen3-vl:8b-instruct \
        --prompts conservative,lenient,balanced

Each arm -> benchmark/runs/judge-<model>-<prompt>/. Collect with
tools/collect_judge.py.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def slug(model: str) -> str:
    return (model.split("/")[-1].replace(":", "-").replace(".", "")
            .replace("_", "-").lower())


def arm_name(model: str, prompt: str) -> str:
    return f"judge-{slug(model)}-{prompt}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--localizer", default="ddgate0.05",
                    help="the FIXED proposer; every arm judges its boxes")
    ap.add_argument("--models", required=True, help="comma list of Ollama tags")
    ap.add_argument("--prompts", default="conservative,lenient,balanced")
    ap.add_argument("--refs", default="all")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    models = [x.strip() for x in args.models.split(",") if x.strip()]
    prompts = [x.strip() for x in args.prompts.split(",") if x.strip()]
    runs = REPO_ROOT / "benchmark" / "runs"

    total = len(models) * len(prompts)
    print(f"{total} arms: {len(models)} models x {len(prompts)} prompts, "
          f"boxes from '{args.localizer}'\n")
    t0 = time.time()
    done, failed = [], []
    for i, model in enumerate(models, 1):
        for j, prompt in enumerate(prompts, 1):
            name = arm_name(model, prompt)
            n = (i - 1) * len(prompts) + j
            if (runs / name / "results.json").exists():
                print(f"[{n}/{total}] {name}  ALREADY DONE, skipping")
                done.append(name)
                continue
            cmd = [sys.executable, "tools/rescore_localizer.py",
                   "--localizer", args.localizer, "--refs", args.refs,
                   "--model", model, "--prompt", prompt, "--out", name]
            print(f"[{n}/{total}] {name}", flush=True)
            if args.dry_run:
                print("        " + " ".join(cmd))
                continue
            r = subprocess.run(cmd, cwd=REPO_ROOT)
            (done if r.returncode == 0 else failed).append(name)
            print(f"        -> {'ok' if r.returncode == 0 else 'FAILED'}  "
                  f"({time.time() - t0:.0f}s elapsed)", flush=True)

    print(f"\nSWEEP DONE  {len(done)} ok, {len(failed)} failed, "
          f"{time.time() - t0:.0f}s")
    if failed:
        print("failed arms:", ", ".join(failed))


if __name__ == "__main__":
    main()
