"""Test CLI for the core engine (the FastAPI backend is the real consumer).

Examples (repo root, venv active):
  python -m arsi_core models
  python -m arsi_core extract --video data/videos/1762-4.mp4 --every-n 25 --out data/app/videos/v4
  python -m arsi_core mask-apply --mask data/app/masks/tram1762.json --image f0001.jpg --out masked.jpg
  python -m arsi_core run --script vlm_05 --reference data/reference/tram_1762_v1_f0227_masked_reference.jpg \
      --frames data/masked/tram_1762_v2_f0037_masked.jpg --model qwen3-vl:8b-instruct
"""
import argparse
import json
import sys
from pathlib import Path

from .adapters import SCRIPTS
from .errors import ArsiError
from .masking import MaskSpec
from .ollama_client import OllamaClient
from .runner import JobConfig, run_job
from .video import extract_frames


def _cmd_models(args):
    health = OllamaClient().health()
    if not health["reachable"]:
        print(f"Ollama: UNREACHABLE ({health.get('detail', '')})")
        return 1
    print("Ollama: OK")
    for name in health["models"]:
        print(f"  {name}")
    return 0


def _cmd_extract(args):
    meta = extract_frames(args.video, args.out, every_n=args.every_n,
                          every_s=args.every_s, start_s=args.start,
                          end_s=args.end, max_side=args.max_side)
    print(f"{len(meta['frames'])} frames -> {args.out} (meta.json written)")
    return 0


def _cmd_mask_apply(args):
    spec = MaskSpec.load(args.mask)
    out = spec.apply_file(args.image, args.out)
    print(f"mask '{spec.name}' (hash {spec.hash}) -> {out}")
    return 0


def _cmd_run(args):
    frames = []
    for f in args.frames:
        p = Path(f)
        frames.extend(sorted(p.glob("*.jpg")) + sorted(p.glob("*.png"))
                      if p.is_dir() else [p])
    params = {}
    for kv in args.param or []:
        k, _, v = kv.partition("=")
        try:
            v = json.loads(v)
        except json.JSONDecodeError:
            pass
        params[k] = v
    if args.max_retries is not None:
        params["max_retries"] = args.max_retries

    cfg = JobConfig(script=args.script, frames=[str(f) for f in frames],
                    model=args.model, reference=args.reference, mask=args.mask,
                    prompt=Path(args.prompt_file).read_text(encoding="utf-8")
                    if args.prompt_file else None,
                    prompt_name=Path(args.prompt_file).stem if args.prompt_file else "default",
                    params=params, job_dir=args.out)

    def show(event):
        if event["event"] == "frame_done":
            flag = {True: "ANOMALY", False: "clean"}.get(event["anomaly"], "FAILED")
            print(f"  [{event['index'] + 1}/{len(frames)}] {event['frame_id']}: "
                  f"{flag}  ({event['n_detections']} boxes, {event['seconds']}s, "
                  f"attempt {event['attempts']})")
        elif event["event"] == "frame_retry":
            print(f"    retry: {event['error']}")

    result = run_job(cfg, on_event=show)
    s = result.summary
    print(f"\n{result.status.upper()}: {s.n_ok}/{s.n_frames} ok, "
          f"{s.n_anomalous} anomalous, {s.n_failed} failed, {s.wall_seconds}s")
    print(f"results: {Path(cfg.job_dir) / 'results.json'}")
    return 0 if result.status == "completed" else 1


def main(argv=None):
    ap = argparse.ArgumentParser(prog="arsi_core", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("models", help="Ollama health + installed models")

    p = sub.add_parser("extract", help="video -> frames")
    p.add_argument("--video", required=True)
    p.add_argument("--out", required=True)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--every-n", type=int, help="keep 1 frame in N")
    g.add_argument("--every-s", type=float, help="1 frame every N seconds")
    p.add_argument("--start", type=float, default=0.0)
    p.add_argument("--end", type=float, default=None)
    p.add_argument("--max-side", type=int, default=None)

    p = sub.add_parser("mask-apply", help="apply a mask preset to one image")
    p.add_argument("--mask", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--out", required=True)

    p = sub.add_parser("run", help="run a pipeline over frames")
    p.add_argument("--script", required=True, choices=sorted(SCRIPTS))
    p.add_argument("--frames", nargs="+", required=True,
                   help="image files and/or directories")
    p.add_argument("--model", default=None, help="default: script's MODEL_NAME")
    p.add_argument("--reference", default=None)
    p.add_argument("--mask", default=None, help="path to a MaskSpec JSON")
    p.add_argument("--prompt-file", default=None)
    p.add_argument("--max-retries", type=int, default=None)
    p.add_argument("--param", action="append",
                   help="KEY=VALUE; UPPER_CASE keys override script config "
                        "(e.g. PERSON_FILTER=false, DIFF_THRESHOLD=35)")
    p.add_argument("--out", default=None, help="job dir (default: data/app/jobs/<id>)")

    args = ap.parse_args(argv)
    try:
        return {"models": _cmd_models, "extract": _cmd_extract,
                "mask-apply": _cmd_mask_apply, "run": _cmd_run}[args.cmd](args)
    except ArsiError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
