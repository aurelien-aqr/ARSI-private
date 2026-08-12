#!/usr/bin/env python3
"""Extract the original, overlay-free H.264 streams from a DTI .dvs container.

For every frame the .dvs stores an ASCII metadata block followed by the raw
H.264 Annex-B payload. I-frames carry a 16-byte binary "DTIS264I" header
before the first start code; P-frames begin directly with 00 00 00 01 61.

Frames are grouped into 12-frame GOPs per camera. The last frame of each GOP
is followed by a container trailer that must be stripped, otherwise the
decoder swallows index bytes and corrupts frames: magic "PIC2DTI4" + 324
bytes, preceded by a 16-byte record (offset + size + FILETIME) and a
variable-length run of zero padding.

We locate the metadata blocks by regex, take a frame's payload to run from
the end of its block to the start of the next one, strip the trailer, and
feed each camera to a dedicated ffmpeg that remuxes to mp4 without
re-encoding.

Usage: carve_dvs.py <file.dvs> <output_dir> [--prefix 1760] [--fps 25]
"""

import argparse
import mmap
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

META = re.compile(
    rb"IC:[0-9a-f]{128};SN:([^;]*);VI:([^;]*);CN:Cam (\d+);"
    rb"DT:(\d{14});LL:([^;]*);Alarms:([^\x00]*)\x00"
)
IFRAME_HDR = b"DTIS264I"
IFRAME_HDR_LEN = 16
START_CODE = b"\x00\x00\x00\x01"
TRAILER_MAGIC = b"PIC2DTI4"
TRAILER_PRE_RECORD = 16  # offset + size + FILETIME, glued in front of the magic


def strip_trailer(payload):
    """Remove the end-of-GOP container trailer. Returns (payload, found)."""
    j = payload.find(TRAILER_MAGIC)
    if j < 0:
        return payload, False
    end = j - TRAILER_PRE_RECORD
    # the zero padding in front does not belong to the NAL: an H.264 NAL cannot
    # contain 00 00 00 (emulation prevention) and ends on a non-zero byte
    # (rbsp_trailing_bits)
    while end > 0 and payload[end - 1] == 0:
        end -= 1
    return payload[:end], True


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("dvs", type=Path)
    p.add_argument("outdir", type=Path)
    p.add_argument("--prefix", default=None, help="output filename prefix (default: the stream's VI)")
    p.add_argument("--fps", default="25", help="frame rate declared at remux time")
    return p.parse_args()


def main():
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    logdir = args.outdir / "_carve_logs"
    logdir.mkdir(exist_ok=True)

    fh = open(args.dvs, "rb")
    buf = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)

    matches = list(META.finditer(buf))
    if not matches:
        sys.exit("no metadata block recognised: unexpected format")
    print(f"{len(matches)} frames found", flush=True)

    vehicle = matches[0].group(2).decode()
    prefix = args.prefix or vehicle
    site = matches[0].group(1).decode()

    procs, logs, first_ts, last_ts = {}, {}, {}, {}
    counts, anomalies, trailers = Counter(), Counter(), Counter()

    def writer(cam):
        if cam not in procs:
            out = args.outdir / f"{prefix}-cam{cam:02d}.mp4"
            log = open(logdir / f"cam{cam:02d}.log", "wb")
            procs[cam] = subprocess.Popen(
                ["ffmpeg", "-hide_banner", "-y", "-r", args.fps, "-f", "h264",
                 "-i", "pipe:0", "-c", "copy", "-movflags", "+faststart", str(out)],
                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log,
            )
            logs[cam] = log
        return procs[cam].stdin

    for i, m in enumerate(matches):
        cam = int(m.group(3))
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(buf)
        payload = buf[start:end]

        if payload.startswith(IFRAME_HDR):
            payload = payload[IFRAME_HDR_LEN:]
        payload, had_trailer = strip_trailer(payload)
        if had_trailer:
            trailers[cam] += 1
        if not payload.startswith(START_CODE):
            # nothing is dropped: ffmpeg resynchronises on the first start code,
            # but we count the anomaly to report it at the end of the pass
            anomalies[payload[:4]] += 1

        writer(cam).write(payload)
        counts[cam] += 1
        ts = m.group(4).decode()
        first_ts.setdefault(cam, ts)
        last_ts[cam] = ts

        if i % 25000 == 0:
            print(f"  {i}/{len(matches)}", flush=True)

    for cam, p in sorted(procs.items()):
        p.stdin.close()
    for cam, p in sorted(procs.items()):
        if p.wait() != 0:
            print(f"  !! ffmpeg failed on cam{cam:02d}, see {logdir}/cam{cam:02d}.log")
        logs[cam].close()

    print(f"\n{vehicle} / {site}")
    for cam in sorted(counts):
        print(f"  cam{cam:02d}: {counts[cam]:6d} frames  "
              f"{first_ts[cam][8:]}->{last_ts[cam][8:]}"
              f"  {trailers[cam]} trailers stripped")
    if anomalies:
        print("\npayloads with no leading start code (resynchronised by ffmpeg):")
        for prefix_bytes, n in anomalies.most_common(10):
            print(f"  {prefix_bytes.hex()}  x{n}")


if __name__ == "__main__":
    main()
