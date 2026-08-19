#!/usr/bin/env python3
"""Turn the harvested RTX job summary into the comparison tables of REPORT.md."""
import json
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
S = json.load(open(os.path.join(HERE, "summary.json")))

# Sources are re-uploads of the same four clips; group them by (frames, size).
SOURCES = {
    (123, 510.0): "1762-1  empty tram (123 f)",
    (176, 769.0): "1762-3  tram with clothes on a seat (176 f)",
    (73, 339.0): "1762-2  STAGED: planted phone/wallet/backpack (73 f)",
    (73, 769.0): "1762-3  same clip, 73-frame extraction",
    (12, 42.8): "1762-4  short clip (12 f)",
}
vid_label = {}
for v, m in S["videos"].items():
    key = (m["frames_extracted"], m["source_mb"])
    vid_label[v] = SOURCES.get(key, f"{v} ({m['frames_extracted']} f)")


def label(v):
    return vid_label.get(v.rstrip("*"), v)


by_src = defaultdict(list)
for j in S["jobs"]:
    by_src[label(j["video"])].append(j)

lines = ["# RTX Studio jobs — harvested comparison", ""]
lines += [
    "Every job run on the 3080 Ti workstation, aggregated from the untracked",
    "`data/app/jobs/` tree. `flag` = frames carrying at least one detection.",
    "`(cache)` marks a replay that made no fresh VLM call (median frame < 1 s).",
    "",
]

for src in sorted(by_src, key=lambda s: -len(by_src[s])):
    jobs = by_src[src]
    lines += [f"## {src}", "", f"{len(jobs)} jobs", ""]
    lines += ["| pipeline | model | prompt | ref | mask | frames | flag | det | s/frame |",
              "|---|---|---|---|---|---:|---:|---:|---:|"]
    for j in sorted(jobs, key=lambda x: (x["script"], -x["detections"])):
        note = " *(cache)*" if j["cache_replay"] else ""
        lines.append(
            f"| {j['script']} | {j['model'][:26]}{note} | {j['prompt']} | "
            f"{'yes' if j['has_ref'] else '-'} | {'yes' if j['has_mask'] else '-'} | "
            f"{j['n_frames']} | {j['flagged']} | {j['detections']} | {j['sec_per_frame']} |"
        )
    lines.append("")

out = os.path.join(HERE, "REPORT.md")
with open(out, "w") as fh:
    fh.write("\n".join(lines) + "\n")
print(f"wrote {out} ({len(lines)} lines)")
