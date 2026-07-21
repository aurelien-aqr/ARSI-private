"""Human-readable explanation of each pipeline, served with /api/pipelines.

Lives next to the backend rather than in the frontend so the wording stays with
the code it describes: when a pipeline's behaviour changes, this file is in the
same review as the change. The frontend only renders it.

Every entry is optional-by-key; the UI skips sections that are missing.
Numbers quoted here come from benchmark/report.md (29-case ground truth) and
must be updated when that report is regenerated.
"""

PIPELINE_DOCS = {

    "vlm_01": {
        "summary": "The simplest baseline: one frame goes to the VLM, which "
                   "answers a fixed structured report. No reference, no "
                   "localization, no boxes.",
        "inputs": "One inspection frame.",
        "output": "Structured text per frame: GRAFFITI / VANDALISM / FORGOTTEN "
                  "OBJECT (yes/no + note), DESCRIPTION, SEVERITY 1-5.",
        "steps": [
            ("Send the frame", "The whole image is passed to the local VLM "
             "through Ollama with a prompt that fixes the answer format."),
            ("Parse the answer", "The reply is read back into the five fields. "
             "A frame counts as anomalous if any of the three yes/no lines is "
             "yes."),
        ],
        "strengths": [
            "No reference image needed — works on any camera, any framing.",
            "One VLM call per frame: the cheapest pipeline here.",
        ],
        "limits": [
            "No bounding box: you learn that something is wrong, not where.",
            "The model judges the scene from scratch every time, so normal tram "
            "fittings (worn seats, scratched panels, posters) are regularly "
            "read as anomalies.",
        ],
    },

    "vlm_02": {
        "summary": "Same idea as vlm_01, but the model is shown a clean "
                   "reference next to the inspection frame and asked only for "
                   "what differs.",
        "inputs": "A clean reference frame + one inspection frame.",
        "output": "Structured text describing the differences only.",
        "steps": [
            ("Send both images", "Reference and inspection frame go to the VLM "
             "in the same call."),
            ("Ask for differences only", "The prompt tells the model to ignore "
             "anything present in both images and report only what is new in "
             "the inspection frame."),
        ],
        "strengths": [
            "The reference suppresses most false alarms on permanent fittings: "
            "a scratched panel is in both images, so it is not reported.",
        ],
        "limits": [
            "Still no bounding box.",
            "The comparison is done inside the model, over two full frames. "
            "Small objects occupy very few tokens and are routinely missed — "
            "this is the failure mode vlm_05 exists to fix.",
        ],
    },

    "vlm_03": {
        "summary": "Asks the VLM itself to output bounding boxes as JSON over "
                   "the whole frame — the model does both the finding and the "
                   "localizing.",
        "inputs": "One inspection frame.",
        "output": "JSON list of detections (label, normalized box, severity) "
                  "plus an annotated image, boxes coloured green→red by "
                  "severity.",
        "steps": [
            ("Ask for JSON", "The prompt requests a list of anomalies, each "
             "with a normalized 0-1 bounding box and a 1-5 severity."),
            ("Parse and draw", "The JSON is parsed, denormalized to pixels and "
             "drawn with Pillow."),
        ],
        "strengths": [
            "Boxes without any extra detector or reference image.",
        ],
        "limits": [
            "Box coordinates from a general-purpose VLM are approximate — the "
            "box often lands near the object rather than on it.",
            "Output can be unparseable JSON, which costs a retry.",
        ],
    },

    "vlm_04": {
        "summary": "Hybrid two-stage POC for forgotten personal objects: an "
                   "open-vocabulary detector proposes candidates, the VLM "
                   "confirms each one.",
        "inputs": "One inspection frame, optionally a clean reference.",
        "output": "Confirmed objects as JSON + an annotated image.",
        "steps": [
            ("Localize with YOLO-World", "An open-vocabulary detector is asked "
             "for a fixed class list (cell phone, wallet, handbag, backpack, "
             "bag, suitcase, laptop) at several image scales, with a very low "
             "confidence threshold so small objects survive."),
            ("Filter against the reference (optional)", "Candidates already "
             "present in the clean reference are dropped, keeping only what is "
             "new."),
            ("Confirm with the VLM", "Each surviving box is cropped with "
             "context and sent to the VLM for a short yes/no + label."),
        ],
        "strengths": [
            "Needs no reference image if the filter step is off — usable on a "
            "moving camera.",
            "The detector gives tight, honest boxes.",
        ],
        "limits": [
            "The detector is the ceiling: it misses small objects at CCTV "
            "distance (a wallet on a seat), and nothing downstream can recover "
            "what was never proposed.",
            "Closed to its class list: graffiti, litter and seat damage are not "
            "detectable classes, so this pipeline only covers forgotten "
            "objects.",
        ],
    },

    "vlm_05": {
        "summary": "For a FIXED camera. Instead of asking a model to find "
                   "anomalies, it computes what physically changed since a "
                   "clean reference frame, then asks the VLM to judge each "
                   "changed region. Recommended.",
        "inputs": "A clean reference frame (same camera, empty tram) + the "
                  "inspection frames.",
        "output": "One box per kept region with a short label, plus a "
                  "frame-level anomalous/clean verdict.",
        "steps": [
            ("Photometric diff", "Both frames are converted to grayscale, "
             "blurred (BLUR_RADIUS) and subtracted. A pixel counts as changed "
             "when the difference exceeds DIFF_THRESHOLD (default 40)."),
            ("Group into regions", "Changed pixels are grouped into connected "
             "components on a downscaled mask, dilated so a fragmented object "
             "becomes one box. Specks below MIN_AREA are dropped; degenerate "
             "regions above MAX_AREA (a global exposure shift) are dropped too."),
            ("Two extra channels", "A second photometric pass at a lower "
             "threshold catches low-contrast objects (a dark bottle on a dark "
             "floor). An added-edge channel — extra edge energy where the "
             "reference is locally flat — catches faint graffiti that no global "
             "threshold reaches. Both only ADD boxes; they never modify the "
             "base ones."),
            ("Person veto", "YOLOv8-nano finds people once per frame. Regions "
             "mostly inside a person box are dropped before any VLM call. This "
             "is what separates 'jacket worn by a passenger' (vetoed) from "
             "'jacket forgotten on a seat' (kept) — a label blacklist cannot "
             "make that distinction."),
            ("Salience cap", "At most MAX_REGIONS regions per frame are kept, "
             "ranked by salience (mean diff intensity × √area), so a bright "
             "small phone is not evicted by a dim large lighting blob."),
            ("Merge fragments", "One object rarely produces one region — a "
             "backpack and its strap land as separate components. Neighbouring "
             "regions are fused before any VLM call, so the judge sees the "
             "object rather than a fragment. A fill guard stops the merge from "
             "chaining neighbours into one frame-sized box."),
            ("VLM judges each region", "Each region is cropped from BOTH images "
             "with context padding and sent side by side (reference | now). The "
             "model answers YES/NO plus a 2-4 word name. In 'filter' mode the "
             "NO regions are dropped; in 'label' mode every region is kept and "
             "just named."),
            ("Verdict", "Surviving boxes that still overlap heavily are merged "
             "a second time, and the frame is flagged anomalous if at least one "
             "box remains."),
        ],
        "strengths": [
            "The diff finds everything: measured localization recall is 45/45 "
            "instances on the 29-case ground truth, including faint tags and "
            "objects too small for any detector.",
            "The VLM never has to search the frame — it only answers a yes/no "
            "about one small crop, which it does far more reliably.",
            "Frame-level accuracy 1.000 on the benchmark (17 anomalous, 12 "
            "clean, no false alarm and no miss).",
        ],
        "limits": [
            "Needs a fixed camera and a clean reference of the same scene. A "
            "different session (exposure, onboard displays) produces many more "
            "candidate regions — the VLM still rejects them, but the cost rises.",
            "Object-level precision is the weak point: 0.663 region precision "
            "and 0.889 instance recall. The pre-VLM merge reduces the "
            "fragmentation behind those numbers but has not yet been scored "
            "end to end — the figures here predate it.",
            "Cost scales with the number of changed regions: roughly 20 VLM "
            "calls per frame on a busy frame.",
        ],
    },
}
