"""Result dataclasses — the single JSON contract shared by the runner, the
backend, the report and the exports (docs/SPEC.md "FrameResult schema")."""
from dataclasses import dataclass, field, asdict
from typing import Optional

ANOMALY_TYPES = ("object", "graffiti", "damage", "litter", "unknown")


@dataclass
class Detection:
    label: str                          # what the model said ("phone on seat")
    type: str = "unknown"               # one of ANOMALY_TYPES
    bbox: Optional[list] = None         # [x0, y0, x1, y1] pixels, None for whole-frame pipelines
    zone: Optional[str] = None          # free-text zone ("left-hand seats") from vlm_01/02
    severity: Optional[int] = None      # 1-5 when the pipeline reports one
    confidence: Optional[float] = None  # detector confidence (vlm_04)
    channel: Optional[str] = None       # vlm_05 localizer channel (photo/photo_lo/edge)


@dataclass
class FrameResult:
    frame_id: str
    image: str
    status: str = "ok"                  # ok | failed | skipped
    attempts: int = 1
    seconds: float = 0.0
    anomaly: Optional[bool] = None      # None when status != ok
    detections: list = field(default_factory=list)   # [Detection]
    raw_response: str = ""
    error: Optional[str] = None

    def to_dict(self):
        return asdict(self)


@dataclass
class JobSummary:
    n_frames: int = 0
    n_ok: int = 0
    n_anomalous: int = 0
    n_failed: int = 0
    wall_seconds: float = 0.0


@dataclass
class JobResult:
    job_id: str
    config: dict
    started: str = ""                   # ISO timestamps
    finished: str = ""
    status: str = "completed"           # completed | failed | cancelled
    frames: list = field(default_factory=list)       # [FrameResult]
    summary: JobSummary = field(default_factory=JobSummary)

    def to_dict(self):
        return asdict(self)


def guess_type(label: str) -> str:
    """Map a free-text VLM label to an anomaly type (vlm_05 labels carry no
    explicit type; keyword match is enough for filtering/reporting)."""
    low = label.lower()
    if any(w in low for w in ("graffiti", "tag", "scribble", "drawn", "marking")):
        return "graffiti"
    if any(w in low for w in ("torn", "slash", "broken", "damage", "crack", "vandal")):
        return "damage"
    if any(w in low for w in ("litter", "trash", "wrapper", "can ", "debris", "crumpled")):
        return "litter"
    if any(w in low for w in ("bag", "backpack", "phone", "wallet", "bottle", "suitcase",
                              "package", "jacket", "object", "umbrella", "box")):
        return "object"
    return "unknown"
