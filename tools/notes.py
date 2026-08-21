"""The standalone notes under docs/: one row each, read by everything.

The builder that writes a note, the Studio that serves it, and the printer that
turns it into a PDF all need the same facts about it. They used to be spelled
out separately in three files, which is how the sidebar ended up offering a note
whose PDF was named after another one.

    from tools.notes import NOTES, MARKDOWN_NOTES

Two kinds of note sit in this list. Three are assembled from measurements by a
builder of their own (tools/build_alignment_doc.py and friends), so every number
on the page is read out of a JSON file rather than typed. Two were written as
prose in Markdown and are still edited that way: they carry a `source`, and
tools/build_md_note.py converts them. Both kinds render through
tools/report_style.py and are served identically.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS = REPO_ROOT / "docs"


@dataclass(frozen=True)
class Note:
    #: URL key: /notes/<key> for the page, /notes/<key>.pdf for the print
    key: str
    #: shown in the bar, the tab, and the Studio sidebar
    title: str
    #: the self-contained page its builder writes
    html: Path
    #: what a browser saves the PDF as, and the running foot of every sheet
    pdf_name: str
    running_head: str
    #: the Markdown this note is converted from, for the ones that have one
    source: Path | None = None
    #: the kicker above the title. The measured notes carry their own; a
    #: converted one has no place to put it, since its Markdown starts at the
    #: title, so it is stated here.
    eyebrow: str = ""

    @property
    def pdf(self) -> Path:
        return self.html.with_suffix(".pdf")

    #: keyword arguments of report_style.head(), so a note cannot title its
    #: page one thing and its PDF another
    @property
    def head_args(self) -> dict:
        return {"title": self.title, "key": self.key, "filename": self.pdf_name}


NOTES = {n.key: n for n in [
    Note("camera-alignment", "Framing drift 1760 → 3333",
         DOCS / "camera_alignment" / "camera_framing_drift_1760_3333.html",
         "arsi-framing-drift-1760-3333.pdf", "Framing drift 1760 → 3333 · ARSI"),
    Note("dino-models", "Dinomaly & AnomalyDINO",
         DOCS / "dino_models" / "dino_models.html",
         "arsi-dino-models.pdf", "Dinomaly & AnomalyDINO · ARSI"),
    Note("vlm-benchmark", "VLM benchmark",
         DOCS / "vlm_benchmark" / "vlm_benchmark.html",
         "arsi-vlm-benchmark.pdf", "VLM benchmark · ARSI"),
    Note("public-datasets", "Public datasets",
         DOCS / "PUBLIC_DATASETS.html",
         "arsi-public-datasets.pdf", "Public datasets · ARSI",
         source=DOCS / "PUBLIC_DATASETS.md",
         eyebrow="ARSI · publication · evaluation plan"),
    Note("lora-plan", "LoRA plan",
         DOCS / "LORA_PLAN.html",
         "arsi-lora-plan.pdf", "LoRA plan · ARSI",
         source=DOCS / "LORA_PLAN.md",
         eyebrow="ARSI · judging step · fine-tuning"),
]}

#: the ones tools/build_md_note.py builds
MARKDOWN_NOTES = {k: n for k, n in NOTES.items() if n.source}
