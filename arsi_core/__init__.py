"""ARSI Studio core engine: video -> frames -> pipeline (vlm_01..05) -> results.

Pure-Python, no UI. See docs/SPEC.md for the contract this package implements.
The vlm_0x scripts at the repository root are imported as modules (never run
as subprocesses); their CLI behaviour and defaults are untouched.
"""
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ARSI_APP_DATA relocates all runtime state (jobs, masks, videos, cache,
# settings) - the test suite points it at a temp dir to leave data/app alone.
APP_DATA = Path(os.environ.get("ARSI_APP_DATA") or REPO_ROOT / "data" / "app")

# --- tram ids -----------------------------------------------------------
# The 2026-08-11 multi-camera capture is filed under the tram id "3333".
# That id is a PLACEHOLDER: the real fleet number of that vehicle is not
# known. It used to be filed under "39T", which is worse - 39T is the Skoda
# type (tram 1760 is a 39T too), not a vehicle. Everything that shows the id
# to a human marks it with an asterisk and one of the notes below.
PLACEHOLDER_TRAM_IDS = ("3333",)
#: Matches the id as an id ("3333", "3333-cam52", "3333_cam52_083517") and not
#: as four digits inside a number - a box coordinate like 817.3333740234375
#: must not drag the footnote into a report.
PLACEHOLDER_TRAM_RE = re.compile(r"(?<![\d.])3333(?!\d)")


def names_placeholder_tram(text) -> bool:
    """True when `text` carries the placeholder tram id."""
    return bool(PLACEHOLDER_TRAM_RE.search(str(text)))
TRAM_ID_NOTE = ("3333 is a placeholder, not the tram's real fleet number - the "
                "vehicle number of the 2026-08-11 capture is unknown. It was "
                "called 39T before, but 39T is the Skoda type, which tram 1760 "
                "shares.")
#: Markdown footnote for any report/doc whose text carries the id.
TRAM_ID_NOTE_MD = "\\* " + TRAM_ID_NOTE
#: `<abbr>` wrapper for HTML/UI: the id, an asterisk, the note on hover.
def tram_id_html(text):
    """`text` with the placeholder id starred and explained on hover."""
    if not names_placeholder_tram(text):
        return text
    return (f'<abbr title="{TRAM_ID_NOTE}" style="text-decoration:underline dotted;'
            f' cursor:help;">{text}<sup>*</sup></abbr>')

__version__ = "0.1.0"
