"""ARSI Studio core engine: video -> frames -> pipeline (vlm_01..05) -> results.

Pure-Python, no UI. See docs/SPEC.md for the contract this package implements.
The vlm_0x scripts at the repository root are imported as modules (never run
as subprocesses); their CLI behaviour and defaults are untouched.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

APP_DATA = REPO_ROOT / "data" / "app"

__version__ = "0.1.0"
