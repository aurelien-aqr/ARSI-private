#!/usr/bin/env python3
"""Prompt variants for the judge sweep, kept OUT of vlm_05_reference_diff.py.

The shipped module owns the two prompts the pipeline actually uses (PROMPT, the
conservative default, and PROMPT_LENIENT). Anything under test lives here until
it wins, so a sweep can never quietly change what a normal run does.

WHY A THIRD ONE. Measured 2026-08-19 end to end on the 68 cases: object recall
is 55/73 and does not move with the localizer, and the loss is entirely on
`object` (37/53) while damage is 6/6. The judge sees ONE region at a time, so
this is not a "several objects per frame" problem - it is per-crop rejection of
genuine objects, concentrated on the smaller ones (median lost 10,100 px vs
15,000 px kept) and on the v2 frames that carry four staged items.

The conservative prompt was written to fix a PRECISION problem: with pixel-diff
boxes, region precision was ~0.75. With the feature-proposed boxes it is 0.896
before any prompt change. So the hypothesis this sweep tests is that the
conservative wording is now paying for a problem the localizer already solved.

`balanced` therefore keeps every NO condition that protects specificity (person,
lighting, reflection, scratches on metal) and drops only the two clauses that
bias toward NO on a real but unobtrusive object: the "be conservative" framing
and "if you are unsure, answer NO".
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
import vlm_05_reference_diff as m                                # noqa: E402

BALANCED = """You are shown the SAME spot in a tram twice, side by side:
- LEFT half  = the clean reference (normal, empty, undamaged tram).
- RIGHT half = the same spot now.

Report a new ANOMALY that is present on the RIGHT and absent on the LEFT. An
anomaly is one of:
- a forgotten object (bag, backpack, phone, wallet, bottle, package),
- litter left behind (can, bottle, wrapper, paper, food),
- graffiti: painted letters, a tag or a scribble on a surface,
- damage / vandalism: a torn or slashed seat, a broken part.

The object may be SMALL, partly hidden behind a seat or a pole, or lying flat on
the floor. Size does not matter: a phone or a bottle counts exactly as much as a
suitcase. Judge only this crop, on its own.

Answer NO (it is NOT an anomaly) if ANY of these hold:
- the RIGHT shows the SAME empty seat, floor, wall, pole or panel as the LEFT and
  only the brightness, shadow, reflection or colour differs,
- the difference is a person, or anything a person is wearing, holding or sitting
  on,
- it is a scratch, glare or reflection on metal or glass (these are NOT graffiti).

Reply with YES or NO, then name what appeared in 2-4 words."""

#: name -> prompt text. `conservative` and `lenient` are read from the shipped
#: module rather than copied, so a change there cannot silently desync the sweep.
PROMPTS = {
    "conservative": m.PROMPT,
    "lenient": m.PROMPT_LENIENT,
    "balanced": BALANCED,
}


def get(name: str) -> str:
    if name not in PROMPTS:
        raise SystemExit(f"unknown prompt '{name}'. Known: {sorted(PROMPTS)}")
    return PROMPTS[name]
