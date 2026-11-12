"""The home page's public data: the wins reel (D99).

It serves the file the daily refresh writes (``python -m Scripts.data_platform home-wins``) and never reads the
platform database, so a visit cannot slow the board or the refresh. No file, no reel: the page hides the section.
"""
from __future__ import annotations

import sys
from pathlib import Path

from fastapi import APIRouter

_SCRIPTS = Path(__file__).resolve().parents[2]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from data_platform.services import home_wins  # noqa: E402

router = APIRouter(prefix="/api/home", tags=["home"])
WINS_PATH = home_wins.DEFAULT_PATH


@router.get("/wins")
def wins() -> dict:
    """``home_wins.v1``: last full week's best wins as the board showed them at kickoff, and the week's record."""
    return home_wins.load(WINS_PATH)
