"""Engine switches: the environment settings that can change what the engine reads or predicts.

Every switch is read here so that each Match Read result and each published pick can
record which were in effect (rework A3.7, tracker D16 rule 3 and D26): the model version
fingerprints code files only, so a scorecard could not otherwise split picks by stage.
An unrecognised value falls back to the default, and the snapshot names it as ignored.
"""
from __future__ import annotations

import os
from typing import Dict

VERSION = "engine-switches.v1"

# name -> (default, accepted values). Defaults are today's behaviour.
SWITCHES: Dict[str, tuple] = {
    "DOCUMENT_STORE": ("chroma", frozenset({"chroma", "sqlite"})),          # A2a
    "SPIX_COUNT_DISPERSION": ("floors", frozenset({"floors", "fixed"})),    # A3.5, stage 1
    "SPIX_DEVIG_METHOD": ("proportional", frozenset({"proportional", "power"})),  # A3.1 (D17), stage 2
    # D21/D26 stage 3 (strict); D29 stage 4 (goals); D35: full = goals plus corners at the line consensus.
    "SPIX_MARKET_ANCHOR": ("off", frozenset({"off", "strict", "goals", "full"})),
    "SPIX_SELECTION_POLICY": ("legacy", frozenset({"legacy", "market_value"})),  # D33 (b), D35: stage 4 bundle
    "SPIX_BOOKMAKER_SET": ("all", frozenset({"all", "romania"})),              # D32, D35: picks only at these books
}


def _raw(name: str) -> str:
    return os.environ.get(name, "").strip().lower()


def value(name: str) -> str:
    """The value in effect for one switch."""
    default, accepted = SWITCHES[name]
    raw = _raw(name)
    return raw if raw in accepted else default


def snapshot() -> dict:
    """Every switch's value in effect, plus any setting that was not recognised."""
    out = {"version": VERSION, **{name: value(name) for name in SWITCHES}}
    ignored = {name: _raw(name) for name in SWITCHES if _raw(name) and _raw(name) not in SWITCHES[name][1]}
    if ignored:
        out["ignored"] = ignored
    return out
