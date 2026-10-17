"""Bookmakers a member can bet with, per country (rework D32, D35).

A pick is published only at a bookmaker in the active set (``SPIX_BOOKMAKER_SET``); the market
consensus still uses every bookmaker, because more books make it sharper.

Romania (owner's choice, 17 Oct 2026): Betano and Superbet, both on the ONJN register of
Class I licensees (onjn.gov.ro/licentiati-clasa-i, checked 17 Oct 2026). API-Football carries
Betano for most fixtures and, since 6 Oct 2026, no Superbet prices; Superbet counts as soon as
a source provides it.
"""
from __future__ import annotations

from . import engine_switches

VERSION = "bookmaker-sets.v1"
SETS = {"romania": frozenset({"betano", "superbet"})}


def active() -> str:
    return engine_switches.value("SPIX_BOOKMAKER_SET")


def allowed(bookmaker) -> bool:
    """True when a pick may be published at this bookmaker under the active set."""
    name = active()
    return name == "all" or str(bookmaker or "").strip().casefold() in SETS[name]
