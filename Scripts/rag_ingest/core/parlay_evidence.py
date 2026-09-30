"""Shared deterministic evidence and confidence from canonical quote assessment."""
from .quote_assessment import assess_leg, quote_evidence


def leg_standalone_confidence(leg, league):
    return assess_leg(leg, league)


def leg_evidence(leg, league):
    return quote_evidence(leg, league)
