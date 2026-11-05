"""Pick notes (tracker D75): one line for each pick and one against it, in plain words.

Each Most likely and Value pick gets a small fact pack: why Spix lists it (the likeliest option in its market, or a
price above what the market's chance says it is worth), the market's chance, and how often that exact option would
have won in each team's finished matches in this competition this season (its last ``LAST`` and the season, each
match read with the team in its role tonight). Recent form is the only outside fact the owner allowed, and every pick
gets the same kinds of fact, so none is chosen because it flatters the pick.

GPT-5.6 may word the notes from those facts only. A note is refused when it uses a figure that is not in its pick's
facts, a number in words, a promise, a tipster word or an instruction to bet; that pick then takes the fixed wording
built here. Notes are reused while the facts are unchanged (``fact_key``), as the Match Read briefing does. No engine
output changes: the notes describe picks the layer already made.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from .recommendations import MIN_PRICE, wins

VERSION = "pick-notes.v1"
MODEL = "gpt-5.6"  # the Match Read briefing's model
LAST = 5
MIN_PLAYED = 3
MAX_WORDS = 32
MAX_CHARACTERS = 220

_REASONS = {
    "most_likely": f"The likeliest option in its market that still pays at least {MIN_PRICE:.2f}.",
    "value": "The best price found is above what the market's chance says it is worth.",
}
_REFUSED = re.compile(
    r"\b(locks?|bankers?|guarantee\w*|certain\w*|sure|can'?t lose|cannot lose|free money|models?|ai|"
    r"artificial intelligence|back it|bet|bets|betting|stake|must)\b", re.IGNORECASE)
_NUMBER_WORDS = re.compile(r"\b(two|three|four|five|six|seven|eight|nine|ten)\b", re.IGNORECASE)
_NUMBER = re.compile(r"\d+(?:\.\d+)?")

_INSTRUCTIONS = """You write short notes for the picks on one football match for Spix, a website where bettors
browse picks and build their own bets. For each pick you get its label, why Spix lists it (reason), the market's
chance in percent, about how many times in 10 it loses at that chance, and each team's recent form in this
competition this season, over its last few matches and the whole season. "counted" says what was counted: with
"about": "match", how many of that team's matches had it; with "about": "team", in how many the team did it.
"opposite" is what happened in the other matches, which is what cut against the pick.

For every pick write:
- "for": one or two sentences, at most 30 words: the reason with the chance, and the form figure that supports it
  most, if one does.
- "against": one sentence, at most 30 words: the form that cuts most against it, said as what happened (the
  opposite, counted as played minus landed); if none cuts against it, about how many times in 10 it still loses at
  the market's chance.

Use only the figures supplied for that pick and write every number in digits exactly as given. Call the form counts
matches and name the competition. With "about": "match" the count is of whole matches, never of the team alone:
write "4 of Ferencvaros's last 5 Europa League matches had 8 or more corners", never "Ferencvaros had 8 or more
corners". Say the reason in the words it is given, with no extra terms such as "qualifying". Never promise or suggest that a pick will win, never tell the reader to bet, and
never use the words lock, banker, guaranteed, sure, certain, model or AI. Plain English, short sentences, no
exclamation marks."""


def landed(option: Mapping[str, Any], *, home_goals, away_goals, home_corners=None, away_corners=None) -> Optional[bool]:
    """Whether ``option`` would have won on a finished match; None when it cannot be told (a whole or quarter line,
    which can be returned or split, or a corners count missing)."""
    line = option.get("line")
    if line is not None and float(line) % 1 != 0.5:
        return None
    if option.get("market") == "corners":
        if home_corners is None or away_corners is None:
            return None
        total = home_corners + away_corners
        return total > line if option.get("side") == "over" else total < line
    if home_goals is None or away_goals is None:
        return None
    return wins(option, home_goals, away_goals)


def counted(option: Mapping[str, Any], role: str) -> Optional[Tuple[str, str]]:
    """What a team's matches are counted for, in words from that team's side (``role``: its side tonight):
    ``("match", "had 3 or more goals")`` or ``("team", "lost by 2 or more")``. None for a whole or quarter line."""
    market, side, line = option.get("market"), option.get("side"), option.get("line")
    if market in ("goals", "corners"):
        if line is None or float(line) % 1 != 0.5:
            return None
        if side == "over":
            return "match", f"had {int(line) + 1} or more {market}"
        return "match", f"had {int(line)} {market} or fewer" if int(line) else f"had no {market}"
    if market == "btts":
        return "match", "saw both teams score" if side == "yes" else "had a team fail to score"
    if market == "moneyline":
        if side == "draw":
            return "team", "drew"
        line = -0.5
    elif market != "spreads" or line is None or float(line) % 1 != 0.5:
        return None
    margin = int(abs(float(line)) + 0.5)
    by = "" if margin == 1 else f" by {margin} or more"
    if float(line) < 0:
        return "team", f"won{by}" if role == side else f"lost{by}"
    if role == side:
        return "team", "avoided defeat" if margin == 1 else f"avoided losing{by}"
    return "team", f"failed to win{by}"


def opposite_counted(option: Mapping[str, Any], role: str) -> Optional[str]:
    """What happened in the matches ``option`` would have lost, in words from that team's side: the other side of a
    total or of both teams to score, the other team's side of a handicap (a win is the handicap at −0.5)."""
    market, side, line = option.get("market"), option.get("side"), option.get("line")
    if market == "moneyline":
        if side == "draw":
            return "did not draw"
        market, line = "spreads", -0.5
    other = {"over": "under", "under": "over", "yes": "no", "no": "yes", "home": "away", "away": "home"}.get(side)
    if other is None:
        return None
    flipped = {"market": market, "side": other,
               "line": -float(line) if market == "spreads" and line is not None else line}
    found = counted(flipped, role)
    return found[1] if found else None


def team_form(option: Mapping[str, Any], side: str, matches: Sequence[Mapping[str, Any]]) -> Optional[dict]:
    """How often ``option`` would have won in one team's finished matches (newest first): its last ``LAST`` and the
    season. Each match is read with the team in its role tonight (``side``) and its opponent in the other, so
    "Arsenal to win" counts Arsenal's wins in Arsenal's matches and the opponent's defeats in the opponent's. None
    under ``MIN_PLAYED`` countable matches."""
    counted: List[bool] = []
    for row in matches:
        own = (row.get("goals_for"), row.get("corners_for"))
        other = (row.get("goals_against"), row.get("corners_against"))
        home, away = (own, other) if side == "home" else (other, own)
        result = landed(option, home_goals=home[0], away_goals=away[0], home_corners=home[1], away_corners=away[1])
        if result is not None:
            counted.append(result)
    if len(counted) < MIN_PLAYED:
        return None
    last = counted[:LAST]
    return {"last": {"played": len(last), "landed": sum(last)},
            "season": {"played": len(counted), "landed": sum(counted)}}


def _percent(chance: float) -> int:
    return int(chance * 100 + 0.5)


def _loses_in_10(chance: float) -> int:
    return max(1, int((1 - chance) * 10 + 0.5))


def match_facts(block: Mapping[str, Any], fixture: Mapping[str, Any],
                histories: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict:
    """The fact pack of one match's picks: ``block`` is its ``recommendations.v1`` block, ``fixture`` names the teams
    and the competition, ``histories`` holds each side's finished matches in this competition this season."""
    teams = {"home": fixture.get("home_team"), "away": fixture.get("away_team")}
    picks = []
    for kind in ("most_likely", "value"):
        for index, item in enumerate(block.get(kind) or []):
            chance = item.get("chance")
            if not isinstance(chance, (int, float)) or not 0 < chance < 1:
                continue
            named = item.get("side") is not None
            form = [{"team": teams[side], "about": what[0], "counted": what[1],
                     "opposite": opposite_counted(item, side), **found}
                    for side in ("home", "away")
                    if named and (what := counted(item, side))
                    and (found := team_form(item, side, histories.get(side) or []))]
            pick = {"id": f"{kind}-{index}", "kind": kind, "label": item.get("label"), "reason": _REASONS[kind],
                    "chance_percent": _percent(chance), "loses_about_in_10": _loses_in_10(chance), "form": form}
            if kind == "value" and isinstance(item.get("worth_it_at"), (int, float)):
                pick["worth_it_at"] = item["worth_it_at"]
            picks.append(pick)
    return {"fixture": teams, "competition": fixture.get("competition"), "picks": picks}


def _rate(entry: Mapping[str, Any]) -> float:
    return entry["last"]["landed"] / entry["last"]["played"]


_RESULTS = ("won", "lost", "drew", "failed to win")


def _possessive(team: Any) -> str:
    return f"{team}'" if str(team).endswith("s") else f"{team}'s"


def _form_sentence(entry: Mapping[str, Any], competition: Optional[str], *, against: bool = False) -> str:
    """"3 of Arsenal's last 5 Premier League matches had 3 or more goals." or "Brighton avoided defeat in 3 of their
    last 5 Premier League matches."; ``against`` says what happened in the matches it would have lost."""
    landed, played = entry["last"]["landed"], entry["last"]["played"]
    count, what = (played - landed, entry["opposite"]) if against else (landed, entry["counted"])
    where = f"{competition} matches" if competition else "matches"
    if entry["about"] == "match":
        return f"{count} of {_possessive(entry['team'])} last {played} {where} {what}."
    joint = "" if what in _RESULTS else " in"
    return f"{entry['team']} {what}{joint} {count} of their last {played} {where}."


def fixed_note(pick: Mapping[str, Any], competition: Optional[str]) -> dict:
    """The fixed wording: the reason with the chance and the form most for it (if it won at least half the time);
    against, the form most against it (if under half), else how often it still loses at the market's chance."""
    percent = pick["chance_percent"]
    if pick.get("kind") == "value":
        first = (f"The best price found is above the {pick['worth_it_at']:.2f} it is worth at the market's {percent}%."
                 if pick.get("worth_it_at") else f"The best price found is above what the market's {percent}% says it is worth.")
    else:
        first = f"At {percent}%, this is the likeliest option in its market that still pays at least {MIN_PRICE:.2f}."
    form = pick.get("form") or []
    best = max(form, key=lambda e: (_rate(e), e["last"]["played"]), default=None)
    worst = min(form, key=lambda e: (_rate(e), -e["last"]["played"]), default=None)
    note_for = first if best is None or _rate(best) < 0.5 else f"{first} {_form_sentence(best, competition)}"
    if worst is not None and _rate(worst) < 0.5 and worst.get("opposite"):
        against = _form_sentence(worst, competition, against=True)
    else:
        times = pick["loses_about_in_10"]
        against = f"At {pick['chance_percent']}% it still loses about {times} {'time' if times == 1 else 'times'} in 10."
    return {"for": note_for, "against": against}


def _figures(text: Any) -> set:
    return {float(n) for n in _NUMBER.findall(str(text or ""))}


def _allowed(pick: Mapping[str, Any], facts: Mapping[str, Any]) -> set:
    allowed = {float(pick["chance_percent"]), float(pick["loses_about_in_10"]), 10.0, float(MIN_PRICE)}
    if pick.get("worth_it_at"):
        allowed |= {float(pick["worth_it_at"]), round(float(pick["worth_it_at"]), 2)}
    for entry in pick.get("form") or []:
        allowed |= _figures(entry.get("counted")) | _figures(entry.get("opposite"))
        for span in ("last", "season"):
            played, landed = entry[span]["played"], entry[span]["landed"]
            allowed |= {float(played), float(landed), float(played - landed)}
    for named in (pick.get("label"), facts.get("competition"), *(facts.get("fixture") or {}).values()):
        allowed |= _figures(named)
    return allowed


def _clean(text: Any) -> str:
    return " ".join(str(text or "").split())


def _team_owns_match_count(text: str, pick: Mapping[str, Any]) -> bool:
    """A whole-match count said as one team's own ("Arsenal had 3 or more goals"), which misstates who did it."""
    return any(re.search(rf"\b{re.escape(str(entry['team']))}\s+(had|saw)\b", text)
               for entry in pick.get("form") or [] if entry.get("about") == "match")


def valid_note(note: Mapping[str, Any], pick: Mapping[str, Any], facts: Mapping[str, Any]) -> bool:
    """True when both lines are present, short, free of promises and tipster words, use only the pick's own figures
    in digits, and never give a whole-match count to one team."""
    allowed = _allowed(pick, facts)
    for key in ("for", "against"):
        text = _clean(note.get(key))
        if (not text or len(text.split()) > MAX_WORDS or len(text) > MAX_CHARACTERS or _REFUSED.search(text)
                or _NUMBER_WORDS.search(text) or not _figures(text) <= allowed or _team_owns_match_count(text, pick)):
            return False
    return True


def fact_key(facts: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(facts, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:24]


def notes_for_match(facts: Mapping[str, Any], *, generator: Optional[Callable[[Mapping[str, Any]], Mapping]] = None,
                    cached: Optional[Mapping[str, Any]] = None) -> dict:
    """``{version, fact_key, model, notes: {pick id: {for, against, source}}}``: GPT's wording where it passes
    ``valid_note``, the fixed wording elsewhere (``source`` says which). ``cached`` notes with the same facts are
    returned as they are, so nothing is asked twice."""
    key = fact_key(facts)
    if cached and cached.get("version") == VERSION and cached.get("fact_key") == key:
        return dict(cached)
    try:
        written = (generator or generate_gpt_notes)(facts) or {}
    except Exception:  # a missing note must never cost the match its picks
        written = {}
    by_id = {n.get("id"): n for n in written.get("notes") or [] if isinstance(n, Mapping)}
    notes = {}
    for pick in facts.get("picks") or []:
        candidate = by_id.get(pick["id"])
        if candidate is not None and valid_note(candidate, pick, facts):
            notes[pick["id"]] = {"for": _clean(candidate["for"]), "against": _clean(candidate["against"]),
                                 "source": "gpt"}
        else:
            notes[pick["id"]] = {**fixed_note(pick, facts.get("competition")), "source": "fixed"}
    return {"version": VERSION, "fact_key": key, "model": _model(), "notes": notes}


def _model() -> str:
    return os.environ.get("PICK_NOTES_MODEL", "").strip() or MODEL


def generate_gpt_notes(facts: Mapping[str, Any]) -> Mapping[str, Any]:
    """One GPT-5.6 call per match: ``{"notes": [{id, for, against}]}`` from the fact pack alone. The client is created
    lazily, so nothing else needs OpenAI or its key; OpenAI does not store the request."""
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured")
    from openai import OpenAI

    note = {"type": "object", "additionalProperties": False, "required": ["id", "for", "against"],
            "properties": {"id": {"type": "string"}, "for": {"type": "string"}, "against": {"type": "string"}}}
    response = OpenAI(api_key=api_key).responses.create(
        model=_model(), instructions=_INSTRUCTIONS, input=json.dumps(facts, sort_keys=True, separators=(",", ":")),
        reasoning={"effort": "low"}, max_output_tokens=1500, store=False,
        text={"format": {"type": "json_schema", "name": "pick_notes", "strict": True,
                         "schema": {"type": "object", "additionalProperties": False, "required": ["notes"],
                                    "properties": {"notes": {"type": "array", "items": note}}}}},
    )
    output = str(getattr(response, "output_text", "") or "").strip()
    if not output:
        raise RuntimeError("GPT returned no notes")
    value = json.loads(output)
    if not isinstance(value, Mapping):
        raise RuntimeError("GPT returned a non-object")
    return value
