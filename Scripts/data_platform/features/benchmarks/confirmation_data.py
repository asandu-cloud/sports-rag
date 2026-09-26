"""One-time confirmation inputs; later outcome stores never enter this reader.

The audit archive is inspected only after the permanent opening claim. Its
later history payloads and duplicated fixture outcomes are structurally skipped,
not decoded. This is an auditable trusted-code boundary, not a secrecy sandbox.
"""
from __future__ import annotations

from collections import Counter
from datetime import timedelta
import re

from Scripts.rag_ingest.core import model_features as core
from Scripts.data_platform.features.phase3_features import HistoryIndex, build_reference
from . import baselines as b
from .artifacts import sha
from .data import MARKETS, COMPETITIONS, _decode, _count, _validate_feature_row

PARTITION = "phase3_confirmation"
FEATURES = f"lockbox/{PARTITION}/features.jsonl"
LABELS = f"lockbox/{PARTITION}/labels.jsonl"
HISTORY = "audit/inputs.json"
_TOKEN = re.compile(r'"(?:[^"\\]|\\.)*"|[{}\[\]]')
_STRING = re.compile(r'"(?:[^"\\]|\\.)*"')


def _white(text, pos):
    while pos < len(text) and text[pos].isspace():
        pos += 1
    return pos


def _end(text, pos):
    """Skip a JSON value without constructing its nested values."""
    pos = _white(text, pos)
    if pos >= len(text):
        raise ValueError("Truncated archive")
    if text[pos] == '"':
        match = _STRING.match(text, pos)
        if not match:
            raise ValueError("Unterminated archive string")
        return match.end()
    if text[pos] in "[{":
        stack = []
        for token in _TOKEN.finditer(text, pos):
            value = token.group()
            if value.startswith('"'):
                continue
            if value in "[{":
                stack.append(value)
            elif not stack or stack.pop() != {"}": "{", "]": "["}[value]:
                raise ValueError("Mismatched archive container")
            if not stack:
                return token.end()
        raise ValueError("Truncated archive container")
    stop = pos
    while stop < len(text) and text[stop] not in ",]} \t\r\n":
        stop += 1
    if stop == pos:
        raise ValueError("Missing archive value")
    return stop


def _members(text, start=0):
    pos = _white(text, start)
    if text[pos] != "{":
        raise ValueError("Expected archive object")
    pos = _white(text, pos + 1)
    seen = set()
    while text[pos] != "}":
        stop = _end(text, pos)
        key = _decode(text[pos:stop])
        if not isinstance(key, str) or key in seen:
            raise ValueError("Duplicate/invalid archive key")
        seen.add(key)
        pos = _white(text, stop)
        if text[pos] != ":":
            raise ValueError("Missing archive colon")
        pos = _white(text, pos + 1)
        stop = _end(text, pos)
        yield key, pos, stop
        pos = _white(text, stop)
        if text[pos] == "}":
            return
        if text[pos] != ",":
            raise ValueError("Missing archive separator")
        pos = _white(text, pos + 1)
        if text[pos] == "}":
            raise ValueError("Trailing archive comma")


def history_before(text, boundary):
    """Decode only history with assumed label availability strictly before end.

    No call to a JSON object decoder receives the fixtures array, full archive,
    or a history object on/after the later calibration boundary.
    """
    fields = {key: (a, z) for key, a, z in _members(text)}
    if set(fields) != {"competitions", "fixtures", "history"} or _white(text, _end(text, 0)) != len(text):
        raise ValueError("Unexpected archive structure")
    a, z = fields["competitions"]
    competitions = _decode(text[a:z])
    if competitions != COMPETITIONS:
        raise ValueError("Archive competition contract mismatch")
    a, z = fields["history"]
    if text[a] != "[":
        raise ValueError("History must be an array")
    pos, previous, result, skipped = _white(text, a + 1), None, [], 0
    boundary = core.utc(boundary)
    while text[pos] != "]":
        stop = _end(text, pos)
        metadata = {key: (lo, hi) for key, lo, hi in _members(text, pos) if key == "kickoff"}
        if set(metadata) != {"kickoff"}:
            raise ValueError("Missing history kickoff")
        lo, hi = metadata["kickoff"]
        kickoff = core.utc(_decode(text[lo:hi]))
        if previous is not None and kickoff < previous:
            raise ValueError("History chronology changed")
        previous = kickoff
        if kickoff + timedelta(hours=3) < boundary:
            row = _decode(text[pos:stop])
            if row.get("status") != "FT":
                raise ValueError("Unexpected non-regulation history")
            result.append(row)
        else:
            skipped += 1
        pos = _white(text, stop)
        if text[pos] == "]":
            break
        if text[pos] != ",":
            raise ValueError("Missing history separator")
        pos = _white(text, pos + 1)
        if text[pos] == "]":
            raise ValueError("Trailing history comma")
    if pos + 1 != z:
        raise ValueError("History span mismatch")
    return result, competitions, {"decoded_history_rows": len(result), "opaque_later_history_rows": skipped,
        "fixtures_array_decoded": False, "later_outcomes_decoded": False, "availability_end": boundary.isoformat()}


def checked_read(dataset, name, expected):
    if name not in {FEATURES, LABELS, HISTORY}:
        raise ValueError("Confirmation reader refuses this data store")
    path = dataset / name
    if path.resolve() != path or sha(path) != expected:
        raise ValueError("Confirmation artifact checksum/path mismatch")
    # Recheck the actual bytes returned, not just a previous stat/read.
    body = path.read_bytes()
    import hashlib
    if hashlib.sha256(body).hexdigest() != expected:
        raise ValueError("Confirmation artifact changed while reading")
    return body


def feature_rows(body, data):
    rows = [_decode(line) for line in body.splitlines() if line.strip()]
    seen, snapshots = set(), set()
    for row in rows:
        _validate_feature_row(data, row, partitions={PARTITION},
            start=data.boundaries["phase3_confirmation_start"], end=data.boundaries["calibration_start"])
        fid = row["fixture"]["fixture_id"]
        if fid in seen or row["snapshot_id"] in snapshots or any(k in row for k in ("labels", "target", "team_labels")):
            raise ValueError("Duplicate confirmation version or outcome in features")
        if any(k in row["fixture"] for k in ("home", "away", "goals", "score")):
            raise ValueError("Outcome embedded in fixture identity")
        seen.add(fid); snapshots.add(row["snapshot_id"])
    expected = {fid for fid, m in data.memberships.items() if m["partition"] == PARTITION}
    if seen != expected:
        raise ValueError("Confirmation feature membership mismatch")
    return sorted(rows, key=lambda r: (core.utc(r["as_of"]), r["fixture"]["fixture_id"]))


def prepare_rows(rows, history, competitions, *, scoring, end):
    """Replay features and reconstruct the exact two predeclared comparators."""
    if any(b._available(r, "assumed_final") >= core.utc(end) for r in history):
        raise ValueError("Later history must stay opaque")
    index, average = HistoryIndex(history), b._AverageIndex(history, "assumed_final")
    output = []
    for number, row in enumerate(rows, 1):
        eligible = [m for m in MARKETS if row["market_eligibility"][m]["eligible"]]
        if not eligible:
            continue
        snapshot, _, view = build_reference(row["fixture"], index, competitions)
        if (snapshot["snapshot_id"] != row["snapshot_id"] or view["values"] != row["values"]
                or view["support"] != row["support"] or view["feature_contract_id"] != row["feature_contract_id"]):
            raise ValueError("Confirmation feature/snapshot replay mismatch")
        cutoff = core.utc(row["as_of"])
        if any(b._available(r, "assumed_final") >= cutoff for r in snapshot["history"]):
            raise ValueError("Future result leaked into confirmation features")
        inputs = b.reconstruct_snapshot(snapshot, scoring=scoring)
        for market in eligible:
            league = average.predict(row["fixture"], market, cutoff)
            stat = b.statistical_projection(inputs["profiles"]["home"], inputs["profiles"]["away"],
                inputs["recent"]["home"], inputs["recent"]["away"], market, scoring=scoring)
            from .data import _finite
            if any(not _finite(v) or v <= 0 for v in (league["value"], stat["value"])):
                raise ValueError("Baseline unavailable; cannot silently shrink confirmation cohort")
            output.append({"fixture_id": row["fixture"]["fixture_id"], "snapshot_id": row["snapshot_id"],
                "feature_contract_id": row["feature_contract_id"], "market": market,
                "league_average": league["value"], "statistical": stat["value"],
                "evidence": {"as_of": row["as_of"], "availability": "assumed_final",
                    "baseline_version": b.VERSION, "scoring_sha256": core.digest(scoring),
                    "league_average": league, "statistical": stat, "profile_audit": inputs["evidence"],
                    "inputs_sha256": core.digest(inputs), "history_sha256": core.digest(snapshot["history"]),
                    "omitted_context": list(b.OMITTED_CONTEXT), "ml_weight": 0.0}})
        if number % 250 == 0:
            print(f"Confirmation features replayed {number}/{len(rows)}", flush=True)
    return output


def labels_by_fixture(body, rows):
    expected = {r["fixture"]["fixture_id"]: r for r in rows}
    result = {}
    for line in body.splitlines():
        if not line.strip():
            continue
        label = _decode(line)
        fid, labels, teams = label.get("fixture_id"), label.get("labels", {}), label.get("team_labels", {})
        if type(fid) is not int or fid not in expected or fid in result:
            raise ValueError("Confirmation label membership mismatch")
        if set(labels) != {*MARKETS, "cards"} or set(teams) != {"home", "away"}:
            raise ValueError("Invalid confirmation target schema")
        if any(set(teams[s]) != {*MARKETS, "cards"} for s in teams):
            raise ValueError("Invalid confirmation team target schema")
        for market in (*MARKETS, "cards"):
            parts = [teams[s][market] for s in ("home", "away")]
            if any(v is not None and not _count(v) for v in [labels[market], *parts]):
                raise ValueError("Invalid confirmation count")
            if labels[market] != (sum(parts) if all(v is not None for v in parts) else None):
                raise ValueError("Confirmation team targets disagree")
            if market == "cards" and any(v is not None for v in [labels[market], *parts]):
                raise ValueError("Card target remains unqualified")
            if expected[fid]["market_eligibility"][market]["eligible"] and labels[market] is None:
                raise ValueError("Eligible confirmation target missing")
        result[fid] = labels
    if set(result) != set(expected):
        raise ValueError("Confirmation labels missing fixtures")
    return result


def coverage(rows):
    result = {}
    for market in MARKETS:
        result[market] = {}
        for league in COMPETITIONS:
            selected = [r for r in rows if r["fixture"]["competition"] == league]
            result[market][league] = {"fixtures": len(selected),
                "eligible": sum(r["market_eligibility"][market]["eligible"] for r in selected),
                "primary_exclusions": dict(Counter(r["market_eligibility"][market]["primary_reason"]
                    for r in selected if not r["market_eligibility"][market]["eligible"]))}
    return result
