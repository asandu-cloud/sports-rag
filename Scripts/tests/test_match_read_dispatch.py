"""Tests for the fixture-first Match Read generation dispatcher."""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT / "Scripts" / "rag_ingest", ROOT / "Scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from rendering import match_read_dispatch  # noqa: E402


def _event(event_id: str, *, kickoff: str = "2026-09-06T15:00:00Z") -> dict:
    return {
        "id": event_id,
        "home_team": f"Home {event_id}",
        "away_team": f"Away {event_id}",
        "commence_time": kickoff,
        "bookmakers": [],
    }


def _draft(event_id: str, *, status: str = "recommended"):
    return SimpleNamespace(
        status=status,
        canonical_results=(),
        stage="pre_match",
        event_id=event_id,
    )


def test_dispatch_fetches_and_enriches_once_then_evaluates_every_market_once_per_fixture():
    first = _event("fixture-1", kickoff="2026-09-06T17:00:00Z")
    second = _event("fixture-2", kickoff="2026-09-06T12:00:00Z")
    evaluation_calls = []
    compilation_calls = []

    def evaluate(event, league, *, markets, fixture_date, context):
        evaluation_calls.append((event["id"], league, tuple(markets), fixture_date, context))
        return [{"event_id": event["id"], "market": market} for market in markets]

    def compile_read(results, *, stage):
        compilation_calls.append((tuple(results), stage))
        return _draft(results[0]["event_id"])

    with mock.patch.object(
        match_read_dispatch, "fetch_events", return_value=([first, second], ["provider note"])
    ) as fetch, mock.patch.object(
        match_read_dispatch, "filter_events_by_exact_date", return_value=[first, second]
    ) as date_filter, mock.patch.object(
        match_read_dispatch, "enrich_events_for_groups", return_value=([first, second], ["odds refreshed"])
    ) as enrich:
        run = match_read_dispatch.generate_match_reads_sync(
            "EPL",
            date(2026, 9, 6),
            evaluator=evaluate,
            compiler=compile_read,
        )

    fetch.assert_called_once_with("EPL", target_date=date(2026, 9, 6))
    date_filter.assert_called_once_with([first, second], date(2026, 9, 6))
    enrich.assert_called_once_with(
        [first, second], "EPL", set(match_read_dispatch.CANONICAL_ODDS_GROUPS),
    )
    assert [call[0] for call in evaluation_calls] == ["fixture-2", "fixture-1"]
    assert all(call[2] == tuple(match_read_dispatch.SUPPORTED_MARKETS) for call in evaluation_calls)
    assert all(call[3] == "2026-09-06" for call in evaluation_calls)
    assert all(call[4]["delivery_surface"] == "match_read_shadow" for call in evaluation_calls)
    assert len(compilation_calls) == 2
    assert [row.event_id for row in run.fixtures] == ["fixture-2", "fixture-1"]
    assert len(run.drafts) == 2
    assert run.records == ()
    assert run.notes == ("provider note", "odds refreshed")


def test_dispatch_persists_only_when_explicitly_requested_and_keeps_fixture_draft_on_failure():
    event = _event("fixture-1")
    draft = _draft("fixture-1")
    persisted = {"id": "match-read-1", "created": True}
    persist_calls = []

    def fetcher(_league, *, target_date):
        assert target_date == date(2026, 9, 6)
        return [event], []

    def evaluate(*_args, **_kwargs):
        return [{"event_id": "fixture-1", "market": "goals"}]

    def compiler(*_args, **_kwargs):
        return draft

    def persister(saved_draft, *, service):
        persist_calls.append((saved_draft, service))
        return persisted

    shadow = match_read_dispatch.generate_match_reads_sync(
        "EPL",
        date(2026, 9, 6),
        fetcher=fetcher,
        date_filter=lambda events, _target: events,
        enricher=lambda events, _league, _groups: (events, []),
        evaluator=evaluate,
        compiler=compiler,
        persister=persister,
    )
    assert persist_calls == []
    assert shadow.fixtures[0].record is None
    assert shadow.fixtures[0].draft is draft

    service = object()
    stored = match_read_dispatch.generate_match_reads_sync(
        "EPL",
        date(2026, 9, 6),
        persist=True,
        service=service,
        fetcher=fetcher,
        date_filter=lambda events, _target: events,
        enricher=lambda events, _league, _groups: (events, []),
        evaluator=evaluate,
        compiler=compiler,
        persister=persister,
    )
    assert persist_calls == [(draft, service)]
    assert stored.fixtures[0].record == persisted
    assert stored.records == (persisted,)


def test_dispatch_records_a_fixture_local_error_without_losing_other_fixture_drafts():
    broken = _event("broken", kickoff="2026-09-06T12:00:00Z")
    healthy = _event("healthy", kickoff="2026-09-06T17:00:00Z")

    def evaluate(event, *_args, **_kwargs):
        if event["id"] == "broken":
            raise RuntimeError("malformed odds")
        return [{"event_id": "healthy", "market": "goals"}]

    run = match_read_dispatch.generate_match_reads_sync(
        "EPL",
        date(2026, 9, 6),
        fetcher=lambda _league, *, target_date: ([broken, healthy], []),
        date_filter=lambda events, _target: events,
        enricher=lambda events, _league, _groups: (events, []),
        evaluator=evaluate,
        compiler=lambda results, *, stage: _draft(results[0]["event_id"]),
    )

    assert run.fixtures[0].event_id == "broken"
    assert run.fixtures[0].draft is None
    assert "canonical fixture evaluation failed" in run.fixtures[0].error
    assert run.fixtures[1].event_id == "healthy"
    assert run.fixtures[1].status == "recommended"
    assert len(run.drafts) == 1
    assert any("malformed odds" in note for note in run.notes)


def test_confirmed_lineups_stage_requires_verified_lineup_input_before_generation():
    fetcher = mock.Mock()

    run = match_read_dispatch.generate_match_reads_sync(
        "EPL",
        date(2026, 9, 6),
        stage="confirmed_lineups",
        fetcher=fetcher,
    )

    fetcher.assert_not_called()
    assert run.fixtures == ()
    assert "requires a verified lineup provider" in run.notes[0]


def test_confirmed_lineups_stage_passes_only_a_verified_context_to_evaluation():
    event = _event("fixture-1")
    context = SimpleNamespace(source="lineups", is_available=True)
    captured = {}

    def evaluate(_event, _league, **kwargs):
        captured.update(kwargs)
        return [{"event_id": "fixture-1", "market": "goals"}]

    run = match_read_dispatch.generate_match_reads_sync(
        "EPL",
        date(2026, 9, 6),
        stage="confirmed_lineups",
        fetcher=lambda _league, *, target_date: ([event], []),
        date_filter=lambda events, _target: events,
        enricher=lambda events, _league, _groups: (events, []),
        evaluator=evaluate,
        compiler=lambda results, *, stage: _draft(results[0]["event_id"]),
        lineup_provider=lambda _event, _league, _date: context,
    )

    assert run.fixtures[0].status == "recommended"
    assert captured["lineup_ctx"] is context
    assert captured["context"]["match_read_stage"] == "confirmed_lineups"
