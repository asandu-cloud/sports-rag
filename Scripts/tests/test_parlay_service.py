from __future__ import annotations

import tempfile
import unittest
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from Scripts.rag_ingest.core import parlay_service as service
from Scripts.rag_ingest.core.parlay_models import (
    ParlayBuildRequest,
    ParlayBuildResult,
    ParlayLegResult,
    ParlaySessionRecord,
)


def _candidate(event_id: str, fixture: str, market_key: str, outcome: str, odds: float, point: float | None, league: str = "EPL"):
    home, away = fixture.split(" vs ")
    return service.rag.CandidateLeg(
        event_id=event_id,
        fixture=fixture,
        home_team=home,
        away_team=away,
        market_key=market_key,
        outcome=outcome,
        odds=odds,
        point=point,
        bookmaker="Bet365",
        league=league,
        event={"id": event_id, "_league": league, "home_team": home, "away_team": away},
    )


class ParlayServiceTests(unittest.TestCase):
    def test_build_parlay_request_cross_league(self):
        request = service.build_parlay_request(
            league_value="cross-league",
            target_date=date(2026, 4, 2),
            legs=4,
            odds=4.5,
            market="mixed",
            match=None,
            source_command="parlay",
            target_mode="around",
        )
        self.assertEqual(request.display_league, "Cross-League")
        self.assertEqual(request.leagues, service.TOP5_LEAGUES)
        self.assertEqual(request.default_league, "EPL")
        self.assertTrue(request.require_unique_events)
        self.assertFalse(request.same_game_only)
        self.assertEqual(request.target_mode, "around")

    def test_build_parlay_request_supports_ranges_and_league_override(self):
        request = service.build_parlay_request(
            league_value="cross-league",
            leagues_override=["EPL", "UCL"],
            display_league="Cross-League",
            target_date=date(2026, 4, 2),
            legs=2,
            odds=None,
            target_min=1.4,
            target_max=1.8,
            market="mixed",
            source_command="trial",
            min_model_prob=0.6,
            allowed_confidences=["high"],
            allowed_groups=["moneyline", "btts"],
            soft_prefer_groups=["moneyline", "btts"],
        )
        self.assertEqual(request.leagues, ["EPL", "UCL"])
        self.assertEqual(request.target_min, 1.4)
        self.assertEqual(request.target_max, 1.8)
        self.assertEqual(request.allowed_groups, ["moneyline", "btts"])
        self.assertEqual(request.allowed_confidences, ["high"])
        self.assertEqual(request.min_model_prob, 0.6)

    def test_session_store_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = service.ParlaySessionStore(Path(tmpdir) / "predictions.db")
            request = service.build_parlay_request(
                league_value="EPL",
                target_date=date(2026, 4, 3),
                legs=3,
                odds=2.8,
                market="totals",
                match=None,
                source_command="build",
            )
            result = ParlayBuildResult(
                request=request,
                selected_legs=[
                    ParlayLegResult(
                        event_id="1",
                        league="EPL",
                        fixture="Arsenal vs Chelsea",
                        home_team="Arsenal",
                        away_team="Chelsea",
                        market_key="totals",
                        market_group="totals",
                        outcome="Over",
                        point=2.5,
                        pick_display="Over 2.5 (totals/totals)",
                        odds=1.75,
                        bookmaker="Bet365",
                        quality_score=0.9,
                        confidence="high",
                        model_prob=0.66,
                        implied_prob=0.57,
                        value_edge=0.09,
                        projected_total=3.1,
                        reasons=["Projected goals: 3.1."],
                    )
                ],
                combined_odds=1.75,
                deterministic_summary="Deterministic summary.",
            )
            session_id = store.create_session(request, result)
            store.update_message_context(session_id, message_id=555, channel_id=777, thread_id=999)
            loaded = store.get_session(session_id)
            assert loaded is not None
            self.assertIsInstance(loaded, service.ParlaySessionRecord)
            self.assertEqual(loaded.id, session_id)
            self.assertEqual(loaded.message_id, 555)
            self.assertEqual(loaded.thread_id, 999)
            self.assertEqual(loaded.result.selected_legs[0].fixture, "Arsenal vs Chelsea")
            self.assertEqual(loaded.result.session_id, session_id)

    def test_followup_request_derivation(self):
        request = service.build_parlay_request(
            league_value="EPL",
            target_date=date(2026, 4, 4),
            legs=3,
            odds=3.5,
            market="mixed",
            match=None,
            source_command="parlay",
        )
        legs = [
            ParlayLegResult(
                event_id="1",
                league="EPL",
                fixture="Arsenal vs Chelsea",
                home_team="Arsenal",
                away_team="Chelsea",
                market_key="totals",
                market_group="totals",
                outcome="Over",
                point=2.5,
                pick_display="Over 2.5 (totals/totals)",
                odds=1.72,
                bookmaker="Bet365",
                quality_score=0.8,
                confidence="high",
                model_prob=0.64,
                implied_prob=0.58,
                value_edge=0.06,
                projected_total=3.0,
            ),
            ParlayLegResult(
                event_id="2",
                league="EPL",
                fixture="Liverpool vs Brighton",
                home_team="Liverpool",
                away_team="Brighton",
                market_key="corners",
                market_group="corners",
                outcome="Over",
                point=9.5,
                pick_display="Over 9.5 (corners/corners)",
                odds=1.85,
                bookmaker="Bet365",
                quality_score=0.4,
                confidence="medium",
                model_prob=0.58,
                implied_prob=0.54,
                value_edge=0.04,
                projected_total=10.4,
            ),
            ParlayLegResult(
                event_id="3",
                league="EPL",
                fixture="Spurs vs Everton",
                home_team="Spurs",
                away_team="Everton",
                market_key="cards",
                market_group="cards",
                outcome="Over",
                point=3.5,
                pick_display="Over 3.5 (cards/cards)",
                odds=1.66,
                bookmaker="Bet365",
                quality_score=0.7,
                confidence="high",
                model_prob=0.62,
                implied_prob=0.60,
                value_edge=0.02,
                projected_total=4.1,
            ),
        ]
        result = ParlayBuildResult(request=request, selected_legs=legs, combined_odds=5.28)
        record = ParlaySessionRecord(
            id=44,
            parent_session_id=None,
            discord_user_id=10,
            guild_id=20,
            channel_id=30,
            thread_id=40,
            message_id=50,
            source_command="parlay",
            action_type="initial",
            created_at="2026-04-01 12:00:00",
            request=request,
            result=result,
        )

        add_req = service.derive_followup_request(record, "add_leg")
        self.assertEqual(add_req.legs_requested, 4)
        self.assertEqual(len(add_req.locked_legs), 3)
        self.assertTrue(add_req.prefer_new_fixture)

        swap_req = service.derive_followup_request(record, "swap_weakest")
        self.assertEqual(len(swap_req.locked_legs), 2)
        self.assertEqual(len(swap_req.excluded_legs), 1)
        self.assertEqual(swap_req.excluded_legs[0].event_id, "2")

        safer_req = service.derive_followup_request(record, "safer")
        self.assertEqual(set(safer_req.allowed_event_ids), {"1", "2", "3"})
        self.assertLess(safer_req.target_odds or 0.0, result.combined_odds)

        riskier_req = service.derive_followup_request(record, "riskier")
        self.assertEqual(set(riskier_req.allowed_event_ids), {"1", "2", "3"})
        self.assertGreater(riskier_req.target_odds or 0.0, result.combined_odds)

    def test_constraint_spec_uses_ranges_and_group_overrides(self):
        request = service.build_parlay_request(
            league_value="cross-league",
            leagues_override=["EPL", "UCL"],
            display_league="Cross-League",
            target_date=date(2026, 4, 5),
            legs=4,
            odds=None,
            target_min=4.0,
            target_max=6.0,
            market="mixed",
            source_command="auto_push",
            allowed_groups=["totals", "corners", "cards"],
            soft_prefer_groups=["totals", "corners", "cards"],
            required_group_counts={"totals": 1, "corners": 1, "cards": 1},
        )
        constraint = service._build_constraint_spec(request)
        self.assertEqual(constraint.target_min, 4.0)
        self.assertEqual(constraint.target_max, 6.0)
        self.assertEqual(constraint.leagues, ["EPL", "UCL"])
        self.assertEqual(constraint.required_group_counts["totals"], 1)
        self.assertIn("totals", constraint.soft_prefer_groups)
        self.assertIn("sot", constraint.hard_exclude_groups)

    def test_llm_summary_rejects_unsafe_output(self):
        request = service.build_parlay_request(
            league_value="EPL",
            target_date=date(2026, 4, 5),
            legs=2,
            odds=2.4,
            market="totals",
            match=None,
            source_command="parlay",
        )
        result = ParlayBuildResult(
            request=request,
            selected_legs=[
                ParlayLegResult(
                    event_id="1",
                    league="EPL",
                    fixture="Arsenal vs Chelsea",
                    home_team="Arsenal",
                    away_team="Chelsea",
                    market_key="totals",
                    market_group="totals",
                    outcome="Over",
                    point=2.5,
                    pick_display="Over 2.5 (totals/totals)",
                    odds=1.72,
                    bookmaker="Bet365",
                    quality_score=0.8,
                    confidence="high",
                    model_prob=0.64,
                    implied_prob=0.58,
                    value_edge=0.06,
                    projected_total=3.0,
                    reasons=["Projected combined goals: 3.0."],
                )
            ],
            combined_odds=1.72,
        )
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **kwargs: SimpleNamespace(
                        choices=[SimpleNamespace(message=SimpleNamespace(content="Take Over 2.5 instead and swap the line."))]
                    )
                )
            )
        )
        with mock.patch.object(service.rag, "client", fake_client):
            summary, note = service._generate_llm_summary(result)
        self.assertIsNone(summary)
        self.assertIn("rejected", note or "")

    def test_build_parlay_uses_structured_selection_path(self):
        request = service.build_parlay_request(
            league_value="EPL",
            target_date=date(2026, 4, 6),
            legs=2,
            odds=3.0,
            market="mixed",
            match=None,
            source_command="parlay",
        )
        events = [{"id": "1", "_league": "EPL", "home_team": "Arsenal", "away_team": "Chelsea"}]
        selected = [
            _candidate("1", "Arsenal vs Chelsea", "totals", "Over", 1.72, 2.5),
            _candidate("2", "Liverpool vs Brighton", "corners", "Over", 1.80, 9.5),
        ]
        with mock.patch.object(service, "_fetch_events_for_request", return_value=(events, [])), \
             mock.patch.object(service, "_filter_events_for_request", return_value=events), \
             mock.patch.object(service, "_enrich_events", return_value=(events, [])), \
             mock.patch.object(service.rag, "build_candidates", return_value=selected), \
             mock.patch.object(service.rag, "check_feasibility", return_value=(True, [])), \
             mock.patch.object(service.rag, "select_parlay", return_value=(selected, ["mock selection"])), \
             mock.patch.object(service.rag, "combo_valid", return_value=True), \
             mock.patch.object(service.rag, "combined_odds", return_value=3.10), \
             mock.patch.object(service.rag, "kb_leg_quality", side_effect=[0.8, 0.6]), \
             mock.patch.object(service, "_generate_llm_summary", return_value=(None, "LLM unavailable")), \
             mock.patch.object(service, "leg_standalone_confidence", side_effect=[
                 {"confidence": "high", "model_prob": 0.64, "projected": 3.0, "side_agrees": True, "warning": None},
                 {"confidence": "medium", "model_prob": 0.58, "projected": 10.4, "side_agrees": True, "warning": None},
             ]), \
             mock.patch.object(service, "leg_evidence", side_effect=[
                 ["Projected combined goals: 3.0."],
                 ["Projected combined corners: 10.4."],
             ]):
            result = service.build_parlay(request)

        self.assertEqual(len(result.selected_legs), 2)
        self.assertEqual(result.selected_legs[0].fixture, "Arsenal vs Chelsea")
        self.assertEqual(result.notes[-1], "LLM unavailable")
        self.assertIn("Combined odds landed", result.deterministic_summary)


if __name__ == "__main__":
    unittest.main()
