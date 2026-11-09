# Pick Evidence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** put the approved match-card evidence (How Spix reads it, verdicts, details, past matches at similar odds)
on the website's cards behind `SPIX_PICK_EVIDENCE`, with no pick changing.

**Architecture:** a pure module computes groups, outcomes, verdicts and wording from a card's
`recommendations.v1` block and a list of past matches; a service loads that history (football-data file plus this
season from the platform database, kept for an hour) and attaches the result to the card in the website router.

**Tech Stack:** Python 3.11, sqlite3, SQLAlchemy sessions (`data_platform.db.session_scope`), pytest.

**Spec:** `docs/superpowers/specs/2026-11-09-pick-evidence-design.md`

## Global Constraints

- Switch `SPIX_PICK_EVIDENCE`, on only when the value is `on`; pinned off in `Scripts/tests/conftest.py`.
- Team group 12 matches (minimum 8), league group 40 (minimum 20), at least two groups per pick.
- Backs at 60% or more, against under 50%; lead at most 14 words, detail at most 25.
- Safer-line floor is `recommendations.MIN_PRICE` (1.30).
- No engine, worker or stored-record change; a card must never fail because of evidence.
- Never log exception text, only its type name.

## Review Focus

- A promoted team with few priced matches: its group is dropped, the pick keeps two groups or shows none.
- A handicap with a negative line: "won by 2 or more", never a positive-line phrase.
- A European cup card: gets `story`, never `evidence`.
- A card whose moneyline has no typical price: no `evidence`, card unchanged otherwise.
- The history load fails (database busy): card served unchanged, retried on the next request.

---

### Task 1: The pure module `Scripts/data_platform/pick_evidence.py`

**Files:** Create `Scripts/data_platform/pick_evidence.py`; Test `Scripts/tests/data_platform/test_pick_evidence.py`.

**Produces:**
- `team_group(history, team_id, price, before, *, n=12) -> list[dict]` (rows with `venue`, `team_price`,
  `goals_for`, `goals_against`), `league_group(history, league, home_price, away_price, before, *, n=40)`.
- `won(pick, match, *, view) -> bool` with `view` one of `"home"`, `"away"`, `"league"`.
- `means(pick, *, view, home, away) -> str`, `verdict(rates: list[float]) -> str`,
  `lead(home, away, chances, goals) -> str`, `detail(pick, groups, options, home, away) -> str`.
- `build(block, fixture, *, history, team_ids, competition, figures) -> dict` returning
  `{"story": ..., "evidence": ... or None, "picks": {index: evidence}}`.

- [ ] Tests first: group selection by log distance and recency ties; `won` for every market in the spec's table
  including a −1.5 handicap from each view; every verdict row; lead thresholds; detail sentences and the 25-word
  cap; `build` drops short groups and needs two.
- [ ] Run, watch fail; implement; run, pass; commit with the spec, plan and tracker D82.

### Task 2: The service `Scripts/data_platform/services/pick_evidence_service.py`

**Consumes:** Task 1's `build`. **Produces:** `enabled()`, `History(football_data_path, platform_loader, keep)`
with `rows(now)` and `teams(event_id)`, `PickEvidence(history).attach(card, canonical_results, now)`.

- [ ] Tests first with a temporary football-data file and a stub platform loader: rows merge without double
  counting, prices from the snapshot medians, the hour's cache, incremental price reads, `attach` adds only
  `story`/`evidence` keys, a European cup card gets only `story`, a failing loader leaves the card unchanged.
- [ ] Implement the platform reader with `session_scope` and `retry_database_busy`, snapshots read 20 at a time.
- [ ] Commit.

### Task 3: Router wiring

**Files:** `Scripts/web_app/routers/match_reads.py` (`_card_from_read`), `Scripts/tests/conftest.py`, test
`Scripts/tests/data_platform/test_pick_evidence_api.py`.

- [ ] Test first: with the switch on, a board card carries `story` and the picks' `evidence`; every field present
  with the switch off is identical with it on; with the switch off nothing is added.
- [ ] Implement; full suite; pinned golden check; real-data check on 8200 with the switch on the command line.
- [ ] Commit with the tracker's checks; message the orchestrator.
