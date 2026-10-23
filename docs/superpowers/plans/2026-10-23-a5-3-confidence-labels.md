# A5.3 Confidence Labels Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the edge-based high/medium/low label with a high/low label built from evidence: market firmness now, plus Spix's agreement in markets that have earned it. The new label sits behind a switch that stays off, and two research checks set its cut-offs.

**Architecture:**
- One dict-based module, `data_platform/confidence_labels.py`, reads a finished market result the way price confirmation does. It works out bookmaker spread, market value, Spix's value and price age, and applies one rule.
- The engine labels each result when it is built, with price age unknown. The Match Read dispatcher labels again after price confirmation, when the age is known.
- Two studies validate the rule on recorded data:
  - `ops/confidence_study.py`: spread cut-off and the per-market Spix check.
  - `ops/confidence_age_study.py`: price age, on cached OddsPapi histories.

**Tech Stack:** Python 3.13, stdlib `statistics`, numpy (fixture bootstrap via `disagreement.fixture_bootstrap`), pytest.

**Spec:** `docs/superpowers/specs/2026-10-23-a5-3-confidence-labels-design.md` (tracker D44).

## Global Constraints

- **Switch** `SPIX_CONFIDENCE_LABELS`:
  - Values: `edge` (default, today's `confidence_from_edge`) or `evidence`.
  - Registered in `Scripts/rag_ingest/core/engine_switches.py:SWITCHES`.
- **Two labels only:** `"high"` and `"low"`. "Medium" is not produced under `evidence`.
- **High** needs all of the following. Anything else is low:
  - At least `min_books` (4) bookmakers price the line two-sided.
  - The bookmaker spread is at or under `max_spread`.
  - The price age is known and at or under `fresh_minutes` (30).
  - The market's own value at the published price is at least 2%.
  - In markets listed in `SPIX_VOTE_MARKETS`, Spix's own value at that price is also at least 2%.
- **Unknown or missing evidence:**
  - An unknown price age is low.
  - `max_spread=None` (no cut-off has passed its check) makes every label low.
- **Minimum value** is 2%, equal to `line_selection.MARKET_VALUE_MIN_EV`.
- **`SPIX_VOTE_MARKETS`** starts as `frozenset()`. A market is added only by an owner-approved tracker decision, never by a study.
- **A label never changes a decision.** Status, quote, odds, model probability and reason stay byte-identical. Only `decision.confidence` and `context["confidence_evidence"]` change.
- **With the switch off nothing changes:** `golden_snapshots check` stays "no changes across 54 fixtures".
- **Each label's share:** each label must hold at least 20% of picks for a spread cut-off to be eligible.
- **Reads of `Index/platform.db`:**
  - Only inside `refresh_coordination.data_access()`, in short statements (`calibration_data.latest_read_rows`).
  - Never while `refresh-season` runs.
- **OddsPapi in the age study:**
  - Cached answers only: `client.budget = client._usage()["billable"]`, so no billable request is made.
  - The key is read from `.env` with `load_dotenv`. Never print it, and redact `apiKey=` from any error text.
- **Commits:**
  - Add files by name. Never add `Index/`, `Output/`, `Research/` or `.impeccable/`.
  - Each message ends with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Commit and push every task.
- **Tracker:** `docs/rework-tracker.md` is updated in the same commit as any finding or plan change.

## Review Focus

1. **The quoting bookmaker is missing from the two-sided books.** For example, its price is one-sided. The label must still be computed from the other books without crashing. Pinned in Task 2: `test_a_quote_whose_own_book_is_one_sided_is_still_labelled`.
2. **Results with no decision audit or input options.** This covers old stored reads and unavailable markets. `line_evidence` returns `None`, and the label is low. Pinned in Task 2: `test_a_result_without_audited_prices_is_low`.
3. **Asian quarter lines** (2.25, 2.75) on totals and the handicap. Margin removal per book must not fail, and a label results. Pinned in Task 2: `test_a_quarter_line_is_labelled`.
4. **Timestamps written with `Z` against `+00:00`** in `price_confirmation` records. The age must parse both. Pinned in Task 2: `test_price_age_is_known_only_from_a_confirmed_price`.
5. **A repriced pick** (its odds were moved to the bookmaker's shorter price). The label must use the published, repriced odds. If the market's value falls under 2% there, it is low. Pinned in Task 3: `test_the_dispatcher_labels_after_confirmation_with_the_published_price`.

---

### Task 1: Bookmaker spread on the consensus, carried by the A5 rows

**Files:**
- Modify: `Scripts/data_platform/reference_price.py` (add `consensus_spread` after `devig_consensus`)
- Modify: `Scripts/data_platform/disagreement.py` (`read_lines`: add `"book_spread"` to each row)
- Test: `Scripts/tests/data_platform/test_reference_price.py`, `Scripts/tests/data_platform/test_disagreement.py`

**Interfaces:**
- Produces:
  - `reference_price.consensus_spread(consensus: Optional[dict]) -> Optional[float]`: the interquartile range of `consensus["books"][i]["fair"]`, using `statistics.quantiles(fairs, n=4, method="inclusive")`, q3 − q1. Returns `None` when `consensus` is `None` or has fewer than 2 books.
  - Rows from `disagreement.read_lines` gain `"book_spread"`: the spread of that row's own consensus. For the other side of a two-way line, the books' fair probabilities are 1 − f, so the spread is identical.

- [ ] **Step 1: Write the failing tests**

```python
# test_reference_price.py
def test_the_spread_is_the_interquartile_range_of_the_books_fair_probabilities():
    consensus = {"books": [{"fair": f} for f in (0.50, 0.52, 0.53, 0.60)]}
    assert consensus_spread(consensus) == pytest.approx(0.0325)  # q1 0.515, q3 0.5475
    assert consensus_spread({"books": [{"fair": 0.5}]}) is None and consensus_spread(None) is None

# test_disagreement.py (TOTALS: books A, B, C on over/under 2.5)
def test_both_sides_of_a_line_carry_the_same_book_spread():
    rows = dis.read_lines(read([result("goals", TOTALS, [candidate("totals", "over", 2.5, 0.6)])]), all_sides=True)
    over, under = (next(r for r in rows if r["side"] == s) for s in ("over", "under"))
    fairs = [power([o, u])[0] for o, u in ((1.90, 1.95), (2.10, 1.75), (2.00, 1.85))]
    q = statistics.quantiles(fairs, n=4, method="inclusive")
    assert over["book_spread"] == pytest.approx(q[2] - q[0]) == pytest.approx(under["book_spread"])
```

- [ ] **Step 2: Run them to verify they fail**
  - Run: `.venv/bin/python -m pytest -q Scripts/tests/data_platform/test_reference_price.py Scripts/tests/data_platform/test_disagreement.py`
  - Expected: FAIL, with `consensus_spread` undefined and `KeyError: 'book_spread'`.

- [ ] **Step 3: Implement `consensus_spread`, and set `"book_spread": consensus_spread(consensus)` on every row `read_lines` appends**

- [ ] **Step 4: Run the tests and the study tests to verify they pass**
  - Run: `.venv/bin/python -m pytest -q Scripts/tests/data_platform -k "reference_price or disagreement or selection or m1_gate or calibration"`
  - Expected: all pass.

- [ ] **Step 5: Commit** `reference_price.py`, `disagreement.py` and the two test files: "Bookmaker spread (M3) on the consensus and on the A5 rows".

### Task 2: The label rule and its switch

**Files:**
- Create: `Scripts/data_platform/confidence_labels.py`
- Modify:
  - `Scripts/rag_ingest/core/engine_switches.py`: add `"SPIX_CONFIDENCE_LABELS": ("edge", frozenset({"edge", "evidence"}))  # A5.3 (D44)`.
  - `Scripts/data_platform/scorecard.py`: `STAGE_SWITCHES` gains `"SPIX_CONFIDENCE_LABELS": "edge"`.
- Test:
  - Create `Scripts/tests/data_platform/test_confidence_labels.py`.
  - Modify `Scripts/tests/test_engine_switches.py` (`test_defaults_are_todays_behaviour` expects `"SPIX_CONFIDENCE_LABELS": "edge"`) and `Scripts/tests/data_platform/test_scorecard.py`.

**Interfaces:**
- Consumes:
  - `reference_price.consensus_spread`, `devig_consensus`, `audit_books` and `norm`.
  - `disagreement.MARKET_KEYS`.
  - `settlement.utc_datetime`.
- Produces (all in `confidence_labels.py`):
  - `VERSION = "confidence-labels.v1"`.
  - `@dataclass(frozen=True) class Cutoffs: max_spread: Optional[float]; min_books: int = 4; fresh_minutes: float = 30.0`.
  - `CUTOFFS = Cutoffs(max_spread=None)`, set in Task 6 from the study.
  - `SPIX_VOTE_MARKETS: frozenset = frozenset()`.
  - `MIN_VALUE = 0.02`.
  - `line_evidence(result: Mapping) -> Optional[dict]`:
    - Returns keys `market`, `book_count`, `book_spread`, `market_probability`, `market_value`, `spix_probability`, `spix_value` and `method`. Returns `None` unless the result is recommended with a quote and a two-sided consensus exists.
    - `method` is the result's `context.engine_switches.SPIX_DEVIG_METHOD`, or `"proportional"`.
    - Side and point are `norm(quote.side)` and `quote.line`, with `spreads=` set for the handicap.
    - `market_value` is `market_probability * odds - 1`, the same as the A5.2 study.
    - Spix's probability is found in this order:
      1. For 1X2, `context.market_anchor.spix[side]`.
      2. The selected decision-audit candidate matching the quote's side/team, point and bookmaker: its option's `_spix_prob`, else its `_model_prob`.
      3. `decision.model_probability`.
  - `price_age_minutes(result: Mapping) -> Optional[float]`: minutes from `context.price_confirmation.bookmaker_seen_at` to its `checked_at`, only when the status is `confirmed` or `repriced`.
  - `label(evidence, age_minutes, *, cutoffs=CUTOFFS, spix_vote=SPIX_VOTE_MARKETS, min_value=MIN_VALUE) -> str`: implements the Global Constraints rule.
  - `apply_label(result)` and `apply_labels(results) -> list`:
    - They return the same type they are given: a `MarketResult` (via `to_dict`/`from_dict`) or a dict.
    - A result is unchanged unless it is recommended and its recorded `context.engine_switches.SPIX_CONFIDENCE_LABELS == "evidence"`.
    - Otherwise they set `decision.confidence`, and set `context["confidence_evidence"]` to `{**evidence, "price_age_minutes", "label", "version", "cutoffs": asdict(cutoffs)}`.

- [ ] **Step 1: Write the failing tests** in `test_confidence_labels.py`. Build results as dicts with `input_options` for 4–5 books, and `EVIDENCE = dict(market="goals", book_count=5, book_spread=0.01, market_value=0.03, spix_value=-0.04)`.
  - `test_a_firm_fresh_market_with_value_is_high`: `label(EVIDENCE, 12, cutoffs=Cutoffs(0.02)) == "high"`.
  - `test_scattered_few_books_old_or_unknown_price_or_no_cutoff_is_low`: each of the following gives `"low"`:
    - `book_spread=0.03`
    - `book_count=3`
    - age 45
    - age `None`
    - `Cutoffs(None)`
    - `market_value=0.01`
    - `evidence=None`
  - `test_spix_votes_only_in_markets_that_earned_it`, with `market="corners"` and `spix_value=-0.04`:
    - high with `spix_vote=frozenset()`;
    - low with `spix_vote=frozenset({"corners"})`;
    - high with the vote and `spix_value=0.03`.
  - `test_line_evidence_reads_the_books_and_spix_from_the_result`: four books on totals 2.5 and a quote of over 2.5 @ 2.05 (Betano), with a selected candidate whose option has `_spix_prob=0.47`.
    - `book_count == 4`.
    - `book_spread == consensus_spread(...)`.
    - `market_value == approx(median_fair * 2.05 - 1)`.
    - `spix_value == approx(0.47 * 2.05 - 1)`.
  - `test_moneyline_spix_probability_comes_from_the_anchor_record`: `spix_probability == 0.29` from `market_anchor.spix.away`.
  - `test_a_quote_whose_own_book_is_one_sided_is_still_labelled`: the evidence is not `None`, and `book_count` counts the other books.
  - `test_a_result_without_audited_prices_is_low`: `line_evidence` is `None`, and `apply_label` gives `"low"` under `evidence`.
  - `test_a_quarter_line_is_labelled`: totals 2.25 over/under from 4 books, so the evidence is not `None`.
  - `test_price_age_is_known_only_from_a_confirmed_price`:
    - confirmed, with `checked_at="2026-10-24T12:30:00+00:00"` and `bookmaker_seen_at="2026-10-24T12:18:00Z"`, gives `12.0`;
    - `not_confirmed` gives `None`;
    - no record gives `None`.
  - `test_labels_change_nothing_with_the_switch_off_and_never_change_a_decision`:
    - With the switch recorded `edge`, the result equals its input.
    - With `evidence`, every decision key except `confidence` is equal, and `context` gains only `confidence_evidence`.
  - `test_min_value_matches_the_selection_threshold`: `MIN_VALUE == line_selection.MARKET_VALUE_MIN_EV`.
  - In `test_scorecard.py`: `engine_stage({"engine_switches": {"SPIX_CONFIDENCE_LABELS": "evidence"}}) == "SPIX_CONFIDENCE_LABELS=evidence"`.

- [ ] **Step 2: Run them to verify they fail**
  - Run: `.venv/bin/python -m pytest -q Scripts/tests/data_platform/test_confidence_labels.py Scripts/tests/test_engine_switches.py Scripts/tests/data_platform/test_scorecard.py`
  - Expected: FAIL, as the module, the switch and the stage entry don't exist yet.

- [ ] **Step 3: Implement `confidence_labels.py`, register the switch, and add the scorecard stage entry**
  - Copy `price_confirmation._as_dict`/`_rebuild`, or import them, to round-trip `MarketResult`.
  - Parse times with `settlement.utc_datetime`.

- [ ] **Step 4: Run the tests to verify they pass, then the full suite**
  - Run: `.venv/bin/python -m pytest -q`
  - Expected: all pass, since `Scripts/tests/conftest.py` pins every registered switch to its default.

- [ ] **Step 5: Commit** the new module, the two modified sources and the three test files: "A5.3 label rule behind SPIX_CONFIDENCE_LABELS (off)".

### Task 3: Labels in the engine, after confirmation, and in the golden preview

**Files:**
- Modify:
  - `Scripts/rag_ingest/core/market_service.py:338-364` (`_build_result`): return `apply_label(...)` of the built `MarketResult`. Import with the module's `data_platform` / `Scripts.data_platform` fallback.
  - `Scripts/rag_ingest/rendering/match_read_dispatch.py`: right after the price-confirmation block (around line 399–415), set `canonical_results = tuple(apply_labels(canonical_results))`. This runs whether or not a confirmer ran.
  - `Scripts/ops/golden_snapshots.py`:
    - `pick()` gains `"confidence": decision.get("confidence")`.
    - `summarize_preview` adds one line after the table: `labels on kept picks: <before>→<after> <count>, …`, or `labels unchanged`.
- Test:
  - `Scripts/tests/test_match_read_dispatch.py`
  - `Scripts/tests/test_golden_snapshots.py`
  - `Scripts/tests/data_platform/test_confidence_labels.py` (engine hook)

**Interfaces:**
- Consumes: `confidence_labels.apply_label`, `apply_labels`, `CUTOFFS` (Task 2).

- [ ] **Step 1: Write the failing tests**
  - `test_the_engine_labels_a_result_as_it_is_built` (monkeypatch `SPIX_CONFIDENCE_LABELS=evidence`):
    - Call `market_service._build_result` with a recommended decision and a context holding `engine_switches.snapshot()` and audited prices.
    - `result.decision.confidence == "low"`, because the age is unknown.
    - `"confidence_evidence" in result.context`.
  - `test_the_dispatcher_labels_after_confirmation_with_the_published_price`:
    - Monkeypatch `confidence_labels.CUTOFFS = Cutoffs(0.05)`.
    - The confirmer returns the evaluator's result with a fresh `price_confirmation`. The compiled result is `"high"`.
    - With `status="repriced"` and odds lowered so that `market_value < 0.02`, it is `"low"`.
  - `test_plain_results_pass_the_label_step_unchanged`: the existing `_confirmation_run` dicts compile unchanged. This keeps `test_stored_reads_have_their_prices_confirmed_between_evaluation_and_compilation` true.
  - `test_preview_reports_label_changes_on_kept_picks`: rows with a kept pick going low → high give `"labels on kept picks: low→high 1"`.

- [ ] **Step 2: Run them to verify they fail**
  - Run: `.venv/bin/python -m pytest -q Scripts/tests/test_match_read_dispatch.py Scripts/tests/test_golden_snapshots.py Scripts/tests/data_platform/test_confidence_labels.py`
  - Expected: the new tests FAIL.

- [ ] **Step 3: Implement the three hooks**

- [ ] **Step 4: Verify**
  - Run the full suite: `.venv/bin/python -m pytest -q`. Expected: all pass.
  - Run `.venv/bin/python -m Scripts.ops.golden_snapshots check`. Expected: `no changes across 54 fixtures`.
  - Run `SPIX_COUNT_DISPERSION=fixed SPIX_DEVIG_METHOD=power SPIX_CONFIDENCE_LABELS=evidence .venv/bin/python -m Scripts.ops.golden_snapshots preview --baseline SPIX_CONFIDENCE_LABELS=edge`. Expected:
    - Every market row shows Added 0, Removed 0, Moved 0 and Price only 0.
    - The label line shows kept picks moving to `low`, since the replay has no price confirmation.

- [ ] **Step 5: Commit** the three sources and three test files: "A5.3 labels applied when results are built and after price confirmation".

### Task 4: The firmness and Spix checks

**Files:**
- Create: `Scripts/data_platform/label_checks.py` (pure functions) and `Scripts/ops/confidence_study.py` (CLI).
- Test: `Scripts/tests/data_platform/test_label_checks.py`

**Interfaces:**
- Consumes:
  - `disagreement.build(db, backfill_dir=..., all_sides=True)` rows with `book_spread` (Task 1).
  - `selection_policies.pick` and `policies()`, using the policy named `"market"`.
  - `disagreement.fixture_bootstrap(rows, statistic, resamples=..., seed=...)`.
  - `confidence_labels.MIN_VALUE`.
- Produces (in `label_checks.py`):
  - `pick_rows(rows, *, min_value=MIN_VALUE) -> list[dict]`: the market-policy picks per stage, flattened to:
    - `fixture_id`, `stage`, `market`, `kickoff` and `odds` (`best_odds`);
    - `p` (the published probability), `y`, `w`, `book_count`, `book_spread`;
    - `spix_value` (`p_spix * odds - 1`);
    - `value_at_close` (`p_close * odds - 1`, or `None`).
  - `split_by_week(picks, *, holdout_days=7) -> tuple[list, list]`: the holdout is the picks whose kickoff falls in the last `holdout_days` before the latest kickoff.
  - `compare(high, low, *, resamples=1000) -> dict`, with keys:
    - `high` and `low`, each `{picks, value_at_close, stated, happened}`;
    - `value_diff` and `value_diff_interval95` (high − low mean value at the close, over picks with a close);
    - `calibration_diff` and `calibration_diff_interval95` (high's happened − stated, minus low's).
    - All intervals resample whole fixtures over the combined sample.
  - `choose_spread_cutoff(train, *, min_books=4, min_share=0.2) -> Optional[float]`:
    - Candidates are the 20th–80th percentiles of train spreads, in steps of 10, among picks with ≥ `min_books` books.
    - High is `book_count >= min_books and book_spread <= c`.
    - Keep candidates where each label holds ≥ `min_share` of picks, and pick the one with the largest train `value_diff`.
    - Returns `None` if no candidate qualifies.
  - `passes(comparison) -> bool`: `value_diff_interval95[0] > 0 and calibration_diff_interval95[1] >= 0`.
  - `spix_check(picks, *, min_value=MIN_VALUE, min_picks=30, resamples=1000) -> dict[str, dict]`: per market, "agrees" (`spix_value >= min_value`) is compared against "does not". Each entry is the `compare(...)` output plus `{"agree": n, "disagree": n, "passes": bool}`, and `passes` is false when either group has fewer than `min_picks`.
- `ops/confidence_study.py main(argv)`:
  - Flags: `--out`, `--db` (default `Index/platform.db`), `--backfill` (default `Index/closing_backfill`), `--resamples 1000`.
  - Reads rows inside `with data_access():`.
  - Writes `confidence.json` with:
    - `cutoff`, `train` and `holdout` comparisons, and `passes`;
    - the per-market high/low comparison at the chosen cut-off;
    - `spix` per market;
    - the share of picks in each label.
  - Prints a markdown summary.

- [ ] **Step 1: Write the failing tests** (synthetic picks with `fixture_id`s, two weeks of kickoffs):
  - `test_holdout_is_the_last_week`.
  - `test_the_cutoff_keeps_a_fifth_of_picks_in_each_label`: with spreads skewed so that 90% sit under the 20th-percentile candidate, the chosen cut-off still leaves ≥ 20% in each label.
  - `test_a_clearly_better_high_label_passes_and_an_equal_one_does_not`.
  - `test_the_spix_check_needs_enough_picks_on_both_sides`: 10 agreeing picks give `passes is False`, even if their value is higher.
  - `test_pick_rows_takes_the_market_policy_and_carries_the_evidence`.

- [ ] **Step 2: Run them to verify they fail**
  - Run: `.venv/bin/python -m pytest -q Scripts/tests/data_platform/test_label_checks.py`
  - Expected: FAIL, as the module doesn't exist yet.

- [ ] **Step 3: Implement `label_checks.py` and `ops/confidence_study.py`**

- [ ] **Step 4: Run the tests to verify they pass**, then the full suite. Expected: all pass.

- [ ] **Step 5: Commit** the two new modules and the test: "A5.3 firmness and Spix checks (study)".

### Task 5: The price-age check

**Files:**
- Create: `Scripts/ops/confidence_age_study.py`
- Test: `Scripts/tests/test_confidence_age_study.py`

**Interfaces:**
- Consumes:
  - `price_check.picks_by_policy(rows, ("market",))["market"]`, `price_check.fixture_ids` and `pinnacle_check.our_fixtures`.
  - `oddspapi.OddsPapiClient`, `histories`, `series`, `locate`, `market_index`, `BOOKS`, `EXCHANGES` and `utc`.
  - `disagreement.fixture_bootstrap`.
- Produces (in `confidence_age_study.py`):
  - `seen_age_minutes(moments: Sequence[datetime], at: datetime) -> Optional[float]`: minutes since the bookmaker's newest record on any line of the match at or before `at`.
  - `next_check_verdict(points, moments, at, kickoff, quoted, *, tolerance=0.005) -> str`: one of `"still there"`, `"gone"` or `"no check before kickoff"`, judged at the bookmaker's first record after `at`.
  - `still_there_comparison(fresh, older, *, resamples=1000) -> dict`, with keys:
    - `fresh` and `older`, each `{picks, still_there_share, value_at_close}`;
    - `diff` and `diff_interval95` (fresh − older share still there, resampling whole fixtures).
  - `main(argv)`, with flags `--out`, `--from 2026-10-08`, `--to <yesterday>` and `--fresh-minutes 30`:
    - For every market-value pick whose quoted price was on offer at the read (the bookmaker's own price ≥ quoted × (1 − 0.005), active), record its age, verdict, value at the close, `y`/`w` and fixture.
    - Compare fresh (≤ 30 min) against older with `still_there_comparison`.
    - Write `confidence_age.json` and print a summary.
  - The age rule is supported when fresh prices are still there more often than older ones, with the difference's interval above zero.

- [ ] **Step 1: Write the failing tests:**
  - `test_fresh_prices_still_there_more_often_show_a_positive_difference`.
  - `test_age_is_minutes_since_the_newest_record`.
  - `test_no_record_before_the_read_is_unknown`.
  - `test_the_verdict_reads_the_first_check_after_the_read`.
  - `test_a_price_gone_or_suspended_at_the_next_check_is_gone`.
  - `test_no_check_before_kickoff`.

- [ ] **Step 2: Run them to verify they fail**
  - Run: `.venv/bin/python -m pytest -q Scripts/tests/test_confidence_age_study.py`
  - Expected: FAIL.

- [ ] **Step 3: Implement the module**
  - Use cached answers only.
  - Keep one fixture's histories in memory at a time.
  - While the UTC minute is between 25 and 40, wait, so the hourly true close keeps the endpoint.

- [ ] **Step 4: Run the tests to verify they pass**, then the full suite.

- [ ] **Step 5: Commit**: "A5.3 price-age check (cached OddsPapi histories)".

### Task 6: Run the checks, set the cut-offs or report, and log

**Files:**
- Modify: `Scripts/data_platform/confidence_labels.py` (`CUTOFFS`) only if the firmness check passes.
- Modify: `docs/rework-tracker.md` and `docs/left-to-do-2026-10-23.md`.

- [ ] **Step 1: Run** `.venv/bin/python -m Scripts.ops.confidence_study --out Research/confidence-<date>`. Check `pgrep -f refresh-season` first: it must print nothing.
- [ ] **Step 2: Run** `.venv/bin/python -m Scripts.ops.confidence_age_study --out Research/confidence-age-<date>`. Then check OddsPapi usage: the billable count must be unchanged.
- [ ] **Step 3: Decide by the spec's rules:**
  - **Holdout passes:** set `CUTOFFS = Cutoffs(max_spread=<chosen>)`. Keep `fresh_minutes = 30` if the age check supports it. If it doesn't, keep the value but log it for the owner.
  - **Holdout fails:** leave `max_spread=None`, and write the result up for the owner.
  - **Spix check:** log its per-market results. Do **not** add any market to `SPIX_VOTE_MARKETS`.
- [ ] **Step 4: Verify the cut-off and log the results**
  - If `CUTOFFS` changed, rerun the full suite and `golden_snapshots check`. The check must report no changes.
  - Update the tracker: the A5 status row, the A5.3 open item, a finding with both studies' numbers, and D44's follow-up. Update the left-to-do calendar too.
- [ ] **Step 5: Commit and push**: "A5.3 checks: <passed/failed>, cut-offs <value/none>". Then tell the owner the result, plus any Spix-check market that is a candidate for their approval.
