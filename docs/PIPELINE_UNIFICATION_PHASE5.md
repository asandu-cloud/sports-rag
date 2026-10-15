# Pipeline Unification — Phase 5

Phase 5 establishes regression gates around the canonical pre-match decision
boundary. Its purpose is to make prediction and delivery changes reviewable;
it does not claim that the current model has been backtested or calibrated for
the new season.

## Gates now in place

### 1. Recorded canonical-decision replay

`Scripts/tests/test_market_regression.py` replays the scenarios under:

```text
Scripts/tests/fixtures/pipeline_v1/
```

Each scenario freezes a normalized fixture/odds payload, the requested
markets, a deterministic generation time, an input-snapshot identifier, and an
approved expected decision view. The replay runs the real canonical service,
odds extraction, and line-selection code. It asserts:

- market ordering and identity;
- projection values and components that are exposed to users;
- recommendation / no-bet / unavailable state;
- selected side, line, bookmaker, odds, probabilities, edge, EV, confidence,
  and reason;
- relevant provenance and evidence codes.

The initial coverage is deliberately small but representative:

| Scenario | Gate |
|---|---|
| `priced_standard` | Priced goals, BTTS, and moneyline recommendations |
| `no_bet` | A real price with insufficient conviction remains a pass |
| `missing_odds` | A projected market without prices remains unavailable |
| `insufficient_profile` | Missing model inputs cannot create a recommendation |

### 2. Cross-surface contract gate

`Scripts/tests/test_market_delivery.py` verifies that:

- market dispatch evaluates the canonical service;
- legacy text is only a one-way presentation adapter;
- the website exposes the unmodified `market_results` contract alongside its
  compatibility fields; and
- Discord bet cards read the result attributes directly, with no parser in the
  decision path.

### 3. Contract-level comparison helper

`core/market_regression.py` owns the semantic comparison view. It deliberately
excludes changing timestamps and presentation details, while making every
decision-relevant difference fail clearly.

## Recording policy

The current fixture set is labelled `synthetic_selector_delivery`: it freezes
projection outputs because the repository has no stable archived combination
of pre-match team/profile snapshots and odds snapshots yet. This is useful for
protecting selection and delivery behaviour, but it is not a substitute for a
historical replay/backtest.

When historical source data is selected, add recordings using the same schema
with the actual sanitized profile/context/odds inputs. They should supplement,
not overwrite, the synthetic gates.

## Intentional model corrections

Historical output is not required to stay identical when a model correction is
intentional. A correction must instead:

1. preserve the frozen input and show the old-versus-new decision difference;
2. increment `baseline_revision` and update `baseline_reason` in the affected
   recording;
3. document why the previous outcome was wrong or unsafe; and
4. pass the full selector, delivery, and regression suite.

The replay test requires this metadata to be present; revision discipline is
then enforced through normal code review. Layout, prose, and generation-time
changes are not behavioural parity targets.

## Remaining Phase 5 work

- Add archived pre-match recordings once a historic odds/profile source is
  selected.
- Add a walk-forward performance gate (calibration, ROI, and sample-size
  reporting) before changing weights or thresholds for the season.
- Add a deployment smoke test using a non-production API/Discord environment.
