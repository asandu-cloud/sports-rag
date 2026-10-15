# Canonical selector release policy

The early-season guardrails are a deliberate model correction. They must be
evaluated from decisions captured before kick-off, not from a reconstruction
using today's profiles or prices.

## Release modes

`PREDICTION_RELEASE_MODE` is interpreted at the canonical market-service
boundary, so it applies equally to the website and Discord delivery paths.

| Mode | Behaviour | Use |
|---|---|---|
| `live` | Eligible canonical recommendations can be displayed/published. This is the default when the variable is absent. | Established release only. |
| `shadow` | Every otherwise eligible pick is retained as `context.release_candidate`, converted to a no-bet for delivery, and therefore cannot be published. | Evaluate a revision safely. |
| `disabled` | Recommendations are withheld and not treated as shadow candidates. | Data/provider incident or emergency stop. |

An explicitly configured but invalid value fails closed to `disabled` and is
included in the result context as a configuration error.

## Shadow operation

Start the bot/web process with `PREDICTION_RELEASE_MODE=shadow`. It will not
publish canonical recommendations. Capture the candidates once per matchday
before fixtures begin:

```bash
.venv/bin/python Scripts/rag_ingest/shadow_release.py --league EPL --date 2026-09-01
```

Repeat `--league` for each covered competition. The command:

- forces shadow mode only for its own process;
- writes deduplicated canonical decision snapshots to
  `Index/shadow_runs/canonical_shadow.jsonl`; and
- logs candidates to `Index/predictions.db` with source `canonical_shadow`, so
  the existing outcome resolver can settle them.

Run it on the same matchday as the fixtures. The legacy prediction tracker
resolves by prediction date, so it is not currently suitable for logging a
future date several days in advance.

After results are final, resolve the shadow picks as usual, then review the
chronological report:

```bash
.venv/bin/python Scripts/rag_ingest/prediction_tracker.py --resolve-date 2026-09-01
.venv/bin/python Scripts/rag_ingest/walk_forward.py \
  --predictions-db Index/predictions.db --source canonical_shadow --strict
```

The report includes ROI, hit rate, Brier score, probability-bucket calibration,
market/league/source slices, and any snapshot captured at or after kick-off.
It holds promotion by default unless there are at least 50 resolved stake units,
at least 20 in a reportable probability bucket, and no qualifying calibration
gap over 10 percentage points. These are release-safety thresholds, not model
weights and not an optimisation target.

## Promotion and rollback

Before setting the environment to `live`, attach the generated report to the
model-change record and state the baseline revision, reason, sample size,
calibration result, ROI/CLV result where available, and affected markets.

If a regression, provider issue, or suspect output appears:

1. Set `PREDICTION_RELEASE_MODE=shadow` to continue collecting candidates
   without publishing them, or `disabled` to stop canonical recommendations
   completely.
2. Preserve the affected shadow snapshots and report; do not rewrite them.
3. Roll back the application deployment to the last approved revision only
   after recording the reason and scope of the rollback.
4. Re-run the full test suite and a new shadow period before another promotion.

This policy intentionally does not claim an archived historical selector
replay exists yet. Creating immutable fixture, odds, profile, and lineup input
snapshots remains required for that deeper backtest.
