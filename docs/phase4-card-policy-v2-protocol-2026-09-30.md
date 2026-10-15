# Card research amendment: two minutes and league phases

Owner-approved on 30 September 2026, before this amendment's arithmetic run.

- Stage `spix-participation-settlement.v2` / `phase4-participation-card-target.v2`
  with recorded minutes **>= 2**, yellow 1, straight red 2 and a maximum of 3
  per player for the two-yellow dismissal case. Missing eligible-player counts
  and unknown minutes for carded players remain unknown. Yellow-only markets
  remain separate. This changes participation, not yellow/red event semantics.
- New research targets, derived team/opponent/referee histories and evaluation
  must identify and use this same definition. Do not mix v1 and v2 histories.
- Production keeps v1, including old publication policy snapshots and grades.
  Promote the rebuilt calculation, compatible profiles and new engine together
  in a later approved release. The shared parser's optional minimum is not a
  production activation: existing callers retain their one-minute default.
- Include ordinary domestic league rounds, numbered integral Belgian league
  split groups, and European group/league phases. Reuse the existing versioned
  competition/round classifier. Exclude separate playoffs, European qualifying,
  knockouts and unknown rounds from both targets and historical context for now.
- For an exact saved fixture identity with trusted competition/season/round
  metadata and completed FT status, these known formats establish regulation
  scope. Do not require an additional historical regulation certificate solely
  because the earlier frozen export omitted it. Contradictions and non-FT status
  still exclude the fixture. Source completeness of the player records remains
  an independent gate; this amendment does not invent an original response or
  automatically certify legacy exports.
- Recompute only previously admitted pre-2024 development player rows from the
  sealed local reconstruction. Compare policy effects without fitting weights.
  New all-season reads are identity, round and player/archive presence metadata
  only. Never decode later player/team outcome values or archived responses.
- Preserve old artifacts. Write a new immutable amendment artifact with inputs,
  source hashes, policy, scoped targets, exclusions, archive-gap inventory,
  comparison support and tests. No API calls or canonical database mutations.
- The archive-gap inventory identifies `/fixtures/players?fixture=<id>` responses
  and normalized/legacy presence separately. It is not an approved download plan
  or proof that an existing archive is complete. Future acquisition should save
  the full response before applying the profile/minutes filter.

Fitting/evaluation cutoffs, source requirements, support minimums and the
final-system reservation remain unchanged. This amendment does not launch a
backfill, change public delivery, retune weights or activate the candidate.
