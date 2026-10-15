# Founder skills for Codex

Installed for this repository under `.agents/skills/founder-*` from [emotixco/claude-skills-founder](https://github.com/emotixco/claude-skills-founder), version 2.0.2, commit `41ee742d8d683f93daa1994af3698881af472a2a`.

These are founder/business analysis workflows. They do not validate forecasting models or establish profitable recommendations. Installation does not run an analysis or change the application.

## Use

On the next turn, type `$founder-` to select a skill. If the list does not refresh, restart Codex. For example:

```text
$founder-product-brief Analyse Spix as it exists today. Verify current implementation claims, distinguish live features from previews, and respect our UK audience and single-subscription decision. Identify unknowns rather than inventing traction or predictive value.
```

Outputs are saved under `founder/` by the invoked skill. Later skills can reuse them. The original framework has defaults such as three pricing tiers; the user's one-subscription decision takes precedence.

## Compatibility changes

- Names and cross-skill references use `$founder-<name>` instead of Claude's `/founder:<name>`.
- Each skill bundles `references/conventions.md`; no `${CLAUDE_PLUGIN_ROOT}` dependency remains.
- User input comes from the request and context rather than `$ARGUMENTS` substitution.
- Claude-specific argument and tool-name metadata was replaced with normal Codex metadata. The original analysis-output scope remains in the instructions.
- Each skill retains MIT attribution and an `UPSTREAM.json` record. These are locally adapted copies; upstream updates are not automatic and must preserve the compatibility changes.

## Installed skills

- `$founder-competitor-matrix`
- `$founder-email-sequence`
- `$founder-fundraise-prep`
- `$founder-go-to-market`
- `$founder-landing-page`
- `$founder-metrics-dashboard`
- `$founder-mvp-scope`
- `$founder-persona-gen`
- `$founder-pitch-deck`
- `$founder-pricing-strategy`
- `$founder-product-brief`
- `$founder-user-interviews`
- `$founder-validate-idea`

All 13 pass the bundled Codex skill validator. Required references, metadata and installed file hashes were checked. Discovery in the next-turn skill picker and complete workflow execution remain runtime checks; no business assessment was run as an installation test.

References: [Codex skill discovery and invocation](https://learn.chatgpt.com/docs/build-skills), [upstream repository](https://github.com/emotixco/claude-skills-founder).
