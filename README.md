# Seagate KQI Diffusion-Augmentation Audit — Code and Machine-Readable Results

Companion repository for the IEEE Access submission:

> **When Validation-Gated Diffusion Augmentation Fails to Generalize:
> A Retrospective Audit of Rare Seagate KQI Classification**
> Tae-Yoon Kim, Young-Shin Han, and Jong-Sik Lee, Inha University

This repository provides numerical traceability for every reported result:
the analysis scripts, the archived audit outputs they produced, the exact
input tables they consume, and machine-readable summaries with SHA-256
manifests. It does **not** regenerate the historical diffusion candidates;
the underlying synthetic training matrices were not retained, and the paper
states this limitation explicitly.

## Layout

| Directory | Contents |
|---|---|
| `analysis_code/` | Audit and figure scripts, including the matched-objective attribution audit and its pre-frozen specification (`MATCHED_OBJECTIVE_GATE_AUDIT_SPEC.md`) |
| `audit_artifacts/` | Archived outputs of each audit run (CSV results, README, SHA-256 manifests, environment records) |
| `machine_readable_supplement/` | Flat machine-readable summary tables and `claim_verification.json` (42 checks), which binds the principal numeric claims in the manuscript to their source artifacts |
| `input_tables/` | The exact archived candidate/result tables the audit scripts read |

## Data

The Seagate `time-series-2` arrays are public and are **not** redistributed
here. They are available from the original repository at the exact commit
recorded in the paper:

```
https://github.com/Seagate/softsensing_data
commit 3731abbf0111f62ee10b28cdb2be089f32a99625  (Apache-2.0)
```

SECOM is available from the UCI Machine Learning Repository
(DOI 10.24432/C54305).

## Re-running the audits

The scripts were archived from a larger research workspace and resolve their
inputs relative to that workspace as
`experiments/seagate_kqi/artifacts/condition_aware_main/<run_dir>/<file>`.
To re-run an audit, recreate that relative layout from `input_tables/`
(each subdirectory keeps its original run-directory name), or edit the
path constants at the top of the script. All audits are deterministic;
bootstrap draws use NumPy `default_rng(20260714)` with 10,000 resamples.

| Script | Reported in |
|---|---|
| `run_ieee_conventional_baseline_audit.py` | Conventional all-endpoint audit (Table 2) |
| `build_ieee_task_independent_nested_audit.py` | Primary task-independent nested audit (Table 4) |
| `build_ieee_v10_uniform_grid_loto_audit.py` | Equal-budget outer LOTO sensitivity (Table 4) |
| `build_ieee_matched_objective_gate_audit.py` | Matched-objective attribution audit (Table 5); the hardened builder writes `..._run2`, which is byte-identical to the cited `..._run1` |
| `verify_matched_audit_claims.py` | Appends the matched-audit checks to `claim_verification.json` |
| `build_ieee_revision_figures.py` | Figures 1–3 |

## Integrity

Each audit directory carries its own README and SHA-256 manifest.
`machine_readable_supplement/claim_verification.json` records the
claim-by-claim check that the 42 principal numeric claims in the manuscript match their source
artifact. The matched-objective audit additionally binds its pre-analysis
specification and its input tables by SHA-256 in each run's
`audit_manifest.json`; the specification's local file metadata predates the
audit output, but no independently timestamped preregistration exists.

## License

MIT (see `LICENSE`). The Seagate source data remain under their own
Apache-2.0 license at the repository above.

## v1.5 additions (2026-09-01, review response)

- `machine_readable_supplement/primary_nested_audit_spec.json`: complete
  machine-readable specification of the primary task-independent nested audit
  (candidate families, the internal 24-configuration grid and
  validation-winner rule, the seven gates with exact thresholds, fold roster,
  objectives and deterministic tie-breaks, metric definitions, run-local
  reference provenance, input/output SHA-256 hashes, and the mapping to the
  primary rows of the article).
- `analysis_code/build_primary_nested_audit_spec.py`: the script that
  generated the specification from the archived audit outputs.
- `audit_artifacts/secom_external_frozen_seal/v11_frozen_rule_manifest.json`:
  the frozen-rule manifest (sealed 2026-07-11) that preceded the blind SECOM
  transfer audit.
