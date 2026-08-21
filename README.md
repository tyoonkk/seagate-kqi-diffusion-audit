# Seagate KQI Diffusion-Augmentation Audit — Code and Machine-Readable Results

Companion repository for the IEEE Access submission:

> **When Validation-Gated Diffusion Augmentation Fails to Generalize:
> A Retrospective Audit of Rare Seagate KQI Classification**
> Taeyoon Kim, Inha University

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
| `machine_readable_supplement/` | Flat machine-readable summary tables and `claim_verification.json`, which binds every numeric claim in the manuscript to a source artifact |
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
| `build_ieee_matched_objective_gate_audit.py` | Matched-objective attribution audit (Table 5) |
| `build_ieee_revision_figures.py` | Figures 1–3 |

## Integrity

Each audit directory carries its own README and SHA-256 manifest.
`machine_readable_supplement/claim_verification.json` records the
claim-by-claim check that every number in the manuscript matches its source
artifact. The matched-objective audit additionally binds its pre-analysis
specification (frozen before any result was viewed) and its input tables by
SHA-256 in `audit_artifacts/ieee_matched_objective_gate_audit_run1/audit_manifest.json`.

## License

MIT (see `LICENSE`). The Seagate source data remain under their own
Apache-2.0 license at the repository above.
