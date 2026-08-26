# V16.6 Time-Series-1 Frozen Procedure-Transfer Audit

V16.6 applies the frozen V16.4 validation-only score, 112-candidate grid,
acceptance certificate, and safe fallback to five objectively eligible,
anonymous `time-series-1` endpoints. Candidate selection and score sealing were
completed against deterministic dummy test labels before row-level outcomes
were loaded.

## Result

- validation-gate accepted: `1/5`
- macro mean policy delta vs safe PR-AUC: `+0.011483`
- minimum task delta: `+0.000000`
- policy harm: `0/5`
- mean/max-task frozen-threshold FPR excess:
  `-0.001699` / `+0.000000`
- locked pilot decision: `fails_locked_ts1_transfer_pilot`
- next-experiment rule: `stop_v166_seed_expansion`

- task 2: `augment`, policy delta `+0.057414`, FPR excess `-0.008496`
- task 4: `safe_reject`, policy delta `+0.000000`, FPR excess `+0.000000`
- task 5: `safe_reject`, policy delta `+0.000000`, FPR excess `+0.000000`
- task 6: `safe_reject`, policy delta `+0.000000`, FPR excess `+0.000000`
- task 10: `safe_reject`, policy delta `+0.000000`, FPR excess `+0.000000`

## Claim boundary

This is a first project-use procedure transfer to a slightly different Seagate
tool family, not a semantic mapping of task IDs and not an external-factory
validation. Aggregate class counts were known for objective endpoint
eligibility, but row-level test outcomes were masked during selection. The five
tasks share a tool family and may share rows, so they are breadth cases rather
than independent datasets; no micro-pooled or binomial inference is reported.

Generic pipeline test metrics used dummy labels and are invalid/excluded. Only
the sealed float64 score arrays were evaluated here. Regardless of the numeric
result, V15 remains the main result. Paper files were not modified.

## Files

- `v166_task_case_rows.csv`
- `v166_macro_summary.csv`
- `v166_decision_matrix.csv`
- `v166_unblind_manifest.csv`
- `gpu_status.csv`
