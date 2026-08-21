# V16.5 Task0 Train-Only Temporal Backtest Audit

This is a rolling-origin, array-index-order replay. Fold A predictions were
sealed before Q4 was allowed to enter Fold B as validation; Fold B predictions
were then sealed before Q5 row-level outcomes were loaded. Aggregate class
counts were known at design time. The frozen V16.4 validation-only score and
certificate were used without human retuning between folds.

## Result

- accepted: `3/10`
- mean policy delta vs safe PR-AUC: `-0.006550`
- minimum per-seed policy delta: `-0.053332`
- policy harm: `2/10`
- mean frozen-threshold test FPR excess: `-0.002775`
- locked stress decision: `fails_locked_retrospective_stress_checks`

- fold_a: accepted `2/5`, mean delta `-0.015621`, harm `2/5`, ensemble delta `-0.023010`, bootstrap p05 `-0.043342`, mean/max-seed FPR excess `-0.002227`/`+0.000000`
- fold_b: accepted `1/5`, mean delta `+0.002520`, harm `0/5`, ensemble delta `+0.011830`, bootstrap p05 `-0.001720`, mean/max-seed FPR excess `-0.003322`/`+0.000000`

The ten seed-fold rows are paired stochastic repeats on two nested blocks, not
ten independent generalization samples; no binomial harm-rate inference is
attached to them. The ensemble endpoint is a separate secondary deployment
object formed by arithmetic-mean soft voting across the five policy models and
their five paired safe models within each fold. Its paired stratified bootstrap
p05 is a conditional-i.i.d.-row, fixed-class-count descriptive percentile only;
it ignores temporal/wafer clustering and algorithmic seed uncertainty.

## Decision boundary

This is retrospective row-level own-test-label-masked, array-index-order stress
evidence only. The public NPY data were preprocessed upstream using the original
70-week cohort, timestamps and wafer/lot IDs are unavailable, and upstream
imputation/encoding cannot be replayed. Consequently V16.5 cannot promote task0
or replace V15 regardless of the numeric result. Task0 remains a safe reject
until a post-week92 cohort with provenance is evaluated once under a frozen
policy. The paper directory was not modified.

Q1-Q5 are equal-count ordered task0-valid observation blocks, not verified
calendar windows or entity-disjoint cohorts. Generic test metrics produced
during modeling used deterministic dummy labels and are invalid/excluded; only
the sealed score arrays are evaluated here against the row-level replay labels.

## Files

- `v165_temporal_case_rows.csv`
- `v165_per_fold_summary.csv`
- `v165_overall_summary.csv`
- `v165_ensemble_paired_bootstrap.csv`
- `v165_decision_matrix.csv`
- `v165_unblind_manifest.csv`
- `gpu_status.csv`
