# IEEE V10 Uniform-Grid LOTO Audit Run 1

Date: 2026-07-14

## Design

All 9 archived tasks and 5 seeds use the
same V10 validation budget: six augmentation ratios by four keep rates, or
24 candidates per task-seed.  A candidate is fixed by
validation PR-AUC before applying a task-ID-free V11/V12 gate.

For each outer held-out task, the gate is selected on the other eight tasks by
the deterministic order recorded in `fixed_general_gate_manifest.json`.

## Result

- selected hybrids: 11/45
- negative test PR-AUC changes: 0/45
- task-macro mean delta: +0.008579
- 95% task-cluster bootstrap CI: [+0.001152,
  +0.017543]
- minimum task mean delta: +0.000000

## Interpretation boundary

This result is a retrospective task-ID-free sensitivity analysis, not clean
confirmatory evidence.  Although the outer held-out task is not used to choose
its gate in this calculation, the five-rule family itself was historically
developed after exposure to archived test outcomes.  The result supports only
the narrower claim that a validation-derived abstention gate can reproduce a
nonnegative archival profile under an equal candidate budget.

The V12 then-unseen tasks 6 and 9 produced no hybrid selections; therefore they
confirm abstention behavior but provide no positive utility evidence.

## Reproduction

`python experiments/seagate_kqi/build_ieee_v10_uniform_grid_loto_audit.py`
