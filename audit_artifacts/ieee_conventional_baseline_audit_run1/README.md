# IEEE Conventional Baseline Audit (Merged)

Two classifier runs were executed independently and merged only after each
completed its full 11-task by 5-seed by 7-method grid.

## Integrity checks

- Rows: 770/770.
- Duplicate keys: 0.
- Failed fits: 0.
- Eligibility uses training valid-row count only; all 11 endpoints are included.
- Every method has one fixed configuration and 200 trees.

## Validation-selected conventional results

- lightgbm: task-macro test PR-AUC 0.103870 (95% task-bootstrap CI 0.072461 to 0.134726); Recall@FPR=0.01 0.093671.
- xgboost: task-macro test PR-AUC 0.110877 (95% task-bootstrap CI 0.078604 to 0.143263); Recall@FPR=0.01 0.101499.

This conventional audit does not imply that archived diffusion candidates
were compared with both classifiers; complete synthetic matrices were not retained.
