# Machine-Readable Supplement

These CSV and JSON files support the values reported in the IEEE Access revision. They are summary-level research artifacts, not raw Seagate observations.

## Principal groups

- `baseline_*`: 770-fit conventional LightGBM/XGBoost audit, task eligibility, method summaries, validation-selected rows, and selection frequencies.
- `loto_*`: 45-case equal-budget outer leave-one-task-out sensitivity analysis and gate manifest.
- `nested_*`: stricter task-independent held-out-task audit.
- `v15_*`, `v16_*`, `v161_*`, `v165_*`, `v166_*`: historical ceiling and stress/transfer summaries in their stated evidential scopes.
- `secom_*`: 60 blind repeated splits used as an external-dataset negative control.
- `claim_verification.json`: expected-versus-observed checks for the numerical claims promoted into the manuscript.
- `SHA256_MANIFEST.json`: source hashes recorded when these extracts were generated.

Seeds on fixed Seagate arrays are stochastic repeats, not independent cohorts. The task is the primary inference unit for the main interval. See the manuscript and supplement before reusing any value outside its stated scope.
