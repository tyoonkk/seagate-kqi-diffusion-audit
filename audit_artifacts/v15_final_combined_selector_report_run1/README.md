# V15 Final Combined Selector Report Run1

Date: 2026-07-02

Purpose:
Freeze the strongest post-V14 selector candidate into a 45-case report:
use V14 everywhere except the prior task2 gate and the new task10 low-canary
score selector.

Selector:
- task2: previous canary-shift gate from V15 signal mining
- task10: low-canary microgrid score, canary_p95 <= 0.10 and score = validation PR-AUC - 0.10 * canary_p95
- other tasks: keep V14 current main selection

GPU after summary:
- NVIDIA GeForce RTX 3070 / 1004 MiB / util 16%

Main comparison:
- V12 fixed baseline mean delta: 0.012217
- V14 current main mean delta: 0.019847
- V15 final combined mean delta: 0.022639
- V15 lift over V14: +0.002792
- V15 lift over V12: +0.010423
- V15 harm: 0/45
- V15 worse than V14 rows: 0/45
- V15 replacements: 7/45

Task10 contribution:
- V15 task10 mean delta vs V12 safe: 0.076304
- task10 lift over V14 task mean delta: +0.022848

Interpretation:
This is now the strongest deployable post-V14 selector candidate in the current
experiment thread. It improves the 45-case mean PR-AUC delta while keeping harm
and worse-than-V14 rows at zero.

Files:
- `v15_final_combined_selected_rows.csv`
- `v15_final_combined_comparison_rows.csv`
- `v15_final_combined_overall_summary.csv`
- `v15_final_combined_summary_by_task.csv`
- `v15_final_combined_replacements.csv`
- `gpu_status_after_summary.csv`
