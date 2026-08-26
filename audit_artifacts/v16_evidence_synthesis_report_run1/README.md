# V16 Evidence Synthesis Report Run1

Generated: 2026-07-03T13:13:45

Purpose:
Clarify the status of V15 and synthesize the post-V15 fresh evidence.

Status:

- V15 is **not discarded**.
- V15 final combined remains the best 45-case deployable main selector.
- V16 is an evidence package and next-direction label built from task2/task10
  fresh validation, not a replacement for the frozen V15 table yet.

45-case main result:

- V12 mean delta vs safe: 0.012217
- V14 mean delta vs safe: 0.019847
- V15 mean delta vs safe: 0.022639
- V15 lift over V14: +0.002792
- V15 lift over V12: +0.010423
- V15 harm: 0/45
- V15 worse than V14 rows: 0/45
- V15 replacements: 7/45

Fresh evidence:

- task2 seeds 47-61: selected 14/15, mean delta +0.023048, harm 0/15
- task10 seeds 47-61: selected 14/15, mean delta +0.061518, harm 0/15
- combined task2+task10 fresh evidence: n=30, selected 28/30, mean delta +0.042283, harm 0/30

Interpretation:

V15 should be kept as the current main result because it improves V14 on the
original 45-case table while preserving zero harm and zero worse-than-V14 rows.
The new V15.2/V16 evidence shows that the improvement is not a one-off table
artifact: task2 and task10 repeatedly show positive fresh-seed gains with zero
harm against the run-local safe reference.

Next direction:

Use this as the V16 research package:

1. Keep V15 as the frozen 45-case main selector.
2. Treat task2 and task10 as validated opportunity tasks.
3. Design a future V16 selector that can incorporate fresh-generalizing task2
   behavior without weakening the original 45-case V15 safety guarantee.

Files:

- `v16_main_45case_comparison.csv`
- `v16_task2_task10_fresh_summary.csv`
- `v16_task2_task10_combined_fresh_score.csv`
- `v16_v15_replacement_summary_by_task.csv`
- `v16_task2_seed_table.csv`
- `v16_task10_seed_table.csv`
- `gpu_status.csv`
