# IEEE Task-Independent Nested Audit Run 1

Date: 2026-07-14

## Scope

This audit uses only the archival `quick_mean_alltasks` candidate source.  Each of the
9 tasks and 5 seeds has exactly the same
three candidate families: `ig_clean, ig_smote, s1_plus_ig`.  The outer split is
leave-one-task-out; no task ID is an input to any gate.

This is an archival nested audit, not prospective or external validation.

## Results

- Conservative gate selection: task-macro mean delta
  +0.000000, 95% task-bootstrap CI
  [+0.000000,
  +0.000000], negative tasks
  0/9,
  negative rows 0/45,
  hybrid selections 0/45.
- Mean-only gate selection: task-macro mean delta
  -0.002087, 95% task-bootstrap CI
  [-0.005395,
  +0.000000], negative tasks
  2/9,
  negative rows 5/45,
  hybrid selections 7/45.

## Interpretation

The task-independent audit does not validate a generally beneficial hybrid
selector.  Under the conservative criterion the learned policy falls back to
the conventional reference in every held-out fold.  Removing the no-negative
constraint causes negative held-out performance.  The later V15 result must
therefore be labelled a retrospective, task-specific development result rather
than independent evidence for a general selector.

## Reproduction

Run:

`python experiments/seagate_kqi/build_ieee_task_independent_nested_audit.py`

The task-cluster bootstrap uses NumPy `default_rng(20260714)` with
10,000 resamples.  Candidate ranking is deterministic and fully
specified in the script.
