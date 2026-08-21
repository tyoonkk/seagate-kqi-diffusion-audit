#!/usr/bin/env python
"""Task-independent nested audit for the IEEE Access revision.

This audit deliberately uses only the common ``quick_mean_alltasks`` source
from the V9 evidence table.  That source contains the same three candidate
families for every one of the nine archived tasks and five stochastic seeds.

The outer unit is the task.  For each held-out task, a small, fixed family of
task-ID-free validation gates is compared using test outcomes from the other
eight development tasks only.  The selected gate is then evaluated on the
held-out task.  Test fields from the held-out task never enter gate selection.

Two selection regimes are reported:

* ``conservative``: a non-reference gate is eligible only when it has no
  negative task mean and no negative row on the eight development tasks.
* ``mean_only``: select the gate with the largest development-task macro mean.

The conservative regime tests whether a no-negative-development requirement
generalizes.  The mean-only regime is a sensitivity analysis that removes that
constraint.  Neither regime is described as prospective validation because
the underlying arrays and candidate suite are archival.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
INPUT = (
    ROOT
    / "experiments/seagate_kqi/artifacts/condition_aware_main/"
    "selector_v9_certificate_mining_run1/v9_candidate_evidence_table.csv"
)
OUT_DIR = (
    ROOT
    / "experiments/seagate_kqi/artifacts/condition_aware_main/"
    "ieee_task_independent_nested_audit_run1"
)
SOURCE = "quick_mean_alltasks"
EXPECTED_METHODS = ("ig_clean", "ig_smote", "s1_plus_ig")
EXPECTED_TASKS = (0, 1, 2, 3, 5, 6, 7, 9, 10)
EXPECTED_SEEDS = (42, 43, 44, 45, 46)
BOOTSTRAP_SEED = 20260714
BOOTSTRAP_REPS = 10_000


@dataclass(frozen=True)
class Gate:
    name: str
    description: str
    min_val_gain: float | None = None
    max_val_fpr_excess: float | None = None
    positive_only: bool = False
    min_tstr_pr_auc: float | None = None
    pass_all: bool = False


GATES = (
    Gate("reference_only", "Always retain the validation-selected conventional reference."),
    Gate("validation_best", "Deploy the validation-best hybrid without an additional gate.", pass_all=True),
    Gate(
        "gain0_fpr0",
        "Require nonnegative validation PR-AUC gain and no validation FPR increase.",
        min_val_gain=0.0,
        max_val_fpr_excess=0.0,
    ),
    Gate(
        "gain005_fpr0",
        "Require validation PR-AUC gain >= 0.005 and no validation FPR increase.",
        min_val_gain=0.005,
        max_val_fpr_excess=0.0,
    ),
    Gate(
        "gain01_fpr01",
        "Require validation PR-AUC gain >= 0.01 and validation FPR excess <= 0.01.",
        min_val_gain=0.01,
        max_val_fpr_excess=0.01,
    ),
    Gate(
        "positive_gain0_fpr01",
        "Positive-only generation with nonnegative validation gain and FPR excess <= 0.01.",
        min_val_gain=0.0,
        max_val_fpr_excess=0.01,
        positive_only=True,
    ),
    Gate(
        "positive_tstr_gain0",
        "Positive-only generation with TSTR >= 0.008, nonnegative gain, and FPR excess <= 0.01.",
        min_val_gain=0.0,
        max_val_fpr_excess=0.01,
        positive_only=True,
        min_tstr_pr_auc=0.008,
    ),
)


def load_common_candidates() -> pd.DataFrame:
    if not INPUT.exists():
        raise FileNotFoundError(INPUT)
    candidates = pd.read_csv(INPUT, low_memory=False)
    candidates = candidates[candidates["source"].eq(SOURCE)].copy()
    required = {
        "task_id",
        "seed",
        "method",
        "candidate_key",
        "ddpm_train_mode",
        "val_PR_AUC",
        "val_gain",
        "val_fpr_excess",
        "tstr_pr_auc",
        "test_PR_AUC",
        "safe_baseline_test_PR_AUC",
        "delta_vs_safe_baseline_test",
    }
    missing = sorted(required.difference(candidates.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    tasks = tuple(sorted(candidates["task_id"].astype(int).unique()))
    seeds = tuple(sorted(candidates["seed"].astype(int).unique()))
    if tasks != EXPECTED_TASKS:
        raise ValueError(f"Expected tasks {EXPECTED_TASKS}, found {tasks}")
    if seeds != EXPECTED_SEEDS:
        raise ValueError(f"Expected seeds {EXPECTED_SEEDS}, found {seeds}")

    pair_counts = candidates.groupby(["task_id", "seed"]).size()
    if not (pair_counts == len(EXPECTED_METHODS)).all():
        raise ValueError(f"Unequal candidate counts:\n{pair_counts}")
    method_sets = candidates.groupby(["task_id", "seed"])["method"].agg(
        lambda values: tuple(sorted(set(values)))
    )
    if not method_sets.map(lambda values: values == EXPECTED_METHODS).all():
        raise ValueError(f"Candidate families differ across rows:\n{method_sets}")
    if candidates.duplicated(["task_id", "seed", "method"]).any():
        raise ValueError("Duplicate task-seed-method candidates found")
    return candidates.sort_values(["task_id", "seed", "method"]).reset_index(drop=True)


def gate_mask(frame: pd.DataFrame, gate: Gate) -> pd.Series:
    if gate.name == "reference_only":
        return pd.Series(False, index=frame.index)
    mask = pd.Series(True, index=frame.index)
    if gate.pass_all:
        return mask
    if gate.min_val_gain is not None:
        mask &= frame["val_gain"].ge(gate.min_val_gain)
    if gate.max_val_fpr_excess is not None:
        mask &= frame["val_fpr_excess"].le(gate.max_val_fpr_excess)
    if gate.positive_only:
        mask &= frame["ddpm_train_mode"].eq("positive_only")
    if gate.min_tstr_pr_auc is not None:
        mask &= frame["tstr_pr_auc"].fillna(-np.inf).ge(gate.min_tstr_pr_auc)
    return mask


def apply_gate(candidates: pd.DataFrame, gate: Gate) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (task_id, seed), group in candidates.groupby(["task_id", "seed"], sort=True):
        passed = group[gate_mask(group, gate)].copy()
        if passed.empty:
            sample = group.iloc[0]
            selected_kind = "reference"
            selected_method = str(sample["safe_baseline_method"])
            selected_candidate_key = "reference"
            selected_val_gain = 0.0
            selected_val_fpr_excess = 0.0
            selected_test_pr_auc = float(sample["safe_baseline_test_PR_AUC"])
            reference_test_pr_auc = selected_test_pr_auc
            delta = 0.0
        else:
            pick = passed.sort_values(
                ["val_PR_AUC", "val_fpr_excess", "tstr_pr_auc", "candidate_key"],
                ascending=[False, True, False, True],
                na_position="last",
                kind="mergesort",
            ).iloc[0]
            selected_kind = "hybrid"
            selected_method = str(pick["method"])
            selected_candidate_key = str(pick["candidate_key"])
            selected_val_gain = float(pick["val_gain"])
            selected_val_fpr_excess = float(pick["val_fpr_excess"])
            selected_test_pr_auc = float(pick["test_PR_AUC"])
            reference_test_pr_auc = float(pick["safe_baseline_test_PR_AUC"])
            delta = float(pick["delta_vs_safe_baseline_test"])
        rows.append(
            {
                "gate": gate.name,
                "task_id": int(task_id),
                "seed": int(seed),
                "selected_kind": selected_kind,
                "selected_method": selected_method,
                "selected_candidate_key": selected_candidate_key,
                "selected_val_gain": selected_val_gain,
                "selected_val_fpr_excess": selected_val_fpr_excess,
                "selected_test_PR_AUC": selected_test_pr_auc,
                "reference_test_PR_AUC": reference_test_pr_auc,
                "delta_vs_reference_test": delta,
                "negative_row": bool(delta < 0.0),
            }
        )
    return pd.DataFrame(rows)


def summarize(frame: pd.DataFrame) -> dict[str, object]:
    task_delta = frame.groupby("task_id")["delta_vs_reference_test"].mean()
    return {
        "n_tasks": int(task_delta.size),
        "n_rows": int(len(frame)),
        "task_macro_mean_delta": float(task_delta.mean()),
        "task_median_delta": float(task_delta.median()),
        "negative_task_count": int(task_delta.lt(0.0).sum()),
        "negative_row_count": int(frame["negative_row"].sum()),
        "hybrid_selection_count": int(frame["selected_kind"].eq("hybrid").sum()),
        "hybrid_selection_rate": float(frame["selected_kind"].eq("hybrid").mean()),
    }


def choose_gate(scores: pd.DataFrame, regime: str) -> str:
    eligible = scores.copy()
    if regime == "conservative":
        eligible = eligible[
            eligible["negative_task_count"].eq(0)
            & eligible["negative_row_count"].eq(0)
        ].copy()
        if eligible.empty:
            return "reference_only"
    elif regime != "mean_only":
        raise ValueError(regime)

    # Explicit deterministic tie break: larger task-macro mean, then fewer
    # negative tasks/rows, then fewer hybrid deployments, then lexical name.
    eligible = eligible.sort_values(
        [
            "task_macro_mean_delta",
            "negative_task_count",
            "negative_row_count",
            "hybrid_selection_count",
            "gate",
        ],
        ascending=[False, True, True, True, True],
        kind="mergesort",
    )
    return str(eligible.iloc[0]["gate"])


def task_bootstrap_ci(frame: pd.DataFrame) -> tuple[float, float]:
    task_delta = frame.groupby("task_id")["delta_vs_reference_test"].mean().to_numpy()
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, len(task_delta), size=(BOOTSTRAP_REPS, len(task_delta)))
    draws = task_delta[indices].mean(axis=1)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return float(lo), float(hi)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    candidates = load_common_candidates()

    gate_rows = {gate.name: apply_gate(candidates, gate) for gate in GATES}
    definitions = pd.DataFrame([asdict(gate) for gate in GATES])
    definitions.to_csv(OUT_DIR / "policy_definitions.csv", index=False)

    source_audit = (
        candidates.groupby(["task_id", "seed"])
        .agg(candidate_count=("method", "size"), methods=("method", lambda x: ",".join(sorted(x))))
        .reset_index()
    )
    source_audit.to_csv(OUT_DIR / "candidate_source_audit.csv", index=False)

    train_scores: list[dict[str, object]] = []
    choices: list[dict[str, object]] = []
    heldout_frames: list[pd.DataFrame] = []
    tasks = sorted(candidates["task_id"].astype(int).unique())
    for regime in ("conservative", "mean_only"):
        for heldout_task in tasks:
            fold_scores: list[dict[str, object]] = []
            for gate in GATES:
                train = gate_rows[gate.name][gate_rows[gate.name]["task_id"].ne(heldout_task)]
                row = {
                    "regime": regime,
                    "heldout_task_id": int(heldout_task),
                    "gate": gate.name,
                    **summarize(train),
                }
                fold_scores.append(row)
                train_scores.append(row)
            fold_scores_df = pd.DataFrame(fold_scores)
            chosen = choose_gate(fold_scores_df, regime)
            heldout = gate_rows[chosen][gate_rows[chosen]["task_id"].eq(heldout_task)].copy()
            heldout.insert(0, "regime", regime)
            heldout.insert(1, "heldout_task_id", int(heldout_task))
            heldout.insert(2, "chosen_gate", chosen)
            heldout_frames.append(heldout)
            choice = {
                "regime": regime,
                "heldout_task_id": int(heldout_task),
                "chosen_gate": chosen,
                **{f"heldout_{k}": v for k, v in summarize(heldout).items()},
            }
            choices.append(choice)

    train_scores_df = pd.DataFrame(train_scores)
    choices_df = pd.DataFrame(choices)
    heldout_df = pd.concat(heldout_frames, ignore_index=True)
    train_scores_df.to_csv(OUT_DIR / "outer_train_policy_scores.csv", index=False)
    choices_df.to_csv(OUT_DIR / "outer_fold_choices.csv", index=False)
    heldout_df.to_csv(OUT_DIR / "outer_heldout_rows.csv", index=False)

    task_summary = (
        heldout_df.groupby(["regime", "task_id"])
        .agg(
            chosen_gate=("chosen_gate", "first"),
            n=("seed", "size"),
            mean_delta=("delta_vs_reference_test", "mean"),
            median_delta=("delta_vs_reference_test", "median"),
            negative_rows=("negative_row", "sum"),
            hybrid_selections=("selected_kind", lambda x: int(x.eq("hybrid").sum())),
        )
        .reset_index()
    )
    task_summary.to_csv(OUT_DIR / "task_summary.csv", index=False)

    summaries: list[dict[str, object]] = []
    for regime, frame in heldout_df.groupby("regime", sort=True):
        lo, hi = task_bootstrap_ci(frame)
        summaries.append(
            {
                "regime": regime,
                **summarize(frame),
                "task_bootstrap_ci_low": lo,
                "task_bootstrap_ci_high": hi,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "bootstrap_reps": BOOTSTRAP_REPS,
            }
        )
    summary = pd.DataFrame(summaries)
    summary.to_csv(OUT_DIR / "summary.csv", index=False)

    conservative = summary[summary["regime"].eq("conservative")].iloc[0]
    mean_only = summary[summary["regime"].eq("mean_only")].iloc[0]
    readme = f"""# IEEE Task-Independent Nested Audit Run 1

Date: {date.today().isoformat()}

## Scope

This audit uses only the archival `{SOURCE}` candidate source.  Each of the
{len(EXPECTED_TASKS)} tasks and {len(EXPECTED_SEEDS)} seeds has exactly the same
three candidate families: `{', '.join(EXPECTED_METHODS)}`.  The outer split is
leave-one-task-out; no task ID is an input to any gate.

This is an archival nested audit, not prospective or external validation.

## Results

- Conservative gate selection: task-macro mean delta
  {conservative['task_macro_mean_delta']:+.6f}, 95% task-bootstrap CI
  [{conservative['task_bootstrap_ci_low']:+.6f},
  {conservative['task_bootstrap_ci_high']:+.6f}], negative tasks
  {int(conservative['negative_task_count'])}/{int(conservative['n_tasks'])},
  negative rows {int(conservative['negative_row_count'])}/{int(conservative['n_rows'])},
  hybrid selections {int(conservative['hybrid_selection_count'])}/{int(conservative['n_rows'])}.
- Mean-only gate selection: task-macro mean delta
  {mean_only['task_macro_mean_delta']:+.6f}, 95% task-bootstrap CI
  [{mean_only['task_bootstrap_ci_low']:+.6f},
  {mean_only['task_bootstrap_ci_high']:+.6f}], negative tasks
  {int(mean_only['negative_task_count'])}/{int(mean_only['n_tasks'])},
  negative rows {int(mean_only['negative_row_count'])}/{int(mean_only['n_rows'])},
  hybrid selections {int(mean_only['hybrid_selection_count'])}/{int(mean_only['n_rows'])}.

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

The task-cluster bootstrap uses NumPy `default_rng({BOOTSTRAP_SEED})` with
{BOOTSTRAP_REPS:,} resamples.  Candidate ranking is deterministic and fully
specified in the script.
"""
    (OUT_DIR / "README.md").write_text(readme, encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"Wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
