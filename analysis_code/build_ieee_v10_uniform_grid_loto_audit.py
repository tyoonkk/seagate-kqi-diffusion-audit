#!/usr/bin/env python
"""Uniform-grid, task-ID-free gate audit for the IEEE revision.

The V10 experiment evaluated the same 24 validation candidates for every
archived task/seed (six augmentation ratios by four keep rates).  One hybrid
candidate per task/seed was fixed by validation PR-AUC.  This script applies
the five legacy V11/V12 task-ID-free gates to those 45 fixed cases, verifies
the historical output, and performs an outer leave-one-task-out (LOTO) gate
selection sensitivity analysis.

The gate family was historically developed with exposure to archived test
outcomes.  Consequently, the LOTO result is explicitly labelled retrospective
and is not treated as prospective confirmation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "experiments/seagate_kqi/artifacts/condition_aware_main"
CASE_PATH = (
    BASE
    / "selector_v12_final_validation_report_run1/"
    "selector_v12_final_candidates_with_task_context.csv"
)
LEGACY_RESULT_PATH = (
    BASE
    / "selector_v12_final_validation_report_run1/"
    "selector_v12_final_selection_results.csv"
)
GRID_PATHS = (
    BASE
    / "task10_full_summarystats_v10_s1plus_featurewise_quantile_run1/"
    "task10_v10_s1plus_featurewise_candidate_val_grid.csv",
    BASE
    / "task0_task2_full_summarystats_v10_s1plus_featurewise_quantile_run1/"
    "task0_task2_v10_s1plus_featurewise_candidate_val_grid.csv",
    BASE
    / "task1_task7_full_summarystats_v10_s1plus_featurewise_quantile_run1/"
    "task1_task7_v10_s1plus_featurewise_candidate_val_grid.csv",
    BASE
    / "task3_task5_full_summarystats_v10_s1plus_featurewise_quantile_run1/"
    "task3_task5_v10_s1plus_featurewise_candidate_val_grid.csv",
    BASE
    / "task6_task9_full_summarystats_v10_s1plus_featurewise_quantile_run1/"
    "task6_task9_v10_s1plus_featurewise_candidate_val_grid.csv",
)
OUT_DIR = BASE / "ieee_v10_uniform_grid_loto_audit_run1"
EXPECTED_TASKS = (0, 1, 2, 3, 5, 6, 7, 9, 10)
EXPECTED_SEEDS = (42, 43, 44, 45, 46)
EXPECTED_GRID_PER_CASE = 24
BOOTSTRAP_SEED = 20260714
BOOTSTRAP_REPS = 10_000


@dataclass(frozen=True)
class GateDefinition:
    name: str
    formula: str
    historical_role: str


GATES = (
    GateDefinition("safe_only", "False", "reference"),
    GateDefinition("strict_certificate", "strict", "legacy comparator"),
    GateDefinition(
        "v11_grid_or_profile_strict",
        "strict and moderate",
        "legacy conservative gate",
    ),
    GateDefinition(
        "v11_rare_signal_aggressive_else_strict",
        "broad if rare else (strict and moderate)",
        "frozen before tasks 3/5/6/9",
    ),
    GateDefinition(
        "v12_signal_aggressive_else_strict",
        "broad if opportunity else (strict and moderate)",
        "expanded after tasks 3/5; frozen before tasks 6/9",
    ),
)

EXCLUDED_PREDICTOR_PATTERNS = (
    "task_id (grouping key only)",
    "seed (grouping key only)",
    "test_*",
    "safe_baseline_test_*",
    "selected_test_*",
    "delta_vs_*",
    "lift_vs_*",
    "harm*",
    "worse_*",
    "oracle*",
    "pos_rate_test",
    "n_test",
    "test_pos",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = [CASE_PATH, LEGACY_RESULT_PATH, *GRID_PATHS]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing inputs: {missing}")
    cases = pd.read_csv(CASE_PATH, low_memory=False)
    legacy = pd.read_csv(LEGACY_RESULT_PATH, low_memory=False)
    grids = []
    for path in GRID_PATHS:
        frame = pd.read_csv(path, low_memory=False)
        frame["input_file"] = str(path.relative_to(ROOT))
        grids.append(frame)
    return cases, legacy, pd.concat(grids, ignore_index=True)


def audit_grid(grid: pd.DataFrame) -> pd.DataFrame:
    tasks = tuple(sorted(grid["task_id"].astype(int).unique()))
    seeds = tuple(sorted(grid["seed"].astype(int).unique()))
    if tasks != EXPECTED_TASKS or seeds != EXPECTED_SEEDS:
        raise ValueError(f"Unexpected tasks/seeds: {tasks}, {seeds}")
    coverage = (
        grid.groupby(["task_id", "seed"], as_index=False)
        .agg(
            grid_count=("ratio", "size"),
            ratio_count=("ratio", "nunique"),
            keep_rate_count=("s3_ig_keep_rate", "nunique"),
            methods=("method", lambda x: ",".join(sorted(set(map(str, x))))),
            modes=("ddpm_train_mode", lambda x: ",".join(sorted(set(map(str, x))))),
            postprocess=("s3_postprocess", lambda x: ",".join(sorted(set(map(str, x))))),
        )
        .sort_values(["task_id", "seed"])
    )
    if not coverage["grid_count"].eq(EXPECTED_GRID_PER_CASE).all():
        raise ValueError("The validation budget is not 24 candidates for every case")
    if not coverage["ratio_count"].eq(6).all() or not coverage["keep_rate_count"].eq(4).all():
        raise ValueError("The grid is not the expected 6 x 4 design")
    return coverage


def derive_validation_signals(cases: pd.DataFrame) -> pd.DataFrame:
    out = cases.copy()
    # These task descriptors use train/validation data only.
    out["derived_rare"] = out["pos_rate_train"].le(0.015) & out[
        "delta_S1_minus_S0"
    ].ge(0.030)
    out["derived_strong"] = out["S0_val_PR_AUC_mean"].ge(0.100) & out[
        "delta_S1_minus_S0"
    ].ge(0.030)
    out["derived_opportunity"] = out["derived_rare"] | out["derived_strong"]
    out["derived_moderate"] = out["grid_mean_val_gain"].ge(0.0) | out[
        "delta_S1_minus_S0"
    ].ge(0.020)
    out["derived_risk"] = out["delta_S1_minus_S0"].lt(0.020) & out[
        "grid_mean_val_gain"
    ].lt(0.0)

    out["derived_strict"] = (
        out["val_gain"].ge(0.005)
        & out["val_fpr_excess"].le(0.0)
        & out["val_recall_fpr010_gain"].ge(0.0)
        & out["tstr_pr_auc"].ge(0.040)
        & out["certificate_red_flags_v9"].eq(0)
    )
    out["derived_broad"] = (
        out["method"].eq("s1_plus_ig")
        & out["ddpm_train_mode"].eq("positive_only")
        & out["val_gain"].ge(-0.060)
        & out["val_fpr_excess"].le(0.020)
        & out["val_recall_fpr010_gain"].ge(-0.050)
        & out["tstr_pr_auc"].ge(0.008)
        & out["certificate_score_v9"].ge(-3.0)
        & out["certificate_red_flags_v9"].le(1)
    )
    return out


def gate_decision(frame: pd.DataFrame, gate: str) -> pd.Series:
    if gate == "safe_only":
        return pd.Series(False, index=frame.index)
    if gate == "strict_certificate":
        return frame["derived_strict"]
    if gate == "v11_grid_or_profile_strict":
        return frame["derived_strict"] & frame["derived_moderate"]
    if gate == "v11_rare_signal_aggressive_else_strict":
        return np.where(
            frame["derived_rare"],
            frame["derived_broad"],
            frame["derived_strict"] & frame["derived_moderate"],
        )
    if gate == "v12_signal_aggressive_else_strict":
        return np.where(
            frame["derived_opportunity"],
            frame["derived_broad"],
            frame["derived_strict"] & frame["derived_moderate"],
        )
    raise ValueError(gate)


def apply_gate(cases: pd.DataFrame, gate: str) -> pd.DataFrame:
    selected = np.asarray(gate_decision(cases, gate), dtype=bool)
    out = pd.DataFrame(
        {
            "rule_name": gate,
            "task_id": cases["task_id"].astype(int),
            "seed": cases["seed"].astype(int),
            "validation_stage": cases["validation_stage"],
            "selected_kind": np.where(selected, "hybrid", "safe_baseline"),
            "selected_reason": np.where(selected, "gate_pass", "reference_fallback"),
            "selected_candidate_key": np.where(
                selected,
                cases["method"].astype(str)
                + "/mode="
                + cases["ddpm_train_mode"].astype(str)
                + "/ratio="
                + cases["ratio"].astype(str)
                + "/keep="
                + cases["s3_ig_keep_rate"].astype(str),
                "safe_baseline",
            ),
            "selected_test_PR_AUC": np.where(
                selected, cases["test_PR_AUC"], cases["safe_baseline_test_PR_AUC"]
            ),
            "safe_baseline_test_PR_AUC": cases["safe_baseline_test_PR_AUC"],
            "delta_vs_safe_baseline_test": np.where(
                selected, cases["delta_vs_safe_baseline_test"], 0.0
            ),
        }
    )
    out["harm_flag"] = out["delta_vs_safe_baseline_test"].lt(0.0)
    return out


def verify_legacy(recomputed: pd.DataFrame, legacy: pd.DataFrame) -> None:
    keys = ["rule_name", "task_id", "seed"]
    merged = recomputed.merge(
        legacy[keys + ["selected_kind", "delta_vs_safe_baseline_test"]],
        on=keys,
        how="outer",
        suffixes=("_new", "_legacy"),
        indicator=True,
    )
    kind_ok = merged["selected_kind_new"].eq(merged["selected_kind_legacy"])
    delta_ok = np.isclose(
        merged["delta_vs_safe_baseline_test_new"],
        merged["delta_vs_safe_baseline_test_legacy"],
        atol=1e-12,
        rtol=0.0,
        equal_nan=True,
    )
    if not merged["_merge"].eq("both").all() or not kind_ok.all() or not delta_ok.all():
        bad = merged[~(merged["_merge"].eq("both") & kind_ok & delta_ok)]
        raise ValueError(f"Legacy reproduction failed:\n{bad}")


def summarize(frame: pd.DataFrame) -> dict[str, object]:
    task_delta = frame.groupby("task_id")["delta_vs_safe_baseline_test"].mean()
    return {
        "n_tasks": int(task_delta.size),
        "n_rows": int(len(frame)),
        "task_macro_mean_delta": float(task_delta.mean()),
        "task_median_delta": float(task_delta.median()),
        "minimum_task_mean_delta": float(task_delta.min()),
        "negative_task_count": int(task_delta.lt(0.0).sum()),
        "harm_row_count": int(frame["harm_flag"].sum()),
        "selected_count": int(frame["selected_kind"].eq("hybrid").sum()),
        "coverage": float(frame["selected_kind"].eq("hybrid").mean()),
    }


def choose_rule(train_scores: pd.DataFrame) -> str:
    # Deterministic retrospective selection: minimize negative rows first,
    # then maximize task-macro gain, minimum task gain, and coverage.
    ranked = train_scores.sort_values(
        [
            "harm_row_count",
            "task_macro_mean_delta",
            "minimum_task_mean_delta",
            "selected_count",
            "rule_name",
        ],
        ascending=[True, False, False, False, True],
        kind="mergesort",
    )
    return str(ranked.iloc[0]["rule_name"])


def task_bootstrap(frame: pd.DataFrame) -> tuple[float, float]:
    values = frame.groupby("task_id")["delta_vs_safe_baseline_test"].mean().to_numpy()
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_REPS, len(values)))
    draws = values[indices].mean(axis=1)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return float(lo), float(hi)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cases, legacy, grid = load_inputs()
    coverage = audit_grid(grid)
    coverage.to_csv(OUT_DIR / "candidate_coverage_matrix.csv", index=False)

    cases = derive_validation_signals(cases)
    rule_frames = {gate.name: apply_gate(cases, gate.name) for gate in GATES}
    recomputed = pd.concat(rule_frames.values(), ignore_index=True)
    verify_legacy(recomputed, legacy)
    recomputed.to_csv(OUT_DIR / "legacy_rule_case_results_recomputed.csv", index=False)

    legacy_summary_rows = []
    for gate in GATES:
        legacy_summary_rows.append({"rule_name": gate.name, **summarize(rule_frames[gate.name])})
    legacy_summary = pd.DataFrame(legacy_summary_rows)
    legacy_summary.to_csv(OUT_DIR / "legacy_rule_summary.csv", index=False)

    inner_rows: list[dict[str, object]] = []
    choice_rows: list[dict[str, object]] = []
    heldout_frames: list[pd.DataFrame] = []
    for heldout_task in EXPECTED_TASKS:
        scores = []
        for gate in GATES:
            train = rule_frames[gate.name][rule_frames[gate.name]["task_id"].ne(heldout_task)]
            row = {
                "heldout_task_id": heldout_task,
                "rule_name": gate.name,
                **summarize(train),
            }
            scores.append(row)
            inner_rows.append(row)
        score_frame = pd.DataFrame(scores)
        chosen = choose_rule(score_frame)
        heldout = rule_frames[chosen][rule_frames[chosen]["task_id"].eq(heldout_task)].copy()
        heldout.insert(0, "heldout_task_id", heldout_task)
        heldout.insert(1, "chosen_rule", chosen)
        heldout_frames.append(heldout)
        choice_rows.append(
            {
                "heldout_task_id": heldout_task,
                "chosen_rule": chosen,
                **{f"heldout_{key}": value for key, value in summarize(heldout).items()},
            }
        )

    inner = pd.DataFrame(inner_rows)
    choices = pd.DataFrame(choice_rows)
    heldout = pd.concat(heldout_frames, ignore_index=True)
    inner.to_csv(OUT_DIR / "outer_development_rule_scores.csv", index=False)
    choices.to_csv(OUT_DIR / "outer_loto_rule_choices.csv", index=False)
    heldout.to_csv(OUT_DIR / "outer_loto_case_results.csv", index=False)

    task_summary = (
        heldout.groupby("task_id", as_index=False)
        .agg(
            chosen_rule=("chosen_rule", "first"),
            validation_stage=("validation_stage", "first"),
            n=("seed", "size"),
            selected_count=("selected_kind", lambda x: int(x.eq("hybrid").sum())),
            harm_count=("harm_flag", "sum"),
            mean_delta=("delta_vs_safe_baseline_test", "mean"),
            median_delta=("delta_vs_safe_baseline_test", "median"),
            minimum_delta=("delta_vs_safe_baseline_test", "min"),
        )
    )
    task_summary.to_csv(OUT_DIR / "outer_loto_summary_by_task.csv", index=False)

    lo, hi = task_bootstrap(heldout)
    overall = pd.DataFrame(
        [
            {
                "analysis": "retrospective_outer_leave_one_task_out",
                **summarize(heldout),
                "task_bootstrap_ci_low": lo,
                "task_bootstrap_ci_high": hi,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "bootstrap_reps": BOOTSTRAP_REPS,
            }
        ]
    )
    overall.to_csv(OUT_DIR / "outer_loto_overall_summary.csv", index=False)

    chronology = pd.DataFrame(
        [
            {
                "phase": "V11 gate development",
                "tasks": "0,1,2,7,10",
                "role": "retrospective development",
                "test_exposure": "yes",
            },
            {
                "phase": "V11 forward application",
                "tasks": "3,5",
                "role": "then-unseen expansion",
                "test_exposure": "after V11 freeze",
            },
            {
                "phase": "V12 gate development",
                "tasks": "0,1,2,3,5,7,10",
                "role": "retrospective expansion",
                "test_exposure": "yes",
            },
            {
                "phase": "V12 final holdout",
                "tasks": "6,9",
                "role": "then-unseen abstention check",
                "test_exposure": "after V12 freeze",
            },
            {
                "phase": "V13-V15 development",
                "tasks": "2,3,10 emphasis",
                "role": "task-specific retrospective refinement",
                "test_exposure": "yes",
            },
        ]
    )
    chronology.to_csv(OUT_DIR / "test_exposure_ledger.csv", index=False)

    manifest = {
        "analysis_label": "retrospective task-ID-free gate sensitivity",
        "gate_selection_order": [
            "minimum harm_row_count",
            "maximum task_macro_mean_delta",
            "maximum minimum_task_mean_delta",
            "maximum selected_count",
            "lexical rule_name",
        ],
        "gate_definitions": [asdict(gate) for gate in GATES],
        "input_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in [CASE_PATH, LEGACY_RESULT_PATH, *GRID_PATHS]},
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_reps": BOOTSTRAP_REPS,
    }
    (OUT_DIR / "fixed_general_gate_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    (OUT_DIR / "excluded_columns_and_leakage_manifest.json").write_text(
        json.dumps(
            {
                "excluded_predictor_patterns": EXCLUDED_PREDICTOR_PATTERNS,
                "historical_contamination": (
                    "The fixed legacy gate family was developed after archived test exposure; "
                    "LOTO is therefore a retrospective sensitivity analysis."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    result = overall.iloc[0]
    readme = f"""# IEEE V10 Uniform-Grid LOTO Audit Run 1

Date: {date.today().isoformat()}

## Design

All {len(EXPECTED_TASKS)} archived tasks and {len(EXPECTED_SEEDS)} seeds use the
same V10 validation budget: six augmentation ratios by four keep rates, or
{EXPECTED_GRID_PER_CASE} candidates per task-seed.  A candidate is fixed by
validation PR-AUC before applying a task-ID-free V11/V12 gate.

For each outer held-out task, the gate is selected on the other eight tasks by
the deterministic order recorded in `fixed_general_gate_manifest.json`.

## Result

- selected hybrids: {int(result['selected_count'])}/{int(result['n_rows'])}
- negative test PR-AUC changes: {int(result['harm_row_count'])}/{int(result['n_rows'])}
- task-macro mean delta: {result['task_macro_mean_delta']:+.6f}
- 95% task-cluster bootstrap CI: [{result['task_bootstrap_ci_low']:+.6f},
  {result['task_bootstrap_ci_high']:+.6f}]
- minimum task mean delta: {result['minimum_task_mean_delta']:+.6f}

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
"""
    (OUT_DIR / "README.md").write_text(readme, encoding="utf-8")
    print(overall.to_string(index=False))
    print(f"Wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
