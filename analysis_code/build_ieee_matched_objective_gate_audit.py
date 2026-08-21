#!/usr/bin/env python
"""Matched-objective, matched-family gate audit on the equal-budget archive.

Frozen pre-analysis specification: MATCHED_OBJECTIVE_GATE_AUDIT_SPEC.md
(SHA-256 recorded in the output manifest).  This script decomposes the
difference between the primary task-independent nested audit (V9 common
candidates) and the secondary equal-budget LOTO audit (V10 uniform grid)
along two of the three differing axes: gate family and selection objective,
both evaluated on the SAME V10 45-case archive.

Six arms are always reported (no selective reporting):
  {legacy5, nested7_veto} x {conservative, mean_only, harm_first}

The candidate-level nested audit cannot be reproduced on this archive
because only the validation-best hybrid of each 24-candidate grid has an
archived test outcome; the nested gates are therefore applied as case-level
VETO rules (deploy the fixed hybrid iff it passes), which is strictly more
conservative than the original select-best-passing semantics.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
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
LEGACY_SUMMARY_PATH = BASE / "ieee_v10_uniform_grid_loto_audit_run1/legacy_rule_summary.csv"
SPEC_PATH = ROOT / "experiments/seagate_kqi/MATCHED_OBJECTIVE_GATE_AUDIT_SPEC.md"
OUT_DIR = BASE / "ieee_matched_objective_gate_audit_run1"

EXPECTED_TASKS = (0, 1, 2, 3, 5, 6, 7, 9, 10)
EXPECTED_SEEDS = (42, 43, 44, 45, 46)
BOOTSTRAP_SEED = 20260714
BOOTSTRAP_REPS = 10_000

LEGACY_GATES = (
    "safe_only",
    "strict_certificate",
    "v11_grid_or_profile_strict",
    "v11_rare_signal_aggressive_else_strict",
    "v12_signal_aggressive_else_strict",
)


@dataclass(frozen=True)
class NestedGate:
    name: str
    min_val_gain: float | None = None
    max_val_fpr_excess: float | None = None
    positive_only: bool = False
    min_tstr_pr_auc: float | None = None
    pass_all: bool = False
    never: bool = False


NESTED_GATES = (
    NestedGate("reference_only", never=True),
    NestedGate("validation_best", pass_all=True),
    NestedGate("gain0_fpr0", min_val_gain=0.0, max_val_fpr_excess=0.0),
    NestedGate("gain005_fpr0", min_val_gain=0.005, max_val_fpr_excess=0.0),
    NestedGate("gain01_fpr01", min_val_gain=0.01, max_val_fpr_excess=0.01),
    NestedGate(
        "positive_gain0_fpr01",
        min_val_gain=0.0,
        max_val_fpr_excess=0.01,
        positive_only=True,
    ),
    NestedGate(
        "positive_tstr_gain0",
        min_val_gain=0.0,
        max_val_fpr_excess=0.01,
        positive_only=True,
        min_tstr_pr_auc=0.008,
    ),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_cases() -> pd.DataFrame:
    if not CASE_PATH.exists():
        raise FileNotFoundError(CASE_PATH)
    cases = pd.read_csv(CASE_PATH, low_memory=False)
    if len(cases) != 45:
        raise ValueError(f"Expected 45 cases, found {len(cases)}")
    tasks = tuple(sorted(cases["task_id"].astype(int).unique()))
    seeds = tuple(sorted(cases["seed"].astype(int).unique()))
    if tasks != EXPECTED_TASKS or seeds != EXPECTED_SEEDS:
        raise ValueError(f"Unexpected tasks/seeds: {tasks}, {seeds}")
    if cases.duplicated(["task_id", "seed"]).any():
        raise ValueError("Duplicate task-seed cases found")
    return cases.sort_values(["task_id", "seed"]).reset_index(drop=True)


# --- legacy5 gate logic, copied verbatim from build_ieee_v10_uniform_grid_loto_audit.py ---

def derive_validation_signals(cases: pd.DataFrame) -> pd.DataFrame:
    out = cases.copy()
    out["derived_rare"] = out["pos_rate_train"].le(0.015) & out["delta_S1_minus_S0"].ge(0.030)
    out["derived_strong"] = out["S0_val_PR_AUC_mean"].ge(0.100) & out["delta_S1_minus_S0"].ge(0.030)
    out["derived_opportunity"] = out["derived_rare"] | out["derived_strong"]
    out["derived_moderate"] = out["grid_mean_val_gain"].ge(0.0) | out["delta_S1_minus_S0"].ge(0.020)
    out["derived_risk"] = out["delta_S1_minus_S0"].lt(0.020) & out["grid_mean_val_gain"].lt(0.0)
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


def legacy_gate_decision(frame: pd.DataFrame, gate: str) -> pd.Series:
    if gate == "safe_only":
        return pd.Series(False, index=frame.index)
    if gate == "strict_certificate":
        return frame["derived_strict"]
    if gate == "v11_grid_or_profile_strict":
        return frame["derived_strict"] & frame["derived_moderate"]
    if gate == "v11_rare_signal_aggressive_else_strict":
        return pd.Series(
            np.where(
                frame["derived_rare"],
                frame["derived_broad"],
                frame["derived_strict"] & frame["derived_moderate"],
            ),
            index=frame.index,
        )
    if gate == "v12_signal_aggressive_else_strict":
        return pd.Series(
            np.where(
                frame["derived_opportunity"],
                frame["derived_broad"],
                frame["derived_strict"] & frame["derived_moderate"],
            ),
            index=frame.index,
        )
    raise ValueError(gate)


# --- nested7 gates applied as case-level vetoes on the fixed hybrid ---

def nested_gate_decision(frame: pd.DataFrame, gate: NestedGate) -> pd.Series:
    if gate.never:
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


def apply_case_gate(cases: pd.DataFrame, family: str, gate_name: str, selected: pd.Series) -> pd.DataFrame:
    sel = np.asarray(selected, dtype=bool)
    out = pd.DataFrame(
        {
            "family": family,
            "gate": gate_name,
            "task_id": cases["task_id"].astype(int),
            "seed": cases["seed"].astype(int),
            "selected_kind": np.where(sel, "hybrid", "safe_baseline"),
            "selected_test_PR_AUC": np.where(
                sel, cases["test_PR_AUC"], cases["safe_baseline_test_PR_AUC"]
            ),
            "safe_baseline_test_PR_AUC": cases["safe_baseline_test_PR_AUC"],
            "delta_vs_safe_baseline_test": np.where(
                sel, cases["delta_vs_safe_baseline_test"], 0.0
            ),
        }
    )
    out["negative_row"] = out["delta_vs_safe_baseline_test"].lt(0.0)
    return out


def summarize(frame: pd.DataFrame) -> dict[str, object]:
    task_delta = frame.groupby("task_id")["delta_vs_safe_baseline_test"].mean()
    return {
        "n_tasks": int(task_delta.size),
        "n_rows": int(len(frame)),
        "task_macro_mean_delta": float(task_delta.mean()),
        "task_median_delta": float(task_delta.median()),
        "minimum_task_mean_delta": float(task_delta.min()),
        "negative_task_count": int(task_delta.lt(0.0).sum()),
        "negative_row_count": int(frame["negative_row"].sum()),
        "hybrid_selection_count": int(frame["selected_kind"].eq("hybrid").sum()),
        "coverage": float(frame["selected_kind"].eq("hybrid").mean()),
    }


def choose_gate(scores: pd.DataFrame, objective: str, fallback: str) -> str:
    eligible = scores.copy()
    if objective == "conservative":
        eligible = eligible[
            eligible["negative_task_count"].eq(0) & eligible["negative_row_count"].eq(0)
        ].copy()
        if eligible.empty:
            return fallback
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
    if objective == "mean_only":
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
    if objective == "harm_first":
        eligible = eligible.sort_values(
            [
                "negative_row_count",
                "task_macro_mean_delta",
                "minimum_task_mean_delta",
                "hybrid_selection_count",
                "gate",
            ],
            ascending=[True, False, False, False, True],
            kind="mergesort",
        )
        return str(eligible.iloc[0]["gate"])
    raise ValueError(objective)


def task_bootstrap(frame: pd.DataFrame) -> tuple[float, float]:
    values = frame.groupby("task_id")["delta_vs_safe_baseline_test"].mean().to_numpy()
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_REPS, len(values)))
    draws = values[indices].mean(axis=1)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return float(lo), float(hi)


def verify_against_legacy_summary(gate_frames: dict[str, pd.DataFrame]) -> None:
    """Fail closed unless the legacy5 full-archive recomputation reproduces run1."""
    if not LEGACY_SUMMARY_PATH.exists():
        raise FileNotFoundError(LEGACY_SUMMARY_PATH)
    legacy = pd.read_csv(LEGACY_SUMMARY_PATH).set_index("rule_name")
    for gate_name in LEGACY_GATES:
        got = summarize(gate_frames[gate_name])
        want = legacy.loc[gate_name]
        checks = {
            "task_macro_mean_delta": (got["task_macro_mean_delta"], float(want["task_macro_mean_delta"])),
            "negative_row_count": (got["negative_row_count"], int(want["harm_row_count"])),
            "hybrid_selection_count": (got["hybrid_selection_count"], int(want["selected_count"])),
            "negative_task_count": (got["negative_task_count"], int(want["negative_task_count"])),
        }
        for key, (new, old) in checks.items():
            if isinstance(old, float):
                ok = bool(np.isclose(new, old, atol=1e-12, rtol=0.0))
            else:
                ok = new == old
            if not ok:
                raise ValueError(f"Legacy reproduction failed: {gate_name}.{key} new={new} old={old}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cases = load_cases()
    cases = derive_validation_signals(cases)

    families: dict[str, tuple[dict[str, pd.DataFrame], str]] = {}

    legacy_frames = {
        name: apply_case_gate(cases, "legacy5", name, legacy_gate_decision(cases, name))
        for name in LEGACY_GATES
    }
    verify_against_legacy_summary(legacy_frames)
    families["legacy5"] = (legacy_frames, "safe_only")

    nested_frames = {
        gate.name: apply_case_gate(cases, "nested7_veto", gate.name, nested_gate_decision(cases, gate))
        for gate in NESTED_GATES
    }
    families["nested7_veto"] = (nested_frames, "reference_only")

    full_rows = []
    for family, (frames, _) in families.items():
        for gate_name, frame in frames.items():
            full_rows.append({"family": family, "gate": gate_name, **summarize(frame)})
    full_summary = pd.DataFrame(full_rows)
    full_summary.to_csv(OUT_DIR / "full_archive_gate_summary.csv", index=False)

    objectives = ("conservative", "mean_only", "harm_first")
    dev_rows: list[dict[str, object]] = []
    choice_rows: list[dict[str, object]] = []
    heldout_frames: list[pd.DataFrame] = []
    overall_rows: list[dict[str, object]] = []

    for family, (frames, fallback) in families.items():
        for objective in objectives:
            fold_heldouts: list[pd.DataFrame] = []
            for heldout_task in EXPECTED_TASKS:
                scores = []
                for gate_name, frame in frames.items():
                    train = frame[frame["task_id"].ne(heldout_task)]
                    row = {
                        "family": family,
                        "objective": objective,
                        "heldout_task_id": heldout_task,
                        "gate": gate_name,
                        **summarize(train),
                    }
                    scores.append(row)
                    dev_rows.append(row)
                chosen = choose_gate(pd.DataFrame(scores), objective, fallback)
                heldout = frames[chosen][frames[chosen]["task_id"].eq(heldout_task)].copy()
                heldout.insert(0, "objective", objective)
                heldout.insert(1, "heldout_task_id", heldout_task)
                heldout.insert(2, "chosen_gate", chosen)
                fold_heldouts.append(heldout)
                choice_rows.append(
                    {
                        "family": family,
                        "objective": objective,
                        "heldout_task_id": heldout_task,
                        "chosen_gate": chosen,
                        **{f"heldout_{k}": v for k, v in summarize(heldout).items()},
                    }
                )
            arm = pd.concat(fold_heldouts, ignore_index=True)
            heldout_frames.append(arm)
            lo, hi = task_bootstrap(arm)
            overall_rows.append(
                {
                    "family": family,
                    "objective": objective,
                    **summarize(arm),
                    "task_bootstrap_ci_low": lo,
                    "task_bootstrap_ci_high": hi,
                    "bootstrap_seed": BOOTSTRAP_SEED,
                    "bootstrap_reps": BOOTSTRAP_REPS,
                }
            )

    pd.DataFrame(dev_rows).to_csv(OUT_DIR / "outer_development_gate_scores.csv", index=False)
    pd.DataFrame(choice_rows).to_csv(OUT_DIR / "outer_loto_gate_choices.csv", index=False)
    pd.concat(heldout_frames, ignore_index=True).to_csv(
        OUT_DIR / "outer_loto_case_results.csv", index=False
    )
    overall = pd.DataFrame(overall_rows)
    overall.to_csv(OUT_DIR / "outer_loto_overall_summary.csv", index=False)

    manifest = {
        "audit": "ieee_matched_objective_gate_audit_run1",
        "date": str(date.today()),
        "spec_file": str(SPEC_PATH.relative_to(ROOT)),
        "spec_sha256": sha256(SPEC_PATH),
        "case_input": str(CASE_PATH.relative_to(ROOT)),
        "case_input_sha256": sha256(CASE_PATH),
        "legacy_summary_input": str(LEGACY_SUMMARY_PATH.relative_to(ROOT)),
        "legacy_summary_sha256": sha256(LEGACY_SUMMARY_PATH),
        "legacy_reproduction_check": "passed_fail_closed",
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_reps": BOOTSTRAP_REPS,
        "semantics_note": (
            "nested7 gates applied as case-level vetoes on the fixed "
            "validation-best hybrid; strictly more conservative than the "
            "original select-best-passing-candidate semantics."
        ),
        "evidence_role": (
            "retrospective attribution sensitivity on a historically "
            "test-exposed gate family; not prospective validation; does not "
            "replace the primary nested audit or the secondary LOTO audit."
        ),
    }
    (OUT_DIR / "audit_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    print()
    print(overall.to_string(index=False))


if __name__ == "__main__":
    main()
