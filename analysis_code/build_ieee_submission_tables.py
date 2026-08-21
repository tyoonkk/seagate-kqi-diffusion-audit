#!/usr/bin/env python
"""Generate source-checked LaTeX tables and machine-readable IEEE extracts."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "experiments/seagate_kqi/artifacts/condition_aware_main"
PAPER = ROOT / "experiments/seagate_kqi/paper/ieee_access_revision_2026"
MACHINE = PAPER / "machine_readable"
BASELINE = BASE / "ieee_conventional_baseline_audit_run1"
LOTO = BASE / "ieee_v10_uniform_grid_loto_audit_run1"
NESTED = BASE / "ieee_task_independent_nested_audit_run1"
V15 = BASE / "v15_final_combined_selector_report_run1"
V16 = BASE / "v16_selector_promotion_audit_run1"
V161 = BASE / "v161_task2_task3_task10_promotion_audit_run1"
V165 = BASE / "v165_task0_temporal_backtest_audit_run1"
V166 = BASE / "v166_ts1_frozen_transfer_audit_run1"
SECOM = Path(
    r"C:\Users\taeyoon\Desktop\pythonproject_tabddpm-secom\experiments\secom_external\artifacts\external_selector_v11_blind_fresh_summary_322_381_run1"
)

METHOD_ORDER = [
    "unweighted",
    "class_weighted",
    "random_over",
    "random_under_10to1",
    "smote",
    "borderline_smote",
    "adasyn",
    "validation_selected",
]
METHOD_LABEL = {
    "unweighted": "Unweighted",
    "class_weighted": "Class weighted",
    "random_over": "Random oversampling",
    "random_under_10to1": "Random undersampling (10:1)",
    "smote": "SMOTE",
    "borderline_smote": "Borderline-SMOTE",
    "adasyn": "ADASYN",
    "validation_selected": "Validation selected",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def close(actual: float, expected: float, atol: float = 5e-7) -> bool:
    return bool(np.isclose(float(actual), float(expected), atol=atol, rtol=0.0))


def signed(value: float, digits: int = 4) -> str:
    return f"{value:+.{digits}f}"


def build_main_baseline_table(overall: pd.DataFrame) -> str:
    values: dict[tuple[str, str], pd.Series] = {}
    for row in overall.itertuples(index=False):
        values[(str(row.classifier), str(row.method))] = pd.Series(row._asdict())
    unweighted = {
        clf: float(values[(clf, "unweighted")]["task_macro_test_PR_AUC"])
        for clf in ("lightgbm", "xgboost")
    }
    selected_notes = []
    classifier_label = {"lightgbm": "LightGBM", "xgboost": "XGBoost"}
    lines = [
        r"\begin{table*}[!t]",
        r"\centering",
        r"\caption{Fixed-budget conventional audit across all 11 endpoints and five seeds. Values are task-macro test PR-AUC; $\Delta_0$ is the difference from the same classifier's unweighted result. The final row selects a method using validation PR-AUC within each task--seed.}",
        r"\label{tab:baseline}",
        r"\small",
        r"\begin{tabular}{l rr rr}",
        r"\toprule",
        r"& \multicolumn{2}{c}{LightGBM} & \multicolumn{2}{c}{XGBoost} \\",
        r"Method & PR-AUC & $\Delta_0$ & PR-AUC & $\Delta_0$ \\",
        r"\midrule",
    ]
    for method in METHOD_ORDER:
        parts = []
        for clf in ("lightgbm", "xgboost"):
            row = values[(clf, method)]
            score = float(row["task_macro_test_PR_AUC"])
            parts.extend([f"{score:.4f}", signed(score - unweighted[clf], 4)])
            if method == "validation_selected":
                selected_notes.append(
                    f"{classifier_label[clf]} {float(row['task_bootstrap_ci_low']):.4f}--"
                    f"{float(row['task_bootstrap_ci_high']):.4f}"
                )
        label = METHOD_LABEL[method]
        if method == "validation_selected":
            lines.append(r"\midrule")
            lines.append(
                r"\textbf{" + label + r"} & "
                + " & ".join(r"\textbf{" + value + "}" for value in parts)
                + r" \\"
            )
        else:
            lines.append(label + " & " + " & ".join(parts) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\vspace{2pt}",
            r"\begin{minipage}{0.96\textwidth}\footnotesize Validation-selected 95\% task-bootstrap intervals: "
            + "; ".join(selected_notes)
            + r". The audit compares conventional methods across classifiers; archived diffusion candidates were not rerun with XGBoost.\end{minipage}",
            r"\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def build_supplement_endpoint_table(
    task_summary: pd.DataFrame, selected: pd.DataFrame
) -> str:
    unweighted = task_summary[task_summary["method"].eq("unweighted")][
        ["classifier", "task_id", "mean_test_PR_AUC"]
    ].rename(columns={"mean_test_PR_AUC": "unweighted"})
    selected_task = (
        selected.groupby(["classifier", "task_id"], as_index=False)
        .agg(selected=("test_PR_AUC", "mean"))
    )
    merged = unweighted.merge(selected_task, on=["classifier", "task_id"], validate="one_to_one")
    pivot: dict[tuple[str, int], pd.Series] = {
        (str(row.classifier), int(row.task_id)): pd.Series(row._asdict())
        for row in merged.itertuples(index=False)
    }
    lines = [
        r"\begin{table}[!ht]",
        r"\centering",
        r"\caption{Endpoint-level unweighted and validation-selected conventional test PR-AUC, averaged over five seeds.}",
        r"\label{tab:s_baseline_endpoint}",
        r"\small",
        r"\begin{tabular}{c rrr rrr}",
        r"\toprule",
        r"& \multicolumn{3}{c}{LightGBM} & \multicolumn{3}{c}{XGBoost} \\",
        r"Task & Unweighted & Selected & $\Delta$ & Unweighted & Selected & $\Delta$ \\",
        r"\midrule",
    ]
    for task in range(11):
        cells = []
        for clf in ("lightgbm", "xgboost"):
            row = pivot[(clf, task)]
            u = float(row["unweighted"])
            s = float(row["selected"])
            cells.extend([f"{u:.6f}", f"{s:.6f}", signed(s - u, 6)])
        lines.append(str(task) + " & " + " & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def main() -> None:
    PAPER.mkdir(parents=True, exist_ok=True)
    MACHINE.mkdir(parents=True, exist_ok=True)

    overall_path = require(BASELINE / "overall_summary.csv")
    rows_path = require(BASELINE / "baseline_rows.csv")
    task_summary_path = require(BASELINE / "method_task_summary.csv")
    selected_path = require(BASELINE / "validation_selected_rows.csv")
    overall = pd.read_csv(overall_path)
    rows = pd.read_csv(rows_path, low_memory=False)
    task_summary = pd.read_csv(task_summary_path)
    selected = pd.read_csv(selected_path, low_memory=False)

    if len(rows) != 770 or rows.duplicated(["classifier", "task_id", "seed", "method"]).any():
        raise ValueError("Combined conventional baseline row audit failed")
    if not rows["status"].eq("ok").all() or len(selected) != 110:
        raise ValueError("Conventional baseline contains failures or missing selections")

    (PAPER / "generated_baseline_table.tex").write_text(
        build_main_baseline_table(overall), encoding="utf-8"
    )
    (PAPER / "generated_baseline_endpoint_table.tex").write_text(
        build_supplement_endpoint_table(task_summary, selected), encoding="utf-8"
    )

    loto_overall_path = require(LOTO / "outer_loto_overall_summary.csv")
    loto_cases_path = require(LOTO / "outer_loto_case_results.csv")
    loto_tasks_path = require(LOTO / "outer_loto_summary_by_task.csv")
    nested_summary_path = require(NESTED / "summary.csv")
    v15_overall_path = require(V15 / "v15_final_combined_overall_summary.csv")
    v15_rows_path = require(V15 / "v15_final_combined_selected_rows.csv")
    v16_candidates_path = require(V16 / "v16_selector_promotion_candidates.csv")
    v161_summary_path = require(V161 / "v161_combined_summary.csv")
    v165_summary_path = require(V165 / "v165_overall_summary.csv")
    v166_summary_path = require(V166 / "v166_macro_summary.csv")
    secom_summary_path = require(SECOM / "v11_fresh_pooled_summary.csv")
    secom_cases_path = require(SECOM / "v11_fresh_case_results.csv")

    loto = pd.read_csv(loto_overall_path).iloc[0]
    nested = pd.read_csv(nested_summary_path).set_index("regime")
    v15_overall = pd.read_csv(v15_overall_path)
    v15 = v15_overall[v15_overall["selector"].eq("v15_final_combined")].iloc[0]
    v15_rows = pd.read_csv(v15_rows_path)
    v15_selected = int(v15_rows["method"].ne("safe_baseline").sum())
    v16 = pd.read_csv(v16_candidates_path).set_index("policy_name").loc[
        "v16_locked_task2_task10"
    ]
    v165 = pd.read_csv(v165_summary_path).iloc[0]
    v166 = pd.read_csv(v166_summary_path).iloc[0]
    secom_cases = pd.read_csv(secom_cases_path)

    checks: list[dict[str, object]] = []

    def add_check(name: str, actual: object, expected: object, passed: bool) -> None:
        checks.append(
            {"name": name, "actual": actual, "expected": expected, "passed": bool(passed)}
        )

    add_check("baseline_rows", len(rows), 770, len(rows) == 770)
    add_check("baseline_failed_rows", int(rows["status"].ne("ok").sum()), 0, rows["status"].eq("ok").all())
    add_check("loto_rows", int(loto["n_rows"]), 45, int(loto["n_rows"]) == 45)
    add_check("loto_selected", int(loto["selected_count"]), 11, int(loto["selected_count"]) == 11)
    add_check("loto_negative", int(loto["harm_row_count"]), 0, int(loto["harm_row_count"]) == 0)
    add_check("loto_mean", float(loto["task_macro_mean_delta"]), 0.008579306689453232, close(loto["task_macro_mean_delta"], 0.008579306689453232))
    add_check("loto_ci_low", float(loto["task_bootstrap_ci_low"]), 0.0011522156885808799, close(loto["task_bootstrap_ci_low"], 0.0011522156885808799))
    add_check("loto_ci_high", float(loto["task_bootstrap_ci_high"]), 0.017543314565515247, close(loto["task_bootstrap_ci_high"], 0.017543314565515247))
    add_check("nested_conservative_selected", int(nested.loc["conservative", "hybrid_selection_count"]), 0, int(nested.loc["conservative", "hybrid_selection_count"]) == 0)
    add_check("nested_mean_only_selected", int(nested.loc["mean_only", "hybrid_selection_count"]), 7, int(nested.loc["mean_only", "hybrid_selection_count"]) == 7)
    add_check("nested_mean_only_negative", int(nested.loc["mean_only", "negative_row_count"]), 5, int(nested.loc["mean_only", "negative_row_count"]) == 5)
    add_check("nested_mean_only_mean", float(nested.loc["mean_only", "task_macro_mean_delta"]), -0.0020867161789478116, close(nested.loc["mean_only", "task_macro_mean_delta"], -0.0020867161789478116))
    add_check("v15_selected", v15_selected, 16, v15_selected == 16)
    add_check("v15_mean", float(v15["mean_delta_vs_v12_safe"]), 0.02263936514973358, close(v15["mean_delta_vs_v12_safe"], 0.02263936514973358))
    add_check("v15_negative", int(v15["harm_vs_v12_safe_count"]), 0, int(v15["harm_vs_v12_safe_count"]) == 0)
    add_check("v16_selected", int(v16["fresh_selected_count"]), 28, int(v16["fresh_selected_count"]) == 28)
    add_check("v16_mean", float(v16["fresh_mean_delta_vs_safe"]), 0.04228293082156776, close(v16["fresh_mean_delta_vs_safe"], 0.04228293082156776))
    add_check("v165_negative", int(v165["policy_harm_count"]), 2, int(v165["policy_harm_count"]) == 2)
    add_check("v165_mean", float(v165["mean_policy_delta_PR_AUC"]), -0.006550431635838083, close(v165["mean_policy_delta_PR_AUC"], -0.006550431635838083))
    add_check("v166_selected", int(v166["accepted_count"]), 1, int(v166["accepted_count"]) == 1)
    add_check("v166_mean", float(v166["macro_mean_policy_delta_PR_AUC"]), 0.011482883411032051, close(v166["macro_mean_policy_delta_PR_AUC"], 0.011482883411032051))
    add_check("secom_rows", len(secom_cases), 60, len(secom_cases) == 60)
    add_check("secom_selected", int(secom_cases["selected_kind"].eq("hybrid").sum()), 2, int(secom_cases["selected_kind"].eq("hybrid").sum()) == 2)
    delta_column = "delta_vs_safe_baseline_test"
    add_check("secom_negative", int(secom_cases[delta_column].lt(0).sum()), 1, int(secom_cases[delta_column].lt(0).sum()) == 1)
    add_check("secom_mean", float(secom_cases[delta_column].mean()), -0.0003938214394657333, close(secom_cases[delta_column].mean(), -0.0003938214394657333))

    failed_checks = [row for row in checks if not row["passed"]]
    if failed_checks:
        raise ValueError(f"Claim verification failed: {failed_checks}")
    (PAPER / "claim_verification.json").write_text(
        json.dumps({"all_passed": True, "checks": checks}, indent=2), encoding="utf-8"
    )

    sources = {
        "baseline_overall_summary.csv": overall_path,
        "baseline_validation_selected_rows.csv": selected_path,
        "baseline_validation_selection_frequency.csv": require(BASELINE / "validation_selection_frequency.csv"),
        "baseline_method_task_summary.csv": task_summary_path,
        "baseline_task_eligibility.csv": require(BASELINE / "task_eligibility.csv"),
        "loto_overall_summary.csv": loto_overall_path,
        "loto_task_summary.csv": loto_tasks_path,
        "loto_case_results.csv": loto_cases_path,
        "loto_gate_manifest.json": require(LOTO / "fixed_general_gate_manifest.json"),
        "loto_test_exposure_ledger.csv": require(LOTO / "test_exposure_ledger.csv"),
        "nested_summary.csv": nested_summary_path,
        "nested_task_summary.csv": require(NESTED / "task_summary.csv"),
        "nested_heldout_rows.csv": require(NESTED / "outer_heldout_rows.csv"),
        "v15_overall_summary.csv": v15_overall_path,
        "v15_task_summary.csv": require(V15 / "v15_final_combined_summary_by_task.csv"),
        "v16_task2_task10_candidates.csv": v16_candidates_path,
        "v161_combined_summary.csv": v161_summary_path,
        "v165_overall_summary.csv": v165_summary_path,
        "v166_macro_summary.csv": v166_summary_path,
        "secom_pooled_summary.csv": secom_summary_path,
        "secom_case_results.csv": secom_cases_path,
    }
    for target_name, source in sources.items():
        shutil.copy2(source, MACHINE / target_name)
    shutil.copy2(PAPER / "claim_verification.json", MACHINE / "claim_verification.json")

    manifest = {
        "source_sha256": {str(path): sha256(path) for path in sources.values()},
        "submission_extract_sha256": {
            path.name: sha256(path) for path in sorted(MACHINE.iterdir()) if path.is_file()
        },
    }
    (MACHINE / "SHA256_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print("All manuscript claim checks passed")
    print(overall.to_string(index=False))
    print(f"Wrote {PAPER / 'generated_baseline_table.tex'}")
    print(f"Wrote {PAPER / 'generated_baseline_endpoint_table.tex'}")
    print(f"Wrote {MACHINE}")


if __name__ == "__main__":
    main()
