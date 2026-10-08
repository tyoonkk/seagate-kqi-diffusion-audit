#!/usr/bin/env python
"""Analyze task10 targeted v9 S1+IG positive-only sweep.

The sweep is performance-oriented: it expands ratio/keep-rate for the one
family that v9 mining identified as promising on task10:

    summary_stats + S3_HYBRID/s1_plus_ig + positive_only DDPM

The strict runner tests only the validation-selected S3_HYBRID winner per seed.
It still stores validation metrics for the whole ratio/keep grid. Therefore this
analysis separates:

- full validation grid diagnostics for all candidates
- test deltas for the validation-selected hybrid winner only
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_RUN_DIR = Path(
    "experiments/seagate_kqi/outputs/paper_strict/"
    "condition_task10_full_summarystats_v9_s1plus_positive_sweep_seed42_46_run1"
)
DEFAULT_OUT_DIR = Path(
    "experiments/seagate_kqi/outputs/condition_aware/hybrid_experiments/"
    "task10_full_summarystats_v9_s1plus_positive_sweep_run1"
)
DEFAULT_ARTIFACT_DIR = Path(
    "experiments/seagate_kqi/artifacts/condition_aware_main/"
    "task10_full_summarystats_v9_s1plus_positive_sweep_run1"
)


def as_float(value: object, default: float = np.nan) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "(empty)"
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(lambda value: f"{value:.6g}")
        else:
            out[col] = out[col].astype(str)
    lines = [
        "| " + " | ".join(out.columns) + " |",
        "| " + " | ".join(["---"] * len(out.columns)) + " |",
    ]
    for _, row in out.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in out.columns) + " |")
    return "\n".join(lines)


def pick_safe_baseline(group: pd.DataFrame) -> pd.Series:
    return group.sort_values(
        ["val_PR_AUC", "val_FPR", "val_MCC", "method"],
        ascending=[False, True, False, True],
    ).iloc[0]


def add_candidate_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["val_gain"] = out["val_PR_AUC"] - out["safe_baseline_val_PR_AUC"]
    out["val_fpr_excess"] = out["val_FPR"] - out["safe_baseline_val_FPR"]
    out["val_recall_fpr010_gain"] = (
        out["val_Recall_at_FPR010"].fillna(0.0)
        - out["safe_baseline_val_Recall_at_FPR010"].fillna(0.0)
    )
    out["delta_vs_safe_baseline_test"] = out["test_PR_AUC"] - out["safe_baseline_test_PR_AUC"]
    out["test_fpr_excess"] = out["test_FPR"] - out["safe_baseline_test_FPR"]
    out["harm_flag"] = out["delta_vs_safe_baseline_test"] < 0.0

    score = np.zeros(len(out), dtype=float)
    score += np.where(out["val_gain"] >= 0.005, 2.0, np.where(out["val_gain"] >= 0.0, 1.0, -1.0))
    score += np.where(out["val_fpr_excess"] <= 0.0, 2.0, np.where(out["val_fpr_excess"] <= 0.005, 1.0, -1.0))
    score += np.where(out["val_recall_fpr010_gain"] >= 0.025, 2.0, np.where(out["val_recall_fpr010_gain"] >= 0.0, 1.0, -0.5))
    score += np.where(out["tstr_pr_auc"].fillna(0.0) >= 0.040, 1.0, -1.0)
    score += np.where(out["ddpm_train_mode"].astype(str) == "positive_only", 1.0, -1.0)
    score += np.where(out["real_vs_synth_auc"].fillna(0.99) <= 0.999, 0.5, -0.5)
    out["certificate_score_v9"] = score

    red_flags = np.zeros(len(out), dtype=int)
    red_flags += np.where(out["val_gain"] < -0.060, 1, 0)
    red_flags += np.where(out["val_fpr_excess"] > 0.020, 1, 0)
    red_flags += np.where(out["val_recall_fpr010_gain"] < -0.050, 1, 0)
    red_flags += np.where(out["tstr_pr_auc"].fillna(0.0) < 0.008, 1, 0)
    red_flags += np.where(out["ddpm_train_mode"].astype(str) != "positive_only", 1, 0)
    out["certificate_red_flags_v9"] = red_flags

    out["pass_v9_opportunity_s1plus_anyrep_fpr020"] = (
        (out["method"].astype(str) == "s1_plus_ig")
        & (out["ddpm_train_mode"].astype(str) == "positive_only")
        & (out["val_gain"] >= -0.060)
        & (out["val_fpr_excess"] <= 0.020)
        & (out["val_recall_fpr010_gain"] >= -0.050)
        & (out["tstr_pr_auc"].fillna(0.0) >= 0.008)
        & (out["certificate_score_v9"] >= -3.0)
        & (out["certificate_red_flags_v9"] <= 1)
    )
    out["pass_strict_cert"] = (
        (out["val_gain"] >= 0.005)
        & (out["val_fpr_excess"] <= 0.0)
        & (out["val_recall_fpr010_gain"] >= 0.0)
        & (out["tstr_pr_auc"].fillna(0.0) >= 0.040)
        & (out["certificate_red_flags_v9"] == 0)
    )
    return out


def load_candidates(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    final_path = run_dir / "final_test_results.csv"
    candidate_path = run_dir / "candidate_val_results.csv"
    if not final_path.exists():
        raise FileNotFoundError(f"Missing final_test_results.csv: {final_path}")
    if not candidate_path.exists():
        raise FileNotFoundError(f"Missing candidate_val_results.csv: {candidate_path}")
    raw = pd.read_csv(final_path)
    candidate_raw = pd.read_csv(candidate_path)
    safe_raw = raw[raw["scenario"].isin(["S0", "S1"])].copy()
    hybrid_raw = raw[raw["scenario"] == "S3_HYBRID"].copy()
    if safe_raw.empty:
        raise ValueError("Missing S0/S1 rows.")
    if hybrid_raw.empty:
        raise ValueError("Missing S3_HYBRID rows.")

    safe_rows = []
    for (task_id, seed), group in safe_raw.groupby(["task_id", "seed"]):
        pick = pick_safe_baseline(group)
        safe_rows.append(
            {
                "task_id": int(task_id),
                "seed": int(seed),
                "safe_baseline_scenario": str(pick["scenario"]),
                "safe_baseline_method": str(pick["method"]),
                "safe_baseline_val_PR_AUC": as_float(pick.get("val_PR_AUC")),
                "safe_baseline_val_FPR": as_float(pick.get("val_FPR")),
                "safe_baseline_val_MCC": as_float(pick.get("val_MCC")),
                "safe_baseline_val_Recall_at_FPR010": as_float(pick.get("val_Recall_at_FPR010")),
                "safe_baseline_test_PR_AUC": as_float(pick.get("PR_AUC")),
                "safe_baseline_test_FPR": as_float(pick.get("FPR")),
                "safe_baseline_test_MCC": as_float(pick.get("MCC")),
                "safe_baseline_test_Recall_at_FPR010": as_float(pick.get("Recall_at_FPR010")),
            }
        )
    safe = pd.DataFrame(safe_rows)

    rows = []
    for _, h in hybrid_raw.iterrows():
        safe_match = safe[
            (safe["task_id"] == int(h["task_id"]))
            & (safe["seed"] == int(h["seed"]))
        ].iloc[0]
        rows.append(
            {
                "source": "full_summarystats_task10_v9_s1plus_positive_sweep",
                "dataset": str(h.get("dataset", "seagate_kqi")),
                "toolset": str(h.get("toolset", "time-series-2")),
                "representation": str(h.get("representation", "summary_stats")),
                "task_id": int(h["task_id"]),
                "seed": int(h["seed"]),
                "scenario": str(h["scenario"]),
                "method": str(h["method"]),
                "ratio": as_float(h.get("ratio")),
                "s3_ig_keep_rate": as_float(h.get("s3_ig_keep_rate")),
                "ddpm_train_mode": str(h.get("ddpm_train_mode", "")),
                "s3_postprocess": str(h.get("s3_postprocess", "none")),
                "val_PR_AUC": as_float(h.get("val_PR_AUC")),
                "val_FPR": as_float(h.get("val_FPR")),
                "val_MCC": as_float(h.get("val_MCC")),
                "val_Recall_at_FPR010": as_float(h.get("val_Recall_at_FPR010")),
                "test_PR_AUC": as_float(h.get("PR_AUC")),
                "test_FPR": as_float(h.get("FPR")),
                "test_MCC": as_float(h.get("MCC")),
                "test_Recall_at_FPR010": as_float(h.get("Recall_at_FPR010")),
                "real_vs_synth_auc": as_float(h.get("real_vs_synth_auc")),
                "tstr_pr_auc": as_float(h.get("tstr_pr_auc")),
                **safe_match.to_dict(),
            }
        )
    candidate_grid = candidate_raw[candidate_raw["scenario"] == "S3_HYBRID"].copy()
    candidate_grid = candidate_grid.merge(
        safe[
            [
                "task_id",
                "seed",
                "safe_baseline_val_PR_AUC",
                "safe_baseline_val_FPR",
                "safe_baseline_val_Recall_at_FPR010",
            ]
        ],
        on=["task_id", "seed"],
        how="left",
    )
    candidate_grid["val_gain"] = candidate_grid["PR_AUC"] - candidate_grid["safe_baseline_val_PR_AUC"]
    candidate_grid["val_fpr_excess"] = candidate_grid["FPR"] - candidate_grid["safe_baseline_val_FPR"]
    candidate_grid["val_recall_fpr010_gain"] = (
        candidate_grid["Recall_at_FPR010"].fillna(0.0)
        - candidate_grid["safe_baseline_val_Recall_at_FPR010"].fillna(0.0)
    )
    return add_candidate_features(pd.DataFrame(rows)), safe, candidate_grid


def pick_candidate(candidates: pd.DataFrame, rule_name: str) -> pd.Series | None:
    if candidates.empty:
        return None
    if rule_name == "force_valbest_hybrid":
        pool = candidates.copy()
        sort_cols = ["val_PR_AUC", "val_fpr_excess", "certificate_score_v9", "tstr_pr_auc"]
        ascending = [False, True, False, False]
    elif rule_name == "v9_opportunity_s1plus_anyrep_fpr020":
        pool = candidates[candidates["pass_v9_opportunity_s1plus_anyrep_fpr020"]].copy()
        sort_cols = ["certificate_score_v9", "val_gain", "val_recall_fpr010_gain", "val_fpr_excess", "val_PR_AUC"]
        ascending = [False, False, False, True, False]
    elif rule_name == "strict_certificate":
        pool = candidates[candidates["pass_strict_cert"]].copy()
        sort_cols = ["certificate_score_v9", "val_gain", "val_fpr_excess", "val_PR_AUC"]
        ascending = [False, False, True, False]
    elif rule_name == "posthoc_keep075_or_valgain":
        pool = candidates[
            candidates["pass_v9_opportunity_s1plus_anyrep_fpr020"]
            & (
                (candidates["s3_ig_keep_rate"].fillna(0.0) >= 0.75)
                | (candidates["val_gain"] >= 0.0)
            )
        ].copy()
        sort_cols = ["certificate_score_v9", "val_gain", "val_recall_fpr010_gain", "val_fpr_excess", "val_PR_AUC"]
        ascending = [False, False, False, True, False]
    else:
        raise ValueError(f"Unknown rule: {rule_name}")
    if pool.empty:
        return None
    return pool.sort_values(sort_cols, ascending=ascending).iloc[0]


def evaluate_rules(candidates: pd.DataFrame, safe: pd.DataFrame) -> pd.DataFrame:
    rules = [
        "safe_only",
        "final_validation_best_hybrid",
        "v9_opportunity_s1plus_anyrep_fpr020",
        "posthoc_keep075_or_valgain",
        "strict_certificate",
    ]
    rows = []
    for _, safe_row in safe.iterrows():
        group = candidates[
            (candidates["task_id"] == int(safe_row["task_id"]))
            & (candidates["seed"] == int(safe_row["seed"]))
        ].copy()
        safe_test = as_float(safe_row["safe_baseline_test_PR_AUC"])
        for rule in rules:
            pick_rule = "force_valbest_hybrid" if rule == "final_validation_best_hybrid" else rule
            pick = None if rule == "safe_only" else pick_candidate(group, pick_rule)
            if pick is None:
                rows.append(
                    {
                        "rule_name": rule,
                        "task_id": int(safe_row["task_id"]),
                        "seed": int(safe_row["seed"]),
                        "selected_kind": "safe_baseline",
                        "selected_method": f"{safe_row['safe_baseline_scenario']}/{safe_row['safe_baseline_method']}",
                        "selected_candidate_key": "safe_baseline",
                        "selected_test_PR_AUC": safe_test,
                        "safe_baseline_test_PR_AUC": safe_test,
                        "delta_vs_safe_baseline_test": 0.0,
                        "harm_flag": False,
                    }
                )
            else:
                selected_test = as_float(pick["test_PR_AUC"])
                rows.append(
                    {
                        "rule_name": rule,
                        "task_id": int(pick["task_id"]),
                        "seed": int(pick["seed"]),
                        "selected_kind": "hybrid",
                        "selected_method": str(pick["method"]),
                        "selected_candidate_key": (
                            f"{pick['method']}/mode={pick['ddpm_train_mode']}/"
                            f"ratio={pick['ratio']}/keep={pick['s3_ig_keep_rate']}/post={pick['s3_postprocess']}"
                        ),
                        "selected_test_PR_AUC": selected_test,
                        "safe_baseline_test_PR_AUC": safe_test,
                        "delta_vs_safe_baseline_test": selected_test - safe_test,
                        "harm_flag": bool(selected_test < safe_test),
                        "selected_val_gain": as_float(pick["val_gain"]),
                        "selected_val_fpr_excess": as_float(pick["val_fpr_excess"]),
                        "selected_val_recall_fpr010_gain": as_float(pick["val_recall_fpr010_gain"]),
                        "selected_certificate_score_v9": as_float(pick["certificate_score_v9"]),
                        "selected_ratio": as_float(pick["ratio"]),
                        "selected_keep_rate": as_float(pick["s3_ig_keep_rate"]),
                        "selected_tstr_pr_auc": as_float(pick["tstr_pr_auc"]),
                    }
                )
    return pd.DataFrame(rows)


def summarize_selection(selected: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for key, group in selected.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, key if isinstance(key, tuple) else (key,)))
        delta = group["delta_vs_safe_baseline_test"].astype(float)
        row.update(
            {
                "n": int(len(group)),
                "harm_count": int(group["harm_flag"].sum()),
                "harm_rate": float(group["harm_flag"].mean()),
                "mean_delta_vs_safe_baseline_test": float(delta.mean()),
                "median_delta_vs_safe_baseline_test": float(delta.median()),
                "worst_harm": float(delta.min()),
                "best_gain": float(delta.max()),
                "hybrid_selection_rate": float((group["selected_kind"] == "hybrid").mean()),
                "selected_hybrid_count": int((group["selected_kind"] == "hybrid").sum()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["harm_count", "mean_delta_vs_safe_baseline_test", "selected_hybrid_count"],
        ascending=[True, False, False],
    )


def summarize_winners(candidates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_variant = (
        candidates.groupby(["ratio", "s3_ig_keep_rate", "s3_postprocess"], dropna=False)
        .agg(
            n=("seed", "count"),
            harm_rate=("harm_flag", "mean"),
            mean_delta_vs_safe_baseline_test=("delta_vs_safe_baseline_test", "mean"),
            median_delta_vs_safe_baseline_test=("delta_vs_safe_baseline_test", "median"),
            worst_harm=("delta_vs_safe_baseline_test", "min"),
            best_gain=("delta_vs_safe_baseline_test", "max"),
            mean_val_gain=("val_gain", "mean"),
            mean_val_fpr_excess=("val_fpr_excess", "mean"),
            mean_certificate_score_v9=("certificate_score_v9", "mean"),
        )
        .reset_index()
        .sort_values(["harm_rate", "mean_delta_vs_safe_baseline_test"], ascending=[True, False])
    )
    top_seed = (
        candidates.sort_values(
            ["task_id", "seed", "test_PR_AUC", "val_PR_AUC"],
            ascending=[True, True, False, False],
        )
        .groupby(["task_id", "seed"], as_index=False)
        .head(5)
    )
    return by_variant, top_seed


def summarize_candidate_grid(candidate_grid: pd.DataFrame) -> pd.DataFrame:
    grid = candidate_grid.copy()
    return (
        grid.groupby(["ratio", "s3_ig_keep_rate", "s3_postprocess"], dropna=False)
        .agg(
            n=("seed", "count"),
            mean_val_PR_AUC=("PR_AUC", "mean"),
            median_val_PR_AUC=("PR_AUC", "median"),
            mean_val_gain=("val_gain", "mean"),
            median_val_gain=("val_gain", "median"),
            mean_val_fpr_excess=("val_fpr_excess", "mean"),
            mean_val_recall_fpr010_gain=("val_recall_fpr010_gain", "mean"),
            mean_tstr_pr_auc=("tstr_pr_auc", "mean"),
        )
        .reset_index()
        .sort_values(["mean_val_gain", "mean_val_PR_AUC"], ascending=[False, False])
    )


def write_readme(
    out_dir: Path,
    run_dir: Path,
    title: str,
    description: str,
    selection_summary: pd.DataFrame,
    selection_task_summary: pd.DataFrame,
    selected: pd.DataFrame,
    winner_variant_summary: pd.DataFrame,
    candidate_grid_summary: pd.DataFrame,
    top_seed: pd.DataFrame,
) -> None:
    selected_cols = [
        "rule_name",
        "task_id",
        "seed",
        "selected_kind",
        "selected_candidate_key",
        "selected_test_PR_AUC",
        "safe_baseline_test_PR_AUC",
        "delta_vs_safe_baseline_test",
        "harm_flag",
    ]
    top_cols = [
        "task_id",
        "seed",
        "ratio",
        "s3_ig_keep_rate",
        "test_PR_AUC",
        "safe_baseline_test_PR_AUC",
        "delta_vs_safe_baseline_test",
        "val_gain",
        "val_fpr_excess",
        "tstr_pr_auc",
        "certificate_score_v9",
    ]
    lines = [
        f"# {title}",
        "",
        description,
        "",
        f"Source run: `{run_dir}`",
        "",
        "## Selection Summary",
        "",
        markdown_table(selection_summary),
        "",
        "## Selection Summary by Task",
        "",
        markdown_table(selection_task_summary),
        "",
        "## Seed-Level Rule Decisions",
        "",
        markdown_table(selected[selected_cols]),
        "",
        "## Winner Variant Summary",
        "",
        "This table covers only the validation-selected S3_HYBRID winner per seed,",
        "because strict final test is reported only for selected winners.",
        "",
        markdown_table(winner_variant_summary.head(20)),
        "",
        "## Validation Grid Summary",
        "",
        "This table covers the full ratio/keep grid, but validation metrics only.",
        "",
        markdown_table(candidate_grid_summary.head(20)),
        "",
        "## Top Hybrid Candidates Per Seed",
        "",
        markdown_table(top_seed[top_cols]),
        "",
        "## Reading",
        "",
        "- `final_validation_best_hybrid` is the strict runner's validation-selected hybrid winner.",
        "- `v9_opportunity_s1plus_anyrep_fpr020` is the mined performance rule from v9.",
        "- `posthoc_keep075_or_valgain` is a post-hoc next-rule candidate mined from this run.",
        "- To test non-winner candidates on the final test split, the runner must be extended",
        "  to report final metrics for every validation-passing candidate.",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def copy_artifacts(out_dir: Path, artifact_dir: Path, file_prefix: str) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    for name in [
        f"{file_prefix}_candidates.csv",
        f"{file_prefix}_candidate_val_grid.csv",
        f"{file_prefix}_selection_results.csv",
        f"{file_prefix}_selection_summary.csv",
        f"{file_prefix}_selection_summary_by_task.csv",
        f"{file_prefix}_winner_variant_summary.csv",
        f"{file_prefix}_candidate_val_grid_summary.csv",
        f"{file_prefix}_top_candidates_by_seed.csv",
        "README.md",
    ]:
        src = out_dir / name
        if src.exists():
            shutil.copy2(src, artifact_dir / name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--file-prefix", default="task10_v9_s1plus")
    parser.add_argument("--title", default="Task10 V9 Targeted S1+IG Positive Sweep")
    parser.add_argument(
        "--description",
        default=(
            "This run expands ratio/keep-rate only for the family that v9 mining "
            "identified as promising: `summary_stats`, `S3_HYBRID/s1_plus_ig`, "
            "`positive_only` DDPM, no postprocessing."
        ),
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    candidates, safe, candidate_grid = load_candidates(args.run_dir)
    selected = evaluate_rules(candidates, safe)
    selection_summary = summarize_selection(selected, ["rule_name"])
    selection_task_summary = summarize_selection(selected, ["rule_name", "task_id"])
    winner_variant_summary, top_seed = summarize_winners(candidates)
    candidate_grid_summary = summarize_candidate_grid(candidate_grid)

    prefix = args.file_prefix
    candidates.to_csv(args.out_dir / f"{prefix}_candidates.csv", index=False)
    candidate_grid.to_csv(args.out_dir / f"{prefix}_candidate_val_grid.csv", index=False)
    selected.to_csv(args.out_dir / f"{prefix}_selection_results.csv", index=False)
    selection_summary.to_csv(args.out_dir / f"{prefix}_selection_summary.csv", index=False)
    selection_task_summary.to_csv(args.out_dir / f"{prefix}_selection_summary_by_task.csv", index=False)
    winner_variant_summary.to_csv(args.out_dir / f"{prefix}_winner_variant_summary.csv", index=False)
    candidate_grid_summary.to_csv(args.out_dir / f"{prefix}_candidate_val_grid_summary.csv", index=False)
    top_seed.to_csv(args.out_dir / f"{prefix}_top_candidates_by_seed.csv", index=False)
    write_readme(
        args.out_dir,
        args.run_dir,
        args.title,
        args.description,
        selection_summary,
        selection_task_summary,
        selected,
        winner_variant_summary,
        candidate_grid_summary,
        top_seed,
    )
    copy_artifacts(args.out_dir, args.artifact_dir, prefix)

    print(f"[Done] saved analysis to: {args.out_dir}")
    print("\n[Selection summary]")
    print(selection_summary.to_string(index=False))
    print("\n[Selection summary by task]")
    print(selection_task_summary.to_string(index=False))
    print("\n[Top validation grid variants]")
    print(candidate_grid_summary.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
