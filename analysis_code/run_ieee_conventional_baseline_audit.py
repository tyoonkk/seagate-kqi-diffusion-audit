#!/usr/bin/env python
"""Uniform conventional-baseline audit for the IEEE Access revision.

The audit includes every Seagate KQI endpoint with at least 500 valid training
rows.  Endpoint eligibility therefore uses the training split only; validation
and test positive counts are reported but do not determine inclusion.

Each classifier/resampling pair receives one fixed configuration, so no method
or endpoint receives a larger tuning budget.  The validation split selects the
best conventional method for each task/seed/classifier.  Test outcomes are
read only after that selection.  Per-method test results are also retained as
predefined benchmark results.
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import imblearn
import lightgbm
import numpy as np
import pandas as pd
import sklearn
import xgboost
from imblearn.over_sampling import ADASYN, BorderlineSMOTE, RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier


THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parents[1]
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from run_pipeline import (  # noqa: E402
    PipelineConfig,
    TabularPreprocessor,
    add_recall_at_fpr_caps,
    build_or_load_representation,
    evaluate_at_threshold,
    extract_task_binary,
    load_toolset_arrays,
    task_stats_one_split,
    tune_threshold,
)


DEFAULT_OUT = (
    ROOT
    / "experiments/seagate_kqi/artifacts/condition_aware_main/"
    "ieee_conventional_baseline_audit_run1"
)
METHODS = (
    "unweighted",
    "class_weighted",
    "random_over",
    "random_under_10to1",
    "smote",
    "borderline_smote",
    "adasyn",
)
CLASSIFIERS = ("lightgbm", "xgboost")
BOOTSTRAP_SEED = 20260714
BOOTSTRAP_REPS = 10_000


@dataclass(frozen=True)
class AuditConfig:
    data_root: str
    toolset: str
    representation: str
    min_train_valid: int
    seeds: tuple[int, ...]
    classifiers: tuple[str, ...]
    methods: tuple[str, ...]
    n_estimators: int
    learning_rate: float
    target_add_ratio: float
    under_negative_to_positive: float
    n_jobs: int


def parse_ints(text: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in text.split(",") if part.strip())


def parse_strings(text: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in text.split(",") if part.strip())


def git_state() -> dict[str, object]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.STDOUT
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=ROOT, text=True, stderr=subprocess.STDOUT
            ).strip()
        )
        return {"git_commit": commit, "git_dirty": dirty}
    except Exception as exc:  # pragma: no cover
        return {"git_commit": "unavailable", "git_dirty": None, "git_error": str(exc)}


def environment_record() -> dict[str, object]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "imbalanced_learn": imblearn.__version__,
        "lightgbm": lightgbm.__version__,
        "xgboost": xgboost.__version__,
        **git_state(),
    }


def target_positive_count(y: np.ndarray, add_ratio: float) -> int:
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    return min(n_neg, int(round(n_pos * (1.0 + add_ratio))))


def resample(
    X: np.ndarray,
    y: np.ndarray,
    method: str,
    seed: int,
    add_ratio: float,
    under_negative_to_positive: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    if method in {"unweighted", "class_weighted"}:
        return X, y, 0

    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    if n_pos < 2 or n_neg < 2:
        raise ValueError("Both classes need at least two training rows")

    if method == "random_under_10to1":
        target_neg = min(n_neg, int(round(n_pos * under_negative_to_positive)))
        if target_neg >= n_neg:
            return X, y, 0
        sampler = RandomUnderSampler(
            sampling_strategy=float(n_pos / target_neg), random_state=seed
        )
    else:
        target_pos = target_positive_count(y, add_ratio)
        if target_pos <= n_pos:
            return X, y, 0
        strategy = {1: target_pos}
        k = max(1, min(5, n_pos - 1))
        if method == "random_over":
            sampler = RandomOverSampler(sampling_strategy=strategy, random_state=seed)
        elif method == "smote":
            sampler = SMOTE(sampling_strategy=strategy, random_state=seed, k_neighbors=k)
        elif method == "borderline_smote":
            sampler = BorderlineSMOTE(
                sampling_strategy=strategy,
                random_state=seed,
                k_neighbors=k,
                m_neighbors=max(2, min(10, n_pos - 1)),
                kind="borderline-1",
            )
        elif method == "adasyn":
            sampler = ADASYN(sampling_strategy=strategy, random_state=seed, n_neighbors=k)
        else:
            raise ValueError(method)

    X_res, y_res = sampler.fit_resample(X, y)
    delta = int(len(y_res) - len(y))
    return np.asarray(X_res, dtype=np.float32), np.asarray(y_res, dtype=np.int8), delta


def class_weight_ratio(y: np.ndarray, method: str) -> float:
    if method != "class_weighted":
        return 1.0
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    return float(n_neg / max(n_pos, 1))


def make_model(
    classifier: str,
    seed: int,
    n_estimators: int,
    learning_rate: float,
    scale_pos_weight: float,
    n_jobs: int,
):
    if classifier == "lightgbm":
        return LGBMClassifier(
            objective="binary",
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=63,
            subsample=0.9,
            subsample_freq=1,
            colsample_bytree=0.9,
            scale_pos_weight=scale_pos_weight,
            random_state=seed,
            n_jobs=n_jobs,
            device_type="cpu",
            deterministic=True,
            force_col_wise=True,
            verbosity=-1,
        )
    if classifier == "xgboost":
        return XGBClassifier(
            objective="binary:logistic",
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=6,
            min_child_weight=1.0,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            scale_pos_weight=scale_pos_weight,
            random_state=seed,
            n_jobs=n_jobs,
            tree_method="hist",
            eval_metric="logloss",
            verbosity=0,
        )
    raise ValueError(classifier)


def score_probabilities(
    y_val: np.ndarray,
    p_val: np.ndarray,
    y_test: np.ndarray,
    p_test: np.ndarray,
) -> dict[str, float]:
    threshold = tune_threshold(y_val, p_val, "max_mcc", 0.90)
    val_main = evaluate_at_threshold(y_val, p_val, threshold)
    test_main = evaluate_at_threshold(y_test, p_test, threshold)
    val_metrics = add_recall_at_fpr_caps(val_main, y_val, p_val, y_val, p_val, (0.01, 0.02, 0.05))
    test_metrics = add_recall_at_fpr_caps(test_main, y_val, p_val, y_test, p_test, (0.01, 0.02, 0.05))
    return {
        **{f"val_{key}": value for key, value in val_metrics.items()},
        **{f"test_{key}": value for key, value in test_metrics.items()},
    }


def fit_one(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    classifier: str,
    method: str,
    seed: int,
    cfg: AuditConfig,
) -> dict[str, object]:
    start = time.perf_counter()
    X_fit, y_fit, row_delta = resample(
        X_train,
        y_train,
        method,
        seed,
        cfg.target_add_ratio,
        cfg.under_negative_to_positive,
    )
    spw = class_weight_ratio(y_fit, method)
    model = make_model(
        classifier,
        seed,
        cfg.n_estimators,
        cfg.learning_rate,
        spw,
        cfg.n_jobs,
    )
    model.fit(X_fit, y_fit)
    p_val = np.asarray(model.predict_proba(X_val)[:, 1], dtype=np.float64)
    p_test = np.asarray(model.predict_proba(X_test)[:, 1], dtype=np.float64)
    metrics = score_probabilities(y_val, p_val, y_test, p_test)
    elapsed = time.perf_counter() - start
    del model, X_fit, y_fit, p_val, p_test
    gc.collect()
    return {
        "train_row_delta": row_delta,
        "scale_pos_weight": spw,
        "fit_seconds": elapsed,
        **metrics,
    }


def save_partial(rows: list[dict[str, object]], out_dir: Path) -> None:
    pd.DataFrame(rows).to_csv(out_dir / "baseline_rows.csv", index=False)


def task_bootstrap(values: pd.Series) -> tuple[float, float]:
    array = values.to_numpy(dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    index = rng.integers(0, len(array), size=(BOOTSTRAP_REPS, len(array)))
    draws = array[index].mean(axis=1)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return float(lo), float(hi)


def build_summaries(rows: pd.DataFrame, out_dir: Path) -> None:
    selected = (
        rows.sort_values(
            ["classifier", "task_id", "seed", "val_PR_AUC", "method"],
            ascending=[True, True, True, False, True],
            kind="mergesort",
        )
        .groupby(["classifier", "task_id", "seed"], as_index=False)
        .head(1)
        .copy()
    )
    selected["selection_rule"] = "maximum validation PR-AUC; lexical method tie-break"
    selected.to_csv(out_dir / "validation_selected_rows.csv", index=False)

    task_summary = (
        rows.groupby(["classifier", "method", "task_id"], as_index=False)
        .agg(
            n_seeds=("seed", "size"),
            mean_val_PR_AUC=("val_PR_AUC", "mean"),
            mean_test_PR_AUC=("test_PR_AUC", "mean"),
            sd_test_PR_AUC=("test_PR_AUC", "std"),
            mean_test_Recall_at_FPR010=("test_Recall_at_FPR010", "mean"),
            mean_fit_seconds=("fit_seconds", "mean"),
        )
    )
    task_summary.to_csv(out_dir / "method_task_summary.csv", index=False)

    overall_rows: list[dict[str, object]] = []
    for (classifier, method), group in task_summary.groupby(["classifier", "method"], sort=True):
        lo, hi = task_bootstrap(group["mean_test_PR_AUC"])
        overall_rows.append(
            {
                "result_type": "predefined_method",
                "classifier": classifier,
                "method": method,
                "n_tasks": int(len(group)),
                "task_macro_test_PR_AUC": float(group["mean_test_PR_AUC"].mean()),
                "task_median_test_PR_AUC": float(group["mean_test_PR_AUC"].median()),
                "task_bootstrap_ci_low": lo,
                "task_bootstrap_ci_high": hi,
                "task_macro_Recall_at_FPR010": float(group["mean_test_Recall_at_FPR010"].mean()),
            }
        )

    selected_task = (
        selected.groupby(["classifier", "task_id"], as_index=False)
        .agg(
            mean_test_PR_AUC=("test_PR_AUC", "mean"),
            mean_test_Recall_at_FPR010=("test_Recall_at_FPR010", "mean"),
        )
    )
    for classifier, group in selected_task.groupby("classifier", sort=True):
        lo, hi = task_bootstrap(group["mean_test_PR_AUC"])
        overall_rows.append(
            {
                "result_type": "validation_selected_conventional",
                "classifier": classifier,
                "method": "validation_selected",
                "n_tasks": int(len(group)),
                "task_macro_test_PR_AUC": float(group["mean_test_PR_AUC"].mean()),
                "task_median_test_PR_AUC": float(group["mean_test_PR_AUC"].median()),
                "task_bootstrap_ci_low": lo,
                "task_bootstrap_ci_high": hi,
                "task_macro_Recall_at_FPR010": float(group["mean_test_Recall_at_FPR010"].mean()),
            }
        )
    pd.DataFrame(overall_rows).to_csv(out_dir / "overall_summary.csv", index=False)


def write_readme(cfg: AuditConfig, stats: pd.DataFrame, out_dir: Path) -> None:
    summary = pd.read_csv(out_dir / "overall_summary.csv")
    selected = summary[summary["result_type"].eq("validation_selected_conventional")]
    lines = [
        "# IEEE Conventional Baseline Audit Run 1",
        "",
        "## Protocol",
        "",
        "- Eligibility: at least 500 valid training rows; no validation/test count is used.",
        f"- Included endpoints: {','.join(map(str, sorted(stats.loc[stats['include'], 'task_id'].astype(int))))}.",
        f"- Seeds: {','.join(map(str, cfg.seeds))}.",
        f"- Classifiers: {', '.join(cfg.classifiers)}.",
        f"- Methods: {', '.join(cfg.methods)}.",
        f"- Trees per fit: {cfg.n_estimators}; learning rate: {cfg.learning_rate}.",
        f"- Oversampling adds {cfg.target_add_ratio:.1f} times the original positive count.",
        f"- Random undersampling retains {cfg.under_negative_to_positive:.0f} negatives per positive.",
        "- Every classifier/method pair has one fixed configuration and therefore one fit budget.",
        "- The validation-selected conventional result chooses the method by validation PR-AUC only.",
        "- Operational thresholds are chosen on validation by maximum MCC.",
        f"- Task-cluster bootstrap: {BOOTSTRAP_REPS:,} resamples, seed {BOOTSTRAP_SEED}.",
        "",
        "## Validation-selected conventional results",
        "",
    ]
    for row in selected.itertuples(index=False):
        lines.append(
            f"- {row.classifier}: task-macro test PR-AUC {row.task_macro_test_PR_AUC:.6f} "
            f"(95% task-bootstrap CI {row.task_bootstrap_ci_low:.6f} to "
            f"{row.task_bootstrap_ci_high:.6f}); task-macro Recall@FPR=0.01 "
            f"{row.task_macro_Recall_at_FPR010:.6f}."
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "This audit strengthens the conventional comparison and the all-endpoint eligibility",
            "sensitivity.  It does not provide a cross-classifier evaluation of the archived",
            "synthetic candidates because the corresponding synthetic training matrices were not",
            "retained.  That missing comparison is reported as a limitation rather than inferred.",
            "",
            "## Reproduction",
            "",
            "`python experiments/seagate_kqi/run_ieee_conventional_baseline_audit.py`",
            "",
        ]
    )
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=".venv/softsensing_data_full")
    parser.add_argument("--toolset", default="time-series-2")
    parser.add_argument("--representation", default="mean")
    parser.add_argument("--min-train-valid", type=int, default=500)
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--classifiers", default=",".join(CLASSIFIERS))
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--target-add-ratio", type=float, default=1.0)
    parser.add_argument("--under-negative-to-positive", type=float, default=10.0)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    cfg = AuditConfig(
        data_root=args.data_root,
        toolset=args.toolset,
        representation=args.representation,
        min_train_valid=args.min_train_valid,
        seeds=parse_ints(args.seeds),
        classifiers=parse_strings(args.classifiers),
        methods=parse_strings(args.methods),
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        target_add_ratio=args.target_add_ratio,
        under_negative_to_positive=args.under_negative_to_positive,
        n_jobs=args.n_jobs,
    )
    unknown_classifiers = sorted(set(cfg.classifiers).difference(CLASSIFIERS))
    unknown_methods = sorted(set(cfg.methods).difference(METHODS))
    if unknown_classifiers or unknown_methods:
        raise ValueError(
            f"Unknown classifiers={unknown_classifiers}, methods={unknown_methods}"
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    (out_dir / "environment.json").write_text(
        json.dumps(environment_record(), indent=2), encoding="utf-8"
    )

    X_train_raw, X_val_raw, X_test_raw, y_train_raw, y_val_raw, y_test_raw = load_toolset_arrays(
        cfg.data_root, cfg.toolset, "r"
    )
    pipeline_cfg = PipelineConfig(
        data_root=cfg.data_root,
        toolset=cfg.toolset,
        representation=cfg.representation,
        use_rep_cache=True,
    )
    X_train_rep = build_or_load_representation(X_train_raw, pipeline_cfg, "train")
    X_val_rep = build_or_load_representation(X_val_raw, pipeline_cfg, "val")
    X_test_rep = build_or_load_representation(X_test_raw, pipeline_cfg, "test")

    stats = pd.concat(
        [
            task_stats_one_split(y_train_raw, "train"),
            task_stats_one_split(y_val_raw, "val"),
            task_stats_one_split(y_test_raw, "test"),
        ],
        ignore_index=True,
    )
    pivot = stats.pivot(index="task_id", columns="split", values=["valid", "positive", "negative"])
    pivot.columns = [f"{metric}_{split_name}" for metric, split_name in pivot.columns]
    pivot = pivot.reset_index()
    pivot["include"] = pivot["valid_train"].ge(cfg.min_train_valid)
    pivot["eligibility_rule"] = f"valid_train >= {cfg.min_train_valid}"
    pivot.to_csv(out_dir / "task_eligibility.csv", index=False)
    tasks = sorted(pivot.loc[pivot["include"], "task_id"].astype(int))

    rows: list[dict[str, object]] = []
    completed: set[tuple[int, int, str, str]] = set()
    partial_path = out_dir / "baseline_rows.csv"
    if args.resume and partial_path.exists():
        prior = pd.read_csv(partial_path)
        rows = prior.to_dict("records")
        completed = {
            (int(row.task_id), int(row.seed), str(row.classifier), str(row.method))
            for row in prior.itertuples(index=False)
        }
        print(f"[resume] loaded {len(rows)} completed rows")

    for task_id in tasks:
        train_mask, y_train = extract_task_binary(y_train_raw, task_id)
        val_mask, y_val = extract_task_binary(y_val_raw, task_id)
        test_mask, y_test = extract_task_binary(y_test_raw, task_id)
        prep = TabularPreprocessor(drop_constant=True, imputer_strategy="mean")
        prep.fit(X_train_rep[train_mask])
        X_train = np.asarray(prep.transform_classifier(X_train_rep[train_mask]), dtype=np.float32)
        X_val = np.asarray(prep.transform_classifier(X_val_rep[val_mask]), dtype=np.float32)
        X_test = np.asarray(prep.transform_classifier(X_test_rep[test_mask]), dtype=np.float32)
        print(
            f"[task {task_id}] train={len(y_train)} val={len(y_val)} test={len(y_test)} "
            f"features={X_train.shape[1]}",
            flush=True,
        )

        for seed in cfg.seeds:
            for classifier in cfg.classifiers:
                for method in cfg.methods:
                    key = (task_id, seed, classifier, method)
                    if key in completed:
                        continue
                    print(f"  seed={seed} classifier={classifier} method={method}", flush=True)
                    try:
                        result = fit_one(
                            X_train,
                            y_train,
                            X_val,
                            y_val,
                            X_test,
                            y_test,
                            classifier,
                            method,
                            seed,
                            cfg,
                        )
                        status = "ok"
                        error = ""
                    except Exception as exc:
                        result = {}
                        status = "error"
                        error = repr(exc)
                        print(f"    ERROR {error}", flush=True)
                    rows.append(
                        {
                            "task_id": task_id,
                            "seed": seed,
                            "classifier": classifier,
                            "method": method,
                            "status": status,
                            "error": error,
                            "train_n": len(y_train),
                            "train_pos": int(np.sum(y_train == 1)),
                            "val_n": len(y_val),
                            "val_pos": int(np.sum(y_val == 1)),
                            "test_n": len(y_test),
                            "test_pos": int(np.sum(y_test == 1)),
                            "n_features": X_train.shape[1],
                            **result,
                        }
                    )
                    save_partial(rows, out_dir)

        del X_train, X_val, X_test, y_train, y_val, y_test, prep
        gc.collect()

    all_rows = pd.DataFrame(rows)
    failed = all_rows[all_rows["status"].ne("ok")]
    if not failed.empty:
        failed.to_csv(out_dir / "failed_rows.csv", index=False)
        raise RuntimeError(f"{len(failed)} baseline fits failed; inspect failed_rows.csv")
    build_summaries(all_rows, out_dir)
    write_readme(cfg, pivot, out_dir)
    print(pd.read_csv(out_dir / "overall_summary.csv").to_string(index=False))
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
