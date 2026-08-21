#!/usr/bin/env python
"""Merge and verify the independently executed IEEE conventional audits."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from run_ieee_conventional_baseline_audit import (
    BOOTSTRAP_REPS,
    BOOTSTRAP_SEED,
    build_summaries,
)


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "experiments/seagate_kqi/artifacts/condition_aware_main"
SOURCES = {
    "lightgbm": BASE / "ieee_conventional_baseline_lightgbm_run1",
    "xgboost": BASE / "ieee_conventional_baseline_xgboost_run1",
}
OUT = BASE / "ieee_conventional_baseline_audit_run1"
EXPECTED_TASKS = tuple(range(11))
EXPECTED_SEEDS = (42, 43, 44, 45, 46)
EXPECTED_METHODS = (
    "unweighted",
    "class_weighted",
    "random_over",
    "random_under_10to1",
    "smote",
    "borderline_smote",
    "adasyn",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    configs: dict[str, dict[str, object]] = {}
    environments: dict[str, dict[str, object]] = {}
    eligibility: pd.DataFrame | None = None
    input_hashes: dict[str, str] = {}

    for classifier, source in SOURCES.items():
        required = [
            source / "baseline_rows.csv",
            source / "config.json",
            source / "environment.json",
            source / "task_eligibility.csv",
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing source files: {missing}")
        if (source / "failed_rows.csv").exists():
            failed = pd.read_csv(source / "failed_rows.csv")
            if not failed.empty:
                raise ValueError(f"{classifier} source contains failed rows")

        frame = pd.read_csv(source / "baseline_rows.csv", low_memory=False)
        if set(frame["classifier"].astype(str)) != {classifier}:
            raise ValueError(f"Classifier mismatch in {source}")
        frames.append(frame)
        configs[classifier] = json.loads((source / "config.json").read_text(encoding="utf-8"))
        environments[classifier] = json.loads(
            (source / "environment.json").read_text(encoding="utf-8")
        )
        current_eligibility = pd.read_csv(source / "task_eligibility.csv")
        if eligibility is None:
            eligibility = current_eligibility
        elif not current_eligibility.equals(eligibility):
            raise ValueError("Task eligibility differs between classifier runs")
        for path in required:
            input_hashes[str(path.relative_to(ROOT))] = sha256(path)

    rows = pd.concat(frames, ignore_index=True)
    key = ["classifier", "task_id", "seed", "method"]
    expected_rows = 2 * len(EXPECTED_TASKS) * len(EXPECTED_SEEDS) * len(EXPECTED_METHODS)
    if len(rows) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows, found {len(rows)}")
    if rows.duplicated(key).any():
        raise ValueError("Duplicate classifier/task/seed/method keys")
    if not rows["status"].eq("ok").all():
        raise ValueError("At least one fit did not finish successfully")
    if tuple(sorted(rows["task_id"].astype(int).unique())) != EXPECTED_TASKS:
        raise ValueError("Task coverage is incomplete")
    if tuple(sorted(rows["seed"].astype(int).unique())) != EXPECTED_SEEDS:
        raise ValueError("Seed coverage is incomplete")
    if tuple(sorted(rows["method"].astype(str).unique())) != tuple(sorted(EXPECTED_METHODS)):
        raise ValueError("Method coverage is incomplete")

    rows = rows.sort_values(key, kind="mergesort").reset_index(drop=True)
    rows.to_csv(OUT / "baseline_rows.csv", index=False)
    assert eligibility is not None
    eligibility.to_csv(OUT / "task_eligibility.csv", index=False)
    build_summaries(rows, OUT)

    selection_frequency = (
        pd.read_csv(OUT / "validation_selected_rows.csv")
        .groupby(["classifier", "method"], as_index=False)
        .size()
        .rename(columns={"size": "selected_task_seed_count"})
    )
    selection_frequency["selected_fraction"] = (
        selection_frequency["selected_task_seed_count"]
        / (len(EXPECTED_TASKS) * len(EXPECTED_SEEDS))
    )
    selection_frequency.to_csv(OUT / "validation_selection_frequency.csv", index=False)

    merged_config = {
        "source_runs": {name: str(path.relative_to(ROOT)) for name, path in SOURCES.items()},
        "classifiers": list(SOURCES),
        "tasks": list(EXPECTED_TASKS),
        "seeds": list(EXPECTED_SEEDS),
        "methods": list(EXPECTED_METHODS),
        "expected_rows": expected_rows,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_reps": BOOTSTRAP_REPS,
        "source_configs": configs,
    }
    (OUT / "config.json").write_text(
        json.dumps(merged_config, indent=2), encoding="utf-8"
    )
    (OUT / "environment.json").write_text(
        json.dumps({"source_environments": environments}, indent=2), encoding="utf-8"
    )

    output_files = [
        OUT / "baseline_rows.csv",
        OUT / "task_eligibility.csv",
        OUT / "validation_selected_rows.csv",
        OUT / "validation_selection_frequency.csv",
        OUT / "method_task_summary.csv",
        OUT / "overall_summary.csv",
        OUT / "config.json",
        OUT / "environment.json",
    ]
    manifest = {
        "input_sha256": input_hashes,
        "output_sha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in output_files
        },
        "verification": {
            "row_count": len(rows),
            "unique_key_count": int(rows[key].drop_duplicates().shape[0]),
            "failed_row_count": int(rows["status"].ne("ok").sum()),
            "task_count": int(rows["task_id"].nunique()),
            "classifier_count": int(rows["classifier"].nunique()),
            "method_count": int(rows["method"].nunique()),
        },
    }
    (OUT / "sha256_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    summary = pd.read_csv(OUT / "overall_summary.csv")
    selected = summary[summary["result_type"].eq("validation_selected_conventional")]
    lines = [
        "# IEEE Conventional Baseline Audit (Merged)",
        "",
        "Two classifier runs were executed independently and merged only after each",
        "completed its full 11-task by 5-seed by 7-method grid.",
        "",
        "## Integrity checks",
        "",
        f"- Rows: {len(rows)}/{expected_rows}.",
        "- Duplicate keys: 0.",
        "- Failed fits: 0.",
        "- Eligibility uses training valid-row count only; all 11 endpoints are included.",
        "- Every method has one fixed configuration and 200 trees.",
        "",
        "## Validation-selected conventional results",
        "",
    ]
    for row in selected.itertuples(index=False):
        lines.append(
            f"- {row.classifier}: task-macro test PR-AUC "
            f"{row.task_macro_test_PR_AUC:.6f} (95% task-bootstrap CI "
            f"{row.task_bootstrap_ci_low:.6f} to {row.task_bootstrap_ci_high:.6f}); "
            f"Recall@FPR=0.01 {row.task_macro_Recall_at_FPR010:.6f}."
        )
    lines.extend(
        [
            "",
            "This conventional audit does not imply that archived diffusion candidates",
            "were compared with both classifiers; complete synthetic matrices were not retained.",
            "",
        ]
    )
    (OUT / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
