#!/usr/bin/env python
"""Bind the matched-objective attribution numbers in the manuscript to their artifact.

Appends checks for Table 5 / Table S8 to claim_verification.json.  Every
expected value is the number printed in main.tex and supplement.tex; every
actual value is read from the archived audit output.  Fails closed.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
AUDIT = (
    ROOT
    / "experiments/seagate_kqi/artifacts/condition_aware_main/"
    "ieee_matched_objective_gate_audit_run1/outer_loto_overall_summary.csv"
)
CLAIMS = (
    ROOT
    / "experiments/seagate_kqi/paper/ieee_access_revision_2026/claim_verification.json"
)
MACHINE = ROOT / "experiments/seagate_kqi/paper/ieee_access_revision_2026/machine_readable"
MANIFEST = MACHINE / "SHA256_MANIFEST.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sync_machine_readable() -> None:
    """Copy the 42-check file into machine_readable/ and refresh its manifest entry.

    build_ieee_submission_tables.py writes the manifest before this script appends
    the matched-audit checks, so the copy and the manifest row are refreshed here.
    The manifest never lists itself.
    """
    shutil.copy2(CLAIMS, MACHINE / CLAIMS.name)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    extract = manifest["submission_extract_sha256"]
    extract.pop(MANIFEST.name, None)
    extract[CLAIMS.name] = _sha256(MACHINE / CLAIMS.name)
    manifest["submission_extract_sha256"] = dict(sorted(extract.items()))
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

# (check name, family, objective, column, value printed in the manuscript)
EXPECTED = [
    ("matched_historical_conservative_selected", "legacy5", "conservative", "hybrid_selection_count", 11),
    ("matched_historical_conservative_mean", "legacy5", "conservative", "task_macro_mean_delta", 0.008579306689453232),
    ("matched_historical_mean_seeking_selected", "legacy5", "mean_only", "hybrid_selection_count", 11),
    ("matched_historical_mean_seeking_mean", "legacy5", "mean_only", "task_macro_mean_delta", 0.008579306689453232),
    ("matched_historical_harm_first_selected", "legacy5", "harm_first", "hybrid_selection_count", 11),
    ("matched_historical_harm_first_mean", "legacy5", "harm_first", "task_macro_mean_delta", 0.008579306689453232),
    ("matched_historical_negative_rows", "legacy5", "conservative", "negative_row_count", 0),
    ("matched_simple_conservative_selected", "nested7_veto", "conservative", "hybrid_selection_count", 0),
    ("matched_simple_conservative_mean", "nested7_veto", "conservative", "task_macro_mean_delta", 0.0),
    ("matched_simple_harm_first_selected", "nested7_veto", "harm_first", "hybrid_selection_count", 0),
    ("matched_simple_mean_seeking_selected", "nested7_veto", "mean_only", "hybrid_selection_count", 32),
    ("matched_simple_mean_seeking_mean", "nested7_veto", "mean_only", "task_macro_mean_delta", 0.004942),
    ("matched_simple_mean_seeking_negative_rows", "nested7_veto", "mean_only", "negative_row_count", 11),
    ("matched_simple_mean_seeking_negative_tasks", "nested7_veto", "mean_only", "negative_task_count", 3),
    ("matched_simple_mean_seeking_ci_low", "nested7_veto", "mean_only", "task_bootstrap_ci_low", -0.008831),
    ("matched_simple_mean_seeking_ci_high", "nested7_veto", "mean_only", "task_bootstrap_ci_high", 0.018602),
    ("matched_simple_mean_seeking_minimum_task", "nested7_veto", "mean_only", "minimum_task_mean_delta", -0.034092),
]


def main() -> int:
    frame = pd.read_csv(AUDIT).set_index(["family", "objective"])
    payload = json.loads(CLAIMS.read_text(encoding="utf-8"))
    existing = {c["name"] for c in payload["checks"]}

    added = 0
    for name, family, objective, column, expected in EXPECTED:
        if name in existing:
            continue
        actual = float(frame.loc[(family, objective), column])
        # Manuscript values are rounded to six decimals; compare at that scale.
        passed = abs(actual - float(expected)) <= 5e-7
        payload["checks"].append(
            {
                "name": name,
                "actual": actual,
                "expected": expected,
                "passed": bool(passed),
            }
        )
        added += 1

    payload["all_passed"] = all(c["passed"] for c in payload["checks"])
    CLAIMS.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    failed = [c["name"] for c in payload["checks"] if not c["passed"]]
    if not failed:
        sync_machine_readable()
    print(f"added={added} total={len(payload['checks'])} all_passed={payload['all_passed']}")
    if failed:
        print("FAILED:", failed)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
