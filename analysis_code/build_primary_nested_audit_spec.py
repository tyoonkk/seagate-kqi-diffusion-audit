# -*- coding: utf-8 -*-
"""primary nested audit 의 machine-readable 명세 생성.

심사 Major 1 / Codex M1 대응: 일곱 gate, 세 family(내부 24-grid 와 winner 규칙),
fold 구성, objective 와 tie-break, 정보 경계, run-local reference, 산출물 SHA-256,
Table 5 행 매핑을 하나의 JSON 으로 봉인한다. 모든 수치·규칙은 audit 스크립트와
그 산출물에서 그대로 옮긴 것이며 새 계산은 없다.
"""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
AUDIT_DIR = (ROOT / "experiments/seagate_kqi/artifacts/condition_aware_main/"
             "ieee_task_independent_nested_audit_run1")
SCRIPT = ROOT / "experiments/seagate_kqi/build_ieee_task_independent_nested_audit.py"
OUT = Path(__file__).with_name("primary_nested_audit_spec.json")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def typed(value, kind):
    if value is None or value == "":
        return None
    if kind is bool:
        return value == "True"
    return kind(value)


gates = []
for row in csv.DictReader(open(AUDIT_DIR / "policy_definitions.csv",
                               encoding="utf-8")):
    gates.append({
        "name": row["name"],
        "description": row["description"],
        "min_val_gain": typed(row["min_val_gain"], float),
        "max_val_fpr_excess": typed(row["max_val_fpr_excess"], float),
        "positive_only": typed(row["positive_only"], bool),
        "min_tstr_pr_auc": typed(row["min_tstr_pr_auc"], float),
        "pass_all": typed(row["pass_all"], bool),
    })

artifact_hashes = {
    name: sha(AUDIT_DIR / name)
    for name in ("README.md", "candidate_source_audit.csv",
                 "outer_fold_choices.csv", "outer_heldout_rows.csv",
                 "outer_train_policy_scores.csv", "policy_definitions.csv",
                 "summary.csv", "task_summary.csv")
}

INPUTS = {
    "v9_candidate_evidence_table": (
        "experiments/seagate_kqi/artifacts/condition_aware_main/"
        "selector_v9_certificate_mining_run1/v9_candidate_evidence_table.csv"),
    "raw_validation_grid_1080_rows": (
        "experiments/seagate_kqi/outputs/paper_strict/"
        "condition_hybrid_alltasks_seed42_46_quick_mean_run2/"
        "candidate_val_results.csv"),
    "source_run_log_seed42": (
        "experiments/seagate_kqi/outputs/paper_strict/"
        "condition_hybrid_alltasks_seed42_46_quick_mean_run2/"
        "condition_hybrid_alltasks_seed42_46_quick_mean_run2_seed42.log"),
    "reference_strategy_selection": (
        "experiments/seagate_kqi/outputs/condition_aware/"
        "strategy_selection_results.csv"),
}
input_hashes = {
    key: {"relative_path": rel, "sha256": sha(ROOT / rel)}
    for key, rel in INPUTS.items()
}

spec = {
    "schema": "seagate-audit-primary-nested-audit-spec-v1",
    "audit_script": {
        "relative_path": ("experiments/seagate_kqi/"
                          "build_ieee_task_independent_nested_audit.py"),
        "sha256": sha(SCRIPT),
        "bootstrap": {"rng": "numpy.default_rng", "seed": 20260714,
                      "repetitions": 10000},
    },
    "candidate_archive": {
        "source": "quick_mean_alltasks (V9 candidate evidence table)",
        "raw_grid": {
            "rows": 1080,
            "cases": 45,
            "configurations_per_case": 24,
            "families": {
                "ig_clean": ("real training rows plus cleaned "
                             "importance-guided synthetic positives"),
                "ig_smote": ("importance-guided synthetic positives "
                             "followed by SMOTE"),
                "s1_plus_ig": ("SMOTE S1 base followed by importance-guided "
                               "synthetic positives"),
            },
            "ddpm_train_modes": ["positive_only", "conditional_mixed"],
            "augmentation_ratios": [0.5, 1.0],
            "synthetic_keep_rates": [0.25, 0.75],
        },
        "family_winner_rule": {
            "sort_descending": ["validation PR-AUC", "MCC", "ROC-AUC",
                                "Recall", "Precision"],
            "winners_entering_audit": "45 cases x 3 family winners = 135 rows",
        },
        "run_provenance": {
            "representation": "mean",
            "classifier": "LightGBM, 400 trees (explicit, not quick-reduced)",
            "diffusion_epochs": ("200 requested on the command line; the "
                                 "--quick path overrides training to 10 "
                                 "epochs"),
            "synthetic_filter": "ig_weighted (importance-weighted filtering)",
            "postprocessing": "none",
            "base_smote_ratio": 0.5,
            "s1_base_ratio": 0.5,
            "selection_threshold_rule": "max_mcc",
        },
        "distinct_from": ("the 24-candidate equal-budget grid of the "
                          "secondary audit (6 ratios x 4 keep rates, single "
                          "family s1_plus_ig, positive-only mode)"),
    },
    "run_local_reference": {
        "definition": ("per task-seed validation-best conventional S0/S1 "
                       "strategy recorded in the same archival run "
                       "(descending validation PR-AUC)"),
        "observed_methods": {"S1/smote": 15, "S1/random_under_10to1": 12,
                             "S1/random_over": 10, "S1/random_under_20to1": 8},
        "relation_to_equal_budget_reference": ("numerically different in all "
                                               "45 cases"),
    },
    "gates": gates,
    "metric_definitions": {
        "validation_gain": ("candidate validation PR-AUC minus run-local "
                            "reference validation PR-AUC"),
        "validation_fpr_excess": ("candidate validation FPR minus run-local "
                                  "reference validation FPR "
                                  "(val_FPR - safe_baseline_val_FPR), each "
                                  "model scored at its own "
                                  "validation-selected max-MCC threshold"),
        "threshold_rule": "max_mcc (validation-derived per model)",
        "tstr_pr_auc": ("train-synthetic-test-real PR-AUC evaluated on the "
                        "real validation split; no test outcome involved"),
    },
    "row_level_selection": {
        "inputs": ["validation gain", "validation FPR excess",
                   "generation mode", "TSTR PR-AUC"],
        "test_fields_used": False,
        "order": ["validation PR-AUC desc", "validation FPR excess asc",
                  "TSTR PR-AUC desc", "candidate_key asc"],
        "no_pass_action": "retain run-local reference",
        "tstr_note": ("TSTR PR-AUC is a selection-side diagnostic computed "
                      "without any test outcome"),
    },
    "outer_meta_selection": {
        "folds": "leave-one-task-out over tasks [0,1,2,3,5,6,7,9,10]",
        "development_signal": ("archived test deltas of the eight "
                               "development tasks x five repeats"),
        "held_out_test_access": ("only after the gate and row decisions are "
                                 "fixed"),
        "objectives": {
            "conservative": ("eligible gates have zero negative development "
                             "rows and zero negative development task means; "
                             "reference_only if none"),
            "mean_seeking": "all gates eligible",
        },
        "tie_break_order": ["task-macro mean desc", "negative task count asc",
                            "negative row count asc",
                            "hybrid deployment count asc", "gate name asc"],
    },
    "gate_family_provenance": {
        "defined": ("together with the audit script and executed "
                    "immediately (2026-07-14 file timestamps)"),
        "sealed_prespecification": None,
        "claim_scope": ("archival outer-task replay of the seven evaluated "
                        "gates; not prospective validation"),
    },
    "inputs": input_hashes,
    "output_root": ("experiments/seagate_kqi/artifacts/condition_aware_main/"
                    "ieee_task_independent_nested_audit_run1"),
    "artifact_sha256": artifact_hashes,
    "table5_row_mapping": {
        "nested_conservative": {
            "spec": "this file", "outputs": ["summary.csv",
                                             "outer_fold_choices.csv",
                                             "outer_heldout_rows.csv"]},
        "nested_mean_seeking": {
            "spec": "this file", "outputs": ["summary.csv",
                                             "outer_fold_choices.csv",
                                             "outer_heldout_rows.csv"]},
        "equal_budget_loto": {
            "spec": ("secondary audit; see "
                     "ieee_v10_uniform_grid_loto_audit_run1"),
            "outputs": ["ieee_v10_uniform_grid_loto_audit_run1"]},
    },
}

OUT.write_text(json.dumps(spec, indent=2, ensure_ascii=False) + "\n",
               encoding="utf-8")
print("wrote", OUT)
print("spec sha256:", hashlib.sha256(OUT.read_bytes()).hexdigest())
