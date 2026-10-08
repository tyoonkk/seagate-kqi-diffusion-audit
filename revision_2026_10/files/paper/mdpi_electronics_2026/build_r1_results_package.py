"""수정본(R1) 결과 표·수치 꾸러미를 만든다 — 계열 A (MDPI electronics-4573451). 원고 수치는 이 꾸러미에서만 가져온다.

입력
- 재생성 감사 출력 폴더(build_regen_audits_v1.py 의 --out-dir)
- 재생성 생성기 진단 표 폴더(build_regen_generator_table_v1.py 의 --out-dir)
- 원래 자료(제출본) 감사 결과: artifacts/condition_aware_main/ 의 nested·LOTO·matched 요약
출력(한 번만 쓰는 폴더): results_r1.json(모든 수치, 원값과 표시값), 표 CSV 들, 입력 해시가 든 manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
SEAGATE = HERE.parents[1]
ART = SEAGATE / "artifacts/condition_aware_main"
ORIG = {"nested": ART / "ieee_task_independent_nested_audit_run1/summary.csv",
        "loto": ART / "ieee_v10_uniform_grid_loto_audit_run1/outer_loto_overall_summary.csv",
        "matched": ART / "ieee_matched_objective_gate_audit_run2/outer_loto_overall_summary.csv"}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def f6(x) -> str:
    return f"{x:+.6f}".replace("+-", "−").replace("-", "−") if pd.notna(x) else "NA"


def primary_table(audit: Path) -> pd.DataFrame:
    rows = []
    o = pd.read_csv(ORIG["nested"])
    for _, r in o.iterrows():
        rows.append({"archive": "original", "classifier": "lgbm", "task_set": "archive9", "regime": r["regime"], "n_tasks": r["n_tasks"],
                     "n_rows": r["n_rows"], "hybrid": r["hybrid_selection_count"], "negative_rows": r["negative_row_count"],
                     "negative_tasks": r["negative_task_count"], "task_macro": r["task_macro_mean_delta"],
                     "ci_low": r["task_bootstrap_ci_low"], "ci_high": r["task_bootstrap_ci_high"]})
    for clf in ("lgbm", "xgb"):
        for ts in ("archive9", "all11"):
            p = audit / f"{clf}_{ts}_nested_summary.csv"
            if not p.exists():
                continue
            for _, r in pd.read_csv(p).iterrows():
                rows.append({"archive": "regenerated", "classifier": clf, "task_set": ts, "regime": r["regime"], "n_tasks": r["n_tasks"],
                             "n_rows": r["n_rows"], "hybrid": r["hybrid_selection_count"], "negative_rows": r["negative_row_count"],
                             "negative_tasks": r["negative_task_count"], "task_macro": r["task_macro_mean_delta"],
                             "ci_low": r["ci_low"], "ci_high": r["ci_high"]})
    return pd.DataFrame(rows)


def loto_table(audit: Path) -> pd.DataFrame:
    rows = []
    r = pd.read_csv(ORIG["loto"]).iloc[0]
    rows.append({"archive": "original", "classifier": "lgbm", "task_set": "archive9", "n_rows": r["n_rows"], "selected": r["selected_count"],
                 "negative_rows": r["harm_row_count"], "task_macro": r["task_macro_mean_delta"], "min_task": r["minimum_task_mean_delta"],
                 "ci_low": r["task_bootstrap_ci_low"], "ci_high": r["task_bootstrap_ci_high"]})
    for clf in ("lgbm", "xgb"):
        for ts in ("archive9", "all11"):
            p = audit / f"{clf}_{ts}_loto_summary.csv"
            if not p.exists():
                continue
            r = pd.read_csv(p).iloc[0]
            rows.append({"archive": "regenerated", "classifier": clf, "task_set": ts, "n_rows": r["n_rows"], "selected": r["selected_count"],
                         "negative_rows": r["harm_row_count"], "task_macro": r["task_macro_mean_delta"], "min_task": r["minimum_task_mean_delta"],
                         "ci_low": r["ci_low"], "ci_high": r["ci_high"]})
    return pd.DataFrame(rows)


def matched_table(audit: Path) -> pd.DataFrame:
    rows = []
    for _, r in pd.read_csv(ORIG["matched"]).iterrows():
        rows.append({"archive": "original", "classifier": "lgbm", "task_set": "archive9", "family": r["family"], "objective": r["objective"],
                     "hybrid": r["hybrid_selection_count"], "negative_rows": r["negative_row_count"], "task_macro": r["task_macro_mean_delta"],
                     "ci_low": r["task_bootstrap_ci_low"], "ci_high": r["task_bootstrap_ci_high"]})
    for clf in ("lgbm", "xgb"):
        for ts in ("archive9", "all11"):
            p = audit / f"{clf}_{ts}_matched_summary.csv"
            if not p.exists():
                continue
            for _, r in pd.read_csv(p).iterrows():
                rows.append({"archive": "regenerated", "classifier": clf, "task_set": ts, "family": r["family"], "objective": r["objective"],
                             "hybrid": r["hybrid_selection_count"], "negative_rows": r["negative_row_count"],
                             "task_macro": r["task_macro_mean_delta"], "ci_low": r["ci_low"], "ci_high": r["ci_high"]})
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit-dir", required=True)
    ap.add_argument("--generator-dir", default="")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    audit, out = Path(args.audit_dir), Path(args.out_dir)
    if out.exists():
        print(f"출력 폴더가 이미 있다: {out}")
        return 1
    out.mkdir(parents=True)
    tabs = {"primary": primary_table(audit), "loto": loto_table(audit), "matched": matched_table(audit)}
    agree = audit / "classifier_agreement.csv"
    if agree.exists():
        tabs["classifier_agreement"] = pd.read_csv(agree)
    for name in ("jackknife",):
        frames = []
        for clf in ("lgbm", "xgb"):
            for ts in ("archive9", "all11"):
                p = audit / f"{clf}_{ts}_{name}.csv"
                if p.exists():
                    frames.append(pd.read_csv(p).assign(classifier=clf, task_set=ts))
        if frames:
            tabs[name] = pd.concat(frames, ignore_index=True)
    if args.generator_dir:
        g = Path(args.generator_dir) / "generator_diagnostics_by_task.csv"
        if g.exists():
            tabs["generator"] = pd.read_csv(g)
    for name, df in tabs.items():
        df.to_csv(out / f"T_{name}_r1.csv", index=False)
    numbers = {name: df.to_dict(orient="records") for name, df in tabs.items()}
    inputs = {str(p): sha(p) for p in list(ORIG.values()) + sorted(audit.glob("*.csv")) + sorted(audit.glob("manifest.json"))}
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "audit_dir": str(audit), "inputs_sha256": inputs,
                "code_sha256": sha(Path(__file__).resolve())}
    (out / "results_r1.json").write_text(json.dumps(numbers, ensure_ascii=False, indent=1, default=float) + "\n", encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    p = tabs["primary"]
    for _, r in p.iterrows():
        print(f"{r['archive']:11s} {r['classifier']:4s} {r['task_set']:8s} {r['regime']:12s} hybrid {int(r['hybrid']):2d}/{int(r['n_rows'])} "
              f"neg {int(r['negative_rows'])} task-macro {f6(r['task_macro'])} [{f6(r['ci_low'])}, {f6(r['ci_high'])}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
