"""build_regen_audits_v1.py 가 원래 감사와 같은 계산을 하는지 보관 자료로 검증한다 — 제출한 Seagate 논문 수정 (2026-10).

(a) 감사 함수 재현: 원래 입력표(V9 증거표, V12 사례표)를 이 파일의 감사 래퍼에 넣어 논문 수치를 다시 낸다.
(b) 선택 규칙 재현: 보관 후보표에 이 파일의 우승·reference 규칙을 적용해 보관된 실제 선택과 같은지 본다.
결과는 표준 출력으로만 낸다(파일을 쓰지 않는다).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SEAGATE = HERE.parent
sys.path[:0] = [str(HERE), str(SEAGATE)]

import build_regen_audits_v1 as RA  # noqa: E402

BASE = SEAGATE / "artifacts/condition_aware_main"
OUT = SEAGATE / "outputs"
fails = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'OK' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        fails.append(name)


# (a) 감사 래퍼
v9 = pd.read_csv(BASE / "selector_v9_certificate_mining_run1/v9_candidate_evidence_table.csv", low_memory=False)
q = v9[v9["source"].eq("quick_mean_alltasks")].copy()
na = RA.nested_audit(q, RA.TASKS_ARCHIVE)
s = na["summary"].set_index("regime")
check("nested conservative 0/45", int(s.loc["conservative", "hybrid_selection_count"]) == 0)
check("nested mean_only 7/45", int(s.loc["mean_only", "hybrid_selection_count"]) == 7)
check("nested mean_only -0.002087", round(float(s.loc["mean_only", "task_macro_mean_delta"]), 6) == -0.002087,
      f"{s.loc['mean_only', 'task_macro_mean_delta']:.8f}")
check("nested mean_only CI", (round(float(s.loc["mean_only", "ci_low"]), 6), round(float(s.loc["mean_only", "ci_high"]), 6)) == (-0.005395, 0.0),
      f"{s.loc['mean_only', 'ci_low']:.6f},{s.loc['mean_only', 'ci_high']:.6f}")
check("nested mean_only negative rows 5", int(s.loc["mean_only", "negative_row_count"]) == 5)

case = pd.read_csv(BASE / "selector_v12_final_validation_report_run1/selector_v12_final_candidates_with_task_context.csv", low_memory=False)
la = RA.loto_audit(case, RA.TASKS_ARCHIVE)
ls = la["summary"].iloc[0]
check("LOTO 11/45", int(ls["selected_count"]) == 11)
check("LOTO +0.008579", round(float(ls["task_macro_mean_delta"]), 6) == 0.008579, f"{ls['task_macro_mean_delta']:.8f}")
check("LOTO CI", (round(float(ls["ci_low"]), 6), round(float(ls["ci_high"]), 6)) == (0.001152, 0.017543), f"{ls['ci_low']:.6f},{ls['ci_high']:.6f}")
check("LOTO harm 0", int(ls["harm_row_count"]) == 0)
ch = la["choices"].set_index("heldout_task_id")["chosen_rule"]
check("LOTO choices 8x opportunity + task3 rare", (ch.drop(3) == "v12_signal_aggressive_else_strict").all() and ch.loc[3] == "v11_rare_signal_aggressive_else_strict")
legacy = pd.read_csv(BASE / "ieee_v10_uniform_grid_loto_audit_run1/outer_loto_overall_summary.csv")
check("LOTO == archived LOTO summary", np.isclose(float(legacy["task_macro_mean_delta"].iloc[0]), float(ls["task_macro_mean_delta"]), atol=1e-12))

# matched: 원래 run2 의 overall 표와 legacy5·nested7_veto 6개 arm 대조 (후보 수준 판은 원래 자료에 test 가 없어 건너뛴다)
grid_dummy = pd.DataFrame(columns=["task_id", "seed", "val_gain", "val_fpr_excess", "ddpm_train_mode", "tstr_pr_auc",
                                   "val_PR_AUC", "candidate_key", "delta_vs_safe_baseline_test"])
mo = RA.matched_audit(case, grid_dummy.astype({"task_id": int, "seed": int}), RA.TASKS_ARCHIVE)
ref = pd.read_csv(BASE / "ieee_matched_objective_gate_audit_run2/outer_loto_overall_summary.csv") if (BASE / "ieee_matched_objective_gate_audit_run2/outer_loto_overall_summary.csv").exists() else None
if ref is None:
    cands = sorted((BASE / "ieee_matched_objective_gate_audit_run2").glob("*.csv"))
    print("   matched run2 files:", [c.name for c in cands])
else:
    m = mo.merge(ref, on=["family", "objective"], suffixes=("_new", "_ref"))
    m = m[m["family"].isin(["legacy5", "nested7_veto"])]
    ok = len(m) == 6 and np.allclose(m["task_macro_mean_delta_new"], m["task_macro_mean_delta_ref"], atol=1e-12) \
        and (m["hybrid_selection_count_new"] == m["hybrid_selection_count_ref"]).all()
    check("matched 6 arms == run2", ok, m[["family", "objective", "task_macro_mean_delta_new", "hybrid_selection_count_new"]].to_string(index=False))

# (b) 선택 규칙
qc = pd.read_csv(OUT / "paper_strict/condition_hybrid_alltasks_seed42_46_quick_mean_run2/candidate_val_results.csv", low_memory=False)
qw = pd.read_csv(OUT / "paper_strict/condition_hybrid_alltasks_seed42_46_quick_mean_run2/selected_winners.csv", low_memory=False)
for m_ in RA.METRIC_ORDER + ["FPR"]:
    qc[f"val_{m_}"] = qc[m_]
qc["_key"] = [RA.ckey(a, b, c, d) for a, b, c, d in zip(qc["method"], qc["ddpm_train_mode"], qc["ratio"], qc["s3_ig_keep_rate"])]
qw["_key"] = [RA.ckey(a, b, c, d) for a, b, c, d in zip(qw["method"], qw["ddpm_train_mode"], qw["ratio"], qw["s3_ig_keep_rate"])]
mine = {(t, s, m_): RA.pipeline_winner(g)["_key"] for (t, s, m_), g in qc.groupby(["task_id", "seed", "method"])}
arch = {(t, s, m_): k for t, s, m_, k in zip(qw["task_id"], qw["seed"], qw["method"], qw["_key"])}
same = sum(mine[k] == arch.get(k) for k in mine)
check("grid A family winners == archived selected_winners", same == len(arch) == 135, f"{same}/{len(arch)}")

ref_all = pd.read_csv(OUT / "condition_aware/strategy_selection_candidate_val_all.csv", low_memory=False)
ref_all = ref_all[ref_all["scenario"].isin(["S0", "S1"])].copy()
for m_ in ["PR_AUC", "MCC", "FPR"]:
    ref_all[f"val_{m_}"] = ref_all[m_]
ss = pd.read_csv(OUT / "condition_aware/strategy_selection_results.csv")
ok_n = 0
for (t, s_), g in ref_all.groupby(["task_id", "seed"]):
    r = RA.reference_A(g)
    want = ss[(ss["task_id"] == t) & (ss["seed"] == s_)].iloc[0]
    ok_n += int(RA.ref_label(r) == want["safe_baseline_method"] and np.isclose(r["val_PR_AUC"], want["safe_baseline_val_PR_AUC"], atol=0))
check("reference A == archived strategy selection", ok_n == 45, f"{ok_n}/45")

v10c = []
for p in sorted((OUT / "paper_strict").glob("condition_task*_full_summarystats_v10_s1plus_featurewise_quantile_seed42_46_run1/candidate_val_results.csv")):
    v10c.append(pd.read_csv(p, low_memory=False))
v10c = pd.concat(v10c, ignore_index=True)
for m_ in RA.METRIC_ORDER + ["FPR", "Recall_at_FPR010"]:
    v10c[f"val_{m_}"] = v10c[m_]
v10c["_key"] = [RA.ckey(a, b, c, d) if sc == "S3_HYBRID" else f"{sc}/{a}/ratio={c}"
                for a, b, c, d, sc in zip(v10c["method"], v10c["ddpm_train_mode"], v10c["ratio"], v10c["s3_ig_keep_rate"], v10c["scenario"])]
ok_w = ok_r = 0
for (t, s_), g in v10c.groupby(["task_id", "seed"]):
    want = case[(case["task_id"] == t) & (case["seed"] == s_)].iloc[0]
    w = RA.pipeline_winner(g[g["scenario"].eq("S3_HYBRID")])
    ok_w += int(np.isclose(w["ratio"], want["ratio"]) and np.isclose(w["s3_ig_keep_rate"], want["s3_ig_keep_rate"]))
    r = RA.reference_B(g[g["scenario"].isin(["S0", "S1"])])
    ok_r += int(np.isclose(r["val_PR_AUC"], want["safe_baseline_val_PR_AUC"], atol=0) and str(r["method"]) == str(want["safe_baseline_method"]))
check("grid B fixed winners == archived", ok_w == 45, f"{ok_w}/45")
check("reference B == archived safe baseline", ok_r == 45, f"{ok_r}/45")

print("\n실패:", fails if fails else "없음")
sys.exit(1 if fails else 0)
