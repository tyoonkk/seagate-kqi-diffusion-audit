"""다시 만든 보관 자료에 논문의 감사를 그대로 적용한다 — 제출한 Seagate 논문(MDPI electronics-4573451) 수정 (2026-10).

게이트 판정, 바깥 과제 하나 빼기(LOTO) 선택, 부트스트랩은 원래 감사 스크립트의 함수를 import 해서 그대로 쓴다.
이 파일이 새로 하는 일은 (1) 다시 만든 후보표에서 원래 감사가 읽던 입력표를 같은 규칙으로 만드는 것,
(2) 원래 스크립트의 main 에 박혀 있던 경로·과제 목록·과거 결과 대조·해석 문구를 빼고 같은 계산을 돌리는 것,
(3) 개정에서 추가하는 안정성 분석(과제 하나 빼기 잭나이프, 분류기 사이 일치)을 더하는 것이다.

입력표 규칙 (보관 경로와 같게)
- 주 감사(격자 A)의 reference: S0, 무작위 복제·SMOTE × 비율 {0.5,1,2}, 무작위 줄이기 20:1·10:1 의 9개 중에서
  방법별 우승(검증 PR-AUC↓, MCC↓, FPR↑) → 검증 PR-AUC↓, FPR↑, 복잡도(S0<S1)↑ 순 (select_strategy_noharm.py 와 같음).
- 주 감사 family winner: (과제, seed, 방식)마다 격자 A 8개 후보 중 [PR_AUC, MCC, ROC_AUC, Recall, Precision] 검증값 내림차순 첫 행
  (run_pipeline.select_candidate_winners 와 같음; 완전히 같은 값이면 후보 키 사전순).
- 동일예산 감사(격자 B)의 reference: 15개에서 방법별 우승([PR_AUC, MCC, ROC_AUC, Recall, Precision]↓) → pick_safe_baseline
  (검증 PR-AUC↓, FPR↑, MCC↓, 방법 이름↑) (analyze_task10_v9_targeted_sweep.py 와 같음).
- 격자 B 고정 후보: (과제, seed)마다 24개 중 같은 우승 규칙. 후보 특징(certificate 점수, red flag)은
  analyze_task10_v9_targeted_sweep.add_candidate_features 를 그대로 쓴다.
- 과제 서술자: pos_rate_train = 학습 불량 비율, S0_val_PR_AUC_mean·best_S1·delta_S1_minus_S0 는 compute_task_profiles.baseline_from_candidates 와
  같은 정의로 9개 reference 의 검증 PR-AUC 에서 계산한다(분류기별). grid_mean_val_gain 은 격자 B 24개 후보의 val_gain 평균
  (run_selector_v12_featurewise_signal_gate.add_task_context 와 같음).

분석 (분류기 lgbm, xgb 각각; 과제 묶음은 보관 9개와 전체 11개 각각)
- 주 감사: 7개 게이트 × {conservative, mean_only} 바깥 LOTO (build_ieee_task_independent_nested_audit 의 함수).
- 동일예산 LOTO: 과거 5개 게이트 (build_ieee_v10_uniform_grid_loto_audit 의 함수).
- 맞춘 비교: {과거 5, 단순 7 거부형} × {conservative, mean_only, harm_first} (build_ieee_matched_objective_gate_audit 의 함수).
  다시 만든 자료는 24개 후보 모두 test 값이 있으므로 단순 7 게이트를 후보 수준(통과 후보 중 검증 최고)으로 적용한 판도 함께 낸다.
- 안정성: 바깥 결과의 과제별 평균, 과제 하나를 뺀 과제 거시 평균(잭나이프), 분류기 사이 선택·Δ 일치.

출력 폴더는 한 번만 쓴다. --fake-test 는 코드 경로 시험용으로 test 지표를 무작위 값으로 바꾸며, 그 출력은 결과로 쓰지 않는다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SEAGATE = HERE.parent
ROOT = SEAGATE.parents[1]
sys.path[:0] = [str(SEAGATE)]

import analyze_task10_v9_targeted_sweep as V9S  # noqa: E402
import build_ieee_matched_objective_gate_audit as MA  # noqa: E402
import build_ieee_task_independent_nested_audit as NA  # noqa: E402
import build_ieee_v10_uniform_grid_loto_audit as LA  # noqa: E402

TASKS_ARCHIVE = (0, 1, 2, 3, 5, 6, 7, 9, 10)
TASKS_ALL = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
SEEDS = (42, 43, 44, 45, 46)
METRIC_ORDER = ["PR_AUC", "MCC", "ROC_AUC", "Recall", "Precision"]
REF_A = {("S0", "none", 0.0), ("S1", "random_over", 0.5), ("S1", "random_over", 1.0), ("S1", "random_over", 2.0),
         ("S1", "smote", 0.5), ("S1", "smote", 1.0), ("S1", "smote", 2.0)}
REF_A_UNDER = {"random_under_20to1", "random_under_10to1"}
ROWS_PER_CASE = (15 + 24 + 24) * 2


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def ckey(method, mode, ratio, keep) -> str:
    return f"{method}/mode={mode}/ratio={float(ratio)}/keep={float(keep)}"


# ── 읽기 ─────────────────────────────────────────────────────────────────────
def load_archive(run_dir: Path, tasks, fake_test: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames, cases = [], []
    for t in tasks:
        for s in SEEDS:
            d = run_dir / f"task{t}_seed{s}"
            if not (d / "case.json").exists():
                raise FileNotFoundError(f"사례가 없다: {d}")
            c = pd.read_csv(d / "candidates.csv", low_memory=False)
            if len(c) != ROWS_PER_CASE:
                raise ValueError(f"{d}: 후보 행 수 {len(c)} ≠ {ROWS_PER_CASE}")
            c["task_id"], c["seed"] = int(t), int(s)
            frames.append(c)
            meta = json.loads((d / "case.json").read_text(encoding="utf-8"))
            if not fake_test and (int(meta["task_id"]) != int(t) or int(meta["seed"]) != int(s)):
                raise ValueError(f"{d}: case.json 의 과제·seed 가 폴더 이름과 다르다")
            cases.append({"task_id": int(t), "seed": int(s), **{k: meta[k] for k in ("n_train", "n_pos_train", "n_val", "n_pos_val", "n_features")}})
    cand = pd.concat(frames, ignore_index=True)
    if fake_test:
        rng = np.random.default_rng(12345)
        base = cand["val_PR_AUC"].to_numpy(float)
        cand["test_PR_AUC"] = np.clip(base + rng.normal(0, 0.02, len(cand)), 0, 1)
        for m in ["FPR", "MCC", "Recall_at_FPR010"]:
            cand[f"test_{m}"] = cand.get(f"val_{m}", np.nan)
    need = [f"val_{m}" for m in METRIC_ORDER + ["FPR", "Recall_at_FPR010"]] + ["test_PR_AUC", "classifier", "grid", "scenario", "method"]
    miss = [c for c in need if c not in cand.columns]
    if miss:
        raise ValueError(f"열이 없다: {miss}")
    return cand, pd.DataFrame(cases)


def pipeline_winner(group: pd.DataFrame) -> pd.Series:
    """run_pipeline.select_candidate_winners 의 순서. 완전 동률이면 후보 키 사전순으로 정한다."""
    cols = [f"val_{m}" for m in METRIC_ORDER] + ["_key"]
    return group.sort_values(cols, ascending=[False] * len(METRIC_ORDER) + [True], kind="mergesort").iloc[0]


def ref_label(r: pd.Series) -> str:
    return f"{r['scenario']}/{r['method']}"


# ── reference ───────────────────────────────────────────────────────────────
def reference_A(refs: pd.DataFrame) -> pd.Series:
    """select_strategy_noharm: 방법별 우승(PR_AUC↓, MCC↓, FPR↑) → (PR_AUC↓, FPR↑, 복잡도↑)."""
    in_set = refs.apply(lambda r: (r["scenario"], r["method"], float(r["ratio"])) in REF_A or r["method"] in REF_A_UNDER, axis=1)
    pool = refs[in_set].copy()
    if len(pool) != 9:
        raise ValueError(f"reference A 후보가 9개가 아니다: {len(pool)}")
    pool = pool.sort_values(["val_PR_AUC", "val_MCC", "val_FPR"], ascending=[False, False, True], kind="stable")
    winners = pool.groupby(["scenario", "method"], sort=False).head(1).copy()
    winners["_complexity"] = winners["scenario"].map({"S0": 0, "S1": 1})
    return winners.sort_values(["val_PR_AUC", "val_FPR", "_complexity"], ascending=[False, True, True], kind="stable").iloc[0]


def reference_B(refs: pd.DataFrame) -> pd.Series:
    """파이프라인 방법별 우승 → analyze_task10_v9_targeted_sweep.pick_safe_baseline."""
    if len(refs) != 15:
        raise ValueError(f"reference B 후보가 15개가 아니다: {len(refs)}")
    winners = pd.DataFrame([pipeline_winner(g) for _, g in refs.groupby(["scenario", "method"], sort=True)])
    w = winners.rename(columns={"val_PR_AUC": "val_PR_AUC", "val_FPR": "val_FPR", "val_MCC": "val_MCC"})
    return V9S.pick_safe_baseline(w)


def safe_fields(r: pd.Series) -> dict:
    return {"safe_baseline_method": ref_label(r), "safe_baseline_val_PR_AUC": float(r["val_PR_AUC"]),
            "safe_baseline_val_FPR": float(r["val_FPR"]), "safe_baseline_val_MCC": float(r["val_MCC"]),
            "safe_baseline_val_Recall_at_FPR010": float(r["val_Recall_at_FPR010"]),
            "safe_baseline_test_PR_AUC": float(r["test_PR_AUC"]), "safe_baseline_test_FPR": float(r.get("test_FPR", np.nan)),
            "safe_baseline_test_MCC": float(r.get("test_MCC", np.nan)),
            "safe_baseline_test_Recall_at_FPR010": float(r.get("test_Recall_at_FPR010", np.nan))}


# ── 입력표 만들기 ────────────────────────────────────────────────────────────
def build_inputs(cand: pd.DataFrame, cases: pd.DataFrame, clf: str):
    c = cand[cand["classifier"].eq(clf)].copy()
    c["_key"] = [ckey(m, md, r, k) if g != "ref" else f"{sc}/{m}/ratio={r}"
                 for m, md, r, k, g, sc in zip(c["method"], c.get("ddpm_train_mode", ""), c["ratio"], c.get("keep_rate", np.nan), c["grid"], c["scenario"])]
    nested_rows, case_rows, grid_rows, ref_rows, prof_rows = [], [], [], [], []
    for (t, s), g in c.groupby(["task_id", "seed"], sort=True):
        refs = g[g["grid"].eq("ref")]
        rA, rB = reference_A(refs), reference_B(refs)
        ref_rows.append({"task_id": t, "seed": s, "classifier": clf, "refA": ref_label(rA), "refA_ratio": rA["ratio"],
                         "refB": ref_label(rB), "refB_ratio": rB["ratio"]})
        sA, sB = safe_fields(rA), safe_fields(rB)
        # 격자 A family winner
        A = g[g["grid"].eq("A")]
        for method, gm in A.groupby("method", sort=True):
            if len(gm) != 8:
                raise ValueError(f"격자 A {t}/{s}/{method} 후보 {len(gm)}개")
            w = pipeline_winner(gm)
            nested_rows.append({
                "source": "regen_gridA", "task_id": int(t), "seed": int(s), "method": method, "classifier": clf,
                "candidate_key": w["_key"], "ddpm_train_mode": w["ddpm_train_mode"], "ratio": float(w["ratio"]),
                "s3_ig_keep_rate": float(w["keep_rate"]), "val_PR_AUC": float(w["val_PR_AUC"]), "val_FPR": float(w["val_FPR"]),
                "val_Recall_at_FPR010": float(w["val_Recall_at_FPR010"]), "test_PR_AUC": float(w["test_PR_AUC"]),
                "real_vs_synth_auc": float(w["real_vs_synth_auc"]), "tstr_pr_auc": float(w["tstr_pr_auc"]),
                "with_replacement": bool(w["with_replacement"]), **sA,
                "val_gain": float(w["val_PR_AUC"]) - sA["safe_baseline_val_PR_AUC"],
                "val_fpr_excess": float(w["val_FPR"]) - sA["safe_baseline_val_FPR"],
                "delta_vs_safe_baseline_test": float(w["test_PR_AUC"]) - sA["safe_baseline_test_PR_AUC"]})
        # 격자 B 고정 후보와 24개 격자
        B = g[g["grid"].eq("B")]
        if len(B) != 24:
            raise ValueError(f"격자 B {t}/{s} 후보 {len(B)}개")
        w = pipeline_winner(B)
        case_rows.append({
            "source": "regen_gridB", "task_id": int(t), "seed": int(s), "classifier": clf, "scenario": "S3_HYBRID",
            "method": w["method"], "ratio": float(w["ratio"]), "s3_ig_keep_rate": float(w["keep_rate"]),
            "ddpm_train_mode": w["ddpm_train_mode"], "s3_postprocess": w["s3_postprocess"], "candidate_key": w["_key"],
            "val_PR_AUC": float(w["val_PR_AUC"]), "val_FPR": float(w["val_FPR"]), "val_MCC": float(w["val_MCC"]),
            "val_Recall_at_FPR010": float(w["val_Recall_at_FPR010"]), "test_PR_AUC": float(w["test_PR_AUC"]),
            "test_FPR": float(w.get("test_FPR", np.nan)), "test_MCC": float(w.get("test_MCC", np.nan)),
            "test_Recall_at_FPR010": float(w.get("test_Recall_at_FPR010", np.nan)),
            "real_vs_synth_auc": float(w["real_vs_synth_auc"]), "tstr_pr_auc": float(w["tstr_pr_auc"]),
            "with_replacement": bool(w["with_replacement"]), "safe_baseline_scenario": rB["scenario"], **sB})
        for _, r in B.iterrows():
            grid_rows.append({"task_id": int(t), "seed": int(s), "classifier": clf, "method": r["method"], "ratio": float(r["ratio"]),
                              "s3_ig_keep_rate": float(r["keep_rate"]), "ddpm_train_mode": r["ddpm_train_mode"],
                              "candidate_key": r["_key"], "val_PR_AUC": float(r["val_PR_AUC"]), "val_FPR": float(r["val_FPR"]),
                              "val_MCC": float(r["val_MCC"]), "val_Recall_at_FPR010": float(r["val_Recall_at_FPR010"]),
                              "test_PR_AUC": float(r["test_PR_AUC"]), "real_vs_synth_auc": float(r["real_vs_synth_auc"]),
                              "tstr_pr_auc": float(r["tstr_pr_auc"]), **sB,
                              "val_gain": float(r["val_PR_AUC"]) - sB["safe_baseline_val_PR_AUC"],
                              "val_fpr_excess": float(r["val_FPR"]) - sB["safe_baseline_val_FPR"],
                              "val_recall_fpr010_gain": float(r["val_Recall_at_FPR010"]) - sB["safe_baseline_val_Recall_at_FPR010"],
                              "delta_vs_safe_baseline_test": float(r["test_PR_AUC"]) - sB["safe_baseline_test_PR_AUC"]})
        # 과제 서술자용 reference 9개 (보관 프로필과 같은 집합)
        in_set = refs.apply(lambda r: (r["scenario"], r["method"], float(r["ratio"])) in REF_A or r["method"] in REF_A_UNDER, axis=1)
        for _, r in refs[in_set].iterrows():
            prof_rows.append({"task_id": int(t), "seed": int(s), "scenario": r["scenario"], "method": r["method"],
                              "PR_AUC": float(r["val_PR_AUC"]), "Recall_at_FPR010": float(r["val_Recall_at_FPR010"])})
    nested = pd.DataFrame(nested_rows)
    grid = pd.DataFrame(grid_rows)
    case = V9S.add_candidate_features(pd.DataFrame(case_rows))
    # 과제 서술자 (compute_task_profiles.baseline_from_candidates 와 같은 정의)
    prof = pd.DataFrame(prof_rows)
    ctx_rows = []
    for t, gp in prof.groupby("task_id"):
        s0 = gp[gp["scenario"].eq("S0")]["PR_AUC"]
        s1 = gp[gp["scenario"].eq("S1")].groupby("method")["PR_AUC"].mean().sort_values(ascending=False)
        cs = cases[cases["task_id"].eq(t)].iloc[0]
        ctx_rows.append({"task_id": int(t), "pos_rate_train": cs["n_pos_train"] / cs["n_train"],
                         "S0_val_PR_AUC_mean": float(s0.mean()), "best_S1_method": s1.index[0],
                         "best_S1_val_PR_AUC_mean": float(s1.iloc[0]), "delta_S1_minus_S0": float(s1.iloc[0] - s0.mean())})
    tctx = pd.DataFrame(ctx_rows).merge(grid.groupby("task_id", as_index=False).agg(grid_mean_val_gain=("val_gain", "mean")), on="task_id")
    case = case.merge(tctx, on="task_id", how="left")
    case["validation_stage"] = "regen"
    return nested, case, grid, tctx, pd.DataFrame(ref_rows)


# ── 감사 ─────────────────────────────────────────────────────────────────────
def nested_audit(nested: pd.DataFrame, tasks) -> dict:
    cand = nested[nested["task_id"].isin(tasks)].sort_values(["task_id", "seed", "method"]).reset_index(drop=True)
    gate_rows = {gate.name: NA.apply_gate(cand, gate) for gate in NA.GATES}
    choices, held = [], []
    for regime in ("conservative", "mean_only"):
        for ht in sorted(tasks):
            scores = pd.DataFrame([{"regime": regime, "heldout_task_id": ht, "gate": gate.name,
                                    **NA.summarize(gate_rows[gate.name][gate_rows[gate.name]["task_id"].ne(ht)])} for gate in NA.GATES])
            chosen = NA.choose_gate(scores, regime)
            h = gate_rows[chosen][gate_rows[chosen]["task_id"].eq(ht)].copy()
            h.insert(0, "regime", regime)
            h.insert(1, "heldout_task_id", ht)
            h.insert(2, "chosen_gate", chosen)
            held.append(h)
            choices.append({"regime": regime, "heldout_task_id": ht, "chosen_gate": chosen})
    held = pd.concat(held, ignore_index=True)
    summ = []
    for regime, f in held.groupby("regime", sort=True):
        lo, hi = NA.task_bootstrap_ci(f)
        summ.append({"regime": regime, **NA.summarize(f), "ci_low": lo, "ci_high": hi})
    return {"summary": pd.DataFrame(summ), "choices": pd.DataFrame(choices), "heldout": held,
            "all_gates": pd.concat([df.assign(gate=name) for name, df in gate_rows.items()], ignore_index=True)}


def loto_audit(case: pd.DataFrame, tasks) -> dict:
    cs = LA.derive_validation_signals(case[case["task_id"].isin(tasks)].sort_values(["task_id", "seed"]).reset_index(drop=True))
    frames = {g.name: LA.apply_gate(cs, g.name) for g in LA.GATES}
    choices, held = [], []
    for ht in sorted(tasks):
        scores = pd.DataFrame([{"heldout_task_id": ht, "rule_name": g.name,
                                **LA.summarize(frames[g.name][frames[g.name]["task_id"].ne(ht)])} for g in LA.GATES])
        chosen = LA.choose_rule(scores)
        h = frames[chosen][frames[chosen]["task_id"].eq(ht)].copy()
        h.insert(0, "heldout_task_id", ht)
        h.insert(1, "chosen_rule", chosen)
        held.append(h)
        choices.append({"heldout_task_id": ht, "chosen_rule": chosen})
    held = pd.concat(held, ignore_index=True)
    lo, hi = LA.task_bootstrap(held)
    full = pd.DataFrame([{"rule_name": g.name, **LA.summarize(frames[g.name])} for g in LA.GATES])
    return {"summary": pd.DataFrame([{**LA.summarize(held), "ci_low": lo, "ci_high": hi}]), "choices": pd.DataFrame(choices),
            "heldout": held, "full_archive_rules": full, "signals": cs}


def matched_audit(case: pd.DataFrame, grid: pd.DataFrame, tasks) -> pd.DataFrame:
    """build_ieee_matched_objective_gate_audit.main 과 같은 계산 (과거 결과 대조만 뺌) + 후보 수준 단순 7 게이트 판."""
    cs = MA.derive_validation_signals(case[case["task_id"].isin(tasks)].sort_values(["task_id", "seed"]).reset_index(drop=True))
    families = {
        "legacy5": ({n: MA.apply_case_gate(cs, "legacy5", n, MA.legacy_gate_decision(cs, n)) for n in MA.LEGACY_GATES}, "safe_only"),
        "nested7_veto": ({g.name: MA.apply_case_gate(cs, "nested7_veto", g.name, MA.nested_gate_decision(cs, g)) for g in MA.NESTED_GATES},
                         "reference_only"),
    }
    # 후보 수준 단순 7 게이트: 24개 후보 중 게이트를 통과한 후보에서 원래 주 감사의 정렬로 하나를 고른다
    gr = grid[grid["task_id"].isin(tasks)].copy()
    cand_frames = {}
    for g in (NA.GATES if len(gr) else ()):
        rows = []
        for (t, s), gg in gr.groupby(["task_id", "seed"], sort=True):
            sel = gg[NA.gate_mask(gg, g)]
            if sel.empty:
                rows.append({"family": "nested7_candidate", "gate": g.name, "task_id": int(t), "seed": int(s),
                             "selected_kind": "safe_baseline", "delta_vs_safe_baseline_test": 0.0})
            else:
                p = sel.sort_values(["val_PR_AUC", "val_fpr_excess", "tstr_pr_auc", "candidate_key"],
                                    ascending=[False, True, False, True], kind="mergesort").iloc[0]
                rows.append({"family": "nested7_candidate", "gate": g.name, "task_id": int(t), "seed": int(s),
                             "selected_kind": "hybrid", "delta_vs_safe_baseline_test": float(p["delta_vs_safe_baseline_test"])})
        f = pd.DataFrame(rows)
        f["negative_row"] = f["delta_vs_safe_baseline_test"].lt(0.0)
        cand_frames[g.name] = f
    if len(gr):  # 검증용으로 원래 자료(후보별 test 없음)를 넣을 때는 이 판을 건너뛴다
        families["nested7_candidate"] = (cand_frames, "reference_only")
    out = []
    for family, (frames, fallback) in families.items():
        for objective in ("conservative", "mean_only", "harm_first"):
            held = []
            for ht in sorted(tasks):
                scores = pd.DataFrame([{"gate": n, **MA.summarize(f[f["task_id"].ne(ht)])} for n, f in frames.items()])
                chosen = MA.choose_gate(scores, objective, fallback)
                h = frames[chosen]
                held.append(h[h["task_id"].eq(ht)].assign(chosen_gate=chosen))
            held = pd.concat(held, ignore_index=True)
            lo, hi = MA.task_bootstrap(held)
            out.append({"family": family, "objective": objective, **MA.summarize(held), "ci_low": lo, "ci_high": hi,
                        "chosen_by_fold": ";".join(f"{t}:{c}" for t, c in held.groupby("task_id")["chosen_gate"].first().items())})
    return pd.DataFrame(out)


def candidate_delta_summary(nested: pd.DataFrame, grid: pd.DataFrame, tasks) -> pd.DataFrame:
    """서술용: 선택 없이 후보 자체가 reference 보다 test 에서 나았는지 과제별로 센다 (격자 A family winner 와 격자 B 24개)."""
    rows = []
    for name, df in (("gridA_family_winners", nested), ("gridB_all_candidates", grid)):
        d = df[df["task_id"].isin(tasks)]
        for t, g in d.groupby("task_id"):
            rows.append({"set": name, "task_id": int(t), "n": len(g), "share_positive_delta": float((g["delta_vs_safe_baseline_test"] > 0).mean()),
                         "mean_delta": float(g["delta_vs_safe_baseline_test"].mean()), "max_delta": float(g["delta_vs_safe_baseline_test"].max()),
                         "min_delta": float(g["delta_vs_safe_baseline_test"].min())})
    return pd.DataFrame(rows)


def jackknife(held: pd.DataFrame, delta_col: str) -> pd.DataFrame:
    tm = held.groupby("task_id")[delta_col].mean()
    return pd.DataFrame([{"left_out_task": int(t), "task_macro_mean_delta": float(tm.drop(t).mean())} for t in tm.index])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tasks", default=",".join(map(str, TASKS_ALL)))
    ap.add_argument("--fake-test", action="store_true")
    args = ap.parse_args()
    run_dir, out = Path(args.run_dir), Path(args.out_dir)
    if out.exists():
        print(f"출력 폴더가 이미 있다: {out}")
        return 1
    out.mkdir(parents=True)
    tasks_avail = tuple(int(x) for x in args.tasks.split(","))
    cand, cases = load_archive(run_dir, tasks_avail, args.fake_test)
    task_sets = {"archive9": tuple(t for t in TASKS_ARCHIVE if t in tasks_avail), "all11": tasks_avail}
    manifest = {"schema_id": "seagate-revision-regen-audits-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
                "run_dir": str(run_dir), "fake_test": bool(args.fake_test), "task_sets": task_sets,
                "code_sha256": {p.name: sha(p) for p in [Path(__file__).resolve(), SEAGATE / "build_ieee_task_independent_nested_audit.py",
                                                       SEAGATE / "build_ieee_v10_uniform_grid_loto_audit.py",
                                                       SEAGATE / "build_ieee_matched_objective_gate_audit.py",
                                                       SEAGATE / "analyze_task10_v9_targeted_sweep.py"]},
                "input_sha256": {f"task{t}_seed{s}": sha(run_dir / f"task{t}_seed{s}" / "candidates.csv") for t in tasks_avail for s in SEEDS},
                "results": {}}
    for clf in ("lgbm", "xgb"):
        nested, case, grid, tctx, refs = build_inputs(cand, cases, clf)
        for name, df in (("nested_input", nested), ("loto_case_input", case), ("gridB_candidates", grid), ("task_context", tctx), ("references", refs)):
            df.to_csv(out / f"{clf}_{name}.csv", index=False)
        for set_name, tasks in task_sets.items():
            if len(tasks) < 3:
                continue
            tag = f"{clf}_{set_name}"
            na = nested_audit(nested, tasks)
            la = loto_audit(case, tasks)
            ma = matched_audit(case, grid, tasks)
            na["summary"].to_csv(out / f"{tag}_nested_summary.csv", index=False)
            na["choices"].to_csv(out / f"{tag}_nested_choices.csv", index=False)
            na["heldout"].to_csv(out / f"{tag}_nested_heldout_rows.csv", index=False)
            la["summary"].to_csv(out / f"{tag}_loto_summary.csv", index=False)
            la["choices"].to_csv(out / f"{tag}_loto_choices.csv", index=False)
            la["heldout"].to_csv(out / f"{tag}_loto_heldout_rows.csv", index=False)
            la["full_archive_rules"].to_csv(out / f"{tag}_loto_full_archive_rules.csv", index=False)
            ma.to_csv(out / f"{tag}_matched_summary.csv", index=False)
            candidate_delta_summary(nested, grid, tasks).to_csv(out / f"{tag}_candidate_delta_summary.csv", index=False)
            jk = []
            for regime, f in na["heldout"].groupby("regime"):
                jk.append(jackknife(f, "delta_vs_reference_test").assign(analysis=f"nested_{regime}"))
            jk.append(jackknife(la["heldout"], "delta_vs_safe_baseline_test").assign(analysis="loto"))
            pd.concat(jk, ignore_index=True).to_csv(out / f"{tag}_jackknife.csv", index=False)
            manifest["results"][tag] = {
                "nested": na["summary"].to_dict(orient="records"), "loto": la["summary"].to_dict(orient="records"),
                "matched": ma[["family", "objective", "task_macro_mean_delta", "hybrid_selection_count", "negative_row_count", "ci_low", "ci_high"]].to_dict(orient="records")}
    # 분류기 사이 일치 (같은 사례에서 고른 행동과 Δ)
    agree = []
    for set_name, tasks in task_sets.items():
        for kind, fn in (("nested", "nested_heldout_rows"), ("loto", "loto_heldout_rows")):
            a = pd.read_csv(out / f"lgbm_{set_name}_{fn}.csv")
            b = pd.read_csv(out / f"xgb_{set_name}_{fn}.csv")
            keys = ["regime", "task_id", "seed"] if kind == "nested" else ["task_id", "seed"]
            dcol = "delta_vs_reference_test" if kind == "nested" else "delta_vs_safe_baseline_test"
            m = a.merge(b, on=keys, suffixes=("_lgbm", "_xgb"))
            for grp, mm in (m.groupby("regime") if kind == "nested" else [("loto", m)]):
                agree.append({"task_set": set_name, "analysis": kind if kind == "loto" else f"{kind}_{grp}", "n": len(mm),
                              "same_kind": int(mm["selected_kind_lgbm"].eq(mm["selected_kind_xgb"]).sum()),
                              "both_hybrid": int((mm["selected_kind_lgbm"].eq("hybrid") & mm["selected_kind_xgb"].eq("hybrid")).sum()),
                              "delta_sign_agree_when_both_hybrid": int(((np.sign(mm[f"{dcol}_lgbm"]) == np.sign(mm[f"{dcol}_xgb"]))
                                                                        & mm["selected_kind_lgbm"].eq("hybrid") & mm["selected_kind_xgb"].eq("hybrid")).sum())})
    pd.DataFrame(agree).to_csv(out / "classifier_agreement.csv", index=False)
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(manifest["results"], ensure_ascii=False, indent=1, default=str)[:4000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
