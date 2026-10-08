"""수정본(R1) 원고 본문 숫자 — 계열 A (MDPI electronics-4573451).

원고 문장·표에 들어가는 숫자는 모두 이 스크립트가 고정된 출력 파일에서 읽어 만든다(손으로 옮기지 않는다).
원고 내용 파일은 ⟦키⟧ 자리표시를 쓰고, 빌드할 때 이 JSON 의 표시 문자열로 바뀐다. 없는 키가 있으면 빌드가 멈춘다.

입력(모두 한 번만 쓰는 출력 폴더나 보관 자료)
- outputs/revision_2026_10/regen_results_r1/results_r1.json          주 감사·LOTO·맞춘 비교·분류기 일치·과제 하나 빼기
- outputs/revision_2026_10/regen_audits_v1/*_candidate_delta_summary.csv  선택 없이 본 후보의 test 차이
- outputs/revision_2026_10/regen_archive_v1_run1/task*_seed*/case.json  고친 생성기 진단(격자 A)
- outputs/revision_2026_10/archived_generator_replay_v2/rows.csv        보관 생성기 재학습 진단(v1 은 도구 시간 제한으로 중단, 아래 기록 참고) (rows.sha256 와 대조)
- outputs/paper_strict/condition_task*_full_summarystats_v10_s1plus_featurewise_quantile_seed42_46_run1/*_seed4?.log
  동일예산 격자를 만든 보관 200 epoch 실행의 생성기 학습 손실
- paper/mdpi_electronics_2026/figures_r1/figure4_gate_effects_r1_data.csv  게이트 전체 적용 값(그림 4)
- paper/mdpi_electronics_2026/table1_inventory_r1.csv                    표 1
출력: paper/mdpi_electronics_2026/r1_text_numbers.json  {"values": {키: 표시 문자열}, "raw": {키: 원값}, "inputs_sha256": {...}}
"""

from __future__ import annotations

import glob
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SEAGATE = HERE.parents[1]
OUTR = SEAGATE / "outputs/revision_2026_10"
RES = OUTR / "regen_results_r1/results_r1.json"
AUD = OUTR / "regen_audits_v1"
RUN = OUTR / "regen_archive_v1_run1"
REPLAY = OUTR / "archived_generator_replay_v2"
LOGS_ROOT = SEAGATE / "outputs/paper_strict"
FIG3 = HERE / "figures_r1/figure4_gate_effects_r1_data.csv"
T1 = HERE / "table1_inventory_r1.csv"
OUT = HERE / "r1_text_numbers.json"
MINUS = "−"
ARCH = {"original": "orig", "regenerated": "regen"}
REG = {"conservative": "cons", "mean_only": "mean", "harm_first": "harm"}
FAM = {"legacy5": "leg", "nested7_veto": "veto", "nested7_candidate": "cand"}
ANA = {"nested_conservative": "cons", "nested_mean_only": "mean", "loto": "loto"}

V: dict[str, str] = {}
RAW: dict[str, object] = {}
SHA: dict[str, str] = {}


def sha(p: Path) -> str:
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    SHA[str(p.relative_to(SEAGATE)).replace("\\", "/")] = h
    return h


def _fmt(x: float, d: int, signed: bool) -> str:
    if x == 0:
        return "0." + "0" * d if d else "0"
    s = f"{x:+.{d}f}" if signed else f"{x:.{d}f}"
    return s.replace("-", MINUS)


def put(key: str, raw, text: str) -> None:
    if key in V:
        raise RuntimeError(f"키가 두 번 정의됐다: {key}")
    V[key], RAW[key] = text, raw


def num(key: str, x: float) -> None:
    """같은 값을 여러 형식으로: s6/s4 부호 있음, u6/u4 부호 없음, a 자동(작은 값은 6자리)."""
    x = float(x)
    put(key + ".s6", x, _fmt(x, 6, True))
    put(key + ".s4", x, _fmt(x, 4, True))
    put(key + ".u6", x, _fmt(x, 6, False))
    put(key + ".u4", x, _fmt(x, 4, False))
    put(key + ".a", x, _fmt(x, 6 if (x != 0 and abs(x) < 0.0005) else 4, True))
    put(key + ".p1", x, f"{x:.1f}")


def cnt(key: str, x) -> None:
    put(key, int(x), str(int(x)))


def results_numbers() -> None:
    sha(RES)
    r = json.loads(RES.read_text(encoding="utf-8"))
    for p in r["primary"]:
        b = f"P.{ARCH[p['archive']]}.{p['classifier']}.{p['task_set']}.{REG[p['regime']]}"
        cnt(b + ".hyb", p["hybrid"]); cnt(b + ".neg", p["negative_rows"]); cnt(b + ".negt", p["negative_tasks"])
        cnt(b + ".n", p["n_rows"]); cnt(b + ".ntask", p["n_tasks"])
        num(b + ".tm", p["task_macro"]); num(b + ".lo", p["ci_low"]); num(b + ".hi", p["ci_high"])
    for p in r["loto"]:
        b = f"L.{ARCH[p['archive']]}.{p['classifier']}.{p['task_set']}"
        cnt(b + ".sel", p["selected"]); cnt(b + ".neg", p["negative_rows"]); cnt(b + ".n", p["n_rows"])
        num(b + ".tm", p["task_macro"]); num(b + ".lo", p["ci_low"]); num(b + ".hi", p["ci_high"]); num(b + ".min", p["min_task"])
    above = []
    for p in r["matched"]:
        b = f"M.{ARCH[p['archive']]}.{p['classifier']}.{p['task_set']}.{FAM[p['family']]}.{REG[p['objective']]}"
        cnt(b + ".hyb", p["hybrid"]); cnt(b + ".neg", p["negative_rows"])
        num(b + ".tm", p["task_macro"]); num(b + ".lo", p["ci_low"]); num(b + ".hi", p["ci_high"])
        if p["archive"] == "regenerated" and p["ci_low"] > 0:
            above.append(b)
    regen_arms = [p for p in r["matched"] if p["archive"] == "regenerated"]
    cnt("M.regen.n_arms", len(regen_arms))
    cnt("M.regen.n_arms_ci_above0", len(above))
    RAW["M.regen.arms_ci_above0_list"] = above
    for p in r["classifier_agreement"]:
        b = f"A.{p['task_set']}.{ANA[p['analysis']]}"
        cnt(b + ".n", p["n"]); cnt(b + ".same", p["same_kind"]); cnt(b + ".both", p["both_hybrid"])
        cnt(b + ".agree", p["delta_sign_agree_when_both_hybrid"])
    jk = pd.DataFrame(r["jackknife"])
    for (clf, ts, ana), g in jk.groupby(["classifier", "task_set", "analysis"]):
        b = f"J.{clf}.{ts}.{ANA[ana]}"
        g = g.sort_values("left_out_task")
        imin, imax = g["task_macro_mean_delta"].idxmin(), g["task_macro_mean_delta"].idxmax()
        num(b + ".min", g.loc[imin, "task_macro_mean_delta"]); num(b + ".max", g.loc[imax, "task_macro_mean_delta"])
        cnt(b + ".argmin", g.loc[imin, "left_out_task"]); cnt(b + ".argmax", g.loc[imax, "left_out_task"])
        cnt(b + ".npos", (g["task_macro_mean_delta"] > 0).sum()); cnt(b + ".nneg", (g["task_macro_mean_delta"] < 0).sum())
    jm = jk[jk["analysis"] == "nested_mean_only"].groupby(["classifier", "task_set"])["task_macro_mean_delta"].agg(["min", "max"])
    cnt("J.n_mean", len(jm))
    cnt("J.n_mean_sign_change", ((jm["min"] < 0) & (jm["max"] > 0)).sum())


def candidate_numbers() -> None:
    for clf in ("lgbm", "xgb"):
        for ts in ("archive9", "all11"):
            p = AUD / f"{clf}_{ts}_candidate_delta_summary.csv"
            sha(p)
            d = pd.read_csv(p)
            for gs, g in d.groupby("set"):
                k = {"gridA_family_winners": "A", "gridB_all_candidates": "B"}[gs]
                b = f"C.{clf}.{ts}.{k}"
                for row in g.itertuples():
                    put(f"{b}.t{row.task_id}.share", float(row.share_positive_delta), f"{100 * row.share_positive_delta:.0f}%")
                    num(f"{b}.t{row.task_id}.mean", row.mean_delta)
                    cnt(f"{b}.t{row.task_id}.n", row.n)
                cnt(b + ".n_tasks_all_pos", (g["share_positive_delta"] == 1.0).sum())
                cnt(b + ".n_tasks_none_pos", (g["share_positive_delta"] == 0.0).sum())
                put(b + ".share_min", float(g["share_positive_delta"].min()), f"{100 * g['share_positive_delta'].min():.0f}%")
                put(b + ".share_max", float(g["share_positive_delta"].max()), f"{100 * g['share_positive_delta'].max():.0f}%")
                cnt(b + ".share_argmin", g.loc[g["share_positive_delta"].idxmin(), "task_id"])
                cnt(b + ".share_argmax", g.loc[g["share_positive_delta"].idxmax(), "task_id"])


def gate_numbers() -> None:
    sha(FIG3)
    d = pd.read_csv(FIG3)
    for row in d.itertuples():
        b = f"G.{row.family}.{row.gate}.{row.archive}"
        cnt(b + ".sel", row.selected); cnt(b + ".neg", row.negative); num(b + ".tm", row.task_macro)


def table1_numbers() -> None:
    sha(T1)
    d = pd.read_csv(T1)
    put("T1.posrate_min", float(d.train_pos_rate_pct.min()), f"{d.train_pos_rate_pct.min():.2f}")
    put("T1.posrate_max", float(d.train_pos_rate_pct.max()), f"{d.train_pos_rate_pct.max():.2f}")
    cnt("T1.nfeat_min", d.n_features.min()); cnt("T1.nfeat_max", d.n_features.max())
    put("T1.bin_min", float(d.binary_feature_pct.min()), f"{d.binary_feature_pct.min():.0f}")
    put("T1.bin_max", float(d.binary_feature_pct.max()), f"{d.binary_feature_pct.max():.0f}")
    cnt("T1.n_tasks", len(d))


def corrected_generator() -> pd.DataFrame:
    rows = []
    for case in sorted(RUN.glob("task*_seed*/case.json")):
        sha(case)
        m = json.loads(case.read_text(encoding="utf-8"))
        for g in m["generator_diagnostics"]:
            if g["grid"] != "A":
                continue
            hl = g["holdout_pos_loss"]
            rows.append({"task_id": m["task_id"], "seed": m["seed"], "mode": g["mode"],
                         "cont_ratio": hl["cont_model"] / hl["cont_gauss_indep"] if g["n_cont"] else np.nan,
                         "cont_zero_ratio": hl["cont_model"] / hl["cont_zero"] if g["n_cont"] else np.nan,
                         "bin_ratio": hl["bin_model"] / min(hl["bin_marginal"], hl["bin_copy"]) if g["n_bin"] else np.nan,
                         "learned": bool((not g["n_cont"] or hl["cont_model"] < hl["cont_gauss_indep"]) and
                                         (not g["n_bin"] or hl["bin_model"] < min(hl["bin_marginal"], hl["bin_copy"]))),
                         "c2st": g["c2st_hold"], "tstr_ratio": g["tstr_internal_syn"] / g["tstr_internal_real"],
                         "n_const": g["n_const_syn"], "steps": g["steps_run"], "best_step": g["best_step"]})
    return pd.DataFrame(rows)


def archived_replay() -> pd.DataFrame:
    rows_p, sha_p = REPLAY / "rows.csv", REPLAY / "rows.sha256"
    if not sha_p.exists():
        raise RuntimeError("보관 생성기 재학습이 아직 끝나지 않았다 (rows.sha256 없음)")
    if sha(rows_p) != sha_p.read_text(encoding="utf-8").strip():
        raise RuntimeError("rows.csv 해시가 rows.sha256 과 다르다")
    d = pd.read_csv(rows_p)
    d["zero_ratio"] = d["loss_model"] / d["loss_zero"]
    d["tstr_ratio"] = d["tstr_internal_syn"] / d["tstr_internal_real"]
    d = d.rename(columns={"c2st_hold": "c2st", "n_const_syn": "n_const"})
    return d


def archived_logs() -> pd.DataFrame:
    """동일예산 격자(보관 200 epoch, positive-only)를 실제로 만든 실행 5개의 로그에서 생성기 학습 손실(1·200 epoch)을 읽는다."""
    out = []
    files = sorted(glob.glob(str(LOGS_ROOT / "*v10_s1plus_featurewise_quantile_seed42_46_run1" / "*_seed4?.log")))
    for f in files:
        sha(Path(f))
        seed = int(re.search(r"seed(\d+)\.log$", f).group(1))
        task, ep1 = None, None
        for line in open(f, encoding="utf-8", errors="replace"):
            m = re.search(r"running task=(\d+)", line)
            if m:
                task, ep1 = int(m.group(1)), None
                continue
            m = re.search(r"\[TorchDDPM\] epoch=(\d+) loss=([\d.]+)", line)
            if m and task is not None:
                ep, loss = int(m.group(1)), float(m.group(2))
                if ep == 1:
                    ep1 = loss
                elif ep == 200:
                    out.append({"task_id": task, "seed": seed, "mode": "positive_only", "loss_ep1": ep1, "loss_ep200": loss})
    d = pd.DataFrame(out)
    if len(d) != 9 * 5 or d.duplicated(["task_id", "seed"]).any():
        raise RuntimeError(f"동일예산 격자 로그에서 과제·seed 마다 학습 손실 하나씩(45개)을 읽지 못했다: {len(d)}")
    return d


def generator_numbers() -> None:
    nfeat = pd.read_csv(T1).set_index("task_id")["n_features"]
    cg = corrected_generator()
    cg["const_pct"] = 100 * cg["n_const"] / cg["task_id"].map(nfeat)
    cnt("GEN.corr.n_fits", len(cg)); cnt("GEN.corr.n_learned", cg["learned"].sum())
    if int(cg["learned"].sum()) != len(cg):
        raise RuntimeError("학습 확인을 통과하지 못한 고친 생성기가 있다 — 본문 서술을 바꿔야 한다")
    for col in ("cont_ratio", "bin_ratio", "cont_zero_ratio", "c2st", "tstr_ratio", "const_pct"):
        num(f"GEN.corr.{col}.min", cg[col].min()); num(f"GEN.corr.{col}.max", cg[col].max()); num(f"GEN.corr.{col}.med", cg[col].median())
    cnt("GEN.corr.n_c2st_ge099", (cg["c2st"] >= 0.99).sum())
    cnt("GEN.corr.steps.max", cg["steps"].max()); cnt("GEN.corr.best_step.max", cg["best_step"].max())
    max_steps = json.loads((SEAGATE / "revision_2026_10/generator_config_v3_final.json").read_text(encoding="utf-8"))["train_steps"]
    cnt("GEN.corr.n_hit_max_steps", (cg["steps"] >= max_steps).sum())
    for t, g in cg.groupby("task_id"):
        num(f"GEN.corr.t{t}.c2st.med", g["c2st"].median()); num(f"GEN.corr.t{t}.tstr_ratio.med", g["tstr_ratio"].median())
    ar = archived_replay()
    ar["const_pct"] = 100 * ar["n_const"] / ar["task_id"].map(nfeat)
    cnt("GEN.arch.n_rows", len(ar))
    # 원고의 보관 자료에 실제로 쓰인 설정: 주 감사 후보원 = 10 epoch × 두 모드, 동일예산 격자 = 200 epoch × positive-only
    subsets = {"quick": ar[ar["setting"] == "quick_10ep"],
               "full_pos": ar[(ar["setting"] == "full_200ep") & (ar["mode"] == "positive_only")],
               "full_cm": ar[(ar["setting"] == "full_200ep") & (ar["mode"] == "conditional_mixed")]}
    if len(subsets["quick"]) != 110 or len(subsets["full_pos"]) != 55:
        raise RuntimeError("보관 생성기 재학습 행 수가 예상(10 epoch 110, 200 epoch positive-only 55)과 다르다")
    for k, g in subsets.items():
        b = f"GEN.arch.{k}"
        cnt(b + ".n", len(g))
        for col in ("zero_ratio", "c2st", "tstr_ratio", "const_pct"):
            num(f"{b}.{col}.min", g[col].min()); num(f"{b}.{col}.max", g[col].max()); num(f"{b}.{col}.med", g[col].median())
        tasks_nl = sorted(g.groupby("task_id")["zero_ratio"].min().loc[lambda x: x >= 0.995].index.tolist())
        RAW[b + ".tasks_no_learning"] = tasks_nl
        V[b + ".tasks_no_learning"] = " and ".join(map(str, tasks_nl)) if 0 < len(tasks_nl) <= 2 else (", ".join(map(str, tasks_nl)) or "none")
        cnt(b + ".n_tasks_no_learning", len(tasks_nl))
        cnt(b + ".n_tasks_learned", g["task_id"].nunique() - len(tasks_nl))
        learned = g.groupby("task_id")["zero_ratio"].min().loc[lambda x: x < 0.995]
        if len(learned):
            sub = g[g["task_id"].isin(learned.index)]
            num(f"{b}.learned_tasks.zero_ratio.min", sub["zero_ratio"].min()); num(f"{b}.learned_tasks.zero_ratio.max", sub["zero_ratio"].max())
        if tasks_nl:
            sub = g[g["task_id"].isin(tasks_nl)]
            num(f"{b}.nl_tasks.zero_ratio.min", sub["zero_ratio"].min()); num(f"{b}.nl_tasks.zero_ratio.max", sub["zero_ratio"].max())
        for t, gt in g.groupby("task_id"):
            num(f"{b}.t{t}.c2st.med", gt["c2st"].median()); num(f"{b}.t{t}.tstr_ratio.med", gt["tstr_ratio"].median())
        m = g.merge(cg, on=["task_id", "seed", "mode"], suffixes=("_a", "_c"))
        bp = f"GEN.pair.{k}"
        cnt(bp + ".n", len(m))
        cnt(bp + ".c2st_corr_lower", (m["c2st_c"] < m["c2st_a"]).sum())
        cnt(bp + ".tstr_corr_higher", (m["tstr_ratio_c"] > m["tstr_ratio_a"]).sum())
        cnt(bp + ".const_corr_lower", (m["n_const_c"] < m["n_const_a"]).sum())
        num(bp + ".c2st_diff_med", (m["c2st_c"] - m["c2st_a"]).median())
        num(bp + ".tstr_diff_med", (m["tstr_ratio_c"] - m["tstr_ratio_a"]).median())
    lg = archived_logs()
    cnt("GEN.log.n", len(lg))
    per = lg.groupby("task_id")["loss_ep200"].agg(["min", "max"])
    flat = sorted(per[per["min"] >= 0.995].index.tolist())
    RAW["GEN.log.tasks_flat"] = flat
    V["GEN.log.tasks_flat"] = " and ".join(map(str, flat)) if len(flat) <= 2 else ", ".join(map(str, flat))
    num("GEN.log.flat.min", per.loc[flat, "min"].min()); num("GEN.log.flat.max", per.loc[flat, "max"].max())
    other = per.drop(index=flat)
    num("GEN.log.other.min", other["min"].min()); num("GEN.log.other.max", other["max"].max())
    for t, row in per.iterrows():
        num(f"GEN.log.t{t}.min", row["min"]); num(f"GEN.log.t{t}.max", row["max"])
    # 잡음 일정 끝의 신호 비율 ᾱ_T: 보관 설정은 보관 실행의 config.json, 고친 설정은 생성기 설정 파일에서 읽는다
    arch_cfg_p = SEAGATE / "outputs/paper_strict/condition_hybrid_alltasks_seed42_46_quick_mean_run2_seed42/config.json"
    corr_cfg_p = SEAGATE / "revision_2026_10/generator_config_v3_final.json"
    sha(arch_cfg_p)
    ac, cc = json.loads(arch_cfg_p.read_text(encoding="utf-8")), json.loads(corr_cfg_p.read_text(encoding="utf-8"))
    sup = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
    for k, (T, b0, b1) in {"arch": (ac["ddpm_T"], ac["ddpm_beta_start"], ac["ddpm_beta_end"]),
                           "corr": (cc["T"], cc["beta_start"], cc["beta_end"])}.items():
        ab = float(np.prod(1.0 - np.linspace(b0, b1, int(T))))
        if ab > 0.01:
            txt = f"{ab:.2f}"
        else:
            m, e = f"{ab:.0e}".split("e")
            txt = f"{int(m)} × 10" + str(int(e)).translate(sup)
        put(f"GEN.abarT.{k}", ab, txt)
    for k in ("ddpm_T", "ddpm_hidden_dim", "ddpm_n_layers"):
        cnt(f"GEN.archcfg.{k}", ac[k])
    put("GEN.archcfg.ddpm_beta_end", ac["ddpm_beta_end"], f"{ac['ddpm_beta_end']:g}")


def dev_numbers() -> None:
    """원래 연구의 과제별 개발 정책(최종 결합 선택기) — 보관 선택 행에서 다시 계산."""
    p = SEAGATE / "artifacts/condition_aware_main/v15_final_combined_selector_report_run1/v15_final_combined_selected_rows.csv"
    sha(p)
    d = pd.read_csv(p)
    x = d["delta_vs_v12_safe"]
    if len(d) != 45:
        raise RuntimeError("개발 정책 선택 행이 45개가 아니다")
    cnt("DEV.sel", (x != 0).sum()); cnt("DEV.neg", (x < 0).sum())
    num("DEV.tm", d.groupby("task_id")["delta_vs_v12_safe"].mean().mean()); num("DEV.selmean", x[x != 0].mean())


def det_numbers() -> None:
    p = SEAGATE / "outputs/revision_2026_10/classifier_determinism_v1/result.json"
    sha(p)
    r = json.loads(p.read_text(encoding="utf-8"))["results"]
    if not (r["lightgbm_cpu"]["identical"] and r["xgboost_cuda"]["identical"]):
        raise RuntimeError("CPU LightGBM 또는 CUDA XGBoost 가 결정적이지 않다 — 4.8절 서술을 바꿔야 한다")
    put("DET.lgbm_gpu_maxdiff", r["lightgbm_gpu"]["max_abs_diff"], f"{r['lightgbm_gpu']['max_abs_diff']:.2f}")


def archive_fact_numbers() -> None:
    """원래 자료의 두 사실: 1–99% 자르기로 상수가 되는 특징 비율(count_clip_collapse_v1), test 재학습 때문에 달라진 임계값."""
    cp = SEAGATE / "outputs/revision_2026_10/clip_collapse_v1/clip_collapse.csv"
    sha(cp)
    c = pd.read_csv(cp)
    put("CLIP.pct_min", float(c.pct_collapsed.min()), f"{c.pct_collapsed.min():.0f}")
    put("CLIP.pct_max", float(c.pct_collapsed.max()), f"{c.pct_collapsed.max():.0f}")
    put("CLIP.out_min", float(c.pct_pos_rows_outside_range.min()), f"{c.pct_pos_rows_outside_range.min():.0f}")
    put("CLIP.out_max", float(c.pct_pos_rows_outside_range.max()), f"{c.pct_pos_rows_outside_range.max():.0f}")
    q = SEAGATE / "outputs/paper_strict/condition_hybrid_alltasks_seed42_46_quick_mean_run2/final_test_results.csv"
    sha(q)
    d = pd.read_csv(q)
    v = []
    for f in sorted(glob.glob(str(LOGS_ROOT / "*v10_s1plus_featurewise_quantile_seed42_46_run1" / "final_test_results.csv"))):
        sha(Path(f))
        v.append(pd.read_csv(f))
    v = pd.concat(v)
    if len(d) != 135 or len(v) != 270:
        raise RuntimeError(f"원래 자료 test 결과 행 수가 예상과 다르다: {len(d)}, {len(v)}")
    same = int((d.Threshold == d.val_Threshold).sum()) + int((v.Threshold == v.val_Threshold).sum())
    cnt("THR.n_total", len(d) + len(v)); cnt("THR.n_same", same)
    rel = ((d.Threshold - d.val_Threshold).abs() / d.val_Threshold.abs()).median()
    put("THR.quick.rel_med_pct", float(100 * rel), f"{100 * rel:.1f}")


def misc_numbers() -> None:
    r = json.loads(RES.read_text(encoding="utf-8"))
    mean_rows = [p["task_macro"] for p in r["primary"] if p["regime"] == "mean_only"]
    num("X.primary_mean_min", min(mean_rows)); num("X.primary_mean_max", max(mean_rows))
    d = pd.read_csv(FIG3)
    for fam in ("historical", "simple"):
        for arch in ("orig", "lgbm", "xgb"):
            g = d[(d.family == fam) & (d.archive == arch) & ~d.gate.isin(["safe_only", "reference_only"])]
            b = f"G.{fam}.{arch}.nonref"
            cnt(b + ".sel_min", g.selected.min()); cnt(b + ".sel_max", g.selected.max())
            cnt(b + ".neg_min", g.negative.min()); cnt(b + ".neg_max", g.negative.max())
            num(b + ".tm_min", g.task_macro.min()); num(b + ".tm_max", g.task_macro.max())
    h = pd.read_csv(AUD / "xgb_all11_nested_heldout_rows.csv")
    tm = h[h.regime == "mean_only"].groupby("task_id")["delta_vs_reference_test"].mean()
    cnt("H.regen.xgb.all11.mean.argmax_task", tm.idxmax())


def heldout_numbers() -> None:
    """주 감사의 바깥(held-out) 행: 보관·과제·목표별 반복 5개의 평균, 최솟값, 최댓값, 하이브리드 수, 음수 수."""
    srcs = {("orig", "lgbm", "archive9"): SEAGATE / "artifacts/condition_aware_main/ieee_task_independent_nested_audit_run1/outer_heldout_rows.csv"}
    for clf in ("lgbm", "xgb"):
        for ts in ("archive9", "all11"):
            srcs[("regen", clf, ts)] = AUD / f"{clf}_{ts}_nested_heldout_rows.csv"
    for (arch, clf, ts), path in srcs.items():
        sha(path)
        d = pd.read_csv(path)
        for (reg, t), g in d.groupby(["regime", "task_id"]):
            if len(g) != 5:
                raise RuntimeError(f"{path} {reg} 과제 {t}: 반복 5행이 아니다")
            b = f"H.{arch}.{clf}.{ts}.{REG[reg]}.t{t}"
            x = g["delta_vs_reference_test"]
            num(b + ".mean", x.mean()); num(b + ".min", x.min()); num(b + ".max", x.max())
            cnt(b + ".nhyb", (g["selected_kind"] != "reference").sum()); cnt(b + ".nneg", (x < 0).sum())


def config_numbers() -> None:
    cfgp = SEAGATE / "revision_2026_10/generator_config_v3_final.json"
    sha(cfgp)
    c = json.loads(cfgp.read_text(encoding="utf-8"))
    for k in ("T", "hidden_dim", "n_blocks", "train_steps", "batch_size", "warmup_steps", "eval_every", "patience", "min_holdout"):
        put(f"GEN.cfg.{k}", c[k], f"{int(c[k]):,}")
    for k in ("dropout", "grad_clip", "ema_decay", "beta_end", "holdout_frac"):
        put(f"GEN.cfg.{k}", c[k], f"{c[k]:g}")
    put("GEN.cfg.holdout_pct", c["holdout_frac"], f"{100 * c['holdout_frac']:.0f}%")
    def sci(x):  # 1e-4 → 1 × 10⁻⁴
        m, e = f"{x:.0e}".split("e")
        sup = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
        return f"{int(m)} × 10" + str(int(e)).translate(sup)
    for k in ("beta_start", "lr", "weight_decay"):
        put(f"GEN.cfg.{k}", c[k], sci(c[k]))
    m = json.loads((RUN / "manifest.json").read_text(encoding="utf-8"))
    if m["generator_cfg"] != c:
        raise RuntimeError("재생성 실행 manifest 의 생성기 설정이 설정 파일과 다르다")
    env = m["environment"]
    for k in ("python", "torch", "torch_cuda", "numpy", "pandas", "scikit_learn", "imbalanced_learn", "lightgbm", "xgboost", "scipy", "gpu"):
        put(f"ENV.{k}", env[k], str(env[k]).split("+")[0] if k == "torch" else str(env[k]))
    sha(RUN / "manifest.json")
    log = (RUN / "run.log").read_text(encoding="utf-8")
    sha(RUN / "run.log")
    mins = [float(x) for x in re.findall(r"과제 \d+ seed \d+: 끝 \(([\d.]+)분\)", log)]
    if len(mins) != 55:
        raise RuntimeError(f"run.log 에서 끝난 사례 55개의 시간을 읽지 못했다: {len(mins)}")
    put("RUN.case_min_min", min(mins), f"{min(mins):.0f}"); put("RUN.case_min_max", max(mins), f"{max(mins):.0f}")
    put("RUN.total_hours", sum(mins) / 60, f"{sum(mins) / 60:.0f}")


def main() -> int:
    results_numbers()
    candidate_numbers()
    gate_numbers()
    table1_numbers()
    generator_numbers()
    heldout_numbers()
    config_numbers()
    dev_numbers()
    det_numbers()
    archive_fact_numbers()
    misc_numbers()
    words = "zero one two three four five six seven eight nine ten eleven twelve".split()
    for k, v in list(RAW.items()):
        if isinstance(v, (int, np.integer)) and not isinstance(v, bool) and 0 <= int(v) < len(words) and k in V and k + ".w" not in V:
            V[k + ".w"] = words[int(v)]
    OUT.write_text(json.dumps({"schema_id": "seagate-mdpi-r1-text-numbers-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
                               "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                               "inputs_sha256": SHA, "values": V, "raw": RAW}, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"ok: {len(V)} values, {len(SHA)} inputs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
