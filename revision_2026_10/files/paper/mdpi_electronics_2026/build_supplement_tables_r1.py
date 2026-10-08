"""수정본 보충자료의 생성 표 — 계열 A (MDPI electronics-4573451).

고정된 출력에서 LaTeX 표 조각을 만든다(손으로 옮기지 않는다). 출력은 supplement_src/generated_*_r1.tex.
- S8  맞춘 비교: 원래 자료 6개 팔 + 재생성 자료(LightGBM·XGBoost × 9·11개 과제 × 세 게이트 묶음 × 세 목표)
- S15 생성기 진단: 과제 × 생성기(보관 10 epoch, 보관 200 epoch positive-only, 고친 생성기)의 중앙값
- S16 선택 없이 본 후보의 test 차이: 과제 × 분류기 × 격자
- S17 과제 하나 빼기 민감도와 분류기 일치
- S18 재생성 자료 주 감사의 과제별 바깥 효과(9·11개 과제)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SEAGATE = HERE.parents[1]
OUTR = SEAGATE / "outputs/revision_2026_10"
AUD = OUTR / "regen_audits_v1"
RUN = OUTR / "regen_archive_v1_run1"
REPLAY = OUTR / "archived_generator_replay_v2"
SUP = HERE / "supplement_src"
RES = json.loads((OUTR / "regen_results_r1/results_r1.json").read_text(encoding="utf-8"))
NFEAT = pd.read_csv(HERE / "table1_inventory_r1.csv").set_index("task_id")["n_features"]
M = "$-$"


def f6(x: float) -> str:
    if x == 0:
        return "0.000000"
    return (f"{x:+.6f}").replace("-", M)


def f4(x: float) -> str:
    if x == 0:
        return "0.0000"
    if abs(x) < 5e-5:  # 4자리로 줄이면 ±0.0000 이 되는 작은 값은 6자리로 보인다
        return f6(x)
    return (f"{x:+.4f}").replace("-", M)


def u3(x: float) -> str:
    return f"{x:.3f}"


def ci(lo: float, hi: float) -> str:
    return f"$[{f6(lo).replace(M, '-')}, {f6(hi).replace(M, '-')}]$"


FAM = {"legacy5": "Historical", "nested7_veto": "Simple, veto", "nested7_candidate": "Simple, candidate"}
OBJ = {"conservative": "conservative", "mean_only": "mean-seeking", "harm_first": "harm-first"}
CLF = {"lgbm": "LightGBM", "xgb": "XGBoost"}


def s8() -> str:
    rows = []
    for p in RES["matched"]:
        arch = "Original" if p["archive"] == "original" else "Rebuilt"
        n = 45 if p["task_set"] == "archive9" else 55
        rows.append(f"{arch} & {CLF[p['classifier']]} & {9 if p['task_set'] == 'archive9' else 11} & {FAM[p['family']]} & "
                    f"{OBJ[p['objective']]} & {p['hybrid']}/{n} & {p['negative_rows']} & {f6(p['task_macro'])} & {ci(p['ci_low'], p['ci_high'])} \\\\")
    head = (r"\begin{longtable}{l l c l l c c r c}" "\n"
            r"\caption{Matched-objective comparison on the equal-budget candidates: all family--objective arms for both archives. "
            r"In the original archive, only the validation-best candidate has a test outcome, so the simple gates act as case-level vetoes; "
            r"in the rebuilt archive, they are also applied at the candidate level. Negative counts selected runs with a negative test "
            r"difference. Intervals are descriptive endpoint-resampling intervals (10{,}000 resamples, seed 20260714).}\label{tab:s_matched}\\" "\n"
            r"\toprule Archive & Classifier & End. & Gate family & Objective & Hybrids & Neg. & Task-macro $\Delta$ & 95\% interval \\ \midrule"
            "\n" r"\endfirsthead" "\n" r"\toprule Archive & Classifier & End. & Gate family & Objective & Hybrids & Neg. & Task-macro $\Delta$ & 95\% interval \\ \midrule"
            "\n" r"\endhead" "\n" r"\bottomrule" "\n" r"\endfoot" "\n")
    return "{\\footnotesize\n" + head + "\n".join(rows) + "\n\\end{longtable}\n}\n"


def generator_frame() -> pd.DataFrame:
    rows = []
    for case in sorted(RUN.glob("task*_seed*/case.json")):
        m = json.loads(case.read_text(encoding="utf-8"))
        for g in m["generator_diagnostics"]:
            if g["grid"] != "A":
                continue
            hl = g["holdout_pos_loss"]
            rows.append({"gen": "Corrected", "task_id": m["task_id"], "seed": m["seed"], "mode": g["mode"],
                         "learn": hl["total_model"] / hl["total_baseline"], "c2st": g["c2st_hold"],
                         "tstr": g["tstr_internal_syn"] / g["tstr_internal_real"], "n_const": g["n_const_syn"]})
    r = pd.read_csv(REPLAY / "rows.csv")
    r["learn"] = r["loss_model"] / np.minimum(r["loss_zero"], r["loss_gauss"])
    r["c2st"] = r["c2st_hold"]
    r["tstr"] = r["tstr_internal_syn"] / r["tstr_internal_real"]
    r["n_const"] = r["n_const_syn"]
    a10 = r[r["setting"] == "quick_10ep"].assign(gen="Archived, 10 epochs")
    a200 = r[(r["setting"] == "full_200ep") & (r["mode"] == "positive_only")].assign(gen="Archived, 200 epochs (pos.-only)")
    cols = ["gen", "task_id", "seed", "mode", "learn", "c2st", "tstr", "n_const"]
    d = pd.concat([pd.DataFrame(rows)[cols], a10[cols], a200[cols]], ignore_index=True)
    d["const_pct"] = 100 * d["n_const"] / d["task_id"].map(NFEAT)
    return d


def s15() -> str:
    d = generator_frame()
    order = ["Archived, 10 epochs", "Archived, 200 epochs (pos.-only)", "Corrected"]
    out = []
    for t in sorted(d.task_id.unique()):
        for k, gname in enumerate(order):
            g = d[(d.task_id == t) & (d.gen == gname)]
            out.append(f"{t if k == 0 else ''} & {gname} & {len(g)} & {u3(g.learn.median())} & {u3(g.c2st.median())} & "
                       f"{u3(g.tstr.median())} & {g.const_pct.median():.1f} \\\\")
        out.append(r"\midrule" if t != max(d.task_id) else "")
    head = (r"\begin{longtable}{c l c c c c c}" "\n"
            r"\caption{Generator diagnostics on held-out training positives, by endpoint (medians over repeats and training modes). "
            r"Loss ratio: held-out denoising loss divided by that of the best trivial predictor (archived: predicting zero noise; "
            r"corrected: independent-Gaussian predictor for continuous features plus the better of the training-rate and copy "
            r"predictors for binary features); values below 1 indicate learning. Two-sample AUC: five-fold LightGBM AUC between held-out "
            r"real positives and generated positives, averaged over five subsamples. TSTR ratio: PR-AUC of a classifier trained with "
            r"generated positives divided by that with real positives. Constant: percentage of an endpoint's predictors that are constant "
            r"in every generated row although they vary among real training positives. The archived configurations were refitted on the "
            r"same rows as the corrected generator.}\label{tab:s_generator}\\" "\n"
            r"\toprule Endpoint & Generator & Fits & Loss ratio & Two-sample AUC & TSTR ratio & Constant (\%) \\ \midrule" "\n"
            r"\endfirsthead" "\n" r"\toprule Endpoint & Generator & Fits & Loss ratio & Two-sample AUC & TSTR ratio & Constant (\%) \\ \midrule"
            "\n" r"\endhead" "\n" r"\bottomrule" "\n" r"\endfoot" "\n")
    return "{\\footnotesize\n" + head + "\n".join(x for x in out if x) + "\n\\end{longtable}\n}\n"


def s16() -> str:
    parts = {}
    for clf in ("lgbm", "xgb"):
        d = pd.read_csv(AUD / f"{clf}_all11_candidate_delta_summary.csv")
        for gs, k in (("gridA_family_winners", "A"), ("gridB_all_candidates", "B")):
            parts[(clf, k)] = d[d.set == gs].set_index("task_id")
    out = []
    for t in range(11):
        cells = []
        for clf in ("lgbm", "xgb"):
            for k in ("A", "B"):
                r = parts[(clf, k)].loc[t]
                cells.append(f"{100 * r.share_positive_delta:.0f}\\% & {f4(r.mean_delta)}")
        out.append(f"{t} & " + " & ".join(cells) + r" \\")
    head = (r"\begin{table}[H]" "\n" r"\centering" "\n"
            r"\caption{Test differences of individual hybrid candidates from the run-local reference without any selection, rebuilt archive, "
            r"by endpoint: share of candidates with a positive difference and mean difference. Grid A: the three family winners per "
            r"endpoint--repeat that enter the nested audit (15 per endpoint); grid B: all 24 equal-budget candidates per endpoint--repeat "
            r"(120 per endpoint).}\label{tab:s_candidates}" "\n" r"\footnotesize" "\n"
            r"\begin{tabular}{c c r c r c r c r}" "\n" r"\toprule" "\n"
            r"& \multicolumn{4}{c}{LightGBM} & \multicolumn{4}{c}{XGBoost} \\ \cmidrule(lr){2-5}\cmidrule(lr){6-9}" "\n"
            r"& \multicolumn{2}{c}{Grid A} & \multicolumn{2}{c}{Grid B} & \multicolumn{2}{c}{Grid A} & \multicolumn{2}{c}{Grid B} \\" "\n"
            r"Endpoint & Pos. & Mean $\Delta$ & Pos. & Mean $\Delta$ & Pos. & Mean $\Delta$ & Pos. & Mean $\Delta$ \\ \midrule" "\n")
    return head + "\n".join(out) + "\n" + r"\bottomrule" + "\n" + r"\end{tabular}" + "\n" + r"\end{table}" + "\n"


def s17() -> str:
    jk = pd.DataFrame(RES["jackknife"])
    ana = {"nested_conservative": "Nested, conservative", "nested_mean_only": "Nested, mean-seeking", "loto": "Historical LOTO"}
    full = {}
    for p in RES["primary"]:
        if p["archive"] == "regenerated":
            full[(p["classifier"], p["task_set"], "nested_conservative" if p["regime"] == "conservative" else "nested_mean_only")] = p["task_macro"]
    for p in RES["loto"]:
        if p["archive"] == "regenerated":
            full[(p["classifier"], p["task_set"], "loto")] = p["task_macro"]
    out = []
    for (clf, ts, a), g in jk.groupby(["classifier", "task_set", "analysis"]):
        g = g.set_index("left_out_task")["task_macro_mean_delta"]
        out.append(f"{CLF[clf]} & {9 if ts == 'archive9' else 11} & {ana[a]} & {f6(full[(clf, ts, a)])} & {f6(g.min())} ({g.idxmin()}) & "
                   f"{f6(g.max())} ({g.idxmax()}) \\\\")
    agree = []
    for p in RES["classifier_agreement"]:
        agree.append(f"{9 if p['task_set'] == 'archive9' else 11} & {ana[p['analysis']]} & {p['n']} & {p['same_kind']} & "
                     f"{p['both_hybrid']} & {p['delta_sign_agree_when_both_hybrid']} \\\\")
    t1 = (r"\begin{table}[H]" "\n" r"\centering" "\n"
          r"\caption{Leave-one-endpoint-out sensitivity of the task-macro summaries, rebuilt archive. Each summary is recomputed with one "
          r"endpoint's held-out block removed while the outer choices are kept fixed; parentheses give the omitted endpoint.}"
          r"\label{tab:s_jackknife}" "\n" r"\footnotesize" "\n" r"\begin{tabular}{l c l r r r}" "\n" r"\toprule" "\n"
          r"Classifier & End. & Analysis & All endpoints & Minimum (omitted) & Maximum (omitted) \\ \midrule" "\n"
          + "\n".join(out) + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n" r"\end{table}" "\n")
    t2 = (r"\begin{table}[H]" "\n" r"\centering" "\n"
          r"\caption{Agreement between LightGBM and XGBoost on the same held-out rows of the rebuilt archive. Same kind: both keep the "
          r"reference or both select a hybrid. Sign agreement is counted among the rows in which both select a hybrid.}"
          r"\label{tab:s_agreement}" "\n" r"\footnotesize" "\n" r"\begin{tabular}{c l c c c c}" "\n" r"\toprule" "\n"
          r"End. & Analysis & Rows & Same kind & Both hybrid & Same sign \\ \midrule" "\n"
          + "\n".join(agree) + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n" r"\end{table}" "\n")
    return t1 + "\n" + t2


def s18() -> str:
    blocks = []
    for ts in ("archive9", "all11"):
        rows = []
        dd = {clf: pd.read_csv(AUD / f"{clf}_{ts}_nested_heldout_rows.csv") for clf in ("lgbm", "xgb")}
        for t in sorted(dd["lgbm"].task_id.unique()):
            cells = []
            for clf in ("lgbm", "xgb"):
                for reg in ("mean_only", "conservative"):
                    g = dd[clf][(dd[clf].task_id == t) & (dd[clf].regime == reg)]
                    nh = int((g.selected_kind != "reference").sum())
                    cells.append(f"{nh}/5 & {f4(g.delta_vs_reference_test.mean())}")
            rows.append(f"{t} & " + " & ".join(cells) + r" \\")
        blocks.append((ts, rows))
    out = (r"\begin{table}[H]" "\n" r"\centering" "\n"
           r"\caption{Held-out effects of the primary nested audit by endpoint in the rebuilt archive: hybrids selected among the five "
           r"repeats and the mean test difference. The outer choice for each endpoint is made on the other endpoints of the same set, so "
           r"the nine- and 11-endpoint analyses can differ for the same endpoint.}\label{tab:s_heldout_regen}" "\n" r"\footnotesize" "\n"
           r"\begin{tabular}{c c r c r c r c r}" "\n" r"\toprule" "\n"
           r"& \multicolumn{4}{c}{LightGBM} & \multicolumn{4}{c}{XGBoost} \\ \cmidrule(lr){2-5}\cmidrule(lr){6-9}" "\n"
           r"& \multicolumn{2}{c}{Mean-seeking} & \multicolumn{2}{c}{Conservative} & \multicolumn{2}{c}{Mean-seeking} & \multicolumn{2}{c}{Conservative} \\" "\n"
           r"Endpoint & Hyb. & Mean $\Delta$ & Hyb. & Mean $\Delta$ & Hyb. & Mean $\Delta$ & Hyb. & Mean $\Delta$ \\ \midrule" "\n")
    for ts, rows in blocks:
        out += r"\multicolumn{9}{l}{\emph{" + ("Nine original endpoints (primary analysis)" if ts == "archive9" else "All 11 endpoints") + r"}} \\" "\n"
        out += "\n".join(rows) + "\n" + (r"\midrule" + "\n" if ts == "archive9" else "")
    out += r"\bottomrule" "\n" r"\end{tabular}" "\n" r"\end{table}" "\n"
    return out


def macros() -> str:
    v = json.loads((HERE / "r1_text_numbers.json").read_text(encoding="utf-8"))["values"]
    return ("% build_supplement_tables_r1.py 가 r1_text_numbers.json 에서 만든다\n"
            + r"\newcommand{\ClipMin}{" + v["CLIP.pct_min"] + "}\n"
            + r"\newcommand{\ClipMax}{" + v["CLIP.pct_max"] + "}\n")


def main() -> int:
    if not (REPLAY / "rows.sha256").exists():
        raise SystemExit("보관 생성기 재학습(v2)이 아직 끝나지 않았다")
    (SUP / "generated_numbers_r1.tex").write_text(macros(), encoding="utf-8")
    for name, fn in (("generated_s8_matched_r1.tex", s8), ("generated_s15_generator_r1.tex", s15),
                     ("generated_s16_candidates_r1.tex", s16), ("generated_s17_stability_r1.tex", s17),
                     ("generated_s18_heldout_r1.tex", s18)):
        (SUP / name).write_text(fn(), encoding="utf-8")
        print("wrote", name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
