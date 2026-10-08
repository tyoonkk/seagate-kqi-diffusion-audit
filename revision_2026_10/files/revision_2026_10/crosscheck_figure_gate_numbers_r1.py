"""수정본 그림 3(게이트 전체 적용) 숫자의 독립 재계산 — 계열 A (MDPI electronics-4573451) 수정.

그림 스크립트는 원래 감사 함수(NA.apply_gate, NA.summarize)와 감사 래퍼 출력(*_loto_full_archive_rules.csv)을 쓴다.
여기서는 그 함수를 쓰지 않고 crosscheck_regen_audits_v1.py 의 독립 구현(load, GATES7, 과거 게이트 정의)으로
보관 9개 과제 전체에 게이트를 그대로 적용해 선택 수, 음수 행, 과제 거시 평균을 다시 계산하고 대조한다.
원래 자료 값은 보관 감사의 접힘별 점수(outer_train_policy_scores.csv)에서 거꾸로 복원해 대조한다.
다르면 0 이 아닌 코드로 끝난다.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SEAGATE = HERE.parent
sys.path.insert(0, str(SEAGATE))
spec = importlib.util.spec_from_file_location("X", HERE / "crosscheck_regen_audits_v1.py")
X = importlib.util.module_from_spec(spec)
spec.loader.exec_module(X)
import build_ieee_task_independent_nested_audit as NA  # noqa: E402  (대조 대상 쪽)

RUN = SEAGATE / "outputs/revision_2026_10/regen_archive_v1_run1"
AUD = SEAGATE / "outputs/revision_2026_10/regen_audits_v1"
ART = SEAGATE / "artifacts/condition_aware_main"
TASKS9 = X.TASKS9
TOL = 1e-12


def simple_whole(fam: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for g, f in X.GATES7.items():
        out = []
        for (t, s), grp in fam.groupby(["task", "seed"]):
            ok = grp[[False if f is None else bool(f(r)) for _, r in grp.iterrows()]]
            if ok.empty:
                out.append((t, s, 0.0, False))
            else:
                p = ok.assign(tt=ok.tstr.fillna(-np.inf)).sort_values(["vp", "fpr", "tt", "k"], ascending=[False, True, False, True],
                                                                    kind="mergesort").iloc[0]
                out.append((t, s, float(p.delta), True))
        d = pd.DataFrame(out, columns=["task", "seed", "delta", "hyb"])
        rows.append({"gate": g, "sel": int(d.hyb.sum()), "neg": int((d.delta < 0).sum()), "tm": float(d.groupby("task").delta.mean().mean())})
    return pd.DataFrame(rows)


def legacy_whole(case: pd.DataFrame) -> pd.DataFrame:
    c = case.copy()
    rare = (c.pos_rate <= 0.015) & (c.delta_s1 >= 0.030)
    strong = (c.s0 >= 0.100) & (c.delta_s1 >= 0.030)
    opp = rare | strong
    mod = (c.grid_mean >= 0) | (c.delta_s1 >= 0.020)
    strict = (c.gain >= 0.005) & (c.fpr <= 0) & (c.rec >= 0) & (c.tstr >= 0.040) & (c["nflags"] == 0)
    broad = (c.method == "s1_plus_ig") & c.pos & (c.gain >= -0.060) & (c.fpr <= 0.020) & (c.rec >= -0.050) & (c.tstr >= 0.008) & \
            (c.score >= -3.0) & (c["nflags"] <= 1)
    dec = {"safe_only": pd.Series(False, index=c.index), "strict_certificate": strict, "v11_grid_or_profile_strict": strict & mod,
           "v11_rare_signal_aggressive_else_strict": pd.Series(np.where(rare, broad, strict & mod), index=c.index),
           "v12_signal_aggressive_else_strict": pd.Series(np.where(opp, broad, strict & mod), index=c.index)}
    rows = []
    for g, d in dec.items():
        delta = np.where(d, c.delta, 0.0)
        tm = pd.Series(delta).groupby(c.task.to_numpy()).mean().mean()
        rows.append({"gate": g, "sel": int(d.sum()), "neg": int((delta < 0).sum()), "tm": float(tm)})
    return pd.DataFrame(rows)


def compare(name: str, mine: pd.DataFrame, theirs: pd.DataFrame) -> int:
    m = mine.merge(theirs, on="gate", suffixes=("_x", "_y"), how="outer", indicator=True)
    bad = m[(m["_merge"] != "both") | (m.sel_x != m.sel_y) | (m.neg_x != m.neg_y) | ((m.tm_x - m.tm_y).abs() > TOL)]
    print(f"{name}: {'OK' if bad.empty else 'MISMATCH'} ({len(m)} gates)")
    if not bad.empty:
        print(bad.to_string(index=False))
    return int(not bad.empty)


def main() -> int:
    fails = 0
    # 원래 자료: 접힘별 8개 과제 점수에서 9개 과제 전체 값을 복원 (각 과제는 9개 접힘 중 8개에 들어간다)
    s = pd.read_csv(ART / "ieee_task_independent_nested_audit_run1/outer_train_policy_scores.csv")
    s = s[s.regime.eq("conservative")]
    rec = s.groupby("gate").agg(sel=("hybrid_selection_count", "sum"), neg=("negative_row_count", "sum"),
                                tm=("task_macro_mean_delta", "sum")).reset_index()
    rec["sel"], rec["neg"], rec["tm"] = rec.sel / 8, rec.neg / 8, rec.tm / 9
    v9 = pd.read_csv(ART / "selector_v9_certificate_mining_run1/v9_candidate_evidence_table.csv", low_memory=False)
    v9 = v9[v9["source"].eq("quick_mean_alltasks")].sort_values(["task_id", "seed", "method"]).reset_index(drop=True)
    fig = []
    for g in NA.GATES:
        r = NA.summarize(NA.apply_gate(v9, g))
        fig.append({"gate": g.name, "sel": int(r["hybrid_selection_count"]), "neg": int(r["negative_row_count"]), "tm": float(r["task_macro_mean_delta"])})
    rec["tm"] = rec["tm"].astype(float)
    fails += compare("original, simple gates (figure vs archived fold scores)", pd.DataFrame(fig), rec.assign(sel=rec.sel.round().astype(int), neg=rec.neg.round().astype(int)))
    for clf in ("lgbm", "xgb"):
        fam, case = X.load(RUN, clf, TASKS9)
        ni = pd.read_csv(AUD / f"{clf}_nested_input.csv")
        ni = ni[ni["task_id"].isin(TASKS9)].sort_values(["task_id", "seed", "method"]).reset_index(drop=True)
        fig = []
        for g in NA.GATES:
            r = NA.summarize(NA.apply_gate(ni, g))
            fig.append({"gate": g.name, "sel": int(r["hybrid_selection_count"]), "neg": int(r["negative_row_count"]), "tm": float(r["task_macro_mean_delta"])})
        fails += compare(f"{clf}, simple gates (figure vs independent)", pd.DataFrame(fig), simple_whole(fam))
        lr = pd.read_csv(AUD / f"{clf}_archive9_loto_full_archive_rules.csv")
        lr = pd.DataFrame({"gate": lr.rule_name, "sel": lr.selected_count.astype(int), "neg": lr.harm_row_count.astype(int),
                           "tm": lr.task_macro_mean_delta.astype(float)})
        fails += compare(f"{clf}, historical gates (figure vs independent)", lr, legacy_whole(case))
    print("ALL OK" if fails == 0 else f"{fails} MISMATCH")
    return int(fails > 0)


if __name__ == "__main__":
    raise SystemExit(main())
