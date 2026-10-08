"""재생성 감사 결과의 독립 재계산 — 계열 A 수정 (크로스체크 규칙).

build_regen_audits_v1.py 는 원래 감사 스크립트의 함수를 import 해서 쓴다. 이 파일은 그 함수를 쓰지 않고, 정의(원고·사전 등록 문서)만 보고
처음부터 다시 계산한 뒤 래퍼 출력과 대조한다. 대조 항목:
- 주 감사: 7개 게이트 × conservative·mean_only 바깥 LOTO 의 선택 수, 음수 행, 과제 거시 평균, 부트스트랩 구간
- 과거 게이트 LOTO: 5개 게이트, 선택 수, 음수 행, 과제 거시 평균, 구간
분류기 2개 × 과제 묶음 2개. 다르면 0 이 아닌 코드로 끝난다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SEEDS = (42, 43, 44, 45, 46)
TASKS9 = (0, 1, 2, 3, 5, 6, 7, 9, 10)
TASKS11 = tuple(range(11))
WIN = ["val_PR_AUC", "val_MCC", "val_ROC_AUC", "val_Recall", "val_Precision"]


def key(r) -> str:
    return f"{r['method']}/mode={r['ddpm_train_mode']}/ratio={float(r['ratio'])}/keep={float(r['keep_rate'])}"


def boot(vals: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(20260714)
    idx = rng.integers(0, len(vals), size=(10_000, len(vals)))
    lo, hi = np.quantile(vals[idx].mean(axis=1), [0.025, 0.975])
    return float(lo), float(hi)


def ref_a(refs: pd.DataFrame) -> pd.Series:
    keep = refs[(refs.scenario == "S0") | (refs.method.isin(["random_under_20to1", "random_under_10to1"])) |
                (refs.method.isin(["random_over", "smote"]) & refs.ratio.isin([0.5, 1.0, 2.0]))]
    assert len(keep) == 9
    best = []
    for (sc, m), g in keep.groupby(["scenario", "method"], sort=False):
        best.append(g.sort_values(["val_PR_AUC", "val_MCC", "val_FPR"], ascending=[False, False, True], kind="stable").iloc[0])
    b = pd.DataFrame(best)
    b["cx"] = (b.scenario == "S1").astype(int)
    return b.sort_values(["val_PR_AUC", "val_FPR", "cx"], ascending=[False, True, True], kind="stable").iloc[0]


def ref_b(refs: pd.DataFrame) -> pd.Series:
    best = []
    for (sc, m), g in refs.groupby(["scenario", "method"], sort=True):
        g = g.assign(k=[f"{sc}/{m}/ratio={r}" for r in g.ratio])
        best.append(g.sort_values(WIN + ["k"], ascending=[False] * 5 + [True], kind="mergesort").iloc[0])
    b = pd.DataFrame(best)
    return b.sort_values(["val_PR_AUC", "val_FPR", "val_MCC", "method"], ascending=[False, True, False, True]).iloc[0]


def load(run: Path, clf: str, tasks) -> tuple[pd.DataFrame, pd.DataFrame]:
    fam, case = [], []
    ctx_rows = []
    for t in tasks:
        s0, s1 = [], {}
        for s in SEEDS:
            c = pd.read_csv(run / f"task{t}_seed{s}" / "candidates.csv", low_memory=False)
            c = c[c.classifier == clf]
            refs = c[c.grid == "ref"]
            ra, rb = ref_a(refs), ref_b(refs)
            A = c[c.grid == "A"].copy()
            A["k"] = [key(r) for _, r in A.iterrows()]
            for m, g in A.groupby("method"):
                w = g.sort_values(WIN + ["k"], ascending=[False] * 5 + [True], kind="mergesort").iloc[0]
                fam.append({"task": t, "seed": s, "method": m, "k": w["k"], "mode": w["ddpm_train_mode"], "vp": w["val_PR_AUC"],
                            "gain": w["val_PR_AUC"] - ra["val_PR_AUC"], "fpr": w["val_FPR"] - ra["val_FPR"], "tstr": w["tstr_pr_auc"],
                            "delta": w["test_PR_AUC"] - ra["test_PR_AUC"]})
            B = c[c.grid == "B"].copy()
            B["k"] = [key(r) for _, r in B.iterrows()]
            w = B.sort_values(WIN + ["k"], ascending=[False] * 5 + [True], kind="mergesort").iloc[0]
            rec = (0.0 if pd.isna(w["val_Recall_at_FPR010"]) else w["val_Recall_at_FPR010"]) - \
                  (0.0 if pd.isna(rb["val_Recall_at_FPR010"]) else rb["val_Recall_at_FPR010"])
            gain, fpr, tstr = w["val_PR_AUC"] - rb["val_PR_AUC"], w["val_FPR"] - rb["val_FPR"], (0.0 if pd.isna(w["tstr_pr_auc"]) else w["tstr_pr_auc"])
            pos = w["ddpm_train_mode"] == "positive_only"
            rvs = 0.99 if pd.isna(w["real_vs_synth_auc"]) else w["real_vs_synth_auc"]
            score = (2 if gain >= 0.005 else 1 if gain >= 0 else -1) + (2 if fpr <= 0 else 1 if fpr <= 0.005 else -1) + \
                    (2 if rec >= 0.025 else 1 if rec >= 0 else -0.5) + (1 if tstr >= 0.04 else -1) + (1 if pos else -1) + (0.5 if rvs <= 0.999 else -0.5)
            flags = int(gain < -0.06) + int(fpr > 0.02) + int(rec < -0.05) + int(tstr < 0.008) + int(not pos)
            case.append({"task": t, "seed": s, "method": w["method"], "pos": pos, "gain": gain, "fpr": fpr, "rec": rec, "tstr": tstr,
                         "score": score, "nflags": flags, "delta": w["test_PR_AUC"] - rb["test_PR_AUC"],
                         "grid_gains": list(B["val_PR_AUC"] - rb["val_PR_AUC"])})
            # 과제 서술자용 9개 reference
            keep = refs[(refs.scenario == "S0") | (refs.method.isin(["random_under_20to1", "random_under_10to1"])) |
                        (refs.method.isin(["random_over", "smote"]) & refs.ratio.isin([0.5, 1.0, 2.0]))]
            s0.append(float(keep[keep.scenario == "S0"]["val_PR_AUC"].iloc[0]))
            for m, g in keep[keep.scenario == "S1"].groupby("method"):
                s1.setdefault(m, []).extend(g["val_PR_AUC"].tolist())
        meta = pd.read_json(run / f"task{t}_seed42" / "case.json", typ="series")
        best_s1 = max(np.mean(v) for v in s1.values())
        ctx_rows.append({"task": t, "pos_rate": meta["n_pos_train"] / meta["n_train"], "s0": float(np.mean(s0)), "delta_s1": best_s1 - float(np.mean(s0))})
    fam, case, ctx = pd.DataFrame(fam), pd.DataFrame(case), pd.DataFrame(ctx_rows)
    gm = case.groupby("task")["grid_gains"].apply(lambda x: float(np.mean(np.concatenate([np.asarray(v) for v in x]))))
    case = case.merge(ctx, on="task").merge(gm.rename("grid_mean").reset_index(), on="task")
    return fam, case


GATES7 = {"reference_only": None, "validation_best": lambda r: True,
          "gain0_fpr0": lambda r: r.gain >= 0 and r.fpr <= 0, "gain005_fpr0": lambda r: r.gain >= 0.005 and r.fpr <= 0,
          "gain01_fpr01": lambda r: r.gain >= 0.01 and r.fpr <= 0.01,
          "positive_gain0_fpr01": lambda r: r["mode"] == "positive_only" and r.gain >= 0 and r.fpr <= 0.01,
          "positive_tstr_gain0": lambda r: r["mode"] == "positive_only" and r.gain >= 0 and r.fpr <= 0.01 and (not pd.isna(r.tstr)) and r.tstr >= 0.008}


def nested(fam: pd.DataFrame, tasks) -> dict:
    per = {}
    for g, f in GATES7.items():
        out = []
        for (t, s), grp in fam.groupby(["task", "seed"]):
            ok = grp[[False if f is None else bool(f(r)) for _, r in grp.iterrows()]]
            if ok.empty:
                out.append((t, s, 0.0, False))
            else:
                p = ok.assign(tt=ok.tstr.fillna(-np.inf)).sort_values(["vp", "fpr", "tt", "k"], ascending=[False, True, False, True],
                                                                    kind="mergesort").iloc[0]
                out.append((t, s, float(p.delta), True))
        per[g] = pd.DataFrame(out, columns=["task", "seed", "delta", "hyb"])
    res = {}
    for regime in ("conservative", "mean_only"):
        held = []
        for ht in tasks:
            sc = []
            for g, d in per.items():
                tr = d[d.task != ht]
                tm = tr.groupby("task").delta.mean()
                sc.append((g, tm.mean(), int((tm < 0).sum()), int((tr.delta < 0).sum()), int(tr.hyb.sum())))
            sc = pd.DataFrame(sc, columns=["g", "m", "nt", "nr", "h"])
            if regime == "conservative":
                sc = sc[(sc.nt == 0) & (sc.nr == 0)]
            ch = sc.sort_values(["m", "nt", "nr", "h", "g"], ascending=[False, True, True, True, True], kind="mergesort").iloc[0].g
            held.append(per[ch][per[ch].task == ht])
        h = pd.concat(held)
        tm = h.groupby("task").delta.mean().to_numpy()
        lo, hi = boot(tm)
        res[regime] = {"hybrid": int(h.hyb.sum()), "neg": int((h.delta < 0).sum()), "macro": float(tm.mean()), "lo": lo, "hi": hi}
    return res


def loto(case: pd.DataFrame, tasks) -> dict:
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
    per = {g: pd.DataFrame({"task": c.task, "seed": c.seed, "delta": np.where(d, c.delta, 0.0), "hyb": d.astype(bool)}) for g, d in dec.items()}
    held = []
    for ht in tasks:
        sc = []
        for g, d in per.items():
            tr = d[d.task != ht]
            tm = tr.groupby("task").delta.mean()
            sc.append((g, int((tr.delta < 0).sum()), tm.mean(), tm.min(), int(tr.hyb.sum())))
        sc = pd.DataFrame(sc, columns=["g", "nr", "m", "mn", "h"])
        ch = sc.sort_values(["nr", "m", "mn", "h", "g"], ascending=[True, False, False, False, True], kind="mergesort").iloc[0].g
        held.append(per[ch][per[ch].task == ht])
    h = pd.concat(held)
    tm = h.groupby("task").delta.mean().to_numpy()
    lo, hi = boot(tm)
    return {"selected": int(h.hyb.sum()), "neg": int((h.delta < 0).sum()), "macro": float(tm.mean()), "lo": lo, "hi": hi}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--audit-dir", required=True)
    args = ap.parse_args()
    run, audit = Path(args.run_dir), Path(args.audit_dir)
    bad = 0
    for clf in ("lgbm", "xgb"):
        for ts_name, tasks in (("archive9", TASKS9), ("all11", TASKS11)):
            fam, case = load(run, clf, tasks)
            n = nested(fam, tasks)
            w = pd.read_csv(audit / f"{clf}_{ts_name}_nested_summary.csv").set_index("regime")
            for regime, v in n.items():
                ok = (v["hybrid"] == w.loc[regime, "hybrid_selection_count"] and v["neg"] == w.loc[regime, "negative_row_count"]
                      and abs(v["macro"] - w.loc[regime, "task_macro_mean_delta"]) < 1e-12 and abs(v["lo"] - w.loc[regime, "ci_low"]) < 1e-12
                      and abs(v["hi"] - w.loc[regime, "ci_high"]) < 1e-12)
                bad += not ok
                print(f"[{'OK' if ok else 'DIFF'}] {clf} {ts_name} nested {regime}: hybrid {v['hybrid']} neg {v['neg']} macro {v['macro']:+.6f} [{v['lo']:+.6f}, {v['hi']:+.6f}]")
            l = loto(case, tasks)
            w = pd.read_csv(audit / f"{clf}_{ts_name}_loto_summary.csv").iloc[0]
            ok = (l["selected"] == w["selected_count"] and l["neg"] == w["harm_row_count"] and abs(l["macro"] - w["task_macro_mean_delta"]) < 1e-12
                  and abs(l["lo"] - w["ci_low"]) < 1e-12 and abs(l["hi"] - w["ci_high"]) < 1e-12)
            bad += not ok
            print(f"[{'OK' if ok else 'DIFF'}] {clf} {ts_name} LOTO: selected {l['selected']} neg {l['neg']} macro {l['macro']:+.6f} [{l['lo']:+.6f}, {l['hi']:+.6f}]")
    print("불일치:", bad)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
