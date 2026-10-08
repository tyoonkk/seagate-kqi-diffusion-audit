"""수정본 그림 3: 주 감사(과제 ID 없는 게이트 7개, 바깥 LOTO)의 과제별·반복별 test 효과 — 계열 A (MDPI electronics-4573451) 수정
(심사위원 2 의견 3·4·6).

패널: (a) 원래 자료(제출본, LightGBM) (b) 재생성 자료 LightGBM (c) 재생성 자료 XGBoost.
끝점(과제)마다 두 목표(mean-seeking 위, conservative 아래)의 반복 5번을 seed 순서로 조금씩 위아래로 벌려 찍는다.
하이브리드를 쓴 반복은 찬 점(그 반복의 Δ PR-AUC), reference 를 지킨 반복은 0 위의 작은 회색 빈 점, 세로 막대는 반복 5번 평균이다.
과제 묶음은 --task-set (archive9 = 주 분석, all11 = 보충). 원래 자료는 9개 과제뿐이다.
출력: figure3_task_effects_r1{suffix}.{pdf,svg,png}, figure3_task_effects_r1{suffix}_data.csv (그림에 찍힌 행 그대로)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = Path(__file__).resolve().parent
SEAGATE = HERE.parents[2]
ORIG_ROWS = SEAGATE / "artifacts/condition_aware_main/ieee_task_independent_nested_audit_run1/outer_heldout_rows.csv"
WIDTH_IN = 6.7
SEEDS = (42, 43, 44, 45, 46)
plt.rcParams.update({"font.family": "Arial", "font.size": 7.5, "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "path",
                     "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False, "axes.linewidth": 0.8,
                     "xtick.major.width": 0.8, "xtick.major.size": 3})
# 두 목표의 색: 색각 이상 모의 뒤에도 OKLab 거리 ≥ 19 (protan 19.1, deutan 23.7, tritan 28.1). 모양(원/네모)으로도 구분한다.
OBJ = [("mean_only", "Mean-seeking objective", "#B3412E", "o", -0.22),
       ("conservative", "Conservative objective", "#1F5FA8", "s", 0.22)]
FALLBACK_C, BAND, GRID = "#9A9A9A", "#F5F5F5", "#E6E6E6"


def panel(ax, rows: pd.DataFrame, tasks, title: str, xlim):
    for i, t in enumerate(tasks):
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color=BAND, zorder=0, lw=0)
    for i, t in enumerate(tasks):
        d = rows[rows["task_id"].eq(t)]
        if d.empty:
            ax.text(0.5, i, "not in the original archive", transform=ax.get_yaxis_transform(), va="center", ha="center",
                    fontsize=6.8, color="#8A8A8A", style="italic", zorder=4)
            continue
        for regime, _, col, mk, off in OBJ:
            r = d[d["regime"].eq(regime)].set_index("seed").reindex(list(SEEDS))
            if r["delta_vs_reference_test"].isna().any():
                raise RuntimeError(f"과제 {t} {regime}: 반복 5개가 다 있지 않다")
            y0 = i + off
            for k, s in enumerate(SEEDS):
                y = y0 + (k - 2) * 0.06
                if r.loc[s, "selected_kind"] == "reference":
                    ax.plot(0, y, "o", ms=2.3, markerfacecolor="white", markeredgecolor=FALLBACK_C, markeredgewidth=0.6, zorder=2)
                else:
                    ax.plot(r.loc[s, "delta_vs_reference_test"], y, mk, ms=(3.6 if mk == "o" else 3.3), color=col,
                            markeredgecolor="white", markeredgewidth=0.35, zorder=4)
            if (r["selected_kind"] != "reference").any():  # 다섯 번 모두 reference 면 평균은 0 이고 막대를 그리지 않는다
                m = float(r["delta_vs_reference_test"].mean())
                ax.plot([m, m], [y0 - 0.17, y0 + 0.17], color=col, lw=1.5, solid_capstyle="butt", zorder=3)
    ax.axvline(0, color="#3C3C3C", lw=0.7, zorder=1)
    ax.set_yticks(range(len(tasks)))
    ax.set_yticklabels([f"Endpoint {t}" for t in tasks])
    ax.tick_params(axis="y", length=0, pad=3)
    ax.set_ylim(len(tasks) - 0.5, -0.5)
    ax.set_xlim(*xlim)
    ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title(title, fontsize=7.8, loc="left", fontweight="bold", pad=4)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit-dir", required=True)
    ap.add_argument("--task-set", default="archive9", choices=["archive9", "all11"])
    ap.add_argument("--suffix", default="")
    args = ap.parse_args()
    audit = Path(args.audit_dir)
    src = {"Original archive, LightGBM": pd.read_csv(ORIG_ROWS),
           "Rebuilt archive, LightGBM": pd.read_csv(audit / f"lgbm_{args.task_set}_nested_heldout_rows.csv"),
           "Rebuilt archive, XGBoost": pd.read_csv(audit / f"xgb_{args.task_set}_nested_heldout_rows.csv")}
    for k, d in src.items():
        if "task_id" not in d or (d.groupby(["regime", "task_id"]).size() != len(SEEDS)).any():
            raise RuntimeError(f"{k}: 과제·목표마다 반복 5행이 아니다")
    tasks = sorted(set().union(*[set(d["task_id"]) for d in src.values()]))
    pd.concat([d.assign(panel=k) for k, d in src.items()]).to_csv(HERE / f"figure3_task_effects_r1{args.suffix}_data.csv", index=False)
    allv = pd.concat([d["delta_vs_reference_test"] for d in src.values()])
    lo, hi = float(allv.min()), float(allv.max())
    pad = 0.06 * (hi - lo)
    xlim = (min(lo, 0) - pad, max(hi, 0) + pad)
    h = 0.62 + 0.30 * len(tasks) + 0.62
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH_IN, h), sharey=True, sharex=True, gridspec_kw={"wspace": 0.07})
    for ax, (lab, d), tag in zip(axes, src.items(), "abc"):
        panel(ax, d, tasks, f"({tag}) {lab}", xlim)
        ax.set_xlabel("Test ΔPR-AUC vs. reference")
    fig.subplots_adjust(left=0.105, right=0.99, top=1 - 0.30 / h, bottom=0.62 / h + 0.04)
    handles = [Line2D([], [], ls="none", marker=mk, ms=(3.6 if mk == "o" else 3.3), color=col, markeredgecolor="white",
                      markeredgewidth=0.35) for _, _, col, mk, _ in OBJ]
    labels = [f"{name}: run that used a hybrid" for _, name, *_ in OBJ]
    handles += [Line2D([], [], ls="none", marker="o", ms=2.6, markerfacecolor="white", markeredgecolor=FALLBACK_C, markeredgewidth=0.6),
                Line2D([], [], ls="none", color="#555555", marker="|", ms=8, markeredgewidth=1.5)]
    labels += ["Run that kept the reference (Δ = 0)", "Mean over the five repeats (shown when a hybrid was used)"]
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=7.3, bbox_to_anchor=(0.55, 0.0),
               handletextpad=0.5, columnspacing=2.0, handlelength=1.4)
    for ext in ("pdf", "svg"):
        fig.savefig(HERE / f"figure3_task_effects_r1{args.suffix}.{ext}")
    fig.savefig(HERE / f"figure3_task_effects_r1{args.suffix}.png", dpi=600)
    print("ok", len(tasks))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
