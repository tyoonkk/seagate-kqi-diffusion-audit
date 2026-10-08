"""수정본 그림 2: 생성기 진단 — 계열 A (MDPI electronics-4573451) 수정 (심사위원 2 의견 6).

끝점(과제)마다 학습 자료 안 떼어 둔 불량에서 잰 세 지표를 생성기 셋으로 나란히 놓는다.
(a) 떼어 둔 잡음 예측 손실 ÷ 가장 나은 단순 기준선 손실 (1 보다 작으면 학습했다)
    - 보관 생성기: loss_model / min(0-예측, 특징 독립 가우시안)
    - 고친 생성기: total_model / total_baseline (연속 MSE + T×이진 KL, 기준선은 연속 가우시안 + 이진 min(복사, 학습 비율))
(b) 떼어 둔 실제 불량 대 생성 불량 C2ST AUC (0.5 = 구별 불가)
(c) 내부 TSTR 비 = 생성 불량으로 학습한 PR-AUC ÷ 실제 불량으로 학습한 PR-AUC
생성기: 보관 10 epoch(주 감사 후보원, 두 모드), 보관 200 epoch positive-only(동일예산 격자), 고친 생성기(재생성 실행 격자 A, 두 모드).
보관 생성기는 diag_archived_generator_replay_v2 가 고친 생성기와 같은 행으로 다시 학습한 값이다. 점은 중앙값, 선은 최솟값–최댓값.
출력: figure2_generator_r1{suffix}.{pdf,svg,png}, figure2_generator_r1{suffix}_data.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = Path(__file__).resolve().parent
WIDTH_IN = 6.7
plt.rcParams.update({"font.family": "Arial", "font.size": 7.5, "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "path",
                     "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False, "axes.linewidth": 0.8,
                     "xtick.major.width": 0.8, "xtick.major.size": 3})
# 색각 이상 모의(Machado 2009) 뒤 OKLab 거리 ≥ 11.0, 모양으로도 구분한다. 황갈색 = 원래 자료, 청록 = 재생성(그림 3·4 와 같은 뜻).
SERIES = [("arch10", "Archived generator, 10 epochs", "#B8860B", "o", "none", -0.24),
          ("arch200", "Archived generator, 200 epochs (positive-only)", "#A63603", "s", "none", 0.0),
          ("corr", "Corrected generator", "#00695C", "o", "full", 0.24)]
BAND, GRID = "#F5F5F5", "#E6E6E6"


def load(run_dir: Path, replay_rows: Path) -> pd.DataFrame:
    rows = []
    for case in sorted(run_dir.glob("task*_seed*/case.json")):
        m = json.loads(case.read_text(encoding="utf-8"))
        for g in m["generator_diagnostics"]:
            if g["grid"] != "A":
                continue
            hl = g["holdout_pos_loss"]
            rows.append({"series": "corr", "task_id": m["task_id"], "seed": m["seed"], "mode": g["mode"],
                         "learn": hl["total_model"] / hl["total_baseline"], "c2st": g["c2st_hold"],
                         "tstr": g["tstr_internal_syn"] / g["tstr_internal_real"]})
    r = pd.read_csv(replay_rows)
    r["learn"] = r["loss_model"] / np.minimum(r["loss_zero"], r["loss_gauss"])
    r["c2st"] = r["c2st_hold"]
    r["tstr"] = r["tstr_internal_syn"] / r["tstr_internal_real"]
    a10 = r[r["setting"] == "quick_10ep"].assign(series="arch10")
    a200 = r[(r["setting"] == "full_200ep") & (r["mode"] == "positive_only")].assign(series="arch200")
    cols = ["series", "task_id", "seed", "mode", "learn", "c2st", "tstr"]
    return pd.concat([pd.DataFrame(rows)[cols], a10[cols], a200[cols]], ignore_index=True)


def panel(ax, d: pd.DataFrame, col: str, tasks, title: str, ref: float, ref_label: str, xlim, xlabel: str, label_side: str):
    for i in range(len(tasks)):
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color=BAND, zorder=0, lw=0)
    for key, _, c, mk, fill, off in SERIES:
        g = d[d["series"] == key].groupby("task_id")[col].agg(["median", "min", "max"])
        for i, t in enumerate(tasks):
            if t not in g.index:
                continue
            r = g.loc[t]
            y = i + off
            ax.plot([r["min"], r["max"]], [y, y], color=c, lw=1.1, solid_capstyle="butt", zorder=2)
            ax.plot(r["median"], y, mk, ms=(4.0 if mk == "s" else 4.6), color=c, markerfacecolor=("white" if fill == "none" else c),
                    markeredgecolor=c, markeredgewidth=(1.1 if fill == "none" else 0.4), zorder=3)
    ax.axvline(ref, color="#555555", lw=0.8, ls=(0, (3, 2)), zorder=1)
    ax.text(ref + (-0.012 if label_side == "left" else 0.012) * (xlim[1] - xlim[0]), -0.62, ref_label,
            ha=("right" if label_side == "left" else "left"), va="bottom", fontsize=6.8, color="#555555")
    ax.set_yticks(range(len(tasks)))
    ax.set_yticklabels([f"Endpoint {t}" for t in tasks])
    ax.tick_params(axis="y", length=0, pad=3)
    ax.set_ylim(len(tasks) - 0.5, -0.5)
    ax.set_xlim(*xlim)
    ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title(title, fontsize=7.8, loc="left", fontweight="bold", pad=12)
    ax.set_xlabel(xlabel)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--replay-rows", required=True)
    ap.add_argument("--suffix", default="")
    args = ap.parse_args()
    d = load(Path(args.run_dir), Path(args.replay_rows))
    d.to_csv(HERE / f"figure2_generator_r1{args.suffix}_data.csv", index=False)
    tasks = sorted(d["task_id"].unique())
    h = 0.75 + 0.27 * len(tasks) + 0.62
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH_IN, h), sharey=True, gridspec_kw={"wspace": 0.08})
    hi_learn = max(1.05, float(d["learn"].max()) * 1.03)
    hi_tstr = max(1.05, float(d["tstr"].max()) * 1.05)
    panel(axes[0], d, "learn", tasks, "(a) Denoising loss ÷ best trivial", 1.0, "no learning", (0, hi_learn),
          "Held-out loss ratio", "left")
    panel(axes[1], d, "c2st", tasks, "(b) Two-sample AUC", 0.5, "indistinguishable", (0.45, 1.02),
          "Real vs. generated positives", "right")
    panel(axes[2], d, "tstr", tasks, "(c) Internal TSTR ratio", 1.0, "as useful as real", (0, hi_tstr),
          "Generated ÷ real positives", "left")
    fig.subplots_adjust(left=0.105, right=0.985, top=1 - 0.45 / h, bottom=0.62 / h + 0.07)
    handles = [Line2D([], [], color=c, marker=mk, lw=1.1, ms=(4.0 if mk == "s" else 4.6),
                      markerfacecolor=("white" if fill == "none" else c), markeredgecolor=c,
                      markeredgewidth=(1.1 if fill == "none" else 0.4)) for _, _, c, mk, fill, _ in SERIES]
    fig.legend(handles, [lab for _, lab, *_ in SERIES], loc="lower center", ncol=3, frameon=False, fontsize=7.2,
               bbox_to_anchor=(0.55, 0.0), handletextpad=0.5, columnspacing=1.4)
    for ext in ("pdf", "svg"):
        fig.savefig(HERE / f"figure2_generator_r1{args.suffix}.{ext}")
    fig.savefig(HERE / f"figure2_generator_r1{args.suffix}.png", dpi=600)
    print("ok", len(tasks), d.groupby("series").size().to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
