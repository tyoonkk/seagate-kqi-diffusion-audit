"""수정본 그림 4: 게이트를 보관 9개 과제 전체에 그대로 적용했을 때(바깥 선택 없음)의 과제 거시 효과 — 계열 A (MDPI electronics-4573451) 수정.

제출본 그림 3(과거 게이트의 적용 비율–이득 산점도)을 바꾼다. 같은 게이트를 원래 자료와 재생성 자료(LightGBM, XGBoost)에 적용한 값을
행마다 나란히 놓고, 오른쪽에 하이브리드 선택 수 / 음수 행 수(45개 중)를 적는다. 서술용(사후) 그림이다.

(a) 과거 게이트 5개: 원래 자료는 동일예산 격자 감사(legacy_rule_summary.csv), 재생성 자료는 격자 B (감사 래퍼의 *_archive9_loto_full_archive_rules.csv).
(b) 과제 ID 없는 단순 게이트 7개(표 4): 원래 자료는 주 감사 원천(quick_mean_alltasks), 재생성 자료는 격자 A family winner.
게이트 판정은 원래 감사 스크립트의 함수를 그대로 쓴다. 숫자는 crosscheck_figure_gate_numbers_r1.py 가 독립 재계산으로 대조한다.
출력: figure4_gate_effects_r1.{pdf,svg,png}, figure4_gate_effects_r1_data.csv (그림에 찍힌 값 그대로)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = Path(__file__).resolve().parent
SEAGATE = HERE.parents[2]
sys.path.insert(0, str(SEAGATE))
import build_ieee_task_independent_nested_audit as NA  # noqa: E402

ART = SEAGATE / "artifacts/condition_aware_main"
TASKS9 = (0, 1, 2, 3, 5, 6, 7, 9, 10)
WIDTH_IN = 6.7          # 원고에 넣는 폭 그대로 그린다(글자 크기가 그대로 유지된다)
plt.rcParams.update({"font.family": "Arial", "font.size": 7.5, "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "path",
                     "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False, "axes.linewidth": 0.8,
                     "xtick.major.width": 0.8, "xtick.major.size": 3})
INK, MUTED, GRID = "#222222", "#6B6B6B", "#E6E6E6"
# 원래 자료 / 재생성 LightGBM / 재생성 XGBoost. 색각 이상 모의(Machado 2009) 뒤 OKLab 거리 ≥ 8.8, 모양으로도 구분한다.
SERIES = [("orig", "Original archive (LightGBM)", "#B8860B", "o", "none"),
          ("lgbm", "Rebuilt archive, LightGBM", "#00695C", "o", "full"),
          ("xgb", "Rebuilt archive, XGBoost", "#5E35B1", "D", "full")]
LEGACY = [("safe_only", "Always fallback"), ("strict_certificate", "Strict only"),
          ("v11_grid_or_profile_strict", "Combined conservative"),
          ("v11_rare_signal_aggressive_else_strict", "Rare-signal (Algorithm 1)"),
          ("v12_signal_aggressive_else_strict", "Opportunity extension")]
SIMPLE = [(g.name, g.name) for g in NA.GATES]   # 표 4 의 이름을 그대로 쓴다


def simple_rows(cands: pd.DataFrame, key: str) -> list[dict]:
    out = []
    for g in NA.GATES:
        s = NA.summarize(NA.apply_gate(cands, g))
        out.append({"family": "simple", "gate": g.name, "archive": key, "task_macro": float(s["task_macro_mean_delta"]),
                    "selected": int(s["hybrid_selection_count"]), "negative": int(s["negative_row_count"]), "rows": int(s["n_rows"])})
    return out


def legacy_rows(df: pd.DataFrame, key: str) -> list[dict]:
    return [{"family": "historical", "gate": r.rule_name, "archive": key, "task_macro": float(r.task_macro_mean_delta),
             "selected": int(r.selected_count), "negative": int(r.harm_row_count), "rows": int(r.n_rows)} for r in df.itertuples()]


def collect(audit: Path) -> pd.DataFrame:
    v9 = pd.read_csv(ART / "selector_v9_certificate_mining_run1/v9_candidate_evidence_table.csv", low_memory=False)
    v9 = v9[v9["source"].eq("quick_mean_alltasks")].sort_values(["task_id", "seed", "method"]).reset_index(drop=True)
    rows = simple_rows(v9, "orig") + legacy_rows(pd.read_csv(ART / "ieee_v10_uniform_grid_loto_audit_run1/legacy_rule_summary.csv"), "orig")
    for clf in ("lgbm", "xgb"):
        ni = pd.read_csv(audit / f"{clf}_nested_input.csv")
        ni = ni[ni["task_id"].isin(TASKS9)].sort_values(["task_id", "seed", "method"]).reset_index(drop=True)
        rows += simple_rows(ni, clf) + legacy_rows(pd.read_csv(audit / f"{clf}_archive9_loto_full_archive_rules.csv"), clf)
    d = pd.DataFrame(rows)
    if not (d["rows"] == 45).all():
        raise RuntimeError("보관 9개 과제 × seed 5개 = 45행이 아닌 묶음이 있다")
    return d


def panel(ax, tab, d: pd.DataFrame, family: str, gates, title: str, xlim):
    n = len(gates)
    dodge = {"orig": -0.24, "lgbm": 0.0, "xgb": 0.24}
    for i, (g, _) in enumerate(gates):
        ax.axhspan(i - 0.5, i + 0.5, color="#F6F6F6" if i % 2 == 0 else "white", zorder=0, lw=0)
        for key, _, col, mk, fill in SERIES:
            r = d[(d.family == family) & (d.gate == g) & (d.archive == key)].iloc[0]
            ax.plot(r.task_macro, i + dodge[key], mk, ms=(4.6 if mk == "D" else 5.2), color=col,
                    markerfacecolor=("white" if fill == "none" else col), markeredgecolor=col,
                    markeredgewidth=(1.2 if fill == "none" else 0.5), zorder=3)
        cells = []
        for key, *_ in SERIES:
            r = d[(d.family == family) & (d.gate == g) & (d.archive == key)].iloc[0]
            cells.append(f"{r.selected} / {r.negative}")
        for j, txt in enumerate(cells):
            tab.text(j, i, txt, ha="center", va="center", fontsize=7.2, color=INK)
    ax.axvline(0, color="#3C3C3C", lw=0.8, zorder=1)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_xlim(*xlim)
    ax.set_yticks(range(n))
    ax.set_yticklabels([lab for _, lab in gates])
    ax.tick_params(axis="y", length=0, pad=4)
    ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title(title, fontsize=7.8, loc="left", fontweight="bold", pad=4)
    tab.set_xlim(-0.5, len(SERIES) - 0.5)
    tab.set_ylim(n - 0.5, -0.5)
    for i in range(n):
        tab.axhspan(i - 0.5, i + 0.5, color="#F6F6F6" if i % 2 == 0 else "white", zorder=0, lw=0)
    tab.axis("off")
    for j, (key, _, col, mk, fill) in enumerate(SERIES):
        tab.plot(j, -0.95, mk, ms=(4.2 if mk == "D" else 4.8), color=col, markerfacecolor=("white" if fill == "none" else col),
                 markeredgecolor=col, markeredgewidth=(1.1 if fill == "none" else 0.5), clip_on=False)
    tab.text(1, -1.55, "Hybrids / negative runs (of 45)", ha="center", va="bottom", fontsize=7.2, color=MUTED)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit-dir", required=True)
    ap.add_argument("--suffix", default="")
    args = ap.parse_args()
    d = collect(Path(args.audit_dir))
    d.to_csv(HERE / f"figure4_gate_effects_r1{args.suffix}_data.csv", index=False)
    lo, hi = float(d.task_macro.min()), float(d.task_macro.max())
    pad = 0.06 * (hi - lo)
    xlim = (min(lo, 0) - pad, max(hi, 0) + pad)
    na, nb = len(LEGACY), len(SIMPLE)
    fig = plt.figure(figsize=(WIDTH_IN, 4.55))
    gs = fig.add_gridspec(2, 2, width_ratios=[3.05, 1.45], height_ratios=[na, nb], left=0.215, right=0.995, top=0.905,
                          bottom=0.165, wspace=0.04, hspace=0.42)
    ax_a, tab_a = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    ax_b, tab_b = fig.add_subplot(gs[1, 0], sharex=ax_a), fig.add_subplot(gs[1, 1])
    panel(ax_a, tab_a, d, "historical", LEGACY, "(a) Historical gate family, equal-budget candidate grid", xlim)
    panel(ax_b, tab_b, d, "simple", SIMPLE, "(b) Task-identifier-free gates (Table 4), primary-audit candidates", xlim)
    ax_a.tick_params(axis="x", labelbottom=True)
    ax_b.set_xlabel("Task-macro test ΔPR-AUC vs. run-local reference (nine endpoints, gate applied to every run)")
    handles = [Line2D([], [], ls="none", marker=mk, ms=(4.6 if mk == "D" else 5.2), color=col,
                      markerfacecolor=("white" if fill == "none" else col), markeredgecolor=col,
                      markeredgewidth=(1.2 if fill == "none" else 0.5)) for _, _, col, mk, fill in SERIES]
    fig.legend(handles, [lab for _, lab, *_ in SERIES], loc="lower center", ncol=3, frameon=False, fontsize=7.4,
               bbox_to_anchor=(0.5, 0.0), handletextpad=0.4, columnspacing=1.6)
    for ext in ("pdf", "svg"):
        fig.savefig(HERE / f"figure4_gate_effects_r1{args.suffix}.{ext}")
    fig.savefig(HERE / f"figure4_gate_effects_r1{args.suffix}.png", dpi=600)
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
