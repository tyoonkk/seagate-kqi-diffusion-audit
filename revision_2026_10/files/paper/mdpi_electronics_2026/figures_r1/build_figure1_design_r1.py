"""수정본 그림 1: 연구 설계와 정보 경계 — 계열 A (MDPI electronics-4573451) 수정 (심사위원 1 의견 5, 그림 개선).

벡터(PDF, SVG)와 고해상도 PNG(600 dpi)를 같은 그림으로 만든다. 수치는 넣지 않는다(설계도).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

OUT = Path(__file__).resolve().parent
plt.rcParams.update({"font.family": "Arial", "font.size": 7.4, "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "path"})

C = {
    "train": ("#2E7D32", "#EDF7EE"), "val": ("#D35400", "#FEF3E9"), "test": ("#B71C1C", "#FCECEC"),
    "decide": ("#1F5FA8", "#EAF2FB"), "report": ("#3C3C3C", "#F3F3F3"), "regen": ("#00695C", "#E6F4F2"),
    "orig": ("#7A5C00", "#FBF5E3"), "ink": "#1E1E1E", "muted": "#5F6368", "line": "#4A4A4A",
}
TITLE_GAP = 0.052  # 상자 위쪽에서 제목까지


def box(ax, x, y, w, h, key, title, body, title_size=8.0, body_size=7.2):
    edge, face = C[key]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.010", linewidth=1.05,
                                edgecolor=edge, facecolor=face))
    ax.text(x + w / 2, y + h - 0.030, title, ha="center", va="top", fontsize=title_size, fontweight="bold", color=edge)
    ax.text(x + w / 2, y + (h - TITLE_GAP) / 2, body, ha="center", va="center", fontsize=body_size, color=C["ink"],
            linespacing=1.35)


def arrow(ax, p0, p1, color=None, style="-", lw=1.05, ms=8):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=ms, linewidth=lw, color=color or C["line"],
                                 linestyle=style, shrinkA=0, shrinkB=0))


def main() -> None:
    fig = plt.figure(figsize=(6.7, 4.35))   # 원고에 넣는 폭 그대로
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # ── 위: 공식 분할 (제공자 설명상 연속된 시간 창) ──
    x0, x1, yb, hb = 0.040, 0.960, 0.846, 0.074
    ax.text(x0, 0.958, "Seagate time-series-2 (ion-milling tool family), 11 binary KQI endpoints. Fixed splits that the data "
            "providers describe as consecutive time windows:", ha="left", va="top", fontsize=7.0, color=C["muted"])
    x = x0
    for label, wk, key in [("training split\nweeks 1–70", 70, "train"), ("validation\nweeks 71–84", 14, "val"), ("test\n85–92", 8, "test")]:
        w = (x1 - x0) * wk / 92
        edge, face = C[key]
        ax.add_patch(Rectangle((x, yb), w, hb, facecolor=face, edgecolor=edge, linewidth=1.0))
        ax.text(x + w / 2, yb + hb / 2, label, ha="center", va="center", fontsize=7.2, color=edge, fontweight="bold",
                linespacing=1.15)
        x += w

    # ── 가운데: 행 하나의 결정이 쓰는 정보 ──
    # 시험 정보가 쓰이는 곳은 모두 빨간색으로 그린다: (1) 바깥 게이트 선택에 다른 과제의 시험 결과,
    # (2) 남겨 둔 과제에서 고른 행동의 채점, (3) 원래 연구에서 게이트를 다듬은 회고적 고리(점선).
    tx, tw, th = 0.040, 0.300, 0.230
    box(ax, tx, 0.565, tw, th, "train", "Training rows only",
        "constant-column removal, imputation\nSMOTE and other resampling\ndiffusion generator fit (rebuilt\n"
        "archive: 20% of its rows held out\nfor early stopping and checks)")
    box(ax, tx, 0.295, tw, th, "val", "Validation rows",
        "feature weights for ranking\ngenerated rows\ncandidate ranking by PR-AUC\nmaximum-MCC thresholds\ngate checks and task profiles")
    dx, dw = 0.400, 0.190      # 행 결정
    sx, sw = 0.640, 0.195      # 시험 행
    rx, rw = 0.870, 0.090      # 보고
    dy, dh = 0.385, 0.250
    cy, ch = 0.690, 0.118      # 바깥 게이트 선택
    box(ax, dx, cy, sx + sw - dx, ch, "decide", "Outer gate choice (leave-one-task-out)",
        "chosen on test outcomes of the other endpoints only;\nthe held-out endpoint's test values are not used",
        title_size=7.8, body_size=7.1)
    box(ax, dx, dy, dw, dh, "decide", "Row decision",
        "one endpoint–repeat run:\nhybrid candidate or\nconventional reference;\nthe gate never sees\nthe task identifier")
    box(ax, sx, dy, sw, dh, "test", "Test rows",
        "held-out endpoint:\nΔ PR-AUC of the chosen\naction vs. the reference;\nno row decision uses\ntest values")
    box(ax, rx, dy, rw, dh, "report", "Report",
        "task-macro\neffect over\nheld-out\nendpoints")
    mid = dy + dh / 2
    red = C["test"][0]
    arrow(ax, (tx + tw, 0.565 + th / 2), (dx, mid + 0.045))
    arrow(ax, (tx + tw, 0.295 + th / 2), (dx, mid - 0.045))
    arrow(ax, (dx + dw / 2, cy), (dx + dw / 2, dy + dh), color=C["decide"][0])
    ux = sx + sw / 2
    arrow(ax, (ux, dy + dh), (ux, cy), color=red)
    ax.text(ux + 0.010, (dy + dh + cy) / 2, "other endpoints", ha="left", va="center", fontsize=6.9, color=red,
            style="italic")
    arrow(ax, (dx + dw, mid), (sx, mid))
    arrow(ax, (sx + sw, mid), (rx, mid))

    # 회고적 개발 고리 (꺾인 점선): 시험 → 검증 쪽 게이트 개발
    ey = 0.318
    ax.plot([ux, ux, 0.360], [dy, ey, ey], color=red, lw=1.0, ls=(0, (3, 2)), solid_capstyle="butt")
    arrow(ax, (0.360, ey), (tx + tw + 0.002, ey), color=red, style="-", lw=1.0, ms=7)
    ax.text(0.362, ey - 0.014, "in the original study, later gate families were refined after earlier test results",
            ha="left", va="top", fontsize=6.9, color=red, style="italic")

    # ── 아래: 이 연구의 두 증거 묶음 (같은 감사 코드) ──
    ax.text(0.040, 0.228, "Evidence in this paper — the same audit code is applied to both candidate archives:",
            ha="left", va="bottom", fontsize=7.0, color=C["muted"])
    box(ax, 0.040, 0.040, 0.450, 0.168, "orig", "Original archive (June 2026; submitted analysis)",
        "archived generator learned little (almost nothing with 10 epochs);\n"
        "test metrics from a separate, non-deterministic refit;\ngate families developed after earlier test comparisons",
        title_size=7.8, body_size=7.1)
    box(ax, 0.510, 0.040, 0.450, 0.168, "regen", "Rebuilt archive (October 2026; this revision)",
        "corrected generator, all 11 endpoints, LightGBM and XGBoost;\n"
        "validation and test metrics from the same deterministic fit;\nprotocol and code fixed before any rebuilt test outcome",
        title_size=7.8, body_size=7.1)

    for ext in ("pdf", "svg"):
        fig.savefig(OUT / f"figure1_study_design_r1.{ext}")
    fig.savefig(OUT / "figure1_study_design_r1.png", dpi=600)
    plt.close(fig)


if __name__ == "__main__":
    main()
