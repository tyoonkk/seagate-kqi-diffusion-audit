#!/usr/bin/env python
"""Build neutral, data-driven figures for the IEEE Access revision.

2026-08-20 revision pass:
- Figure 1 redrawn on an aligned grid with symmetric arrow routing.
- Reader-facing labels no longer use internal version codenames (V10/V11/
  V12/V15); descriptive gate names are used instead.  Internal rule keys in
  the archived CSVs are unchanged.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "experiments/seagate_kqi/artifacts/condition_aware_main"
OUT = ROOT / "experiments/seagate_kqi/paper/ieee_access_revision_2026/figures"
V12_RESULTS = (
    BASE
    / "selector_v12_final_validation_report_run1/"
    "selector_v12_final_selection_results.csv"
)
V15_ROWS = (
    BASE
    / "v15_final_combined_selector_report_run1/"
    "v15_final_combined_selected_rows.csv"
)
V15_SUMMARY = (
    BASE
    / "v15_final_combined_selector_report_run1/"
    "v15_final_combined_overall_summary.csv"
)


BLUE = "#00629B"
LIGHT_BLUE = "#DCECF5"
ORANGE = "#D97706"
RED = "#B91C1C"
GREEN = "#1B7F5A"
GRAY = "#5B6573"
LIGHT_GRAY = "#EEF1F4"


def setup() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "legend.fontsize": 7.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    OUT.mkdir(parents=True, exist_ok=True)


def add_box(ax, xy, width, height, title, body, facecolor, edgecolor=BLUE):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.008,rounding_size=0.015",
        linewidth=1.2,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(patch)
    x, y = xy
    ax.text(x + width / 2, y + height * 0.72, title, ha="center", va="center", weight="bold")
    ax.text(x + width / 2, y + height * 0.34, body, ha="center", va="center", fontsize=7.3)


def arrow(ax, start, end, color=GRAY, style="-", width=1.2, rad=0.0):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=width,
            linestyle=style,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=0.0,
            shrinkB=0.0,
        )
    )


def elbow(ax, pts, color=GRAY, style="-", width=1.2):
    """Orthogonal connector: straight segments through pts, arrowhead on the last."""
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    if len(pts) > 2:
        ax.plot(xs[:-1], ys[:-1], color=color, linestyle=style, linewidth=width,
                solid_capstyle="round", zorder=2)
    ax.add_patch(
        FancyArrowPatch(
            pts[-2],
            pts[-1],
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=width,
            linestyle=style,
            color=color,
            shrinkA=0.0,
            shrinkB=0.0,
            zorder=2,
        )
    )


def information_boundaries() -> None:
    fig, ax = plt.subplots(figsize=(13.2, 5.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Strict grid.  Main pipeline on one horizontal line (center y=0.62);
    # training-only fit above, validation selection below; the historical
    # feedback loop runs along the bottom band.  Every connector is
    # orthogonal (right angles only).
    add_box(ax, (0.02, 0.50), 0.14, 0.24, "Public Seagate arrays",
            "fixed train / validation\n/ test splits", LIGHT_BLUE)
    add_box(ax, (0.24, 0.74), 0.17, 0.24, "Training only",
            "imputation, constants,\nmodel and generator fit", "#E7F4EC", GREEN)
    add_box(ax, (0.24, 0.26), 0.17, 0.24, "Validation",
            "candidate ranking,\nthresholds, task profiles", "#FFF1D6", ORANGE)
    add_box(ax, (0.47, 0.50), 0.17, 0.24, "Frozen row decision",
            "hybrid candidate or\nconventional reference", LIGHT_BLUE)
    add_box(ax, (0.70, 0.50), 0.11, 0.24, "Test",
            "one row-level score\nafter selection", "#FDE8E8", RED)
    add_box(ax, (0.86, 0.50), 0.12, 0.24, "Reported result",
            "task-cluster summaries\nand coverage", LIGHT_GRAY, GRAY)

    # Forward flow.
    elbow(ax, [(0.09, 0.74), (0.09, 0.86), (0.24, 0.86)])          # arrays -> training
    elbow(ax, [(0.09, 0.50), (0.09, 0.38), (0.24, 0.38)])          # arrays -> validation
    elbow(ax, [(0.41, 0.86), (0.555, 0.86), (0.555, 0.74)])        # training -> frozen (top)
    elbow(ax, [(0.41, 0.38), (0.555, 0.38), (0.555, 0.50)])        # validation -> frozen (bottom)
    elbow(ax, [(0.64, 0.62), (0.70, 0.62)])                         # frozen -> test
    elbow(ax, [(0.81, 0.62), (0.86, 0.62)])                         # test -> reported

    # Historical feedback loop (dashed red, bottom band, right angles).
    add_box(
        ax,
        (0.44, 0.02),
        0.36,
        0.18,
        "Retrospective policy development",
        "archived test outcomes informed later gate families\nand task-specific refinements",
        "#FFF0F0",
        RED,
    )
    elbow(ax, [(0.755, 0.50), (0.755, 0.20)], color=RED, style="--", width=1.4)   # test -> feedback
    elbow(ax, [(0.44, 0.11), (0.325, 0.11), (0.325, 0.26)], color=RED, style="--", width=1.4)  # feedback -> validation

    ax.text(
        0.02,
        0.045,
        "Independent evidence is not implied by new RNG seeds on the same arrays.\n"
        "Time-series-1 and SECOM are reported separately as transfer stress tests.",
        ha="left",
        va="center",
        color=GRAY,
        fontsize=8,
    )

    fig.tight_layout(pad=0.4)
    for suffix in ("pdf", "png"):
        fig.savefig(OUT / f"study_information_boundaries.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def task_effects() -> None:
    v12 = pd.read_csv(V12_RESULTS)
    v11 = v12[v12["rule_name"].eq("v11_rare_signal_aggressive_else_strict")].copy()
    v15 = pd.read_csv(V15_ROWS)

    v11_task = v11.groupby("task_id")["delta_vs_safe_baseline_test"].agg(["mean", "std", "count"])
    v15_task = v15.groupby("task_id")["delta_vs_v12_safe"].agg(["mean", "std", "count"])
    tasks = list(sorted(set(v11_task.index).intersection(v15_task.index)))
    y = np.arange(len(tasks))

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for yi, task in enumerate(tasks):
        ax.plot(
            [v11_task.loc[task, "mean"], v15_task.loc[task, "mean"]],
            [yi - 0.10, yi + 0.10],
            color="#C7CDD4",
            linewidth=1.0,
            zorder=1,
        )
    ax.scatter(
        v11_task.loc[tasks, "mean"],
        y - 0.10,
        s=34,
        marker="o",
        color=BLUE,
        label="Task-identifier-free rare-signal gate",
        zorder=3,
    )
    ax.scatter(
        v15_task.loc[tasks, "mean"],
        y + 0.10,
        s=38,
        marker="D",
        color=ORANGE,
        label="Task-specific development policy",
        zorder=3,
    )
    ax.axvline(0.0, color="#333333", linewidth=0.8)
    labels = [f"Task {task}" + ("*" if task in {6, 9} else "") for task in tasks]
    ax.set_yticks(y, labels)
    ax.set_xlabel("Mean test PR-AUC difference from the validation-selected reference")
    ax.set_title("Task concentration of retrospective augmentation gains")
    ax.grid(axis="x", color="#E6E8EB", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=False)
    ax.text(
        0.01,
        -0.16,
        "* Tasks 6 and 9 were evaluated only after the expanded gate had been frozen;\n"
        "  neither policy selected a hybrid on them.",
        transform=ax.transAxes,
        fontsize=7.2,
        color=GRAY,
    )
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(OUT / f"task_level_policy_effects.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def gate_tradeoff() -> None:
    v12 = pd.read_csv(V12_RESULTS)
    rows = []
    short = {
        "strict_certificate": "Candidate-strict gate",
        "v11_grid_or_profile_strict": "Combined conservative gate",
        "v11_rare_signal_aggressive_else_strict": "Rare-signal gate",
        "v12_signal_aggressive_else_strict": "Opportunity-extension gate",
    }
    for rule, label in short.items():
        group = v12[v12["rule_name"].eq(rule)]
        rows.append(
            {
                "label": label,
                "coverage": group["selected_kind"].eq("hybrid").mean(),
                "mean": group["delta_vs_safe_baseline_test"].mean(),
                "selected": int(group["selected_kind"].eq("hybrid").sum()),
                "harm": int(group["harm_flag"].sum()),
                "kind": "task-ID-free",
            }
        )
    v15 = pd.read_csv(V15_ROWS)
    v15_selected = v15["delta_vs_v12_safe"].ne(0.0)
    rows.append(
        {
            "label": "Task-specific development policy",
            "coverage": float(v15_selected.mean()),
            "mean": v15["delta_vs_v12_safe"].mean(),
            "selected": int(v15_selected.sum()),
            "harm": int(v15["harm_vs_v12_safe"].sum()),
            "kind": "task-specific",
        }
    )
    frame = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    colors = frame["harm"].map(lambda value: RED if value else BLUE)
    markers = frame["kind"].map(lambda value: "D" if value == "task-specific" else "o")
    for row, color, marker in zip(frame.itertuples(index=False), colors, markers):
        ax.scatter(
            row.coverage,
            row.mean,
            s=45 + 12 * row.selected,
            color=color if row.kind == "task-ID-free" else ORANGE,
            marker=marker,
            alpha=0.88,
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        # Right-align the annotation for points near the right edge so the
        # text stays inside the axes.
        right_edge = row.coverage > 0.33
        ax.annotate(
            f"{row.label}\n{row.selected}/45 selected; {row.harm} negative",
            (row.coverage, row.mean),
            xytext=(-8, 6) if right_edge else (5, 6),
            textcoords="offset points",
            ha="right" if right_edge else "left",
            fontsize=7.2,
        )
    ax.axhline(0, color="#333333", linewidth=0.8)
    # Extra headroom so the top-right annotation stays inside the axes.
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax + 0.22 * (ymax - ymin))
    ax.set_xlabel("Hybrid coverage")
    ax.set_ylabel("Mean test PR-AUC difference")
    ax.set_title("Retrospective coverage--gain trade-off")
    ax.grid(color="#E6E8EB", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.text(
        0.01,
        -0.18,
        "Bubble area scales with selections. Orange diamond: task-specific development; red: observed negative rows.",
        transform=ax.transAxes,
        fontsize=7.2,
        color=GRAY,
    )
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(OUT / f"gate_coverage_gain_tradeoff.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    setup()
    information_boundaries()
    task_effects()
    gate_tradeoff()
    print(f"Wrote revision figures to {OUT}")


if __name__ == "__main__":
    main()
