#!/usr/bin/env python3
"""Generate publication-quality paper figures from CSVs using Seaborn.

Produces:
  - leaderboard_errorbars.pdf  (horizontal bar chart with error bars, rank-band shading)
  - rubric_heatmap.pdf         (annotated heatmap of per-rubric pass rates)
  - variance_dotplot.pdf       (dot-plot of per-model run-to-run sigma)

Template: CEURART single-column (textwidth ≈ 160 mm ≈ 6.3 in).
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns

HERE = Path(__file__).resolve().parent

# ── Seaborn theme ────────────────────────────────────────────────────────────

sns.set_theme(
    style="whitegrid",
    context="paper",
    font_scale=1.0,
    rc={
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "font.size": 9,
        "axes.linewidth": 0.6,
        "grid.linewidth": 0.4,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    },
)

# CEURART single-column text width in inches (~160 mm)
COLUMN_WIDTH = 6.3

# ── Band palette (ranks 1–7 vs 8–12; not a 4.6 score cut) ────────────────────

BAND_COLORS = {
    "high": "#3a86a8",  # teal-blue for ranks 1–7
    "low": "#7fb685",   # muted green for ranks 8–12
}


# ── Data readers ─────────────────────────────────────────────────────────────

def read_leaderboard() -> tuple[list[str], np.ndarray, np.ndarray]:
    path = HERE / "leaderboard.csv"
    models: list[str] = []
    means: list[float] = []
    sigmas: list[float] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            models.append(row["model"])
            means.append(float(row["mean"]))
            sigmas.append(float(row["sigma_run"]))
    return models, np.array(means), np.array(sigmas)


def read_rubric_matrix() -> tuple[list[str], list[str], np.ndarray]:
    """Return model names (sorted by total desc), rubric columns, matrix (n, 5)."""
    path = HERE / "rubric_scores.csv"
    rubric_cols = ["intent", "clarification", "recommendation", "cart", "grounding"]
    rows: list[dict[str, str]] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    rows.sort(key=lambda r: float(r["total"]), reverse=True)
    models = [r["model"] for r in rows]
    mat = np.array([[float(r[c]) for c in rubric_cols] for r in rows], dtype=float)
    labels = ["Intent", "Clarification", "Recommendation", "Add-to-Cart", "Grounding"]
    return models, labels, mat


# ── Figure 3: Leaderboard with error bars ────────────────────────────────────

def plot_leaderboard() -> None:
    models, means, sigmas = read_leaderboard()
    # Sort ascending for bottom-to-top layout (highest score at top after invert_yaxis)
    order = np.argsort(means)
    models = [models[i] for i in order]
    means = means[order]
    sigmas = sigmas[order]

    n = len(models)
    # Ascending sort: last 7 entries are ranks 1–7 (highest scores).
    colors = [
        BAND_COLORS["high"] if i >= n - 7 else BAND_COLORS["low"]
        for i in range(n)
    ]
    # 0.20 in per row + 0.8 in margins — compact enough for a workshop paper
    fig_h = max(2.8, 0.20 * n + 0.8)
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, fig_h))

    y = np.arange(n)
    ax.barh(
        y, means, xerr=sigmas,
        color=colors, edgecolor="white", linewidth=0.3,
        error_kw=dict(ecolor="#444444", capsize=2.0, capthick=0.6, elinewidth=0.6),
        height=0.55, zorder=3,
    )

    # Value annotations
    for i, (mean, sigma) in enumerate(zip(means, sigmas)):
        ax.text(
            mean + sigma + 0.02, i, f"{mean:.2f}",
            va="center", ha="left", fontsize=7, color="#333333",
        )

    ax.set_yticks(y)
    ax.set_yticklabels(models, fontsize=7.5)
    ax.set_xlabel("Mean rubric score (/5)", fontsize=8.5, labelpad=3)
    ax.set_xlim(2.8, 5.25)
    ax.invert_yaxis()
    ax.xaxis.grid(True, linestyle="--", alpha=0.4, linewidth=0.4)
    ax.yaxis.grid(False)
    ax.set_axisbelow(True)

    legend_patches = [
        mpatches.Patch(color=BAND_COLORS["high"], label="Ranks 1–7"),
        mpatches.Patch(color=BAND_COLORS["low"], label="Ranks 8–12"),
    ]
    ax.legend(
        handles=legend_patches, loc="upper center",
        bbox_to_anchor=(0.5, -0.07), ncol=2,
        fontsize=7.5, framealpha=0.9, edgecolor="#cccccc",
        borderpad=0.3, columnspacing=0.8,
    )

    sns.despine(left=True, bottom=False)
    fig.tight_layout(pad=0.4)
    out = HERE / "leaderboard_errorbars.pdf"
    fig.savefig(out)
    plt.close(fig)


# ── Figure 4: Per-rubric heatmap ─────────────────────────────────────────────

def plot_rubric_heatmap() -> None:
    models, col_labels, mat = read_rubric_matrix()

    n = len(models)
    fig_h = max(2.8, 0.19 * n + 0.8)
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, fig_h))

    sns.heatmap(
        mat,
        ax=ax,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        vmin=0.4, vmax=1.0,
        linewidths=0.4,
        linecolor="white",
        cbar_kws={
            "label": "Pass rate",
            "shrink": 0.75,
            "aspect": 30,
            "pad": 0.02,
        },
        annot_kws={"size": 7},
        square=False,
    )

    ax.set_xticklabels(col_labels, rotation=30, ha="right", fontsize=8)
    ax.set_yticklabels(models, rotation=0, fontsize=7.5)
    ax.tick_params(axis="both", length=0)

    fig.tight_layout(pad=0.4)
    out = HERE / "rubric_heatmap.pdf"
    fig.savefig(out)
    plt.close(fig)


# ── Figure 5: Variance dot-plot ──────────────────────────────────────────────

def plot_variance_dotplot() -> None:
    models, means, sigmas = read_leaderboard()
    order = np.argsort(means)[::-1]
    models = [models[i] for i in order]
    means = means[order]
    sigmas = sigmas[order]

    n = len(models)
    colors = [
        BAND_COLORS["high"] if i < 7 else BAND_COLORS["low"]
        for i in range(n)
    ]
    fig_h = max(3.5, 0.28 * n + 0.8)
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, fig_h))

    y = np.arange(n)
    ax.scatter(sigmas, y, c=colors, s=55, zorder=3, edgecolors="white", linewidths=0.5)

    for yi in y:
        ax.axhline(yi, color="#eeeeee", linewidth=0.4, zorder=1)

    for i, (s, _) in enumerate(zip(sigmas, models)):
        ax.text(s + 0.006, i, f"{s:.2f}", va="center", ha="left", fontsize=8, color="#555555")

    ax.set_yticks(y)
    ax.set_yticklabels(models, fontsize=8)
    ax.set_xlabel("Run-to-run σ (5 runs × 20 scenarios)", fontsize=9, labelpad=4)
    ax.set_xlim(-0.01, max(sigmas) + 0.07)
    ax.invert_yaxis()
    ax.xaxis.grid(True, linestyle="--", alpha=0.4, linewidth=0.4)
    ax.yaxis.grid(False)
    ax.set_axisbelow(True)

    sns.despine(left=True, bottom=False)
    fig.tight_layout(pad=0.4)
    out = HERE / "variance_dotplot.pdf"
    fig.savefig(out)
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    plot_leaderboard()
    print(f"  Wrote {HERE / 'leaderboard_errorbars.pdf'}")
    plot_rubric_heatmap()
    print(f"  Wrote {HERE / 'rubric_heatmap.pdf'}")
    plot_variance_dotplot()
    print(f"  Wrote {HERE / 'variance_dotplot.pdf'}")


if __name__ == "__main__":
    main()
