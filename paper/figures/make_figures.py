#!/usr/bin/env python3
"""Generate publication-quality paper figures from CSVs using Seaborn.

Produces:
  - leaderboard_errorbars.pdf  (horizontal bar chart with error bars, tier shading)
  - rubric_heatmap.pdf         (annotated heatmap of per-rubric pass rates)
  - variance_dotplot.pdf       (dot-plot of per-model run-to-run sigma)

Style reference: RecBench+ (Huang et al., WSDM 2026).
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
    font_scale=0.95,
    rc={
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "axes.linewidth": 0.6,
        "grid.linewidth": 0.4,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    },
)

COLUMN_WIDTH = 3.35  # acmart sigconf single-column width in inches

# ── Tier palette ─────────────────────────────────────────────────────────────

TIER_COLORS = {
    "top": "#3a86a8",     # teal-blue for top tier (>=4.6)
    "mid": "#7fb685",     # muted green for mid tier (>=3.5)
    "low": "#d4856a",     # warm coral for lower tier
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
    order = np.argsort(means)
    models = [models[i] for i in order]
    means = means[order]
    sigmas = sigmas[order]

    colors = []
    for m in means:
        if m >= 4.6:
            colors.append(TIER_COLORS["top"])
        elif m >= 3.5:
            colors.append(TIER_COLORS["mid"])
        else:
            colors.append(TIER_COLORS["low"])

    fig_h = max(2.4, 0.28 * len(models) + 0.8)
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, fig_h))

    y = np.arange(len(models))
    bars = ax.barh(
        y, means, xerr=sigmas,
        color=colors, edgecolor="white", linewidth=0.3,
        error_kw=dict(ecolor="#444444", capsize=2.5, capthick=0.7, elinewidth=0.7),
        height=0.62, zorder=3,
    )

    # Value annotations
    for i, (mean, sigma) in enumerate(zip(means, sigmas)):
        ax.text(
            mean + sigma + 0.03, i, f"{mean:.2f}",
            va="center", ha="left", fontsize=6.5, color="#333333",
        )

    ax.set_yticks(y)
    ax.set_yticklabels(models, fontsize=7.5)
    ax.set_xlabel("Mean rubric score (/5)", fontsize=8.5, labelpad=4)
    ax.set_xlim(2.8, 5.15)
    ax.invert_yaxis()
    ax.xaxis.grid(True, linestyle="--", alpha=0.4, linewidth=0.4)
    ax.yaxis.grid(False)
    ax.set_axisbelow(True)

    # Tier legend
    legend_patches = [
        mpatches.Patch(color=TIER_COLORS["top"], label="Top tier (≥4.6)"),
        mpatches.Patch(color=TIER_COLORS["mid"], label="Mid tier (≥3.5)"),
        mpatches.Patch(color=TIER_COLORS["low"], label="Lower tier (<3.5)"),
    ]
    ax.legend(
        handles=legend_patches, loc="lower right",
        fontsize=6, framealpha=0.9, edgecolor="#cccccc",
    )

    sns.despine(left=True, bottom=False)
    fig.tight_layout()
    out = HERE / "leaderboard_errorbars.pdf"
    fig.savefig(out)
    plt.close(fig)


# ── Figure 4: Per-rubric heatmap ─────────────────────────────────────────────

def plot_rubric_heatmap() -> None:
    models, col_labels, mat = read_rubric_matrix()

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.8))

    sns.heatmap(
        mat,
        ax=ax,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        vmin=0.4, vmax=1.0,
        linewidths=0.8,
        linecolor="white",
        cbar_kws={
            "label": "Pass rate",
            "shrink": 0.85,
            "aspect": 20,
        },
        annot_kws={"size": 7},
        square=False,
    )

    ax.set_xticklabels(col_labels, rotation=30, ha="right", fontsize=7.5)
    ax.set_yticklabels(models, rotation=0, fontsize=7.5)
    ax.tick_params(axis="both", length=0)

    fig.tight_layout()
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

    colors = []
    for m in means:
        if m >= 4.6:
            colors.append(TIER_COLORS["top"])
        elif m >= 3.5:
            colors.append(TIER_COLORS["mid"])
        else:
            colors.append(TIER_COLORS["low"])

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.4))

    y = np.arange(len(models))
    ax.scatter(sigmas, y, c=colors, s=45, zorder=3, edgecolors="white", linewidths=0.5)

    # Horizontal reference lines
    for yi in y:
        ax.axhline(yi, color="#eeeeee", linewidth=0.4, zorder=1)

    # Annotations
    for i, (s, m) in enumerate(zip(sigmas, models)):
        ax.text(s + 0.008, i, f"{s:.2f}", va="center", ha="left", fontsize=6.5, color="#555555")

    ax.set_yticks(y)
    ax.set_yticklabels(models, fontsize=7.5)
    ax.set_xlabel("Run-to-run σ (5 runs × 20 scenarios)", fontsize=8, labelpad=4)
    ax.set_xlim(-0.01, max(sigmas) + 0.08)
    ax.invert_yaxis()
    ax.xaxis.grid(True, linestyle="--", alpha=0.4, linewidth=0.4)
    ax.yaxis.grid(False)
    ax.set_axisbelow(True)

    sns.despine(left=True, bottom=False)
    fig.tight_layout()
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
