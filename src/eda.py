"""Exploratory data analysis used by both the CLI and notebook 01."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .data_loader import resolve_path
from .data_validation import TargetInfo
from .feature_engineering import add_engineered_features

logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid", context="notebook")
KEY_PARAMETERS = ["FOut", "FIn", "DPPress", "HLoad", "WoBit", "FDensity", "Flow_Imbalance"]


def generate_figures(
    frame: pd.DataFrame,
    target: TargetInfo,
    config: dict[str, Any],
) -> list[str]:
    """Save publication-quality EDA figures and return their filenames."""
    figures_dir = resolve_path(config["paths"]["figures_dir"], config)
    figures_dir.mkdir(parents=True, exist_ok=True)
    featured = add_engineered_features(frame, config)
    labels = target.labels.to_numpy()
    created: list[str] = []

    # Target distribution
    fig, axis = plt.subplots(figsize=(6, 4))
    counts = pd.Series(labels).map(
        {target.negative_label: "Normal drilling", target.positive_label: "Potential kick"}
    ).value_counts()
    sns.barplot(
        x=counts.index, y=counts.to_numpy(), hue=counts.index,
        legend=False, ax=axis, palette="Blues_d",
    )
    axis.set_ylabel("Observations")
    axis.set_title(f"Proxy target distribution ({target.positive_fraction:.1%} positive)")
    fig.tight_layout()
    fig.savefig(figures_dir / "target_distribution.png", dpi=200)
    plt.close(fig)
    created.append("target_distribution.png")

    # Missing values and duplicates overview
    missing = frame.isna().sum()
    fig, axis = plt.subplots(figsize=(8, 4))
    axis.bar(missing.index, missing.to_numpy())
    axis.set_ylabel("Missing values")
    axis.set_title("Missing-value analysis")
    axis.tick_params(axis="x", rotation=75)
    fig.tight_layout()
    fig.savefig(figures_dir / "missing_values.png", dpi=200)
    plt.close(fig)
    created.append("missing_values.png")

    # Correlation heatmap (numeric variables only)
    numeric = featured.select_dtypes(include=[np.number])
    fig, axis = plt.subplots(figsize=(13, 10))
    mask = np.triu(np.ones_like(numeric.corr(), dtype=bool), k=1)
    sns.heatmap(
        numeric.corr(), mask=mask, cmap="coolwarm", center=0, vmin=-1, vmax=1,
        square=True, cbar_kws={"shrink": 0.75}, ax=axis,
    )
    axis.set_title("Pearson correlation of recorded drilling variables")
    fig.tight_layout()
    fig.savefig(figures_dir / "correlation_heatmap.png", dpi=200)
    plt.close(fig)
    created.append("correlation_heatmap.png")

    # Feature distributions
    plot_columns = [c for c in KEY_PARAMETERS if c in featured.columns][:12]
    fig, axes = plt.subplots(3, 4, figsize=(14, 9))
    for axis, column in zip(axes.ravel(), plot_columns):
        sns.histplot(featured[column].dropna(), bins=40, ax=axis, color="#2b6cb0")
        axis.set_title(column)
    for axis in axes.ravel()[len(plot_columns):]:
        axis.set_visible(False)
    fig.suptitle("Distributions of important drilling parameters")
    fig.tight_layout()
    fig.savefig(figures_dir / "feature_distributions.png", dpi=200)
    plt.close(fig)
    created.append("feature_distributions.png")

    # Boxplots by proxy class
    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    for axis, column in zip(axes.ravel(), plot_columns):
        sns.boxplot(
            x=labels, y=featured[column], ax=axis,
            hue=labels, legend=False, palette=["#4c78a8", "#e45756"],
        )
        axis.set_xticks([0, 1])
        axis.set_xticklabels(["Normal", "Kick"])
        axis.set_title(column)
    for axis in axes.ravel()[len(plot_columns):]:
        axis.set_visible(False)
    fig.suptitle("Key drilling parameters by experimental proxy class")
    fig.tight_layout()
    fig.savefig(figures_dir / "boxplots_by_class.png", dpi=200)
    plt.close(fig)
    created.append("boxplots_by_class.png")

    # Flow and pressure relationships
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].scatter(featured["FIn"], featured["FOut"], c=labels, cmap="coolwarm", s=8, alpha=0.6)
    axes[0].set_xlabel("FIn")
    axes[0].set_ylabel("FOut")
    axes[0].set_title("Flow in vs flow out")
    axes[1].scatter(featured["DPPress"], featured["HLoad"], c=labels, cmap="coolwarm", s=8, alpha=0.6)
    axes[1].set_xlabel("DPPress")
    axes[1].set_ylabel("HLoad")
    axes[1].set_title("Pressure vs hook load")
    axes[2].scatter(featured["Flow_Imbalance"], target.target_values, c=labels, cmap="coolwarm", s=8, alpha=0.6)
    axes[2].set_xlabel("Flow_Imbalance (FOut - FIn)")
    axes[2].set_ylabel("ActiveGL (target signal)")
    axes[2].set_title("Flow imbalance vs pit gain")
    fig.tight_layout()
    fig.savefig(figures_dir / "flow_pressure_relationships.png", dpi=200)
    plt.close(fig)
    created.append("flow_pressure_relationships.png")

    logger.info("Generated %d EDA figures in %s", len(created), figures_dir)
    return created


def write_eda_report(
    frame: pd.DataFrame,
    target: TargetInfo,
    config: dict[str, Any],
    figures: list[str],
) -> Path:
    """Write an EDA report grounded in computed observations."""
    reports_dir = resolve_path(config["paths"]["reports_dir"], config)
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / "eda_report.md"
    numeric = frame.select_dtypes(include=[np.number])
    correlations = numeric.corr(numeric_only=True)[target.column].drop(target.column)
    top = correlations.reindex(correlations.abs().sort_values(ascending=False).index).head(8)
    lines = [
        "# Exploratory data analysis report",
        "",
        "> Generated from the raw DataDRILL CSV. Correlations are observations, not causal claims.",
        "",
        f"- Dataset shape: {frame.shape[0]} rows x {frame.shape[1]} columns.",
        f"- Missing values: {int(frame.isna().sum().sum())} total.",
        f"- Exact duplicate rows: {int(frame.duplicated().sum())}.",
        f"- Target: `{target.column}` ({target.mode}); {target.n_negative} normal / "
        f"{target.n_positive} proxy kick ({target.positive_fraction:.1%} positive).",
        f"- Proxy onset row: {target.onset_index}; depth: {target.onset_depth} ft.",
        "",
        "## Strongest observed associations with the continuous target signal",
        "",
        "| Variable | Pearson r |",
        "| --- | ---: |",
    ]
    for name, value in top.items():
        lines.append(f"| {name} | {value:.4f} |")
    imbalance_phrase = (
        "present but moderate" if 0.2 <= target.positive_fraction <= 0.8 else "substantial"
    )
    lines += [
        "",
        "## Petroleum-engineering interpretation",
        "",
        "- `ATVolume` is a near-exact copy of `ActiveGL` plus a nearly constant offset "
        "(~199.083), so its apparent association is redundancy, not independent evidence.",
        "- Increasing `FOut` relative to `FIn` (positive `Flow_Imbalance`) is consistent "
        "with formation fluid entering the wellbore and increasing return flow.",
        "- Pressure and mechanical variables (`DPPress`, `HLoad`, `WoBit`) change as the "
        "simulated formation pressure exceeds hydrostatic pressure. These are associations "
        "within one simulated scenario, not universal physical laws.",
        f"- Class imbalance is {imbalance_phrase}; "
        "recall and precision are therefore more informative than accuracy.",
        "- No timestamp or event identifier is available, so this EDA cannot validate "
        "early-warning lead time.",
        "",
        "## Figures",
        "",
    ]
    lines += [f"- `results/figures/{name}`" for name in figures]
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
