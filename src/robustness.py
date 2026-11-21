"""Controlled robustness experiments on evaluation inputs.

Perturbations are applied to copies of test observations only. The raw CSV and
the fitted model are never modified. Noise levels are fractions of the training
standard deviation of each sensor, not verified sensor specifications.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data_loader import load_raw_data, resolve_path
from .data_validation import validate_target
from .evaluate import compute_metrics
from .feature_engineering import add_engineered_features
from .preprocessing import split_data
from .train import _scores

logger = logging.getLogger(__name__)


def _load_test_parts(config: dict[str, Any], feature_columns: list[str]) -> dict[str, Any]:
    """Reconstruct raw, featured and split test observations."""
    frame = load_raw_data(config=config)
    featured = add_engineered_features(frame, config)
    target = validate_target(frame, config)
    X = featured[feature_columns]
    split = split_data(X, target.labels, config)
    return {
        "raw_train": frame.iloc[split.train_indices].reset_index(drop=True),
        "raw_test": frame.iloc[split.test_indices].reset_index(drop=True),
        "featured_test": featured.iloc[split.test_indices].reset_index(drop=True),
        "y_test": split.y_test,
        "X_train": split.X_train,
    }


def perturb_sensors(
    featured: pd.DataFrame,
    sensor_columns: list[str],
    train_std: pd.Series,
    level: float,
    seed: int,
    kind: str,
) -> pd.DataFrame:
    """Apply Gaussian noise or MCAR missingness to sensor readings.

    Args:
        featured: Featured test frame (copy is made).
        sensor_columns: Physical sensor columns to perturb.
        train_std: Training-set standard deviations used for noise scaling.
        level: Fraction (noise) or probability (missingness).
        seed: Reproducible random seed.
        kind: ``noise`` or ``missing``.

    Returns:
        Perturbed frame with deterministic relationships (MRFlow) recomputed.
    """
    output = featured.copy()
    rng = np.random.default_rng(seed)
    for column in sensor_columns:
        if column not in output.columns:
            continue
        if kind == "noise":
            scale = float(train_std.get(column, 1.0)) * level
            output[column] = output[column].to_numpy(dtype=float) + rng.normal(0.0, scale, len(output))
        elif kind == "missing":
            mask = rng.random(len(output)) < level
            output.loc[mask, column] = np.nan
        else:
            raise ValueError(f"Unsupported perturbation kind: {kind!r}")
    # MRFlow is an exact function of FIn/FOut in the simulator; keep it consistent.
    if {"MRFlow", "FIn", "FOut"}.issubset(output.columns):
        output["MRFlow"] = output["FIn"] - output["FOut"]
    return output


def run_robustness(
    bundle: dict[str, Any],
    config: dict[str, Any],
) -> pd.DataFrame:
    """Run all configured noise/missingness experiments and save results."""
    parts = _load_test_parts(config, bundle["feature_columns"])
    sensor_columns = list(config["robustness"].get("sensor_columns", []))
    seed = int(config["robustness"].get("seed", config["project"]["seed"]))
    train_std = parts["X_train"].std()
    y_test = parts["y_test"].to_numpy()

    records: list[dict[str, Any]] = []

    def evaluate_variant(variant: pd.DataFrame, model_name: str, perturbation: str, level: float) -> None:
        entry = bundle["models"][model_name]
        X_variant = variant[bundle["feature_columns"]]
        scores = _scores(entry["pipeline"], X_variant)
        labels = (scores >= float(entry["threshold"])).astype(int)
        metrics = compute_metrics(y_test, scores, labels)
        records.append({
            "model": model_name,
            "perturbation": perturbation,
            "level": level,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "roc_auc": metrics["roc_auc"],
            "pr_auc": metrics["pr_auc"],
            "far": metrics["far"],
            "missed_kicks": metrics["fn"],
            "false_warnings": metrics["fp"],
        })

    baseline_frame = add_engineered_features(parts["raw_test"], config)
    for model_name in bundle["models"]:
        evaluate_variant(baseline_frame, model_name, "baseline", 0.0)

    for level in config["robustness"].get("noise_levels", []):
        noisy = perturb_sensors(
            parts["featured_test"], sensor_columns, train_std, float(level), seed, "noise"
        )
        for model_name in bundle["models"]:
            evaluate_variant(noisy, model_name, "sensor_noise", float(level))
    for level in config["robustness"].get("missing_levels", []):
        missing = perturb_sensors(
            parts["featured_test"], sensor_columns, train_std, float(level), seed + 1, "missing"
        )
        for model_name in bundle["models"]:
            evaluate_variant(missing, model_name, "missing_data", float(level))

    table = pd.DataFrame(records)
    results_dir = resolve_path(config["paths"]["results_dir"], config)
    results_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(results_dir / "robustness_results.csv", index=False)
    logger.info("Saved robustness results: %d rows", len(table))
    return table


def save_robustness_plots(table: pd.DataFrame, config: dict[str, Any]) -> None:
    """Plot recall and PR-AUC degradation versus perturbation level."""
    figures_dir = resolve_path(config["paths"]["figures_dir"], config)
    figures_dir.mkdir(parents=True, exist_ok=True)
    for perturbation, filename in (("sensor_noise", "robustness_noise.png"),
                                   ("missing_data", "robustness_missing.png")):
        subset = table[table["perturbation"] == perturbation]
        if subset.empty:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        for metric, axis in zip(("recall", "pr_auc"), axes):
            for model_name, group in subset.groupby("model"):
                group = group.sort_values("level")
                axis.plot(group["level"], group[metric], marker="o", lw=1.6, label=model_name)
            axis.set_xlabel("Noise level (x training SD)" if perturbation == "sensor_noise"
                            else "Missing-data fraction")
            axis.set_ylabel(metric.replace("_", " ").title())
            axis.set_ylim(-0.02, 1.05)
            axis.grid(alpha=0.3)
            axis.legend(fontsize=7)
        fig.suptitle(f"Robustness to {perturbation.replace('_', ' ')} - test inputs")
        fig.tight_layout()
        fig.savefig(figures_dir / filename, dpi=200)
        plt.close(fig)


def write_robustness_report(table: pd.DataFrame, config: dict[str, Any]) -> Path:
    """Write a Markdown summary of the robustness results."""
    reports_dir = resolve_path(config["paths"]["reports_dir"], config)
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / "robustness_results.md"
    lines = [
        "# Robustness testing",
        "",
        "Noise is Gaussian with standard deviation `level x training-set standard deviation` "
        "of each listed sensor. Missingness is masked completely at random. Neither level is "
        "a verified sensor specification. All perturbations are applied to copies of test rows; "
        "models and raw data are unchanged. Seed: "
        f"{config['robustness'].get('seed', config['project']['seed'])}.",
        "",
        "| Model | Perturbation | Level | Recall | PR-AUC | FAR | Missed kicks | False warnings |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in table.iterrows():
        lines.append(
            f"| {row['model']} | {row['perturbation']} | {row['level']} | {row['recall']:.3f} | "
            f"{row['pr_auc']:.3f} | {row['far']:.3f} | {int(row['missed_kicks'])} | "
            f"{int(row['false_warnings'])} |"
        )
    lines += ["", "Full numerical results: `results/robustness_results.csv`.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
