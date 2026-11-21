"""Model evaluation, comparison tables, plots and research reports."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.metrics import ConfusionMatrixDisplay

from .data_loader import load_raw_data, resolve_path
from .data_validation import validate_target
from .feature_engineering import add_engineered_features
from .preprocessing import split_data
from .train import _scores

logger = logging.getLogger(__name__)


def load_test_data(config: dict[str, Any], feature_columns: list[str]) -> tuple[pd.DataFrame, pd.Series, Any]:
    """Reconstruct the exact test split used during training."""
    frame = load_raw_data(config=config)
    target = validate_target(frame, config)
    featured = add_engineered_features(frame, config)
    X = featured[feature_columns]
    split = split_data(X, target.labels, config)
    return split.X_test, split.y_test, split


def compute_metrics(y_true: np.ndarray, scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Compute all configured classification metrics.

    FPR and FAR are reported separately even though they are algebraically
    identical in this binary setting (FP/(FP+TN)); this avoids ambiguity about
    which false-alarm definition is being used.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, labels, labels=[0, 1]).ravel()
    denominator_fpr = fp + tn
    denominator_precision = tp + fp
    return {
        "accuracy": float(accuracy_score(y_true, labels)),
        "precision": float(precision_score(y_true, labels, zero_division=0)),
        "recall": float(recall_score(y_true, labels, zero_division=0)),
        "f1": float(f1_score(y_true, labels, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "pr_auc": float(average_precision_score(y_true, scores)),
        "fpr": float(fp / denominator_fpr) if denominator_fpr else 0.0,
        "far": float(fp / denominator_fpr) if denominator_fpr else 0.0,
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "precision_denominator": int(denominator_precision),
    }


def evaluate_all(
    bundle: dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    config: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Evaluate every saved model on the untouched test set."""
    results: dict[str, dict[str, Any]] = {}
    for name, entry in bundle["models"].items():
        pipeline = entry["pipeline"]
        threshold = float(entry["threshold"])
        scores = _scores(pipeline, X_test)
        labels = (scores >= threshold).astype(int)
        metrics = compute_metrics(y_test.to_numpy(), scores, labels)
        metrics.update({
            "model": name,
            "family": entry["family"],
            "decision_threshold": threshold,
        })
        curve_fpr, curve_tpr, _ = roc_curve(y_test, scores)
        curve_precision, curve_recall, _ = precision_recall_curve(y_test, scores)
        metrics["_roc_curve"] = (curve_fpr, curve_tpr)
        metrics["_pr_curve"] = (curve_recall, curve_precision)
        metrics["_scores"] = scores
        metrics["_labels"] = labels
        results[name] = metrics
        logger.info(
            "%s: recall=%.3f precision=%.3f FAR=%.3f PR-AUC=%.3f",
            name, metrics["recall"], metrics["precision"], metrics["far"], metrics["pr_auc"],
        )
    return results


def save_results_tables(results: dict[str, dict[str, Any]], config: dict[str, Any]) -> pd.DataFrame:
    """Save the model comparison CSV and a JSON copy of all numeric metrics."""
    results_dir = resolve_path(config["paths"]["results_dir"], config)
    results_dir.mkdir(parents=True, exist_ok=True)
    columns = [
        "model", "family", "decision_threshold", "accuracy", "precision", "recall", "f1",
        "roc_auc", "pr_auc", "fpr", "far", "tp", "fp", "tn", "fn",
    ]
    table = pd.DataFrame([
        {column: result[column] for column in columns} for result in results.values()
    ])
    table.to_csv(resolve_path(config["paths"]["comparison_csv"], config), index=False)

    serialisable = {
        name: {key: value for key, value in result.items() if not key.startswith("_")}
        for name, result in results.items()
    }
    path = resolve_path(config["paths"]["metrics_json"], config)
    path.write_text(json.dumps(serialisable, indent=2), encoding="utf-8")
    logger.info("Saved comparison table with %d models.", len(table))
    return table


def _plot_confusions(results: dict[str, dict[str, Any]], y_test: pd.Series, figures_dir: Path) -> None:
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(4.6 * n_models, 4.2), squeeze=False)
    for axis, (name, result) in zip(axes[0], results.items()):
        matrix = np.array([[result["tn"], result["fp"]], [result["fn"], result["tp"]]])
        display = ConfusionMatrixDisplay(
            confusion_matrix=matrix, display_labels=["Normal", "Kick"]
        )
        display.plot(ax=axis, cmap="Blues", colorbar=False, values_format="d")
        axis.set_title(name.replace("_", " "))
    fig.suptitle("Test-set confusion matrices (experimental proxy labels)")
    fig.tight_layout()
    fig.savefig(figures_dir / "confusion_matrices.png", dpi=200)
    plt.close(fig)


def _plot_roc(results: dict[str, dict[str, Any]], figures_dir: Path) -> None:
    fig, axis = plt.subplots(figsize=(6.4, 5.2))
    for name, result in results.items():
        fpr, tpr = result["_roc_curve"]
        axis.plot(fpr, tpr, lw=1.8, label=f"{name} (AUC={result['roc_auc']:.3f})")
    axis.plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
    axis.set_xlabel("False positive rate")
    axis.set_ylabel("True positive rate (kick recall)")
    axis.set_title("ROC curves - independent test set")
    axis.legend(fontsize=8, loc="lower right")
    axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures_dir / "roc_curves.png", dpi=200)
    plt.close(fig)


def _plot_pr(results: dict[str, dict[str, Any]], y_test: pd.Series, figures_dir: Path) -> None:
    fig, axis = plt.subplots(figsize=(6.4, 5.2))
    baseline = float(y_test.mean())
    for name, result in results.items():
        recall, precision = result["_pr_curve"]
        axis.plot(recall, precision, lw=1.8, label=f"{name} (AP={result['pr_auc']:.3f})")
    axis.axhline(baseline, color="k", ls="--", lw=1, label=f"Chance ({baseline:.3f})")
    axis.set_xlabel("Recall (kick)")
    axis.set_ylabel("Precision")
    axis.set_title("Precision-recall curves - independent test set")
    axis.legend(fontsize=8, loc="lower left")
    axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures_dir / "precision_recall_curves.png", dpi=200)
    plt.close(fig)


def save_evaluation_plots(
    results: dict[str, dict[str, Any]], y_test: pd.Series, config: dict[str, Any]
) -> None:
    """Save confusion matrices, ROC and precision-recall curves."""
    figures_dir = resolve_path(config["paths"]["figures_dir"], config)
    figures_dir.mkdir(parents=True, exist_ok=True)
    _plot_confusions(results, y_test, figures_dir)
    _plot_roc(results, figures_dir)
    _plot_pr(results, y_test, figures_dir)


def write_evaluation_report(
    results: dict[str, dict[str, Any]],
    table: pd.DataFrame,
    config: dict[str, Any],
) -> Path:
    """Write a Markdown evaluation report using only computed results."""
    reports_dir = resolve_path(config["paths"]["reports_dir"], config)
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / "model_evaluation.md"
    lines = [
        "# Model evaluation report",
        "",
        "> Experimental research prototype. Labels are proxies derived from a "
        "continuous pit-gain signal; see `reports/data_audit.md`.",
        "",
        "## Test-set comparison",
        "",
        "| Model | Family | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | FAR | TP | FP | FN | TN |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in table.iterrows():
        lines.append(
            f"| {row['model']} | {row['family']} | {row['accuracy']:.3f} | {row['precision']:.3f} | "
            f"{row['recall']:.3f} | {row['f1']:.3f} | {row['roc_auc']:.3f} | {row['pr_auc']:.3f} | "
            f"{row['far']:.3f} | {int(row['tp'])} | {int(row['fp'])} | {int(row['fn'])} | {int(row['tn'])} |"
        )
    supervised = [r for r in results.values() if r["family"] == "supervised"]
    best = max(supervised, key=lambda r: r["pr_auc"]) if supervised else max(
        results.values(), key=lambda r: r["pr_auc"]
    )
    lines += [
        "",
        "Decision thresholds were tuned on a validation split only; the test set was used once "
        "for reporting. Where validation scores were separable, the threshold was the midpoint "
        "between the largest normal score and the smallest kick score; otherwise the PR-curve "
        "metric was optimised. Operating points were constrained to at least "
        f"{float(config['evaluation'].get('minimum_recall', 0.0)):.0%} validation kick recall. "
        "FAR is defined as FP/(FP+TN), identical numerically to the false "
        "positive rate in this binary setting. PR-AUC is emphasised because the kick class is "
        "the minority and missed kicks are more serious than false warnings.",
        "",
        "## Engineering interpretation",
        "",
        f"- Highest supervised PR-AUC: **{best['model']}** ({best['pr_auc']:.3f}). "
        "PR-AUC, not accuracy, is the selection criterion.",
        f"- Missed kicks (false negatives) for that model: {best['fn']}; "
        f"false warnings: {best['fp']}.",
        "- A high recall with many false warnings may be preferable during investigation, "
        "but a high false-alarm count can cause alarm fatigue. The operating threshold must "
        "be chosen with drilling personnel, not from accuracy alone.",
        "- Both Isolation Forest variants are unsupervised anomaly detectors; one is fitted "
        "on mixed training blocks and one only on normal training rows. Their agreement "
        "with the proxy labels is a post-hoc comparison, not supervised classification.",
        "- The conventional rule uses an experimental flow-imbalance threshold tuned on "
        "validation data. It is not an accepted field threshold.",
        "- Near-perfect scores must be interpreted cautiously: the proxy label is derived "
        "from a pit-gain signal that is physically coupled to flow imbalance in this "
        "simulator. Separability is therefore expected and is not evidence of real-world "
        "detection performance.",
        "- These results come from one simulated rig scenario. They do not demonstrate "
        "readiness for real operations or validated early warning.",
        "",
        "## Figures",
        "",
        "- `results/figures/confusion_matrices.png`",
        "- `results/figures/roc_curves.png`",
        "- `results/figures/precision_recall_curves.png`",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote evaluation report to %s", path)
    return path
