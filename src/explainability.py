"""SHAP-based model explainability with physically cautious reporting."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data_loader import resolve_path
from .evaluate import load_test_data
from .train import _scores

logger = logging.getLogger(__name__)


def _transformed_features(pipeline: Any, X_test: pd.DataFrame) -> tuple[Any, list[str]]:
    """Return the matrix actually seen by the final estimator."""
    if "preprocess" in pipeline.named_steps:
        preprocessor = pipeline.named_steps["preprocess"]
        transformed = preprocessor.transform(X_test)
        names = [str(name) for name in preprocessor.get_feature_names_out()]
        return transformed, names
    return X_test, list(X_test.columns)


def generate_explanations(bundle: dict[str, Any], config: dict[str, Any]) -> Path | None:
    """Generate SHAP plots for the best validated model.

    Returns the path to the explainability report, or ``None`` if SHAP cannot
    explain the selected model (in which case a short note is written).
    """
    output_dir = resolve_path(config["paths"]["explainability_dir"], config)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "explainability_report.md"
    if not config.get("explainability", {}).get("enabled", True):
        report_path.write_text("SHAP explainability disabled in config.\n", encoding="utf-8")
        return report_path

    model_name = bundle["best_model"]
    pipeline = bundle["models"][model_name]["pipeline"]
    X_test, y_test, _ = load_test_data(config, bundle["feature_columns"])
    X_transformed, feature_names = _transformed_features(pipeline, X_test)
    estimator = pipeline.steps[-1][1]

    try:
        import shap
    except ImportError:
        report_path.write_text(
            "SHAP is not installed (`pip install shap`); no explanations were generated.\n",
            encoding="utf-8",
        )
        logger.warning("SHAP unavailable; skipping explanations.")
        return report_path

    try:
        tree_types = tuple(
            cls for cls in (
                getattr(__import__("sklearn.ensemble", fromlist=["RandomForestClassifier"]),
                        "RandomForestClassifier", None),
                getattr(__import__("sklearn.ensemble", fromlist=["ExtraTreesClassifier"]),
                        "ExtraTreesClassifier", None),
            ) if cls is not None
        )
        try:
            from xgboost import XGBClassifier
            tree_types += (XGBClassifier,)
        except ImportError:
            pass

        if isinstance(estimator, tree_types):
            explainer = shap.TreeExplainer(estimator)
        elif hasattr(estimator, "coef_"):
            explainer = shap.LinearExplainer(estimator, X_transformed)
        else:  # pragma: no cover - model-dependent fallback
            explainer = shap.Explainer(estimator, X_transformed)

        explanation = explainer(X_transformed)
        values = explanation.values
        if isinstance(values, list):
            values = values[1]
        if np.asarray(values).ndim == 3:
            # Multi-output classifiers: keep only the positive-class explanation.
            explanation = explanation[..., 1]
            values = explanation.values
        if np.asarray(values).ndim != 2:
            raise ValueError(f"Unexpected SHAP value shape: {np.asarray(values).shape}")

        mean_abs = np.abs(values).mean(axis=0)
        importance = sorted(
            zip(feature_names, mean_abs), key=lambda item: item[1], reverse=True
        )
        top_n = int(config.get("explainability", {}).get("top_features", 12))

        fig = plt.figure()
        shap.summary_plot(values, X_transformed, feature_names=feature_names, show=False, max_display=top_n)
        fig.savefig(output_dir / "shap_summary.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

        fig = plt.figure()
        shap.plots.bar(explanation, max_display=top_n, show=False)
        fig.savefig(output_dir / "shap_feature_importance.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

        top_feature = importance[0][0]
        top_index = feature_names.index(top_feature)
        fig = plt.figure()
        shap.dependence_plot(
            top_index, values, X_transformed,
            feature_names=feature_names, show=False,
        )
        fig.savefig(output_dir / f"shap_dependence_{top_feature}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

        scores = _scores(pipeline, X_test)
        labels = (scores >= float(bundle["models"][model_name]["threshold"])).astype(int)
        candidates = np.flatnonzero((y_test.to_numpy() == 1) & (labels == 1))
        row_index = int(candidates[0]) if candidates.size else 0
        fig = plt.figure()
        shap.plots.waterfall(explanation[row_index], max_display=top_n, show=False)
        fig.savefig(output_dir / "shap_individual_explanation.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

        lines = [
            "# SHAP explainability report",
            "",
            f"- Model explained: `{model_name}` (selected on validation PR-AUC/ROC-AUC).",
            f"- Explainability method: `{type(explainer).__name__}`.",
            f"- Test observations explained: {len(X_test)}; example row: {row_index}.",
            "",
            "## Mean absolute SHAP magnitude (test set)",
            "",
            "| Rank | Feature | Mean |SHAP| |",
            "| ---: | --- | ---: |",
        ]
        for rank, (name, value) in enumerate(importance[:top_n], start=1):
            lines.append(f"| {rank} | {name} | {value:.4f} |")
        lines += [
            "",
            "## Engineering interpretation",
            "",
            "Features with large SHAP magnitude strongly influence the model output on this "
            "test set. For a kick, a positive shift is physically consistent with increasing "
            "return flow, pit gain, or changing pressure/force readings. SHAP importance is "
            "associational: it does not prove that a variable physically causes a kick.",
            "",
            "Because `ActiveGL` and `ATVolume` were excluded as target-derived variables, "
            "the conclusions concern precursor-like measurements such as flow imbalance, "
            "pressure and mechanical response, not the label itself.",
            "",
            "## Outputs",
            "",
            "- `shap_summary.png`",
            "- `shap_feature_importance.png`",
            f"- `shap_dependence_{top_feature}.png`",
            "- `shap_individual_explanation.png`",
            "",
        ]
        report_path.write_text("\n".join(lines), encoding="utf-8")
        logger.info("Wrote SHAP explanations to %s", output_dir)
        return report_path
    except Exception as exc:  # pragma: no cover - depends on installed versions
        message = f"SHAP explanation failed for {model_name}: {type(exc).__name__}: {exc}\n"
        logger.exception(message.strip())
        report_path.write_text(message, encoding="utf-8")
        return report_path
