"""Model training with validation-only threshold selection."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_curve, roc_auc_score
from sklearn.pipeline import Pipeline

from .data_loader import resolve_path
from .feature_engineering import add_engineered_features, select_feature_columns
from .preprocessing import build_preprocessor, make_validation_split, split_data
from .data_validation import validate_target

logger = logging.getLogger(__name__)


class AnomalyClassifier(BaseEstimator, ClassifierMixin):
    """Expose an unsupervised anomaly detector with a classifier-like interface.

    The model is fitted WITHOUT labels. Its scores are converted to an empirical
    percentile of the training anomaly-score distribution so that a single
    threshold can be tuned on validation data. This remains an anomaly detector:
    labels are used only for post-hoc evaluation and threshold calibration.
    """

    def __init__(self, estimator: Any = None):
        self.estimator = estimator if estimator is not None else IsolationForest(random_state=42)

    def fit(self, X: Any, y: Any = None) -> "AnomalyClassifier":
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X)
        self.fit_scores_ = -np.asarray(self.estimator_.decision_function(X), dtype=float)
        self.classes_ = np.array([0, 1])
        return self

    def decision_function(self, X: Any) -> np.ndarray:
        return -np.asarray(self.estimator_.decision_function(X), dtype=float)

    def predict_proba(self, X: Any) -> np.ndarray:
        scores = self.decision_function(X)
        percentile = np.searchsorted(np.sort(self.fit_scores_), scores) / len(self.fit_scores_)
        percentile = np.clip(percentile, 0.0, 1.0)
        return np.column_stack([1.0 - percentile, percentile])

    def predict(self, X: Any) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {"estimator": self.estimator}


class ConventionalBaseline(BaseEstimator, ClassifierMixin):
    """Experimental threshold rule: kick when a single indicator exceeds a limit.

    This is NOT an operational drilling threshold. The limit is tuned on
    validation data and reported as a research-only comparison.
    """

    def __init__(self, indicator: str = "Flow_Imbalance", threshold: float = 0.0):
        self.indicator = indicator
        self.threshold = threshold

    def fit(self, X: Any, y: Any = None) -> "ConventionalBaseline":
        if self.indicator not in X.columns:
            raise ValueError(f"Baseline indicator '{self.indicator}' is not available.")
        scores = self.decision_function(X)
        self.score_min_ = float(scores.min())
        self.score_span_ = max(float(np.ptp(scores)), 1e-9)
        self.classes_ = np.array([0, 1])
        return self

    def decision_function(self, X: Any) -> np.ndarray:
        return X[self.indicator].to_numpy(dtype=float)

    def predict_proba(self, X: Any) -> np.ndarray:
        scores = self.decision_function(X)
        scaled = np.clip((scores - self.score_min_) / self.score_span_, 0.0, 1.0)
        return np.column_stack([1.0 - scaled, scaled])

    def predict(self, X: Any) -> np.ndarray:
        return (self.decision_function(X) > self.threshold).astype(int)


@dataclass
class TrainingResult:
    """Trained pipelines, tuned thresholds and provenance metadata."""

    bundle: dict[str, Any]
    feature_columns: list[str]
    threshold_details: dict[str, dict[str, float]]
    validation_summary: dict[str, dict[str, float]]
    models_used: list[str]


def _estimators(config: dict[str, Any], y_train: pd.Series) -> dict[str, Any]:
    """Instantiate configured estimators, with a documented XGBoost fallback."""
    model_cfg = config["models"]
    estimators: dict[str, Any] = {}
    if model_cfg["logistic_regression"].get("enabled", True):
        cfg = model_cfg["logistic_regression"]
        estimators["logistic_regression"] = LogisticRegression(
            C=float(cfg["C"]), max_iter=int(cfg["max_iter"]),
            class_weight=cfg.get("class_weight"), random_state=int(config["project"]["seed"]),
        )
    if model_cfg["random_forest"].get("enabled", True):
        cfg = model_cfg["random_forest"]
        estimators["random_forest"] = RandomForestClassifier(
            n_estimators=int(cfg["n_estimators"]), min_samples_leaf=int(cfg["min_samples_leaf"]),
            max_depth=cfg.get("max_depth"), class_weight=cfg.get("class_weight"),
            n_jobs=int(cfg.get("n_jobs", -1)), random_state=int(config["project"]["seed"]),
        )
    if model_cfg["xgboost"].get("enabled", True):
        cfg = model_cfg["xgboost"]
        try:
            from xgboost import XGBClassifier
        except ImportError:
            logger.warning("XGBoost unavailable; using HistGradientBoostingClassifier fallback.")
            estimators["hist_gradient_boosting"] = HistGradientBoostingClassifier(
                max_iter=int(cfg["n_estimators"]), max_depth=cfg.get("max_depth"),
                learning_rate=float(cfg["learning_rate"]), l2_regularization=float(cfg.get("reg_lambda", 1.0)),
                random_state=int(config["project"]["seed"]),
            )
        else:
            pos = int(y_train.sum())
            neg = int(len(y_train) - pos)
            estimators["xgboost"] = XGBClassifier(
                n_estimators=int(cfg["n_estimators"]), max_depth=int(cfg["max_depth"]),
                learning_rate=float(cfg["learning_rate"]), subsample=float(cfg["subsample"]),
                colsample_bytree=float(cfg["colsample_bytree"]), reg_lambda=float(cfg.get("reg_lambda", 1.0)),
                scale_pos_weight=max(neg / max(pos, 1), 1.0), eval_metric=cfg.get("eval_metric", "logloss"),
                n_jobs=int(cfg.get("n_jobs", -1)), random_state=int(config["project"]["seed"]),
            )
    if model_cfg["isolation_forest"].get("enabled", True) or model_cfg.get(
        "isolation_forest_normal_fit", {}
    ).get("enabled", False):
        cfg = model_cfg["isolation_forest"]
        forest = IsolationForest(
            n_estimators=int(cfg["n_estimators"]), max_samples=cfg.get("max_samples", 256),
            contamination=cfg.get("contamination", "auto"), n_jobs=int(cfg.get("n_jobs", -1)),
            random_state=int(config["project"]["seed"]),
        )
        estimators["isolation_forest"] = AnomalyClassifier(estimator=forest)
        if model_cfg.get("isolation_forest_normal_fit", {}).get("enabled", False):
            normal_forest = IsolationForest(
                n_estimators=int(cfg["n_estimators"]), max_samples=cfg.get("max_samples", 256),
                contamination=cfg.get("contamination", "auto"), n_jobs=int(cfg.get("n_jobs", -1)),
                random_state=int(config["project"]["seed"]) + 1,
            )
            estimators["isolation_forest_normal_fit"] = AnomalyClassifier(estimator=normal_forest)
    if model_cfg["conventional_baseline"].get("enabled", True):
        cfg = model_cfg["conventional_baseline"]
        estimators["conventional_baseline"] = ConventionalBaseline(indicator=cfg["indicator"])
    return estimators


def _scores(estimator: Any, X: pd.DataFrame) -> np.ndarray:
    """Return continuous scores where higher means more likely kick."""
    if hasattr(estimator, "predict_proba"):
        return np.asarray(estimator.predict_proba(X)[:, 1], dtype=float)
    return np.asarray(estimator.decision_function(X), dtype=float)


def _metric_value(metric: str, precision: float, recall: float) -> float:
    """Compute the configured threshold-selection metric."""
    if metric == "f1":
        return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    if metric == "precision":
        return precision
    if metric == "recall":
        return recall
    raise ValueError(f"Unsupported tuning metric: {metric!r}")


def tune_threshold(
    scores: np.ndarray,
    y_true: np.ndarray,
    metric: str,
    minimum_recall: float,
) -> tuple[float, float]:
    """Select a decision threshold using validation data only.

    When the validation scores are perfectly separable, the midpoint between
    the largest negative and smallest positive score is used. This avoids a
    brittle endpoint threshold such as exactly 1.0. Otherwise the PR-curve
    threshold optimising the configured metric is used, subject to the minimum
    validation recall.
    """
    if len(np.unique(y_true)) < 2:
        raise ValueError("Threshold tuning requires both classes in the validation set.")
    y = np.asarray(y_true)
    max_negative = float(scores[y == 0].max())
    min_positive = float(scores[y == 1].min())
    if min_positive > max_negative:
        threshold = (min_positive + max_negative) / 2.0
        labels = (scores >= threshold).astype(int)
        tp = int(((labels == 1) & (y == 1)).sum())
        fp = int(((labels == 1) & (y == 0)).sum())
        fn = int(((labels == 0) & (y == 1)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        if recall < minimum_recall:
            raise ValueError("Separable validation scores do not meet the minimum recall.")
        value = _metric_value(metric, precision, recall)
        logger.info(
            "Tuned separable midpoint threshold=%.4f (validation %s=%.4f, margin=%.4f)",
            threshold, metric, value, min_positive - max_negative,
        )
        return float(threshold), float(value)

    precision_curve, recall_curve, thresholds = precision_recall_curve(y, scores)
    candidates: list[tuple[float, float]] = []
    for index, threshold in enumerate(thresholds):
        r = float(recall_curve[index])
        p = float(precision_curve[index])
        if r < minimum_recall:
            continue
        candidates.append((_metric_value(metric, p, r), float(threshold)))
    if not candidates:
        raise ValueError("No threshold satisfies the configured minimum recall.")
    best_value, best_threshold = max(candidates, key=lambda item: item[0])
    logger.info("Tuned threshold=%.4f (validation %s=%.4f)", best_threshold, metric, best_value)
    return best_threshold, best_value


def _validate_score(estimator: Any, X: pd.DataFrame, y: pd.Series) -> float:
    """Compute validation ROC-AUC for model selection (never uses test data)."""
    try:
        return float(roc_auc_score(y, _scores(estimator, X)))
    except Exception:
        return float("nan")


def _importances(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, seed: int) -> dict[str, float]:
    """Compute permutation importance on validation data for the app/report."""
    try:
        result = permutation_importance(
            pipeline, X, y, scoring="roc_auc", n_repeats=3, random_state=seed, n_jobs=1
        )
        values = np.abs(result.importances_mean)
        if not np.isfinite(values).any() or float(np.max(values)) <= 1e-9:
            # With perfect validation ROC-AUC, permutation can report zero for
            # every feature. Use the estimator's inherent importance instead.
            estimator = pipeline.steps[-1][1]
            if hasattr(estimator, "feature_importances_"):
                values = np.abs(estimator.feature_importances_)
            elif hasattr(estimator, "coef_"):
                values = np.abs(np.ravel(estimator.coef_))
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.warning("Permutation importance failed: %s", exc)
        estimator = pipeline.steps[-1][1]
        if hasattr(estimator, "feature_importances_"):
            values = np.abs(estimator.feature_importances_)
        elif hasattr(estimator, "coef_"):
            values = np.abs(np.ravel(estimator.coef_))
        else:
            return {}
    return {
        column: float(value)
        for column, value in zip(X.columns, values)
    }


def train_models(config: dict[str, Any]) -> TrainingResult:
    """Run the full training workflow and persist a model bundle.

    Steps: load/verify target, engineer features, split, fit candidates on the
    fit portion, tune each decision threshold on validation, refit on all
    training blocks, and save the bundle. The test set is not touched here.
    """
    from .data_loader import load_raw_data

    seed = int(config["project"]["seed"])
    frame = load_raw_data(config=config)
    target = validate_target(frame, config)
    featured = add_engineered_features(frame, config)
    feature_columns = select_feature_columns(featured, config)
    X = featured[feature_columns]
    y = target.labels

    split = split_data(X, y, config)
    X_fit, X_val, y_fit, y_val = make_validation_split(
        split.X_train, split.y_train, split.train_groups, config
    )
    preprocessing_fit = build_preprocessor(feature_columns, config)
    preprocessing_full = build_preprocessor(feature_columns, config)
    estimators = _estimators(config, split.y_train)
    eval_cfg = config["evaluation"]
    min_recall = float(eval_cfg.get("minimum_recall", 0.0))

    pipelines: dict[str, Pipeline] = {}
    thresholds: dict[str, float] = {}
    threshold_details: dict[str, dict[str, float]] = {}
    validation_summary: dict[str, dict[str, float]] = {}
    importances: dict[str, dict[str, float]] = {}

    for name, estimator in estimators.items():
        normal_only = name == "isolation_forest_normal_fit"
        fit_X = X_fit[y_fit == target.negative_label] if normal_only else X_fit
        fit_y = None if normal_only else y_fit
        if name == "conventional_baseline":
            # No fitting is required; the indicator is already in X.
            pipeline = Pipeline([("model", clone(estimator))])
            pipeline.fit(X_fit)
        else:
            pipeline = Pipeline([
                ("preprocess", clone(preprocessing_fit)),
                ("model", clone(estimator)),
            ])
            pipeline.fit(fit_X, fit_y)
        if normal_only:
            logger.info("Fitted %s on %d normal-only training rows.", name, len(fit_X))

        val_scores = _scores(pipeline, X_val)
        threshold, threshold_value = tune_threshold(
            val_scores, y_val.to_numpy(), str(eval_cfg.get("tuning_metric", "f1")), min_recall
        )
        y_val_array = y_val.to_numpy()
        val_labels = (val_scores >= threshold).astype(int)
        val_tp = int(((val_labels == 1) & (y_val_array == 1)).sum())
        val_fp = int(((val_labels == 1) & (y_val_array == 0)).sum())
        val_fn = int(((val_labels == 0) & (y_val_array == 1)).sum())
        val_tn = int(((val_labels == 0) & (y_val_array == 0)).sum())
        validation_margin = float(
            val_scores[y_val_array == 1].min() - val_scores[y_val_array == 0].max()
        )
        validation_summary[name] = {
            "validation_roc_auc": _validate_score(pipeline, X_val, y_val),
            f"validation_{eval_cfg.get('tuning_metric', 'f1')}_at_threshold": threshold_value,
            "validation_precision_at_threshold": (
                val_tp / (val_tp + val_fp) if val_tp + val_fp else 0.0
            ),
            "validation_recall_at_threshold": (
                val_tp / (val_tp + val_fn) if val_tp + val_fn else 0.0
            ),
            "validation_far_at_threshold": (
                val_fp / (val_fp + val_tn) if val_fp + val_tn else 0.0
            ),
            "validation_score_margin": validation_margin,
        }
        pipelines[name] = pipeline
        thresholds[name] = threshold
        threshold_details[name] = {
            "threshold": threshold,
            "validation_score": threshold_value,
            "tuning_metric": str(eval_cfg.get("tuning_metric", "f1")),
        }
        importances[name] = _importances(pipeline, X_val, y_val, seed)

        # Refit on every training block using the validation-selected threshold.
        if name == "conventional_baseline":
            final = Pipeline([("model", clone(estimator))])
            final.fit(split.X_train)
        else:
            final = Pipeline([
                ("preprocess", clone(preprocessing_full)),
                ("model", clone(estimator)),
            ])
            final_X = split.X_train[split.y_train == target.negative_label] if normal_only else split.X_train
            final_y = None if normal_only else split.y_train
            final.fit(final_X, final_y)
        pipelines[name] = final

    anomaly_names = {"isolation_forest", "isolation_forest_normal_fit"}
    supervised = {name: value for name, value in validation_summary.items()
                  if name not in anomaly_names and name != "conventional_baseline"}
    valid = {name: value for name, value in supervised.items() if np.isfinite(value["validation_roc_auc"])}
    metric_key = f"validation_{eval_cfg.get('tuning_metric', 'f1')}_at_threshold"
    best_model = max(
        valid,
        key=lambda name: (
            valid[name]["validation_roc_auc"],
            valid[name].get(metric_key, 0.0),
            valid[name]["validation_recall_at_threshold"],
            -valid[name]["validation_far_at_threshold"],
            valid[name]["validation_score_margin"],
        ),
    ) if valid else (
        "random_forest" if "random_forest" in pipelines else next(iter(pipelines))
    )

    bundle = {
        "version": str(config["project"]["version"]),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "feature_columns": feature_columns,
        "target": {
            "column": target.column,
            "mode": target.mode,
            "rule": target.rule,
            "positive_label": target.positive_label,
            "negative_label": target.negative_label,
            "proxy_confirmation_required": bool(config["target"].get("confirmation_required", True)),
            "onset_index": target.onset_index,
            "onset_depth_ft": target.onset_depth,
            "proxy_disclaimer": (
                "WARNING: labels used for training were experimental proxies derived from "
                "pit gain/loss. Confirm the target definition with a drilling supervisor "
                "before interpreting results."
            ),
        },
        "models": {
            name: {
                "pipeline": pipeline,
                "threshold": thresholds[name],
                "family": (
                    "unsupervised_anomaly" if name in anomaly_names
                    else "conventional_rule" if name == "conventional_baseline"
                    else "supervised"
                ),
            }
            for name, pipeline in pipelines.items()
        },
        "best_model": best_model,
        "feature_bounds": {
            column: {
                "min": float(split.X_train[column].min()),
                "max": float(split.X_train[column].max()),
            }
            for column in feature_columns
        },
        "feature_importance": importances,
        "validation_summary": validation_summary,
        "train_rows": int(len(split.X_train)),
        "test_rows": int(len(split.X_test)),
        "split_indices": {
            "train": split.train_indices.tolist(),
            "test": split.test_indices.tolist(),
        },
        "config": config,
    }

    bundle_path = resolve_path(config["paths"]["bundle_path"], config)
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    import joblib
    joblib.dump(bundle, bundle_path)
    logger.info("Saved model bundle to %s (best model: %s)", bundle_path, best_model)

    return TrainingResult(
        bundle=bundle,
        feature_columns=feature_columns,
        threshold_details=threshold_details,
        validation_summary=validation_summary,
        models_used=list(pipelines),
    )
