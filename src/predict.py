"""Single-record prediction interface used by both the CLI and Streamlit app."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data_loader import load_config, resolve_path
from .train import _scores

logger = logging.getLogger(__name__)

DERIVED_PREFIXES = ("Flow_Imbalance", "roll")


def load_bundle(path: str | Path | None = None, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Load a saved training bundle, failing clearly when it is absent."""
    import joblib

    if config is None:
        config = load_config()
    bundle_path = Path(path) if path is not None else resolve_path(config["paths"]["bundle_path"], config)
    if not bundle_path.is_file():
        raise FileNotFoundError(
            f"Trained model not found: {bundle_path}. Run `python -m src.cli train` first."
        )
    bundle = joblib.load(bundle_path)
    if not isinstance(bundle, dict) or "models" not in bundle:
        raise ValueError(f"Unrecognised model bundle: {bundle_path}")
    return bundle


def required_raw_features(bundle: dict[str, Any]) -> list[str]:
    """Return the user-supplied inputs needed before derived features are built."""
    return [
        column for column in bundle["feature_columns"]
        if not column.startswith(DERIVED_PREFIXES)
    ]


def validate_input(
    values: dict[str, Any],
    bundle: dict[str, Any],
    strict_bounds: bool = True,
) -> dict[str, float]:
    """Validate numeric ranges and presence of all required inputs.

    Args:
        values: Mapping of feature name to value.
        bundle: Training bundle containing feature bounds.
        strict_bounds: Reject values far outside the training envelope.

    Returns:
        Cleaned mapping of floats.

    Raises:
        ValueError: With a clear message for missing, non-numeric, non-finite,
            or implausible values.
    """
    required = required_raw_features(bundle)
    missing = [column for column in required if column not in values or values[column] is None]
    if missing:
        raise ValueError(f"Missing required drilling parameters: {', '.join(missing)}")

    cleaned: dict[str, float] = {}
    bounds = bundle.get("feature_bounds", {})
    for column in required:
        raw = values[column]
        if isinstance(raw, bool):
            raise ValueError(f"'{column}' must be numeric, not a boolean.")
        try:
            number = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"'{column}' must be numeric; got {raw!r}.") from exc
        if not np.isfinite(number):
            raise ValueError(f"'{column}' must be finite; got {raw!r}.")
        if strict_bounds and column in bounds:
            lower = float(bounds[column]["min"])
            upper = float(bounds[column]["max"])
            padding = max(0.5 * (upper - lower), abs(upper) * 0.05, 1e-6)
            if number < lower - padding or number > upper + padding:
                raise ValueError(
                    f"'{column}'={number:.6g} is outside the plausible training range "
                    f"[{lower:.6g}, {upper:.6g}] (with 50% tolerance)."
                )
        cleaned[column] = number
    return cleaned


def prepare_feature_row(values: dict[str, float], bundle: dict[str, Any], config: dict[str, Any]) -> pd.DataFrame:
    """Rebuild engineered features from validated raw inputs."""
    row = dict(values)
    feature_columns = bundle["feature_columns"]
    engineered = config.get("features", {}).get("engineered", {})
    if engineered.get("flow_imbalance", False) and "Flow_Imbalance" in feature_columns:
        if "FIn" not in row or "FOut" not in row:
            raise ValueError("FIn and FOut are required to compute Flow_Imbalance.")
        row["Flow_Imbalance"] = row["FOut"] - row["FIn"]
        if engineered.get("flow_imbalance_ratio", False) and "Flow_Imbalance_Ratio" in feature_columns:
            if row["FIn"] == 0:
                raise ValueError("FIn must be non-zero to compute relative flow imbalance.")
            row["Flow_Imbalance_Ratio"] = row["Flow_Imbalance"] / row["FIn"]
    missing = [column for column in feature_columns if column not in row]
    if missing:
        raise ValueError(f"Cannot construct model input; missing derived features: {missing}")
    return pd.DataFrame([[row[column] for column in feature_columns]], columns=feature_columns)


def predict_one(
    values: dict[str, Any],
    bundle: dict[str, Any],
    config: dict[str, Any] | None = None,
    model_name: str | None = None,
    strict_bounds: bool = True,
) -> dict[str, Any]:
    """Return a structured prediction for one set of drilling measurements."""
    if config is None:
        config = load_config()
    cleaned = validate_input(values, bundle, strict_bounds=strict_bounds)
    X = prepare_feature_row(cleaned, bundle, config)
    name = model_name or bundle["best_model"]
    if name not in bundle["models"]:
        raise ValueError(f"Unknown model '{name}'. Available: {', '.join(bundle['models'])}")
    entry = bundle["models"][name]
    score = float(_scores(entry["pipeline"], X)[0])
    threshold = float(entry["threshold"])
    positive = int(bundle["target"]["positive_label"])
    is_kick = int(score >= threshold)
    band = float(config["evaluation"].get("uncertainty_band", 0.0))
    uncertain = bool(abs(score - threshold) <= band)
    importance = bundle.get("feature_importance", {}).get(name, {})
    contributors = sorted(importance.items(), key=lambda item: item[1], reverse=True)[:3]
    if uncertain:
        warning = (
            "UNCERTAIN: the model score is close to its decision threshold. Check pit volume, "
            "flow-out/flow-in and pressure trends before drawing a conclusion."
        )
    elif is_kick == positive:
        warning = (
            "POTENTIAL KICK: compare with pit volume, flow-out/flow-in and standpipe pressure "
            "trends, and follow approved well-control procedures."
        )
    else:
        warning = "No kick indicated by this experimental model; continue standard monitoring."
    label_key = (
        "uncertainty_class_name" if uncertain
        else "positive_class_name" if is_kick == positive
        else "negative_class_name"
    )
    return {
        "model": name,
        "predicted_class": int(is_kick),
        "predicted_label": config["evaluation"][label_key],
        "uncertain": uncertain,
        "score": score,
        "decision_threshold": threshold,
        "model_family": entry["family"],
        "warning": warning,
        "top_global_contributors": [{"feature": key, "importance": float(value)} for key, value in contributors],
        "disclaimer": (
            "Experimental research prediction only. Not validated for field use and not a "
            "substitute for qualified personnel or approved well-control procedures."
        ),
    }


def parse_values_json(payload: str) -> dict[str, Any]:
    """Parse a JSON object of input values with a clear error message."""
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError(f"--values must be valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("--values must be a JSON object such as {\"FIn\": 0.95, ...}")
    return parsed
