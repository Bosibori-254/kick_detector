"""Domain-informed feature engineering for drilling kick indicators.

Only information available at the current observation is used. No future
rows, target values, or target-derived columns enter the feature set.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def verify_flow_conventions(frame: pd.DataFrame) -> dict[str, Any]:
    """Verify the sign convention required for ``Flow_Imbalance = FOut - FIn``.

    The DataDRILL documentation describes FIn as fluid entering the drill pipe
    and FOut as fluid returning from it. The apparent return-flow variable
    MRFlow is checked against ``FIn - FOut``. If that identity holds, a positive
    ``FOut - FIn`` means more fluid leaves the well than enters it, consistent
    with an influx of formation fluid.
    """
    required = {"FIn", "FOut", "MRFlow"}
    if not required.issubset(frame.columns):
        return {
            "verified": False,
            "reason": f"Missing required columns: {sorted(required - set(frame.columns))}",
        }
    residual = (frame["MRFlow"] - (frame["FIn"] - frame["FOut"])).abs()
    max_residual = float(residual.max())
    verified = bool(max_residual < 1e-4)
    return {
        "verified": verified,
        "max_abs_residual": max_residual,
        "interpretation": (
            "MRFlow = FIn - FOut; therefore Flow_Imbalance = FOut - FIn = -MRFlow. "
            "Positive values mean return flow exceeds input flow (pit gain)."
            if verified
            else "MRFlow identity not confirmed; Flow_Imbalance sign must be reviewed manually."
        ),
    }


def add_engineered_features(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Add verified, causally-safe engineered features.

    Args:
        frame: Input dataframe (not modified).
        config: Project configuration with ``features.engineered`` settings.

    Returns:
        Copy of the dataframe with new columns appended.

    Raises:
        ValueError: If flow imbalance is requested but its sign convention
            cannot be verified from the available columns.
    """
    output = frame.copy()
    engineered = config.get("features", {}).get("engineered", {})

    if engineered.get("flow_imbalance", False):
        convention = verify_flow_conventions(output)
        if not convention["verified"]:
            raise ValueError(
                "Cannot create Flow_Imbalance: " + convention.get("reason", convention["interpretation"])
            )
        output["Flow_Imbalance"] = output["FOut"] - output["FIn"]
        if engineered.get("flow_imbalance_ratio", False):
            denominator = output["FIn"].replace(0, np.nan)
            output["Flow_Imbalance_Ratio"] = (output["Flow_Imbalance"] / denominator).fillna(0.0)
        logger.info("Added Flow_Imbalance (sign convention verified).")

    rolling_cfg = config.get("features", {}).get("rolling", {})
    if rolling_cfg.get("enabled", False):
        window = int(rolling_cfg.get("window", 20))
        if window < 2:
            raise ValueError("features.rolling.window must be at least 2.")
        for column in rolling_cfg.get("columns", []):
            if column not in output.columns:
                raise ValueError(f"Rolling feature requested for missing column '{column}'.")
            # pandas rolling is causal: row t uses rows <= t only.
            output[f"roll{window}_mean_{column}"] = (
                output[column].rolling(window=window, min_periods=window).mean()
            )
            output[f"roll{window}_std_{column}"] = (
                output[column].rolling(window=window, min_periods=window).std()
            )
        logger.info("Added causal rolling features with window=%d.", window)

    return output


def select_feature_columns(frame: pd.DataFrame, config: dict[str, Any]) -> list[str]:
    """Select numeric model inputs after applying configured exclusions.

    Extra safety rules remove non-numeric columns, constant columns and any
    column whose name contains the configured target name.
    """
    target_name = str(config["target"]["column"])
    excluded = set(config.get("features", {}).get("excluded_columns", []))
    candidates = []
    for column in frame.columns:
        if column in excluded or target_name.lower() in column.lower():
            continue
        if not pd.api.types.is_numeric_dtype(frame[column]):
            logger.warning("Excluding non-numeric column '%s'.", column)
            continue
        if frame[column].nunique(dropna=False) <= 1:
            logger.info("Excluding constant column '%s'.", column)
            continue
        candidates.append(column)
    if not candidates:
        raise ValueError("No feature columns remain after exclusions; check configs/config.yaml.")
    logger.info("Selected %d model features: %s", len(candidates), candidates)
    return candidates
