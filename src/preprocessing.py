"""Configurable preprocessing and leakage-aware data splitting."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class DataSplit:
    """Container for a train/test split with group identifiers."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    train_groups: np.ndarray
    test_groups: np.ndarray
    train_indices: np.ndarray
    test_indices: np.ndarray


def build_preprocessor(feature_columns: list[str], config: dict[str, Any]) -> ColumnTransformer:
    """Build imputation/scaling that is fitted on training data only.

    Robust scaling is the default because drilling signals can contain genuine
    transient events that should not be erased by mean/variance scaling.
    """
    features_cfg = config.get("features", {})
    strategy = str(features_cfg.get("imputer", "median"))
    scaler_name = str(features_cfg.get("scaler", "robust")).lower()
    if scaler_name == "standard":
        scaler: Any = StandardScaler()
    elif scaler_name == "robust":
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unsupported scaler: {scaler_name!r}")

    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy=strategy, keep_empty_features=True)),
        ("scaler", scaler),
    ])
    transformer = ColumnTransformer(
        transformers=[("numeric", numeric, feature_columns)],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return transformer


def create_blocks(n_rows: int, block_size: int) -> np.ndarray:
    """Assign contiguous rows to blocks so adjacent samples stay together."""
    if block_size < 1:
        raise ValueError("split.block_size must be a positive integer.")
    return np.arange(n_rows) // int(block_size)


def _fold_split(
    X: pd.DataFrame,
    y: pd.Series,
    groups: np.ndarray,
    n_splits: int,
    seed: int,
    purge_blocks: int,
    fold: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return one fold of a stratified group split with optional block purge."""
    if n_splits < 2:
        raise ValueError("At least two splits are required.")
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = list(splitter.split(X, y, groups))
    train_idx, test_idx = folds[fold]
    test_groups = set(groups[test_idx].tolist())
    if purge_blocks:
        forbidden = set()
        for block in test_groups:
            forbidden.update(range(int(block) - purge_blocks, int(block) + purge_blocks + 1))
        train_idx = np.array([i for i in train_idx if int(groups[i]) not in forbidden], dtype=int)
    return train_idx, test_idx


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    config: dict[str, Any],
    groups: np.ndarray | None = None,
) -> DataSplit:
    """Create a leakage-aware train/test split.

    Contiguous blocks are kept intact and stratified by class, preventing
    near-duplicate neighbouring measurements from appearing on both sides.
    A strict forward-chaining split is impossible here because the single
    simulation contains one kick near the end; a purely chronological split
    would leave the training set without positive examples.
    """
    split_cfg = config["split"]
    if split_cfg.get("strategy") != "stratified_group_blocked":
        raise ValueError(f"Unsupported split strategy: {split_cfg.get('strategy')!r}")
    seed = int(config["project"]["seed"])
    block_ids = create_blocks(len(X), int(split_cfg["block_size"])) if groups is None else np.asarray(groups)
    n_splits = max(2, int(round(1.0 / float(split_cfg["test_size"]))))
    train_idx, test_idx = _fold_split(
        X, y, block_ids, n_splits, seed, int(split_cfg.get("purge_blocks", 0))
    )
    for name, indices in (("train", train_idx), ("test", test_idx)):
        classes = set(np.unique(y.iloc[indices]))
        if len(classes) < 2:
            raise ValueError(f"{name} split contains only one class; adjust split configuration.")

    logger.info(
        "Split: train=%d rows, test=%d rows, positive fractions %.3f/%.3f",
        len(train_idx), len(test_idx), y.iloc[train_idx].mean(), y.iloc[test_idx].mean(),
    )
    return DataSplit(
        X_train=X.iloc[train_idx].reset_index(drop=True),
        X_test=X.iloc[test_idx].reset_index(drop=True),
        y_train=y.iloc[train_idx].reset_index(drop=True),
        y_test=y.iloc[test_idx].reset_index(drop=True),
        train_groups=block_ids[train_idx],
        test_groups=block_ids[test_idx],
        train_indices=train_idx,
        test_indices=test_idx,
    )


def make_validation_split(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups_train: np.ndarray,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Carve a validation subset out of the training data for threshold tuning."""
    size = float(config["split"].get("validation_size", 0.2))
    n_splits = max(2, int(round(1.0 / size)))
    seed = int(config["project"]["seed"])
    fit_idx, val_idx = _fold_split(X_train, y_train, groups_train, n_splits, seed, 0)
    if len(set(np.unique(y_train.iloc[val_idx]))) < 2:
        raise ValueError("Validation split contains only one class; adjust validation_size.")
    logger.info("Validation split: fit=%d, validation=%d", len(fit_idx), len(val_idx))
    return (
        X_train.iloc[fit_idx].reset_index(drop=True),
        X_train.iloc[val_idx].reset_index(drop=True),
        y_train.iloc[fit_idx].reset_index(drop=True),
        y_train.iloc[val_idx].reset_index(drop=True),
    )
