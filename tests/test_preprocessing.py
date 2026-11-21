"""Tests for feature engineering, preprocessing and splitting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_loader import load_config, load_raw_data
from src.feature_engineering import add_engineered_features, select_feature_columns, verify_flow_conventions
from src.preprocessing import build_preprocessor, create_blocks, make_validation_split, split_data
from src.data_validation import validate_target


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture(scope="module")
def prepared(config):
    frame = load_raw_data(config=config)
    featured = add_engineered_features(frame, config)
    target = validate_target(frame, config)
    columns = select_feature_columns(featured, config)
    return featured, target.labels, columns


def test_flow_convention_verified(prepared, config):
    featured, _, _ = prepared
    result = verify_flow_conventions(featured)
    assert result["verified"] is True
    assert result["max_abs_residual"] < 1e-4


def test_flow_imbalance_sign(prepared):
    featured, _, _ = prepared
    expected = featured["FOut"] - featured["FIn"]
    np.testing.assert_allclose(featured["Flow_Imbalance"], expected)
    assert featured["Flow_Imbalance"].iloc[2000] > 0  # during the kick


def test_excluded_columns_never_selected(prepared, config):
    featured, _, columns = prepared
    forbidden = set(config["features"]["excluded_columns"])
    forbidden.add(config["target"]["column"])
    assert not forbidden.intersection(columns)
    assert "Flow_Imbalance" in columns


def test_split_keeps_blocks_intact_and_both_classes(prepared, config):
    featured, labels, columns = prepared
    split = split_data(featured[columns], labels, config)
    assert len(split.X_train) > 0 and len(split.X_test) > 0
    assert set(split.y_train.unique()) == {0, 1} and set(split.y_test.unique()) == {0, 1}
    assert not set(split.train_groups).intersection(split.test_groups)
    assert create_blocks(100, 25)[-1] == 3


def test_validation_split_has_both_classes(prepared, config):
    featured, labels, columns = prepared
    split = split_data(featured[columns], labels, config)
    fit, val, y_fit, y_val = make_validation_split(
        split.X_train, split.y_train, split.train_groups, config
    )
    assert len(fit) > 0 and len(val) > 0
    assert set(y_val.unique()) == {0, 1}


def test_preprocessor_fits_on_training_columns_only(prepared, config):
    featured, labels, columns = prepared
    split = split_data(featured[columns], labels, config)
    transformer = build_preprocessor(columns, config)
    transformed = transformer.fit_transform(split.X_train)
    assert transformed.shape[0] == len(split.X_train)
    assert transformed.shape[1] == len(columns)
    assert np.isfinite(transformed).all()


def test_invalid_scaler_raised(prepared, config):
    _, _, columns = prepared
    bad = {**config, "features": {**config["features"], "scaler": "unknown"}}
    with pytest.raises(ValueError, match="Unsupported scaler"):
        build_preprocessor(columns, bad)
