"""Tests for input validation and prediction output."""

from __future__ import annotations

import json

import pytest

from src.data_loader import load_config, resolve_path
from src.predict import (
    load_bundle,
    parse_values_json,
    predict_one,
    required_raw_features,
    validate_input,
)


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture(scope="module")
def bundle(config):
    if not resolve_path(config["paths"]["bundle_path"], config).is_file():
        pytest.skip("Run `python -m src.cli train` before prediction tests.")
    return load_bundle(config=config)


@pytest.fixture(scope="module")
def valid_values(bundle):
    bounds = bundle["feature_bounds"]
    return {
        feature: (float(bounds[feature]["min"]) + float(bounds[feature]["max"])) / 2.0
        for feature in required_raw_features(bundle)
    }


def test_parse_values_json_rejects_invalid_payload():
    with pytest.raises(ValueError, match="valid JSON"):
        parse_values_json("{not json}")
    with pytest.raises(ValueError, match="JSON object"):
        parse_values_json("[1, 2, 3]")


def test_missing_and_invalid_inputs_rejected(bundle, valid_values):
    incomplete = dict(valid_values)
    incomplete.pop(next(iter(incomplete)))
    with pytest.raises(ValueError, match="Missing required"):
        validate_input(incomplete, bundle)

    invalid = dict(valid_values)
    invalid["FIn"] = "not-a-number"
    with pytest.raises(ValueError, match="numeric"):
        validate_input(invalid, bundle)

    non_finite = dict(valid_values)
    non_finite["FOut"] = float("inf")
    with pytest.raises(ValueError, match="finite"):
        validate_input(non_finite, bundle)


def test_out_of_range_inputs_rejected(bundle, valid_values):
    out_of_range = dict(valid_values)
    out_of_range["FIn"] = 1e12
    with pytest.raises(ValueError, match="plausible training range"):
        validate_input(out_of_range, bundle)


def test_prediction_output_structure(bundle, config, valid_values):
    result = predict_one(valid_values, bundle, config)
    assert result["predicted_class"] in {0, 1}
    assert result["predicted_label"] in {
        config["evaluation"]["negative_class_name"],
        config["evaluation"]["positive_class_name"],
        config["evaluation"]["uncertainty_class_name"],
    }
    assert isinstance(result["score"], float)
    assert "well-control" in result["disclaimer"].lower()
    assert len(result["top_global_contributors"]) <= 3


def test_unknown_model_rejected(bundle, config, valid_values):
    with pytest.raises(ValueError, match="Unknown model"):
        predict_one(valid_values, bundle, config, model_name="not_a_model")
