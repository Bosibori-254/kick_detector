"""Tests for dataset loading, validation and target verification."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_loader import load_config, load_raw_data
from src.data_validation import (
    assess_temporal_support,
    audit_dataset,
    detect_constant_columns,
    validate_target,
)


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture(scope="module")
def frame(config):
    return load_raw_data(config=config)


def test_dataset_loads_with_expected_shape(frame):
    assert frame.shape[0] > 1000
    assert "ActiveGL" in frame.columns
    assert "FOut" in frame.columns


def test_target_is_experimental_proxy_with_documented_onset(frame, config):
    info = validate_target(frame, config)
    assert info.mode == "proxy_from_threshold"
    assert 0 < info.positive_fraction < 1
    assert info.onset_index is not None
    assert info.onset_depth == pytest.approx(config["target"]["documented_kick_depth_ft"], abs=2.0)
    assert any("proxy" in warning.lower() for warning in info.warnings)


def test_explicit_target_mode_rejects_continuous_values(frame, config):
    bad_config = {**config, "target": {**config["target"], "mode": "explicit_column"}}
    with pytest.raises(ValueError, match="not an explicit"):
        validate_target(frame, bad_config)


def test_missing_target_column_raises(frame, config):
    bad_config = {**config, "target": {**config["target"], "column": "NotAColumn"}}
    with pytest.raises(ValueError, match="absent"):
        validate_target(frame, bad_config)


def test_constant_and_duplicate_detection(frame):
    constants = detect_constant_columns(frame)
    assert "CPress" in constants["constant"]
    assert "MPS1" in constants["constant"]


def test_temporal_support_reports_missing_timestamps(frame):
    temporal = assess_temporal_support(frame)
    assert temporal["timestamp_columns"] == []
    assert temporal["supports_validated_early_warning"] is False
    assert temporal["sequence_order_plausible"] is True


def test_audit_contains_leakage_findings(frame, config):
    audit = audit_dataset(frame, config)
    columns = {item["column"] for item in audit["leakage_findings"]}
    assert {"ActiveGL", "ATVolume", "MRFlow"}.issubset(columns)
    assert audit["known_identities"]["MRFlow_equals_FIn_minus_FOut"]["holds"]


def test_proxy_threshold_change_is_documented(frame, config):
    low = validate_target(frame, config).n_positive
    raised = {**config, "target": {**config["target"], "threshold": 5.0}}
    high = validate_target(frame, raised).n_positive
    assert high < low
