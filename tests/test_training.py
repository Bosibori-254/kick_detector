"""Smoke tests for the saved training artefact."""

from __future__ import annotations

import pytest

from src.data_loader import load_config, resolve_path
from src.predict import load_bundle


def test_saved_bundle_contains_required_models():
    config = load_config()
    if not resolve_path(config["paths"]["bundle_path"], config).is_file():
        pytest.skip("Run `python -m src.cli train` first.")
    bundle = load_bundle(config=config)
    families = {entry["family"] for entry in bundle["models"].values()}
    assert "supervised" in families
    assert "unsupervised_anomaly" in families
    assert "conventional_rule" in families
    assert len([e for e in bundle["models"].values() if e["family"] == "supervised"]) >= 3
    assert bundle["best_model"] in bundle["models"]
    assert "ActiveGL" not in bundle["feature_columns"]
    assert "ATVolume" not in bundle["feature_columns"]
