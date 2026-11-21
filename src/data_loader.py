"""Configuration loading and raw dataset access.

The raw CSV is treated as read-only evidence. Every function here either reads
it or reports on it; none of them modify the source file.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the YAML configuration.

    Args:
        path: Optional path to a config file. Defaults to ``configs/config.yaml``.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        ValueError: If the YAML is empty or is not a mapping.
    """
    config_path = Path(path) if path is not None else DEFAULT_CONFIG
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {config_path}")
    logger.info("Loaded configuration from %s", config_path)
    return config


def resolve_path(relative_path: str | Path, config: dict[str, Any] | None = None) -> Path:
    """Resolve a project-relative path against the project root.

    Absolute paths are returned unchanged. Relative paths are resolved from the
    project root so commands work from any working directory.
    """
    path = Path(relative_path)
    if path.is_absolute():
        return path
    root = PROJECT_ROOT
    if config is not None and "project" in config and "root" in config["project"]:
        root = Path(config["project"]["root"])
    return (root / path).resolve()


def get_raw_path(config: dict[str, Any] | None = None) -> Path:
    """Return the configured raw dataset path."""
    if config is None:
        config = load_config()
    return resolve_path(config["data"]["raw_path"], config)


def load_raw_data(path: str | Path | None = None, config: dict[str, Any] | None = None) -> pd.DataFrame:
    """Load the raw drilling dataset without modifying it.

    Args:
        path: Optional explicit CSV path. Otherwise the configured path is used.
        config: Optional configuration dictionary.

    Returns:
        Raw dataframe with the original columns and row order.

    Raises:
        FileNotFoundError: If the CSV is absent, with placement instructions.
        ValueError: If the file cannot be parsed as a non-empty CSV.
    """
    if config is None:
        config = load_config()
    csv_path = Path(path) if path is not None else get_raw_path(config)
    if not csv_path.is_absolute():
        csv_path = resolve_path(csv_path, config)

    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {csv_path}\n"
            "Download Kick_Detection.csv from https://zenodo.org/records/12759014 "
            f"and place it at {config['data']['raw_path']}."
        )

    try:
        frame = pd.read_csv(csv_path, low_memory=False)
    except Exception as exc:  # pragma: no cover - defensive parsing guard
        raise ValueError(f"Could not parse CSV file {csv_path}: {exc}") from exc

    if frame.empty or frame.columns.empty:
        raise ValueError(f"Dataset is empty: {csv_path}")

    logger.info("Loaded %d rows x %d columns from %s", *frame.shape, csv_path)
    return frame


def file_md5(path: str | Path) -> str:
    """Return the MD5 checksum of a file, useful for dataset provenance checks."""
    file_path = Path(path)
    digest = hashlib.md5()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
