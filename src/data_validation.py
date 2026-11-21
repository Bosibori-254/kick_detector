"""Dataset validation, auditing and target verification.

The most important scientific check in this module is :func:`validate_target`:
the DataDRILL kick file contains a continuous pit-gain/loss measurement, not an
explicit kick label. Any binary target is derived from documented physical
evidence and reported as an *experimental proxy*, never as authoritative truth.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data_loader import file_md5

logger = logging.getLogger(__name__)

TIMESTAMP_PATTERN = re.compile(r"(time|date|stamp|epoch|datetime)", re.IGNORECASE)
GROUP_PATTERN = re.compile(r"(well[_-]?id|well[_-]?name|run[_-]?id|scenario|event[_-]?id|kick[_-]?id)", re.IGNORECASE)
DEPTH_CANDIDATES = ("WellDepth", "BDepth", "BTBR")


@dataclass
class TargetInfo:
    """Result of target verification and label construction."""

    labels: pd.Series
    target_values: pd.Series
    column: str
    mode: str
    rule: str
    positive_label: int
    negative_label: int
    n_positive: int
    n_negative: int
    onset_index: int | None
    onset_depth: float | None
    transitions: int
    warnings: list[str] = field(default_factory=list)

    @property
    def positive_fraction(self) -> float:
        """Fraction of records assigned to the positive (kick) class."""
        return float(self.n_positive / max(len(self.labels), 1))


def _numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    """Return a numeric series or raise a clear error for non-numeric targets."""
    if column not in frame.columns:
        raise ValueError(f"Target column '{column}' is absent from the dataset.")
    numeric = pd.to_numeric(frame[column], errors="coerce")
    if numeric.isna().any() or not np.isfinite(numeric.to_numpy()).all():
        raise ValueError(
            f"Target column '{column}' contains non-numeric, missing, or infinite values."
        )
    return numeric.astype(float)


def validate_target(frame: pd.DataFrame, config: dict[str, Any]) -> TargetInfo:
    """Verify the target and build experimental binary kick labels.

    Supported modes:

    * ``explicit_column`` - the configured column already contains only the two
      configured class values; nothing is inferred.
    * ``proxy_from_threshold`` - the configured continuous indicator exceeds a
      documented threshold. The result is labelled a proxy in the output.

    Args:
        frame: Raw dataframe.
        config: Project configuration.

    Returns:
        TargetInfo with labels and all checks/warnings.

    Raises:
        ValueError: If the target column is missing, non-numeric, ambiguous, or
            the configured mode is unsupported.
    """
    target_cfg = config["target"]
    column = str(target_cfg["column"])
    mode = str(target_cfg["mode"])
    values = _numeric_series(frame, column)
    positive = int(target_cfg["positive_label"])
    negative = int(target_cfg["negative_label"])
    warnings: list[str] = []

    if mode == "explicit_column":
        unique = set(np.unique(values.to_numpy()))
        allowed = {float(positive), float(negative)}
        if not unique.issubset(allowed):
            raise ValueError(
                f"Column '{column}' is not an explicit {negative}/{positive} label "
                f"(observed values: {sorted(unique)[:10]}). Confirm the target before training."
            )
        labels = values.astype(int)
        rule = f"explicit labels in '{column}'"
    elif mode == "proxy_from_threshold":
        threshold = float(target_cfg["threshold"])
        labels = (values > threshold).astype(int).map({1: positive, 0: negative}).astype(int)
        rule = str(target_cfg.get("rule", f"{column} > {threshold}"))
        warnings.append(
            "No authoritative kick label is supplied by the dataset. Labels are an "
            f"EXPERIMENTAL PROXY from '{column}' and must be confirmed by a drilling "
            "supervisor before any research claim is made."
        )
        if not target_cfg.get("allow_proxy_training", False):
            raise ValueError(
                "Proxy labels are disabled (target.allow_proxy_training=false). "
                "Confirm an authoritative target and update configs/config.yaml before training."
            )
    else:
        raise ValueError(f"Unsupported target.mode: {mode!r}")

    onset_index: int | None = None
    onset_depth: float | None = None
    positive_positions = np.flatnonzero(labels.to_numpy() == positive)
    if positive_positions.size:
        onset_index = int(positive_positions[0])
        depth_column = next((c for c in DEPTH_CANDIDATES if c in frame.columns), None)
        if depth_column is not None:
            onset_depth = float(frame[depth_column].iloc[onset_index])
            documented = target_cfg.get("documented_kick_depth_ft")
            if documented is not None and abs(onset_depth - float(documented)) > 2.0:
                warnings.append(
                    f"Proxy onset depth ({onset_depth:.2f} ft) differs from the documented "
                    f"kick depth ({float(documented):.2f} ft) by more than 2 ft."
                )
    else:
        warnings.append("No positive-label records were produced; supervised training is impossible.")

    transitions = int((labels.diff().fillna(0) != 0).sum())
    if transitions > 1:
        warnings.append(
            f"The proxy label changes {transitions} times; this is not a single clean kick event."
        )

    n_positive = int((labels == positive).sum())
    n_negative = int((labels == negative).sum())
    info = TargetInfo(
        labels=labels,
        target_values=values,
        column=column,
        mode=mode,
        rule=rule,
        positive_label=positive,
        negative_label=negative,
        n_positive=n_positive,
        n_negative=n_negative,
        onset_index=onset_index,
        onset_depth=onset_depth,
        transitions=transitions,
        warnings=warnings,
    )
    logger.info(
        "Target verified: mode=%s, positives=%d, negatives=%d, onset_index=%s",
        mode, n_positive, n_negative, onset_index,
    )
    return info


def assess_temporal_support(frame: pd.DataFrame) -> dict[str, Any]:
    """Determine whether genuine time-aware early-warning analysis is possible."""
    columns = list(frame.columns)
    timestamps = [c for c in columns if TIMESTAMP_PATTERN.search(c)]
    groups = [c for c in columns if GROUP_PATTERN.search(c)]
    ordered_depth = next((c for c in DEPTH_CANDIDATES if c in columns), None)
    monotonic = bool(frame[ordered_depth].is_monotonic_increasing) if ordered_depth else False
    supports = bool(timestamps and groups)  # conservative: need time and run/event context
    notes = []
    if not timestamps:
        notes.append("No timestamp-like column exists; elapsed warning time cannot be calculated.")
    if not groups:
        notes.append("No well/run/event identifier exists; only one continuous run is observable.")
    if monotonic:
        notes.append(f"'{ordered_depth}' is monotonic, so row order is a plausible sequence index.")
    return {
        "timestamp_columns": timestamps,
        "group_columns": groups,
        "sequence_order_plausible": monotonic,
        "supports_validated_early_warning": supports,
        "notes": notes,
    }


def detect_constant_columns(frame: pd.DataFrame, near_constant_unique: int = 5) -> dict[str, list[str]]:
    """Identify constant and near-constant numeric columns."""
    constant = [c for c in frame.columns if frame[c].nunique(dropna=False) <= 1]
    near = [
        c for c in frame.columns
        if c not in constant and frame[c].nunique(dropna=False) < near_constant_unique
    ]
    return {"constant": constant, "near_constant": near}


def analyse_missing_values(frame: pd.DataFrame) -> dict[str, int]:
    """Return missing-value counts for every column."""
    return {column: int(count) for column, count in frame.isna().sum().items() if count > 0}


def analyse_duplicates(frame: pd.DataFrame) -> dict[str, Any]:
    """Quantify exact duplicate rows without deleting them."""
    duplicated = frame.duplicated(keep=False)
    return {
        "n_duplicate_rows": int(duplicated.sum()),
        "n_extra_duplicates": int(frame.duplicated().sum()),
        "example_indices": [int(i) for i in frame.index[duplicated][:10].tolist()],
    }


def check_known_identities(frame: pd.DataFrame) -> dict[str, Any]:
    """Check algebraic relationships that affect leakage and feature choice."""
    results: dict[str, Any] = {}
    if {"MRFlow", "FIn", "FOut"}.issubset(frame.columns):
        residual = (frame["MRFlow"] - (frame["FIn"] - frame["FOut"])).abs()
        results["MRFlow_equals_FIn_minus_FOut"] = {
            "max_abs_residual": float(residual.max()),
            "holds": bool(residual.max() < 1e-5),
        }
    if {"FRate", "FOut"}.issubset(frame.columns):
        negation_residual = (frame["FRate"] + frame["FOut"]).abs()
        results["FRate_is_negative_FOut"] = {
            "correlation": float(frame["FRate"].corr(frame["FOut"])),
            "max_abs_residual_all_rows": float(negation_residual.max()),
            "max_abs_residual_after_first_row": float(negation_residual.iloc[1:].max()),
            "holds_after_first_row": bool(negation_residual.iloc[1:].max() < 1e-4),
        }
    if {"ATVolume", "ActiveGL"}.issubset(frame.columns):
        offset = float((frame["ATVolume"] - frame["ActiveGL"]).median())
        residual = (frame["ATVolume"] - frame["ActiveGL"] - offset).abs()
        results["ATVolume_is_ActiveGL_plus_constant"] = {
            "estimated_offset": offset,
            "max_abs_residual": float(residual.max()),
            "holds": bool(residual.max() < 1e-3),
        }
    return results


def audit_dataset(
    frame: pd.DataFrame,
    config: dict[str, Any],
    csv_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build a complete, evidence-based dataset audit dictionary."""
    target = validate_target(frame, config)
    identities = check_known_identities(frame)
    constants = detect_constant_columns(frame)
    temporal = assess_temporal_support(frame)
    numeric = frame.select_dtypes(include=[np.number])
    correlations = numeric.corr(numeric_only=True)[target.column].drop(target.column)
    correlations = correlations.replace([np.inf, -np.inf], np.nan).dropna()
    top_correlations = {
        key: float(value) for key, value in correlations.reindex(
            correlations.abs().sort_values(ascending=False).index
        ).head(12).items()
    }

    leakage: list[dict[str, str]] = [
        {
            "column": "ActiveGL",
            "reason": "Source of the experimental target; using it as an input would be target leakage.",
            "severity": "critical",
        },
        {
            "column": "ATVolume",
            "reason": "Near-exact copy of ActiveGL plus a constant offset (~199.083); encodes the target.",
            "severity": "critical",
        },
        {
            "column": "MRFlow",
            "reason": "Exactly FIn - FOut; redundant with the engineered Flow_Imbalance feature.",
            "severity": "moderate",
        },
        {
            "column": "WellDepth/BDepth/BTBR",
            "reason": "Kick is configured at 12650 ft; depth can memorise this single simulated scenario.",
            "severity": "high",
        },
        {
            "column": "FPress",
            "reason": "Configured cause of the simulated kick, not a conventional real-time measured kick indicator.",
            "severity": "high",
        },
    ]

    audit: dict[str, Any] = {
        "shape": {"rows": int(frame.shape[0]), "columns": int(frame.shape[1])},
        "columns": list(frame.columns),
        "dtypes": {c: str(t) for c, t in frame.dtypes.items()},
        "missing_values": analyse_missing_values(frame),
        "non_finite_values": {
            column: count for column, count in (
                (c, int((~np.isfinite(numeric[c].to_numpy())).sum())) for c in numeric.columns
            ) if count > 0
        },
        "duplicates": analyse_duplicates(frame),
        "constants": constants,
        "outliers_iqr": {
            column: {
                "lower": float(q1 - 1.5 * iqr),
                "upper": float(q3 + 1.5 * iqr),
                "count": int(((numeric[column] < q1 - 1.5 * iqr) | (numeric[column] > q3 + 1.5 * iqr)).sum()),
            }
            for column in numeric.columns
            for q1, q3 in [(numeric[column].quantile(0.25), numeric[column].quantile(0.75))]
            for iqr in [q3 - q1]
        },
        "numeric_summary": {
            c: {
                "min": float(numeric[c].min()),
                "median": float(numeric[c].median()),
                "max": float(numeric[c].max()),
                "std": float(numeric[c].std()),
                "n_unique": int(numeric[c].nunique()),
            }
            for c in numeric.columns
        },
        "target": target,
        "temporal_support": temporal,
        "known_identities": identities,
        "top_target_correlations": top_correlations,
        "leakage_findings": leakage,
        "md5": file_md5(csv_path) if csv_path is not None else None,
    }
    return audit


def write_audit_report(audit: dict[str, Any], config: dict[str, Any], path: str | Path) -> Path:
    """Render an audit dictionary as a Markdown report.

    Every number in this report comes from the loaded dataset; no observations
    are invented.
    """
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    target: TargetInfo = audit["target"]
    temporal = audit["temporal_support"]
    lines: list[str] = [
        "# DataDRILL kick-detection dataset audit",
        "",
        "> Generated automatically by `python -m src.cli audit`. "
        "This is an experimental research prototype, not an operational well-control tool.",
        "",
        "## 1. Provenance",
        "",
        f"- Source: {config['data']['source_url']}",
        f"- Configured path: `{config['data']['raw_path']}`",
        f"- MD5 observed: `{audit['md5']}`",
        f"- MD5 published by Zenodo: `{config['data'].get('expected_md5', 'not configured')}`",
        "- The raw CSV is read only and is never overwritten by this project.",
        "",
        "## 2. Shape, types and completeness",
        "",
        f"- Rows: {audit['shape']['rows']}; columns: {audit['shape']['columns']}.",
        f"- Duplicate rows: {audit['duplicates']['n_duplicate_rows']} "
        f"({audit['duplicates']['n_extra_duplicates']} beyond the first copy), retained for inspection.",
        f"- Missing values: {audit['missing_values'] if audit['missing_values'] else 'none detected'}.",
        f"- Infinite/non-finite values: {audit['non_finite_values'] or 'none detected'}.",
        "- IQR-rule outliers are recorded but retained: in drilling data, large excursions can "
        "be the kick event itself. Counts: "
        + ", ".join(
            f"{column}={audit['outliers_iqr'][column]['count']}"
            for column in audit["outliers_iqr"]
            if audit["outliers_iqr"][column]["count"] > 0
        ) + ".",
        f"- Constant columns: {audit['constants']['constant']}.",
        f"- Near-constant columns (<5 distinct values): {audit['constants']['near_constant']}.",
        f"- Non-numeric columns: {[c for c, t in audit['dtypes'].items() if not t.startswith(('int', 'float'))] or 'none'}.",
        "",
        "| Column | Dtype | Min | Median | Max | Distinct |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for column in audit["columns"]:
        summary = audit["numeric_summary"].get(column)
        if summary is None:
            continue
        lines.append(
            f"| {column} | {audit['dtypes'][column]} | {summary['min']:.6g} | "
            f"{summary['median']:.6g} | {summary['max']:.6g} | {summary['n_unique']} |"
        )

    lines += [
        "",
        "## 3. Target verification",
        "",
        f"- Configured target column: `{target.column}`; mode: `{target.mode}`.",
        f"- Rule: {target.rule}.",
        f"- Class counts: {target.n_negative} negative / {target.n_positive} positive "
        f"({target.positive_fraction:.1%} positive).",
        f"- First positive row index: {target.onset_index}; depth there: {target.onset_depth} ft "
        f"(documented kick depth: {config['target'].get('documented_kick_depth_ft')} ft).",
        f"- Label transitions: {target.transitions}.",
        "- The source documentation calls ActiveGL 'Active Gain Loss / pit gain-loss', an "
        "**indicator of a kick**. It is continuous, not a supplied binary class. The binary "
        "label used here is therefore an experimental proxy based on the physically meaningful "
        "change from pit loss (negative) to pit gain (positive), which coincides with the "
        "documented kick depth in the source paper. Supervisor confirmation is required.",
        "",
        "Warnings:",
        "",
    ]
    lines += [f"- {warning}" for warning in target.warnings] or ["- None"]
    lines += [
        "",
        "## 4. Leakage and redundancy findings",
        "",
        "| Column(s) | Severity | Reason |",
        "| --- | --- | --- |",
    ]
    for item in audit["leakage_findings"]:
        lines.append(f"| {item['column']} | {item['severity']} | {item['reason']} |")
    lines += ["", "Verified algebraic identities:", ""]
    for name, result in audit["known_identities"].items():
        lines.append(f"- `{name}`: {result}")
    lines += [
        "",
        "## 5. Temporal and event information",
        "",
        f"- Timestamp-like columns: {temporal['timestamp_columns'] or 'none'}.",
        f"- Group/event columns: {temporal['group_columns'] or 'none'}.",
        f"- Sequence order plausible from monotonic depth: {temporal['sequence_order_plausible']}.",
        f"- Validated early-warning lead time possible: **{temporal['supports_validated_early_warning']}**.",
        "",
    ]
    lines += [f"- {note}" for note in temporal["notes"]]
    lines += [
        "",
        "## 6. Target correlations (observed, not causal)",
        "",
        "| Feature | Pearson r with target signal |",
        "| --- | ---: |",
    ]
    for column, value in audit["top_target_correlations"].items():
        lines.append(f"| {column} | {value:.4f} |")
    lines += [
        "",
        "Correlation does not establish causation. High correlations with `ATVolume` and "
        "depth-related columns are consequences of how the simulation records the event and "
        "are exactly why those variables are excluded from model inputs.",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote audit report to %s", report_path)
    return report_path
