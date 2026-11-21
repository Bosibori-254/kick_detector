"""Command-line interface for the drilling kick-detection research pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from .data_loader import load_config, load_raw_data, resolve_path
from .data_validation import audit_dataset, validate_target, write_audit_report
from .eda import generate_figures, write_eda_report
from .feature_engineering import add_engineered_features, select_feature_columns
from .predict import load_bundle, parse_values_json, predict_one, required_raw_features
from .preprocessing import split_data

logger = logging.getLogger(__name__)


def _print_proxy_warning(config: dict[str, Any]) -> None:
    if config["target"].get("mode") == "proxy_from_threshold":
        print(
            "WARNING: target labels are EXPERIMENTAL PROXIES derived from "
            f"'{config['target']['column']}', not authoritative kick labels.",
            file=sys.stderr,
        )


def cmd_audit(config: dict[str, Any], _args: argparse.Namespace) -> int:
    """Audit the raw dataset and write reports/data_audit.md."""
    frame = load_raw_data(config=config)
    audit = audit_dataset(frame, config, resolve_path(config["data"]["raw_path"], config))
    path = write_audit_report(
        audit, config, resolve_path(config["paths"]["reports_dir"], config) / "data_audit.md"
    )
    print(f"Audit written to {path}")
    print(f"Rows: {audit['shape']['rows']}, columns: {audit['shape']['columns']}")
    print(f"Proxy positives: {audit['target'].n_positive} / {audit['shape']['rows']}")
    print(f"Validated early-warning support: {audit['temporal_support']['supports_validated_early_warning']}")
    return 0


def cmd_eda(config: dict[str, Any], _args: argparse.Namespace) -> int:
    """Generate EDA figures and reports/eda_report.md."""
    frame = load_raw_data(config=config)
    target = validate_target(frame, config)
    figures = generate_figures(frame, target, config)
    path = write_eda_report(frame, target, config, figures)
    print(f"EDA report written to {path}")
    print(f"Figures: {', '.join(figures)}")
    return 0


def cmd_preprocess(config: dict[str, Any], _args: argparse.Namespace) -> int:
    """Create and save the leakage-aware feature matrices."""
    frame = load_raw_data(config=config)
    target = validate_target(frame, config)
    featured = add_engineered_features(frame, config)
    columns = select_feature_columns(featured, config)
    split = split_data(featured[columns], target.labels, config)
    output_dir = resolve_path(config["data"]["processed_dir"], config)
    output_dir.mkdir(parents=True, exist_ok=True)
    split.X_train.assign(target=split.y_train.to_numpy()).to_csv(
        output_dir / "train.csv", index=False
    )
    split.X_test.assign(target=split.y_test.to_numpy()).to_csv(
        output_dir / "test.csv", index=False
    )
    print(f"Saved processed splits in {output_dir} ({len(split.X_train)} train / {len(split.X_test)} test rows)")
    print(f"Model features ({len(columns)}): {', '.join(columns)}")
    return 0


def cmd_train(config: dict[str, Any], _args: argparse.Namespace) -> int:
    """Train, tune thresholds on validation, and save the model bundle."""
    _print_proxy_warning(config)
    from .train import train_models

    result = train_models(config)
    print(f"Trained models: {', '.join(result.models_used)}")
    print(f"Best model by validation ROC-AUC: {result.bundle['best_model']}")
    for name, details in result.threshold_details.items():
        print(f"  {name}: threshold={details['threshold']:.4f} "
              f"(validation {details['tuning_metric']}={details['validation_score']:.4f})")
    print(f"Bundle: {resolve_path(config['paths']['bundle_path'], config)}")
    return 0


def cmd_evaluate(config: dict[str, Any], _args: argparse.Namespace) -> int:
    """Evaluate all saved models on the independent test split."""
    from .evaluate import evaluate_all, save_evaluation_plots, save_results_tables, write_evaluation_report, load_test_data

    bundle = load_bundle(config=config)
    X_test, y_test, _ = load_test_data(config, bundle["feature_columns"])
    results = evaluate_all(bundle, X_test, y_test, config)
    table = save_results_tables(results, config)
    save_evaluation_plots(results, y_test, config)
    report = write_evaluation_report(results, table, config)
    print(table.to_string(index=False))
    print(f"Evaluation report: {report}")
    return 0


def cmd_robustness(config: dict[str, Any], _args: argparse.Namespace) -> int:
    """Run sensor-noise and missing-data experiments."""
    from .robustness import run_robustness, save_robustness_plots, write_robustness_report

    bundle = load_bundle(config=config)
    table = run_robustness(bundle, config)
    save_robustness_plots(table, config)
    report = write_robustness_report(table, config)
    print(f"Robustness rows: {len(table)}; report: {report}")
    return 0


def cmd_explain(config: dict[str, Any], _args: argparse.Namespace) -> int:
    """Generate SHAP explanations for the selected model."""
    from .explainability import generate_explanations

    bundle = load_bundle(config=config)
    report = generate_explanations(bundle, config)
    print(f"Explainability report: {report}")
    return 0


def cmd_predict(config: dict[str, Any], args: argparse.Namespace) -> int:
    """Predict a single drilling condition from supplied parameter values."""
    bundle = load_bundle(config=config)
    if args.values_file:
        path = Path(args.values_file)
        if not path.is_file():
            raise FileNotFoundError(f"Input JSON file not found: {path}")
        values = parse_values_json(path.read_text(encoding="utf-8"))
    elif args.values:
        values = parse_values_json(args.values)
    else:
        print("Enter drilling parameter values (blank input is rejected):")
        values = {}
        for column in required_raw_features(bundle):
            bound = bundle.get("feature_bounds", {}).get(column, {})
            hint = ""
            if bound:
                hint = f" [training range {bound['min']:.4g} to {bound['max']:.4g}]"
            values[column] = input(f"{column}{hint}: ")

    result = predict_one(
        values, bundle, config, model_name=args.model, strict_bounds=not args.allow_unbounded
    )
    print(json.dumps(result, indent=2))
    return 0


def cmd_all(config: dict[str, Any], args: argparse.Namespace) -> int:
    """Run the complete research pipeline in sequence."""
    commands = [cmd_audit, cmd_eda, cmd_preprocess, cmd_train, cmd_evaluate, cmd_robustness, cmd_explain]
    for command in commands:
        code = command(config, args)
        if code:
            return code
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(
        prog="python -m src.cli",
        description="Drilling kick detection research pipeline",
    )
    parser.add_argument("--config", default=None, help="Path to an alternative YAML config")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("audit", "Audit the raw dataset"),
        ("eda", "Generate exploratory analysis figures and report"),
        ("preprocess", "Build leakage-aware train/test features"),
        ("train", "Train models and save the bundle"),
        ("evaluate", "Evaluate models on the test set"),
        ("robustness", "Run controlled input perturbation experiments"),
        ("explain", "Generate SHAP explanations"),
        ("predict", "Predict one drilling condition"),
        ("all", "Run the complete pipeline"),
    ):
        subparsers.add_parser(name, help=help_text)

    predict_parser = subparsers.choices["predict"]
    predict_parser.add_argument("--values", help="JSON object of parameter values")
    predict_parser.add_argument("--values-file", help="Path to a JSON file of parameter values")
    predict_parser.add_argument("--model", default=None, help="Model name (default: best model)")
    predict_parser.add_argument(
        "--allow-unbounded", action="store_true",
        help="Do not reject values outside the training envelope (use with caution)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    handlers = {
        "audit": cmd_audit, "eda": cmd_eda, "preprocess": cmd_preprocess,
        "train": cmd_train, "evaluate": cmd_evaluate, "robustness": cmd_robustness,
        "explain": cmd_explain, "predict": cmd_predict, "all": cmd_all,
    }
    try:
        return handlers[args.command](config, args)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
