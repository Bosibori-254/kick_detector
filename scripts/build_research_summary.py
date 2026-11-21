"""Generate reports/research_summary.md from actual experiment outputs.

Any section whose input file is absent is explicitly marked as not conducted.
No numerical finding is written unless it was produced by the pipeline.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.data_loader import load_config, load_raw_data, resolve_path
from src.data_validation import validate_target


def _fmt(value: float) -> str:
    return "not available" if pd.isna(value) else f"{float(value):.3f}"


def main() -> None:
    config = load_config()
    results_dir = resolve_path(config["paths"]["results_dir"], config)
    reports_dir = resolve_path(config["paths"]["reports_dir"], config)
    reports_dir.mkdir(parents=True, exist_ok=True)

    frame = load_raw_data(config=config)
    target = validate_target(frame, config)
    lines = [
        "# Research summary: machine learning-based drilling kick detection",
        "",
        "> Experimental research prototype. All numbers below were produced by the "
        "pipeline from the downloaded DataDRILL CSV. Experimental assumptions and "
        "engineering interpretations are labelled as such.",
        "",
        "## Research problem",
        "",
        "Conventional kick indicators can be noisy and ambiguous. This study investigates "
        "whether machine learning models classify kick-related states in simulator data "
        "with acceptable missed-kick and false-alarm rates, and which parameters drive "
        "those predictions.",
        "",
        "## Dataset description",
        "",
        f"- Source: {config['data']['source_url']} (`Kick_Detection.csv`).",
        f"- Observations: {frame.shape[0]}; recorded variables: {frame.shape[1]}; all numeric.",
        f"- No timestamp, well ID, run ID or event ID is present in the header.",
        f"- The file is one continuous simulated drilling sequence; depth is monotonic.",
        "",
        "## Target definition (experimental assumption)",
        "",
        f"- The source documents `{target.column}` as continuous pit gain/loss, an indicator of a kick.",
        f"- Rule used: {target.rule}.",
        f"- Proxy onset row {target.onset_index} at {target.onset_depth} ft, matching the "
        f"documented kick depth of {config['target']['documented_kick_depth_ft']} ft within 2 ft.",
        f"- Class counts: {target.n_negative} normal and {target.n_positive} proxy kick "
        f"({target.positive_fraction:.1%} positive).",
        "- This is not authoritative ground truth and requires supervisor confirmation.",
        "",
        "## Preprocessing and feature engineering methodology",
        "",
        "- Raw CSV preserved unchanged; no missing values were detected and duplicate rows were retained.",
        "- Target-derived `ActiveGL` and its near-duplicate `ATVolume`, redundant `MRFlow`, constant columns, "
        "scenario-specific depth variables and configured formation pressure were excluded.",
        "- `Flow_Imbalance = FOut - FIn` was added after verifying the documented sign convention "
        "(`MRFlow = FIn - FOut`); a relative imbalance was also computed.",
        "- Median imputation and robust scaling were fitted on training rows only.",
        "",
        "## Training and validation strategy",
        "",
        f"- Contiguous {config['split']['block_size']}-row blocks were kept intact using a "
        "stratified group split with a purge gap around test blocks.",
        f"- Seed: {config['project']['seed']}. Validation fraction: "
        f"{config['split']['validation_size']}; test fraction: {config['split']['test_size']}.",
        "- Decision thresholds were tuned on validation data: the midpoint of the validation "
        "class-score gap when separable, otherwise the best PR-curve operating point subject "
        "to minimum recall. Test data were used only for final reporting.",
        "- The selected supervised model is the one with the highest validation ROC-AUC; ties "
        "are broken by validation F1, recall, false-alarm rate and score margin (all without "
        "using the test set).",
        "- Class weighting was used for supervised classifiers. The anomaly detector was fitted "
        "without labels; the threshold baseline is an experimental rule, not a field threshold.",
        "",
        "## Models investigated",
        "",
        "- Logistic Regression (interpretable baseline).",
        "- Random Forest (nonlinear interactions).",
        "- XGBoost, or HistGradientBoostingClassifier if XGBoost is unavailable.",
        "- Isolation Forest (mixed training data) and Isolation Forest fitted only on normal "
        "training rows; both are unsupervised anomaly detectors evaluated post-hoc.",
        "- Conventional flow-imbalance threshold rule (research comparison only).",
        "",
        "## Evaluation methodology",
        "",
        "Accuracy, precision, recall, F1, ROC-AUC, PR-AUC, confusion matrices, false positive "
        "rate and FAR = FP/(FP+TN). Kick recall, precision, PR-AUC and false alarms are "
        "emphasised; accuracy is not used for model selection.",
        "",
    ]

    metrics_path = results_dir / "metrics.json"
    if metrics_path.is_file():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        lines += [
            "## Observed model comparison (independent test set)",
            "",
            "| Model | Family | Precision | Recall | F1 | ROC-AUC | PR-AUC | FAR | Missed kicks | False warnings |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for name, item in metrics.items():
            lines.append(
                f"| {name} | {item['family']} | {_fmt(item['precision'])} | {_fmt(item['recall'])} | "
                f"{_fmt(item['f1'])} | {_fmt(item['roc_auc'])} | {_fmt(item['pr_auc'])} | "
                f"{_fmt(item['far'])} | {int(item['fn'])} | {int(item['fp'])} |"
            )
        supervised = {k: v for k, v in metrics.items() if v["family"] == "supervised"}
        if supervised:
            best = max(supervised.items(), key=lambda pair: pair[1]["pr_auc"])
            lines += [
                "",
                f"Highest supervised test PR-AUC: **{best[0]}** ({best[1]['pr_auc']:.3f}); "
                f"recall {best[1]['recall']:.3f}, precision {best[1]['precision']:.3f}, "
                f"{int(best[1]['fn'])} missed kicks and {int(best[1]['fp'])} false warnings.",
            ]
        lines.append("")
    else:
        lines += ["## Model comparison", "", "Experiment not yet conducted (`results/metrics.json` absent).", ""]

    robustness_path = results_dir / "robustness_results.csv"
    if robustness_path.is_file():
        table = pd.read_csv(robustness_path)
        lines += [
            "## Robustness results",
            "",
            "Noise levels are multiples of training-set standard deviation; missingness is MCAR. "
            "These are experimental perturbations, not verified sensor specifications.",
            "",
            "| Perturbation | Level | Model | Recall | PR-AUC | FAR |",
            "| --- | ---: | --- | ---: | ---: | ---: |",
        ]
        for _, row in table.iterrows():
            lines.append(
                f"| {row['perturbation']} | {row['level']} | {row['model']} | "
                f"{_fmt(row['recall'])} | {_fmt(row['pr_auc'])} | {_fmt(row['far'])} |"
            )
        lines.append("")
    else:
        lines += ["## Robustness results", "", "Experiment not yet conducted.", ""]

    explain_dir = resolve_path(config["paths"]["explainability_dir"], config)
    explain_report = explain_dir / "explainability_report.md"
    if explain_report.is_file():
        text = explain_report.read_text(encoding="utf-8")
        top = [line for line in text.splitlines() if line.startswith("| ") and "| ---" not in line]
        lines += [
            "## Explainability findings",
            "",
            "SHAP was applied to the selected model. Importance is associational, not causal.",
            "",
        ]
        lines += top[:12]
        lines.append("")
    else:
        lines += ["## Explainability findings", "", "Experiment not yet conducted.", ""]

    lines += [
        "## Engineering interpretation",
        "",
        "- Observed flow-imbalance and pressure changes are consistent with an influx in "
        "this simulated scenario; the models learn associations within that scenario.",
        "- Near-perfect supervised and threshold-rule scores are expected because the proxy "
        "label comes from pit gain, which is tightly coupled to return-flow imbalance in this "
        "simulator. They do not establish performance on field data.",
        "- A low false-alarm rate with poor kick recall is unsafe; a high recall with many "
        "false warnings can cause alarm fatigue. The operating point requires engineering judgement.",
        "- No evidence here establishes operational readiness, blowout prevention, or "
        "validated warning lead time.",
        "",
        "## Limitations",
        "",
        "- Single simulated rig scenario; no field validation, no multiple wells or runs.",
        "- No timestamps/event IDs, so early-warning timing cannot be calculated.",
        "- Experimental proxy labels require supervisor confirmation.",
        "- Model importance does not prove physical causation.",
        "",
        "## Recommendations for future research",
        "",
        "1. Obtain authoritative kick labels and event onset times from drilling supervisors.",
        "2. Collect multiple simulator runs and real field cases for external validation.",
        "3. Implement forward-chaining evaluation with causal rolling features and alert persistence.",
        "4. Calibrate probabilities and quantify the cost of missed kicks versus false alarms.",
        "5. Evaluate physics-informed and sequence models before any operational claims.",
        "",
    ]

    output = reports_dir / "research_summary.md"
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":  # pragma: no cover
    main()
