"""Regenerate the five research notebooks from source code strings.

Run from the project root:
    python notebooks/_build_notebooks.py

The notebooks call the same functions as the CLI, so results stay reproducible.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

NOTEBOOKS_DIR = Path(__file__).resolve().parent

BOOTSTRAP = """\
import sys
from pathlib import Path
PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
"""


def build(filename: str, title: str, cells: list[tuple[str, str]]) -> None:
    """Write one notebook from (kind, source) pairs."""
    notebook = nbf.v4.new_notebook()
    notebook.cells = [
        nbf.v4.new_markdown_cell(source) if kind == "md" else nbf.v4.new_code_cell(source)
        for kind, source in cells
    ]
    notebook.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    }
    path = NOTEBOOKS_DIR / filename
    nbf.write(notebook, path)
    print(f"Wrote {path}")


DISCLAIMER = (
    "> Research prototype only. The dataset is simulated and labels are experimental proxies. "
    "No result here validates field deployment or early-warning lead time."
)

build(
    "01_exploratory_data_analysis.ipynb",
    "Exploratory data analysis",
    [
        ("md", f"# 01 - Exploratory Data Analysis\n\n{DISCLAIMER}"),
        ("code", BOOTSTRAP + "\nimport matplotlib.pyplot as plt\nimport pandas as pd\nimport seaborn as sns\nfrom src.data_loader import load_config, load_raw_data\nfrom src.data_validation import validate_target\n\nconfig = load_config()\nframe = load_raw_data(config=config)\ntarget = validate_target(frame, config)\nprint('shape:', frame.shape)"),
        ("md", "## 1. Schema, missing values and duplicates"),
        ("code", "frame_info = pd.DataFrame({\n    'dtype': frame.dtypes.astype(str),\n    'missing': frame.isna().sum(),\n    'unique': frame.nunique(),\n})\nframe_info"),
        ("code", "print('Duplicate rows:', frame.duplicated().sum())\nframe.describe().T"),
        ("md", "## 2. Target verification and class balance\n\n`ActiveGL` is a continuous pit gain/loss indicator. The sign change from pit loss to "
               "pit gain coincides with the kick depth documented in the source paper. The resulting label "
               "is an experimental proxy, not authoritative ground truth."),
        ("code", "print('Rule:', target.rule)\nprint('Onset index:', target.onset_index, 'depth (ft):', target.onset_depth)\nprint('Positives:', target.n_positive, 'Negatives:', target.n_negative)\nprint('Warnings:', target.warnings)\nax = target.labels.value_counts().sort_index().plot(kind='bar', color='#4c78a8')\nax.set_xlabel('Proxy class'); ax.set_ylabel('Rows'); plt.show()"),
        ("md", "## 3. Feature distributions and correlation structure"),
        ("code", "from src.feature_engineering import add_engineered_features\nfeatured = add_engineered_features(frame, config)\nfeatured[['FIn', 'FOut', 'Flow_Imbalance', 'DPPress', 'HLoad']].hist(bins=40, figsize=(12, 8))\nplt.tight_layout(); plt.show()"),
        ("code", "numeric = featured.select_dtypes(include='number')\nplt.figure(figsize=(13, 10))\nsns.heatmap(numeric.corr(), cmap='coolwarm', center=0, vmin=-1, vmax=1, square=True)\nplt.title('Pearson correlation (observation, not causation)')\nplt.tight_layout(); plt.show()"),
        ("md", "## 4. Parameter behaviour by proxy class"),
        ("code", "for column in ['FOut', 'Flow_Imbalance', 'DPPress', 'HLoad']:\n    sns.boxplot(x=target.labels.to_numpy(), y=featured[column], showfliers=False)\n    plt.xlabel('Proxy class (0=normal, 1=kick)')\n    plt.title(column)\n    plt.show()"),
        ("md", "## 5. Flow, pressure and pit-gain relationships\n\nThe identity `MRFlow = FIn - FOut` justifies `Flow_Imbalance = FOut - FIn`. "
               "A positive imbalance means return flow exceeds input flow, consistent with an influx."),
        ("code", "plt.scatter(featured['FIn'], featured['FOut'], c=target.labels, s=8, alpha=0.6, cmap='coolwarm')\nplt.xlabel('FIn'); plt.ylabel('FOut'); plt.title('Flow in vs flow out'); plt.show()\nplt.scatter(featured['DPPress'], featured['HLoad'], c=target.labels, s=8, alpha=0.6, cmap='coolwarm')\nplt.xlabel('DPPress'); plt.ylabel('HLoad'); plt.title('Pressure vs hook load'); plt.show()"),
        ("md", "## 6. Reproduce published figures and report"),
        ("code", "from src.eda import generate_figures, write_eda_report\nfigures = generate_figures(frame, target, config)\nreport = write_eda_report(frame, target, config, figures)\nprint(report)"),
    ],
)

build(
    "02_data_preprocessing.ipynb",
    "Data preprocessing",
    [
        ("md", f"# 02 - Data Preprocessing\n\n{DISCLAIMER}"),
        ("code", BOOTSTRAP + "\nfrom src.data_loader import load_config, load_raw_data\nfrom src.data_validation import validate_target\nfrom src.feature_engineering import add_engineered_features, select_feature_columns, verify_flow_conventions\n\nconfig = load_config()\nframe = load_raw_data(config=config)\ntarget = validate_target(frame, config)"),
        ("md", "## Cleaning decisions\n\nThere are no missing values in the raw file; five exact duplicate rows are retained "
               "because they may represent genuine steady-state simulator samples. Constant columns and "
               "target-derived columns are excluded from model inputs."),
        ("code", "print('Missing values:', int(frame.isna().sum().sum()))\nprint('Duplicate rows:', frame.duplicated().sum())\nfeatured = add_engineered_features(frame, config)\nprint(verify_flow_conventions(featured))\ncolumns = select_feature_columns(featured, config)\nprint('Selected features:', columns)"),
        ("md", "## Scaling and imputation\n\nThe `ColumnTransformer` below is fitted on training data only. Robust scaling is used "
               "because genuine transient kick responses should not be treated as errors."),
        ("code", "from src.preprocessing import build_preprocessor, split_data, make_validation_split\nsplit = split_data(featured[columns], target.labels, config)\npreprocessor = build_preprocessor(columns, config)\nX_train_t = preprocessor.fit_transform(split.X_train)\nprint('Transformed training matrix:', X_train_t.shape)\nfit, val, y_fit, y_val = make_validation_split(split.X_train, split.y_train, split.train_groups, config)\nprint('Fit/validation rows:', len(fit), len(val))"),
        ("md", "## Group-aware split\n\nContiguous 25-row blocks remain intact and are stratified by the proxy class, "
               "reducing leakage from adjacent near-duplicate measurements. The test set is not used "
               "for any fitting or tuning."),
        ("code", "print('Train/test rows:', len(split.X_train), len(split.X_test))\nprint('Positive fraction:', split.y_train.mean(), split.y_test.mean())\nsplit.X_train.assign(target=split.y_train.to_numpy()).head()"),
    ],
)

build(
    "03_model_training.ipynb",
    "Model training",
    [
        ("md", f"# 03 - Model Training\n\n{DISCLAIMER}\n\nDecision thresholds are tuned on a validation split; "
               "the test split is never used during training."),
        ("code", BOOTSTRAP + "\nfrom src.data_loader import load_config\nfrom src.train import train_models\n\nconfig = load_config()\nresult = train_models(config)\nprint('Models:', result.models_used)\nprint('Best:', result.bundle['best_model'])"),
        ("md", "## Validation-only threshold summary"),
        ("code", "result.threshold_details"),
        ("code", "result.validation_summary"),
        ("md", "## Saved artefact\n\nThe bundle contains the fitted pipelines, tuned thresholds, feature order and "
               "provenance metadata. It is loaded by the evaluation, robustness, explainability and "
               "Streamlit components."),
        ("code", "from src.data_loader import resolve_path\nprint(resolve_path(config['paths']['bundle_path'], config))\nprint('Training rows:', result.bundle['train_rows'], 'Test rows:', result.bundle['test_rows'])"),
    ],
)

build(
    "04_model_evaluation.ipynb",
    "Model evaluation",
    [
        ("md", f"# 04 - Model Evaluation\n\n{DISCLAIMER}\n\nKick recall, precision, PR-AUC and false alarms "
               "take priority over accuracy."),
        ("code", BOOTSTRAP + "\nfrom src.data_loader import load_config\nfrom src.predict import load_bundle\nfrom src.evaluate import load_test_data, evaluate_all, save_results_tables, save_evaluation_plots, write_evaluation_report\n\nconfig = load_config()\nbundle = load_bundle(config=config)\nX_test, y_test, _ = load_test_data(config, bundle['feature_columns'])\nresults = evaluate_all(bundle, X_test, y_test, config)"),
        ("code", "table = save_results_tables(results, config)\ntable"),
        ("code", "save_evaluation_plots(results, y_test, config)\nfrom IPython.display import Image, display\ndisplay(Image(filename='../results/figures/confusion_matrices.png'))\ndisplay(Image(filename='../results/figures/precision_recall_curves.png'))"),
        ("code", "print(write_evaluation_report(results, table, config))"),
    ],
)

build(
    "05_model_explainability.ipynb",
    "Model explainability",
    [
        ("md", f"# 05 - Model Explainability\n\n{DISCLAIMER}\n\nSHAP explains model outputs; it does not "
               "establish physical causation."),
        ("code", BOOTSTRAP + "\nfrom pathlib import Path\nfrom IPython.display import Image, display\nfrom src.data_loader import load_config\nfrom src.predict import load_bundle\nfrom src.explainability import generate_explanations\n\nconfig = load_config()\nbundle = load_bundle(config=config)\nreport = generate_explanations(bundle, config)\nprint(report)"),
        ("code", "output = Path('../results/explainability')\nfor name in ['shap_summary.png', 'shap_feature_importance.png', 'shap_individual_explanation.png']:\n    path = output / name\n    if path.exists():\n        display(Image(filename=str(path)))"),
        ("md", "## Interpretation\n\nLarge SHAP magnitudes identify measurements that most affect predictions on "
               "this simulated test set. Flow imbalance, pressure and mechanical responses are physically "
               "plausible indicators, but model importance is not proof of a physical cause."),
    ],
)
