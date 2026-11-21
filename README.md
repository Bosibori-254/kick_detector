# Development of a Machine Learning-Based Early Warning Framework for Drilling Kick Detection

A modular, reproducible research prototype for investigating whether machine learning can
recognise kick-related patterns in the public **DataDRILL** simulator dataset.

> **Important safety notice**
> This is an experimental academic research project. Its outputs are model predictions,
> not confirmed kicks, and **must not replace approved well-control procedures, pit-volume
> and flow monitoring, or qualified drilling personnel**. The dataset is simulated and has
> no timestamps or event identifiers, so the project evaluates **kick-condition
> classification**, not a validated early-warning lead time.

---

## 1. Background and research problem

A drilling kick occurs when formation pressure exceeds the hydrostatic pressure of the
mud column and formation fluid enters the wellbore. Detecting a kick early reduces the
risk of loss of well control, but conventional indicators (flow out versus flow in, pit
volume, standpipe/drill-pipe pressure, pump speed, mud density, ROP) can be affected by
noise, changing drilling conditions and false alarms.

This project investigates:

1. Whether supervised models can classify kick-related states from real-time drilling
   parameters while controlling false alarms and missed kicks.
2. Whether an unsupervised anomaly detector and a simple threshold rule provide useful
   comparison points.
3. Which measured parameters most influence model predictions.
4. How performance changes under controlled sensor noise and missing data.

## 2. Objectives

- Audit and document the dataset before any modelling.
- Build a leakage-aware preprocessing and feature-engineering pipeline.
- Compare Logistic Regression, Random Forest, XGBoost (with a documented
  scikit-learn fallback), Isolation Forest and an experimental threshold baseline.
- Evaluate recall, precision, PR-AUC and false alarms rather than accuracy alone.
- Produce research reports, figures, model explanations and a prediction interface.

## 3. Dataset source and installation

- Source: [DataDRILL on Zenodo](https://zenodo.org/records/12759014), DOI `10.5281/zenodo.12759014`.
- Companion paper: Arifeen et al., *DataDRILL: Formation Pressure Prediction and Kick
  Detection for Drilling Rigs*, arXiv:2409.19724.
- Licence: CC BY 4.0. Cite the creators when using the data.
- Primary file: `Kick_Detection.csv` (expected MD5 `8bbd611ca9d66c397d43c61a152bed93`).

If the file is not already present, download it from:

```text
https://zenodo.org/records/12759014/files/Kick_Detection.csv?download=1
```

Place it at:

```text
data/raw/Kick_Detection.csv
```

The raw CSV is never overwritten. The project reads it in a fixed row order and
records its checksum in `reports/data_audit.md`.

### Critical target note

The dataset does **not** contain an explicit kick-label column. `ActiveGL` is documented
as continuous *Active Gain Loss / pit gain-loss* and is a physical kick indicator. In
this dataset it changes from a small pit loss to a growing pit gain at the depth where
the source paper documents the kick (about 12,650 ft). The project therefore uses an
**experimental proxy label**:

```text
potential kick = 1 if ActiveGL > 0 bbl, otherwise 0
```

This decision is configurable in `configs/config.yaml`. It is **not authoritative
ground truth** and requires confirmation by a petroleum engineering supervisor before
results are used in a paper or presentation. Set
`target.allow_proxy_training: false` to stop training until confirmation is obtained.
`ActiveGL`, its near-exact duplicate `ATVolume`, scenario depth variables and the configured
formation pressure are excluded from all model inputs to avoid target leakage and
scenario memorisation.

## 4. Software requirements

- Windows with PowerShell.
- Python 3.10+ (the project was developed with Python 3.12).
- Dependencies are listed in `requirements.txt`: numpy, pandas, scikit-learn, xgboost,
  shap, matplotlib, seaborn, PyYAML, joblib, Streamlit, pytest, Jupyter.

If XGBoost is unavailable, training automatically substitutes scikit-learn's
`HistGradientBoostingClassifier` and logs the change. If SHAP is unavailable, the
explainability command writes a note instead of fabricating explanations.

## 5. Environment setup (Windows PowerShell)

```powershell
cd C:\Users\dnyak\OneDrive\Fifth_year_project
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

All commands below assume the virtual environment is active and the working
directory is the project root.

## 6. Project structure

```text
drilling-kick-detection/
├── data/
│   ├── raw/Kick_Detection.csv
│   └── processed/
├── notebooks/
│   ├── _build_notebooks.py
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_model_evaluation.ipynb
│   └── 05_model_explainability.ipynb
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── data_validation.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── eda.py
│   ├── train.py
│   ├── evaluate.py
│   ├── robustness.py
│   ├── explainability.py
│   ├── predict.py
│   └── cli.py
├── app/streamlit_app.py
├── configs/config.yaml
├── models/
├── results/{figures,explainability,predictions}
├── reports/
├── tests/
├── requirements.txt
├── README.md
└── .gitignore
```

## 7. Command-line workflow

```powershell
python -m src.cli audit        # dataset audit -> reports/data_audit.md
python -m src.cli eda          # EDA figures + reports/eda_report.md
python -m src.cli preprocess   # leakage-aware train/test CSV files
python -m src.cli train        # train and save the model bundle
python -m src.cli evaluate     # test metrics, plots, model comparison
python -m src.cli robustness   # noise and missing-data experiments
python -m src.cli explain      # SHAP explanations
python -m src.cli all          # run the complete pipeline
```

Every command fails with a clear message if its input file or trained model is
missing; none silently succeeds.

### Single prediction from PowerShell

```powershell
python -m src.cli predict --values '{"FIn": 0.95, "FOut": 1.8, "DPPress": 1500}'
```

Repeat with **all** parameters listed by `required_raw_features` (the CLI prompts for
missing values when `--values` is omitted). A JSON file can be supplied with
`--values-file`.

### Streamlit application

```powershell
streamlit run app/streamlit_app.py
```

The app validates every input, rejects missing or implausible values, displays the
predicted condition, model score, threshold, warning message, global contributors and
a prominent research-only disclaimer.

## 8. Testing

```powershell
python -m pytest tests -v
```

Tests cover dataset loading, target verification, feature engineering, missing-value
handling in the pipeline, splitting, prediction validation and saved-model contents.

## 9. Reports and outputs

| Output | Location |
| --- | --- |
| Dataset audit | `reports/data_audit.md` |
| Temporal feasibility assessment | `reports/temporal_analysis.md` |
| EDA report and figures | `reports/eda_report.md`, `results/figures/` |
| Model comparison | `results/model_comparison.csv`, `results/metrics.json` |
| Evaluation report | `reports/model_evaluation.md` |
| Robustness results | `results/robustness_results.csv`, `reports/robustness_results.md` |
| SHAP explanations | `results/explainability/` |
| Research summary | `reports/research_summary.md` |

## 10. Methodological decisions

- The raw file has one continuous simulated run, no timestamps and no well/run IDs.
  Contiguous 25-row blocks are kept together in a stratified group split, with a
  purgegap around the test blocks. A strictly chronological training split is impossible
  because all negative samples precede the kick.
- Preprocessing (median imputation and robust scaling) is fitted only on training rows.
- Decision thresholds are chosen on a validation subset; the test set is used once for
  final reporting.
- Flow imbalance is only created after verifying `MRFlow = FIn - FOut`; therefore
  `Flow_Imbalance = FOut - FIn` is positive when return flow exceeds input flow.
- Constant columns, target-derived columns and scenario-specific depth/pressure
  variables are excluded. Correlation is not interpreted as causation.

## 11. Limitations

- One simulated drilling scenario from a digital twin; no field data, no multiple wells.
- No timestamps or event identifiers: no validated detection lead time can be claimed.
- Labels are experimental proxies derived from a continuous pit-gain signal.
- False alarms and missed kicks are reported, but the operating point is not validated
  against real operational costs.
- High classification scores do not demonstrate that the system would prevent a blowout.
- SHAP importance is associational, not causal.

## 12. Future improvements

- Confirm the target definition and obtain event onset times with a drilling supervisor.
- Validate on multiple simulator runs, wells and real operational datasets.
- Develop time-aware models with causal rolling windows and forward-chaining evaluation.
- Calibrate probabilities and study alert persistence (consecutive positive predictions).
- Investigate physics-informed constraints and domain adaptation across formations.
