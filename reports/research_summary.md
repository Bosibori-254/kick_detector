# Research summary: machine learning-based drilling kick detection

> Experimental research prototype. All numbers below were produced by the pipeline from the downloaded DataDRILL CSV. Experimental assumptions and engineering interpretations are labelled as such.

## Research problem

Conventional kick indicators can be noisy and ambiguous. This study investigates whether machine learning models classify kick-related states in simulator data with acceptable missed-kick and false-alarm rates, and which parameters drive those predictions.

## Dataset description

- Source: https://zenodo.org/records/12759014 (`Kick_Detection.csv`).
- Observations: 2337; recorded variables: 28; all numeric.
- No timestamp, well ID, run ID or event ID is present in the header.
- The file is one continuous simulated drilling sequence; depth is monotonic.

## Target definition (experimental assumption)

- The source documents `ActiveGL` as continuous pit gain/loss, an indicator of a kick.
- Rule used: ActiveGL > 0 bbl (pit gain rather than pit loss).
- Proxy onset row 1501 at 12650.15 ft, matching the documented kick depth of 12650.0 ft within 2 ft.
- Class counts: 1501 normal and 836 proxy kick (35.8% positive).
- This is not authoritative ground truth and requires supervisor confirmation.

## Preprocessing and feature engineering methodology

- Raw CSV preserved unchanged; no missing values were detected and duplicate rows were retained.
- Target-derived `ActiveGL` and its near-duplicate `ATVolume`, redundant `MRFlow`, constant columns, scenario-specific depth variables and configured formation pressure were excluded.
- `Flow_Imbalance = FOut - FIn` was added after verifying the documented sign convention (`MRFlow = FIn - FOut`); a relative imbalance was also computed.
- Median imputation and robust scaling were fitted on training rows only.

## Training and validation strategy

- Contiguous 25-row blocks were kept intact using a stratified group split with a purge gap around test blocks.
- Seed: 42. Validation fraction: 0.2; test fraction: 0.2.
- Decision thresholds were tuned on validation data: the midpoint of the validation class-score gap when separable, otherwise the best PR-curve operating point subject to minimum recall. Test data were used only for final reporting.
- The selected supervised model is the one with the highest validation ROC-AUC; ties are broken by validation F1, recall, false-alarm rate and score margin (all without using the test set).
- Class weighting was used for supervised classifiers. The anomaly detector was fitted without labels; the threshold baseline is an experimental rule, not a field threshold.

## Models investigated

- Logistic Regression (interpretable baseline).
- Random Forest (nonlinear interactions).
- XGBoost, or HistGradientBoostingClassifier if XGBoost is unavailable.
- Isolation Forest (mixed training data) and Isolation Forest fitted only on normal training rows; both are unsupervised anomaly detectors evaluated post-hoc.
- Conventional flow-imbalance threshold rule (research comparison only).

## Evaluation methodology

Accuracy, precision, recall, F1, ROC-AUC, PR-AUC, confusion matrices, false positive rate and FAR = FP/(FP+TN). Kick recall, precision, PR-AUC and false alarms are emphasised; accuracy is not used for model selection.

## Observed model comparison (independent test set)

| Model | Family | Precision | Recall | F1 | ROC-AUC | PR-AUC | FAR | Missed kicks | False warnings |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| logistic_regression | supervised | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0 | 0 |
| random_forest | supervised | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0 | 0 |
| xgboost | supervised | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0 | 0 |
| isolation_forest | unsupervised_anomaly | 0.694 | 1.000 | 0.820 | 0.931 | 0.879 | 0.257 | 0 | 77 |
| isolation_forest_normal_fit | unsupervised_anomaly | 1.000 | 0.623 | 0.768 | 1.000 | 1.000 | 0.000 | 66 | 0 |
| conventional_baseline | conventional_rule | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0 | 0 |

Highest supervised test PR-AUC: **random_forest** (1.000); recall 1.000, precision 1.000, 0 missed kicks and 0 false warnings.

## Robustness results

Noise levels are multiples of training-set standard deviation; missingness is MCAR. These are experimental perturbations, not verified sensor specifications.

| Perturbation | Level | Model | Recall | PR-AUC | FAR |
| --- | ---: | --- | ---: | ---: | ---: |
| baseline | 0.0 | logistic_regression | 1.000 | 1.000 | 0.000 |
| baseline | 0.0 | random_forest | 1.000 | 1.000 | 0.000 |
| baseline | 0.0 | xgboost | 1.000 | 1.000 | 0.000 |
| baseline | 0.0 | isolation_forest | 1.000 | 0.879 | 0.257 |
| baseline | 0.0 | isolation_forest_normal_fit | 0.623 | 1.000 | 0.000 |
| baseline | 0.0 | conventional_baseline | 1.000 | 1.000 | 0.000 |
| sensor_noise | 0.01 | logistic_regression | 1.000 | 1.000 | 0.000 |
| sensor_noise | 0.01 | random_forest | 1.000 | 1.000 | 0.000 |
| sensor_noise | 0.01 | xgboost | 1.000 | 1.000 | 0.000 |
| sensor_noise | 0.01 | isolation_forest | 1.000 | 0.847 | 0.347 |
| sensor_noise | 0.01 | isolation_forest_normal_fit | 0.623 | 1.000 | 0.000 |
| sensor_noise | 0.01 | conventional_baseline | 1.000 | 1.000 | 0.000 |
| sensor_noise | 0.03 | logistic_regression | 1.000 | 1.000 | 0.000 |
| sensor_noise | 0.03 | random_forest | 1.000 | 1.000 | 0.000 |
| sensor_noise | 0.03 | xgboost | 1.000 | 1.000 | 0.000 |
| sensor_noise | 0.03 | isolation_forest | 1.000 | 0.839 | 0.403 |
| sensor_noise | 0.03 | isolation_forest_normal_fit | 0.617 | 1.000 | 0.000 |
| sensor_noise | 0.03 | conventional_baseline | 1.000 | 1.000 | 0.000 |
| sensor_noise | 0.05 | logistic_regression | 1.000 | 1.000 | 0.000 |
| sensor_noise | 0.05 | random_forest | 1.000 | 1.000 | 0.000 |
| sensor_noise | 0.05 | xgboost | 1.000 | 1.000 | 0.000 |
| sensor_noise | 0.05 | isolation_forest | 1.000 | 0.835 | 0.420 |
| sensor_noise | 0.05 | isolation_forest_normal_fit | 0.611 | 1.000 | 0.000 |
| sensor_noise | 0.05 | conventional_baseline | 1.000 | 1.000 | 0.000 |
| sensor_noise | 0.1 | logistic_regression | 1.000 | 1.000 | 0.000 |
| sensor_noise | 0.1 | random_forest | 1.000 | 1.000 | 0.000 |
| sensor_noise | 0.1 | xgboost | 1.000 | 1.000 | 0.000 |
| sensor_noise | 0.1 | isolation_forest | 1.000 | 0.828 | 0.457 |
| sensor_noise | 0.1 | isolation_forest_normal_fit | 0.611 | 1.000 | 0.000 |
| sensor_noise | 0.1 | conventional_baseline | 1.000 | 1.000 | 0.000 |
| missing_data | 0.05 | logistic_regression | 0.971 | 0.982 | 0.000 |
| missing_data | 0.05 | random_forest | 1.000 | 1.000 | 0.000 |
| missing_data | 0.05 | xgboost | 0.909 | 1.000 | 0.000 |
| missing_data | 0.05 | isolation_forest | 1.000 | 0.892 | 0.243 |
| missing_data | 0.05 | isolation_forest_normal_fit | 0.451 | 1.000 | 0.000 |
| missing_data | 0.05 | conventional_baseline | 1.000 | 1.000 | 0.000 |
| missing_data | 0.1 | logistic_regression | 0.954 | 0.971 | 0.000 |
| missing_data | 0.1 | random_forest | 1.000 | 1.000 | 0.000 |
| missing_data | 0.1 | xgboost | 0.869 | 1.000 | 0.000 |
| missing_data | 0.1 | isolation_forest | 1.000 | 0.896 | 0.243 |
| missing_data | 0.1 | isolation_forest_normal_fit | 0.394 | 1.000 | 0.000 |
| missing_data | 0.1 | conventional_baseline | 1.000 | 1.000 | 0.000 |
| missing_data | 0.2 | logistic_regression | 0.869 | 0.915 | 0.000 |
| missing_data | 0.2 | random_forest | 0.994 | 1.000 | 0.000 |
| missing_data | 0.2 | xgboost | 0.771 | 1.000 | 0.000 |
| missing_data | 0.2 | isolation_forest | 1.000 | 0.904 | 0.237 |
| missing_data | 0.2 | isolation_forest_normal_fit | 0.303 | 1.000 | 0.000 |
| missing_data | 0.2 | conventional_baseline | 1.000 | 1.000 | 0.000 |

## Explainability findings

SHAP was applied to the selected model. Importance is associational, not causal.

| Rank | Feature | Mean |SHAP| |
| 1 | HLoad | 0.0918 |
| 2 | FOut | 0.0772 |
| 3 | Flow_Imbalance | 0.0705 |
| 4 | WoBit | 0.0674 |
| 5 | FRate | 0.0658 |
| 6 | Flow_Imbalance_Ratio | 0.0495 |
| 7 | MVis | 0.0374 |
| 8 | FDensity | 0.0237 |
| 9 | ATMYP | 0.0098 |
| 10 | ATMPV | 0.0058 |
| 11 | DPPress | 0.0009 |

## Engineering interpretation

- Observed flow-imbalance and pressure changes are consistent with an influx in this simulated scenario; the models learn associations within that scenario.
- Near-perfect supervised and threshold-rule scores are expected because the proxy label comes from pit gain, which is tightly coupled to return-flow imbalance in this simulator. They do not establish performance on field data.
- A low false-alarm rate with poor kick recall is unsafe; a high recall with many false warnings can cause alarm fatigue. The operating point requires engineering judgement.
- No evidence here establishes operational readiness, blowout prevention, or validated warning lead time.

## Limitations

- Single simulated rig scenario; no field validation, no multiple wells or runs.
- No timestamps/event IDs, so early-warning timing cannot be calculated.
- Experimental proxy labels require supervisor confirmation.
- Model importance does not prove physical causation.

## Recommendations for future research

1. Obtain authoritative kick labels and event onset times from drilling supervisors.
2. Collect multiple simulator runs and real field cases for external validation.
3. Implement forward-chaining evaluation with causal rolling features and alert persistence.
4. Calibrate probabilities and quantify the cost of missed kicks versus false alarms.
5. Evaluate physics-informed and sequence models before any operational claims.
