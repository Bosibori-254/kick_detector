# Model evaluation report

> Experimental research prototype. Labels are proxies derived from a continuous pit-gain signal; see `reports/data_audit.md`.

## Test-set comparison

| Model | Family | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | FAR | TP | FP | FN | TN |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| logistic_regression | supervised | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 175 | 0 | 0 | 300 |
| random_forest | supervised | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 175 | 0 | 0 | 300 |
| xgboost | supervised | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 175 | 0 | 0 | 300 |
| isolation_forest | unsupervised_anomaly | 0.838 | 0.694 | 1.000 | 0.820 | 0.931 | 0.879 | 0.257 | 175 | 77 | 0 | 223 |
| isolation_forest_normal_fit | unsupervised_anomaly | 0.861 | 1.000 | 0.623 | 0.768 | 1.000 | 1.000 | 0.000 | 109 | 0 | 66 | 300 |
| conventional_baseline | conventional_rule | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 175 | 0 | 0 | 300 |

Decision thresholds were tuned on a validation split only; the test set was used once for reporting. Where validation scores were separable, the threshold was the midpoint between the largest normal score and the smallest kick score; otherwise the PR-curve metric was optimised. Operating points were constrained to at least 90% validation kick recall. FAR is defined as FP/(FP+TN), identical numerically to the false positive rate in this binary setting. PR-AUC is emphasised because the kick class is the minority and missed kicks are more serious than false warnings.

## Engineering interpretation

- Highest supervised PR-AUC: **random_forest** (1.000). PR-AUC, not accuracy, is the selection criterion.
- Missed kicks (false negatives) for that model: 0; false warnings: 0.
- A high recall with many false warnings may be preferable during investigation, but a high false-alarm count can cause alarm fatigue. The operating threshold must be chosen with drilling personnel, not from accuracy alone.
- Both Isolation Forest variants are unsupervised anomaly detectors; one is fitted on mixed training blocks and one only on normal training rows. Their agreement with the proxy labels is a post-hoc comparison, not supervised classification.
- The conventional rule uses an experimental flow-imbalance threshold tuned on validation data. It is not an accepted field threshold.
- Near-perfect scores must be interpreted cautiously: the proxy label is derived from a pit-gain signal that is physically coupled to flow imbalance in this simulator. Separability is therefore expected and is not evidence of real-world detection performance.
- These results come from one simulated rig scenario. They do not demonstrate readiness for real operations or validated early warning.

## Figures

- `results/figures/confusion_matrices.png`
- `results/figures/roc_curves.png`
- `results/figures/precision_recall_curves.png`
