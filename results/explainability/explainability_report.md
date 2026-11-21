# SHAP explainability report

- Model explained: `random_forest` (selected on validation PR-AUC/ROC-AUC).
- Explainability method: `TreeExplainer`.
- Test observations explained: 475; example row: 300.

## Mean absolute SHAP magnitude (test set)

| Rank | Feature | Mean |SHAP| |
| ---: | --- | ---: |
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
| 12 | RoPen | 0.0008 |

## Engineering interpretation

Features with large SHAP magnitude strongly influence the model output on this test set. For a kick, a positive shift is physically consistent with increasing return flow, pit gain, or changing pressure/force readings. SHAP importance is associational: it does not prove that a variable physically causes a kick.

Because `ActiveGL` and `ATVolume` were excluded as target-derived variables, the conclusions concern precursor-like measurements such as flow imbalance, pressure and mechanical response, not the label itself.

## Outputs

- `shap_summary.png`
- `shap_feature_importance.png`
- `shap_dependence_HLoad.png`
- `shap_individual_explanation.png`
