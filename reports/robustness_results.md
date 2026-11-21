# Robustness testing

Noise is Gaussian with standard deviation `level x training-set standard deviation` of each listed sensor. Missingness is masked completely at random. Neither level is a verified sensor specification. All perturbations are applied to copies of test rows; models and raw data are unchanged. Seed: 42.

| Model | Perturbation | Level | Recall | PR-AUC | FAR | Missed kicks | False warnings |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| logistic_regression | baseline | 0.0 | 1.000 | 1.000 | 0.000 | 0 | 0 |
| random_forest | baseline | 0.0 | 1.000 | 1.000 | 0.000 | 0 | 0 |
| xgboost | baseline | 0.0 | 1.000 | 1.000 | 0.000 | 0 | 0 |
| isolation_forest | baseline | 0.0 | 1.000 | 0.879 | 0.257 | 0 | 77 |
| isolation_forest_normal_fit | baseline | 0.0 | 0.623 | 1.000 | 0.000 | 66 | 0 |
| conventional_baseline | baseline | 0.0 | 1.000 | 1.000 | 0.000 | 0 | 0 |
| logistic_regression | sensor_noise | 0.01 | 1.000 | 1.000 | 0.000 | 0 | 0 |
| random_forest | sensor_noise | 0.01 | 1.000 | 1.000 | 0.000 | 0 | 0 |
| xgboost | sensor_noise | 0.01 | 1.000 | 1.000 | 0.000 | 0 | 0 |
| isolation_forest | sensor_noise | 0.01 | 1.000 | 0.847 | 0.347 | 0 | 104 |
| isolation_forest_normal_fit | sensor_noise | 0.01 | 0.623 | 1.000 | 0.000 | 66 | 0 |
| conventional_baseline | sensor_noise | 0.01 | 1.000 | 1.000 | 0.000 | 0 | 0 |
| logistic_regression | sensor_noise | 0.03 | 1.000 | 1.000 | 0.000 | 0 | 0 |
| random_forest | sensor_noise | 0.03 | 1.000 | 1.000 | 0.000 | 0 | 0 |
| xgboost | sensor_noise | 0.03 | 1.000 | 1.000 | 0.000 | 0 | 0 |
| isolation_forest | sensor_noise | 0.03 | 1.000 | 0.839 | 0.403 | 0 | 121 |
| isolation_forest_normal_fit | sensor_noise | 0.03 | 0.617 | 1.000 | 0.000 | 67 | 0 |
| conventional_baseline | sensor_noise | 0.03 | 1.000 | 1.000 | 0.000 | 0 | 0 |
| logistic_regression | sensor_noise | 0.05 | 1.000 | 1.000 | 0.000 | 0 | 0 |
| random_forest | sensor_noise | 0.05 | 1.000 | 1.000 | 0.000 | 0 | 0 |
| xgboost | sensor_noise | 0.05 | 1.000 | 1.000 | 0.000 | 0 | 0 |
| isolation_forest | sensor_noise | 0.05 | 1.000 | 0.835 | 0.420 | 0 | 126 |
| isolation_forest_normal_fit | sensor_noise | 0.05 | 0.611 | 1.000 | 0.000 | 68 | 0 |
| conventional_baseline | sensor_noise | 0.05 | 1.000 | 1.000 | 0.000 | 0 | 0 |
| logistic_regression | sensor_noise | 0.1 | 1.000 | 1.000 | 0.000 | 0 | 0 |
| random_forest | sensor_noise | 0.1 | 1.000 | 1.000 | 0.000 | 0 | 0 |
| xgboost | sensor_noise | 0.1 | 1.000 | 1.000 | 0.000 | 0 | 0 |
| isolation_forest | sensor_noise | 0.1 | 1.000 | 0.828 | 0.457 | 0 | 137 |
| isolation_forest_normal_fit | sensor_noise | 0.1 | 0.611 | 1.000 | 0.000 | 68 | 0 |
| conventional_baseline | sensor_noise | 0.1 | 1.000 | 1.000 | 0.000 | 0 | 0 |
| logistic_regression | missing_data | 0.05 | 0.971 | 0.982 | 0.000 | 5 | 0 |
| random_forest | missing_data | 0.05 | 1.000 | 1.000 | 0.000 | 0 | 0 |
| xgboost | missing_data | 0.05 | 0.909 | 1.000 | 0.000 | 16 | 0 |
| isolation_forest | missing_data | 0.05 | 1.000 | 0.892 | 0.243 | 0 | 73 |
| isolation_forest_normal_fit | missing_data | 0.05 | 0.451 | 1.000 | 0.000 | 96 | 0 |
| conventional_baseline | missing_data | 0.05 | 1.000 | 1.000 | 0.000 | 0 | 0 |
| logistic_regression | missing_data | 0.1 | 0.954 | 0.971 | 0.000 | 8 | 0 |
| random_forest | missing_data | 0.1 | 1.000 | 1.000 | 0.000 | 0 | 0 |
| xgboost | missing_data | 0.1 | 0.869 | 1.000 | 0.000 | 23 | 0 |
| isolation_forest | missing_data | 0.1 | 1.000 | 0.896 | 0.243 | 0 | 73 |
| isolation_forest_normal_fit | missing_data | 0.1 | 0.394 | 1.000 | 0.000 | 106 | 0 |
| conventional_baseline | missing_data | 0.1 | 1.000 | 1.000 | 0.000 | 0 | 0 |
| logistic_regression | missing_data | 0.2 | 0.869 | 0.915 | 0.000 | 23 | 0 |
| random_forest | missing_data | 0.2 | 0.994 | 1.000 | 0.000 | 1 | 0 |
| xgboost | missing_data | 0.2 | 0.771 | 1.000 | 0.000 | 40 | 0 |
| isolation_forest | missing_data | 0.2 | 1.000 | 0.904 | 0.237 | 0 | 71 |
| isolation_forest_normal_fit | missing_data | 0.2 | 0.303 | 1.000 | 0.000 | 122 | 0 |
| conventional_baseline | missing_data | 0.2 | 1.000 | 1.000 | 0.000 | 0 | 0 |

Full numerical results: `results/robustness_results.csv`.
