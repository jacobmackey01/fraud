# Fraud Detection Model Report

Generated: 2026-06-19T15:01:23+00:00

## Dataset

Source: https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv

This project targets the classic credit-card fraud dataset: anonymized PCA transaction features (`V1` through `V28`), `Time`, `Amount`, and a binary `Class` label.

| split | rows | legitimate | fraud | fraud_rate |
| --- | --- | --- | --- | --- |
| all | 284807 | 284315 | 492 | 0.001727 |
| train | 170884 | 170524 | 360 | 0.002107 |
| validation | 56961 | 56904 | 57 | 0.001001 |
| test | 56962 | 56887 | 75 | 0.001317 |

## Modeling Approach

- Split strategy: `chronological`. The default chronological split is closer to production monitoring than a random split because future transactions are held out from earlier training data.
- Feature engineering: `log1p(Amount)`, zero-amount indicator, and hour-of-day sine/cosine, alongside the anonymized PCA features. Raw time and day index are deliberately excluded from the model to avoid learning a split artifact from this short two-day dataset.
- Imbalance handling: the pipeline compares an unweighted logistic baseline, a class-weighted logistic model, and an optional boosted-tree challenger when scikit-learn is installed.
- Thresholding: the operating threshold is selected on the validation set with policy `cost`. The default cost policy uses illustrative false-positive cost `8.00` and amount-weighted missed-fraud cost `Amount + 35.00`. The test set is used only after that threshold is chosen. In production, fraud operations, risk/finance, product, and data science should jointly own these assumptions.
- Primary metrics: precision, recall, F1, F2, average precision / PR-AUC, ROC-AUC, false positive rate, flagged rate, and expected operating cost. Accuracy is reported only to show why it is a poor fraud metric.

## Model Narrative

- Logistic models are kept as interpretable baselines; the optional boosted-tree challenger reflects a common fraud-team production pattern.
- HistGradientBoostingClassifier internal early stopping is disabled so the chronological validation split remains the only validation boundary.
- Validation expected-cost spread is treated as a model-selection tie when models are within 5% of the best validation cost. Near-tied models here: unweighted_logistic, class_weighted_logistic, hist_gradient_boosting.
- `class_weighted_logistic` has the numerical validation-cost minimum, but `unweighted_logistic` is selected as the operating model because validation cannot separate the near-tied candidates and the baseline is easier to audit.
- Model selection uses validation expected cost over 57 validation frauds and 75 test frauds; close rankings should be treated as noisy rather than proof of model superiority.
- Amount-weighted missed-fraud cost is economically better than a flat false-negative penalty, but it increases variance because a few high-value frauds can move the validation cost ranking.

## Accuracy Trap

An "always legitimate" classifier gets **0.9987 accuracy** on the test split while catching **0.0000** of fraud. That is why this project optimizes precision/recall behavior instead of accuracy.

## Model Comparison

| model | split | threshold | precision | recall | f1 | f2 | average_precision | roc_auc | flagged_rate | false_positive_rate | missed_fraud_cost | expected_cost | cost_savings_vs_no_model |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unweighted_logistic | validation | 0.0537 | 0.8776 | 0.7544 | 0.8113 | 0.7762 | 0.7718 | 0.9734 | 0.0009 | 0.0001 | 4965.0900 | 5013.0900 | 9213.0000 |
| unweighted_logistic | test | 0.0537 | 0.8261 | 0.7600 | 0.7917 | 0.7724 | 0.8075 | 0.9804 | 0.0012 | 0.0002 | 3032.0100 | 3128.0100 | 7226.2500 |
| class_weighted_logistic | validation | 0.9510 | 0.9556 | 0.7544 | 0.8431 | 0.7875 | 0.7794 | 0.9754 | 0.0008 | 0.0000 | 4965.0900 | 4981.0900 | 9245.0000 |
| class_weighted_logistic | test | 0.9510 | 0.9273 | 0.6800 | 0.7846 | 0.7183 | 0.7878 | 0.9840 | 0.0010 | 0.0001 | 4841.9500 | 4873.9500 | 5480.3100 |
| hist_gradient_boosting | validation | 0.9716 | 0.9149 | 0.7544 | 0.8269 | 0.7818 | 0.7798 | 0.9756 | 0.0008 | 0.0001 | 4964.8600 | 4996.8600 | 9229.2300 |
| hist_gradient_boosting | test | 0.9716 | 0.8983 | 0.7067 | 0.7910 | 0.7382 | 0.7701 | 0.9798 | 0.0010 | 0.0001 | 3736.7700 | 3784.7700 | 6569.4900 |

## Champion

Champion: `unweighted_logistic`

- Validation precision: 0.8776
- Validation recall: 0.7544
- Validation expected cost: 5013.09
- Test precision: 0.8261
- Test recall: 0.7600
- Test average precision / PR-AUC: 0.8075
- Test expected cost: 3128.01
- Alert threshold: 0.053669
- Near-optimal validation cost band: thresholds 0.032815 to 0.114460 stay within 5% of selected validation cost, with precision 0.4945-0.9535 and recall 0.7193-0.7895.

Cost alone does not fully determine the operating point when the cost curve is flat. The final threshold should be pinned with `--min-precision` or `--threshold-policy capacity` once fraud operations confirms the review budget and acceptable customer-friction level.

## What Drives Flags

For logistic models, driver values are standardized coefficients: positive values push a transaction toward a fraud flag and negative values push it away. For the boosted-tree challenger, driver values are validation PR-AUC drops from permutation importance. The `V*` fields are anonymized PCA components, so they are useful for model debugging but are not human business concepts like "merchant category" or "cardholder velocity."

| feature | driver_value | driver_type | interpretation |
| --- | --- | --- | --- |
| V14 | -0.1415 | standardized_logistic_coefficient | higher values lower fraud score |
| V12 | -0.1087 | standardized_logistic_coefficient | higher values lower fraud score |
| V17 | -0.0966 | standardized_logistic_coefficient | higher values lower fraud score |
| V10 | -0.0843 | standardized_logistic_coefficient | higher values lower fraud score |
| V16 | -0.0629 | standardized_logistic_coefficient | higher values lower fraud score |
| V11 | 0.0627 | standardized_logistic_coefficient | higher values raise fraud score |
| V4 | 0.0611 | standardized_logistic_coefficient | higher values raise fraud score |
| V3 | -0.0542 | standardized_logistic_coefficient | higher values lower fraud score |
| V7 | -0.0457 | standardized_logistic_coefficient | higher values lower fraud score |
| V9 | -0.0376 | standardized_logistic_coefficient | higher values lower fraud score |
| V8 | -0.0255 | standardized_logistic_coefficient | higher values lower fraud score |
| V2 | 0.0254 | standardized_logistic_coefficient | higher values raise fraud score |

For the selected linear operating model, local transaction-level `x_i * w_i` log-odds contributions are written to `interpretation/linear_log_odds_contributions_top_flags.csv`. These are not customer-facing reason codes; real reason codes require business features such as merchant category, cardholder velocity, device reputation, and chargeback history.

High-value false positives are written to `interpretation/high_value_false_positive_examples.csv`. These are useful monitoring cases because they turn false-positive-rate discussion into customer-friction and segment-risk review, not just a scalar metric.

## Artifacts

- `artifacts/fraud_model.json`: weights, preprocessing statistics, selected threshold, and metrics.
- `reports/metrics_summary.csv`: validation/test metrics for each model.
- `reports/precision_recall_curve.svg`: validation precision-recall curve for the champion.
- `interpretation/global_feature_drivers.csv`: global feature drivers, using signed standardized coefficients for logistic models or permutation PR-AUC drop for the boosted-tree model.
- `interpretation/linear_log_odds_contributions_top_flags.csv`: exact local log-odds contributions for the highest-risk test transactions scored by the selected linear operating model.
- `interpretation/high_value_false_positive_examples.csv`: flagged legitimate high-value transactions to inspect for customer-friction and segment-level false-positive monitoring.

Runtime: 42.7 seconds.
