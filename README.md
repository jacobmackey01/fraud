# Credit Card Fraud Detection Project

This is an end-to-end fraud detection project for the classic public credit-card fraud dataset. It trains imbalance-aware models, chooses an alert threshold from business cost or alert-capacity assumptions, and writes interpretation artifacts that explain what pushes transactions toward a fraud flag.

## Dataset

The project uses the Kaggle/ULB credit-card fraud dataset mirrored by TensorFlow for its imbalanced-data tutorial:

`https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv`

Expected schema:

- `Time`
- `V1` through `V28`
- `Amount`
- `Class`, where `1` is fraud

The dataset is extremely imbalanced: the commonly cited version has 492 frauds out of 284,807 transactions.

For this workspace's execution status, see `RUN_STATUS.md`.

## Quick Start

From this folder:

```powershell
python src/fraud_pipeline.py
```

If network access is blocked, place `creditcard.csv` at `data/raw/creditcard.csv`, then run:

```powershell
python src/fraud_pipeline.py --no-download
```

For a local code sanity check that does not use the real dataset:

```powershell
python src/smoke_test.py
```

## Optional ML Packages

The checked-in pipeline is runnable with NumPy and Pandas only, then automatically adds the boosted-tree challenger when scikit-learn is available. On a normal machine, install the usual ML stack with:

```powershell
python -m pip install -r requirements-full.txt
```

When scikit-learn is installed, the pipeline automatically adds a `HistGradientBoostingClassifier` challenger. The logistic models remain as interpretable baselines.

## What The Pipeline Does

1. Downloads or loads the public transaction CSV.
2. Validates the expected fraud schema.
3. Engineers lightweight transaction features:
   - `log1p(Amount)`
   - zero-amount flag
   - hour-of-day sine/cosine
4. Uses a train/validation/test split. The default is chronological, which better resembles production monitoring than a random split.
5. Trains two logistic models:
   - unweighted baseline
   - class-weighted model for the extreme imbalance
6. Adds a boosted-tree challenger when scikit-learn is available.
7. Selects the fraud alert threshold on validation data using a cost policy by default:
   - false positive cost: analyst review / customer friction
   - false negative cost: transaction `Amount` plus an illustrative handling cost
   - optional capacity policy: cap alerts to the review budget
8. Reports test precision, recall, F1, F2, PR-AUC / average precision, ROC-AUC, false positive rate, flagged rate, and expected cost.
9. Writes interpretation outputs:
   - global feature drivers
   - local linear log-odds contributions for high-risk transactions

## Why Accuracy Is Not The Main Metric

Fraud is rare enough that an "always legitimate" classifier can look excellent by accuracy while catching no fraud. This project reports accuracy only as a cautionary baseline and uses precision, recall, F-scores, PR-AUC, and false positive rate for model selection.

## Verified operating point

[Validation precision–recall curve and untouched-test operating point](reports/fraud_operating_point.svg)

*The threshold was selected on chronological validation data and applied once to the untouched chronological test set: 75 frauds across 56,962 transactions. Costs remain illustrative assumptions and no real fraud-review capacity was supplied. Final production threshold ownership belongs jointly to fraud operations, risk/finance, product and data science. V1–V28 are anonymised PCA components, not customer-facing reason codes; this is a portfolio case study, not a deployed bank fraud system.*

## Outputs

After a real run, inspect:

- `reports/model_report.md`
- `reports/metrics_summary.csv`
- `reports/fraud_operating_point.svg`
- `reports/precision_recall_curve.svg`
- `reports/feature_drivers.svg`
- `artifacts/fraud_model.json`
- `interpretation/global_feature_drivers.csv`
- `interpretation/linear_log_odds_contributions_top_flags.csv`
- `interpretation/high_value_false_positive_examples.csv`
- `MONITORING.md`

The operating-point figure is regenerated from the committed report artifacts with:

```powershell
python src/generate_operating_point_figure.py
```

## Interpretation Note

The `V1`-`V28` fields are PCA-anonymized, so they are valid statistical drivers but not direct business concepts. The local contribution file is therefore not a customer-facing reason-code system. In a production bank setting, the same structure would be enriched with explainable fields such as merchant category, merchant country, cardholder velocity, account age, device reputation, and historical chargeback patterns.

## Repository Note

The raw CSV is included in the delivered zip for convenience, but `data/raw/` is ignored by Git because `creditcard.csv` is larger than GitHub's normal file-size limit. A portfolio repo should rely on the download command or local placement instructions rather than committing the raw dataset.
