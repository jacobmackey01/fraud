# Run Status

What was completed in this workspace:

- Built the end-to-end fraud detection project.
- Installed the optional ML stack into `work/vendor_py` for this local run.
- Downloaded the real public credit-card fraud CSV to `data/raw/creditcard.csv`.
- Added cost/capacity thresholding, monitoring notes, leakage cleanup, and a boosted-tree challenger.
- Ran the real dataset with amount-weighted cost thresholding. The validation costs are a near-tie, so the operating model is `unweighted_logistic`; test precision `0.8261`, test recall `0.7600`, test PR-AUC `0.8075`. The report includes the model-selection caveat and near-optimal threshold band.
- Verified the code path with `src/smoke_test.py`.

Smoke-test artifacts, if generated, are only code-validation artifacts. They are not portfolio model results because the smoke fixture is synthetic and deliberately easy.

The raw CSV is intentionally ignored by `.gitignore` for repository use, but it is left in the zip archive for this handoff.

Real run command after the CSV is available:

```powershell
python src/fraud_pipeline.py --no-download
```

Real run command when network access is available:

```powershell
python src/fraud_pipeline.py
```
