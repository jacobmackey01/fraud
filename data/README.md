# Data

The main project expects the classic credit-card fraud CSV at:

`data/raw/creditcard.csv`

The pipeline can download a public no-login copy from:

`https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv`

If the runtime blocks network access, download the file separately and place it at the raw path, then run:

```powershell
python src/fraud_pipeline.py --no-download
```

The smoke test creates `data/sample/smoke_transactions.csv` only to verify that the code path works. Do not treat the smoke fixture as a real fraud dataset.
