# GCI In-Class Competition — HomeCredit Default Risk

Solution pipeline for the GCI in-class competition (due Nov 20), based on the
[HomeCredit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk) dataset.

**Metric:** ROC AUC — 5-fold CV AUC target **> 0.80** for the ensemble.

---

## Pipeline

| # | Notebook | Runtime (Colab T4) | Description |
|---|----------|--------------------|-------------|
| 1 | `01_feature_engineering.ipynb` | ~15 min | Feature engineering, adversarial validation, saves parquet |
| 2 | `02_optuna_tuning.ipynb` | ~2–3 h | 50 trials x 3 models (LightGBM, XGBoost, CatBoost) |
| 3 | `03_ensemble_training.ipynb` | ~1–2 h | 5-fold CV per model, stacking, SHAP analysis |
| 4 | `04_final_submission.ipynb` | ~10 min | Builds `submission_final.csv` + competition notebook |

Total: **~4–5.5 h**, which fits in a single 12-hour Colab session.
See [`QUICKSTART.md`](QUICKSTART.md) for the full runbook (Drive layout, keepalive tip, troubleshooting).

### Models
- **LightGBM** 4.3.0
- **XGBoost** 2.1.0
- **CatBoost** 1.2.3
- **Stacked ensemble** of the three above

---

## Getting the data

The raw `train.csv` / `test.csv` are **not** committed (they are ~60MB).
Place them under the competition input folder:

```
competition_unzipped/
└── 04. Competition (due Nov 20)/
    └── Competition/
        └── input/
            ├── train.csv        <- not tracked by git
            ├── test.csv         <- not tracked by git
            └── sample_submission.csv
```

In Colab, upload them to Drive at `MyDrive/gci_competition/input/` instead.

### Install

```bash
pip install -r requirements.txt
```

---

## Outputs

- `submission_final.csv` — predictions for upload to the competition platform
- `improved_lgbm_pipeline_v2.ipynb` — competition notebook
- `improved_lgbm_pipeline_v2.zip` — zipped notebook for the honors award

---

## Repo layout

```
.
├── 01_feature_engineering.ipynb
├── 02_optuna_tuning.ipynb
├── 03_ensemble_training.ipynb
├── 04_final_submission.ipynb
├── QUICKSTART.md
├── requirements.txt
└── competition_unzipped/
    └── 04. Competition (due Nov 20)/
        ├── competition_tutorial.pdf
        └── Competition/
            ├── README.ipynb
            ├── tutorial.ipynb
            ├── HomeCredit_columns_description.xlsx
            └── input/sample_submission.csv
```
