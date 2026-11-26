# `pipeline/` — historical build evidence

These are **exact copies** of the scripts, tuned parameter files, and blend recipes that
produced the v3/v4 artifacts, taken from the development session workspace
(`%TEMP%\opencode\`) on 2026-11-18. They are kept so the audit trail does not depend on a
Windows temp cleanup. They are **not** a packaged CLI.

The authoritative, self-contained reproduction is [`../gci_honors_reproduction.ipynb`](../gci_honors_reproduction.ipynb).
It rebuilds features, members, and the blend from the raw competition CSVs on CPU.

## What is here

| File | Purpose |
|---|---|
| `bench_members2.py` | Trains and evaluates the E2 member candidates with fold-local target encoding; writes member OOF arrays. |
| `bench_test_preds.py` | Refits the frozen members on the training folds, writes test predictions, and assembles the submission. |
| `rebuild_v3.py` | Rebuilds the v3 recipe end-to-end (CV **0.76224**). |
| `rebuild_v4_members.py` | Builds the new-tuned CatBoost and HGB members (OOF + test) used by v4. |
| `hgb_fix.py` | Rebuilds HGB with the corrected parameter set (`max_iter`, early stopping, validation fraction); produces its OOF/test arrays. |
| `overnight_retune.py` | Optuna tuning for LightGBM and XGBoost on E2. |
| `overnight_retune2.py` | Optuna tuning for CatBoost and HGB on E2. |
| `best_params_{lgb,xgb,cat,hgb}_e2.json` | Tuned parameters actually used by v4. |
| `best_val_*_e2.txt` | Best Optuna objective value for each family. |
| `bench_winner_params.json` | Earlier E2 LightGBM winner parameters (v2 era). |
| `v3_recipe.json` | v3 member weights and OOF (0.762242). |
| `v4_recipe.json` | v4 member weights, OOF (0.76244), and member descriptions. |
| `ENVIRONMENT.txt` | Exact library versions that produced the artifacts. |

## Hardcoded paths (read before running)

The scripts retain the original absolute Windows paths:

```python
BASE = r"C:\Users\amanuel\AppData\Local\Temp\opencode\drive\GCI Competition"
F    = BASE + r"\features"
T    = r"C:\Users\amanuel\AppData\Local\Temp\opencode"
```

They expect:

```
%TEMP%\opencode\
├── X_e2.parquet                  # 171,202 × 98 training matrix
├── X_test_e2.parquet             # 61,500 × 98 test matrix
└── drive\GCI Competition\features\
    ├── train_engineered.parquet
    ├── test_engineered.parquet
    ├── y_train.npy
    ├── feature_cols.json
    └── high_card_cols.json       # columns that receive target encoding
```

Also expected: member OOF/test `.npy` files from the original run (not in git because of
size). To rerun from raw data instead, first produce the feature matrices with
`01_feature_engineering.ipynb` (or the Honors notebook's feature cells), then point `BASE/T/F`
at the new location.

## Provenance and determinism

- Folds: `StratifiedKFold(5, shuffle=True, random_state=42)` everywhere.
- Target encoding: fold-train means only; validation/test mapped with those means.
- LightGBM is bit-identical across thread counts. **XGBoost `hist` and CatBoost are not**:
  on fold 0, XGBoost `n_jobs=8` (the shipped v4 run) gave AUC 0.759029 / 2,134 trees vs
  `n_jobs=2` 0.759578 / 2,250 trees (corr 0.996); CatBoost `thread_count=2` vs `8` gave
  0.758955 vs 0.759364 (corr 0.9936). That is why a rerun reproduces the recipe and CV within
  ~±0.004, not the CSV byte-for-byte. The shipped `submissions/submission_v4_cv0.76244.csv`
  is authoritative.

## Audit scripts

`audit/` contains the checks described in [`../AUDIT_FINDINGS.md`](../AUDIT_FINDINGS.md):

| Script | Check |
|---|---|
| `audit_deep.py` | Rebuild v4/v3, bootstrap, correlations, submission cross-checks, weight sensitivity |
| `audit_honors_fe.py` | Execute Honors feature cells; diff against `X_e2` / `X_test_e2` |
| `audit_data.py` | ID/order/duplicate checks, adversarial validation |
| `audit_extra.py` | HGB contribution, CatBoost per-fold, submission correlations, enrichment means |
| `audit_threads.py` | Fold-0 LightGBM/XGBoost/CatBoost thread determinism |
| `thread_test_output.txt` | Saved output of `audit_threads.py` (the numbers cited in AUDIT_FINDINGS §C2) |
| `run_nb1.py` | Execute `01_feature_engineering.ipynb` headlessly; diff its parquet output |

They also keep the `%TEMP%` paths of the development session; rewrite the same `BASE/F/T`
constants before re-running.
