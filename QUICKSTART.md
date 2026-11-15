# GCI Competition - Quick Start Guide for Colab T4

## 🚀 Tomorrow Morning Checklist

### 1. Open Colab & Setup (5 min)
```bash
# In Colab notebook 1 (01_feature_engineering.ipynb):
# Runtime → Change runtime type → GPU (T4)
# Run first 3 cells: imports, Drive mount, load data
```

### 2. Upload Files to Google Drive (Before Colab)
Create this folder structure in your Google Drive:
```
MyDrive/
└── gci_competition/
    ├── input/
    │   ├── train.csv
    │   ├── test.csv
    │   └── sample_submission.csv
    ├── features/          (auto-created)
    ├── models/            (auto-created)
    ├── optuna_studies/    (auto-created)
    └── submissions/       (auto-created)
```

**Upload the 3 CSV files from the competition zip to `MyDrive/gci_competition/input/`**

### 3. Execution Order (Estimated Times on T4)

| Notebook | Time | Description |
|----------|------|-------------|
| `01_feature_engineering.ipynb` | ~15 min | Feature engineering, adversarial validation, save parquet |
| `02_optuna_tuning.ipynb` | ~2-3 hours | 50 trials × 3 models (LightGBM, XGBoost, CatBoost) |
| `03_ensemble_training.ipynb` | ~1-2 hours | 5-fold CV for each model, stacking, SHAP |
| `04_final_submission.ipynb` | ~10 min | Create submission, competition notebook, zip |

**Total: ~4-5.5 hours** (fits in one 12-hour Colab session)

### 4. Colab Pro Tips

#### Keep Session Alive
```javascript
// Paste in browser console (F12) to prevent 90-min timeout:
setInterval(() => document.querySelector('#connect-button').click(), 60000);
```

#### Save Progress Frequently
- Notebooks auto-save to Drive every few minutes
- Manually save: `Ctrl+S` or File → Save
- Copy critical outputs to Drive with `shutil.copy2()`

#### If Session Crashes
1. Reconnect to runtime
2. Re-run Drive mount cell
3. Continue from last completed notebook
4. All parquet/pickle files persist on Drive

#### GPU Memory Issues
```python
# If OOM, reduce batch size in Optuna:
'num_leaves': trial.suggest_int('num_leaves', 16, 128),  # Lower max
'max_depth': trial.suggest_int('max_depth', 3, 8),       # Lower max
```

### 5. Expected Scores (Targets)

| Model | 5-Fold CV AUC Target |
|-------|---------------------|
| LightGBM | > 0.79 |
| XGBoost | > 0.79 |
| CatBoost | > 0.79 |
| **Ensemble** | **> 0.80** |

### 6. Key Files to Submit

1. **`submission_final.csv`** - Your predictions (upload to competition platform)
2. **`improved_lgbm_pipeline_v2.ipynb`** - Competition notebook
3. **`improved_lgbm_pipeline_v2.zip`** - Zipped notebook for honors award

### 7. Troubleshooting

| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError` | Run `!pip install -r requirements.txt` in first cell |
| Drive mount fails | Re-run mount cell, allow permissions |
| Optuna too slow | Reduce trials to 30, use 2-fold CV |
| GPU not detected | Runtime → Change runtime type → GPU (T4) |
| OOM error | Restart runtime, reduce `num_leaves`/`max_depth` |
| 90-min timeout | Use console script above, keep tab active |

### 8. Tomorrow's First Steps

1. **Open Colab**: https://colab.research.google.com/
2. **Upload**: `01_feature_engineering.ipynb` → File → Upload notebook
3. **GPU**: Runtime → Change runtime type → T4 GPU
4. **Drive**: Run mount cell, authorize
5. **Data**: Verify CSV files in `MyDrive/gci_competition/input/`
6. **Run**: Execute cells top to bottom

### 9. Parallel Work (While Optuna Runs)

Since Optuna takes 2-3 hours, you can:
- Open second Colab tab for research/experimentation
- Test pseudo-labeling ideas
- Analyze SHAP values from previous runs
- Prepare competition writeup

---

## 📞 Need Help?
- Check Colab logs for errors
- All intermediate results saved to Drive
- Can resume from any notebook

**Good luck! 🏆**