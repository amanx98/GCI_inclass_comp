# GCI In-Class Competition — HomeCredit Default Risk

Solution pipeline for the GCI in-class competition (**due Nov 20, 01:00 UTC**), based on the
[HomeCredit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk) dataset.

**Metric:** ROC AUC · **Data:** 171,202 train rows × 101 engineered features, 61,500 test rows
**Compute:** Google Colab Free (T4 GPU) + local Ryzen 7 5800U (8c/16t) for CPU experiments

---

## 📊 Current Status

| Item | Value |
|---|---|
| **Current public LB** | **0.7627** — from `submission_anchor_blend_v1.csv` (✅ uploaded) |
| Previous dummy submission | 0.66819 (tutorial RF, 5 features) — replaced |
| **Verified true CV (full 5-fold OOF)** | **0.7587** for the anchor blend |
| CV → LB calibration | 0.7587 CV → 0.7627 LB (**+0.004**, slightly optimistic but usable) |
| **Best recipe built (not yet submitted)** | **CV 0.76203** — `submission_v2_recipe_cv0.76203.csv` |
| Submission rules | Unlimited submissions, **last file counts**, public LB is a subset of test |
| Time left | ~4 days (deadline Nov 20) |

### Verified model results (5-fold Stratified OOF, full 171,202 rows)

| Model / Blend | OOF AUC | Note |
|---|---|---|
| LightGBM (Optuna-tuned, depth 3) | 0.75822 | retrained on full data; saved original was 0.75546 |
| XGBoost (tuned) | 0.76137 | **best single model** |
| CatBoost (tuned) | 0.76033 | |
| HistGradientBoosting (sklearn) | 0.75588 | added for diversity, rank-corr ~0.957 vs LGB |
| LGB ×3 seeds averaged | 0.75989 | |
| **Anchor blend** (LGB .30 / XGB .55 / CAT .15) | **0.7587** | submitted → LB 0.7627 |
| **Final recipe v2** (LGB3 .06 / XGB3 .64 / CAT .30 / HGB .00) | **0.76203** | **+0.0033 over anchor — submit this next** |

> **Note on the original pipeline:** the previous 3-GBM blend looked like it gained a lot over its
> best member (+0.0002). The real gain is that the three GBMs are **97–99% rank-correlated** —
> they are nearly the same model, so diversity is the binding constraint, not tuning.

---

## 📈 Score Ladder

| Stage | CV AUC | Est. LB | Status |
|---|---|---|---|
| Dummy RF (tutorial, 5 features) | ~0.66 | 0.668 | ✅ submitted, superseded |
| Anchor blend | 0.7587 | **0.7627** | ✅ **submitted and verified** |
| NB3 v2 recipe | 0.76203 | ~0.757–0.762 | ✅ built, **awaiting upload** |
| + XGB/CAT seed-averaging, finer grid | 0.762–0.763 | ~0.758–0.763 | ⏳ in progress |
| Day-3/4 extras (re-tune on GPU, pseudo-labeling) | 0.763–0.766 | ~0.759–0.762 | ⏳ planned |

**Realistic ceiling for this dataset: ~0.775 CV → ~0.770 LB.** The application table only
(171k rows, no auxiliary tables) is a narrow signal pool. The #1 entry on the public LB at
**1.000** is a known leak/hand-labeling artifact and is explicitly not a target.

---

## 🧭 Process & Key Decisions

This section records *why* the pipeline looks the way it does, so decisions aren't silently
re-litigated. The full experiment log with every benchmark lives in
[`MASTER_PLAN.md`](MASTER_PLAN.md); the inter-agent status file is [`HANDOFF_TO_GLM.md`](HANDOFF_TO_GLM.md).

### 1. Audit-first: we found and fixed 3 real bugs before submitting

The original notebooks (NB1–NB4) were built and then **audited cell-by-cell** against the
competition PDF and the saved Drive artifacts. Three genuine bugs were found:

| # | Bug | Impact | Fix |
|---|---|---|---|
| **1** | NB3 cell 12 "retrain on full data" used XGBoost's `num_boost_round=lgb_final.best_iteration_` (LightGBM's iteration count!) and a hard-coded 1000 iters for CatBoost | Produced `final_full_preds.npy` with a **squashed distribution** (0.346–0.777, std 0.099 vs the correct blend's 0.213); corr with the good blend only 0.981. NB4 loaded this file **by default**, so running NB4 as-is would have submitted the worse variant | Cell 12 **removed from the submission path entirely**. Final submission is now always the **CV-fold-averaged** test predictions, never a one-shot retrain |
| **2** | Blend strategy was selected on a **10% holdout** (6.1k rows → ±0.005 noise) | Reported holdout AUC 0.76502 was **inflated by ~+0.006**; the true value is 0.7587 | Selection moved to **full OOF** via hill-climb + exhaustive weight grid. The holdout is retained only as a sanity check and **no selection decision leans on it** |
| **3** | Holdout target encoding was computed from the full training set | Leakage-adjacent | Kept as sanity check only, same as above |

**Decision rule that came out of bug #1 and now governs everything:**
> *Final submission = CV-fold-averaged test predictions, never one-shot retrains.*

A single full-data model is not the same object the blend weights were calibrated on, and
that mismatch costs AUC.

### 2. Selection discipline: CV decides, LB only sanity-checks

Five standing rules govern every choice in this repo:

1. **CV decides everything.** The public LB is a *subset* of test and is mildly optimistic
   (observed +0.004). We never pick a recipe because it scored well on the public LB.
2. **Never select on the holdout.** Select on full OOF. The holdout is a final sanity check.
3. **Every artifact is saved** — per-seed OOF predictions, test predictions, the recipe JSON,
   and an experiment log. Reproducibility is also an honors-award requirement.
4. **Fixed seeds everywhere**, library versions recorded.
5. **One submission per validated recipe.** Unlimited submissions ≠ free probing; overfitting
   the public subset is exactly how people drop on the private LB.

### 3. Verified locally before spending GPU: drop the holdout, don't go deeper

All tuning decisions were validated on the local Ryzen 7 (LightGBM 4.7 CPU, 5-fold, full 171k)
before touching Colab time.

| Experiment | Config | OOF AUC | Conclusion |
|---|---|---|---|
| **Holdout removal** | Optuna-tuned depth-3 params + TE, CV on full 171k | **0.75822** | vs 0.75546 for the same params/folds trained on the 90% subset → **+0.0028 free** just from dropping the holdout carve-out |
| **Depth** | leaves=63 / leaves=127, deeper | 0.75546 / 0.75580 | Deeper trees **lose**. The signal is saturated for plain GBMs → gains must come from **new features** and **model diversity**, not depth |
| **FE: EXT×AGE + NaN-safe EXT poly + burden ratios + NAN_COUNT** | +17 features | 0.75881 (+0.00059) | Keep — modest but real |
| **FE: categorical-combo TE (fold-safe)** | +4 combo TEs | 0.75920 | Keep |
| **FE: drop 18 dead features** | no combo TE | 0.75963 | Keep (feature pruning helps) |
| **Diversity: HGB / LogReg / RF as 4th member** | HGB 0.75582, LogReg 0.74213, RF 0.73687 | — | Only **near-equal-strength** members earn a seat: LGB+HGB 0.65/0.35 = 0.75937 (**+0.0012**), but *adding* LogReg or RF **hurts** (0.75703 / 0.75499). LogReg meta-stacking over 4 members = 0.75604 (flat stacking dilutes) |

**Decision:** the Optuna depth-3 parameters are *correct* and were kept. The plan is to add
signal and diversity, not capacity.

### 4. Feature engineering is validated batch-by-batch, not by intuition

Round 2 features were ranked by expected value and each batch was gated on measured ΔOOF:

- **Tier A (dominant signal):** `EXT_SOURCE_1/2/3 × DAYS_BIRTH` age interactions, NaN-safe
  `EXT_x * EXT_y` products and `(EXT_x - EXT_y)` diffs (fill NaN with the column mean **for the
  interaction only**, plus an `EXT_x_MISS & EXT_y_MISS` indicator), degree-2 polynomials on the
  three EXT_SOURCE columns, `EXT_SOURCES_PROD_NONAN`.
- **Tier B (financial ratios):** `AMT_ANNUITY / AMT_CREDIT`, `AMT_CREDIT / AMT_INCOME_TOTAL`,
  `AMT_CREDIT / (CNT_FAM_MEMBERS+1)`, `AMT_INCOME_TOTAL / (CNT_CHILDREN+1)`,
  `AMT_ANNUITY / (|DAYS_EMPLOYED|/365 + 1)` (payment burden vs. work history),
  `CREDIT_TERM × REGION_RATING_CLIENT`, plus row-level `NAN_COUNT`.
- **Tier C (categorical combos, fold-safe TE, smoothing=20):**
  `OCCUPATION_TYPE × REGION_RATING_CLIENT`, `ORGANIZATION_TYPE × REGION_RATING_CLIENT`,
  `NAME_INCOME_TYPE × NAME_EDUCATION_TYPE`.

**Gate:** one quick 5-fold LGBM per batch; keep a batch only if **ΔOOF ≥ +0.0003**. Every
experiment is logged in `MASTER_PLAN.md` §8.

### 5. The final blend recipe was searched exhaustively, not hand-picked

With the enriched E2 feature set and full-data members (LGB×3 seeds, XGB×3 seeds, CAT×2,
HGB), **all pairs and triples were searched exhaustively at 0.01 weight resolution**:

> **`LGB3 0.06 + XGB3 0.64 + CAT 0.30 + HGB 0.00` → CV 0.76203**
> (stable across search resolutions: coarse grid 0.76194, hill-climb 0.76186, fine grid ~0.76203)

HGB earns **zero weight** once XGB and CatBoost are at full strength on the new features — the
two strongest GBMs dominate. `03_ensemble_training_v2.ipynb` re-selects the recipe on OOF at
runtime, so it adapts if member strengths shift on Colab hardware.

### 6. Optuna's tuning was mis-specified — re-tuning is scheduled for Nov 17

Audit finding: **NB2's Optuna objective never applied target encoding** ("simplified version —
full implementation in notebook 3"). The tuned parameters were therefore selected on a feature
space *without* the `_TE` columns that NB3 actually trains with, and NB2's reported AUCs are
**not comparable** to NB3's.

**Decision:** the Nov 17 re-tune must use the **same fold-level TE as NB3**, with a **5-fold OOF
objective** (not 3-fold — selection noise matters here), 30 trials × 3 models.

Also noted: NB2's `MedianPruner` never actually pruned (`trial.report(cv_score, step=0)` reports
once per trial, so there are no intermediate values to prune on). Harmless, but the re-run ignores
pruning rather than pretending it works.

### 7. The honors notebook has to actually reproduce the submission

The original "polished notebook" was just a copy of NB4 with cleared outputs — it loads `.npy`
files from Drive, so it **does not reproduce the submitted CSV end-to-end**. The competition PDF
confirms the Honors/Outstanding track requires submitted code.

**Decision:** build `gci_honors_reproduction.ipynb` as a **single self-contained notebook**:
raw CSVs → feature engineering → 4 members (hardcoded tuned params, seeds LGB×3, XGB×3, CAT×2)
→ OOF weight grid → `submission.csv` → zip for the Honors upload. A citation list is included for
any referenced public code.

### 8. Data integrity checks that closed off exploitation paths

The five `FLAG_*` columns and the renumbered IDs were confirmed against
`HomeCredit_columns_description.xlsx` and by direct data inspection: **no hidden columns to
exploit**. The signal genuinely lives in feature engineering, so that is where the effort goes.

---

## 🗂 Files & Notebooks

| File | Role | Status |
|---|---|---|
| `01_feature_engineering.ipynb` | Feature engineering + adversarial validation, saves parquet | ✅ working (cell 12 syntax error fixed) |
| `02_optuna_tuning.ipynb` | 50 trials × 3 models | ⚠️ **stale** — tuned without TE; re-tune scheduled Nov 17 |
| `03_ensemble_training.ipynb` | Original 5-fold ensemble + stacking + SHAP | ⚠️ superseded by v2 (contains bugs 1–3) |
| **`03_ensemble_training_v2.ipynb`** | **Evidence-based rebuild.** Seed-averaged members, 4th member HGB, OOF hill-climb + 4-way grid, self-selecting recipe | ✅ **primary pipeline** |
| **`gci_honors_reproduction.ipynb`** | **Self-contained end-to-end reproduction** → `submission.csv` → zip for Honors | ✅ **submission artifact** |
| `04_final_submission.ipynb` | Legacy finalizer | ⛔ **DO NOT RUN** — loads the buggy squashed `final_full_preds.npy` |
| `MASTER_PLAN.md` | Full experiment log, benchmarks, day-by-day plan | ✅ source of truth |
| `HANDOFF_TO_GLM.md` | Inter-agent status handoff | ✅ source of truth |
| `QUICKSTART.md` | Colab T4 runbook (Drive layout, keepalive, troubleshooting) | ✅ |
| `submissions/submission_anchor_blend_v1.csv` | Anchor submission — **LB 0.7627** | ✅ uploaded |
| `submissions/submission_v2_recipe_cv0.76203.csv` | Best recipe — **CV 0.76203** | ⏳ **upload next** |

### Engineered features (on Drive)

`features/train_engineered.parquet`, `features/test_engineered.parquet` — 101 columns.
Optuna params: `optuna_studies/all_best_params.json`.
Original OOF/test predictions: `models/{lgb,xgb,cat}_test_preds.npy`.

---

## 🗓 Timeline

| Day | Focus | Status |
|---|---|---|
| **Nov 15** | Audit, bug fixes, anchor submission, NB3 v2 + honors notebook built, FE Round 2 validated locally | ✅ done |
| **Nov 16** | Run NB3 v2 on Colab T4 (~2–3 h); FE Round 2 on CPU | 🔄 in progress |
| **Nov 17** | Re-tune Optuna on new feature set (5-fold OOF objective, same TE as NB3); full NB3 v2 re-run | ⏳ |
| **Nov 18** | Squeeze: hill-climb over all saved OOF pred sets; pseudo-labeling on p>0.9 / p<0.02 if stable | ⏳ |
| **Nov 19** | Final submission (CV decides), polish + zip honors notebook, submit early evening — scoring can lag | ⏳ |

---

## ⚙️ Setup

```bash
pip install -r requirements.txt
```

### Getting the data

Raw `train.csv` / `test.csv` are **not committed** (~60MB). Place them at:

```
competition_unzipped/04. Competition (due Nov 20)/Competition/input/
├── train.csv              <- not tracked by git
├── test.csv               <- not tracked by git
└── sample_submission.csv
```

In Colab, upload them to Drive at `MyDrive/gci_competition/input/` instead.

### Expected runtimes (Colab T4)

| Notebook | Time | Description |
|---|---|---|
| `01_feature_engineering.ipynb` | ~15 min | Feature engineering, adversarial validation, saves parquet |
| `02_optuna_tuning.ipynb` | ~2–3 h | 50 trials × 3 models |
| `03_ensemble_training_v2.ipynb` | ~2–3 h | Seed averaging, 4 members, OOF recipe selection |
| `gci_honors_reproduction.ipynb` | ~3–4 h | End-to-end reproduction (full pipeline, no Drive deps) |

---

## 📁 Repo layout

```
.
├── 01_feature_engineering.ipynb
├── 02_optuna_tuning.ipynb
├── 03_ensemble_training.ipynb          # legacy (superseded)
├── 03_ensemble_training_v2.ipynb       # primary pipeline
├── 04_final_submission.ipynb            # legacy — DO NOT RUN
├── gci_honors_reproduction.ipynb        # self-contained honors artifact
├── MASTER_PLAN.md                       # experiment log + plan
├── HANDOFF_TO_GLM.md                    # inter-agent handoff
├── QUICKSTART.md                        # Colab T4 runbook
├── README.md
├── requirements.txt
├── submissions/
│   ├── submission_anchor_blend_v1.csv       # LB 0.7627
│   └── submission_v2_recipe_cv0.76203.csv   # CV 0.76203
└── competition_unzipped/04. Competition (due Nov 20)/
    ├── competition_tutorial.pdf
    └── Competition/
        ├── README.ipynb
        ├── tutorial.ipynb
        ├── HomeCredit_columns_description.xlsx
        └── input/sample_submission.csv
```

---

## ⚠️ Known risks

| Risk | Mitigation |
|---|---|
| Colab session dies mid-run | All intermediates written to Drive after each fold; resume = re-run remaining folds |
| T4 too slow for seed averaging | Seeds only for XGB/CAT; LightGBM on CPU is fast at this data size |
| Overfitting the holdout / OOF | Hill-climb capped at ≤6 members; holdout untouched as a final sanity check |
| Deadline crunch | The anchor blend is already ≥ 0.76 on the LB, so a late failure still lands a strong score |
| Honors notebook not reproducible | `gci_honors_reproduction.ipynb` is end-to-end and must be validated before Nov 19 |
