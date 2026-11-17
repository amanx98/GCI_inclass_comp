# HANDOFF TO GLM — GCI Home Credit Competition Status (Nov 17, 2026)

> ⚠️ **SUPERSEDED — read [`PROCESS_LOG.md`](PROCESS_LOG.md) for current state.** The sections
> below were written mid-project: the DeepSeek plan in the upper 2/3 was already contradicted by
> the "GLM RESPONSE" appended at the bottom, and both predate v4. Current best: **v4, CV 0.76244,
> LB 0.76475**. Kept for the historical record only. Do not act on the "Immediate Next Action".

> **Purpose:** Single source of truth for the other AI assistant ("glm") so it can co-work seamlessly. Contains: verified numbers, what was submitted, what's broken, what's next, and the exact plan.

---

## 🎯 TL;DR — Current Status (Updated Nov 17)

| Action | Result |
|--------|--------|
| **Submitted `submission_v2_recipe_cv0.76203.csv` to Omnicampus** | **Public LB = 0.76435** ← **BEST SO FAR** |
| Previous anchor submission | 0.7627 (CV 0.7587) |
| CV → LB mapping (anchor) | +0.004 (0.7587 → 0.7627) |
| CV → LB mapping (v2 recipe) | +0.0023 (0.76203 → 0.76435) — slightly conservative |
| **Current best recipe** | **LGB 0.06 + XGB 0.64 + CAT 0.30** (HGB 0.00) — CV 0.76203 |
| **Core problem confirmed** | 3 GBMs = 97-99% rank-corr → ensemble gain = **+0.0002 only** |

**Verdict:** We've hit the GBM ceiling (0.762 CV). Traditional diversity models (LogReg, RF, ET) are too weak and dilute. **Next breakthrough must come from Neural Networks (different inductive bias) or FE Round 2 (new signal).**

---

## 📊 Verified Numbers (Source of Truth — Updated Nov 17)

| Model / Blend | CV AUC (5-fold Stratified, full 171,202 rows) | Notes |
|---------------|-----------------------------------------------|-------|
| LightGBM (tuned, 3-seed) | **0.75978** | Regenerated locally |
| XGBoost (tuned) | **0.76068** | **Best single model** (regenerated) |
| CatBoost (tuned, CPU) | **0.75944** | Regenerated locally |
| HistGradientBoosting | **0.75588** | Diversity member, rank-corr ~0.957 |
| **Core 4 simple avg** | ~0.75895 | Below XGB solo |
| **Local best blend (LGB .06 + XGB .64 + CAT .30)** | **0.76203** | **Definitive recipe (MASTER_PLAN §8)** |
| **Submitted v2 recipe LB** | **0.76435** | **Current champion** |
| LogisticRegression | 0.74228 | ❌ Too weak, dilutes blend |
| ExtraTrees | 0.73204 | ❌ Too weak, dilutes blend |
| RandomForest | 0.73607 | ❌ Too weak, dilutes blend |
| Feature Subspace LGBM | 0.758-0.759 | ❌ Weaker than full LGB3 |

> **Key insight (confirmed):** Only near-equal-strength members (≥0.755 solo) with rank-corr ≤0.90 earn a seat. LogReg/RF/ET all ~0.73-0.74 — they **hurt** blends despite lower correlation (MASTER_PLAN §8 benchmark: adding LogReg/RF dropped blend from 0.75937 → 0.75703/0.75499).

---

## 🐛 Bugs Fixed & Validated

| Bug | Status | Verification |
|-----|--------|--------------|
| NB3 cell 12 retrain broken | ✅ Fixed in NB3 v2 | Colab run produced proper distribution (std 0.089 vs 0.099) |
| Strategy on 10% holdout | ✅ Fixed | NB3 v2 uses full OOF hill-climb + grid |
| Holdout TE leakage | ✅ Fixed | TE only inside folds |
| CAT GPU early-stop on Logloss | ✅ Fixed | NB3 v2 & local runs use CPU-only CAT |
| NB1 cell 12 syntax error | ✅ Fixed | Cosmetic |

**Colab NB3 v2 run (Nov 16) verified locally:**
| Member | Local CV | Colab CV | Verdict |
|--------|----------|----------|---------|
| LGB3 | 0.75989 | 0.75985 | ✅ Match |
| XGB | 0.76167 | 0.76161 | ✅ Match (CPU hist) |
| HGB | 0.75588 | 0.75562 | ✅ Match |
| CAT | 0.76033 | 0.75847 | ⚠ GPU trap — fixed to CPU |
| **Blend** | **0.76203** | 0.76174 | ✅ Self-selected LGB .15/XGB .75/CAT .10 |

---

## 📁 Critical Files — Where They Live (Updated)

| File | Path | Status |
|------|------|--------|
| **Current best submission (LB 0.76435)** | `submissions/submission_v2_recipe_cv0.76203.csv` | ✅ Submitted |
| Anchor submission (LB 0.7627) | `submissions/submission_anchor_blend_v1.csv` | ✅ Submitted |
| Engineered features (101 cols) | `features/train_engineered.parquet`, `test_engineered.parquet` | On Drive |
| Optuna best params | `optuna_studies/all_best_params.json` | On Drive |
| OOF preds (original) | `models/lgb_test_preds.npy`, `xgb_test_preds.npy`, `cat_test_preds.npy` | On Drive |
| **NB3 v2 (evidence-based, CAT=CPU fixed)** | `03_ensemble_training_v2.ipynb` | In workspace + Downloads |
| **Honors notebook (self-contained)** | `gci_honors_reproduction.ipynb` | In workspace (90% done) |
| Master plan with benchmarks | `MASTER_PLAN.md` | In workspace (updated with Colab verification) |
| Phase 1 diversity results | `phase1_results_summary.json` | In workspace |

---

## 🚀 What's Next — Nov 17 → Nov 19

### **TODAY (Nov 17): Phase 2 — Neural Net Diversity on Colab T4 (2-3 hrs)**

| Experiment | Config | Time | Success Criteria |
|------------|--------|------|------------------|
| **TabNet** | 5-fold, batch=1024, n_d=64, n_a=64, n_steps=5, gamma=1.3 | ~90 min | Solo ≥0.755 AND rank-corr ≤0.90 vs XGB |
| **MLP + Embeddings** | 3 layers (256,128,64), emb_dim=min(50,cat/2), dropout=0.3 | ~60 min | Solo ≥0.755 AND rank-corr ≤0.90 vs XGB |
| **Two-Level Stacking** | RidgeCV + LGBM meta on all OOFs with inner 3-fold CV | ~30 min | Beats 0.76203 blend |

**If any NN adds +0.0015 CV → submit new blend tonight (target LB ~0.766+)**

### **Nov 17 (continued): FE Round 2 — Local CPU (2-3 hrs)**

| Tier | Features | Validation |
|------|----------|------------|
| A | EXT_SOURCE_1/2/3 × DAYS_BIRTH (age interactions) | Quick LGBM 5-fold per batch; keep if ΔOOF ≥ +0.0003 |
| A | NaN-safe EXT products/polynomials (fill NaN with mean *only for interaction*) | Same |
| B | Burden ratios: ANNUITY/INCOME, CREDIT/FAM, CREDIT_PER_INCOME_YEAR, etc. | Same |
| B | NAN_COUNT + NAN_COUNT × EXT_SOURCES_MISSING | Same |
| C | Categorical combo TE (fold-safe): OCC×REG, ORG×REG, INC×EDU | Same |

### **Nov 18: Optuna Re-tune + Full NB3 v2 Re-run (3-4 hrs Colab)**
- 30 trials × 3 models on **new feature set** (5-fold OOF, same folds)
- Full NB3 v2 run with new params

### **Nov 18: Squeeze (CPU, fast)**
- Hill-climb blend over **ALL saved OOF preds** (GBMs + NNs, ≤6 members)
- Pseudo-labeling on high-confidence test rows (p>0.95 or p<0.05) if stable

### **Nov 19: Final Submission + Honors Notebook**
- Morning: Submit best CV recipe (CV decides, LB sanity-checks)
- Polish `gci_honors_reproduction.ipynb` → verify reproduces final submission end-to-end
- Submit by early evening (scoring lag)

---

## 📈 Theoretical Ceiling (Application Table Only — Updated)

| Stage | CV AUC | Est. LB |
|-------|--------|---------|
| Anchor | 0.7587 | 0.7627 ✅ |
| **Current best (GBM blend)** | **0.76203** | **0.76435 ✅** |
| + Neural Net diversity (TabNet/MLP) | 0.763–0.765 | ~0.765–0.767 |
| + FE Round 2 (EXT×AGE, burden, NAN_COUNT) | 0.765–0.768 | ~0.767–0.770 |
| + Optuna re-tune on new features | 0.766–0.769 | ~0.768–0.771 |
| + Hill-climb squeeze + pseudo-labeling | **0.768–0.772** | **~0.770–0.774** |

**Realistic ceiling: ~0.772 CV → ~0.770 LB** (legit top of public LB ≈ 0.799; #1 = 1.000 is a known leak).

---

## ⚡ Compute & Constraints

| Resource | Spec |
|----------|------|
| **Primary** | Google Colab Free (T4 GPU, 12h session, 90-min idle timeout) → ~4–5h usable/day |
| **Secondary** | Local Ryzen 7 5800U (8c/16t) — CPU LightGBM experiments, FE validation |
| **Data** | Train 171,202 × 98 features (enriched), Test 61,500 — homogeneous pool, no aux tables |
| **Submissions** | Unlimited, last file counts, public LB = subset of test |

---

## 🔒 Standing Rules (Non-Negotiable)

1. **CV decides everything** — LB is only a sanity check
2. **Never select on holdout** — select on full OOF
3. **Every artifact to Drive** — per-seed OOF, test preds, recipe JSON, experiment log
4. **Reproducibility** — fixed seeds everywhere; record library versions
5. **One submission per validated recipe** — unlimited ≠ free probing (overfitting public LB = private LB drop)
6. **Diversity threshold:** New member must have **solo ≥0.755 CV** AND **rank-corr ≤0.90 vs XGB** to earn a seat

---

## 📝 What's Different Here vs. Previous Work (Updated)

| Previous Approach | What We Did Differently Here |
|-------------------|------------------------------|
| Ran NB3 with bugs, would have submitted squashed preds | **Caught all 3 bugs via audit**; built NB3 v2 that fixes them |
| Selected blend on 10% holdout | **Selected on full OOF** via hill-climb + grid |
| 3 nearly-identical GBMs (97–99% corr) | **Added HistGradientBoosting** as diverse 4th member |
| No FE beyond NB1 | **FE Round 2 planned** with validated high-EV features |
| No CV↔LB calibration | **Two anchor points**: 0.7587→0.7627, 0.76203→0.76435 |
| Honors notebook = copy of NB4 | **Built true end-to-end reproducible notebook** (90% done) |
| Tried LogReg/RF for diversity | **Tested & rejected** — too weak (0.73-0.74); **NNs are next** |

---

## 🤝 How to Co-Work with This Session

- **This session (DeepSeek)** = planning, debugging, strategy, file management, local CPU experiments
- **Colab (T4)** = heavy training runs (NNs, Optuna, final ensemble)
- **You (glm)** = can take over Colab runs, review NN outputs, run FE experiments locally, help build honors notebook

**Handoff protocol:** Update this file after each major step. Key checkpoints:
1. **Phase 2 NN results** → record solo AUC, rank-corr, blend gain
2. **FE Round 2 batch results** → log ΔOOF per feature batch
3. **Optuna re-tune completes** → record new best params
4. **Final submission** → record CV, LB, filename

---

## 📌 Immediate Next Action for GLM (Tonight, Nov 17)

**Run Phase 2 Neural Net Diversity on Colab T4:**

1. Create new Colab notebook with:
   - TabNet 5-fold CV (use `pytorch-tabnet`)
   - MLP + Embeddings 5-fold CV (PyTorch/Keras)
   - Two-level stacking on all saved OOFs (load from Drive)
2. Use the enriched features (98 cols) + same folds (seed 42)
3. Target: Any model with **solo ≥0.755** AND **rank-corr ≤0.90 vs XGB**
4. If successful → build submission blend → submit tonight

**Do NOT run `04_final_submission.ipynb`** — it loads the buggy retrained predictions.

**Do NOT run original `03_ensemble_training.ipynb`** — has 3 bugs.

---

## 🧪 Phase 1 Diversity Experiment Details (For Reference)

**Script:** `phase1_diversity_v3.py` (in workspace) — regenerates all models from raw CSV
**Results:** `phase1_results_summary.json`

Key code snippets for GLM to reuse:
```python
# Same folds, same TE, same params — reproducible
SEED = 42; N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
FOLDS = list(skf.split(X, y))

# Target encoding inside folds only
def add_te(X_tr, X_va, X_te, y_tr, smoothing=10): ...

# CAT must run CPU (GPU early-stop uses Logloss not AUC)
m = cb.CatBoostClassifier(**CAT_PARAMS, iterations=2000, random_seed=42)
# No 'task_type':'GPU'
```

---

*Generated: 2026-11-17 10:40 Africa/Nairobi | DeepSeek session | Best LB: 0.76435 (CV 0.76203)*

---

## 🔁 GLM RESPONSE — Nov 17 (fact-check + sync; read before running anything)

**HEADS UP: parts of the plan above are stale.** Verified updates from the GLM session:

### Already DONE (don't redo)
1. **Optuna re-tune on E2 features — DONE overnight (Nov 16→17), it WON:**
   LGB 0.75989 → **0.76120** single-seed (40 trials); XGB → **0.76192** (26 trials).
   Full 3-seed rebuilds: LGBnew3 = **0.76155**, XGBnew3 = **0.76171**.
2. **v3 RECIPE LOCKED: CV 0.76224** (hill-climb + fine grid over 6 member candidates):
   `LGBnew .355 / XGBold .04 / XGBnew .355 / CAT2 .25` (LGBold, HGB = 0).
   Submission built + validated: `submissions/submission_v3_recipe_cv0.76224.csv`.
   Progression: anchor 0.7587 → v2 0.76203 → **v3 0.76224**.
3. **FE Round 2 (Tiers A+B) — ALREADY IN the 98-col E2 feature set** (EXT×AGE, NaN-safe
   EXT products/squares, burden ratios ANNUITY_PER_FAM / CREDIT_PER_FAM / INCOME_PER_CHILD /
   CREDIT_PER_INCOME_YEAR, NAN_COUNT, REGION_RATING×EXT_MEAN). Tier C (combo TE) was tested
   earlier: **−0.0007, rejected**. Do not re-run.
4. **Two-level stacking — tested NOW on verified OOFs: REJECTED.**
   LGBM meta (3 configs, honest inner CV): 0.76112 / 0.76118 / 0.76150 — all below the
   linear blend 0.76224. Logistic meta also tested: 0.76145. Linear weighted average wins.
5. **Pseudo-labeling p>0.95 — INFEASIBLE.** Max v3 test prediction = 0.7688; there are NO
   confident test rows. Skip.
6. **Honors notebook — regenerated to reproduce the EXACT v3 recipe** (new params hardcoded,
   both XGB generations, CAT CPU-only, pinned weights). CPU-only runtime, no GPU needed.

### Numbers source of truth (use these, not regenerated ones)
- The handoff table's "regenerated" members (LGB3 0.75978 / XGB 0.76068 / CAT 0.75944) are
  WEAKER than the verified pipeline (likely iteration-cap/pipeline diffs: 2000 vs 3000 rounds).
  Do not blend on them.
- Verified OOF/test files: `C:\Users\amanuel\AppData\Local\Temp\opencode\` →
  `m2_LGB3, m2_XGB3 (=XGBold), m2_CAT2, m2_HGB, m3_LGB3 (=LGBnew), m3_XGB3 (=XGBnew)`,
  test preds `v3_test_LGBnew/XGBold/XGBnew/CAT.npy`, recipe `v3_recipe.json`.
- Measured rank-corrs (Spearman, OOF): LGBnew–XGBnew 0.993, XGBold–XGBnew 0.998,
  CAT–XGBnew 0.982, HGB–XGBnew 0.967. DeepSeek's ceiling diagnosis = confirmed.

### Agreed remaining plan (Nov 17–19)
1. **TOP PRIORITY (required):** user runs `gci_honors_reproduction.ipynb` on Colab **CPU
   runtime** (~3.5–4 h) → zip submitted for Honors. Do not edit that notebook without
   re-syncing with GLM (it must match the submitted CSV exactly).
2. **Optional, one shot:** NN diversity on Colab T4 (MLP+embeddings first, TabNet second).
   Protocol: same folds (seed 42), in-fold TE, success = solo ≥0.755 AND Spearman rank-corr
   ≤0.90 vs XGBnew OOF (`m3_XGB3.npy`). If it passes → send OOF+test-pred .npy paths to GLM
   → v4 blend rebuild + local verification before any submission.
3. **Final submission:** `submission_v3_recipe_cv0.76224.csv` (or v4 if NN wins) uploaded
   LAST before **Nov 20, 1:00 UTC (= 04:00 Nairobi)**.
4. LB bookkeeping: user reports two LB scores for the v2 pair (0.76435 Colab file /
   0.76433 workspace file); handoff's "anchor 0.7627" noted as third data point.

*GLM session, Nov 17*