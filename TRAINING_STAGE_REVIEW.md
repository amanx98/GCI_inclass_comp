# TRAINING STAGE REVIEW — GCI 2026 Home Credit (Nov 17, 2026)

Full synthesis of every finding from both work sessions (GLM + DeepSeek), stage by stage.
This is the "what we learned → what we changed" record for the final run.

---

## 0. TL;DR

| Recipe | CV (full-data 5-fold OOF) | LB |
|---|---|---|
| Tutorial-style baseline (user's NB4, buggy) | — | 0.668 (dummy RF) |
| Anchor blend (old 3-GBM pipeline) | 0.7587 | 0.7627 |
| v2 recipe (E2 features + full-data OOF selection) | 0.76203 | 0.7643–0.7644 |
| **v3 recipe (re-tuned params + 2-generation ensemble)** | **0.76224** | *(pending — final upload)* |
| Theoretical remaining lever | +0.000–0.0015 via NN diversity seat | — |

---

## 1. Stage-by-stage review

### Stage 1 — Validation protocol
**Finding:** the 10% holdout (a) wasted 34k training rows, (b) made blend selection
high-variance (user's "0.76502 holdout" was inflated noise; true old-pipeline OOF was 0.7587).
**Change:** everything selects on full-data 5-fold StratifiedKFold (seed 42) OOF. Verified: +0.0028.
**Status:** ✅ locked.

### Stage 2 — Preprocessing
**Findings (from tutorial EDA + audit):**
- `DAYS_EMPLOYED = 365243` is a sentinel anomaly → anomaly flag + NaN (kept).
- `OWN_CAR_AGE` > 60 treated as outliers → NaN (kept); clip lower 0.
- Missing values: GBMs get -999 fills; TE inside folds only (no leakage).
- Frequency encoding of categoricals kept only where useful (many `_FREQ` cols proven dead → dropped).
**Status:** ✅ locked.

### Stage 3 — Feature engineering
**Findings:**
- 18 dead features (SHAP≈0 list + FLAG_MOBIL) removed — removing them helped.
- 17 new features validated: EXT×AGE interactions, NaN-safe EXT products/squares, EXT_SUM,
  burden ratios (ANNUITY_PER_FAM, CREDIT_PER_FAM, INCOME_PER_CHILD, CREDIT_PER_INCOME_YEAR),
  NAN_COUNT, REGION_RATING×EXT_MEAN.
- Combo-categorical TE (OCC×REG etc.): **rejected** (−0.0007).
- Feature matrix: 98 columns, train/test column-aligned, asserted.
**Status:** ✅ locked (E2 set). *DeepSeek's proposed "FE Round 2" = these same features; nothing left here.*

### Stage 4 — Hyperparameter tuning (README checklist: "optimize the hyperparameters")
**Findings:**
- Depth-3/4 shallow trees dominate; deeper trees lose (verified A/B/C benchmark).
- Original Optuna params were tuned on the OLD feature set → stale.
- Overnight re-tune on E2 (Nov 16→17): LGB 40 trials → 0.75989→**0.76120** single-seed
  (+0.0013); XGB 26 trials → →**0.76192**. CAT re-tune skipped (params still fine, cost too high).
- 3-seed rebuilds with new params: LGBnew3 **0.76155**, XGBnew3 **0.76171**.
**Change:** v3 recipe uses re-tuned params; old-param XGB kept as a 4%-weight diversity member.
**Status:** ✅ locked. Protocol documented (TPESampler seed 42, spaces in
`Temp\opencode\overnight_retune.py`, best values in `best_params_{lgb,xgb}_e2.json`).

### Stage 5 — Model roster (README checklist: "explore different models")
**Findings (all measured, same folds):**
| Model | OOF AUC | Rank-corr vs XGBnew | Verdict |
|---|---|---|---|
| LGB (re-tuned) ×3 seeds | 0.76155 | 0.993 | ✅ seat (0.355) |
| XGB (re-tuned) ×3 seeds | 0.76171 | 1.000 | ✅ seat (0.355) |
| XGB (old params) ×3 seeds | 0.76167 | 0.998 | ✅ small seat (0.04) |
| CAT (2 seeds, CPU) | 0.76028 | 0.982 | ✅ seat (0.25) |
| HGB | 0.75588 | 0.967 | ❌ 0 weight (too weak despite best corr) |
| LogReg / RF / ExtraTrees | 0.742 / 0.736 / 0.732 | ~0.88–0.90 | ❌ dilute |
| DART / feature-subspace LGBM | < full LGB3 | — | ❌ |
**Seat rule (adopted from DeepSeek, verified empirically):** solo ≥0.755 AND rank-corr ≤~0.98
with real strength; mere decorrelation without strength HURTS (HGB proof).
**Status:** ✅ locked for GBMs; ⏳ NN candidates pending (see Stage 7).

### Stage 6 — Ensembling (README checklist: "explore other ensembling methods")
**Findings (all tested on verified OOFs):**
| Method | CV |
|---|---|
| **Linear weighted average (grid + hill-climb)** | **0.76224** ✅ |
| Rank-normalized weighted average | 0.76200 |
| Logistic regression meta | 0.76145 |
| LGBM meta-stacker (3 configs, inner CV) | 0.76112–0.76150 |
| Simple average | < linear |
**Change:** linear weighted average, weights selected on OOF (never holdout), test preds =
fold-averaged per member then weighted (never one-shot retrain — the old cell-12 bug).
**Status:** ✅ locked.

### Stage 7 — Remaining lever: neural-net diversity seat (the only untested family)
**Rationale:** all GBMs are 0.97–0.998 rank-correlated; DeepSeek's Phase-1 batch had no
strong-enough decorrelated member. An NN with categorical embeddings uses a different
inductive bias and could plausibly land 0.74–0.76 with low correlation.
**Gate (adopted):** solo OOF ≥ 0.755 AND Spearman rank-corr ≤ 0.90 vs XGBnew.
**If passed:** v4 blend rebuilt locally on verified OOFs (hill-climb + grid incl. NN seat),
verified, THEN a submission is considered. No blind submissions.
**Status:** ⏳ notebook `05_nn_diversity.ipynb` built for Colab T4 (~1 h).

---

## 2. Rule-compliance checklist (README + competition PDF)

| Rule | How we comply |
|---|---|
| No external data | Only `train.csv`/`test.csv` (+ provided column-description xlsx for EDA). NNs trained **from scratch** on competition data — no pretrained weights. |
| No hand-labeling | Every prediction comes from trained models; no manual edits anywhere. |
| Reproducibility (seeds) | Seeds 42/1337/2026 everywhere; folds seed 42; library versions recorded in notebook outputs. |
| Honors: submit code reproducing the CSV | `gci_honors_reproduction.ipynb` regenerates the exact v3 recipe from raw CSVs (pinned weights, hardcoded tuned params); zip to be submitted. |
| Cite referenced public code | Tutorial notebook, README, Kaggle Home Credit public EDA conventions (EXT/ratio features), library docs — cited in the honors notebook header. |
| Last submitted file counts | Final action = upload best-verified CSV last; submission log kept (3 files so far, each a validated recipe — no public-LB probing). |
| Compute honesty | All local experiments documented; Colab runs recorded; no hidden selection. |

## 3. Guide-technique mapping (README "How to Improve" + tutorial "Next Steps")

| Guide advice | Our answer |
|---|---|
| "Add more features" | 5 → 98 validated features (dead list pruned, 17 new validated) |
| "Optimize hyperparameters" | Optuna ×2 generations (baseline set + re-tune on final set) |
| "Explore different models" | 4 GBM families + rejected batch (LogReg/RF/ET/DART/subspace) + NN candidates pending |
| "Explore other ensembling methods" | Weighted-average vs rank-avg vs 2-level stacking — all measured; linear weighted average wins |
| "Is holdout the best evaluation?" | No → full-data 5-fold OOF (+0.0028 verified) |
| EDA-driven preprocessing | Anomaly/outlier/missing handling per tutorial, extended and validated |

## 4. What to run, in order

1. `gci_honors_reproduction.ipynb` on Colab **CPU** (~3.5–4 h) → zip → **required for Honors**.
2. `05_nn_diversity.ipynb` on Colab **T4** (~1 h) → if a candidate passes the gate, send
   `nn_diversity/*.npy + nn_summary.json` to GLM → v4 blend, locally verified.
3. Final upload: `submission_v3_recipe_cv0.76224.csv` (or v4 if NN earns a seat) — **last file**.
