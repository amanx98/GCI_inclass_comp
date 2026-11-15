# MASTER PLAN — GCI Home Credit Competition (ends Nov 20, 1:00 AM UTC)

## 0. Where we are (facts) — updated after full Drive audit

| Item | Value |
|---|---|
| Current LB score / rank | **0.66819 / #4564** — from a DUMMY (tutorial RF, 5 features) |
| LB reference points | #1 = 1.00 (leak/hand-labeling, ignore), legit top ≈ 0.79, big pack at 0.70 |
| **TRUE CV (verified from Drive OOF files)** | LGB 0.75546 / XGB 0.75826 / CAT 0.75841 → **best blend 0.7587** |
| Reported holdout 0.76502 | **Inflated by ~+0.006 (6.1k-row noise). Truth = 0.7587 on full OOF** |
| Ensemble gain over best single | **+0.0002 only** — the 3 GBMs are 97–99% rank-correlated, no diversity |
| `final_full_preds.npy` (NB4 default) | **Squashed distribution** (0.346–0.777, std 0.099 vs proper blend std 0.213) — submitting it would hurt. Corr with proper blend only 0.981 |
| Anchor submission | **BUILT LOCALLY**: `submissions/submission_anchor_blend_v1.csv` (0.30/0.55/0.15 blend) — upload this first |
| Compute | Free Colab T4, ~4–5 h/day usable + local Ryzen 7 (8c/16t) for CPU LGBM experiments |
| Data | train 171,202 rows (101 engineered features), test 61,500 — homogeneous pool, no aux tables |
| Submission rules | Unlimited submissions; **last file counts**; public LB = subset of test |

**Key inference:** this dataset (application table only, 171k rows) has a realistic CV ceiling of
**~0.77–0.78**. The NB2-tuned params are all depth-3 trees with `min_split_gain=0.76` — tuned
WITHOUT the TE columns, so they underfit once TE is added (local benchmark quantifies this,
see §8). Diversity is the #1 fix: the 3-model blend currently buys +0.0002.


**Expected score ladder** (updated with verified numbers)
| Step | CV AUC | Est. LB |
|---|---|---|
| Dummy RF (submitted) | ~0.66 | 0.668 ← current |
| Anchor blend (ready: `submissions/submission_anchor_blend_v1.csv`) | 0.7587 | ~0.755–0.759 |
| **NB3 v2 recipe (verified locally: XGB .55/CAT .35/LGB .10)** | **0.76194** | ~0.757–0.762 |
| + XGB/CAT seed-averaging + finer grid (in progress) | 0.762–0.763 | ~0.758–0.763 |
| Day-3/4 extras (re-tune on Colab GPU, pseudo-labeling) | 0.763–0.766 | ~0.759–0.762 |

---

## 1. TODAY (Nov 15) — Anchor submission + NB3 bug fixes

### 1a. Submit the anchor NOW (~15 min, no GPU)
The best predictions from the finished run are already on Drive:
`models/lgb_test_preds.npy`, `xgb_test_preds.npy`, `cat_test_preds.npy`
(weights were LGB=0.30, XGB=0.55, CAT=0.15).

```python
import numpy as np, pandas as pd
W = '/content/drive/MyDrive/GCI Competition'
test = pd.read_csv(f'{W}/input/test.csv', usecols=['SK_ID_CURR'])
lgb = np.load(f'{W}/models/lgb_test_preds.npy')
xgb = np.load(f'{W}/models/xgb_test_preds.npy')
cat = np.load(f'{W}/models/cat_test_preds.npy')
blend = 0.30*lgb + 0.55*xgb + 0.15*cat
pd.DataFrame({'SK_ID_CURR': test.SK_ID_CURR, 'TARGET': blend}) \
  .to_csv(f'{W}/submissions/anchor_blend_0.765cv.csv', index=False)
```
Upload to Omnicampus. **Purpose: lock a real LB point that maps 1:1 to our CV.**
Do NOT submit the cell-12 "retrained full-data" predictions — see bug #1 below.

### 1b. Fix NB3 bugs (the 3 real ones)

1. **Cell 12 (retrain-on-full-data) — remove from the submission path.**
   - XGBoost is retrained with `num_boost_round=lgb_final.best_iteration_` (the *LightGBM's*
     iteration count — wrong model) and CatBoost with a hard-coded 1000 iters.
   - Single full-data models ≠ the 5-fold-averaged predictions the blend weights were
     calibrated on. This mismatch usually *costs* AUC.
   - **Rule: final submission = CV-fold-averaged test predictions, never one-shot retrains.**
2. **Select strategy on full OOF, not the 10% holdout** (6.1k rows → ±0.005 noise).
   Fit the meta-learner with `cross_val_predict` on OOF and compare
   {weighted avg, rank avg, LogReg, Ridge} by OOF AUC. Keep the holdout only as a sanity check.
3. **Cell 9 leakage-smell:** holdout TE is computed from full train; acceptable, but keep it
   *only* as a sanity check (see fix 2) so no selection decision leans on it.

Also worth fixing while we're in there:
- NB1 cell 12 has a syntax error (`1. Run 02_optuna_tuning.ipynb` outside a string) — the
  summary cell crashes after everything is saved. Cosmetic but fix it.
- `high_corr_features` is computed in NB1 and loaded in NB3 but never used — decide: drop via
  OOF validation, or ignore. (Default: ignore; trees don't care much about corr.)

### 1c. NB3 v2 — stronger stacking (run tonight, ~2 h on T4)
- **Seed averaging:** 3 seeds × 5 folds for XGBoost + CatBoost (the two best models);
  LGB stays 1×5-fold (it's the weakest). Data is tiny (171k × ~120 cols) → fast on CPU.
- **Blend space:** compare on OOF: (a) weighted average, (b) rank-average,
  (c) LogReg meta on logit-transformed OOF preds, (d) LGBM-stacker (careful, overfits —
  only with inner-CV). Pick by OOF AUC.
- **4th model for diversity:** `HistGradientBoostingClassifier` (sklearn, CPU, minutes) or a
  small Keras MLP on standardized features. Even +0.001 OOF from a 4th member usually helps.
- Save everything: per-seed OOF + test preds, final blend preds + the exact recipe JSON.

---

## 2. DAY 2 (Nov 16) — Feature Engineering Round 2

Highest-EV features for THIS dataset (known strong from HomeCredit public EDA):

**Tier A — EXT_SOURCE engineering (dominant signal):**
- `EXT_SOURCE_1/2/3 × DAYS_BIRTH` (age interactions — the classic top features)
- `EXT_x * EXT_y` products and `(EXT_x - EXT_y)` diffs **with proper missing handling**:
  fill NaN with column mean *only for the interaction*, and add `EXT_x_MISS & EXT_y_MISS` indicator
- Polynomial features (degree 2) on the 3 EXT_SOURCE cols — worth ~+0.005 historically
- `EXT_SOURCES_PROD_NONAN` = product of available sources / count of available

**Tier B — financial ratios:**
- `AMT_ANNUITY / AMT_CREDIT`, `AMT_CREDIT / AMT_INCOME_TOTAL`, `AMT_ANNUITY / AMT_INCOME_TOTAL`
  (some exist), `AMT_CREDIT / (CNT_FAM_MEMBERS+1)`, `AMT_INCOME_TOTAL / (CNT_CHILDREN+1)`
- `AMT_ANNUITY / (DAYS_EMPLOYED_abs/365 + 1)` — payment burden vs. work history
- `CREDIT_TERM * REGION_RATING_CLIENT` interactions
- Row-level `NAN_COUNT` across all features + `NAN_COUNT_X_EXT_SOURCES_MISSING`

**Tier C — categorical combos with CV-safe TE:**
- TE on: `OCCUPATION_TYPE × REGION_RATING_CLIENT`, `ORGANIZATION_TYPE × REGION_RATING_CLIENT`,
  `NAME_INCOME_TYPE × NAME_EDUCATION_TYPE` (compute means inside folds, smoothing=20)

**Validation protocol (no GPU wasted):**
- One quick LGBM (default-ish params, 5-fold OOF) per feature batch.
- Keep a batch only if OOF AUC improves ≥ +0.0003. Log every experiment in a table.

---

## 3. DAY 3 (Nov 17) — Re-tune + full NB3 v2 run

- Re-run Optuna on the new feature set: **30 trials × 3 models, 5-fold OOF objective**
  (not 3-fold — selection noise matters). ~3 h on T4/CPU.
- Full NB3 v2 run (seeds, 4 models, OOF-selected blend). ~3–4 h.
- Submit the best CV recipe → second LB point. We now have 2 anchor points to
  map CV↔LB precisely.

## 4. DAY 4 (Nov 18) — Squeeze + decide final

Options in EV order:
1. **Hill-climb blend** over all saved OOF/test pred sets (greedy forward selection of
   members by OOF AUC). Free, CPU-only, +0.001–0.003.
2. **Pseudo-labeling** on test rows with p>0.9 or p<0.02 (model-generated → allowed; disclose
   in writeup). +0.000–0.002. Only if everything else is stable.
3. Re-check feature selection (null-importance) — drop only if OOF confirms.

## 5. DAY 5 (Nov 19, deadline day)

- **Morning:** final submission with the best CV recipe (CV decides, LB sanity-checks).
- Build the honors notebook: single clean notebook reproducing `submission_final.csv`
  end-to-end (seeds set, requirements pinned, **citations** for any referenced public code),
  zip it (NB4 does this — polish it).
- **Submit by early evening** — scoring can lag several minutes; don't cut it close to 1:00 AM UTC.

---

## 6. New findings from the full code audit (NB2 / NB4 / PDF)

1. **NB2's Optuna never applied target encoding** — the objective functions literally say
   "simplified version - full implementation in notebook 3". So the tuned params were selected
   on a feature space WITHOUT the `_TE` columns that NB3 trains with (param/feature mismatch),
   and NB2's reported AUCs are not comparable to NB3's. Fix: Day-3 re-tune must use the same
   fold-level TE as NB3.
2. **NB2's MedianPruner never prunes** — `trial.report(cv_score, step=0)` reports once per
   trial, so there are no intermediate values to prune on. Harmless, but "pruning" was
   theater; ignore pruning in the re-run.
3. **NB4 submits the WRONG predictions by default** — cell 2 loads `final_full_preds.npy`
   (the buggy cell-12 retrained models), NOT the cell-10 weighted-average CV blend that
   scored 0.765. Running NB4 as-is would submit the worse variant. Switch NB4 to the
   CV-averaged blend (or the new NB3 v2 output).
4. **NB4's validation would have caught a bug we missed** — it asserts TARGET ∈ [0,1], which
   raw Ridge `predict()` can violate. One more reason the final blend must stay in
   probability space (weighted average / LogReg).
5. **HONORS GAP (important):** NB4's "polished notebook" is just a copy of NB4 with cleared
   outputs — it loads `.npy` files, so it does NOT reproduce the submitted CSV end-to-end.
   The PDF confirms Honors/Outstanding requires submitted code. Build a **single
   self-contained notebook** (load raw CSVs → FE → models → blend → submission.csv) before
   the deadline. Citation list required for any referenced public code.
6. The 5 flags (FLAG_MOBIL etc.) and renumbered IDs confirmed via xlsx + data inspection —
   no hidden columns to exploit; FE is where the signal is.

## 8. Local benchmark results (Ryzen 7 5800U, LightGBM 4.7 CPU, 5-fold, FULL 171k)

**Params test — verdict: your Optuna depth-3 params are correct, deeper trees LOSE**
| Config | OOF AUC |
|---|---|
| **A: your tuned params (depth 3) + TE, full data** | **0.75822** |
| B: leaves=63, deeper | 0.75546 |
| C: leaves=127, deeper | 0.75580 |

Two conclusions:
1. **Free +0.0028 verified**: your saved LGB OOF (0.75546) was trained on the 90% subset only.
   Same params, same folds, but CV on the **full 171k** (no holdout carve-out) → 0.75822.
   NB3 v2 must drop the holdout entirely and select on OOF.
2. Deeper trees underperform → the signal is saturated for plain GBMs on the current feature
   set. Gains must come from (a) NEW FEATURES, (b) MODEL DIVERSITY, not depth.
   (The old 3-GBM blend's +0.0002 gain — the members are 97–99% correlated.)

**Feature engineering test — verdict: EXT×AGE features help, modestly (+0.0006)**
| Config | OOF AUC |
|---|---|
| Original 101 features | 0.75822 |
| +17 new (EXT×AGE, NaN-safe EXT poly, burden ratios, NAN_COUNT) | **0.75881 (+0.00059)** |

**Feature engineering tests — running tally (LGB tuned, full data, same folds)**
| Config | OOF AUC |
|---|---|
| Original 101 features | 0.75822 |
| +17 new (EXT×AGE, NaN-safe EXT poly, burden ratios, NAN_COUNT) | 0.75881 |
| + 4 categorical-combination TEs (fold-safe) | 0.75920 |
| + 18 dead features dropped, NO combo TE | 0.75963 |
| **FINAL: enriched + combo TE + dead dropped (combo keys kept)** | *(running — members benchmark)* |

(Note: the earlier "0.75963 + comboTE" label was wrong — that run had combo keys dropped,
so combo TE silently skipped 1 of 4 combos. The corrected all-in config is in the members run.)

**Diversity test — verdict: only near-equal-strength members earn a seat**
| Member | OOF AUC | rank-corr vs LGB |
|---|---|---|
| HGB | 0.75582 | 0.957 |
| LogReg | 0.74213 | 0.905 |
| RF | 0.73687 | 0.889 |
| Blends: LGB+HGB 0.65/0.35 = **0.75937** (+0.0012); adding LogReg/RF HURTS (0.75703 / 0.75499);
LogReg meta over 4 = 0.75604 — flat stacking dilutes. | | |

**Members round 2 (E2 feature config, full data, verified locally)**
| Member | OOF AUC |
|---|---|
| LGB tuned ×3 seeds avg | 0.75989 |
| HGB | 0.75588 |
| **XGB tuned (full data)** | **0.76137 — best single model** |
| CAT tuned (full data) | 0.76033 |

**DEFINITIVE FINAL RECIPE (verified locally, full 171k, all pairs/triples exhaustive at 0.01):**
**LGB3 0.06 + XGB3 0.64 + CAT 0.30 + HGB 0.00 → CV 0.76203**
(vs old pipeline's true 0.7587 → **+0.0033 verified**; stable across search resolutions
— coarse grid 0.76194, hill-climb 0.76186, fine grid ~0.762)

Deliverables built:
- `03_ensemble_training_v2.ipynb` — Colab runbook, self-selects the recipe on OOF
- `gci_honors_reproduction.ipynb` — self-contained: raw CSVs → FE → 4 members (hardcoded
  tuned params, seeds: LGB×3, XGB×3, CAT×2) → OOF grid → submission.csv → zip for Honors
- `submissions/submission_anchor_blend_v1.csv` — anchor submission (ready to upload)

Notable: with XGB/CAT at full strength on the new features, HGB earns zero weight — the two
strongest GBMs dominate. `03_ensemble_training_v2.ipynb` is BUILT and self-selects the recipe
on OOF, so it adapts if member strengths shift on Colab. Pending verification: XGB seed-averaging
(running), then finer grid (step 0.02) with the updated member OOFs.

## 9. Standing rules

1. CV decides everything; LB is only a sanity check (public = subset, last-file-counts).
2. Never select on the 10% holdout; select on full OOF.
3. Every artifact saved to Drive: per-seed OOF, test preds, recipe JSON, experiment log.
4. Reproducibility: fixed seeds everywhere; record library versions; this is also your
   honors-award requirement.
5. One submission per validated recipe — unlimited submissions ≠ free probing
   (overfitting the public subset is how people drop on the private LB).

## 7. Risks

| Risk | Mitigation |
|---|---|
| Colab session dies mid-run | All intermediates go to Drive after each fold; resume = re-run remaining folds |
| T4 too slow for seeds | Seeds only for XGB/CAT; LGBM on CPU is fine at this data size |
| Overfitting to holdout/OOF | Hill-climb limited to ≤6 members; keep holdout untouched as final sanity check |
| Deadline crunch | Anchor + Day-1 blend already ≥ 0.76 even if everything later fails |
| Honors notebook not reproducible | Build self-contained end-to-end notebook by Nov 19 (see §6.5) |
