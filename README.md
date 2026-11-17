# GCI In-Class Competition — HomeCredit Default Risk

Solution pipeline for the GCI in-class competition (**due Nov 20, 01:00 UTC**), based on the
[HomeCredit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk) dataset.

**Metric:** ROC AUC · **Data:** 171,202 train rows × 98 engineered features (E2 set), 61,500 test rows
**Compute:** Google Colab Free (T4 GPU) + local Ryzen 7 5800U (8c/16t) for CPU experiments

---

## 📊 Current Status

| Item | Value |
|---|---|
| **Current public LB** | **0.76475** — best score so far (✅ uploaded) |
| Previous submissions | 0.76433–0.76435 (v2 recipe) → 0.7627 (anchor) → 0.66819 (dummy RF) |
| **Current best recipe** | **v4 — CV 0.76244** — `submissions/submission_v4_cv0.76244.csv` |
| v4 blend weights | LGB **.30** / XGBnew **.33** / CATnew **.33** / HGB **.04** |
| CV → LB calibration | 0.7587 → 0.7627 (**+0.004**); 0.76203 → 0.76433–0.76435 (**+0.0023**) |
| **Latest LB for v4** | **0.76475** (CV 0.76244 → LB +0.0023, consistent with v2) |
| Honors artifact | `gci_honors_reproduction.ipynb` regenerated for v4 — **still must be run on Colab CPU** |
| Submission rules | Unlimited submissions, **last file counts**, public LB is a subset of test |
| Time left | ~2 days (deadline Nov 20, 01:00 UTC = 04:00 Nairobi) |

### ✅ Local hyperparameter tuning — all ensemble members (completed Nov 17–18)

Tuning ran **locally on the CPU** against the final 98-column E2 feature set. It started with
the three GBM members (LightGBM, XGBoost, CatBoost) and was extended to HistGradientBoosting,
which had never been tuned. Results are recorded in [`PROCESS_LOG.md`](PROCESS_LOG.md) phases 3 and 5.

| Member | Tuning status | Before → after (single-seed unless noted) |
|---|---|---|
| **LightGBM** | ✅ 40 Optuna trials on E2 | 0.75989 → **0.76120**; 3-seed build **0.76155** |
| **XGBoost** | ✅ 26 Optuna trials on E2 | 0.76137 → **0.76192**; 3-seed build **0.76171** |
| **CatBoost** | ✅ First-ever re-tune (20 fold-0 trials, top-3 on full 5-fold) | 0.76033 → **0.76088** full-CV; 2-seed rebuild **0.76063** |
| **HistGradientBoosting** | ✅ First-ever re-tune (40 trials, full CV) | 0.75588 → **0.75820**; rebuild **0.75814** |

The HGB rebuild exposed a reproducibility bug: the tuning JSON stored only the four searched
hyperparameters, so the first rebuild silently used sklearn defaults (`max_iter=100` → 0.74969).
Re-running with the **full** tuned config (`max_iter=1000`, `early_stopping(0.1, 25)`) recovered
0.75814, matching the tuning result. This bug class is now an explicit audit target in
[`AUDIT_REQUEST.md`](AUDIT_REQUEST.md).

**Protocol (identical for all three members):**

1. **Same folds everywhere** — 5-fold `StratifiedKFold`, `random_state=42`, on the full
   171,202 rows. No 10% holdout carve-out (that was worth +0.0028 on its own).
2. **Target encoding inside each fold only** — no leakage from validation rows into training.
3. **Objective = full OOF AUC of the 3-seed-averaged member**, never the 10% holdout. Selection
   noise on 6.1k rows was what produced the fake 0.76502 earlier.
4. **Shallow trees only** (depth 3–4). The verified A/B/C benchmark showed leaves=63/127 lose
   to the tuned depth-3/4 trees, so the search space is deliberately narrow.
5. **CPU-only, fixed seeds (42 / 1337 / 2026)**, every trial written to an experiment log. The
   GPU buys nothing here: on 171k × 98 the models train in minutes, and the one GPU run
   (CatBoost) silently early-stopped on Logloss instead of AUC, costing ~0.0019 AUC.
6. **Acceptance gate:** a new parameter set only replaces the pinned one if the rebuilt
   member improves full OOF AUC by **≥ 0.0003** *and* the blend weights are re-validated on the
   new OOFs. No blind substitution into the submission path.

**Why this mattered:** all GBMs are 97–99.8% rank-correlated, so the blend was saturated —
progression: 0.7587 (anchor) → 0.76203 (v2) → 0.76224 (v3) → **0.76244 (v4)**. Tuning delivered
exactly the expected small gain. The neural-net diversity seat was also tested and **rejected**:
MLP OOF 0.74897 with rank-corr 0.9242 vs XGBnew (fails both gates), TabNet collapsed to a
constant 0.50000, and adding the MLP at any weight moved CV by only +0.00006.

**After tuning:** rebuilt per-seed OOF/test predictions → re-ran hill-climb + weight grid on the
verified OOFs → pinned v4 → validated the submission file (61,500 rows, IDs match sample
submission, no NaNs, preds in [0.00202, 0.77167], Spearman corr vs v3 = 0.99985).

### Verified model results (5-fold Stratified OOF, full 171,202 rows)

| Model / Blend | OOF AUC | Note |
|---|---|---|
| LGBnew ×3 seeds (re-tuned) | 0.76155 | rank-corr 0.993 vs XGBnew |
| XGBnew ×3 seeds (re-tuned) | 0.76171 | **best single model** |
| CATnew ×2 seeds (re-tuned) | 0.76063 | rank-corr ~0.981 vs XGBnew |
| HGBnew (re-tuned) | 0.75814 | rank-corr 0.9756 vs XGBnew |
| MLP + embeddings (NN diversity) | 0.74897 | ❌ rejected — corr 0.9242, fails both gates |
| **Anchor blend** (LGB .30 / XGB .55 / CAT .15) | 0.7587 | submitted → LB 0.7627 |
| **v2** (E2 + full-data OOF selection) | 0.76203 | submitted → LB 0.76433–0.76435 |
| **v3** (re-tuned LGB/XGB, 2 XGB generations) | 0.76224 | built + validated, corr 0.9995 vs v2 |
| **v4 — current** (LGB .30 / XGBn .33 / CATn .33 / HGB .04) | **0.76244** | **submitted → LB 0.76475** |

> **Note on the original pipeline:** the previous 3-GBM blend looked like it gained a lot over its
> best member (+0.0002). The real gain is that the three GBMs are **97–99% rank-correlated** —
> they are nearly the same model, so diversity is the binding constraint, not tuning.

### Colab run 1 — local vs. Colab parity (Nov 16, executed notebook, verified)

| Member | Local | Colab run | Verdict |
|---|---|---|---|
| LGB 3-seed | 0.75989 | 0.75985 | ✅ |
| XGB 3-seed | 0.76167 | 0.76161 | ✅ (CPU `hist`; the GPU probe failed on Colab — harmless) |
| HGB | 0.75588 | 0.75562 | ✅ |
| CAT 2-seed | 0.76033 | **0.75847** | ⚠️ **GPU trap** — AUC early-stopping is not implemented on CatBoost GPU, so it stopped on Logloss instead |
| **Blend** | **0.76203** | **0.76174** | ✅ self-selected LGB .15 / XGB .75 / CAT .10 |

The notebook reproduced the local numbers within noise on all members, and the Colab blend
landed at **CV 0.76174** — a **+0.003 over the anchor's 0.7587**. Colab output:
`Drive/GCI Competition/submissions/submission_v2_cv0.76174.csv`
(test stats mean 0.08990, std 0.08889 — matches local).

**The CatBoost cell has been fixed to CPU-only**, so any re-run now reproduces the strong
0.7603 CAT member instead of the degraded GPU 0.75847. See decision #9 below.

> ⚠️ **Note:** `submission_v2_cv0.76174.csv` lives on Google Drive and has **not** been copied
> into `submissions/` locally yet, so it is not in this repo. Copy it across to have it
> version-controlled and side-by-side with the others.

---

## 📈 Score Ladder

| Stage | CV AUC | Est. LB | Status |
|---|---|---|---|
| Dummy RF (tutorial, 5 features) | ~0.66 | 0.668 | ✅ submitted, superseded |
| Anchor blend | 0.7587 | 0.7627 | ✅ submitted |
| v2 recipe (E2 + OOF selection) | 0.76203 | 0.76433–0.76435 | ✅ submitted |
| v3 recipe (re-tuned LGB/XGB) | 0.76224 | — | ✅ built + validated, not uploaded |
| **v4 recipe (all 4 members tuned)** | **0.76244** | **0.76475** | ✅ **submitted — current best** |
| NN diversity seat (MLP / TabNet) | — | — | ❌ rejected (see phase 6 of `PROCESS_LOG.md`) |
| Final action | — | — | ⏳ run Honors notebook, keep v4 as **last** file |

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

### 9. The GPU is not always faster — and on CatBoost it silently costs accuracy

Colab run 1 surfaced a trap that only appears on real hardware. The CatBoost cell tried GPU
first with a CPU fallback, and the GPU path **did not error** — it just produced a *worse* model:
**OOF 0.75847 on GPU vs 0.76033 on CPU** (≈ −0.0019, enough to move the blend).

**Root cause:** CatBoost's GPU backend does not implement AUC-based early stopping, so the model
silently early-stopped on **Logloss** instead of the intended **AUC** metric. No exception was
raised, so the "GPU if available, CPU otherwise" pattern looked like it was working.

**Decision:** the CatBoost member is now **hard-pinned to CPU** (`task_type` removed, GPU branch
deleted) so the strong 0.7603 member is reproducible on every re-run. The general lesson: on this
dataset, the GPU buys nothing — XGBoost ran on CPU `hist` on Colab anyway, and the one model that
*did* use the GPU (CatBoost) lost ~0.002 AUC. **Local CPU experiments are the fast path, not the
slow fallback.**

Also worth recording: the Colab blend self-selected **LGB .15 / XGB .75 / CAT .10**, which differs
from the local exhaustive-grid recipe (**LGB .06 / XGB .64 / CAT .30**) because the Colab CAT member
was the degraded GPU one. That is precisely why the CAT cell had to be fixed before any further
Colab runs — otherwise the recipe selection would have been fitted to a broken member.

---

## 🗂 Files & Notebooks

| File | Role | Status |
|---|---|---|
| `01_feature_engineering.ipynb` | Feature engineering + adversarial validation, saves parquet | ✅ working (cell 12 syntax error fixed) |
| `02_optuna_tuning.ipynb` | 50 trials × 3 models | ⚠️ **stale** — tuned without TE; replaced by the local re-tunes |
| `03_ensemble_training.ipynb` | Original 5-fold ensemble + stacking + SHAP | ⚠️ superseded (contains bugs 1–3) |
| `03_ensemble_training_v2.ipynb` | Evidence-based rebuild, OOF hill-climb + weight grid | ✅ historical runbook; params overridden with re-tuned values, CAT pinned to CPU |
| **`05_nn_diversity.ipynb`** | Phase 2 NN diversity on Colab T4: MLP + embeddings, TabNet, seat gate | ✅ **executed** — MLP 0.74897 / corr 0.9242, TabNet collapsed → seat rejected |
| **`gci_honors_reproduction.ipynb`** | **Self-contained end-to-end reproduction of v4** (14 cells, CPU-only, tuned params + pinned weights hardcoded) | ✅ **submission artifact — run on Colab CPU (~3.5–4 h) and zip** |
| `04_final_submission.ipynb` | Legacy finalizer | ⛔ **DO NOT RUN** — loads the buggy squashed `final_full_preds.npy` |
| **`PROCESS_LOG.md`** | Full chronological record, all 8 phases, rule-compliance and artifact map | ✅ **primary source of truth** |
| **`AUDIT_REQUEST.md`** | Devil's-advocate audit brief (16 attack surfaces) | ⏳ audit not yet delivered (`AUDIT_FINDINGS.md`) |
| `MASTER_PLAN.md` | Experiment log and day-by-day plan | 📁 historical (superseded by `PROCESS_LOG.md`) |
| `HANDOFF_TO_GLM.md` | Inter-agent handoff | 📁 historical (its early sections were superseded mid-file — read the GLM RESPONSE at the bottom) |
| `TRAINING_STAGE_REVIEW.md` | 7-stage synthesis, rule-compliance and guide-mapping tables | ✅ |
| `QUICKSTART.md` | Colab T4 runbook (Drive layout, keepalive, troubleshooting) | ✅ |
| `phase1_diversity_v{1,2,3}.py`, `phase1_results_summary.json` | Phase 1 diversity experiments (script iterations + results) | 📁 historical; v3 is final, results later found to be weaker regenerated runs |
| `submissions/submission_anchor_blend_v1.csv` | Anchor submission — **LB 0.7627** | ✅ uploaded |
| `submissions/submission_v2_recipe_cv0.76203.csv` | v2 recipe — **LB 0.76433–0.76435** | ✅ uploaded |
| `submissions/submission_v3_recipe_cv0.76224.csv` | v3 recipe — **CV 0.76224**, corr 0.9995 vs v2 | ✅ built, fallback |
| **`submissions/submission_v4_cv0.76244.csv`** | **v4 recipe — CV 0.76244, LB 0.76475** | ✅ **current best; keep as LAST file** |

### Engineered features (on Drive)

`X_e2.parquet` / `X_test_e2.parquet` — 98 columns (E2 set, 16 enrichments + 18 dead features removed).
Tuned params: `best_params_{lgb,xgb,cat,hgb}_e2.json` (session temp dir).
Verified OOF/test `.npy` predictions and `v4_recipe.json` — see the artifact map in `PROCESS_LOG.md` §4.

---

## 🗓 Timeline

| Day | Focus | Status |
|---|---|---|
| **Nov 15** | Audit, bug fixes, anchor submission, NB3 v2 + honors notebook built, FE Round 2 validated locally | ✅ done |
| **Nov 16** | NB3 v2 on Colab T4; FE Round 2 | ✅ Colab run 1 verified — blend CV 0.76174, CAT GPU trap found + fixed |
| **Nov 17** | Re-tune LGB/XGB, first CAT + HGB tune, NN diversity test, v3 build | ✅ done — v3 CV 0.76224; NN seat rejected; v4 development started |
| **Nov 18** | Rebuild all tuned members, hill-climb v4, build + validate submission | ✅ **done — v4 CV 0.76244 → LB 0.76475** |
| **Nov 19** | Run Honors notebook on Colab CPU, zip it, final upload (v4 must be last) | ⏳ |

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
├── 03_ensemble_training_v2.ipynb       # historical runbook (re-tuned params)
├── 04_final_submission.ipynb            # legacy — DO NOT RUN
├── 05_nn_diversity.ipynb                # NN diversity seat (rejected)
├── gci_honors_reproduction.ipynb        # self-contained v4 honors artifact ← RUN THIS
├── PROCESS_LOG.md                       # full chronological record + artifact map
├── AUDIT_REQUEST.md                     # pending devil's-advocate audit brief
├── TRAINING_STAGE_REVIEW.md             # 7-stage synthesis
├── MASTER_PLAN.md                       # historical plan
├── HANDOFF_TO_GLM.md                    # historical handoff
├── QUICKSTART.md                        # Colab T4 runbook
├── phase1_diversity_v{1,2,3}.py         # phase 1 scripts (v3 final)
├── phase1_results_summary.json
├── README.md
├── requirements.txt
├── submissions/
│   ├── submission_anchor_blend_v1.csv       # LB 0.7627
│   ├── submission_v2_recipe_cv0.76203.csv   # LB 0.76433–0.76435
│   ├── submission_v3_recipe_cv0.76224.csv   # CV 0.76224 (fallback)
│   └── submission_v4_cv0.76244.csv          # CV 0.76244 / LB 0.76475 ← LAST FILE
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
