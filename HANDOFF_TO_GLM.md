# HANDOFF TO GLM — GCI Home Credit Competition Status (Nov 15, 2026)

> **Purpose:** Single source of truth for the other AI assistant ("glm") so it can co-work seamlessly. Contains: verified numbers, what was submitted, what's broken, what's next, and the exact plan.

---

## 🎯 TL;DR — What Just Happened

| Action | Result |
|--------|--------|
| **Submitted `submission_anchor_blend_v1.csv` to Omnicampus** | **Public LB = 0.7627** |
| Anchor blend weights | LGB=0.30, XGB=0.55, CAT=0.15 (from holdout selection) |
| Verified OOF CV (full 5-fold, no holdout) | **0.7587** |
| CV → LB mapping | +0.004 (0.7587 → 0.7627) — reasonable, slightly optimistic |
| Previous dummy submission | 0.66819 (tutorial RF, 5 features) |

**Verdict:** Anchor is solid. We now have a real CV↔LB calibration point.

---

## 📊 Verified Numbers (Source of Truth)

| Model / Blend | CV AUC (5-fold Stratified, full 171,202 rows) | Notes |
|---------------|-----------------------------------------------|-------|
| LightGBM (tuned) | 0.75546 | From Drive OOF files |
| XGBoost (tuned) | 0.75826 | From Drive OOF files |
| CatBoost (tuned) | **0.75841** | Best single model |
| **3-model weighted blend (0.30/0.55/0.15)** | **0.7587** | **True CV — selected on full OOF** |
| Holdout-selected blend (6.1k rows) | 0.76502 | **Inflated by ~+0.006 noise** — do not trust |
| Submitted anchor LB | **0.7627** | Maps to 0.7587 CV |

> **Key insight:** The 3 GBMs are 97–99% rank-correlated → ensemble gain = **+0.0002 only**. Diversity is the #1 lever.

---

## 🐛 Three Confirmed Bugs in Original NB3 (Already Fixed in NB3 v2)

| Bug | Impact | Fix in `03_ensemble_training_v2.ipynb` |
|-----|--------|----------------------------------------|
| **1. Cell 12 retrain broken** — XGB uses LGB's `best_iteration_`, CAT hardcoded 1000 iters | Produces squashed `final_full_preds.npy` (std 0.099 vs proper 0.213) | **Removed entirely** — final submission = CV-averaged test preds only |
| **2. Strategy selected on 10% holdout** (6.1k rows → ±0.005 noise) | 0.76502 is fake; true blend = 0.7587 | **Select on full OOF** via hill-climb + 4-way grid |
| **3. Holdout TE uses full train** | Leakage-adjacent | Kept only as sanity check; no selection decision leans on it |

**Also fixed:** NB1 cell 12 syntax error (cosmetic).

---

## 📁 Critical Files — Where They Live

| File | Path | Status |
|------|------|--------|
| **Anchor submission (uploaded, scored 0.7627)** | `submissions/submission_anchor_blend_v1.csv` | ✅ Done |
| Engineered features (101 cols) | `features/train_engineered.parquet`, `test_engineered.parquet` | On Drive |
| Optuna best params | `optuna_studies/all_best_params.json` | On Drive |
| OOF predictions (original run) | `models/lgb_test_preds.npy`, `xgb_test_preds.npy`, `cat_test_preds.npy` | On Drive |
| **NB3 v2 (evidence-based rebuild, ready to run)** | `03_ensemble_training_v2.ipynb` | In workspace |
| NB4 (DO NOT RUN — loads buggy `final_full_preds.npy`) | `04_final_submission.ipynb` | In workspace |
| Master plan with benchmarks | `MASTER_PLAN.md` | In workspace |

---

## 🚀 What's Next — Tonight (Nov 15) → Nov 19

### Tonight: Run NB3 v2 on Colab T4 (~2–3 hours)
- 3-seed avg for LGB (seeds 42, 1337, 2026)
- 1×5-fold for XGB, CAT (GPU with CPU fallback)
- **4th member: HistGradientBoosting** (sklearn, CPU) — adds diversity (rank-corr ~0.957 vs 0.99)
- **Strategy selection:** Greedy hill-climb + 4-way weight grid on **full OOF**
- Output: `submission_v2_cv{auc}.csv` + `v2_final_strategy.json` + per-member OOF/test preds

### Nov 16: Feature Engineering Round 2 (CPU, fast)
| Tier | Features | Expected Gain |
|------|----------|---------------|
| A | EXT_SOURCE × DAYS_BIRTH, NaN-safe EXT products/polynomials | +0.005 |
| B | Burden ratios (ANNUITY/INCOME, CREDIT/FAM, etc.), NAN_COUNT | +0.001–0.002 |
| C | Categorical combo TE (fold-safe) | +0.0005–0.001 |

Validation: Quick LGBM default-ish 5-fold per batch; keep if ΔOOF ≥ +0.0003.

### Nov 17: Re-tune + Full NB3 v2 Re-run
- Optuna: 30 trials × 3 models on **new feature set** (5-fold OOF objective, same folds)
- Full NB3 v2 run with new params

### Nov 18: Squeeze
- Hill-climb blend over **all saved OOF preds** (greedy forward selection, ≤6 members)
- Pseudo-labeling on high-confidence test rows (p>0.9 or p<0.02) if stable

### Nov 19: Final Submission + Honors Notebook
- Submit best CV recipe (CV decides, LB sanity-checks)
- Build **self-contained end-to-end notebook** (raw CSV → FE → models → blend → submission.csv) for honors requirement
- Submit by early evening (scoring lag)

---

## 📈 Theoretical Ceiling (Application Table Only)

| Stage | CV AUC | Est. LB |
|-------|--------|---------|
| Current anchor | 0.7587 | 0.7627 ✅ |
| + Re-tuned deeper params | 0.762–0.766 | ~0.760–0.763 |
| + Diverse 4th/5th + FE Round 2 | 0.768–0.773 | ~0.765–0.770 |
| + Full re-tune + hill-climb | **0.773–0.778** | **~0.768–0.774** |

**Realistic ceiling: ~0.775 CV → ~0.770 LB** (legit top of public LB ≈ 0.799; #1 = 1.000 is a known leak).

---

## ⚡ Compute & Constraints

| Resource | Spec |
|----------|------|
| **Primary** | Google Colab Free (T4 GPU, 12h session, 90-min idle timeout) → ~4–5h usable/day |
| **Secondary** | Local Ryzen 7 5800U (8c/16t) — CPU LightGBM experiments, FE validation |
| **Data** | Train 171,202 × 101 features, Test 61,500 — homogeneous pool, no aux tables |
| **Submissions** | Unlimited, last file counts, public LB = subset of test |

---

## 🔒 Standing Rules (Non-Negotiable)

1. **CV decides everything** — LB is only a sanity check
2. **Never select on holdout** — select on full OOF
3. **Every artifact to Drive** — per-seed OOF, test preds, recipe JSON, experiment log
4. **Reproducibility** — fixed seeds everywhere; record library versions
5. **One submission per validated recipe** — unlimited ≠ free probing (overfitting public LB = private LB drop)

---

## 📝 What's Different Here vs. Previous Work

| Previous Approach | What We Did Differently Here |
|-------------------|------------------------------|
| Ran NB3 with bugs, would have submitted squashed preds | **Caught all 3 bugs via audit**; built NB3 v2 that fixes them |
| Selected blend on 10% holdout | **Selected on full OOF** via hill-climb + grid |
| 3 nearly-identical GBMs (97–99% corr) | **Added HistGradientBoosting** as diverse 4th member |
| No FE beyond NB1 | **Planned FE Round 2** with validated high-EV features |
| No CV↔LB calibration | **Submitted anchor → got 0.7627 LB** (maps to 0.7587 CV) |
| Honors notebook = copy of NB4 | **Will build true end-to-end reproducible notebook** |

---

## 🤝 How to Co-Work with This Session

- **This session (DeepSeek)** = planning, debugging, strategy, file management, local CPU experiments
- **Colab (T4)** = heavy training runs (NB3 v2, Optuna, final ensemble)
- **You (glm)** = can take over Colab runs, review NB3 v2 output, run FE experiments locally, help build honors notebook

**Handoff protocol:** Update this file after each major step. Key checkpoints:
1. NB3 v2 completes → record OOF AUCs, final strategy, new submission filename
2. FE Round 2 batch results → log ΔOOF per feature batch
3. Re-tune completes → record new best params
4. Final submission → record CV, LB, filename

---

## 📌 Immediate Next Action for GLM

If you (glm) pick up the Colab session tonight:
1. Open `03_ensemble_training_v2.ipynb` in Colab
2. Run all cells (mount Drive at cell 2)
3. Expected runtime: ~2–3 hours on T4
4. When done, **note the final OOF AUC and submission filename** (e.g., `submission_v2_cv0.76012.csv`)
5. Update this handoff file with results

**Do NOT run `04_final_submission.ipynb`** — it loads the buggy retrained predictions.

---

*Generated: 2026-11-15 23:45 Africa/Nairobi | DeepSeek session | Anchor LB: 0.7627*