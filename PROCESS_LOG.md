# PROCESS LOG — GCI 2026 In-Class Competition (Home Credit Default Risk)

Full chronological record of the pipeline development, Nov 10 – Nov 18, 2026.
Written so that (a) any third party can verify what was done and why, and (b) an independent
auditor can check every claim against the artifacts on disk.

**Deadline:** Nov 20, 2026, 1:00 AM UTC. Metric: ROC AUC. Platform: Omnicampus, last file counts.

---

## 1. Timeline

### Phase 0 — Baseline (Nov 10–14)
- Tutorial reproduction: 5-feature model, dummy-level submissions (LB 0.668).
- First real pipeline: single LightGBM, 10% holdout validation.

### Phase 1 — Validation overhaul (Nov 14–15)
- **Found:** 10% holdout (a) wasted ~34k training rows, (b) made blend selection
  high-variance. The "0.76502 holdout blend" was inflated noise.
- **Change:** all selection moved to full-data 5-fold StratifiedKFold (seed 42) OOF.
- **Verified gain:** +0.0028 on comparable members.
- Anchor result: old 3-GBM pipeline, OOF CV 0.7587 → LB 0.7627 (calibration point for
  CV→LB gap ≈ +0.004).

### Phase 2 — Feature engineering E2 (Nov 15)
- Base: engineered features from Drive `features/` (anomaly flags, missing indicators,
  domain ratios, EXT_SOURCE interactions, bureau aggregations, quantile bins, freq encoding).
- E2 layer: 16 enrichment features (EXT×AGE interactions, NaN-safe EXT products/squares,
  EXT_SUM, burden ratios, NAN_COUNT, REGION_RATING×EXT_MEAN) + removal of 18 dead features
  (FLAG_MOBIL, ORGANIZATION_TYPE, dead `_WAS_MISSING`/`_FREQ`/`_QBIN` cols).
- Result: 98-column matrix, train/test column-aligned and asserted.
- **Rejected with numbers:** combo-categorical target encoding (OCC×REG etc.) −0.0007.
- Final matrix saved: `X_e2.parquet` (train), `X_test_e2.parquet` (test, built with the
  exact same enrich protocol; `bench_test_preds.py` is the reference implementation).

### Phase 3 — Hyperparameter tuning, generation 1 (Nov 15–16)
- Optuna (TPESampler seed 42) on E2 features, fold-0 proxy search then full-CV validation:
  - LGB: 40 trials → single-seed 0.75989 → **0.76120** (+0.0013)
  - XGB: 26 trials → single-seed **0.76192**
  - CAT: skipped in this round (params kept; cost judged too high — later corrected, see Phase 5)
- Best params saved to `best_params_{lgb,xgb}_e2.json` in the session temp dir.

### Phase 4 — Ensemble generation + v2/v3 recipes (Nov 16)
- Multi-seed rebuilds (3 seeds: 42/1337/2026; CAT 2 seeds: 42/1337), all on identical
  folds/TE protocol:
  - LGBnew3: 0.76155 | XGBnew3: 0.76171 | XGBold3: 0.76167 | CAT2: 0.76028
- Diversity batch measured and **rejected** on the same folds:
  - LogReg 0.742 / RF 0.736 / ExtraTrees 0.732 / DART & feature-subspace LGBM < full LGB3
  - HGB (untuned) 0.75588, best decorrelation (rank-corr 0.967) but too weak solo
- Seat rule adopted: **solo OOF ≥ 0.755 AND Spearman rank-corr ≤ 0.90 vs XGB** for a new
  member type; established members judged by blend contribution instead.
- Ensembling methods compared on verified OOFs:
  - **Linear weighted average: 0.76224** ✅ (winner)
  - Rank-normalized weighted average: 0.76200
  - Logistic regression meta: 0.76145
  - LGBM meta-stacker (3 configs): 0.76112–0.76150
  - Simple average: < linear
- **v2 recipe** (LGB .355/XGBo .04/XGBn .355/CAT .25) CV 0.76203 → **LB 0.76433–0.76435**.
- **v3 recipe** (same members, re-tuned LGB/XGB seeds) CV 0.76224. Built, validated
  (61,500 rows, corr 0.9995 vs v2), saved as `submission_v3_recipe_cv0.76224.csv`.
- Pseudo-labeling evaluated and **rejected** (max test pred 0.769 → no confident rows).

### Phase 5 — Closing the tuning gaps (Nov 17, overnight_retune2.py)
- CAT first-ever re-tune: fold-0 TPE search (20 trials, 9851 s) → best
  `lr 0.0709 / depth 3 / l2 2.53 / bagging 0.38 / random_strength 0.0084 / border 217`;
  fold-0 0.75936 vs baseline fold-0 0.75717.
- Top-3 configs validated on proper full 5-fold CV: top1 **0.76088**, top2 0.75957,
  top3 0.76047. Baseline old params: 0.76033.
- **Adoption gate passed:** new CAT params adopted (0.76088 > 0.7607 threshold; also beats
  baseline outright). Saved to `best_params_cat_e2.json`.
- HGB first-ever tune (40 trials, full CV): **0.75820** (from 0.75588).
  Params: `lr 0.0306 / 31 leaves / min_samples_leaf 93 / l2 9.02 / max_iter 1000 /
  early_stopping(0.1, 25)`. Saved to `best_params_hgb_e2.json`.

### Phase 6 — NN diversity experiment (Nov 17, Colab T4, `05_nn_diversity.ipynb`)
- MLP with categorical embeddings (PyTorch) + TabNet (`pytorch-tabnet`), both trained
  **from scratch** on competition data only (rules: no external data / no pretrained weights).
- Identical FE/folds/TE as the GBM pipeline (feature parity enforced in the notebook).
- **Results:** MLP OOF 0.74897, rank-corr vs XGBnew 0.9242 → fails BOTH gate conditions
  (≥0.755 solo, ≤0.90 corr). TabNet collapsed (OOF 0.50000 — constant output, training failure).
- Blend test (measured, not assumed): best blend incl. MLP at any weight = 5% → CV 0.76230
  vs 0.76224 baseline (+0.00006 = noise). **NN seat rejected.**

### Phase 7 — v4 member rebuild + assembly (Nov 18)
- CAT new params rebuilt on the verified protocol, 2 seeds: s42 **0.76082**,
  s1337 **0.76044** (avg 0.76063 vs old CAT2 0.76028, +0.00035). Rank-corr vs XGBnew ~0.981.
- HGB rebuilt — **bug found and fixed:** the tuning job's JSON stored only the 4 searched
  hyperparameters, so the first rebuild silently used sklearn defaults (max_iter=100 →
  0.74969). Re-run with the full tuned config (`max_iter=1000`, early stopping): **0.75814**,
  rank-corr 0.9756. Matches the tuning result (0.75820) within noise.
- v4 blend search (hill-climb + coarse grids on verified OOFs):
  - v3 recipe re-scored: 0.76224
  - v3 with CATn swap: 0.76241
  - Hill-climb: 0.76244; simple 3-way equal: 0.76244; plateau confirmed across weightings
  - **Pinned v4 recipe: LGB .30 / XGBn .33 / CATn .33 / HGB .04 → CV 0.76244**
  - Old XGB-generation and old CAT members dropped (zero blend contribution once tuned
    versions present).
- `submission_v4_cv0.76244.csv` built and validated: 61,500 rows, IDs match sample
  submission exactly, no NaNs, preds in [0.00202, 0.77167], Spearman corr vs v3 = 0.99985
  (expected: same members, weight shift).
- `gci_honors_reproduction.ipynb` regenerated for v4 (14 cells, CPU-only, hardcoded tuned
  params, pinned weights 0.30/0.33/0.33/0.04, ~3.5–4 h on Colab CPU).

---

## 2. Current state (Nov 18)

| Item | Status |
|---|---|
| Final submission file | `submissions/submission_v4_cv0.76244.csv` (CV 0.76244) |
| **Latest public LB** | **0.76475** (v4 uploaded) — best score so far |
| Earlier LB points | v2 recipe 0.76433–0.76435 · anchor 0.7627 · dummy RF 0.66819 |
| CV → LB transfer for v4 | 0.76244 → 0.76475 = **+0.0023**, consistent with v2's +0.0023 |
| Fallback file | `submissions/submission_v3_recipe_cv0.76224.csv` (CV 0.76224) |
| Honors reproduction code | `gci_honors_reproduction.ipynb` (reproduces v4 end-to-end) |
| Remaining user actions | run honors notebook on Colab CPU → zip; keep v4 as the **last** uploaded file before deadline |

## 3. Rule compliance record

| Rule | Compliance |
|---|---|
| No external data | Only train.csv/test.csv + provided column-description xlsx. NNs trained from scratch (no pretrained weights). TabPFN rejected partly because its weights are pretrained externally (rules-gray). |
| No hand-labeling | All predictions model-generated; no manual edits anywhere in the pipeline. |
| Fixed seeds | 42/1337/2026 for members; folds seed 42; Optuna TPESampler seed 42. |
| Honors code | Self-contained notebook reproduces the submitted CSV from raw CSVs. |
| Cited public code | Tutorial notebook, README, Kaggle Home Credit public EDA conventions (EXT/ratio features), library docs — cited in notebook header. |
| Last file counts | Plan: single final upload of v4 (or v3 if any verification fails). |

## 4. Artifact map (verification pointers)

- Feature protocol reference implementation: `bench_members2.py` (train), `bench_test_preds.py` (test)
- v3 rebuild reference: `rebuild_v3.py`; tuning generation 1: `overnight_retune.py`
- Tuning generation 2: `overnight_retune2.py` (log shows all stage results)
- v4 member rebuild: `rebuild_v4_members.py`; HGB fix: `hgb_fix.py`
- Params: `best_params_{lgb,xgb,cat,hgb}_e2.json` (temp dir)
- Verified OOF/test npy: `m3_LGB3.npy`, `m3_XGB3.npy`, `m2_XGB3.npy`, `m2_CAT2.npy`,
  `v4_CATnew_s{42,1337}_{oof,test}.npy`, `v4_HGBnew_{oof,test}.npy`,
  `v3_test_*.npy` (test preds), `v4_recipe.json`
- NN artifacts: `nn_diversity/` (mlp/tabnet oof+test, nn_summary.json)
- Notebooks: `gci_honors_reproduction.ipynb`, `03_ensemble_training_v2.ipynb`, `05_nn_diversity.ipynb`
- Docs: `TRAINING_STAGE_REVIEW.md`, `MASTER_PLAN.md`, `HANDOFF_TO_GLM.md`

## 5. Known open items / honest uncertainties

1. **CV→LB transfer unverified for v4.** v2 CV 0.76203 → LB 0.76433–0.76435. v4's +0.0002
   CV gain should transfer, but only one LB point is burnable effectively (last file counts).
2. **CAT tuning improvement is modest** (+0.00055 solo CV). Adoption is justified by the
   gate, but the margin is small.
3. **HGB seat rests on a 0.04 weight** chosen at a plateau where many weightings give
   0.76244. Risk of it being pure noise-weight is low but nonzero; it never exceeded the
   solo seat bar (0.75814 < 0.76224).
4. **Test-frame TE/enrichment protocol** must match training exactly — it does (same code
   path as `bench_test_preds.py`), but an auditor should diff these two implementations.
5. **One seed for HGB** vs 3 seeds for LGB/XGB — acceptable given its 4% weight and CPU
   cost (~90 s/run), but it's a documented asymmetry.
