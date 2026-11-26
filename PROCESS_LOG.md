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
- Best params saved to `best_params_{lgb,xgb}_e2.json` (canonical copies in `pipeline/`).

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

### Phase 8 — Independent audit + Honors repair (Nov 18)
- Delivered the audit brief as [`AUDIT_FINDINGS.md`](AUDIT_FINDINGS.md). Key results:
  - v4 reconstructs from the saved OOF/test `.npy` files to `max|diff| = 9.98e-17`; IDs match
    the sample submission exactly; 61,500 rows; no NaNs; range [0.00202, 0.77167].
  - v4 OOF 0.762442, v3 OOF 0.762242; v4−v3 = +0.000199 (paired bootstrap 95% CI
    +0.000085…+0.000327); 4/5 folds positive.
  - HGB's 0.04 seat is +0.000007 (95% CI −0.000025…+0.000040) — noise, but harmless.
  - No data leak: 34 raw columns, no ID overlap, no duplicate rows, row-order AUC 0.5029,
    adversarial AUC 0.5163. The “~0.775 ceiling” is unsupported.
- **Fixed two run-blocking bugs**:
  - `gci_honors_reproduction.ipynb` (and `01_feature_engineering.ipynb`) crashed in
    `add_binned_features` (`.astype(np.int8)` on NaN bins). Fixed with `.fillna(-1)`.
    Both notebooks now rebuild the 98-column E2 matrix with **0 mismatching cells**.
  - `01_feature_engineering.ipynb` still had a literal `1. Run 02_...` syntax error in its
    summary cell; fixed.
- **Quantified the reproducibility limit**: XGBoost `hist` and CatBoost are thread-count-sensitive
  (fold 0 XGB: `n_jobs=2` 0.759578/2250 trees vs `n_jobs=8` 0.759029/2134 trees, corr 0.996;
  CatBoost: `thread_count=2` 0.758955 vs `8` 0.759364, corr 0.9936).
  The Honors notebook reproduces the recipe and CV within ±0.004, not the CSV byte-for-byte.
  LightGBM was bit-identical across thread counts.
- Honest limitations recorded in the README: enrichment fill means are per-frame (2–3%
  relative), HGB's seat is noise, and the realistic remaining headroom is a few 1e-4.
- Build scripts, tuned params, recipes, and audit scripts copied into `pipeline/` so the
  evidence no longer depends on `%TEMP%`.

---

## 2. Current state (Nov 18)

| Item | Status |
|---|---|
| Final submission file | `submissions/submission_v4_cv0.76244.csv` (CV 0.76244) |
| **Latest public LB** | **0.76475** (v4 uploaded) — best score so far |
| Earlier LB points | v2 recipe 0.76433–0.76435 · anchor 0.7627 · dummy RF 0.66819 |
| CV → LB transfer for v4 | 0.76244 → 0.76475 = **+0.0023**, consistent with v2's +0.0023 |
| Fallback file | `submissions/submission_v3_recipe_cv0.76224.csv` (CV 0.76224) |
| Honors reproduction code | `gci_honors_reproduction.ipynb` — **repaired and FE-verified**; must still be run on Colab CPU and zipped |
| Independent audit | [`AUDIT_FINDINGS.md`](AUDIT_FINDINGS.md) — v4 reproduces; no leak; known limits documented |
| Remaining user actions | run Honors notebook on Colab CPU → zip; keep v4 as the **last** uploaded file before deadline (v5 round closed Nov 19, gate-rejected — see above) |

### Nov 18 — code-level notebook review (no submission change)

- **TE A/B** under the v4 folds/params: full fold-mean TE **0.761158** > nested 0.761026 >
  no TE 0.760795; leave-one-out TE **0.526475** is unusable (every fold stops at iteration 1
  because the model inverts the encoding and recovers the row's own label). Keep the shipped
  self-inclusive fold-mean TE; no TE-driven v5.
- **DEAD-feature add-back**: `ORGANIZATION_TYPE` **−0.001257** (LightGBM, 5/5 folds negative;
  CatBoost fold 0 **−0.001402**, with the baseline reproducing the shipped CAT AUC exactly),
  `FLAG_EMP_PHONE` −0.000411, `EXT_SOURCE_3_QBIN` −0.000070, all extras −0.001037. The pruning
  is empirically correct despite the univariate signal; do not re-add them.
- **Notebook repairs**: `01` silent `_FREQ` bug; `02` rewritten to CPU / 5 folds / E2 + TE /
  real pruning; `03` install, path, no-op override, CPU/thread and cat-dtype fixes; `05` NN
  bugs + path fallback + gate/preview. No model or submission artifact was rebuilt.
- **Decision:** keep `submission_v4_cv0.76244.csv` as the last uploaded file.

### Nov 19 — v5 experiment round (all local, Ryzen 7; harness `pipeline/v5/common.py`, reproduces v4 OOF 0.762442 exactly)

Gate for any v5 submission: 500-rep paired bootstrap of v5−v4 OOF AUC, 95% CI entirely > 0.

- **W1 model-based imputation** (EXT_SOURCE_1/3, OWN_CAR_AGE via fold-local LGB regressor on
  observed rows, `_IMP` + `×WAS_MISSING`): LGB 3-seed OOF **0.759901 vs 0.760197 → −0.000296**. Rejected.
- **W2 full-CV Optuna re-tune** (TPESampler seed 42, objective = mean 5-fold OOF, 80–100
  trials/family): XGB **+0.000258** (0.761668; best lr 0.0188, depth 3, mcw 12.5, λ 20.7),
  LGB +0.000152, HGB −0.000174, CAT −0.000502. Only XGB clearly positive; all kept as blend members.
- **W3 seed expansion**: LGB7-new 0.760460, **XGB7-new 0.761714**, CAT4 0.760995, HGB3 0.759125.
- **W4 FE round 3** (LGB 3-seed A/B, keep ≥ +0.0003): b1 KMeans k=8/16 **+0.000109**,
  b3 cat-pair count-joins +0.000045, b4 REGPOP×EXT_MEAN −0.000057. All rejected.
- **W5 blend re-search** over 15 members (hill-climb + 0.05 simplex grid + NNLS): best
  0.762454 (LGB .30 / XGB3 .25 / XGB-extra .10 / CAT4 .35) → bootstrap CI
  [−0.000047, +0.000077], p=0.626 — **fails gate**.
- **W5 member-swap candidates A–I** (XGB7-new / CAT4 / HGB3 / LGB7-new swaps): best
  D_cat_swap_only 0.762454, CI [−0.000038, +0.000062], p=0.680 — **none pass**.
- **Decision (2026-11-19 ~15:30 EAT): v5 REJECTED. `submission_v4_cv0.76244.csv` stays the last
  uploaded file** (re-asserted: 61,500 rows, IDs == sample == test, TARGET ∈ [0.00202, 0.77167],
  no NaN). Artifacts: `pipeline/v5/` + `pipeline/v5/out/` (OOFs, params, `w2_results.jsonl`,
  `w5_blend_result.json`, `w5_candidates.json`). v4 recipe is at the measured ceiling on the
  34-column subset; remaining headroom is a few 1e-4 and every v5 lever landed inside noise.

## 3. Rule compliance record

| Rule | Compliance |
|---|---|
| No external data | Only train.csv/test.csv + provided column-description xlsx. NNs trained from scratch (no pretrained weights). TabPFN rejected partly because its weights are pretrained externally (rules-gray). |
| No hand-labeling | All predictions model-generated; no manual edits anywhere in the pipeline. |
| Fixed seeds | 42/1337/2026 for members; folds seed 42; Optuna TPESampler seed 42. |
| Honors code | Repaired self-contained notebook reproduces the recipe and CV within ±0.004; XGB/CatBoost thread drift prevents a byte-identical CSV. The shipped v4 CSV is authoritative. |
| Cited public code | Tutorial notebook, README, Kaggle Home Credit public EDA conventions (EXT/ratio features), library docs — cited in notebook header. |
| Last file counts | Plan: single final upload of v4 (or v3 if any verification fails). |
| Transductive encoding | Label/frequency encoders are fit on train+test combined (features only, no target). Standard for this competition; stated explicitly rather than hidden. |

## 4. Artifact map (verification pointers)

- **Canonical copies in this repo:** `pipeline/` — `bench_members2.py`, `bench_test_preds.py`,
  `rebuild_v3.py`, `rebuild_v4_members.py`, `hgb_fix.py`, `overnight_retune.py`,
  `overnight_retune2.py`, tuned params, `v3_recipe.json`, `v4_recipe.json`,
  `ENVIRONMENT.txt`, and the audit scripts in `pipeline/audit/`.
- **Session artifacts (outside git, `%TEMP%\opencode\`):** verified OOF/test npy:
  `m3_LGB3.npy`, `m3_XGB3.npy`, `m2_XGB3.npy`, `m2_CAT2.npy`,
  `v4_CATnew_s{42,1337}_{oof,test}.npy`, `v4_HGBnew_{oof,test}.npy`,
  `v3_test_*.npy`, `X_e2.parquet`, `X_test_e2.parquet`, `nn_diversity/`.
  These can be rebuilt from raw data with the Honors notebook.
- Notebooks: `gci_honors_reproduction.ipynb` (authoritative reproduction),
  `01_feature_engineering.ipynb` (verified), `03_ensemble_training_v2.ipynb` (historical),
  `05_nn_diversity.ipynb` (historical).
- Docs: `AUDIT_FINDINGS.md` (current audit), `TRAINING_STAGE_REVIEW.md`, `MASTER_PLAN.md`,
  `HANDOFF_TO_GLM.md` (historical).

## 5. Known open items / honest uncertainties

1. **CV→LB transfer for v4 is now observed:** 0.76244 → LB 0.76475 (+0.0023), consistent with
   v2. The private-LB transfer is still unknown; the public subset is optimistic.
2. **CAT tuning improvement is modest** (+0.00055 solo CV; 4/5 folds positive). Adoption is
   justified by the gate, but the margin is small.
3. **HGB's seat is noise, not a gain.** Measured contribution +0.000007 (95% CI
   −0.000025…+0.000040). It is kept because the validated file already contains it; dropping it
   would not change the score materially.
4. **XGB/CatBoost thread-count sensitivity.** Fold-0 `n_jobs=2` vs `8` for XGB: 0.759578 vs
   0.759029, 2,250 vs 2,134 trees, corr 0.996. CatBoost `thread_count=2` vs `8`: 0.758955 vs
   0.759364, corr 0.9936. The Honors notebook reproduces the recipe/CV within ±0.004, not the
   CSV byte-for-byte. The shipped v4 CSV is authoritative.
5. **Enrichment fill means are per-frame.** Train/test fill values differ by 2.1–3.2% on missing
   EXT_SOURCE/DAYS_EMPLOYED. Not target leakage; negligible under the measured 0.516 adversarial
   AUC. Do not “fix” without a full rebuild and re-validation.
6. **One seed for HGB** vs 3 seeds for LGB/XGB — acceptable given its 4% weight and CPU cost,
   but a documented asymmetry.
7. **The old “~0.775 CV ceiling” is unsupported.** The package has only 33 raw features; the
   measured ceiling of the tried families is 0.76244. Treat remaining headroom as a few 1e-4.
