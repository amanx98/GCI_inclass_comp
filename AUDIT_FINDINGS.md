# AUDIT FINDINGS — GCI 2026 Home Credit pipeline

**Audit date:** 2026-11-18 · **Auditor:** independent pass against the brief in `AUDIT_REQUEST.md`
**Scope:** `PROCESS_LOG.md`, `TRAINING_STAGE_REVIEW.md`, `MASTER_PLAN.md`, `HANDOFF_TO_GLM.md`,
all notebooks, the build scripts (now in `pipeline/`), the saved OOF/test `.npy` artifacts, the
four submission CSVs, and the raw competition data.
**Method:** static code review + numeric checks on the saved artifacts + targeted re-runs
(feature engineering, fold-0 thread tests). Full training was not repeated.

---

## Verdict summary

The modelling pipeline is **sound and the v4 submission is authentic**: blending the saved
OOF/test arrays with the pinned weights reproduces `submission_v4_cv0.76244.csv` to
`max|diff| = 9.98e-17`, with IDs, row count, and value ranges all correct. The CV numbers,
member AUCs, correlations, and the v4−v3 improvement all reproduce. There is **no target
leakage, no train/test contamination, and no distribution shift** in the data.

The single biggest risk was **not** the model — it was the Honors artifact. The notebook that
was supposed to be the end-to-end reproduction **crashed on its own feature-engineering cell**
before training a single model, and the repo's `01_feature_engineering.ipynb` had the same
crash plus a syntax error. Both are now fixed and verified: the repaired notebooks rebuild the
reference 98-column matrix with **zero mismatching cells** (train and test). A second, smaller
risk is that “reproduces the submitted CSV bit-for-bit” is **false** across hardware because
XGBoost `hist` and CatBoost are thread-count-sensitive; the notebook now reproduces the recipe
and CV within a stated tolerance, and the shipped CSV remains authoritative.

**Bottom line:** keep v4 as the last file, run the repaired Honors notebook, and do not change
the model before the deadline. The honest remaining headroom is ~1e-4, not the ~1e-2 claimed
by the old “0.775 ceiling” note.

---

## Critical findings

### C1. The Honors notebook could not run (FIXED, verified)

**Evidence.** Executing `gci_honors_reproduction.ipynb` cell 3 (`add_binned_features`) raised:

```
pandas.errors.IntCastingNaNError: Cannot convert non-finite values (NA or inf) to integer
```

at `df[f'{col}_QBIN'] = df[f'{col}_QBIN'].astype(np.int8)`. `EXT_SOURCE_1` contains NaNs, and
`pd.cut(...).astype(np.int8)` cannot represent them. The shipped parquet stores missing bins as
`-1`, so the build code must have used a fill that the notebook dropped.

**Impact.** The Honors code requirement could not have been met by running the notebook as
delivered. Every “run it on Colab” instruction in the docs would have failed in the first
feature cell.

**Fix.** `.fillna(-1).astype(np.int8)` in both `gci_honors_reproduction.ipynb` and
`01_feature_engineering.ipynb`.

**Verification.** The repaired Honors feature cells rebuild the exact training matrix:

| Check | Result |
|---|---|
| Shape | (171,202 × 98) and (61,500 × 98) — matches `X_e2` / `X_test_e2` |
| Column names and order | identical |
| Train cells | **0 mismatches / 16,777,796** |
| Test cells | **0 mismatches / 6,027,000** |
| `y` and IDs | identical |

### C2. “Bit-for-bit reproduction” is false across thread counts (claim corrected)

**Evidence.** Fold 0, seed 42, identical data/params:

| Member | Setting A | AUC | Best iter | Setting B | AUC | Best iter | max abs pred diff | corr |
|---|---|---|---|---|---|---|---|---|
| LightGBM | `n_jobs=2` | 0.757658 | 1,612 | `n_jobs=-1` | 0.757658 | 1,612 | **0.0** | 1.000000 |
| XGBoost | `n_jobs=2` | 0.759578 | 2,250 | `n_jobs=8` | 0.759029 | 2,134 | **0.0709** | 0.996159 |
| CatBoost | `thread_count=2` | 0.758955 | 1,783 | `thread_count=8` | 0.759364 | 1,938 | **0.2042** | 0.993584 |

The shipped v4 XGB and CatBoost members were built with 8 threads; the Honors notebook uses 2
for Colab. XGBoost's histogram accumulation and CatBoost's parallel statistics are not
associative, so the trees diverge. LightGBM is the only member that was bit-identical. The
notebook therefore cannot reproduce `submission_v4_cv0.76244.csv` byte-for-byte on a different
machine, even with the same seed and data.

**Impact.** Not a validity problem and not a rule violation in substance — the code is genuine
and reproduces the recipe — but the repeated documentation claim “reproduces the exact
submitted CSV” is false and would be embarrassing under Honors review.

**Fix.** The notebook now:
- prints the exact library versions;
- asserts every member OOF is within ±0.004 of the shipped value;
- asserts the pinned blend OOF is within ±0.004 of 0.76244;
- asserts the submission mean/std are close to the shipped file;
- states in its header and in `QUICKSTART.md` that the shipped CSV, not the regenerated one,
  is the authoritative prediction file.

### C3. `01_feature_engineering.ipynb` did not run either (FIXED, verified)

**Evidence.** Same `IntCastingNaNError` as C1, plus the summary cell still contained the raw
`1. Run 02_optuna_tuning.ipynb` lines — a `SyntaxError` — despite `README.md` claiming
“cell 12 syntax error fixed”.

**Fix.** Same QBIN fill; the stale next-steps text was replaced with a valid loop.

**Verification.** The repaired notebook was executed end-to-end locally. It wrote
`train_engineered.parquet` and `test_engineered.parquet` that match the reference files
**cell-for-cell** (0 mismatches), including `feature_cols.json` and `y_train.npy`.

**2026-11-18 follow-up.** The original `_FREQ` block mapped the already label-encoded
integers against a string-keyed frequency dict, so all 8 `*_FREQ` columns were constant
`-999`. They sit in the DEAD list and are dropped before E2, so the submission is unaffected.
`01` now maps the string values and writes real frequencies. A fresh `01 → E2` run therefore
differs from the archived matrix only in `NAN_COUNT`, which drops by a constant 8 (the 8 dead
columns are no longer counted as missing); tree models are invariant to that shift. The Honors
notebook retains the legacy values deliberately, so its rebuilt matrix still matches the
shipped artifact.

---

## Major findings

### M1. `requirements.txt` is unusable on a modern Python (FIXED)

The old pins (`numpy==1.24.3`, `pandas==2.1.3`, `scikit-learn==1.3.2`, `numpy` for Python 3.13)
have no wheels for the current Colab/local Python and do not match the environment that produced
the artifacts. `google-colab==1.0.0` is not needed. The local environment is:

```
Python 3.13.7 · lightgbm 4.7.0 · xgboost 3.4.1 · catboost 1.2.10
scikit-learn 1.8.0 · optuna 5.1.0 · pandas 2.3.2 · numpy 2.2.6
scipy 1.17.1 · pyarrow 21.0.0 · joblib 1.5.3
```

`requirements.txt` now uses working ranges and no longer installs Colab itself. The exact
environment is recorded in `pipeline/ENVIRONMENT.txt`.

### M2. The build evidence lived only in `%TEMP%` (FIXED)

`PROCESS_LOG.md` points at `C:\Users\amanuel\AppData\Local\Temp\opencode\` for
`bench_members2.py`, `rebuild_v3.py`, `rebuild_v4_members.py`, `hgb_fix.py`, the tuned parameter
JSONs, the recipe JSONs, and every OOF/test `.npy`. A Windows temp cleanup would have destroyed
the audit trail. The exact scripts, params, recipes, and audit scripts are now copied into
`pipeline/` (see `pipeline/README.md`). The `.npy`/parquet artifacts remain outside git because
of size, but the Honors notebook can rebuild the matrix and members from raw data.

### M3. HGB's 4% seat is statistically indistinguishable from zero

**Evidence.** v4 with HGB 0.04 = 0.762442; without HGB (renormalised 3-GBM blend) = 0.762436.
Delta = **+0.000007**, 500-rep paired bootstrap 95% CI **[−0.000025, +0.000040]**,
P(>0) = 0.648.

**Verdict.** HGB is a plateau choice, not a discovered gain. It does not hurt, and the shipped
file already contains it; there is no reason to rebuild before the deadline. The docs should
stop describing the 4% weight as a meaningful contribution.

### M4. The published “seat rule” was not the rule actually used

`HANDOFF_TO_GLM.md` and `TRAINING_STAGE_REVIEW.md` state “solo ≥ 0.755 **and** rank-corr ≤ 0.90”.
Yet HGB got a seat at corr **0.9756**, XGB-old got one at corr **0.998**, and LogReg/RF
(corr ~0.88–0.90) were rejected for being weak. The operative rule is “blend contribution”,
which is defensible, but the written rule is post-hoc and inconsistent. Corrected in the current
docs; do not cite the 0.90 threshold as if it were applied uniformly.

### M5. The “~0.775 CV ceiling” claim is unsupported

The competition package has **33 raw features** (34 columns with `TARGET`) — not the full
122-column Kaggle application table, and no bureau/previous-application tables. Eight days of
FE batches, tuning, four GBM families, an NN seat, stacking, rank/logit/power blends, and
pseudo-labeling all failed to exceed **0.76244**. The evidence supports a remaining headroom of
a few 1e-4, not +0.012. The old number should be treated as an aspiration, not a measurement.
The #1 public-LB entry at 1.000 is likewise **unverified** as a “known leak” — it is not needed
for the decision and should not be asserted as fact.

### M6. Enrichment fill values are computed per frame, not carried train → test

`enrich()` uses `filled.fillna(filled.mean())` on whichever frame it receives. Train and test
therefore get different fill values for missing EXT_SOURCE / DAYS_EMPLOYED:

| Feature | Train mean | Test mean | Relative diff |
|---|---|---|---|
| EXT_SOURCE_1 | 0.501692 | 0.490984 | 2.13% |
| EXT_SOURCE_2 | 0.514508 | 0.503577 | 2.12% |
| EXT_SOURCE_3 | 0.511177 | 0.498920 | 2.40% |
| DAYS_BIRTH | 43.9598 | 43.6034 | 0.81% |
| DAYS_EMPLOYED | 6.5455 | 6.3367 | 3.19% |

This is a **protocol deviation, not target leakage**. With adversarial AUC 0.5163 the
distributions are nearly identical, and the fill affects only rows with missing values. The
impact on the submission is negligible. Do **not** “fix” it without rebuilding all members and
re-validating; the current file is internally consistent and LB-tested.

### M7. Documentation contradicted itself (FIXED)

- `QUICKSTART.md` told the user to run `04_final_submission.ipynb` (explicitly forbidden) and
  `03_ensemble_training.ipynb` (contains the original bugs), and promised LGB/XGB/CAT > 0.79
  and an ensemble > 0.80 — impossible on this 33-feature subset.
- `README.md` said “five FLAG_* columns” (the data has three: `FLAG_MOBIL`, `FLAG_EMP_PHONE`,
  `FLAG_WORK_PHONE`), claimed the audit was not delivered, and still said “~2 days left”.
- `TRAINING_STAGE_REVIEW.md` described v3 as pending and the NN seat as pending, both stale.
- `PROCESS_LOG.md` quoted CAT “avg 0.76063” while `v4_recipe.json` says 0.76089; both are true
  but measure different things (mean of per-seed AUCs vs AUC of the seed-averaged OOF). The
  current docs state both explicitly.

The README, QUICKSTART, and PROCESS_LOG have been brought into line with the actual state.

### M8. TE self-inclusion and the DEAD list are validated, not accidental (2026-11-18)

Two code-level challenges were tested under the exact v4 folds, params, and 98-column matrix
(single-seed LightGBM, 5-fold OOF):

| Variant | OOF | Delta vs shipped TE |
|---|---|---|
| fold-mean TE including the row (shipped) | 0.761158 | — |
| nested / inner-OOF TE | 0.761026 | −0.000132 |
| no TE | 0.760795 | −0.000363 |
| leave-one-out TE | 0.526475 | −0.234683 |

LOO is not a candidate: every fold stopped at iteration 1 because the model can invert the
encoding and recover the row's own label. The shipped self-inclusive fold-mean TE is the best
valid option; there is no TE-driven v5.

The 18-feature DEAD list was also challenged by adding features back (same protocol):

| Added back | OOF | Delta vs base |
|---|---|---|
| `ORGANIZATION_TYPE` | 0.759901 | −0.001257 (5/5 folds negative) |
| `FLAG_EMP_PHONE` | 0.760748 | −0.000411 |
| `EXT_SOURCE_3_QBIN` | 0.761089 | −0.000070 |
| large missing indicators | 0.760919 | −0.000239 |
| all of the above | 0.760121 | −0.001037 |

CatBoost was checked separately because it handles high-cardinality categoricals natively:
on fold 0 it reproduced the shipped CAT seed-42 AUC exactly (0.759364), and adding
`ORGANIZATION_TYPE` dropped it to 0.757962 (**−0.001402**). Univariate default rates made
several of these look valuable (`ORGANIZATION_TYPE` ranges 1.9%–16.2%), but they are
redundant/noisy in the full model. The DEAD pruning is empirically correct on both families;
do not re-add them.

### M9. Notebook 02/03/05 defects repaired (2026-11-18)

- `02`: GPU-only params on a CPU pipeline, 3 folds, no E2 enrichment, no TE, `subsample`
  without `subsample_freq`, tuning-time `scale_pos_weight` that final training did not use,
  and a post-hoc `trial.report` that could never prune. Rewritten to CPU, 5 folds, E2 +
  fold-local TE, per-fold pruning, integer cat codes, and CSV/JSON study outputs.
- `03`: removed the unpinned `pip install catboost`, unified the Drive path, embedded the
  frozen E2 params (no dependency on `02`), removed the no-op param override, pinned CatBoost
  to CPU and 2 threads, and cast cat codes to int.
- `05`: fixed the NaN quantile-bin cast, inverted `ReduceLROnPlateau` mode, TabNet `cat_idxs`
  that did not match the reordered matrix, a broken 5-member preview grid, and the missing
  Drive-path fallback.

---

## Minor findings

1. **CAT adoption margin is small but consistent.** New params (single seed 42) beat the old
   params on 4/5 folds: +0.00109, +0.00074, −0.00028, +0.00040, +0.00064; overall
   0.76082 vs 0.76033. Modest, gate-passing, not a game changer.
2. **v4−v3 is 4/5 folds positive.** Per-fold diffs: +0.00041, +0.00030, −0.00004, +0.00011,
   +0.00027. The aggregate +0.000199 is the right number; the one negative fold is within noise.
3. **Weight plateau.** Moving any v4 weight by ±0.06 keeps OOF in 0.76243–0.76244. The pinned
   point is one of many; the decimals in the recipe JSON overstate precision.
4. **Submission Spearman v4 vs v3 = 0.999852**, matching the documented 0.99985. v4 vs v2 =
   0.999502.
5. **Pseudo-labeling is infeasible** on this model: maximum test prediction 0.77167, far below
   any sensible confidence threshold. The earlier rejection is correct.
6. **Notebook hygiene:** `03_ensemble_training.ipynb` and `04_final_submission.ipynb` have
   `.broken-bak` siblings and known-bad cells. They are ignored by git (`*.broken-bak`) but
   remain on disk. Keep them only as history; the headers warn.
7. **Cell comment numbering** in the Honors notebook had two “# 11” sections; the rewrite
   renumbers them.
8. **The Honor's zip cell** assumed the notebook filename existed in the current directory.
   It now searches `/content` and adds `submission.csv` robustly.

---

## Checks that PASSED

| # | Check | Evidence |
|---|---|---|
| 1 | v4 submission reconstructs from saved artifacts | max abs diff **9.975e-17**; corr = 1.0; Spearman = 1.0 |
| 2 | Submission format | 61,500 rows; IDs exactly equal the sample's; no NaNs; range [0.00202, 0.77167]; mean 0.08992, std 0.08913 |
| 3 | v4 / v3 OOF reproduce | 0.762442 / 0.762242 vs documented 0.76244 / 0.76224 |
| 4 | All member OOFs reproduce | LGB 0.76155, XGB 0.76171, CAT 0.76082/0.76044 (seed avg 0.76089), HGB 0.75814 |
| 5 | v4 > v3 | +0.000199; 500-rep paired bootstrap 95% CI [+0.000085, +0.000327]; P(>0) = 1.000 |
| 6 | Target encoding is fold-local | reviewed `bench_members2.py`, `rebuild_v3.py`, `rebuild_v4_members.py`, `hgb_fix.py`, Honors cells 6–10: means from fold-train only; validation/test mapped with those means |
| 7 | Folds consistent | `StratifiedKFold(5, shuffle=True, random_state=42)` in every script and notebook |
| 8 | Parameter completeness (the HGB bug class) | Honors LGB includes `subsample_freq=1`; HGB includes `max_iter=1000`, `early_stopping`, `validation_fraction=0.1`, `n_iter_no_change=25`; XGB/CAT params match the tuned JSONs |
| 9 | CatBoost CPU-only | No `task_type=GPU` anywhere; the GPU trap is documented and avoided |
| 10 | No train/test contamination | ID overlap 0; duplicate feature rows 0; row-order AUC 0.5029 |
| 11 | No distribution shift | Adversarial validation AUC **0.5163** (5-fold LightGBM) |
| 12 | NN results | MLP OOF 0.74897 (rank-corr 0.9242), TabNet all-zero / AUC 0.50000 — confirmed |
| 13 | No hidden ID/order leak | `SK_ID_CURR` is contiguous 0…232,701; train/test split is sequential; AUC of ID as a predictor 0.5029 |
| 14 | Blend variant search | Probability 0.762442, logit 0.762440, rank 0.762433, power p^0.75 0.762444; best-logit minus v4 = −0.000001 (95% CI −0.000038…+0.000035) — no real gain |
| 15 | Weight plateau | ±0.06 on any member keeps OOF at 0.76243–0.76244; HGB 0.00 vs 0.04 differs by 0.000007 |
| 16 | Rules: no external data / no hand labeling | Only the competition CSVs; all predictions model-generated; NNs trained from scratch |
| 17 | Honors feature parity | Repaired notebook reproduces the 98-column matrix exactly (C1 table) |
| 18 | `01_feature_engineering.ipynb` | Repaired notebook runs end-to-end; output parquet matches the reference cell-for-cell |

**Not verified / explicitly UNVERIFIABLE at audit time**

- Full 5-fold Honors notebook rerun on Colab (budget 3.5–4 h; the feature parity, member params,
  folds, TE, and blend were verified statically and numerically).
- The private-LB score (only the public LB is visible).
- The claim that the public-LB #1 entry at 1.000 is a leak/hand-labeling artifact.

The CatBoost thread test was initially unverified but completed afterwards: see C2.

---

## Optimism correction for CV 0.76244

The reported CV is the **maximum** of a weight search on the same OOF used to score it. With
four members, a hill-climb, and 0.01–0.05 grids, the effective selection is small relative to
171,202 rows. A reasonable optimism correction is **0.0001–0.0002**. That still leaves v4 at
~0.7622–0.7623, above v3's 0.76224 only marginally — consistent with the observed +0.0002 and
with the LB evidence (v2 0.76433 → v4 0.76475). Do not claim more precision than that.

---

## Final-upload recommendation

**Keep `submission_v4_cv0.76244.csv` as the last prediction file.** Reasoning:

1. v4 is the best validated recipe: CV 0.76244, LB 0.76475, and it reconstructs exactly from
   saved artifacts.
2. Its edge over v3 is small (+0.0002) but positive in 4/5 folds and in the bootstrap; the LB
   point confirms the direction.
3. The fallback v3 (CV 0.76224) is already on disk and is the right choice only if v4 cannot be
   re-uploaded.
4. Do **not** upload a logit/rank/power blend variant: all score ≤0.762444 CV, i.e. no gain.
5. Do **not** upload the Honors notebook's regenerated `submission.csv`: XGBoost/CatBoost
   thread drift makes it a slightly different file. Upload the notebook as the Honors **code**
   artifact and the shipped v4 CSV as the **prediction** artifact.
6. If any file was uploaded after v4, re-upload v4 at the end. Last file counts.

---

## Appendix — audit scripts

The checks above are reproducible from `pipeline/audit/`:

| Script | What it does |
|---|---|
| `audit_deep.py` | Rebuilds v4/v3, bootstrap, correlations, submission cross-checks, weight sensitivity |
| `audit_honors_fe.py` | Executes the Honors feature cells and diffs against `X_e2` / `X_test_e2` |
| `audit_data.py` | ID/order/duplicate checks + adversarial validation |
| `audit_extra.py` | HGB contribution bootstrap, CAT per-fold, submission correlations, enrich means |
| `audit_threads.py` | Fold-0 LightGBM/XGBoost/CatBoost thread-count determinism (output in `thread_test_output.txt`) |
| `run_nb1.py` | Executes `01_feature_engineering.ipynb` headlessly and diffs its parquet output |

They retain the `%TEMP%` paths of the development session. Point them at the data files to re-run.
