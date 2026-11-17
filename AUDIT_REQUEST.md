# AUDIT REQUEST — Devil's Advocate Review (GCI 2026 Pipeline)

**To:** an independent AI agent (fresh context, no stake in the outcome)
**Subject:** adversarial audit of the competition pipeline described in `PROCESS_LOG.md`,
`TRAINING_STAGE_REVIEW.md`, and the artifacts in this workspace.

Your job is NOT to be helpful or agreeable. Your job is to **find flaws**. Assume something
is wrong and try to prove it. If the pipeline is actually sound, demonstrate that by
showing the specific checks you ran — not by saying "looks good."

---

## 1. What you have access to

- `PROCESS_LOG.md` — full chronological record with numbers and artifact names
- `TRAINING_STAGE_REVIEW.md` — stage-by-stage rationale
- Notebooks: `gci_honors_reproduction.ipynb` (final repro code), `05_nn_diversity.ipynb`,
  `03_ensemble_training_v2.ipynb`
- Scripts (session temp dir, `C:\Users\amanuel\AppData\Local\Temp\opencode\` unless noted):
  `bench_members2.py`, `bench_test_preds.py`, `rebuild_v3.py`, `overnight_retune.py`,
  `overnight_retune2.py`, `rebuild_v4_members.py`, `hgb_fix.py`
- Params: `best_params_{lgb,xgb,cat,hgb}_e2.json`
- Data matrices: `X_e2.parquet`, `X_test_e2.parquet`, `drive/GCI Competition/features/*`
- Predictions: `m*_*.npy`, `v3_test_*.npy`, `v4_CATnew_*.npy`, `v4_HGBnew_*.npy`
- Submission files: `submissions/submission_v3_recipe_cv0.76224.csv`,
  `submission_v4_cv0.76244.csv`
- Recipe: `v4_recipe.json` (weights LGB .30 / XGBn .33 / CATn .33 / HGB .04)

Competition rules (from `competition_unzipped/04. Competition (due Nov 20)/Competition/README.ipynb`):
no external data, no hand-labeling, fixed seeds, Honors requires code reproducing the
submitted CSV, last submitted file counts, cite referenced public code.

---

## 2. Attack surfaces — check ALL of these, in this order

### A. Leakage / protocol integrity (highest priority)
1. **Target encoding:** verify in every training path that TE means are computed from the
   fold-train split only, and that validation/test mapping uses those fold-local means.
   Check `bench_members2.py`, `bench_test_preds.py`, `rebuild_v4_members.py`, `hgb_fix.py`,
   and the honors notebook cell 6 for consistency. Any global TE = leakage → CV inflated.
2. **Enrichment fill values:** `enrich()` fills NaN EXT sources with column means. For the
   TEST frame, are those means computed on test itself (train/test mismatch) or carried
   from train? Is the same true for `age_years` fill and `emp_years` fill? Quantify the
   impact direction (CV vs test distribution shift).
3. **OOF-selection circularity:** weights were selected on the same OOFs used to report CV.
   This is accepted practice but IS an overfitting channel. Assess: how many effective
   degrees of freedom in the weight search (hill-climb iterations + grids) vs 171k rows?
   Estimate an honest optimism correction for CV 0.76244.
4. **Tuning-fold asymmetry:** CAT tuning searched on fold-0 and validated on full CV — but
   the top-3 selection still used the same 5 folds the ensemble is scored on. Quantify.

### B. Reproducibility (Honors requirement — any failure here is fatal)
5. **Notebook vs submitted CSV:** simulate mentally (or actually run if compute allows)
   whether `gci_honors_reproduction.ipynb` regenerates `submission_v4_cv0.76244.csv` bit-
   for-bit (or within float noise). Check: identical feature list order, identical TE
   smoothing, identical params (all of them, including non-searched constants like
   `max_iter=1000`, `n_iter_no_change=25`), identical seeds, identical weight vector and
   member ORDER in the weight vector.
6. **Known past bug class:** `hgb_fix.py` exists because a params JSON omitted non-searched
   constants. AUDIT THE NOTEBOOK FOR THE SAME BUG CLASS — e.g., does the CAT cell carry
   `thread_count=2` (Colab) vs 8 (local) — is `thread_count` seed-relevant in CatBoost?
   Does XGB `n_jobs=2` vs `n_jobs=-1` change results (hist method is deterministic in
   tree structure but check subsample seeding)?
7. **Environment pinning:** the notebook records library versions but does it pin them?
   On a fresh Colab runtime (late 2025/2026 default versions), would LightGBM/XGBoost/
   CatBoost/sklearn behave identically? Flag any version-sensitive behavior (e.g.,
   `iteration_range`, `early_stopping` API changes).

### C. Statistical validity of decisions
8. **Seat rule consistency:** the rule says solo ≥0.755 AND rank-corr ≤0.90, yet HGB got a
   seat at 0.75814 with rank-corr 0.9756, and XGBold (0.998 corr) had a seat in v3. Is the
   rule applied selectively post-hoc? Is there a principled distinction, or is "blend
   contribution" doing all the work? Is that defensible?
9. **Plateau weights:** multiple weightings give 0.76244. The pinned choice (.30/.33/.33/
   .04) is one of many near-equivalent points. Is HGB's 4% seat distinguishable from noise?
   Propose a concrete test (e.g., paired bootstrap over rows on the OOF diff vs v3).
10. **CAT adoption margin:** 0.76088 vs 0.76033 baseline (+0.00055) — is this outside
    fold-to-fold noise? Check per-fold AUCs if available.
11. **v2→v3→v4 progression:** v3 was adopted on +0.0002 CV over v2; v4 on +0.0002 over
    (a re-scored) v3. Are these within selection noise? Does the LB evidence (v2 = 0.7643)
    support or contradict the increments?

### D. Rules compliance (adversarial reading)
12. **"No external data":** frequency encodings and quantile bins are fit on
    train+test COMBINED in some paths (check the notebook: label encoders fit on
    concatenated train+test, freq maps from combined counts). Is fitting unsupervised
    statistics on test a rules problem in this competition's reading? Is it a leak of
    test distribution into features (transductive learning)? State a verdict either way.
13. **NAN_COUNT / WAS_MISSING flags:** fine on principle, but check none encode TARGET-
    correlated test-specific artifacts.
14. **Colab vs local env claims:** all tuning was done locally on CPU; the submission
    notebook runs on Colab CPU. If thread count or BLAS differences change results even
    slightly, the "reproduces the CSV" claim weakens. How brittle is it, honestly?

### E. Strategic sanity
15. Is uploading v4 (never LB-tested, CV +0.0002 over v3) strictly better than re-uploading
    v3 (whose recipe family has an LB point at 0.7643)? Given "last file counts," what is
    the optimal final-upload decision under uncertainty?
16. Was anything cheap left untried that the timeline allowed (e.g., 5-seed expansion,
    LGB/XGB 100+ trial extension)? Was the stopping decision principled?

---

## 3. Required output format

Write your findings to `AUDIT_FINDINGS.md` in this workspace, structured as:

```
## Verdict summary
(one paragraph: is the pipeline sound, what is the single biggest risk)

## Critical findings   (would invalidate the submission / Honors)
[numbered, each with: evidence file+line/numbers, severity, recommended fix]

## Major findings      (weaken claims, worth fixing if time allows)
## Minor findings      (note for the record)
## Checks that PASSED  (list every attack you ran that failed, with the numbers)
## Final-upload recommendation (v4 vs v3, with reasoning)
```

Severity scale:
- **Critical** = CV is inflated, code doesn't reproduce the CSV, or a rule is violated
- **Major** = a claim in PROCESS_LOG.md is overstated or a decision rests on noise
- **Minor** = documentation/hygiene

## 4. Ground rules for you (the auditor)

- Do NOT rerun full training unless something specific looks broken (deadline Nov 20).
  Prefer static analysis + cheap numeric checks on the saved npy/csv artifacts
  (correlations, per-fold AUCs, weight-sensitivity bootstrap).
- Cite the exact file and the exact numbers for every claim you make, in both directions.
- If you cannot verify something with available artifacts, say so explicitly and mark it
  UNVERIFIABLE rather than assuming it's fine.
- Your credibility comes from finding real problems AND from not inventing fake ones.
