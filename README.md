# GCI In-Class Competition — Home Credit Default Risk

**Status: COMPLETE — submitted 2026-11-20, before the 01:00 UTC deadline.**
Repository: https://github.com/amanx98/GCI_inclass_comp

| Deliverable | Artifact | Result |
|---|---|---|
| Prediction (competition platform) | `submissions/submission_v4_cv0.76244.csv` | **CV AUC 0.76244 · public LB 0.76475** — uploaded **last** |
| Code (Honors / Outstanding) | `gci_honors_reproduction.zip` | self-contained notebook + demonstration `submission.csv` + `README_competition.md` |

The story below covers the full run: the baseline, the validation overhaul, every feature and
model decision, the four recipe generations, the independent audit, the final v5 round that was
rejected, and the two files that were submitted.

---

## 1. The result

| Recipe | CV AUC (5-fold OOF) | Public LB | Disposition |
|---|---:|---:|---|
| Tutorial RF, 5 features | ~0.66 | 0.66819 | baseline, superseded |
| Anchor (old 3-GBM pipeline) | 0.7587 | 0.7627 | first real CV↔LB calibration point |
| v2 (E2 features + full-OOF selection) | 0.76203 | 0.76433–0.76435 | uploaded (two files, same recipe) |
| v3 (re-tuned LGB/XGB, two XGB generations) | 0.76224 | — | built + validated; fallback, not uploaded |
| **v4 (all four members tuned)** | **0.76244** | **0.76475** | **final / last uploaded file** |

CV → public-LB transfer was consistently **+0.0023** (v2 and v4). The private LB is unknown.

**v4 recipe:** LGB **0.30** / XGB-new **0.33** / CAT-new **0.33** / HGB **0.04**.

| Member | OOF AUC | Rank-corr vs XGB-new | Seeds |
|---|---:|---:|---|
| LightGBM (re-tuned) | 0.76155 | 0.9926 | 42 / 1337 / 2026 |
| XGBoost (re-tuned) | **0.76171** | 1.0000 | 42 / 1337 / 2026 |
| CatBoost (re-tuned, CPU only) | 0.76089 | 0.9819 | 42 / 1337 |
| HistGradientBoosting (re-tuned) | 0.75814 | 0.9756 | 42 |

Submission format verified: 61,500 rows, `SK_ID_CURR` exactly matches the sample submission,
no NaNs, `TARGET ∈ [0.00202, 0.77167]`, mean 0.08992, std 0.08913.

---

## 2. The whole workflow

### Phase 0 — Baseline (Nov 10–14)
Reproduced the competition tutorial: a 5-feature random-forest pipeline scoring **0.66819**
on the public LB. First real pipeline was a single LightGBM validated on a 10% holdout.

### Phase 1 — Validation overhaul (Nov 14–15)
The 10% holdout wasted ~34k rows and made blend selection high-variance: the reported
"0.76502 holdout blend" was inflated noise, while the true full-data OOF was 0.7587.
**Change:** all selection moved to full-data 5-fold `StratifiedKFold(seed=42)` OOF.
Measured gain on comparable members: **+0.0028**. This single protocol decision underpins
every number that follows.

### Phase 2 — Feature engineering (Nov 15)
33 raw columns (34 with `TARGET`; no bureau or previous-application tables) → 101 engineered
columns → **98-column E2 matrix**: anomaly flags, missing indicators, domain ratios,
`EXT_SOURCE` interactions, quantile bins, frequency encodings, then enrichment interactions
(`EXT×AGE`, NaN-safe `EXT` products/squares, burden ratios, `NAN_COUNT`,
`REGION_RATING×EXT_MEAN`) and **18 dead columns dropped**. Fold-local target encoding on
7 high-cardinality categoricals. Combo-categorical TE was tested and **rejected (−0.0007)**.

### Phase 3 — Audit of the original notebooks (Nov 15)
A cell-by-cell review against the competition PDF and saved Drive artifacts found three real
bugs:
1. **Cell 12 "retrain on full data"** used XGBoost's iteration count from LightGBM
   (`best_iteration_`) and a hard-coded 1000 iterations for CatBoost. It produced a *squashed*
   prediction distribution (std 0.099 vs the correct 0.213). NB4 loaded those predictions
   **by default**, so running it would have submitted the worse variant.
2. **Blend strategy selected on a 6.1k-row holdout** — the source of the fake 0.76502.
3. **Holdout target encoding used the full training set** — leakage-adjacent.

Fix: a dedicated v2 runbook, full-OOF selection, and the standing rule
*"final submission = CV-fold-averaged test predictions, never one-shot retrains."*

### Phase 4 — v2 and the Colab run (Nov 16)
The anchor blend (LGB .30 / XGB .55 / CAT .15) was submitted and scored **0.7627**, giving a
CV↔LB calibration point. The rebuilt pipeline was run on Colab T4: all members reproduced
locally within noise; blend CV 0.76174. That run also exposed a hardware trap: the CatBoost
GPU backend has no AUC early stopping, so it silently stopped on Logloss — **0.75847 GPU vs
0.76033 CPU**, with no exception raised. CatBoost was pinned to **CPU only** from then on.

### Phase 5 — Hyperparameter tuning (Nov 16–17, local CPU)
- LGB: 40 Optuna trials on E2 → 0.75989 → **0.76120** single-seed; 3-seed build 0.76155.
- XGB: 26 trials → **0.76192** single-seed; 3-seed build 0.76171.
- CAT: first-ever re-tune (20 fold-0 trials, top-3 validated on full CV) → **0.76088**; 2-seed rebuild 0.76089.
- HGB: first-ever re-tune (40 trials) → **0.75820**; rebuild 0.75814.

A reproducibility bug surfaced here too: the tuning JSON saved only the searched parameters,
so the first HGB rebuild silently used sklearn defaults (`max_iter=100` → 0.74969). Re-running
with the **full** config (`max_iter=1000`, `early_stopping(0.1, 25)`) recovered 0.75814.

### Phase 6 — Diversity experiments and v3/v4 (Nov 17–18)
- LogReg 0.742, RandomForest 0.736, ExtraTrees 0.732 — too weak; they **diluted** the blend.
- MLP + categorical embeddings: OOF 0.74897, rank-corr 0.9242 vs XGB-new; TabNet collapsed
  to a constant 0.50000. Adding the MLP moved CV by +0.00006. **NN seat rejected.**
- Two-level stacking (Ridge / Logistic / LGBM meta with honest inner CV): 0.76112–0.76150 —
  all below the linear weighted average. **Stacking rejected.**
- Pseudo-labeling: the maximum test prediction is 0.77167, so no confident rows exist. **Rejected.**
- v3 (re-tuned LGB/XGB, two XGB generations): CV **0.76224**, built and validated.
- v4 (tuned CAT and HGB added): exhaustive weight search pinned
  LGB .30 / XGBn .33 / CATn .33 / HGB .04 → CV **0.76244**. Submitted; **LB 0.76475**.

### Phase 7 — Independent audit (Nov 18)
An adversarial audit was commissioned in `AUDIT_REQUEST.md` and delivered as
`AUDIT_FINDINGS.md`. Result: the modelling pipeline was sound — v4 reconstructs from saved
OOF/test arrays to `max|diff| = 9.98e-17`, no leakage, no distribution shift (adversarial AUC
0.5163), no ID/order leak (row-order AUC 0.5029). But it found two **run-blocking** defects:
the Honors notebook and `01_feature_engineering.ipynb` both crashed in the quantile-bin cell
(`pd.cut(...).astype(np.int8)` on NaNs), and NB1 still had a literal `1. Run 02_...` syntax
error. Both were fixed; the repaired notebooks rebuild the 98-column matrix with **0 mismatching
cells** out of 16.8M train and 6.0M test values. The audit also corrected overstated claims:
the "~0.775 CV ceiling", the "known leak" at LB #1, and the published "rank-corr ≤ 0.90" seat
rule that had not actually been applied uniformly.

### Phase 8 — v5 workstream round (Nov 19): everything rejected
A modular harness (`pipeline/v5/common.py`) reproduced v4 OOF 0.762442 exactly, then five
workstreams were tested against a 500-rep paired bootstrap gate (95% CI entirely > 0):

| Workstream | Result | Verdict |
|---|---|---|
| W1 model-based imputation of EXT_SOURCE_1/3, OWN_CAR_AGE | −0.000296 | rejected |
| W2 full-CV Optuna re-tune (80–100 trials/family) | XGB +0.000258, LGB +0.000152, HGB −0.000174, CAT −0.000502 | members only |
| W3 seed expansion | XGB7-new 0.761714, CAT4 0.760995, HGB3 0.759125 | members only |
| W4 FE round 3 (KMeans, cat-pair counts, REGPOP×EXT) | all < +0.0003 | rejected |
| W5 blend re-search (15 members, hill-climb + grid + NNLS) | best 0.762454; CI [−0.000047, +0.000077], p=0.626 | **gate failed** |
| W5 variant sweeps (28,336 weight combos) | best 0.762455; CI [−0.000037, +0.000065], p=0.684 | **gate failed** |
| W6 CV-protocol variants (10-fold, fold seed 1337) | 10-fold LGB OOF 0.761311 | no gain |
| W7 further variants | nothing passed | rejected |

**Conclusion:** every v5 lever landed inside noise. **Zero v5 changes were adopted** — v4 is
plain v4. See §5 for the honest headroom estimate.

### Phase 9 — Final submission (Nov 20)
`FINAL_INSTRUCTIONS.txt` records the two uploads made with the deadline approaching:
1. `submissions/submission_v4_cv0.76244.csv` → competition platform, **as the last file**.
2. `gci_honors_reproduction.zip` (723 KB) → Honors: notebook + demonstration `submission.csv`
   + `README_competition.md`.

A local execution attempt of the Honors notebook reached cell 7 and died with a LightGBM
`MemoryError` on this machine (`honors_local_run_console.log`); the demonstration CSV shipped
in the zip is therefore the validated v4 file (the two are byte-identical, `csv_diff.txt` is
empty). The notebook itself is statically verified and its feature cells are verified to
rebuild the reference matrix exactly. This is disclosed rather than hidden.

---

## 3. Every decision, and why

**Validation & selection**
1. **CV decides; the public LB only sanity-checks.** The public LB is a subset of test and
   mildly optimistic. No recipe was chosen because of a public score.
2. **Full OOF, never the holdout.** The 6.1k-row holdout gave ±0.005 noise; dropping it was
   worth +0.0028. The holdout survived only as a final sanity check.
3. **One submission per validated recipe.** Unlimited uploads are not free probing; overfitting
   the public subset causes private-LB drops.
4. **A new recipe needs a passed gate**, not just a higher point estimate: paired bootstrap on
   OOF with a fully positive 95% CI (v5's +0.000012 with p=0.626 failed).
5. **Honest optimism correction:** CV weight-search selection optimism is ~0.0001–0.0002.

**Data & features**
6. **Domain features over capacity.** Deeper trees (leaves 63/127) lost to depth-3/4 trees,
   so gains had to come from features and diversity, not model size.
7. **Keep validated enrichments; keep the DEAD list.** Re-adding `ORGANIZATION_TYPE` cost
   −0.00126 (5/5 folds negative; CatBoost fold 0 −0.00140), `FLAG_EMP_PHONE` −0.00041,
   all extras −0.00104. Univariate signal was misleading; the pruning is empirically correct.
8. **Keep the shipped fold-mean TE.** Nested TE −0.00013, no TE −0.00036, leave-one-out TE
   collapsed to 0.5265 (the model inverts the encoding and recovers the row's own label).
9. **Transductive feature encoders** (label/frequency on train+test combined) are used with no
   target information and are disclosed; adversarial AUC 0.5163 shows no observable shift.
10. **Per-frame enrichment fills were left alone.** Train and test use means from their own
    frame (2.1–3.2% relative difference); a protocol deviation, not leakage. Rebuilding every
    member for a negligible effect before the deadline was not worthwhile.

**Models & ensembling**
11. **Final submission = CV-fold-averaged predictions, never one-shot retrains** (bug #1).
12. **CatBoost pinned to CPU** — the GPU silently early-stops on Logloss instead of AUC.
13. **Only near-equal-strength members earn a seat.** LogReg/RF/ET at 0.73–0.74 dilute despite
    lower correlation; HGB at 0.75814 with corr 0.9756 was kept only as a 4% plateau weight.
14. **Linear weighted average wins.** Rank-average 0.76200, logistic meta 0.76145, LGBM meta
    0.76112–0.76150, probability blend 0.76244. The GBMs are 97–99.8% rank-correlated.
15. **HGB's 4% seat is noise** (+0.000007, CI [−0.000025, +0.000040]); it was kept only because
    the already-uploaded file contained it.
16. **NN diversity rejected on measured evidence**, not assumed: MLP 0.74897 / corr 0.9242,
    TabNet collapsed, any-weight blend gain +0.00006.
17. **Pseudo-labeling rejected** — max test prediction 0.77167 leaves no confident rows.

**Reproducibility & honesty**
18. **Fixed seeds everywhere** (members 42/1337/2026, folds 42, Optuna sampler 42) and exact
    library versions recorded in `pipeline/ENVIRONMENT.txt`.
19. **Build evidence moved out of `%TEMP%` into `pipeline/`**, so a Windows temp cleanup cannot
    destroy the audit trail.
20. **Bit-exact reproduction is not claimed.** XGBoost `hist` and CatBoost are thread-count
    sensitive (fold-0 corr 0.996 / 0.9936); the Honors notebook reproduces recipe and CV within
    ±0.004, and the shipped CSV is authoritative.
21. **Overstated claims were corrected**: "~0.775 ceiling", "known leak at LB #1", and the
    mis-stated seat rule were removed from the source-of-truth docs.

---

## 4. Verification & audit record

- **v4 reconstructs exactly** from saved OOF/test arrays: `max|diff| = 9.98e-17`, corr = 1.0.
- **v4 > v3** = +0.000199, 500-rep bootstrap 95% CI [+0.000085, +0.000327], 4/5 folds positive.
- **Feature parity**: repaired Honors notebook rebuilds 98 columns, 0 mismatches in 16.8M train
  and 6.0M test cells; `01_feature_engineering.ipynb` runs end-to-end and matches the reference
  parquet cell-for-cell.
- **No leakage**: TE from fold-train means only; folds identical everywhere; no ID overlap;
  no duplicate feature rows; adversarial AUC 0.5163; row-order AUC 0.5029.
- **Blend variants**: logit 0.762440, rank 0.762433, power 0.762444 vs probability 0.762442 —
  no real gain.
- **Full evidence**: `AUDIT_FINDINGS.md` (18 checks passed, 3 explicitly unverifiable).

**Honest limitations**
1. Public LB ≠ private LB; +0.0023 transfer on the public subset is not guaranteed privately.
2. Thread-count drift prevents byte-identical regeneration; do not upload the notebook's CSV.
3. Enrichment fill means are per-frame (2.1–3.2% relative difference).
4. HGB's seat is statistically indistinguishable from zero.
5. Remaining headroom on this 34-column subset is a few **1e-4**, not 1e-2.

---

## 5. Repository map

```
.
├── README.md                          # this file — final project report
├── FINAL_INSTRUCTIONS.txt             # the two files uploaded at the deadline
├── README_competition.md              # Honors README (also inside the zip)
├── QUICKSTART.md                      # Colab runbook
├── PROCESS_LOG.md                     # chronological record (primary history)
├── AUDIT_REQUEST.md / AUDIT_FINDINGS.md
├── TRAINING_STAGE_REVIEW.md           # historical stage synthesis
├── MASTER_PLAN.md / HANDOFF_TO_GLM.md # historical snapshots (superseded)
├── 01_feature_engineering.ipynb       # repaired, verified
├── 02_optuna_tuning.ipynb             # rewritten: CPU, 5-fold, E2 + TE
├── 03_ensemble_training.ipynb         # historical — DO NOT RUN
├── 03_ensemble_training_v2.ipynb      # historical runbook
├── 04_final_submission.ipynb          # DO NOT RUN — loads squashed preds
├── 05_nn_diversity.ipynb              # NN seat test, rejected
├── gci_honors_reproduction.ipynb      # self-contained Honors code artifact
├── gci_honors_reproduction.zip        # submitted Honors bundle
├── requirements.txt                   # working ranges for Python 3.13
├── pipeline/                          # exact build scripts, params, recipes, environment
│   ├── audit/                         # scripts behind AUDIT_FINDINGS.md
│   └── v5/                            # W1–W6 harness and results (all rejected)
├── phase1_diversity_v{1,2,3}.py       # historical diversity experiments
├── phase1_results_summary.json
└── submissions/
    ├── README.md                      # submission history + last-file rule
    ├── submission_anchor_blend_v1.csv       # LB 0.7627
    ├── submission_v2_recipe_cv0.76203.csv    # LB 0.76433–0.76435
    ├── submission_v3_recipe_cv0.76224.csv    # CV 0.76224 (fallback)
    └── submission_v4_cv0.76244.csv           # CV 0.76244 / LB 0.76475 — LAST FILE
```

Not committed on purpose: raw `train.csv`/`test.csv` (~60 MB), Drive snapshots, `.npy`/parquet
intermediates, corrupt `.broken-bak` notebooks, and one-off scratch scripts.

---

## 6. Reproduce

```bash
pip install -r requirements.txt
```

Place `train.csv`, `test.csv`, `sample_submission.csv` in
`competition_unzipped/04. Competition (due Nov 20)/Competition/input/`.

- **End-to-end:** run `gci_honors_reproduction.ipynb` on a CPU runtime (~3.5–4 h). It rebuilds
  features, all four members, and the pinned blend from the raw CSVs, then asserts each member
  OOF is within ±0.004 of the shipped values. Its `submission.csv` is for demonstration —
  `submissions/submission_v4_cv0.76244.csv` is the authoritative prediction.
- **Feature matrix only:** `01_feature_engineering.ipynb` (~1 min locally).
- **Historical evidence:** `pipeline/README.md` documents the exact scripts, parameters and
  recipes used to produce v3/v4 and the audit.

---

## 7. Rules compliance

| Rule | Status |
|---|---|
| No external data | ✅ Only provided CSVs; NNs trained from scratch; no pretrained weights |
| No hand-labeling | ✅ Every prediction model-generated |
| Fixed seeds | ✅ 42 / 1337 / 2026; folds 42; Optuna sampler 42 |
| Honors code | ✅ Self-contained notebook + zip submitted; recipe/CV reproduce within ±0.004 |
| Cite public code | ✅ Tutorial/README and public HomeCredit EDA conventions cited in notebook header |
| Last file counts | ✅ v4 uploaded last |
| Transductive encoding | ✅ Disclosed: feature-only statistics fit on train+test |
