# GCI In-Class Competition — Home Credit Default Risk

## Progress Summary (as of 2026-11-20 03:38 EAT)

**Deadline:** 2026-11-20 01:00 UTC = 04:00 EAT (32 minutes remaining)

**Current Best Submission:** submission_v4_cv0.76244.csv
- CV AUC: **0.76244**
- Public LB AUC: **0.76475**
- 61,500 predictions rows

### What We Did
1. **Built the end-to-end pipeline** from raw CSVs → 98-column feature matrix
   - 33 raw features → 101 engineered → 98 E2 (15 enrichments, 18 dead drops)
   - Fold-safe target encoding on 7 high-cardinality columns
   - LightGBM / XGBoost / CatBoost / HGB with Optuna-tuned parameters

2. **v2 → v3 → v4 progression**
   - v2: CV 0.76203 → LB 0.76433–0.76435
   - v3: CV 0.76224 → (fallback)
   - v4: CV 0.76244 → LB 0.76475 (best, last file)

3. **v5 experiment round (Nov 19) — ALL REJECTED**
   - W1 imputation: −0.0003 (rejected)
   - W2 re-tune: XGB +0.00026, others negative (kept as members only)
   - W3 seed expansion: new members created
   - W4 FE round 3: all < +0.0003 rejected
   - W5 blend re-search: best +0.000012, **bootstrap gate FAILED** (p=0.63)
   - W6/W7: nothing passed the gate

**Bottom line:** v4 is at the measured ceiling on the 34-column subset. Every v5 lever landed inside noise.

### Honors Artifact
- gci_honors_reproduction.zip contains the single self-contained Honors notebook
- The notebook is CPU-only; reproduces the recipe and CV within ±0.004
- XGBoost/CatBoost thread drift prevents byte-identical CSV; the shipped v4 CSV is authoritative
- The notebook assert every member OOF within ±0.004 of shipped values

### Final Day Checklist
- [x] Competition: upload submission_v4_cv0.76244.csv as LAST prediction file
- [x] Honors: upload gci_honors_reproduction.zip as code artifact
- [x] v4 is the best validated recipe (reconstructs from saved OOF/test arrays to max|diff| = 9.98e-17)
- [ ] Remaining headroom is a few 1e-4, not 1e-2

## Repository
GitHub: https://github.com/amanx98/GCI_inclass_comp

## Rules Compliance
- ✅ No external data — only provided train/test CSVs
- ✅ No hand-labeling — all predictions model-generated
- ✅ Fixed seeds: 42, 1337, 2026
- ✅ Last file counts: v4 uploaded last
- ✅ Honors code reproduces recipe/CV within tolerance (submitted CSV is authoritative)
