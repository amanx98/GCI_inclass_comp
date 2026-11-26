# Submission history

Do not rename or edit these files; the filenames carry the CV score on purpose.
All four have 61,500 prediction rows and use the sample-submission IDs in order.

| File | CV (OOF) | Public LB | Status |
|---|---:|---:|---|
| `submission_anchor_blend_v1.csv` | 0.7587 | 0.7627 | Historical anchor (0.30/0.55/0.15 GBM blend). Superseded. |
| `submission_v2_recipe_cv0.76203.csv` | 0.76203 | 0.76433–0.76435 | Historical. Two uploads of this recipe scored 0.76433 and 0.76435. |
| `submission_v3_recipe_cv0.76224.csv` | 0.76224 | — | Fallback. Not separately uploaded. |
| **`submission_v4_cv0.76244.csv`** | **0.76244** | **0.76475** | **Current best. Keep as the last uploaded file.** |

## Rules that matter here

- **The last uploaded file counts.** If anything is uploaded after v4, re-upload v4 before
  the deadline (2026-11-20 01:00 UTC).
- **Upload the shipped v4 CSV, not the Honor's regenerated `submission.csv`.** The notebook
  reproduces the recipe and CV within ±0.004, but XGBoost and CatBoost are thread-count-sensitive,
  so the regenerated predictions are slightly different. The v4 file here is the one that
  scored 0.76475.
- **The Honors notebook is the code artifact.** Its output CSV is for demonstration only.
- v4 vs v3 Spearman is 0.999852; the improvement is real but small (+0.0002 CV), so there is
  no reason to gamble on a new recipe before the deadline.

See [`../AUDIT_FINDINGS.md`](../AUDIT_FINDINGS.md) for the numeric verification (v4
reconstructs from the saved OOF/test arrays to `max|diff| = 9.98e-17`).
