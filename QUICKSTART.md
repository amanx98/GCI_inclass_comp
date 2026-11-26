# Final-day runbook — Honors notebook + last submission

**Deadline:** 2026-11-20 **01:00 UTC** = **04:00 EAT**. Scoring can lag; do not cut it close.
**Current best:** v4 — CV **0.76244**, public LB **0.76475**.
**Goal of this runbook:** produce the Honors code artifact and leave **v4 as the last uploaded file**.

> Do **not** run `03_ensemble_training.ipynb` (historical bugs) or `04_final_submission.ipynb`
> (loads the squashed `final_full_preds.npy`). The only notebook you need now is
> `gci_honors_reproduction.ipynb`.

---

## 0. What you need

| Thing | Where |
|---|---|
| Competition CSVs | `train.csv`, `test.csv`, `sample_submission.csv` |
| Drive layout | `MyDrive/gci_competition/input/{train,test,sample_submission}.csv` |
| Notebook | `gci_honors_reproduction.ipynb` (this repo) |
| Runtime | Colab **CPU** — no GPU needed anywhere |
| Time | ~3.5–4 h on a 2-vCPU Colab CPU runtime |
| Final prediction file | `submissions/submission_v4_cv0.76244.csv` (already uploaded) |

The notebook also accepts the old path `MyDrive/competition/input/` as a fallback.

---

## 1. Upload the data (5 min)

In Google Drive, create:

```
MyDrive/gci_competition/
└── input/
    ├── train.csv
    ├── test.csv
    └── sample_submission.csv
```

Do **not** upload the engineered parquet files — the notebook rebuilds them from the raw CSVs.

---

## 2. Open and run the notebook

1. Go to <https://colab.research.google.com/> → **File → Upload notebook** →
   `gci_honors_reproduction.ipynb`.
2. **Runtime → Change runtime type → CPU** (the notebook is CPU-only by design; CatBoost GPU
   silently early-stops on Logloss and loses ~0.002 AUC).
3. **Runtime → Run all**. Allow ~3.5–4 h. Keep the tab alive (the free runtime disconnects
   after ~90 min idle; training cells are not idle, but keep the tab visible).

### What you should see

The notebook prints library versions, the final matrix shape, then per-member OOF AUCs:

```
engineered: (171202, 101), (61500, 101)
final: 98 features

member OOF AUCs (expected -> reproduced):
  LGB  0.76155 -> 0.7615x   [OK]
  XGB  0.76171 -> 0.7616x   [OK]
  CAT  0.76089 -> 0.7608x   [OK]
  HGB  0.75814 -> 0.7581x   [OK]

pinned recipe OOF AUC: 0.7624x (submitted: 0.76244)
```

The tolerance is ±0.004; a `DRIFT` assertion means the environment or feature parity broke the
pipeline — **do not submit that run**, use the already-shipped v4 CSV and investigate.

The notebook writes:

- `submission.csv` (the regenerated predictions — **do not upload this as the competition file**)
- `gci_honors_reproduction.zip` (notebook + submission, for the Honors upload)

If the automatic download does not start, use the Colab file browser (left sidebar) to download
the zip.

---

## 3. What to upload where

| Artifact | Destination |
|---|---|
| `submission.csv` **from this notebook** | **Do not upload** — XGBoost/CatBoost thread-count drift makes it slightly different from the validated v4 file |
| `submissions/submission_v4_cv0.76244.csv` | Competition platform — **must be the last prediction file** |
| `gci_honors_reproduction.zip` | Honors code submission |

If anything was uploaded to the competition platform **after** v4, re-upload
`submission_v4_cv0.76244.csv` at the very end. Fallback if v4 cannot be re-uploaded:
`submission_v3_recipe_cv0.76224.csv`.

---

## 4. Why not the notebook's `submission.csv`?

The shipped v4 file was built with XGBoost `n_jobs=8` and CatBoost `thread_count=8` on the local
machine. The notebook uses 2 threads for Colab. Both are thread-count-sensitive: on fold 0,
XGBoost gave **0.759578 vs 0.759029** (2,250 vs 2,134 trees, prediction correlation 0.996) and
CatBoost gave **0.758955 vs 0.759364** (correlation 0.9936). LightGBM was bit-identical. So the
notebook reproduces the **recipe and CV within tolerance**, which is what the Honors code
requirement needs; the validated v4 CSV remains the authoritative prediction file.

---

## 5. Troubleshooting

| Symptom | Fix |
|---|---|
| `AssertionError: put train.csv/test.csv/...` | Check `MyDrive/gci_competition/input/` (or `MyDrive/competition/input/`); re-run cell 2. |
| Drive mount permission loop | Runtime → Restart runtime, re-run cell 2, allow the popup. |
| `label_encoders` / parquet not found | The notebook does not need them; it rebuilds features from raw CSVs. |
| LightGBM/XGBoost API warning | Warnings are suppressed; only an `AssertionError` is fatal. |
| Session dies mid-run | Reconnect, re-run from the top. The notebook does not checkpoint members; budget a full rerun. |
| Runtime disconnects at ~90 min | Keep the tab active; do not close the browser. Run the notebook in one sitting. |
| Need to save time | Do not skip members — the blend weights are pinned and the asserts need all four. |
| Final OOF AUC `DRIFT` | Use the shipped v4 CSV; the code artifact is still valid if the notebook ran end-to-end. |

---

## 6. Local sanity checks (optional, ~1 min)

From the repo root, with the competition CSVs in place:

```bash
pip install -r requirements.txt
# verify the feature engineering notebook / matrix
jupyter nbconvert --to notebook --execute 01_feature_engineering.ipynb --output /tmp/fe_check.ipynb
```

The repaired `01_feature_engineering.ipynb` reproduces the reference 98-column matrix exactly
(0 mismatching cells). The authoritative end-to-end path remains the Honors notebook.

---

## 7. Final checklist

- [ ] Honors notebook ran end-to-end on Colab CPU with all four members `[OK]`.
- [ ] `gci_honors_reproduction.zip` downloaded and contains the notebook (and `submission.csv`).
- [ ] Honors zip uploaded to the Honors channel.
- [ ] Competition platform's **last** prediction file is `submission_v4_cv0.76244.csv`.
- [ ] Deadline confirmed: 2026-11-20 01:00 UTC (04:00 EAT).
