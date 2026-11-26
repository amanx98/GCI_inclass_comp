"""Diagnose the interrupted W6 xgb10 run: is the saved OOF complete or partial?

w6_cv_variants.py saves the OOF incrementally after EVERY model seed.
xgb10 = 10 folds x 3 model seeds. If it died mid-seed, the saved OOF has
zeros for the rows of the unfinished folds -- BUT the script also accumulates
`test` continuously, so we need to check per-fold coverage.
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import common as C  # noqa: E402

OUT = HERE

X10 = np.load(OUT / "w6_xgb10_oof.npy").astype(np.float64)
print(f"xgb10 oof shape {X10.shape}")
print(f"exact zeros: {(X10 == 0.0).sum()}")
print(f"nan: {np.isnan(X10).sum()}")
print(f"overall AUC: {roc_auc_score(C.y, X10):.6f}")

# 10-fold split, same params as the script (fold seed 42)
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
folds = list(skf.split(C.X, C.y))
print(f"\nper-fold coverage (each fold should be fully non-zero if complete):")
n_incomplete = 0
for fi, (_, vai) in enumerate(folds):
    z = int((X10[vai] == 0.0).sum())
    a = roc_auc_score(C.y[vai], X10[vai]) if z == 0 else float("nan")
    flag = "OK " if z == 0 else "PARTIAL"
    if z:
        n_incomplete += 1
    print(f"  fold {fi:2d}: zeros={z:6d} auc={a:.6f} [{flag}]")
print(f"\nincomplete folds: {n_incomplete}/10")
print(f"=> xgb10 is {'USABLE' if n_incomplete == 0 else 'PARTIAL / must not be used'}")

# also check the other w6 members for the same problem
print("\n=== all w6 members: fold coverage on their own n_splits ===")
meta = {
    "lgb10": 10,
    "xgb10": 10,
    "xgb_f1337": 5,
    "lgb_f1337": 5,
}
for tag, nsp in meta.items():
    p = OUT / f"w6_{tag}_oof.npy"
    if not p.exists():
        continue
    o = np.load(p).astype(np.float64)
    fs = 1337 if "f1337" in tag else 42
    skf2 = StratifiedKFold(n_splits=nsp, shuffle=True, random_state=fs)
    bad = 0
    for _, vai in skf2.split(C.X, C.y):
        if int((o[vai] == 0.0).sum()):
            bad += 1
    print(f"  {tag:10s}: {bad}/{nsp} incomplete folds | auc={roc_auc_score(C.y, o):.6f}")
