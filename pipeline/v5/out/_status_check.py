"""Fast status check of v5 W6/W7 artifacts (no training)."""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import common as C  # noqa: E402

OUT = HERE
ART = C.ART


def load(p):
    return np.load(p).astype(np.float64)


print(f"v4 OOF AUC (harness rebuild): {C.auc(C.V4_OOF):.6f}")
print()

# --- W6 artifacts ---
print("=== W6 CV-protocol variants ===")
for tag in ("lgb10", "xgb10", "xgb_f1337", "lgb_f1337", "cat10"):
    op = OUT / f"w6_{tag}_oof.npy"
    jp = OUT / f"w6_{tag}.json"
    if op.exists():
        o = load(op)
        meta = json.load(open(jp)) if jp.exists() else None
        stated = f"{meta['oof_auc']:.6f}" if meta else "NO JSON (run incomplete)"
        print(f"  {tag:10s} oof_auc={C.auc(o):.6f}  json={stated}  "
              f"oof_nan={int(np.isnan(o).sum())}")
    else:
        print(f"  {tag:10s} MISSING")

# --- how many rows have zero predictions (unfilled folds) ---
print()
print("=== W6 coverage check (rows with exactly 0.0 prediction = unfilled) ===")
for tag in ("lgb10", "xgb10", "xgb_f1337", "lgb_f1337", "cat10"):
    op = OUT / f"w6_{tag}_oof.npy"
    if not op.exists():
        continue
    o = load(op)
    zeros = int((o == 0.0).sum())
    print(f"  {tag:10s} zeros={zeros}  min={o.min():.6f}  max={o.max():.6f}")
