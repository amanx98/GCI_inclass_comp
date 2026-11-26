"""W5: blend re-search over all available OOF members (v4 + W1/W2/W3 outputs).

- loads every *.npy OOF in out/ plus the v4 artifacts,
- hill-climb + fine simplex grid + NNLS over members,
- reports best blend + bootstrap gate vs v4,
- if gate passes and --test-dir given, rebuilds test preds for the winning members
  and writes submissions/submission_v5_cvXXXX.csv with full asserts.
"""
import argparse
import glob
import json
import time

import numpy as np
from sklearn.metrics import roc_auc_score

import common as C

ap = argparse.ArgumentParser()
ap.add_argument("--save", action="store_true", help="write submission_v5 if gate passes")
ap.add_argument("--tag", default="v5")
args = ap.parse_args()

t0 = time.time()
members = {
    "v4_LGB3": np.load(C.ART / "m3_LGB3.npy"),
    "v4_XGB3": np.load(C.ART / "m3_XGB3.npy"),
    "v4_CAT2": (np.load(C.ART / "v4_CATnew_s42_oof.npy") + np.load(C.ART / "v4_CATnew_s1337_oof.npy")) / 2,
    "v4_HGB": np.load(C.ART / "v4_HGBnew_oof.npy"),
}
for p in sorted(glob.glob(str(C.OUT / "w3_*_oof.npy"))):
    members["w3_" + p.split("\\")[-1][:-4]] = np.load(p)
for p in sorted(glob.glob(str(C.OUT / "w4_*_oof.npy"))):
    members["w4_" + p.split("\\")[-1][:-4]] = np.load(p)
for p in sorted(glob.glob(str(C.OUT / "w5_*_oof.npy"))):
    members["w5_" + p.split("\\")[-1][:-4]] = np.load(p)
if (C.OUT / "w1_lgb_oof.npy").exists():
    members["w1_LGB"] = np.load(C.OUT / "w1_lgb_oof.npy")

print(f"{len(members)} members:")
for k, v in members.items():
    print(f"  {k}: {roc_auc_score(C.y, v):.6f}", flush=True)

names = list(members.keys())
M = np.column_stack([members[k] for k in names])

# hill-climb (average with replacement)
current, chosen = None, []
for step in range(12):
    best = (None, -1.0)
    for i, k in enumerate(names):
        cand = M[:, i].copy() if current is None else (current + M[:, i]) / 2
        a = roc_auc_score(C.y, cand)
        if a > best[1]:
            best = (i, a)
    if current is not None and best[1] <= roc_auc_score(C.y, current):
        break
    current = M[:, best[0]].copy() if current is None else (current + M[:, best[0]]) / 2
    chosen.append(names[best[0]])
    print(f"hill step {step+1}: +{names[best[0]]} -> {best[1]:.6f}", flush=True)

# fine grid over top 6 members by single AUC
order = sorted(names, key=lambda k: -roc_auc_score(C.y, members[k]))[:6]
print("grid over:", order, flush=True)
idx = [names.index(k) for k in order]
best = (-1, None)
step = 0.05
w = np.arange(0, 1.0001, step)
for w0 in w:
    for w1 in np.arange(0, 1.0001 - w0, step):
        for w2 in np.arange(0, 1.0001 - w0 - w1, step):
            for w3 in np.arange(0, 1.0001 - w0 - w1 - w2, step):
                rem = 1 - w0 - w1 - w2 - w3
                if rem < -1e-9:
                    continue
                ws = np.zeros(len(names))
                ws[idx[0]], ws[idx[1]], ws[idx[2]], ws[idx[3]] = w0, w1, w2, w3
                if len(idx) > 4:
                    ws[idx[4]] = rem
                else:
                    ws[idx[0]] += rem
                a = roc_auc_score(C.y, M @ ws)
                if a > best[0]:
                    best = (a, ws.copy())
print(f"grid best: {best[0]:.6f} weights={dict(zip(names, np.round(best[1], 3)))}", flush=True)

# NNLS
from scipy.optimize import nnls
w_nnls, _ = nnls(M, C.y)
w_nnls /= w_nnls.sum()
a_nnls = roc_auc_score(C.y, M @ w_nnls)
print(f"NNLS: {a_nnls:.6f} weights={dict(zip(names, np.round(w_nnls, 3)))}", flush=True)

cands = {"hill": (roc_auc_score(C.y, current), None), "grid": best, "nnls": (a_nnls, w_nnls)}
# pick best
if best[0] >= a_nnls and best[0] >= cands["hill"][0]:
    final_w, final_a = best[1], best[0]
elif a_nnls >= cands["hill"][0]:
    final_w, final_a = w_nnls, a_nnls
else:
    final_w = None
    final_a = cands["hill"][0]
    # reconstruct hill weights
    final_w = np.zeros(len(names))
    for k in chosen:
        final_w[names.index(k)] = 1.0 / len(chosen)

oof = M @ final_w if final_w is not None else current
print(f"\nFINAL blend OOF: {roc_auc_score(C.y, oof):.6f} (v4 = {roc_auc_score(C.y, C.V4_OOF):.6f})", flush=True)
gate = C.bootstrap_gate(oof)
print("bootstrap gate vs v4:", json.dumps(gate), flush=True)
json.dump({"names": names, "weights": list(map(float, final_w)),
           "oof": float(roc_auc_score(C.y, oof)), "gate": gate},
          open(C.OUT / "w5_blend_result.json", "w"), indent=2)
np.save(C.OUT / "w5_blend_oof.npy", oof)
print(f"W5 DONE ({time.time()-t0:.0f}s) gate_passes={gate['passes']}", flush=True)
