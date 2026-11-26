"""W7: final blend + gate over v4 members and W6 CV-protocol variants.

Memory-safe: loads one member at a time where possible, uses rank-space search
(AUC of a nonneg linear blend is invariant under per-member monotone transforms),
and never materialises more than a couple of 171k float arrays at once.

Gate (unchanged): 500-rep paired bootstrap of (candidate - v4) OOF AUC,
95% CI entirely > 0  AND  >= 4/5 folds positive (5-fold view of the seed-42 split).
"""
import gc
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
import common as C

OUT = C.OUT
ART = C.ART


def load(path):
    return np.load(path).astype(np.float64)


def rankify(v):
    return rankdata(v) / len(v)


# ---- members: name -> (oof_path, test_path or None) ----
members = {
    "v4_LGB3": (ART / "m3_LGB3.npy", None),
    "v4_XGB3": (ART / "m3_XGB3.npy", None),
    "v4_CAT2": (ART / "v4_CATnew_s42_oof.npy", ART / "v4_CATnew_s42_test.npy"),
    "v4_HGB": (ART / "v4_HGBnew_oof.npy", ART / "v4_HGBnew_test.npy"),
    "w5_CAT4": (OUT / "w5_cat4_oof.npy", OUT / "w5_cat4_test.npy"),
    "w5_XGB7": (OUT / "w5_xgb7_new_oof.npy", OUT / "w5_xgb7_new_test.npy"),
    "w5_HGB3": (OUT / "w5_hgb3_oof.npy", OUT / "w5_hgb3_test.npy"),
}
for tag in ("lgb10", "xgb10", "xgb_f1337", "lgb_f1337", "cat10"):
    op = OUT / f"w6_{tag}_oof.npy"
    tp = OUT / f"w6_{tag}_test.npy"
    if op.exists():
        members[f"w6_{tag}"] = (op, tp if tp.exists() else None)

# CAT2 special: average of two seeds -> built explicitly below
print(f"{len(members)} member slots")
oofs = {}
for name, (op, tp) in list(members.items()):
    if name == "v4_CAT2":
        continue
    if op is not None and op.exists():
        oofs[name] = load(op)
        print(f"  {name}: {roc_auc_score(C.y, oofs[name]):.6f}", flush=True)
c1 = load(ART / "v4_CATnew_s42_oof.npy")
c2 = load(ART / "v4_CATnew_s1337_oof.npy")
oofs["v4_CAT2"] = (c1 + c2) / 2
del c1, c2
gc.collect()
print(f"  v4_CAT2: {roc_auc_score(C.y, oofs['v4_CAT2']):.6f}", flush=True)

names = list(oofs)
R = {k: rankify(v) for k, v in oofs.items()}
del oofs
gc.collect()

# ---- candidate blends: family slots (LGB, XGB, CAT, HGB) x member choices ----
FAMS = {
    "LGB": [n for n in names if "LGB" in n or "lgb" in n],
    "XGB": [n for n in names if "XGB" in n or "xgb" in n],
    "CAT": [n for n in names if "CAT" in n or "cat" in n],
    "HGB": [n for n in names if "HGB" in n or "hgb" in n],
}
print("families:", {k: v for k, v in FAMS.items()}, flush=True)

# limit per family to the strongest few to keep the grid sane
for k, v in FAMS.items():
    FAMS[k] = sorted(v, key=lambda n: -roc_auc_score(C.y, R[n]))[:3]

results = []
step = 0.05
wvals = np.arange(0, 1.0001, step)
combos = list(itertools.product(*[FAMS[k] for k in FAMS]))
for combo in combos:
    mats = [R[n] for n in combo]
    for w0 in wvals:
        for w1 in np.arange(0, 1.0001 - w0, step):
            for w2 in np.arange(0, 1.0001 - w0 - w1, step):
                w3 = 1 - w0 - w1 - w2
                if w3 < -1e-9:
                    continue
                a = roc_auc_score(C.y, w0 * mats[0] + w1 * mats[1] + w2 * mats[2] + w3 * mats[3])
                results.append((a, combo, (round(float(w0), 3), round(float(w1), 3),
                                           round(float(w2), 3), round(float(w3), 3))))
results.sort(key=lambda t: -t[0])
print(f"searched {len(results)} blends; top 10 by OOF:")
for a, combo, ws in results[:10]:
    print(f"  {a:.6f} {combo} w={ws}", flush=True)

# ---- gate the top distinct candidates ----
OOF_PATH = {
    "v4_LGB3": ART / "m3_LGB3.npy",
    "v4_XGB3": ART / "m3_XGB3.npy",
    "v4_CAT2": None,          # average of two seeds, built below
    "v4_HGB": ART / "v4_HGBnew_oof.npy",
    "w5_CAT4": OUT / "w5_cat4_oof.npy",
    "w5_XGB7": OUT / "w5_xgb7_new_oof.npy",
    "w5_HGB3": OUT / "w5_hgb3_oof.npy",
}
for tag in ("lgb10", "xgb10", "xgb_f1337", "lgb_f1337", "cat10"):
    OOF_PATH[f"w6_{tag}"] = OUT / f"w6_{tag}_oof.npy"


def load_member(name):
    if name == "v4_CAT2":
        a = load(ART / "v4_CATnew_s42_oof.npy")
        b = load(ART / "v4_CATnew_s1337_oof.npy")
        r = (a + b) / 2
        del a, b
        return r
    return load(OOF_PATH[name])


gated = []
seen = set()
for a, combo, ws in results:
    sig = (combo, ws)
    if sig in seen:
        continue
    seen.add(sig)
    oof = np.zeros(len(C.y))
    for wj, nj in zip(ws, combo):
        oof += wj * load_member(nj)
        gc.collect()
    g = C.bootstrap_gate(oof)
    # 5-fold positivity on the seed-42 split
    fa_new = [roc_auc_score(C.y[vai], oof[vai]) for _, vai in C.FOLDS]
    fa_v4 = [roc_auc_score(C.y[vai], C.V4_OOF[vai]) for _, vai in C.FOLDS]
    npos = sum(1 for a_, b_ in zip(fa_new, fa_v4) if a_ > b_)
    gated.append({"oof": float(a), "combo": list(combo), "weights": list(ws),
                  "gate": g, "folds_positive": npos,
                  "passes": bool(g["passes"] and npos >= 4)})
    print(f"GATE oof={a:.6f} {combo} w={ws} mean={g['mean']:+.6f} "
          f"CI=[{g['ci_lo']:+.6f},{g['ci_hi']:+.6f}] p={g['p_gt0']:.3f} "
          f"folds+={npos}/5 passes={gated[-1]['passes']}", flush=True)
    np.save(OUT / f"w7_cand_{len(gated)}_oof.npy", oof)
    del oof
    gc.collect()
    if len(gated) >= 8:
        break

json.dump(gated, open(OUT / "w7_gated.json", "w"), indent=2, default=float)
passing = [g for g in gated if g["passes"]]
print("\nPASSING CANDIDATES:", len(passing))
for g in passing:
    print("  ", g["oof"], g["combo"], g["weights"], g["gate"])
if not passing:
    print("=> NONE. v4 remains the last uploaded file.")
