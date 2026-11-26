"""W5 phase 3: fine weight search around the v4 recipe with the new members swapped in,
restricted to members that individually beat/near v4 counterparts, then bootstrap-gated.

Members considered:
  LGB: v4_LGB3 (0.761548), w5_LGB7_new (0.760460)
  XGB: v4_XGB3 (0.761707), w5_XGB7_new (0.761714), w3_xgb_extra (0.761653)
  CAT: v4_CAT2 (0.760892), w5_CAT4 (0.760995)
  HGB: v4_HGB (0.758139), w5_HGB3 (0.759125)

Search: full simplex grid step 0.02 over [LGB, XGB, CAT, HGB] where each family slot may be
v4 or new member -> evaluate both variants per family combo; keep top by OOF; gate top 3.
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
import common as C

OUT = C.OUT
ART = C.ART

fam = {
    "LGB": [np.load(ART / "m3_LGB3.npy"), np.load(OUT / "w5_lgb7_new_oof.npy")],
    "XGB": [np.load(ART / "m3_XGB3.npy"), np.load(OUT / "w5_xgb7_new_oof.npy")],
    "CAT": [(np.load(ART / "v4_CATnew_s42_oof.npy") + np.load(ART / "v4_CATnew_s1337_oof.npy")) / 2,
            np.load(OUT / "w5_cat4_oof.npy")],
    "HGB": [np.load(ART / "v4_HGBnew_oof.npy"), np.load(OUT / "w5_hgb3_oof.npy")],
}
fam_names = {k: ["v4", "new"] for k in fam}
keys = list(fam)

# Precompute rank-transform of each member once: AUC of a nonneg linear blend is invariant
# under strictly increasing transforms of the members, so rank-space AUC == probability-space
# AUC for ranking metrics. This cuts per-eval cost to O(n) with tiny arrays already in memory.
def rankify(v):
    from scipy.stats import rankdata
    return rankdata(v) / len(v)

best = []
step = 0.05
w = np.arange(0, 1.0001, step)
ranked = {k: [rankify(v) for v in fam[k]] for k in keys}
# keep only top-200 per combo to bound memory, then merge
for combo in itertools.product([0, 1], repeat=4):  # which member per family
    mats = [ranked[k][c] for k, c in zip(keys, combo)]
    local = []
    for w0 in w:
        for w1 in np.arange(0, 1.0001 - w0, step):
            for w2 in np.arange(0, 1.0001 - w0 - w1, step):
                w3 = 1 - w0 - w1 - w2
                if w3 < -1e-9:
                    continue
                a = roc_auc_score(C.y, w0 * mats[0] + w1 * mats[1] + w2 * mats[2] + w3 * mats[3])
                local.append((a, combo, (round(float(w0), 3), round(float(w1), 3),
                                         round(float(w2), 3), round(float(w3), 3))))
    local.sort(key=lambda t: -t[0])
    best.extend(local[:200])
    del local, mats

best.sort(key=lambda t: -t[0])
print(f"searched {len(best)} combos (top-200 per member combo); top 5 by OOF:")
for a, combo, ws in best[:5]:
    print(f"  {a:.6f} combo={[fam_names[k][c] for k, c in zip(keys, combo)]} w(LGB,XGB,CAT,HGB)={ws}", flush=True)

# gate top distinct weight patterns (rebuild oof on demand, free memory after)
seen = set()
gated = []
for a, combo, ws in best:
    sig = (combo, ws)
    if sig in seen:
        continue
    seen.add(sig)
    mats = [fam[k][c] for k, c in zip(keys, combo)]
    oof = ws[0] * mats[0] + ws[1] * mats[1] + ws[2] * mats[2] + ws[3] * mats[3]
    g = C.bootstrap_gate(oof)
    del oof, mats
    gated.append({"oof": a, "combo": [fam_names[k][c] for k, c in zip(keys, combo)],
                  "weights": list(ws), "gate": g})
    print(f"GATE oof={a:.6f} combo={[fam_names[k][c] for k, c in zip(keys, combo)]} ws={ws} "
          f"-> mean={g['mean']:+.6f} CI=[{g['ci_lo']:+.6f},{g['ci_hi']:+.6f}] p={g['p_gt0']:.3f} "
          f"passes={g['passes']}", flush=True)
    np.save(OUT / f"w5g_{len(gated)}_oof.npy", oof)
    if len(gated) >= 5:
        break

json.dump(gated, open(OUT / "w5_grid_gated.json", "w"), indent=2, default=float)
passing = [g for g in gated if g["gate"]["passes"]]
print("PASSING:", len(passing), "->", [(g["oof"], g["combo"], g["weights"]) for g in passing] if passing else "NONE")
