"""W5 phase 2: candidate v5 blends, each gated vs v4 (500-rep bootstrap, CI>0).

Candidates:
  A: v4 recipe but XGB3 -> XGB7-new (w5_xgb7_new), CAT2 -> CAT4 (w5_cat4), HGB->HGB3, LGB3 kept
  B: A but LGB3 -> LGB7-new
  C: v4 weights kept, swap only XGB member for XGB7-new
  D: v4 weights, swap only CAT for CAT4
  E: v4 weights + HGB->HGB3
  F: v4 weights + all swaps (A==F? A uses v4 LGB3; F swaps all incl LGB) - keep distinct
  G: hill result from w5_blend (0.762454) - already gated, fails
  H: simple mean of v4_XGB3, w5_xgb7_new, v4_LGB3, w5_cat4
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
import common as C

OUT = C.OUT
ART = C.ART

v4 = {
    "LGB3": np.load(ART / "m3_LGB3.npy"),
    "XGB3": np.load(ART / "m3_XGB3.npy"),
    "CAT2": (np.load(ART / "v4_CATnew_s42_oof.npy") + np.load(ART / "v4_CATnew_s1337_oof.npy")) / 2,
    "HGB": np.load(ART / "v4_HGBnew_oof.npy"),
}
new = {
    "XGB7": np.load(OUT / "w5_xgb7_new_oof.npy"),
    "LGB7": np.load(OUT / "w5_lgb7_new_oof.npy"),
    "CAT4": np.load(OUT / "w5_cat4_oof.npy"),
    "HGB3": np.load(OUT / "w5_hgb3_oof.npy"),
}

cands = {
    "A_xgb7new_cat4_hgb3": 0.30 * v4["LGB3"] + 0.33 * new["XGB7"] + 0.33 * new["CAT4"] + 0.04 * new["HGB3"],
    "B_all_new": 0.30 * new["LGB7"] + 0.33 * new["XGB7"] + 0.33 * new["CAT4"] + 0.04 * new["HGB3"],
    "C_xgb_swap_only": 0.30 * v4["LGB3"] + 0.33 * new["XGB7"] + 0.33 * v4["CAT2"] + 0.04 * v4["HGB"],
    "D_cat_swap_only": 0.30 * v4["LGB3"] + 0.33 * v4["XGB3"] + 0.33 * new["CAT4"] + 0.04 * v4["HGB"],
    "E_hgb_swap_only": 0.30 * v4["LGB3"] + 0.33 * v4["XGB3"] + 0.33 * v4["CAT2"] + 0.04 * new["HGB3"],
    "H_mean4": (v4["XGB3"] + new["XGB7"] + v4["LGB3"] + new["CAT4"]) / 4,
    "I_xgb3_xgb7_mean_only": 0.5 * v4["XGB3"] + 0.5 * new["XGB7"],
}

results = {}
for name, oof in cands.items():
    a = float(roc_auc_score(C.y, oof))
    g = C.bootstrap_gate(oof)
    results[name] = {"oof": a, "gate": g}
    print(f"{name}: OOF {a:.6f} gate mean={g['mean']:+.6f} CI=[{g['ci_lo']:+.6f},{g['ci_hi']:+.6f}] "
          f"p={g['p_gt0']:.3f} passes={g['passes']}", flush=True)
    np.save(OUT / f"w5cand_{name}_oof.npy", oof)

json.dump(results, open(OUT / "w5_candidates.json", "w"), indent=2, default=float)
best = max(results, key=lambda k: results[k]["oof"])
print(f"\nbest candidate by OOF: {best} {results[best]['oof']:.6f}")
passing = [k for k, v in results.items() if v["gate"]["passes"]]
print("gate-passing candidates:", passing if passing else "NONE - v4 stays last file")
