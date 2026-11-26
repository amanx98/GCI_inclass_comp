"""W3: seed expansion. Produces OOF (+test) for extra seeds per family using the
shipped (or W2-updated, if better) params, ready for the W5 blend re-search.

Usage: python w3_seeds.py --family lgb --seeds 7,99,123,2027
Defaults expand beyond the v4 seed sets:
  LGB/XGB v4 used 42,1337,2026 -> add 7,99,123,2027 (total 7)
  CAT v4 used 42,1337          -> add 2026,7     (total 4)
  HGB v4 used 42               -> add 1337,2026  (total 3)
"""
import argparse
import json
import time

import numpy as np

import common as C

ap = argparse.ArgumentParser()
ap.add_argument("--family", required=True, choices=["lgb", "xgb", "cat", "hgb"])
ap.add_argument("--seeds", default="")
ap.add_argument("--params", default="", help="path to params JSON (default: shipped e2 params)")
args = ap.parse_args()

seeds = tuple(int(s) for s in args.seeds.split(",") if s)
assert seeds, "no seeds given"
params = json.load(open(args.params)) if args.params else None
if params is None and args.family == "hgb":
    params = C.HGB_PARAMS

t0 = time.time()
folds = C.fold_frames()  # with test
print(f"W3 {args.family} seeds={seeds} ({time.time()-t0:.0f}s)", flush=True)

runner = {"lgb": C.lgb_oof_test, "xgb": C.xgb_oof_test,
          "cat": C.cat_oof_test, "hgb": C.hgb_oof_test}[args.family]
oof, test = runner(folds, seeds=seeds, params=params)
a = C.auc(oof)
tag = "_".join(str(s) for s in seeds)
np.save(C.OUT / f"w3_{args.family}_s{tag}_oof.npy", oof)
np.save(C.OUT / f"w3_{args.family}_s{tag}_test.npy", test)
print(f"W3 {args.family} seeds={seeds} OOF AUC: {a:.6f} ({time.time()-t0:.0f}s) SAVED", flush=True)
