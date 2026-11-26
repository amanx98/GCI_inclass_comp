"""W5 final: given the winning blend (w5_blend_result.json) and the member test
arrays, build submissions/submission_v5_cvXXXX.csv with full asserts.

Member test arrays needed in out/:
  v4: m3_LGB3 -> need v4 LGB test (rebuild if missing), m3_XGB3 -> v4 XGB test,
      v4_CATnew_s42/1337_test, v4_HGBnew_test  (in pipeline/artifacts)
  w5_*: w5_lgb7_new_test, w5_xgb7_new_test, w5_cat4_test, w5_hgb3_test
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
import common as C

ART = C.ART
OUT = C.OUT
SUB = Path(__file__).resolve().parents[2] / "submissions"

res = json.load(open(OUT / "w5_blend_result.json"))
names, w = res["names"], np.array(res["weights"])
print("blend:", res["oof"], "gate:", res["gate"])
assert res["gate"]["passes"], "GATE FAILED - do not submit"

# map member name -> test array
TEST_MAP = {
    "v4_LGB3": None,  # rebuilt below if needed
    "v4_XGB3": None,
    "v4_CAT2": (np.load(ART / "v4_CATnew_s42_test.npy") + np.load(ART / "v4_CATnew_s1337_test.npy")) / 2,
    "v4_HGB": np.load(ART / "v4_HGBnew_test.npy"),
    "w5_LGB7_new": np.load(OUT / "w5_lgb7_new_test.npy"),
    "w5_XGB7_new": np.load(OUT / "w5_xgb7_new_test.npy"),
    "w5_CAT4": np.load(OUT / "w5_cat4_test.npy"),
    "w5_HGB3": np.load(OUT / "w5_hgb3_test.npy"),
}
# v4 LGB/XGB test preds: rebuild quickly if the blend uses them
need_lgb = any(n == "v4_LGB3" and w[i] > 1e-6 for i, n in enumerate(names))
need_xgb = any(n == "v4_XGB3" and w[i] > 1e-6 for i, n in enumerate(names))
if need_lgb or need_xgb:
    print("rebuilding v4 LGB/XGB test preds (3 seeds each)...", flush=True)
    folds = C.fold_frames()
    if need_lgb:
        _, t = C.lgb_oof_test(folds, seeds=(42, 1337, 2026))
        TEST_MAP["v4_LGB3"] = t
        np.save(OUT / "v4_lgb3_test.npy", t)
    if need_xgb:
        _, t = C.xgb_oof_test(folds, seeds=(42, 1337, 2026))
        TEST_MAP["v4_XGB3"] = t
        np.save(OUT / "v4_xgb3_test.npy", t)

# remap blend names to test arrays
def test_of(name):
    if name in TEST_MAP and TEST_MAP[name] is not None:
        return TEST_MAP[name]
    if name.startswith("w3_"):
        # w3_<fam>_s<seeds>_oof -> test file with same tag
        tag = name[3:-4]  # strip w3_ and _oof
        return np.load(OUT / f"w3_{tag}_test.npy")
    if name == "w1_LGB":
        return None  # no test for W1 (rejected anyway)
    raise KeyError(name)

test_pred = np.zeros(len(C.X_TEST))
for i, n in enumerate(names):
    if w[i] < 1e-9:
        continue
    t = test_of(n)
    assert t is not None and len(t) == len(C.X_TEST), f"missing test for {n}"
    test_pred += w[i] * t

# ---- asserts ----
sample_path = Path("competition_unzipped/04. Competition (due Nov 20)/Competition/input/sample_submission.csv")
sample = pd.read_csv(sample_path)
assert len(test_pred) == 61500
assert np.isfinite(test_pred).all()
assert test_pred.min() >= 0 and test_pred.max() <= 1

# SK_ID_CURR from test csv
test_csv = pd.read_csv("competition_unzipped/04. Competition (due Nov 20)/Competition/input/test.csv",
                       usecols=["SK_ID_CURR"])
sub = pd.DataFrame({"SK_ID_CURR": test_csv["SK_ID_CURR"].values, "TARGET": test_pred})
sub = sub.set_index("SK_ID_CURR").loc[sample["SK_ID_CURR"]].reset_index()
assert len(sub) == len(sample) == 61500
assert (sub["SK_ID_CURR"].values == sample["SK_ID_CURR"].values).all()
assert sub["TARGET"].between(0, 1).all() and not sub["TARGET"].isnull().any()

cv = res["oof"]
path = SUB / f"submission_v5_cv{cv:.5f}.csv"
sub.to_csv(path, index=False)
print(f"SAVED {path}")
print(f"stats: mean={test_pred.mean():.5f} std={test_pred.std():.5f} "
      f"min={test_pred.min():.5f} max={test_pred.max():.5f}")
print("ALL ASSERTS PASSED")
