import sys, json, os
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from scipy.stats import spearmanr

T = r"C:\Users\amanuel\AppData\Local\Temp\opencode"
WS = r"C:\Users\amanuel\OneDrive\Documents\deepseek-harness\default-workspace"
F = rf"{T}\drive\GCI Competition\features"

y = np.load(rf"{F}\y_train.npy")
print("y:", y.shape, y.mean())

def ld(n):
    p = rf"{T}\{n}"
    return np.load(p) if os.path.exists(p) else None

oofs = {
    "m2_LGB3": ld("m2_LGB3.npy"), "m3_LGB3": ld("m3_LGB3.npy"),
    "m2_XGB3": ld("m2_XGB3.npy"), "m3_XGB3": ld("m3_XGB3.npy"),
    "m2_CAT2": ld("m2_CAT2.npy"),
    "cat42": ld("v4_CATnew_s42_oof.npy"), "cat1337": ld("v4_CATnew_s1337_oof.npy"),
    "hgb": ld("v4_HGBnew_oof.npy"),
    "m2_HGB": ld("m2_HGB.npy"),
    "mlp": ld("nn_diversity/mlp_oof.npy"), "tabnet": ld("nn_diversity/tabnet_oof.npy"),
    "lgb_s42": ld("nn_diversity/lgb_s42_oof.npy"),
}
print("\n--- OOF AUCs ---")
for k, v in oofs.items():
    if v is not None:
        print(f"{k:12s} n={len(v)} auc={roc_auc_score(y, v):.5f} min={v.min():.4f} max={v.max():.4f} mean={v.mean():.4f}")

# seed-averaged CAT new
catn_oof = (oofs["cat42"] + oofs["cat1337"]) / 2
print(f"\nCATn 2-seed oof avg auc = {roc_auc_score(y, catn_oof):.5f}")
print(f"CATn s42 solo auc = {roc_auc_score(y, oofs['cat42']):.5f}; s1337 = {roc_auc_score(y, oofs['cat1337']):.5f}")

# v4 blend
W4 = {"LGB": 0.30, "XGBn": 0.33, "CATn": 0.33, "HGB": 0.04}
v4_oof = W4["LGB"]*oofs["m3_LGB3"] + W4["XGBn"]*oofs["m3_XGB3"] + W4["CATn"]*catn_oof + W4["HGB"]*oofs["hgb"]
print(f"\nv4 OOF blend = {roc_auc_score(y, v4_oof):.5f}  (claimed 0.76244)")

# v3
W3 = {"LGBnew": .355, "XGBold": .04, "XGBnew": .355, "CAT": .25}
v3_oof = W3["LGBnew"]*oofs["m3_LGB3"] + W3["XGBold"]*oofs["m2_XGB3"] + W3["XGBnew"]*oofs["m3_XGB3"] + W3["CAT"]*oofs["m2_CAT2"]
print(f"v3 OOF blend = {roc_auc_score(y, v3_oof):.5f}  (claimed 0.76224)")

# difference bootstrap (paired by row)
print("\n--- paired bootstrap v4 - v3 (500 reps) ---")
rng = np.random.default_rng(42)
d = []
n = len(y)
for _ in range(500):
    idx = rng.integers(0, n, n)
    yy = y[idx]
    if yy.min() == yy.max():
        continue
    d.append(roc_auc_score(yy, v4_oof[idx]) - roc_auc_score(yy, v3_oof[idx]))
d = np.array(d)
print(f"mean diff {d.mean():+.6f}  sd {d.std():.6f}  95% CI [{np.percentile(d,2.5):+.6f}, {np.percentile(d,97.5):+.6f}]  P(>0)={np.mean(d>0):.3f}")

# old vs new members
for a, b in [("m3_LGB3","m2_LGB3"), ("m3_XGB3","m2_XGB3"), ("m2_CAT2","cat42")]:
    if oofs[a] is not None and oofs[b] is not None:
        print(f"spearman {a} vs {b}: {spearmanr(oofs[a], oofs[b]).statistic:.4f}")

# test preds reconstruction
print("\n--- test predictions ---")
tests = {k: ld(v) for k, v in {
    "lgb": "v3_test_LGBnew.npy", "xgb": "v3_test_XGBnew.npy",
    "cat42": "v4_CATnew_s42_test.npy", "cat1337": "v4_CATnew_s1337_test.npy",
    "hgb": "v4_HGBnew_test.npy"}.items()}
for k, v in tests.items():
    print(f"{k:8s} n={len(v)} mean={v.mean():.5f} std={v.std():.5f} min={v.min():.5f} max={v.max():.5f}")
catn_test = (tests["cat42"] + tests["cat1337"]) / 2
v4_test = W4["LGB"]*tests["lgb"] + W4["XGBn"]*tests["xgb"] + W4["CATn"]*catn_test + W4["HGB"]*tests["hgb"]

sub = pd.read_csv(rf"{WS}\submissions\submission_v4_cv0.76244.csv")
sample = pd.read_csv(rf"{WS}\competition_unzipped\04. Competition (due Nov 20)\Competition\input\sample_submission.csv")
print(f"\nsub rows={len(sub)} cols={list(sub.columns)} ids_match_sample={sub.SK_ID_CURR.equals(sample.SK_ID_CURR)} nan={sub.TARGET.isna().sum()}")
print(f"sub stats mean={sub.TARGET.mean():.5f} std={sub.TARGET.std():.5f} min={sub.TARGET.min():.5f} max={sub.TARGET.max():.5f}")
merged = sub.merge(sample, on="SK_ID_CURR", suffixes=("_sub", "_sample"))
print(f"id set equal: {set(sub.SK_ID_CURR)==set(sample.SK_ID_CURR)}")

if len(v4_test) == len(sub):
    # align test pred order to sub order via test.csv ids
    te = pd.read_parquet(rf"{F}\test_engineered.parquet", columns=["SK_ID_CURR"])
    m = pd.DataFrame({"SK_ID_CURR": te.SK_ID_CURR.values, "p": v4_test}).merge(sub, on="SK_ID_CURR")
    print(f"reconstructed vs submitted: n={len(m)} max|diff|={np.abs(m.p-m.TARGET).max():.3e} corr={np.corrcoef(m.p,m.TARGET)[0,1]:.8f} spearman={spearmanr(m.p,m.TARGET).statistic:.8f}")

# compare each submission file with reconstructed members
print("\n--- submission cross-check ---")
subs = {p.name: pd.read_csv(p) for p in sorted((pd.io.common.Path(rf"{WS}\submissions")).glob("*.csv"))} if False else {}
import glob
for p in sorted(glob.glob(rf"{WS}\submissions\*.csv")):
    s = pd.read_csv(p)
    print(os.path.basename(p), len(s), f"mean={s.TARGET.mean():.5f} std={s.TARGET.std():.5f}", f"min={s.TARGET.min():.5f} max={s.TARGET.max():.5f}")
    if len(s) == len(v4_test):
        pass

# weight sensitivity around v4
print("\n--- v4 weight sensitivity (hold others fixed) ---")
for name in ["LGB","XGBn","CATn","HGB"]:
    base = W4[name]
    row = []
    for delta in [-0.06,-0.03,0.0,0.03,0.06]:
        w = dict(W4); w[name] = base + delta
        # renormalize others proportionally
        s = sum(w.values())
        w = {k: v/s for k,v in w.items()}
        b = w["LGB"]*oofs["m3_LGB3"] + w["XGBn"]*oofs["m3_XGB3"] + w["CATn"]*catn_oof + w["HGB"]*oofs["hgb"]
        row.append(f"{delta:+.2f}:{roc_auc_score(y,b):.5f}")
    print(f"{name:5s} base {base:.2f} | " + "  ".join(row))

# per-fold AUC of v4 vs v3
print("\n--- per-fold AUC v3 vs v4 ---")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for i, (_, vai) in enumerate(skf.split(v4_oof, y)):
    print(f"fold {i}: v3={roc_auc_score(y[vai], v3_oof[vai]):.5f} v4={roc_auc_score(y[vai], v4_oof[vai]):.5f} diff={roc_auc_score(y[vai],v4_oof[vai])-roc_auc_score(y[vai],v3_oof[vai]):+.5f}")

# any single OOF correlation table
print("\n--- spearman matrix (key members) ---")
keys = ["m3_LGB3","m3_XGB3","m2_CAT2","hgb"]
print("            " + "  ".join(f"{k:>8s}" for k in keys))
for a in keys:
    print(f"{a:10s}  " + "  ".join(f"{spearmanr(oofs[a], oofs[b]).statistic:8.4f}" for b in keys))
