import sys, os, json
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
lgb = np.load(rf"{T}\m3_LGB3.npy"); xgb = np.load(rf"{T}\m3_XGB3.npy")
cat = (np.load(rf"{T}\v4_CATnew_s42_oof.npy") + np.load(rf"{T}\v4_CATnew_s1337_oof.npy"))/2
hgb = np.load(rf"{T}\v4_HGBnew_oof.npy")
W = np.array([.30,.33,.33,.04]); P = np.column_stack([lgb,xgb,cat,hgb])

# --- HGB contribution ---
W0 = np.array([.30,.33,.33,0]); W0 = W0/W0.sum()
base = roc_auc_score(y, P@W); no_hgb = roc_auc_score(y, P@W0)
print(f"v4 with HGB .04: {base:.6f}   without HGB: {no_hgb:.6f}   delta {base-no_hgb:+.6f}")
rng = np.random.default_rng(11); d=[]
for _ in range(500):
    idx = rng.integers(0,len(y),len(y)); yy=y[idx]
    if yy.min()==yy.max(): continue
    d.append(roc_auc_score(yy,(P@W)[idx])-roc_auc_score(yy,(P@W0)[idx]))
d=np.array(d); print(f"HGB paired bootstrap: mean {d.mean():+.6f} CI [{np.percentile(d,2.5):+.6f},{np.percentile(d,97.5):+.6f}] P(>0)={np.mean(d>0):.3f}")

# --- CAT adoption per fold ---
old = np.load(rf"{T}\m2_CAT.npy"); new = np.load(rf"{T}\v4_CATnew_s42_oof.npy")
skf = StratifiedKFold(5, shuffle=True, random_state=42)
print("\nCAT old vs new params per fold (s42):")
for i,(_,vai) in enumerate(skf.split(old,y)):
    a=roc_auc_score(y[vai],old[vai]); b=roc_auc_score(y[vai],new[vai])
    print(f"  fold {i}: old {a:.5f}  new {b:.5f}  diff {b-a:+.5f}")

# --- submission correlations ---
v3 = pd.read_csv(rf"{WS}\submissions\submission_v3_recipe_cv0.76224.csv")
v4 = pd.read_csv(rf"{WS}\submissions\submission_v4_cv0.76244.csv")
v2 = pd.read_csv(rf"{WS}\submissions\submission_v2_recipe_cv0.76203.csv")
print("\nsubmission spearman:")
for name,s in [("v2",v2),("v3",v3)]:
    print(f"  v4 vs {name}: {spearmanr(v4.TARGET, s.TARGET).statistic:.6f}  max|diff| {np.abs(v4.TARGET-s.TARGET).max():.6f}")

# --- enrich fill means train vs test ---
tr = pd.read_parquet(rf"{F}\train_engineered.parquet")
te = pd.read_parquet(rf"{F}\test_engineered.parquet")
print("\nenrich NaN-fill means (train vs test):")
for c in ["EXT_SOURCE_1","EXT_SOURCE_2","EXT_SOURCE_3","DAYS_BIRTH","DAYS_EMPLOYED"]:
    a=tr[c].replace(-999,np.nan); b=te[c].replace(-999,np.nan)
    if c=="DAYS_BIRTH" or c=="DAYS_EMPLOYED":
        a=a.abs()/365; b=b.abs()/365
    print(f"  {c:15s} train {a.mean():.6f}  test {b.mean():.6f}  rel {abs(a.mean()-b.mean())/abs(a.mean())*100:.4f}%")

# --- member possible leak: TE target means vs y? not applicable
print("\nHGB rankcorr vs XGBnew:", f"{spearmanr(hgb,xgb).statistic:.4f}")
print("CAT rankcorr vs XGBnew:", f"{spearmanr(cat,xgb).statistic:.4f}")
print("LGB rankcorr vs XGBnew:", f"{spearmanr(lgb,xgb).statistic:.4f}")
