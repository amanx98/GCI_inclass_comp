import sys, time, json
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

T = r"C:\Users\amanuel\AppData\Local\Temp\opencode"
F = rf"{T}\drive\GCI Competition\features"
X = pd.read_parquet(rf"{T}\X_e2.parquet")
X_test = pd.read_parquet(rf"{T}\X_test_e2.parquet")
y = np.load(rf"{F}\y_train.npy")
with open(rf"{F}\high_card_cols.json") as fh:
    te_cols = [c for c in json.load(fh) if c in X.columns]

def add_te(X_tr, X_va, X_te_, y_tr, smoothing=10):
    X_tr, X_va, X_te_ = X_tr.copy(), X_va.copy(), X_te_.copy()
    gs = pd.Series(y_tr, index=X_tr.index); gm = gs.mean()
    for col in te_cols:
        means = gs.groupby(X_tr[col]).mean()
        counts = X_tr[col].value_counts()
        sm = (means * counts + gm * smoothing) / (counts + smoothing)
        X_tr[f"{col}_TE"] = X_tr[col].map(sm).fillna(gm)
        X_va[f"{col}_TE"] = X_va[col].map(sm).fillna(gm)
        X_te_[f"{col}_TE"] = X_te_[col].map(sm).fillna(gm)
    return X_tr, X_va, X_te_

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_data = []
for tri, vai in skf.split(X, y):
    fold_data.append((tri, vai, *add_te(X.iloc[tri], X.iloc[vai], X_test, y[tri])))
print("folds ready", flush=True)

p4 = json.load(open(rf"{T}\best_params_hgb_e2.json"))
params = {**p4, "max_iter": 1000, "early_stopping": True,
          "validation_fraction": 0.1, "n_iter_no_change": 25, "random_state": 42}
print("HGB full params:", json.dumps(params), flush=True)

t0 = time.time()
oof = np.zeros(len(X)); test = np.zeros(len(X_test))
for fold, (tri, vai, X_tr, X_va, X_te_) in enumerate(fold_data):
    m = HistGradientBoostingClassifier(**params)
    m.fit(X_tr, y[tri])
    oof[vai] = m.predict_proba(X_va)[:, 1]
    test += m.predict_proba(X_te_)[:, 1] / 5
    print(f"HGB f{fold+1}: {roc_auc_score(y[vai], oof[vai]):.5f} ({time.time()-t0:.0f}s)", flush=True)
auc = roc_auc_score(y, oof)
ref = np.load(rf"{T}\m3_XGB3.npy")
rc = spearmanr(oof, ref).statistic
np.save(rf"{T}\v4_HGBnew_oof.npy", oof)
np.save(rf"{T}\v4_HGBnew_test.npy", test)
print(f"HGB tuned (fixed) OOF AUC: {auc:.5f} | rankcorr vs XGBnew: {rc:.4f}", flush=True)
print("HGB FIX DONE", flush=True)
