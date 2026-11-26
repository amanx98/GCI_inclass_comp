import sys, os, hashlib
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

WS = r"C:\Users\amanuel\OneDrive\Documents\deepseek-harness\default-workspace"
C = rf"{WS}\competition_unzipped\04. Competition (due Nov 20)\Competition\input"
tr = pd.read_csv(rf"{C}\train.csv")
te = pd.read_csv(rf"{C}\test.csv")
print("shapes", tr.shape, te.shape)
y = tr['TARGET'].values

print("\n--- ordering / ID leak checks ---")
print("row index AUC        :", roc_auc_score(y, np.arange(len(tr))))
print("SK_ID_CURR AUC       :", roc_auc_score(y, tr['SK_ID_CURR']))
print("train ID range", tr.SK_ID_CURR.min(), tr.SK_ID_CURR.max())
print("test  ID range", te.SK_ID_CURR.min(), te.SK_ID_CURR.max())
print("ID overlap           :", len(set(tr.SK_ID_CURR) & set(te.SK_ID_CURR)))
print("train target rate    :", y.mean())
print("train ids monotonic? :", tr.SK_ID_CURR.is_monotonic_increasing)
print("test  ids monotonic? :", te.SK_ID_CURR.is_monotonic_increasing)

# target rate by decile of row order
tr2 = tr[['SK_ID_CURR','TARGET']].copy()
tr2['decile'] = pd.qcut(tr2.index, 10, labels=False)
print("\ntarget rate by row decile:")
print(tr2.groupby('decile').TARGET.mean().round(4).to_string())

# duplicate feature rows within train and across train/test
feat = [c for c in tr.columns if c not in ('SK_ID_CURR','TARGET')]
print(f"\n--- duplicate-row check on {len(feat)} raw features ---")
def hashes(df):
    h = pd.util.hash_pandas_object(df[feat].fillna(-999), index=False)
    return h.values
htr = hashes(tr); hte = hashes(te)
print("train dup rows:", len(htr) - len(np.unique(htr)))
print("test  dup rows:", len(hte) - len(np.unique(hte)))
print("train/test shared feature rows:", len(set(htr) & set(hte)))

# adversarial validation quick
print("\n--- adversarial validation (LightGBM, 5-fold) ---")
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
X = pd.concat([tr[feat], te[feat]], axis=0, ignore_index=True)
a = np.r_[np.zeros(len(tr)), np.ones(len(te))]
X = X.select_dtypes(include=[np.number]).fillna(-999)
skf = StratifiedKFold(5, shuffle=True, random_state=42)
oof = np.zeros(len(X))
for tri, vai in skf.split(X, a):
    m = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31, verbose=-1, n_jobs=-1)
    m.fit(X.iloc[tri], a[tri])
    oof[vai] = m.predict_proba(X.iloc[vai])[:,1]
print("adversarial AUC:", roc_auc_score(a, oof), "(0.5 = no shift)")
imp = pd.Series(m.feature_importances_, index=X.columns).sort_values(ascending=False)
print("top shift features:\n", imp.head(10).to_string())
