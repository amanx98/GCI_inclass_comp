import sys, os, json, time
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

T = r"C:\Users\amanuel\AppData\Local\Temp\opencode"
F = rf"{T}\drive\GCI Competition\features"
X = pd.read_parquet(rf"{T}\X_e2.parquet")
y = np.load(rf"{F}\y_train.npy")
with open(rf"{F}\high_card_cols.json") as fh:
    te_cols = [c for c in json.load(fh) if c in X.columns]
skf = StratifiedKFold(5, shuffle=True, random_state=42)
tri, vai = next(iter(skf.split(X, y)))

def add_te(X_tr, X_va, y_tr, smoothing=10):
    X_tr, X_va = X_tr.copy(), X_va.copy()
    gs = pd.Series(y_tr, index=X_tr.index); gm = gs.mean()
    for col in te_cols:
        means = gs.groupby(X_tr[col]).mean(); counts = X_tr[col].value_counts()
        sm = (means * counts + gm * smoothing) / (counts + smoothing)
        X_tr[f"{col}_TE"] = X_tr[col].map(sm).fillna(gm)
        X_va[f"{col}_TE"] = X_va[col].map(sm).fillna(gm)
    return X_tr, X_va

X_tr, X_va = add_te(X.iloc[tri], X.iloc[vai], y[tri])
CATS = ["NAME_CONTRACT_TYPE","CODE_GENDER","NAME_INCOME_TYPE","NAME_EDUCATION_TYPE",
        "NAME_FAMILY_STATUS","NAME_HOUSING_TYPE","OCCUPATION_TYPE","FLAG_WORK_PHONE"]
ci = [list(X_tr.columns).index(c) for c in CATS if c in X_tr.columns]
yy = y[vai]

print("fold0 train/val:", len(tri), len(vai), flush=True)

# LGB
LGB = json.load(open(rf"{T}\best_params_lgb_e2.json")); LGB["subsample_freq"] = 1
res = {}
for nj in (2, -1):
    t0=time.time()
    m = lgb.LGBMClassifier(**LGB, n_estimators=3000, random_state=42, verbose=-1, n_jobs=nj)
    m.fit(X_tr, y[tri], eval_set=[(X_va, yy)], eval_metric="auc", categorical_feature=ci,
          callbacks=[lgb.early_stopping(200, verbose=False)])
    p = m.predict_proba(X_va)[:,1]; res[nj]=p
    print(f"LGB n_jobs={nj}: auc={roc_auc_score(yy,p):.6f} best_iter={m.best_iteration_} {time.time()-t0:.0f}s", flush=True)
print(f"LGB max|diff| n_jobs 2 vs -1: {np.abs(res[2]-res[-1]).max():.3e} corr={np.corrcoef(res[2],res[-1])[0,1]:.9f}", flush=True)

# XGB
XP = json.load(open(rf"{T}\best_params_xgb_e2.json"))
xp = dict(XP); xp.update({"objective":"binary:logistic","eval_metric":"auc","tree_method":"hist","verbosity":0})
res = {}
for nj in (2, 8):
    t0=time.time()
    dtr=xgb.DMatrix(X_tr, label=y[tri]); dva=xgb.DMatrix(X_va, label=yy)
    m=xgb.train({**xp,"n_jobs":nj,"random_state":42}, dtr, num_boost_round=3000,
                evals=[(dva,"val")], early_stopping_rounds=200, verbose_eval=False)
    p=m.predict(dva, iteration_range=(0,m.best_iteration+1)); res[nj]=p
    print(f"XGB n_jobs={nj}: auc={roc_auc_score(yy,p):.6f} best_iter={m.best_iteration} {time.time()-t0:.0f}s", flush=True)
print(f"XGB max|diff| n_jobs 2 vs 8: {np.abs(res[2]-res[8]).max():.3e} corr={np.corrcoef(res[2],res[8])[0,1]:.9f}", flush=True)

# CAT
CP = json.load(open(rf"{T}\best_params_cat_e2.json"))
res = {}
for tc in (2, 8):
    t0=time.time()
    cp = dict(CP); cp.update({"loss_function":"Logloss","eval_metric":"AUC","random_seed":42,
        "verbose":False,"allow_writing_files":False,"thread_count":tc})
    m=cb.CatBoostClassifier(**cp, iterations=3000)
    m.fit(X_tr, y[tri], eval_set=(X_va, yy), cat_features=ci, early_stopping_rounds=200)
    p=m.predict_proba(X_va)[:,1]; res[tc]=p
    print(f"CAT thread_count={tc}: auc={roc_auc_score(yy,p):.6f} best_iter={m.get_best_iteration()} {time.time()-t0:.0f}s", flush=True)
print(f"CAT max|diff| tc 2 vs 8: {np.abs(res[2]-res[8]).max():.3e} corr={np.corrcoef(res[2],res[8])[0,1]:.9f}", flush=True)
print("THREAD TEST DONE", flush=True)
