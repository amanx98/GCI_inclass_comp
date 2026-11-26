import sys, time, json
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier

BASE = r"C:\Users\amanuel\AppData\Local\Temp\opencode\drive\GCI Competition"
F = f"{BASE}\\features"

tr = pd.read_parquet(f"{F}\\train_engineered.parquet")
y = np.load(f"{F}\\y_train.npy")
with open(f"{F}\\feature_cols.json") as f:
    feature_cols = json.load(f)
with open(f"{F}\\high_card_cols.json") as f:
    high_card_cols = json.load(f)

PARAMS = json.load(open(r"C:\Users\amanuel\AppData\Local\Temp\opencode\bench_winner_params.json"))

X0 = tr[feature_cols].fillna(-999)
dead = ["FLAG_MOBIL","ORGANIZATION_TYPE","EXT_SOURCE_2_WAS_MISSING","CNT_FAM_MEMBERS_WAS_MISSING",
        "AMT_REQ_CREDIT_BUREAU_MON_WAS_MISSING","AMT_REQ_CREDIT_BUREAU_YEAR_WAS_MISSING",
        "EXT_SOURCE_1_WAS_MISSING","EXT_SOURCE_3_WAS_MISSING","FLAG_EMP_PHONE",
        "EXT_SOURCE_3_QBIN","OCCUPATION_TYPE_FREQ","NAME_HOUSING_TYPE_FREQ","NAME_FAMILY_STATUS_FREQ",
        "NAME_EDUCATION_TYPE_FREQ","NAME_INCOME_TYPE_FREQ","CODE_GENDER_FREQ","NAME_CONTRACT_TYPE_FREQ",
        "ORGANIZATION_TYPE_FREQ"]

def enrich2(X):
    X = X.copy()
    eps = 1e-5
    ext = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    filled = X[ext].replace(-999, np.nan)
    f = filled.fillna(filled.mean())
    age_years = (X["DAYS_BIRTH"].replace(-999, np.nan).abs() / 365.0).fillna(
        X["DAYS_BIRTH"].replace(-999, np.nan).abs().mean() / 365.0)
    for c in ext:
        X[f"{c}X_AGE"] = f[c] * age_years
    X["EXT_MEAN_X_AGE"] = f.mean(axis=1) * age_years
    X["EXT_2x3"] = f["EXT_SOURCE_2"] * f["EXT_SOURCE_3"]
    X["EXT_1x2x3_NANSAFE"] = f["EXT_SOURCE_1"] * f["EXT_SOURCE_2"] * f["EXT_SOURCE_3"]
    X["EXT_2_SQ"] = f["EXT_SOURCE_2"] ** 2
    X["EXT_3_SQ"] = f["EXT_SOURCE_3"] ** 2
    X["EXT_SUM"] = f.sum(axis=1)
    emp_years = X["DAYS_EMPLOYED"].replace(-999, np.nan).abs() / 365.0
    X["CREDIT_PER_INCOME_YEAR"] = X["AMT_CREDIT"] / ((X["AMT_INCOME_TOTAL"] + eps) * emp_years.fillna(emp_years.mean()) + eps)
    X["ANNUITY_PER_FAM"] = X["AMT_ANNUITY"] / (X["CNT_FAM_MEMBERS"].replace(0, 1) + eps)
    X["CREDIT_PER_FAM"] = X["AMT_CREDIT"] / (X["CNT_FAM_MEMBERS"].replace(0, 1) + eps)
    X["INCOME_PER_CHILD"] = X["AMT_INCOME_TOTAL"] / (X["CNT_CHILDREN"].clip(lower=0) + 1)
    X["NAN_COUNT"] = (X0[feature_cols] == -999).sum(axis=1).astype(np.int16)
    X["REGION_RATING_X_EXT_MEAN"] = X["REGION_RATING_CLIENT"] * f.mean(axis=1)
    return X

def add_te(X_tr, X_va, y_tr, te_cols, smoothing=10):
    X_tr = X_tr.copy(); X_va = X_va.copy()
    gs = pd.Series(y_tr, index=X_tr.index)
    gm = gs.mean()
    for col in te_cols:
        if col not in X_tr.columns:
            continue
        means = gs.groupby(X_tr[col]).mean()
        counts = X_tr[col].value_counts()
        sm = (means * counts + gm * smoothing) / (counts + smoothing)
        X_tr[f"{col}_TE"] = X_tr[col].map(sm).fillna(gm)
        X_va[f"{col}_TE"] = X_va[col].map(sm).fillna(gm)
    return X_tr, X_va

te_cols = [c for c in high_card_cols if c in X0.columns]
cats = ["NAME_CONTRACT_TYPE","CODE_GENDER","NAME_INCOME_TYPE","NAME_EDUCATION_TYPE",
        "NAME_FAMILY_STATUS","NAME_HOUSING_TYPE","OCCUPATION_TYPE","FLAG_WORK_PHONE"]

# E2 config: enriched + ALL dead dropped, TE per fold, NO combo TE
X = enrich2(X0).drop(columns=[c for c in dead if c in X0.columns])
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_data = []
for fold, (tri, vai) in enumerate(skf.split(X, y)):
    X_tr, X_va = add_te(X.iloc[tri].copy(), X.iloc[vai].copy(), y[tri], te_cols)
    fold_data.append((tri, vai, X_tr, X_va))
print(f"E2 config ready: {X.shape[1]} base cols", flush=True)
X.to_parquet(r"C:\Users\amanuel\AppData\Local\Temp\opencode\X_e2.parquet")

def ci_of(Xc):
    return [list(Xc.columns).index(c) for c in cats if c in Xc.columns]

# ---- LGB 3 seeds ----
t0 = time.time()
oof_lgb3 = np.zeros(len(X))
for seed in [42, 1337, 2026]:
    oof_s = np.zeros(len(X))
    for fold, (tri, vai, X_tr, X_va) in enumerate(fold_data):
        m = lgb.LGBMClassifier(**PARAMS, n_estimators=3000, random_state=seed, verbose=-1, n_jobs=-1)
        m.fit(X_tr, y[tri], eval_set=[(X_va, y[vai])], eval_metric="auc",
              categorical_feature=ci_of(X_tr),
              callbacks=[lgb.early_stopping(200, verbose=False)])
        oof_s[vai] = m.predict_proba(X_va)[:, 1]
    print(f"  LGB seed {seed}: {roc_auc_score(y, oof_s):.5f}", flush=True)
    oof_lgb3 += oof_s / 3
auc_lgb3 = roc_auc_score(y, oof_lgb3)
print(f"{'LGB_tuned seed-avg x3':40s}: OOF AUC = {auc_lgb3:.5f}  ({time.time()-t0:.0f}s)", flush=True)
np.save(r"C:\Users\amanuel\AppData\Local\Temp\opencode\m2_LGB3.npy", oof_lgb3)

# ---- HGB ----
t0 = time.time()
oof_hgb = np.zeros(len(X))
for fold, (tri, vai, X_tr, X_va) in enumerate(fold_data):
    m = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=600, max_leaf_nodes=31,
                                       min_samples_leaf=50, l2_regularization=1.0, random_state=42)
    m.fit(X_tr, y[tri])
    oof_hgb[vai] = m.predict_proba(X_va)[:, 1]
auc_hgb = roc_auc_score(y, oof_hgb)
print(f"{'HGB':40s}: OOF AUC = {auc_hgb:.5f}  ({time.time()-t0:.0f}s)", flush=True)
np.save(r"C:\Users\amanuel\AppData\Local\Temp\opencode\m2_HGB.npy", oof_hgb)

# ---- XGB ----
XGB_PARAMS = json.load(open(f"{BASE}\\optuna_studies\\best_params_xgb.json"))
xp = dict(XGB_PARAMS)
xp.update({"objective": "binary:logistic", "eval_metric": "auc",
           "tree_method": "hist", "random_state": 42, "n_jobs": 8, "verbosity": 0})
t0 = time.time()
oof_xgb = np.zeros(len(X))
for fold, (tri, vai, X_tr, X_va) in enumerate(fold_data):
    dtrain = xgb.DMatrix(X_tr, label=y[tri])
    dval = xgb.DMatrix(X_va, label=y[vai])
    m = xgb.train(xp, dtrain, num_boost_round=3000,
                  evals=[(dval, "val")], early_stopping_rounds=200, verbose_eval=False)
    oof_xgb[vai] = m.predict(dval, iteration_range=(0, m.best_iteration + 1))
auc_xgb = roc_auc_score(y, oof_xgb)
print(f"{'XGB tuned (full data)':40s}: OOF AUC = {auc_xgb:.5f}  ({time.time()-t0:.0f}s)", flush=True)
np.save(r"C:\Users\amanuel\AppData\Local\Temp\opencode\m2_XGB.npy", oof_xgb)

# ---- CAT ----
CAT_PARAMS = json.load(open(f"{BASE}\\optuna_studies\\best_params_cat.json"))
cp = dict(CAT_PARAMS)
cp.update({"loss_function": "Logloss", "eval_metric": "AUC", "random_seed": 42,
           "verbose": False, "allow_writing_files": False, "thread_count": 8})
t0 = time.time()
oof_cat = np.zeros(len(X))
for fold, (tri, vai, X_tr, X_va) in enumerate(fold_data):
    m = cb.CatBoostClassifier(**cp, iterations=3000)
    m.fit(X_tr, y[tri], eval_set=(X_va, y[vai]),
          cat_features=ci_of(X_tr), early_stopping_rounds=200)
    oof_cat[vai] = m.predict_proba(X_va)[:, 1]
auc_cat = roc_auc_score(y, oof_cat)
print(f"{'CAT tuned (full data)':40s}: OOF AUC = {auc_cat:.5f}  ({time.time()-t0:.0f}s)", flush=True)
np.save(r"C:\Users\amanuel\AppData\Local\Temp\opencode\m2_CAT.npy", oof_cat)

# ---- final hill-climb over all members ----
mem = {"LGB3": oof_lgb3, "HGB": oof_hgb, "XGB": oof_xgb, "CAT": oof_cat}
print("\n--- member summary (E2 config) ---")
for k, v in mem.items():
    print(f"  {k}: {roc_auc_score(y, v):.5f}")

print("\n--- greedy hill-climb (averaging with replacement) ---")
current, chosen = None, []
for step in range(10):
    best = (None, -1.0)
    for k, v in mem.items():
        cand = v.copy() if current is None else (current + v) / 2
        a = roc_auc_score(y, cand)
        if a > best[1]:
            best = (k, a)
    if current is not None and best[1] <= roc_auc_score(y, current):
        print("  no improvement, stop")
        break
    current = mem[best[0]].copy() if current is None else (current + mem[best[0]]) / 2
    chosen.append(best[0])
    print(f"  step {step+1}: +{best[0]} -> {best[1]:.5f} | {chosen}")

# 3-way weight grid over top members
names = list(mem.keys())
import itertools
print("\n--- exhaustive 4-way weight grid (step 0.05, simplex) ---")
best = (-1, None)
for w1 in np.arange(0, 1.0001, 0.05):
    for w2 in np.arange(0, 1.0001 - w1, 0.05):
        for w3 in np.arange(0, 1.0001 - w1 - w2 + 1e-9, 0.05):
            w4 = 1 - w1 - w2 - w3
            if w4 < -1e-9: continue
            blend = w1*oof_lgb3 + w2*oof_hgb + w3*oof_xgb + w4*oof_cat
            a = roc_auc_score(y, blend)
            if a > best[0]:
                best = (a, (round(w1,2), round(w2,2), round(w3,2), round(w4,2)))
print(f"best grid blend: weights LGB/HGB/XGB/CAT = {best[1]} -> {best[0]:.5f}")
print("members2 done")
