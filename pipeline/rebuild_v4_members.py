import sys, time, json, os, glob
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import pandas as pd
import catboost as cb
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

T = r"C:\Users\amanuel\AppData\Local\Temp\opencode"
F = rf"{T}\drive\GCI Competition\features"

X = pd.read_parquet(rf"{T}\X_e2.parquet")
y = np.load(rf"{F}\y_train.npy")
te = pd.read_parquet(rf"{F}\test_engineered.parquet")
with open(rf"{F}\feature_cols.json") as fh:
    feature_cols = json.load(fh)
with open(rf"{F}\high_card_cols.json") as fh:
    te_cols = [c for c in json.load(fh) if c in X.columns]

DEAD = ["FLAG_MOBIL","ORGANIZATION_TYPE","EXT_SOURCE_2_WAS_MISSING","CNT_FAM_MEMBERS_WAS_MISSING",
        "AMT_REQ_CREDIT_BUREAU_MON_WAS_MISSING","AMT_REQ_CREDIT_BUREAU_YEAR_WAS_MISSING",
        "EXT_SOURCE_1_WAS_MISSING","EXT_SOURCE_3_WAS_MISSING","FLAG_EMP_PHONE",
        "EXT_SOURCE_3_QBIN","OCCUPATION_TYPE_FREQ","NAME_HOUSING_TYPE_FREQ","NAME_FAMILY_STATUS_FREQ",
        "NAME_EDUCATION_TYPE_FREQ","NAME_INCOME_TYPE_FREQ","CODE_GENDER_FREQ","NAME_CONTRACT_TYPE_FREQ",
        "ORGANIZATION_TYPE_FREQ"]
eps = 1e-5
ext = ["EXT_SOURCE_1","EXT_SOURCE_2","EXT_SOURCE_3"]

def enrich(df, base_cols):
    Xc = df.copy()
    filled = Xc[ext].replace(-999, np.nan)
    f = filled.fillna(filled.mean())
    age_years = (Xc["DAYS_BIRTH"].replace(-999, np.nan).abs() / 365.0).fillna(
        Xc["DAYS_BIRTH"].replace(-999, np.nan).abs().mean() / 365.0)
    for c in ext:
        Xc[f"{c}X_AGE"] = f[c] * age_years
    Xc["EXT_MEAN_X_AGE"] = f.mean(axis=1) * age_years
    Xc["EXT_2x3"] = f["EXT_SOURCE_2"] * f["EXT_SOURCE_3"]
    Xc["EXT_1x2x3_NANSAFE"] = f["EXT_SOURCE_1"] * f["EXT_SOURCE_2"] * f["EXT_SOURCE_3"]
    Xc["EXT_2_SQ"] = f["EXT_SOURCE_2"] ** 2
    Xc["EXT_3_SQ"] = f["EXT_SOURCE_3"] ** 2
    Xc["EXT_SUM"] = f.sum(axis=1)
    emp_years = Xc["DAYS_EMPLOYED"].replace(-999, np.nan).abs() / 365.0
    Xc["CREDIT_PER_INCOME_YEAR"] = Xc["AMT_CREDIT"] / ((Xc["AMT_INCOME_TOTAL"] + eps) * emp_years.fillna(emp_years.mean()) + eps)
    Xc["ANNUITY_PER_FAM"] = Xc["AMT_ANNUITY"] / (Xc["CNT_FAM_MEMBERS"].replace(0, 1) + eps)
    Xc["CREDIT_PER_FAM"] = Xc["AMT_CREDIT"] / (Xc["CNT_FAM_MEMBERS"].replace(0, 1) + eps)
    Xc["INCOME_PER_CHILD"] = Xc["AMT_INCOME_TOTAL"] / (Xc["CNT_CHILDREN"].clip(lower=0) + 1)
    Xc["NAN_COUNT"] = (df[base_cols] == -999).sum(axis=1).astype(np.int16)
    Xc["REGION_RATING_X_EXT_MEAN"] = Xc["REGION_RATING_CLIENT"] * f.mean(axis=1)
    return Xc

# test frame: IDENTICAL protocol to bench_test_preds.py (verified v3 test preds)
X_test = enrich(te[feature_cols].fillna(-999), feature_cols).drop(columns=[c for c in DEAD if c in feature_cols])
X_test = X_test[list(X.columns)]
assert list(X.columns) == list(X_test.columns)
X_test.to_parquet(rf"{T}\X_test_e2.parquet")
print(f"test frame: {X_test.shape}", flush=True)

CATS = ["NAME_CONTRACT_TYPE","CODE_GENDER","NAME_INCOME_TYPE","NAME_EDUCATION_TYPE",
        "NAME_FAMILY_STATUS","NAME_HOUSING_TYPE","OCCUPATION_TYPE","FLAG_WORK_PHONE"]
def ci_of(Xc):
    return [list(Xc.columns).index(c) for c in CATS if c in Xc.columns]

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
    X_tr, X_va, X_te_ = add_te(X.iloc[tri], X.iloc[vai], X_test, y[tri])
    fold_data.append((tri, vai, X_tr, X_va, X_te_))
print("folds ready", flush=True)

# rank-corr reference: verified XGBnew OOF
REF = rf"{T}\m3_XGB3.npy"
ref = np.load(REF) if os.path.exists(REF) else None
print("rankcorr ref:", REF, "loaded" if ref is not None else "MISSING", flush=True)

cat_params = json.load(open(rf"{T}\best_params_cat_e2.json"))
hgb_params = json.load(open(rf"{T}\best_params_hgb_e2.json"))
print("CAT params:", json.dumps(cat_params), flush=True)
print("HGB params:", json.dumps(hgb_params), flush=True)

# ---------- CAT, 2 seeds (new params) ----------
for seed in (42, 1337):
    t0 = time.time()
    oof = np.zeros(len(X)); test = np.zeros(len(X_test))
    for fold, (tri, vai, X_tr, X_va, X_te_) in enumerate(fold_data):
        cp = dict(cat_params); cp.update({"loss_function": "Logloss", "eval_metric": "AUC",
            "random_seed": seed, "verbose": False, "allow_writing_files": False, "thread_count": 8})
        m = cb.CatBoostClassifier(**cp, iterations=3000)
        m.fit(X_tr, y[tri], eval_set=(X_va, y[vai]), cat_features=ci_of(X_tr), early_stopping_rounds=200)
        oof[vai] = m.predict_proba(X_va)[:, 1]
        test += m.predict_proba(X_te_)[:, 1] / 5
        print(f"CAT s{seed} f{fold+1}: {roc_auc_score(y[vai], oof[vai]):.5f} ({time.time()-t0:.0f}s)", flush=True)
    auc = roc_auc_score(y, oof)
    np.save(rf"{T}\v4_CATnew_s{seed}_oof.npy", oof)
    np.save(rf"{T}\v4_CATnew_s{seed}_test.npy", test)
    rc = spearmanr(oof, ref).statistic if ref is not None else float("nan")
    print(f"CAT seed {seed} OOF AUC: {auc:.5f} | rankcorr vs XGBnew: {rc:.4f} ({time.time()-t0:.0f}s)", flush=True)

# ---------- HGB, tuned params, 1 seed (fast) ----------
t0 = time.time()
oof = np.zeros(len(X)); test = np.zeros(len(X_test))
for fold, (tri, vai, X_tr, X_va, X_te_) in enumerate(fold_data):
    m = HistGradientBoostingClassifier(**hgb_params, random_state=42)
    m.fit(X_tr, y[tri])
    oof[vai] = m.predict_proba(X_va)[:, 1]
    test += m.predict_proba(X_te_)[:, 1] / 5
    print(f"HGB f{fold+1}: {roc_auc_score(y[vai], oof[vai]):.5f} ({time.time()-t0:.0f}s)", flush=True)
auc = roc_auc_score(y, oof)
np.save(rf"{T}\v4_HGBnew_oof.npy", oof)
np.save(rf"{T}\v4_HGBnew_test.npy", test)
rc = spearmanr(oof, ref).statistic if ref is not None else float("nan")
print(f"HGB tuned OOF AUC: {auc:.5f} | rankcorr vs XGBnew: {rc:.4f}", flush=True)
print("V4 MEMBER REBUILDS DONE", flush=True)
