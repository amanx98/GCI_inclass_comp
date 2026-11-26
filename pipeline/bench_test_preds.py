import sys, time, json
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

BASE = r"C:\Users\amanuel\AppData\Local\Temp\opencode\drive\GCI Competition"
F = f"{BASE}\\features"
T = r"C:\Users\amanuel\AppData\Local\Temp\opencode"
OUT = r"C:\Users\amanuel\OneDrive\Documents\deepseek-harness\default-workspace\submissions"

X = pd.read_parquet(f"{T}\\X_e2.parquet")
y = np.load(f"{F}\\y_train.npy")
te = pd.read_parquet(f"{F}\\test_engineered.parquet")
with open(f"{F}\\feature_cols.json") as f:
    feature_cols = json.load(f)
with open(f"{F}\\high_card_cols.json") as f:
    high_card_cols = json.load(f)

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

X_test = enrich(te[feature_cols].fillna(-999), feature_cols).drop(columns=[c for c in DEAD if c in feature_cols])
X_test = X_test[list(X.columns)]
print(f"test frame: {X_test.shape}", flush=True)

def add_te(X_tr, X_va, X_te_, y_tr, smoothing=10):
    X_tr = X_tr.copy(); X_va = X_va.copy(); X_te_ = X_te_.copy()
    gs = pd.Series(y_tr, index=X_tr.index); gm = gs.mean()
    for col in te_cols:
        means = gs.groupby(X_tr[col]).mean()
        counts = X_tr[col].value_counts()
        sm = (means * counts + gm * smoothing) / (counts + smoothing)
        X_tr[f"{col}_TE"] = X_tr[col].map(sm).fillna(gm)
        X_va[f"{col}_TE"] = X_va[col].map(sm).fillna(gm)
        X_te_[f"{col}_TE"] = X_te_[col].map(sm).fillna(gm)
    return X_tr, X_va, X_te_

te_cols = [c for c in high_card_cols if c in X.columns]
CATS = ["NAME_CONTRACT_TYPE","CODE_GENDER","NAME_INCOME_TYPE","NAME_EDUCATION_TYPE",
        "NAME_FAMILY_STATUS","NAME_HOUSING_TYPE","OCCUPATION_TYPE","FLAG_WORK_PHONE"]
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
FOLDS = list(skf.split(X, y))
def ci_of(Xc):
    return [list(Xc.columns).index(c) for c in CATS if c in Xc.columns]

LGB_PARAMS = json.load(open(f"{T}\\bench_winner_params.json"))
XGB_PARAMS = json.load(open(f"{BASE}\\optuna_studies\\best_params_xgb.json"))
CAT_PARAMS = json.load(open(f"{BASE}\\optuna_studies\\best_params_cat.json"))

test_pred = np.zeros(len(X_test))
W = {"LGB": 0.06, "XGB": 0.64, "CAT": 0.30}

# ---- LGB 3 seeds ----
t0 = time.time()
tp_lgb = np.zeros(len(X_test))
for seed in [42, 1337, 2026]:
    for fold, (tri, vai) in enumerate(FOLDS):
        X_tr, X_va, X_te_ = add_te(X.iloc[tri], X.iloc[vai], X_test, y[tri])
        m = lgb.LGBMClassifier(**LGB_PARAMS, n_estimators=3000, random_state=seed, verbose=-1, n_jobs=-1)
        m.fit(X_tr, y[tri], eval_set=[(X_va, y[vai])], eval_metric="auc",
              categorical_feature=ci_of(X_tr), callbacks=[lgb.early_stopping(200, verbose=False)])
        tp_lgb += m.predict_proba(X_te_)[:, 1] / 5
    print(f"  LGB seed {seed} test done ({time.time()-t0:.0f}s)", flush=True)
tp_lgb /= 3
test_pred += W["LGB"] * tp_lgb

# ---- XGB 3 seeds ----
t0 = time.time()
tp_xgb = np.zeros(len(X_test))
xp0 = dict(XGB_PARAMS); xp0.update({"objective":"binary:logistic","eval_metric":"auc","tree_method":"hist","n_jobs":8,"verbosity":0})
for seed in [42, 1337, 2026]:
    for fold, (tri, vai) in enumerate(FOLDS):
        X_tr, X_va, X_te_ = add_te(X.iloc[tri], X.iloc[vai], X_test, y[tri])
        dtr = xgb.DMatrix(X_tr, label=y[tri]); dte = xgb.DMatrix(X_te_)
        m = xgb.train({**xp0, "random_state":seed}, dtr, num_boost_round=3000,
                      evals=[(xgb.DMatrix(X_va, label=y[vai]), "val")],
                      early_stopping_rounds=200, verbose_eval=False)
        tp_xgb += m.predict(dte, iteration_range=(0, m.best_iteration + 1)) / 5
    print(f"  XGB seed {seed} test done ({time.time()-t0:.0f}s)", flush=True)
tp_xgb /= 3
test_pred += W["XGB"] * tp_xgb

# ---- CAT 1 seed ----
t0 = time.time()
tp_cat = np.zeros(len(X_test))
cp0 = dict(CAT_PARAMS); cp0.update({"loss_function":"Logloss","eval_metric":"AUC","random_seed":42,
    "verbose":False,"allow_writing_files":False,"thread_count":8})
for fold, (tri, vai) in enumerate(FOLDS):
    X_tr, X_va, X_te_ = add_te(X.iloc[tri], X.iloc[vai], X_test, y[tri])
    m = cb.CatBoostClassifier(**cp0, iterations=3000)
    m.fit(X_tr, y[tri], eval_set=(X_va, y[vai]), cat_features=ci_of(X_tr), early_stopping_rounds=200)
    tp_cat += m.predict_proba(X_te_)[:, 1] / 5
    print(f"  CAT fold {fold+1} test done ({time.time()-t0:.0f}s)", flush=True)
test_pred += W["CAT"] * tp_cat

# ---- build submission ----
sample = pd.read_csv(f"{BASE}\\input\\sample_submission.csv")
sub = pd.DataFrame({"SK_ID_CURR": te["SK_ID_CURR"].values, "TARGET": test_pred})
sub = sub.set_index("SK_ID_CURR").loc[sample["SK_ID_CURR"]].reset_index()
assert len(sub) == len(sample)
assert sub["TARGET"].between(0, 1).all() and not sub["TARGET"].isnull().any()
path = f"{OUT}\\submission_v2_recipe_cv0.76203.csv"
sub.to_csv(path, index=False)
print(f"SAVED: {path}", flush=True)
print(f"stats: mean={test_pred.mean():.5f} std={test_pred.std():.5f} min={test_pred.min():.5f} max={test_pred.max():.5f}")
print("v2 recipe submission done")
