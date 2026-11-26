import sys, time, json, os
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import warnings
warnings.filterwarnings("ignore")
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
T = r"C:\Users\amanuel\AppData\Local\Temp\opencode"
OUT = r"C:\Users\amanuel\OneDrive\Documents\deepseek-harness\default-workspace\submissions"

X = pd.read_parquet(f"{T}\\X_e2.parquet")
y = np.load(f"{F}\\y_train.npy")
with open(f"{F}\\high_card_cols.json") as fh:
    high_card_cols = json.load(fh)
te_cols = [c for c in high_card_cols if c in X.columns]

def add_te(X_tr, X_va, y_tr, smoothing=10):
    X_tr = X_tr.copy(); X_va = X_va.copy()
    gs = pd.Series(y_tr, index=X_tr.index); gm = gs.mean()
    for col in te_cols:
        means = gs.groupby(X_tr[col]).mean()
        counts = X_tr[col].value_counts()
        sm = (means * counts + gm * smoothing) / (counts + smoothing)
        X_tr[f"{col}_TE"] = X_tr[col].map(sm).fillna(gm)
        X_va[f"{col}_TE"] = X_va[col].map(sm).fillna(gm)
    return X_tr, X_va

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
FOLDS = list(skf.split(X, y))
CATS = ["NAME_CONTRACT_TYPE","CODE_GENDER","NAME_INCOME_TYPE","NAME_EDUCATION_TYPE",
        "NAME_FAMILY_STATUS","NAME_HOUSING_TYPE","OCCUPATION_TYPE","FLAG_WORK_PHONE"]

fold_data = []
for tri, vai in FOLDS:
    X_tr, X_va = add_te(X.iloc[tri], X.iloc[vai], y[tri])
    fold_data.append((tri, vai, X_tr, X_va))
print("folds ready", flush=True)
def ci_of(Xc):
    return [list(Xc.columns).index(c) for c in CATS if c in Xc.columns]

# ---------- LGB new params, 3 seeds ----------
LGB_NEW = json.load(open(f"{T}\\best_params_lgb_e2.json"))
LGB_NEW["subsample_freq"] = 1
t0 = time.time()
oof = np.zeros(len(X))
for seed in [42, 1337, 2026]:
    oof_s = np.zeros(len(X))
    for fold, (tri, vai, X_tr, X_va) in enumerate(fold_data):
        m = lgb.LGBMClassifier(**LGB_NEW, n_estimators=3000, random_state=seed, verbose=-1, n_jobs=-1)
        m.fit(X_tr, y[tri], eval_set=[(X_va, y[vai])], eval_metric="auc",
              categorical_feature=ci_of(X_tr), callbacks=[lgb.early_stopping(200, verbose=False)])
        oof_s[vai] = m.predict_proba(X_va)[:, 1]
    oof += oof_s / 3
    print(f"  LGBnew seed {seed}: {roc_auc_score(y, oof_s):.5f} ({time.time()-t0:.0f}s)", flush=True)
np.save(f"{T}\\m3_LGB3.npy", oof)
auc_lgb_new = roc_auc_score(y, oof)
print(f"LGBnew 3-seed: {auc_lgb_new:.5f}", flush=True)

# ---------- XGB new params, 3 seeds ----------
XGB_NEW = json.load(open(f"{T}\\best_params_xgb_e2.json"))
xp0 = dict(XGB_NEW); xp0.update({"objective": "binary:logistic", "eval_metric": "auc",
                                  "tree_method": "hist", "n_jobs": 8, "verbosity": 0})
t0 = time.time()
oof = np.zeros(len(X))
for seed in [42, 1337, 2026]:
    oof_s = np.zeros(len(X))
    for fold, (tri, vai, X_tr, X_va) in enumerate(fold_data):
        dtr = xgb.DMatrix(X_tr, label=y[tri]); dva = xgb.DMatrix(X_va, label=y[vai])
        m = xgb.train({**xp0, "random_state": seed}, dtr, num_boost_round=3000,
                      evals=[(dva, "val")], early_stopping_rounds=200, verbose_eval=False)
        oof_s[vai] = m.predict(dva, iteration_range=(0, m.best_iteration + 1))
    oof += oof_s / 3
    print(f"  XGBnew seed {seed}: {roc_auc_score(y, oof_s):.5f} ({time.time()-t0:.0f}s)", flush=True)
np.save(f"{T}\\m3_XGB3.npy", oof)
auc_xgb_new = roc_auc_score(y, oof)
print(f"XGBnew 3-seed: {auc_xgb_new:.5f}", flush=True)

# ---------- assemble 6 members (old + new generations) ----------
mem = {
    "LGBold": np.load(f"{T}\\m2_LGB3.npy"),
    "LGBnew": np.load(f"{T}\\m3_LGB3.npy"),
    "XGBold": np.load(f"{T}\\m2_XGB3.npy"),
    "XGBnew": np.load(f"{T}\\m3_XGB3.npy"),
    "CAT":    np.load(f"{T}\\m2_CAT2.npy"),
    "HGB":    np.load(f"{T}\\m2_HGB.npy"),
}
names = list(mem.keys())
M = [mem[k] for k in names]
print("--- members ---", flush=True)
for k, v in mem.items():
    print(f"  {k}: {roc_auc_score(y, v):.5f}", flush=True)

# ---------- hill-climb with replacement over 6 members ----------
cur = np.zeros(len(y)); counts = {k: 0 for k in names}
best_auc = roc_auc_score(y, cur)
while True:
    step_best = (best_auc, None)
    for k in names:
        trial_pred = (cur * sum(counts.values()) + mem[k]) / (sum(counts.values()) + 1)
        a = roc_auc_score(y, trial_pred)
        if a > step_best[0]:
            step_best = (a, k)
    if step_best[1] is None:
        break
    k = step_best[1]; counts[k] += 1
    cur = (cur * (sum(counts.values()) - 1) + mem[k]) / sum(counts.values())
    best_auc = step_best[0]
print(f"hill-climb: {counts} -> {best_auc:.5f}", flush=True)

# ---------- fine grid around hill-climb weights ----------
best = (best_auc, counts.copy())
total = sum(counts.values())
for k in names:
    for d in np.arange(-0.10, 0.101, 0.02):
        w = {kk: counts[kk] / total for kk in names}
        w[k] = round(w[k] + float(d), 3)
        if w[k] < 0: continue
        rest = 1 - w[k]
        s = sum(v for kk, v in w.items() if kk != k)
        if s <= 0: continue
        for kk in names:
            if kk != k: w[kk] = w[kk] / s * rest
        pred = sum(mem[kk] * w[kk] for kk in names)
        a = roc_auc_score(y, pred)
        if a > best[0]: best = (a, w.copy())
# second refinement pass at 0.01
ref = best[1].copy()
for k in names:
    for d in np.arange(-0.04, 0.041, 0.01):
        w = ref.copy()
        w[k] = round(w[k] + float(d), 3)
        if w[k] < 0: continue
        rest = 1 - w[k]
        s = sum(v for kk, v in w.items() if kk != k)
        if s <= 0: continue
        for kk in names:
            if kk != k: w[kk] = w[kk] / s * rest
        pred = sum(mem[kk] * w[kk] for kk in names)
        a = roc_auc_score(y, pred)
        if a > best[0]: best = (a, w.copy())
final_w = {k: round(best[1][k], 3) for k in names}
print(f"FINAL v3 recipe: {final_w} -> {best[0]:.5f}", flush=True)
print(f"(v2 recipe was 0.76203)", flush=True)

json.dump({"weights": final_w, "cv": best[0]}, open(f"{T}\\v3_recipe.json", "w"), indent=2)

if best[0] <= 0.76203:
    print("NOT BETTER than v2 (0.76203) -> keeping v2, no new CSV", flush=True)
    sys.exit(0)

# ================= TEST PREDS =================
te = pd.read_parquet(f"{F}\\test_engineered.parquet")
with open(f"{F}\\feature_cols.json") as fh:
    feature_cols = json.load(fh)
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

def add_te3(X_tr, X_va, X_te_, y_tr, smoothing=10):
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

test_pred = np.zeros(len(X_test))

def tp_lgb(seeds):
    tp = np.zeros(len(X_test))
    for seed in seeds:
        for tri, vai, X_tr, X_va in fold_data:
            _, _, X_te_ = add_te3(X.iloc[tri], X.iloc[vai], X_test, y[tri])
            m = lgb.LGBMClassifier(**LGB_NEW, n_estimators=3000, random_state=seed, verbose=-1, n_jobs=-1)
            m.fit(X_tr, y[tri], eval_set=[(X_va, y[vai])], eval_metric="auc",
                  categorical_feature=ci_of(X_tr), callbacks=[lgb.early_stopping(200, verbose=False)])
            tp += m.predict_proba(X_te_)[:, 1] / 5
        print(f"  LGBnew test seed {seed} done", flush=True)
    return tp / len(seeds)

def tp_xgb(seeds):
    tp = np.zeros(len(X_test))
    for seed in seeds:
        for tri, vai, X_tr, X_va in fold_data:
            _, _, X_te_ = add_te3(X.iloc[tri], X.iloc[vai], X_test, y[tri])
            dtr = xgb.DMatrix(X_tr, label=y[tri]); dte = xgb.DMatrix(X_te_)
            m = xgb.train({**xp0, "random_state": seed}, dtr, num_boost_round=3000,
                          evals=[(xgb.DMatrix(X_va, label=y[vai]), "val")],
                          early_stopping_rounds=200, verbose_eval=False)
            tp += m.predict(dte, iteration_range=(0, m.best_iteration + 1)) / 5
        print(f"  XGBnew test seed {seed} done", flush=True)
    return tp / len(seeds)

def tp_hgb():
    tp = np.zeros(len(X_test))
    for tri, vai, X_tr, X_va in fold_data:
        _, _, X_te_ = add_te3(X.iloc[tri], X.iloc[vai], X_test, y[tri])
        m = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=600, max_leaf_nodes=31,
                                           min_samples_leaf=50, l2_regularization=1.0, random_state=42)
        m.fit(X_tr, y[tri])
        tp += m.predict_proba(X_te_)[:, 1] / 5
    return tp

CAT_PARAMS = json.load(open(f"{BASE}\\optuna_studies\\best_params_cat.json"))
def tp_cat(seeds):
    tp = np.zeros(len(X_test))
    for seed in seeds:
        cp0 = dict(CAT_PARAMS); cp0.update({"loss_function": "Logloss", "eval_metric": "AUC",
            "random_seed": seed, "verbose": False, "allow_writing_files": False, "thread_count": 8})
        for tri, vai, X_tr, X_va in fold_data:
            _, _, X_te_ = add_te3(X.iloc[tri], X.iloc[vai], X_test, y[tri])
            m = cb.CatBoostClassifier(**cp0, iterations=3000)
            m.fit(X_tr, y[tri], eval_set=(X_va, y[vai]), cat_features=ci_of(X_tr), early_stopping_rounds=200)
            tp += m.predict_proba(X_te_)[:, 1] / 5
        print(f"  CAT test seed {seed} done", flush=True)
    return tp / len(seeds)

P = final_w
if P["LGBnew"] > 0:
    a = tp_lgb([42, 1337, 2026]); np.save(f"{T}\\v3_test_LGBnew.npy", a); test_pred += P["LGBnew"] * a
if P["LGBold"] > 0:
    # old-params LGB test preds: 3 seeds
    LGB_OLD = json.load(open(f"{T}\\bench_winner_params.json"))
    tp = np.zeros(len(X_test))
    for seed in [42, 1337, 2026]:
        for tri, vai, X_tr, X_va in fold_data:
            _, _, X_te_ = add_te3(X.iloc[tri], X.iloc[vai], X_test, y[tri])
            m = lgb.LGBMClassifier(**LGB_OLD, n_estimators=3000, random_state=seed, verbose=-1, n_jobs=-1)
            m.fit(X_tr, y[tri], eval_set=[(X_va, y[vai])], eval_metric="auc",
                  categorical_feature=ci_of(X_tr), callbacks=[lgb.early_stopping(200, verbose=False)])
            tp += m.predict_proba(X_te_)[:, 1] / 5
        print(f"  LGBold test seed {seed} done", flush=True)
    tp /= 3; np.save(f"{T}\\v3_test_LGBold.npy", tp); test_pred += P["LGBold"] * tp
if P["XGBnew"] > 0:
    a = tp_xgb([42, 1337, 2026]); np.save(f"{T}\\v3_test_XGBnew.npy", a); test_pred += P["XGBnew"] * a
if P["XGBold"] > 0:
    XGB_OLD = json.load(open(f"{BASE}\\optuna_studies\\best_params_xgb.json"))
    xo = dict(XGB_OLD); xo.update({"objective": "binary:logistic", "eval_metric": "auc",
                                    "tree_method": "hist", "n_jobs": 8, "verbosity": 0})
    tp = np.zeros(len(X_test))
    for seed in [42, 1337, 2026]:
        for tri, vai, X_tr, X_va in fold_data:
            _, _, X_te_ = add_te3(X.iloc[tri], X.iloc[vai], X_test, y[tri])
            dtr = xgb.DMatrix(X_tr, label=y[tri]); dte = xgb.DMatrix(X_te_)
            m = xgb.train({**xo, "random_state": seed}, dtr, num_boost_round=3000,
                          evals=[(xgb.DMatrix(X_va, label=y[vai]), "val")],
                          early_stopping_rounds=200, verbose_eval=False)
            tp += m.predict(dte, iteration_range=(0, m.best_iteration + 1)) / 5
        print(f"  XGBold test seed {seed} done", flush=True)
    tp /= 3; np.save(f"{T}\\v3_test_XGBold.npy", tp); test_pred += P["XGBold"] * tp
if P["CAT"] > 0:
    a = tp_cat([42, 1337]); np.save(f"{T}\\v3_test_CAT.npy", a); test_pred += P["CAT"] * a
if P["HGB"] > 0:
    a = tp_hgb(); np.save(f"{T}\\v3_test_HGB.npy", a); test_pred += P["HGB"] * a

sample = pd.read_csv(f"{BASE}\\input\\sample_submission.csv")
sub = pd.DataFrame({"SK_ID_CURR": te["SK_ID_CURR"].values, "TARGET": test_pred})
sub = sub.set_index("SK_ID_CURR").loc[sample["SK_ID_CURR"]].reset_index()
assert len(sub) == len(sample)
assert sub["TARGET"].between(0, 1).all() and not sub["TARGET"].isnull().any()
cv_str = f"{best[0]:.5f}"
path = f"{OUT}\\submission_v3_recipe_cv{cv_str}.csv"
sub.to_csv(path, index=False)
print(f"SAVED: {path}", flush=True)
print(f"stats: mean={test_pred.mean():.5f} std={test_pred.std():.5f} min={test_pred.min():.5f} max={test_pred.max():.5f}", flush=True)
print("V3 ALL DONE", flush=True)
