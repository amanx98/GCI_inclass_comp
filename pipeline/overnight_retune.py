import sys, time, json, os
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

BASE = r"C:\Users\amanuel\AppData\Local\Temp\opencode\drive\GCI Competition"
F = f"{BASE}\\features"
T = r"C:\Users\amanuel\AppData\Local\Temp\opencode"

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

def prep_folds(y_tr_idx=None):
    out = []
    for tri, vai in FOLDS:
        X_tr, X_va = add_te(X.iloc[tri], X.iloc[vai], y[tri])
        out.append((tri, vai, X_tr, X_va))
    return out

FOLD_DATA = prep_folds()
print("folds ready", flush=True)

# ============================================================
# STAGE 1: CAT seed 1337 (completes local 2-seed CAT) ~25 min
# ============================================================
CAT_PARAMS = json.load(open(f"{BASE}\\optuna_studies\\best_params_cat.json"))
cp0 = dict(CAT_PARAMS); cp0.update({"loss_function": "Logloss", "eval_metric": "AUC",
    "random_seed": 1337, "verbose": False, "allow_writing_files": False, "thread_count": 8})
t0 = time.time()
oof_cat_1337 = np.zeros(len(X))
for fold, (tri, vai, X_tr, X_va) in enumerate(FOLD_DATA):
    m = cb.CatBoostClassifier(**cp0, iterations=3000)
    m.fit(X_tr, y[tri], eval_set=(X_va, y[vai]),
          cat_features=[list(X_tr.columns).index(c) for c in
                        ["NAME_CONTRACT_TYPE","CODE_GENDER","NAME_INCOME_TYPE","NAME_EDUCATION_TYPE",
                         "NAME_FAMILY_STATUS","NAME_HOUSING_TYPE","OCCUPATION_TYPE","FLAG_WORK_PHONE"]
                        if c in X_tr.columns],
          early_stopping_rounds=200)
    oof_cat_1337[vai] = m.predict_proba(X_va)[:, 1]
    print(f"  cat s1337 fold {fold+1}/5 ({time.time()-t0:.0f}s)", flush=True)
a = roc_auc_score(y, oof_cat_1337)
np.save(f"{T}\\m2_CAT_s1337.npy", oof_cat_1337)
oof_cat2 = (np.load(f"{T}\\m2_CAT.npy") + oof_cat_1337) / 2
a2 = roc_auc_score(y, oof_cat2)
np.save(f"{T}\\m2_CAT2.npy", oof_cat2)
print(f"CAT seed 1337: {a:.5f} | CAT 2-seed avg: {a2:.5f} ({time.time()-t0:.0f}s)", flush=True)

# ============================================================
# STAGE 2: LGB Optuna re-tune on E2 features (~1-1.5 h)
# ============================================================
def lgb_cv(trial_params):
    aucs = []
    for tri, vai, X_tr, X_va in FOLD_DATA:
        m = lgb.LGBMClassifier(**trial_params, n_estimators=3000, random_state=42,
                               verbose=-1, n_jobs=8)
        m.fit(X_tr, y[tri], eval_set=[(X_va, y[vai])], eval_metric="auc",
              categorical_feature=[list(X_tr.columns).index(c) for c in
                                   ["NAME_CONTRACT_TYPE","CODE_GENDER","NAME_INCOME_TYPE","NAME_EDUCATION_TYPE",
                                    "NAME_FAMILY_STATUS","NAME_HOUSING_TYPE","OCCUPATION_TYPE","FLAG_WORK_PHONE"]
                                   if c in X_tr.columns],
              callbacks=[lgb.early_stopping(200, verbose=False)])
        aucs.append(roc_auc_score(y[vai], m.predict_proba(X_va)[:, 1]))
    return float(np.mean(aucs))

def lgb_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.08, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 16, 48),
        "max_depth": trial.suggest_categorical("max_depth", [3, 4]),
        "min_child_samples": trial.suggest_int("min_child_samples", 40, 200),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "subsample_freq": 1,
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 1.0, log=True),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 1e-3, 1.0, log=True),
    }

def dump(study, tag):
    with open(f"{T}\\best_params_{tag}_e2.json", "w") as fh:
        json.dump(study.best_params, fh, indent=2)
    with open(f"{T}\\best_val_{tag}_e2.txt", "w") as fh:
        fh.write(f"{study.best_value:.6f}")

t0 = time.time()
s_lgb = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
s_lgb.optimize(lambda tr: lgb_cv(lgb_space(tr)), n_trials=40, timeout=5400,
               callbacks=[lambda st, tr: dump(st, "lgb")])
print(f"LGB re-tune done: {s_lgb.best_value:.5f} (old 0.75989) ({time.time()-t0:.0f}s)", flush=True)
print("LGB best:", json.dumps(s_lgb.best_params), flush=True)

# ============================================================
# STAGE 3: XGB Optuna re-tune on E2 features (~3-4 h)
# ============================================================
xp_base = {"objective": "binary:logistic", "eval_metric": "auc",
           "tree_method": "hist", "n_jobs": 8, "verbosity": 0}

def xgb_cv(trial_params):
    aucs = []
    for tri, vai, X_tr, X_va in FOLD_DATA:
        dtr = xgb.DMatrix(X_tr, label=y[tri]); dva = xgb.DMatrix(X_va, label=y[vai])
        m = xgb.train({**xp_base, **trial_params}, dtr, num_boost_round=3000,
                      evals=[(dva, "val")], early_stopping_rounds=200, verbose_eval=False)
        aucs.append(roc_auc_score(y[vai], m.predict(dva, iteration_range=(0, m.best_iteration + 1))))
    return float(np.mean(aucs))

def xgb_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.06, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 5),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 0.9),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.4, 0.9),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 0.5, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 20.0, log=True),
        "gamma": trial.suggest_float("gamma", 0.0, 1.0),
    }

t0 = time.time()
s_xgb = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
s_xgb.optimize(lambda tr: xgb_cv(xgb_space(tr)), n_trials=26, timeout=14400,
               callbacks=[lambda st, tr: dump(st, "xgb")])
print(f"XGB re-tune done: {s_xgb.best_value:.5f} (old 0.76167) ({time.time()-t0:.0f}s)", flush=True)
print("XGB best:", json.dumps(s_xgb.best_params), flush=True)

print("OVERNIGHT RETUNE ALL DONE", flush=True)
