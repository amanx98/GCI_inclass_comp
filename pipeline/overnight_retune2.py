import sys, time, json
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import pandas as pd
import catboost as cb
from sklearn.ensemble import HistGradientBoostingClassifier
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
CATS = ["NAME_CONTRACT_TYPE","CODE_GENDER","NAME_INCOME_TYPE","NAME_EDUCATION_TYPE",
        "NAME_FAMILY_STATUS","NAME_HOUSING_TYPE","OCCUPATION_TYPE","FLAG_WORK_PHONE"]

fold_data = []
for tri, vai in FOLDS:
    X_tr, X_va = add_te(X.iloc[tri], X.iloc[vai], y[tri])
    fold_data.append((tri, vai, X_tr, X_va))
print("folds ready", flush=True)
def ci_of(Xc):
    return [list(Xc.columns).index(c) for c in CATS if c in Xc.columns]

# ============================================================
# STAGE A: CAT search on fold-0 only (fast proxy), 20 trials
# ============================================================
tri0, vai0, Xtr0, Xva0 = fold_data[0]
ci0 = ci_of(Xtr0)

def cat_fold0(params, rounds=2000):
    cp0 = dict(params); cp0.update({"loss_function": "Logloss", "eval_metric": "AUC",
        "random_seed": 42, "verbose": False, "allow_writing_files": False, "thread_count": 8})
    m = cb.CatBoostClassifier(**cp0, iterations=rounds)
    m.fit(Xtr0, y[tri0], eval_set=(Xva0, y[vai0]), cat_features=ci0, early_stopping_rounds=200)
    return roc_auc_score(y[vai0], m.predict_proba(Xva0)[:, 1])

def cat_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.08, log=True),
        "depth": trial.suggest_int("depth", 3, 5),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "random_strength": trial.suggest_float("random_strength", 1e-3, 5.0, log=True),
        "border_count": trial.suggest_int("border_count", 64, 254),
    }

t0 = time.time()
s_cat = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
s_cat.optimize(lambda tr: cat_fold0(cat_space(tr)), n_trials=20, timeout=7200)
print(f"CAT fold-0 search done: best {s_cat.best_value:.5f} (baseline fold-0 ~see next) ({time.time()-t0:.0f}s)", flush=True)
print("CAT search best:", json.dumps(s_cat.best_params), flush=True)

# baseline fold-0 for reference
base_cat = json.load(open(f"{BASE}\\optuna_studies\\best_params_cat.json"))
b0 = cat_fold0(base_cat)
print(f"CAT baseline params on fold-0: {b0:.5f}", flush=True)

# top-3 configs -> full 5-fold CPU validation (AUC early stopping, the proper protocol)
top = [s_cat.best_params] + [t.params for t in s_cat.trials
        if t.value is not None and t.params != s_cat.best_params][:2]
def cat_cv_full(params):
    aucs = []
    for tri, vai, X_tr, X_va in fold_data:
        cp0 = dict(params); cp0.update({"loss_function": "Logloss", "eval_metric": "AUC",
            "random_seed": 42, "verbose": False, "allow_writing_files": False, "thread_count": 8})
        m = cb.CatBoostClassifier(**cp0, iterations=3000)
        m.fit(X_tr, y[tri], eval_set=(X_va, y[vai]), cat_features=ci_of(X_tr), early_stopping_rounds=200)
        aucs.append(roc_auc_score(y[vai], m.predict_proba(X_va)[:, 1]))
    return float(np.mean(aucs))

results = {}
for i, params in enumerate(top):
    a = cat_cv_full(params)
    results[json.dumps(params)] = a
    print(f"CAT full-CV top{i+1}: {a:.5f} ({time.time()-t0:.0f}s)", flush=True)
best_cat_params = max(results, key=results.get)
best_cat_val = results[best_cat_params]
print(f"CAT BASELINE full-CV (old params): see prior runs = 0.76033 single-seed", flush=True)
if best_cat_val > 0.7607:  # adopt only if clearly better than 0.76033 baseline
    json.dump(json.loads(best_cat_params), open(f"{T}\\best_params_cat_e2.json", "w"), indent=2)
    open(f"{T}\\best_val_cat_e2.txt", "w").write(f"{best_cat_val:.6f}")
    print(f"CAT NEW PARAMS ADOPTED: {best_cat_val:.5f} > 0.76033", flush=True)
else:
    print(f"CAT no adoption (best {best_cat_val:.5f} <= baseline)", flush=True)

# ============================================================
# STAGE B: HGB full tune (40 trials, full 5-fold, early stopping)
# ============================================================
def hgb_cv(params):
    aucs = []
    for tri, vai, X_tr, X_va in fold_data:
        m = HistGradientBoostingClassifier(**params, random_state=42)
        m.fit(X_tr, y[tri])
        aucs.append(roc_auc_score(y[vai], m.predict_proba(X_va)[:, 1]))
    return float(np.mean(aucs))

def hgb_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.12, log=True),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 63),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 20, 120),
        "l2_regularization": trial.suggest_float("l2_regularization", 1e-2, 10.0, log=True),
        "max_iter": 1000,
        "early_stopping": True,
        "validation_fraction": 0.1,
        "n_iter_no_change": 25,
    }

t0 = time.time()
s_hgb = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
def dump(st, tag):
    json.dump(st.best_params, open(f"{T}\\best_params_{tag}_e2.json", "w"), indent=2)
    open(f"{T}\\best_val_{tag}_e2.txt", "w").write(f"{st.best_value:.6f}")
s_hgb.optimize(lambda tr: hgb_cv(hgb_space(tr)), n_trials=40, timeout=5400,
               callbacks=[lambda st, tr: dump(st, "hgb")])
print(f"HGB tune done: {s_hgb.best_value:.5f} (old default-ish 0.75588) ({time.time()-t0:.0f}s)", flush=True)
print("HGB best:", json.dumps(s_hgb.best_params), flush=True)
print("OVERNIGHT RETUNE-2 ALL DONE", flush=True)
