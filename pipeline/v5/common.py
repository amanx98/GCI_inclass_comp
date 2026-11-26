"""Shared v5 harness.

Loads the verified E2 matrices from pipeline/artifacts and rebuilds the EXACT
v3/v4 protocol: StratifiedKFold(5, shuffle, seed=42) + fold-local smoothed
target encoding (smoothing=10) on the high-card columns present in X_e2.

Every v5 experiment imports from here so all numbers are directly comparable
to v4 OOF 0.762442.  CPU-only.  Fixed seeds.
"""
import json
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")  # sklearn HGB binning ThreadPool uses named pipes (sandbox-denied)

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
ART = HERE.parent / "artifacts"
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)

SEED = 42
N_SPLITS = 5
SMOOTHING = 10

X = pd.read_parquet(ART / "X_e2.parquet")           # (171202, 98)
X_TEST = pd.read_parquet(ART / "X_test_e2.parquet")  # (61500, 98)
y = np.load(ART / "y_train.npy")
HIGH_CARD = json.load(open(ART / "high_card_cols.json"))
TE_COLS = [c for c in HIGH_CARD if c in X.columns]

LGB_PARAMS = json.load(open(ART / "best_params_lgb_e2.json"))
XGB_PARAMS = json.load(open(ART / "best_params_xgb_e2.json"))
CAT_PARAMS = json.load(open(ART / "best_params_cat_e2.json"))
HGB_PARAMS = json.load(open(ART / "best_params_hgb_e2.json"))
HGB_PARAMS = {**HGB_PARAMS, "max_iter": 1000, "early_stopping": True,
              "validation_fraction": 0.1, "n_iter_no_change": 25}

CATS = ["NAME_CONTRACT_TYPE", "CODE_GENDER", "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE",
        "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE", "OCCUPATION_TYPE", "FLAG_WORK_PHONE"]

SKF = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
FOLDS = list(SKF.split(X, y))


def ci_of(Xc):
    return [list(Xc.columns).index(c) for c in CATS if c in Xc.columns]


def add_te(X_tr, X_va, y_tr, X_te=None, smoothing=SMOOTHING, te_cols=None):
    """Fold-local smoothed target encoding. Means come from fold-train ONLY."""
    te_cols = TE_COLS if te_cols is None else te_cols
    X_tr, X_va = X_tr.copy(), X_va.copy()
    X_te = X_te.copy() if X_te is not None else None
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
        if X_te is not None:
            X_te[f"{col}_TE"] = X_te[col].map(sm).fillna(gm)
    return (X_tr, X_va, X_te) if X_te is not None else (X_tr, X_va)


def fold_frames(X_te=X_TEST):
    """Per-fold TE'd frames incl. test: [(tri, vai, X_tr, X_va, X_te), ...]."""
    out = []
    for tri, vai in FOLDS:
        if X_te is not None:
            X_tr, X_va, X_te_f = add_te(X.iloc[tri], X.iloc[vai], y[tri], X_te.copy())
            out.append((tri, vai, X_tr, X_va, X_te_f))
        else:
            X_tr, X_va = add_te(X.iloc[tri], X.iloc[vai], y[tri])
            out.append((tri, vai, X_tr, X_va))
    return out


def auc(oof):
    return float(roc_auc_score(y, oof))


# ---- verified v4 members (for blending / gate) ----
V4_OOF = 0.30 * np.load(ART / "m3_LGB3.npy") \
       + 0.33 * np.load(ART / "m3_XGB3.npy") \
       + 0.33 * (np.load(ART / "v4_CATnew_s42_oof.npy") + np.load(ART / "v4_CATnew_s1337_oof.npy")) / 2 \
       + 0.04 * np.load(ART / "v4_HGBnew_oof.npy")


def bootstrap_gate(new_oof, base_oof=None, n=500, seed=0):
    """Paired bootstrap of (new - base) OOF AUC. Gate: 95% CI entirely > 0."""
    base_oof = V4_OOF if base_oof is None else base_oof
    rng = np.random.default_rng(seed)
    d = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, len(y), len(y))
        d[i] = roc_auc_score(y[idx], new_oof[idx]) - roc_auc_score(y[idx], base_oof[idx])
    return {"mean": float(d.mean()), "ci_lo": float(np.percentile(d, 2.5)),
            "ci_hi": float(np.percentile(d, 97.5)), "p_gt0": float((d > 0).mean()),
            "passes": bool(np.percentile(d, 2.5) > 0)}


def lgb_oof_test(folds, seeds=(42, 1337, 2026), params=None):
    """LGB OOF (+ test if fold frames have 5 entries). Returns (oof, test|None)."""
    import lightgbm as lgb
    oof = np.zeros(len(X))
    have_te = len(folds[0]) > 4
    test = np.zeros(len(X_TEST)) if have_te else None
    for seed in seeds:
        oof_s = np.zeros(len(X))
        for fr in folds:
            tri, vai, X_tr, X_va = fr[0], fr[1], fr[2], fr[3]
            m = lgb.LGBMClassifier(**(params or LGB_PARAMS), n_estimators=3000,
                                   random_state=seed, verbose=-1, n_jobs=-1)
            m.fit(X_tr, y[tri], eval_set=[(X_va, y[vai])], eval_metric="auc",
                  categorical_feature=ci_of(X_tr),
                  callbacks=[lgb.early_stopping(200, verbose=False)])
            oof_s[vai] = m.predict_proba(X_va)[:, 1]
            if have_te:
                test += m.predict_proba(fr[4])[:, 1] / (5 * len(seeds))
        print(f"  LGB seed {seed}: {auc(oof_s):.5f}", flush=True)
        oof += oof_s / len(seeds)
    return oof, test


def xgb_oof_test(folds, seeds=(42, 1337, 2026), params=None, n_jobs=8):
    import xgboost as xgb
    p = dict(params or XGB_PARAMS)
    p.update({"objective": "binary:logistic", "eval_metric": "auc",
              "tree_method": "hist", "n_jobs": n_jobs, "verbosity": 0})
    oof = np.zeros(len(X))
    have_te = len(folds[0]) > 4
    test = np.zeros(len(X_TEST)) if have_te else None
    for seed in seeds:
        oof_s = np.zeros(len(X))
        for fr in folds:
            tri, vai, X_tr, X_va = fr[0], fr[1], fr[2], fr[3]
            dtr = xgb.DMatrix(X_tr, label=y[tri])
            dva = xgb.DMatrix(X_va, label=y[vai])
            m = xgb.train({**p, "random_state": seed}, dtr, num_boost_round=3000,
                          evals=[(dva, "val")], early_stopping_rounds=200, verbose_eval=False)
            oof_s[vai] = m.predict(dva, iteration_range=(0, m.best_iteration + 1))
            if have_te:
                test += m.predict(xgb.DMatrix(fr[4]),
                                  iteration_range=(0, m.best_iteration + 1)) / (5 * len(seeds))
        print(f"  XGB seed {seed}: {auc(oof_s):.5f}", flush=True)
        oof += oof_s / len(seeds)
    return oof, test


def cat_oof_test(folds, seeds=(42, 1337), params=None, thread_count=8):
    import catboost as cb
    p = dict(params or CAT_PARAMS)
    oof = np.zeros(len(X))
    have_te = len(folds[0]) > 4
    test = np.zeros(len(X_TEST)) if have_te else None
    for seed in seeds:
        oof_s = np.zeros(len(X))
        for fr in folds:
            tri, vai, X_tr, X_va = fr[0], fr[1], fr[2], fr[3]
            cp = dict(p)
            cp.update({"loss_function": "Logloss", "eval_metric": "AUC", "random_seed": seed,
                       "verbose": False, "allow_writing_files": False,
                       "thread_count": thread_count})
            m = cb.CatBoostClassifier(**cp, iterations=3000)
            m.fit(X_tr, y[tri], eval_set=(X_va, y[vai]), cat_features=ci_of(X_tr),
                  early_stopping_rounds=200)
            oof_s[vai] = m.predict_proba(X_va)[:, 1]
            if have_te:
                test += m.predict_proba(fr[4])[:, 1] / (5 * len(seeds))
        print(f"  CAT seed {seed}: {auc(oof_s):.5f}", flush=True)
        oof += oof_s / len(seeds)
    return oof, test


def hgb_oof_test(folds, seeds=(42,), params=None):
    p = dict(params or HGB_PARAMS)
    oof = np.zeros(len(X))
    have_te = len(folds[0]) > 4
    test = np.zeros(len(X_TEST)) if have_te else None
    for seed in seeds:
        oof_s = np.zeros(len(X))
        for fr in folds:
            tri, vai, X_tr, X_va = fr[0], fr[1], fr[2], fr[3]
            m = HistGradientBoostingClassifier(**p, random_state=seed)
            m.fit(X_tr, y[tri])
            oof_s[vai] = m.predict_proba(X_va)[:, 1]
            if have_te:
                test += m.predict_proba(fr[4])[:, 1] / (5 * len(seeds))
        print(f"  HGB seed {seed}: {auc(oof_s):.5f}", flush=True)
        oof += oof_s / len(seeds)
    return oof, test


def log(msg, t0=None):
    el = f" ({time.time()-t0:.0f}s)" if t0 is not None else ""
    print(f"{msg}{el}", flush=True)
