"""W6 (v2): CV-protocol variants -- memory-lean, crash-resilient.

Everything in v3/v4/v5 varied MODEL seeds on the SAME seed-42 folds. Two variants remain:
  V1 10-fold CV  -> each fold model trains on 90% instead of 80% (stronger test preds)
  V2 alternate fold seed (1337) -> different TE + different partitions = real diversity

Fixes vs v1 of this script:
  - float32 matrices (half the RAM of float64)
  - one model seed at a time, OOF saved incrementally after EVERY seed (a crash keeps work)
  - n_jobs=4 for LGB, and frames deleted before the next fold is built
  - test predictions accumulated in float32
"""
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
import common as C

OUT = C.OUT

# float32 working copies (TE cols are float32 too)
X32 = C.X.astype(np.float32)
XT32 = C.X_TEST.astype(np.float32)
print(f"X32 {X32.shape} {X32.memory_usage(deep=True).sum()/1e6:.0f}MB | "
      f"XT32 {XT32.shape} {XT32.memory_usage(deep=True).sum()/1e6:.0f}MB", flush=True)


def add_te3(X_tr, X_va, y_tr, X_te):
    X_tr, X_va, X_te = X_tr.copy(), X_va.copy(), X_te.copy()
    gs = pd.Series(y_tr, index=X_tr.index)
    gm = gs.mean()
    for col in C.TE_COLS:
        if col not in X_tr.columns:
            continue
        means = gs.groupby(X_tr[col]).mean()
        counts = X_tr[col].value_counts()
        sm = (means * counts + gm * C.SMOOTHING) / (counts + C.SMOOTHING)
        X_tr[f"{col}_TE"] = X_tr[col].map(sm).fillna(gm).astype(np.float32)
        X_va[f"{col}_TE"] = X_va[col].map(sm).fillna(gm).astype(np.float32)
        X_te[f"{col}_TE"] = X_te[col].map(sm).fillna(gm).astype(np.float32)
    return X_tr, X_va, X_te


def fit_predict(family, params, X_tr, y_tr, X_va, y_va, X_te, mseed):
    if family == "lgb":
        import lightgbm as lgb
        m = lgb.LGBMClassifier(**params, n_estimators=3000, random_state=mseed,
                               verbose=-1, n_jobs=4)
        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="auc",
              categorical_feature=C.ci_of(X_tr),
              callbacks=[lgb.early_stopping(200, verbose=False)])
        return m.predict_proba(X_va)[:, 1], m.predict_proba(X_te)[:, 1]
    if family == "xgb":
        import xgboost as xgb
        p = dict(params)
        p.update({"objective": "binary:logistic", "eval_metric": "auc",
                  "tree_method": "hist", "n_jobs": 4, "verbosity": 0, "random_state": mseed})
        dtr = xgb.DMatrix(X_tr, label=y_tr)
        dva = xgb.DMatrix(X_va, label=y_va)
        dte = xgb.DMatrix(X_te)
        m = xgb.train(p, dtr, num_boost_round=3000, evals=[(dva, "val")],
                      early_stopping_rounds=200, verbose_eval=False)
        return (m.predict(dva, iteration_range=(0, m.best_iteration + 1)),
                m.predict(dte, iteration_range=(0, m.best_iteration + 1)))
    if family == "cat":
        import catboost as cb
        cp = dict(params)
        cp.update({"loss_function": "Logloss", "eval_metric": "AUC", "random_seed": mseed,
                   "verbose": False, "allow_writing_files": False, "thread_count": 4})
        m = cb.CatBoostClassifier(**cp, iterations=3000)
        m.fit(X_tr, y_tr, eval_set=(X_va, y_va), cat_features=C.ci_of(X_tr),
              early_stopping_rounds=200)
        return m.predict_proba(X_va)[:, 1], m.predict_proba(X_te)[:, 1]
    raise ValueError(family)


def run(tag, family, nsp, fs, model_seeds, params):
    t0 = time.time()
    skf = StratifiedKFold(n_splits=nsp, shuffle=True, random_state=fs)
    folds = list(skf.split(X32, C.y))
    oof = np.zeros(len(X32), dtype=np.float32)
    test = np.zeros(len(XT32), dtype=np.float32)
    nseed = len(model_seeds)
    for mseed in model_seeds:
        for fi, (tri, vai) in enumerate(folds):
            X_tr, X_va, X_te = add_te3(X32.iloc[tri], X32.iloc[vai], C.y[tri], XT32)
            p_va, p_te = fit_predict(family, params, X_tr, C.y[tri], X_va, C.y[vai], X_te, mseed)
            oof[vai] = p_va
            test += p_te.astype(np.float32) / (len(folds) * nseed)
            print(f"  {tag} mseed{mseed} fold{fi+1}/{len(folds)}: "
                  f"{roc_auc_score(C.y[vai], p_va):.5f} ({time.time()-t0:.0f}s)", flush=True)
            del X_tr, X_va, X_te, p_va, p_te
            gc.collect()
        # incremental save after each model seed
        np.save(OUT / f"w6_{tag}_oof.npy", oof)
        np.save(OUT / f"w6_{tag}_test.npy", test)
        a = roc_auc_score(C.y, oof)
        print(f"  {tag} partial ({model_seeds.index(mseed)+1}/{nseed} seeds) OOF so far: {a:.6f} "
              f"({time.time()-t0:.0f}s)", flush=True)
    a = roc_auc_score(C.y, oof)
    json.dump({"family": family, "n_splits": nsp, "fold_seed": fs,
               "model_seeds": list(model_seeds), "oof_auc": float(a), "params": params},
              open(OUT / f"w6_{tag}.json", "w"), indent=2, default=float)
    print(f"W6 {tag} FINAL OOF AUC: {a:.6f} ({time.time()-t0:.0f}s)", flush=True)
    return oof, test


PLAN = [
    ("lgb10", "lgb", 10, 42, (42, 1337, 2026), "w2_best_lgb.json"),
    ("xgb10", "xgb", 10, 42, (42, 1337, 2026), "w2_best_xgb.json"),
    ("xgb_f1337", "xgb", 5, 1337, (42, 1337, 2026), "w2_best_xgb.json"),
    ("lgb_f1337", "lgb", 5, 1337, (42, 1337, 2026), "w2_best_lgb.json"),
    ("cat10", "cat", 10, 42, (42, 1337), None),
]

t_all = time.time()
for tag, family, nsp, fs, ms, pfile in PLAN:
    meta_path = OUT / f"w6_{tag}.json"
    if meta_path.exists():
        print(f"skip {tag} (done)", flush=True)
        continue
    params = json.load(open(OUT / pfile)) if pfile else (
        C.CAT_PARAMS if family == "cat" else None)
    print(f"=== W6 {tag}: {family} {nsp}-fold foldseed={fs} model_seeds={ms} ===", flush=True)
    try:
        run(tag, family, nsp, fs, ms, params)
    except Exception as e:
        print(f"W6 {tag} FAILED: {type(e).__name__}: {e}", flush=True)
        gc.collect()
    gc.collect()

print(f"W6 ALL DONE ({time.time()-t_all:.0f}s)", flush=True)
