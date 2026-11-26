"""W2: full-CV Optuna re-tune for LGB / XGB / HGB (CAT optional via --family cat).

Objective = mean 5-fold OOF AUC with the EXACT v4 protocol (seed-42 folds,
fold-local TE smoothing=10). TPESampler(seed=42). Each trial runs all 5 folds
with early stopping, exactly like the shipped members.

Usage: python w2_tune.py --family lgb --trials 120
       python w2_tune.py --family cat --trials 60
Results appended to pipeline/v5/out/w2_results.jsonl; best params saved per family.
"""
import argparse
import json
import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")  # HGB binning ThreadPool needs named pipes (sandbox-denied)

import numpy as np
import optuna
from sklearn.metrics import roc_auc_score

import common as C

optuna.logging.set_verbosity(optuna.logging.WARNING)

ap = argparse.ArgumentParser()
ap.add_argument("--family", required=True, choices=["lgb", "xgb", "hgb", "cat"])
ap.add_argument("--trials", type=int, default=100)
ap.add_argument("--timeout", type=int, default=20000)
ap.add_argument("--seed-avg", type=int, default=1, help="seeds per trial (1 keeps it fast)")
args = ap.parse_args()

t0 = time.time()
folds = C.fold_frames(X_te=None)  # [(tri, vai, X_tr, X_va), ...]
print(f"W2 {args.family}: {args.trials} trials, folds ready ({time.time()-t0:.0f}s)", flush=True)

SEEDS = [42, 1337, 2026][: args.seed_avg]


def eval_params(params):
    """Mean 5-fold OOF AUC for a candidate param dict (family-specific fit)."""
    aucs = []
    if args.family == "lgb":
        import lightgbm as lgb
        for tri, vai, X_tr, X_va in folds:
            m = lgb.LGBMClassifier(**params, n_estimators=3000, random_state=42,
                                   verbose=-1, n_jobs=-1)
            m.fit(X_tr, C.y[tri], eval_set=[(X_va, C.y[vai])], eval_metric="auc",
                  categorical_feature=C.ci_of(X_tr),
                  callbacks=[lgb.early_stopping(200, verbose=False)])
            aucs.append(roc_auc_score(C.y[vai], m.predict_proba(X_va)[:, 1]))
    elif args.family == "xgb":
        import xgboost as xgb
        p = dict(params)
        p.update({"objective": "binary:logistic", "eval_metric": "auc",
                  "tree_method": "hist", "n_jobs": 8, "verbosity": 0, "random_state": 42})
        for tri, vai, X_tr, X_va in folds:
            dtr = xgb.DMatrix(X_tr, label=C.y[tri])
            dva = xgb.DMatrix(X_va, label=C.y[vai])
            m = xgb.train(p, dtr, num_boost_round=3000, evals=[(dva, "val")],
                          early_stopping_rounds=200, verbose_eval=False)
            aucs.append(roc_auc_score(C.y[vai], m.predict(dva, iteration_range=(0, m.best_iteration + 1))))
    elif args.family == "hgb":
        from sklearn.ensemble import HistGradientBoostingClassifier
        for tri, vai, X_tr, X_va in folds:
            m = HistGradientBoostingClassifier(**params, random_state=42)
            m.fit(X_tr, C.y[tri])
            aucs.append(roc_auc_score(C.y[vai], m.predict_proba(X_va)[:, 1]))
    elif args.family == "cat":
        import catboost as cb
        for tri, vai, X_tr, X_va in folds:
            cp = dict(params)
            cp.update({"loss_function": "Logloss", "eval_metric": "AUC", "random_seed": 42,
                       "verbose": False, "allow_writing_files": False, "thread_count": 8})
            m = cb.CatBoostClassifier(**cp, iterations=3000)
            m.fit(X_tr, C.y[tri], eval_set=(X_va, C.y[vai]), cat_features=C.ci_of(X_tr),
                  early_stopping_rounds=200)
            aucs.append(roc_auc_score(C.y[vai], m.predict_proba(X_va)[:, 1]))
    return float(np.mean(aucs))


def space(trial):
    if args.family == "lgb":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.06, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "max_depth": trial.suggest_int("max_depth", 3, 6),
            "min_child_samples": trial.suggest_int("min_child_samples", 40, 300),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 1e-3, 1.0, log=True),
        }
    if args.family == "xgb":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 6),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 40.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.4, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 30.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 1.5),
        }
    if args.family == "hgb":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.12, log=True),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 63),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 20, 150),
            "l2_regularization": trial.suggest_float("l2_regularization", 1e-2, 10.0, log=True),
            "max_iter": 1000,
            "early_stopping": True,
            "validation_fraction": 0.1,
            "n_iter_no_change": 25,
        }
    # cat
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.12, log=True),
        "depth": trial.suggest_int("depth", 3, 6),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "random_strength": trial.suggest_float("random_strength", 1e-3, 5.0, log=True),
        "border_count": trial.suggest_int("border_count", 64, 254),
    }


# baseline for reference
base = eval_params({**getattr(C, f"{args.family.upper()}_PARAMS")}) if args.family != "cat" else eval_params(C.CAT_PARAMS)
print(f"baseline {args.family} full-CV mean: {base:.6f}", flush=True)

study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))


def on_trial(st, tr):
    line = {"family": args.family, "trial": tr.number, "value": tr.value,
            "params": tr.params, "elapsed": time.time() - t0}
    with open(C.OUT / "w2_results.jsonl", "a") as fh:
        fh.write(json.dumps(line) + "\n")
    print(f"  trial {tr.number}: {tr.value:.6f} ({time.time()-t0:.0f}s)", flush=True)
    if st.best_value == tr.value:
        json.dump(tr.params, open(C.OUT / f"w2_best_{args.family}.json", "w"), indent=2)


study.optimize(lambda tr: eval_params(space(tr)), n_trials=args.trials,
               timeout=args.timeout, callbacks=[on_trial])

print(f"W2 {args.family} DONE: best {study.best_value:.6f} vs baseline {base:.6f} "
      f"delta {study.best_value - base:+.6f} ({time.time()-t0:.0f}s)", flush=True)
print("best params:", json.dumps(study.best_params), flush=True)
json.dump(study.best_params, open(C.OUT / f"w2_best_{args.family}.json", "w"), indent=2)
