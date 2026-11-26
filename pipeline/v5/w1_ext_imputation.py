"""W1: model-based imputation of EXT_SOURCE_1 / EXT_SOURCE_3 / OWN_CAR_AGE.

For each target col T (stored as -999 when missing in E2):
  - fit an LGB regressor fold-locally on rows where T is observed,
    using all other features (features-only target, no leakage of y into features),
  - predict T for rows where T == -999 (train + test),
  - add T_IMP (imputed value, -999 rows get prediction; observed rows keep original)
    and T_IMP_WAS_MISSING * T_IMP interaction.
Then A/B vs the v4 baseline matrix with LGB 5-fold OOF (3 seeds, shipped params).

Keep variant only if OOF delta vs v4 LGB member >= +0.0003.
"""
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import common as C

TARGETS = ["EXT_SOURCE_1", "EXT_SOURCE_3", "OWN_CAR_AGE"]
MISS = -999

t0 = time.time()
print("W1 start", flush=True)

# Build augmented train/test frames
X_aug = C.X.copy()
Xt_aug = C.X_TEST.copy()

for tcol in TARGETS:
    print(f"--- imputing {tcol} ---", flush=True)
    tr_obs = C.X[tcol] != MISS
    te_obs = C.X_TEST[tcol] != MISS
    feat_cols = [c for c in C.X.columns if c != tcol]
    # regressor params: modest, fixed seed, fold-local
    rparams = dict(C.LGB_PARAMS)
    rparams["learning_rate"] = 0.05
    rparams["num_leaves"] = 31
    rparams["max_depth"] = 6
    rparams["min_child_samples"] = 100
    preds_tr = C.X[tcol].astype(float).values.copy()
    preds_te = C.X_TEST[tcol].astype(float).values.copy()
    for fold, (tri, vai) in enumerate(C.FOLDS):
        # fit on observed rows of the TRAIN part only
        fit_idx = tri[C.X.iloc[tri][tcol] != MISS]
        Xf = C.X.iloc[fit_idx][feat_cols]
        yf = C.X.iloc[fit_idx][tcol].astype(float)
        # validation = observed rows in val part (for early stopping)
        val_idx = vai[C.X.iloc[vai][tcol] != MISS]
        Xv = C.X.iloc[val_idx][feat_cols]
        yv = C.X.iloc[val_idx][tcol].astype(float)
        m = lgb.LGBMRegressor(**rparams, n_estimators=2000, random_state=42, verbose=-1, n_jobs=-1)
        m.fit(Xf, yf, eval_set=[(Xv, yv)], eval_metric="rmse",
              categorical_feature=C.ci_of(Xf),
              callbacks=[lgb.early_stopping(100, verbose=False)])
        # predict missing rows in train (val part covers all train via CV) and test
        miss_tr = vai[C.X.iloc[vai][tcol] == MISS]
        if len(miss_tr):
            preds_tr[miss_tr] = m.predict(C.X.iloc[miss_tr][feat_cols])
        miss_te = np.where(C.X_TEST[tcol] == MISS)[0]
        if len(miss_te):
            preds_te[miss_te] = m.predict(C.X_TEST.iloc[miss_te][feat_cols])
        # also fill train fold-train missing (they were in fit? no - only observed used; predict for them too)
        miss_fit = tri[C.X.iloc[tri][tcol] == MISS]
        if len(miss_fit):
            preds_tr[miss_fit] = m.predict(C.X.iloc[miss_fit][feat_cols])
        print(f"  fold {fold+1}: fitted on {len(fit_idx)}, best_iter={m.best_iteration_} ({time.time()-t0:.0f}s)", flush=True)
    was_miss_tr = (C.X[tcol] == MISS).astype(float)
    was_miss_te = (C.X_TEST[tcol] == MISS).astype(float)
    X_aug[f"{tcol}_IMP"] = np.where(was_miss_tr == 1, preds_tr, C.X[tcol])
    Xt_aug[f"{tcol}_IMP"] = np.where(was_miss_te == 1, preds_te, C.X_TEST[tcol])
    X_aug[f"{tcol}_IMP_X_WM"] = X_aug[f"{tcol}_IMP"] * was_miss_tr
    Xt_aug[f"{tcol}_IMP_X_WM"] = Xt_aug[f"{tcol}_IMP"] * was_miss_te
    print(f"  {tcol}: {int(was_miss_tr.sum())} train / {int(was_miss_te.sum())} test imputed", flush=True)

X_aug.to_parquet(C.OUT / "X_e2_w1.parquet")
Xt_aug.to_parquet(C.OUT / "X_test_e2_w1.parquet")
print(f"augmented frames saved ({time.time()-t0:.0f}s)", flush=True)

# ---- A/B: LGB 5-fold OOF on augmented vs baseline ----
def lgb_oof_on(Xm, seeds=(42, 1337, 2026)):
    oof = np.zeros(len(Xm))
    for seed in seeds:
        oof_s = np.zeros(len(Xm))
        for tri, vai in C.FOLDS:
            X_tr, X_va = C.add_te(Xm.iloc[tri], Xm.iloc[vai], C.y[tri])
            m = lgb.LGBMClassifier(**C.LGB_PARAMS, n_estimators=3000, random_state=seed,
                                   verbose=-1, n_jobs=-1)
            m.fit(X_tr, C.y[tri], eval_set=[(X_va, C.y[vai])], eval_metric="auc",
                  categorical_feature=C.ci_of(X_tr),
                  callbacks=[lgb.early_stopping(200, verbose=False)])
            oof_s[vai] = m.predict_proba(X_va)[:, 1]
        print(f"  seed {seed}: {roc_auc_score(C.y, oof_s):.5f}", flush=True)
        oof += oof_s / len(seeds)
    return oof

print("=== baseline LGB (E2, 3 seeds) ===", flush=True)
t1 = time.time()
oof_base = lgb_oof_on(C.X)
print(f"BASELINE LGB OOF: {roc_auc_score(C.y, oof_base):.6f} ({time.time()-t1:.0f}s)", flush=True)

print("=== W1 augmented LGB (3 seeds) ===", flush=True)
t1 = time.time()
oof_w1 = lgb_oof_on(X_aug)
print(f"W1 LGB OOF: {roc_auc_score(C.y, oof_w1):.6f} ({time.time()-t1:.0f}s)", flush=True)

delta = roc_auc_score(C.y, oof_w1) - roc_auc_score(C.y, oof_base)
print(f"\nW1 DELTA vs baseline: {delta:+.6f}  (keep threshold +0.0003)", flush=True)
np.save(C.OUT / "w1_lgb_oof.npy", oof_w1)
np.save(C.OUT / "w1_base_lgb_oof.npy", oof_base)
print(f"W1 DONE ({time.time()-t0:.0f}s)", flush=True)
