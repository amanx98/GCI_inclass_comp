"""Sanity check: reproduce v4 OOF 0.762442 from artifacts, verify folds/TE,
and time one LGB fold so we can size the W2 budget."""
import time
import numpy as np
from sklearn.metrics import roc_auc_score
import common as C

t0 = time.time()
print("v4 OOF reproduced:", f"{roc_auc_score(C.y, C.V4_OOF):.6f}", "(expect 0.762442)")
print("X:", C.X.shape, "X_TEST:", C.X_TEST.shape, "y:", C.y.shape, "TE cols:", C.TE_COLS)

# time one LGB fold with the shipped params
folds = C.fold_frames(X_te=None)
tri, vai, X_tr, X_va = folds[0][0], folds[0][1], folds[0][2], folds[0][3]
print("fold frames cols:", X_tr.shape[1], "(98 base + 6 TE = 104 expected)")
t1 = time.time()
import lightgbm as lgb
m = lgb.LGBMClassifier(**C.LGB_PARAMS, n_estimators=3000, random_state=42, verbose=-1, n_jobs=-1)
m.fit(X_tr, C.y[tri], eval_set=[(X_va, C.y[vai])], eval_metric="auc",
      categorical_feature=C.ci_of(X_tr), callbacks=[lgb.early_stopping(200, verbose=False)])
a = roc_auc_score(C.y[vai], m.predict_proba(X_va)[:, 1])
print(f"LGB fold0: {a:.5f} best_iter={m.best_iteration_} ({time.time()-t1:.0f}s)")
print(f"sanity total {time.time()-t0:.0f}s")
