"""W4: feature-engineering round 3, <=4 batches. Each batch is A/B tested with
LGB 5-fold OOF (3 seeds, shipped params) on the augmented matrix; kept only if
delta vs baseline >= +0.0003.

Batches (all features-only / transductive, no labels outside fold-train):
  B1 KMeans cluster IDs (k=8, k=16) fit on train+test numeric features (disclosed)
  B2 imputation interactions: EXT_MEAN_X_AGE * NAN_COUNT etc. (uses W1 IMP cols if present)
  B3 cat pair count-joins: value_counts of (NAME_INCOME_TYPE, OCCUPATION_TYPE) etc. fit on train+test
  B4 REGION_POPULATION_RELATIVE * EXT_SOURCES_MEAN
"""
import argparse
import time

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

import common as C

ap = argparse.ArgumentParser()
ap.add_argument("--batch", required=True, choices=["b1", "b2", "b3", "b4"])
ap.add_argument("--w1", action="store_true", help="load W1-augmented matrices if present")
args = ap.parse_args()

t0 = time.time()
Xw = pd.read_parquet(C.OUT / "X_e2_w1.parquet") if args.w1 else C.X
Xtw = pd.read_parquet(C.OUT / "X_test_e2_w1.parquet") if args.w1 else C.X_TEST
print(f"W4 {args.batch} w1={args.w1} base={Xw.shape}", flush=True)

NUM = Xw.select_dtypes(include=[np.number]).columns.tolist()


def build(df_tr, df_te):
    out_tr, out_te = df_tr.copy(), df_te.copy()
    if args.batch == "b1":
        for k in (8, 16):
            km = KMeans(n_clusters=k, random_state=42, n_init=5)
            km.fit(pd.concat([df_tr[NUM], df_te[NUM]], ignore_index=True))
            out_tr[f"KM{k}"] = km.predict(df_tr[NUM])
            out_te[f"KM{k}"] = km.predict(df_te[NUM])
    elif args.batch == "b2":
        if "EXT_SOURCE_1_IMP" in df_tr.columns:
            em = df_tr[["EXT_SOURCE_1_IMP", "EXT_SOURCE_2", "EXT_SOURCE_3_IMP"]].replace(-999, np.nan).mean(axis=1)
            em_te = df_te[["EXT_SOURCE_1_IMP", "EXT_SOURCE_2", "EXT_SOURCE_3_IMP"]].replace(-999, np.nan).mean(axis=1)
            out_tr["EXT_MEAN_IMP_X_NANCOUNT"] = em.fillna(em.mean()) * df_tr["NAN_COUNT"]
            out_te["EXT_MEAN_IMP_X_NANCOUNT"] = em_te.fillna(em_te.mean()) * df_te["NAN_COUNT"]
            out_tr["EXT_IMP_SUM"] = (df_tr[["EXT_SOURCE_1_IMP", "EXT_SOURCE_2", "EXT_SOURCE_3_IMP"]]
                                     .replace(-999, np.nan).sum(axis=1)).fillna(0)
            out_te["EXT_IMP_SUM"] = (df_te[["EXT_SOURCE_1_IMP", "EXT_SOURCE_2", "EXT_SOURCE_3_IMP"]]
                                      .replace(-999, np.nan).sum(axis=1)).fillna(0)
    elif args.batch == "b3":
        pairs = [("NAME_INCOME_TYPE", "OCCUPATION_TYPE"),
                 ("NAME_EDUCATION_TYPE", "OCCUPATION_TYPE"),
                 ("NAME_HOUSING_TYPE", "NAME_FAMILY_STATUS")]
        all_df = pd.concat([df_tr, df_te], ignore_index=True)
        for a, b in pairs:
            vc = all_df.groupby([a, b]).size()
            out_tr[f"{a}__{b}_CNT"] = df_tr.set_index([a, b]).index.map(vc).fillna(1).values
            out_te[f"{a}__{b}_CNT"] = df_te.set_index([a, b]).index.map(vc).fillna(1).values
    elif args.batch == "b4":
        out_tr["REGPOP_X_EXTMEAN"] = df_tr["REGION_POPULATION_RELATIVE"] * df_tr["EXT_SOURCES_MEAN"]
        out_te["REGPOP_X_EXTMEAN"] = df_te["REGION_POPULATION_RELATIVE"] * df_te["EXT_SOURCES_MEAN"]
    return out_tr, out_te


X_new, Xt_new = build(Xw, Xtw)
new_cols = [c for c in X_new.columns if c not in Xw.columns]
print(f"new cols: {new_cols}", flush=True)


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
        oof += oof_s / len(seeds)
    return oof


print("baseline LGB on same base...", flush=True)
oof_base = lgb_oof_on(Xw)
a_base = roc_auc_score(C.y, oof_base)
print(f"BASE {a_base:.6f} ({time.time()-t0:.0f}s)", flush=True)

print("batch LGB...", flush=True)
oof_new = lgb_oof_on(X_new)
a_new = roc_auc_score(C.y, oof_new)
print(f"BATCH {args.batch} {a_new:.6f} delta {a_new - a_base:+.6f} ({time.time()-t0:.0f}s)", flush=True)
np.save(C.OUT / f"w4_{args.batch}_oof.npy", oof_new)
X_new.to_parquet(C.OUT / f"X_e2_w4_{args.batch}.parquet")
Xt_new.to_parquet(C.OUT / f"X_test_e2_w4_{args.batch}.parquet")
print(f"W4 {args.batch} DONE delta={a_new - a_base:+.6f} keep={a_new - a_base >= 0.0003}", flush=True)
