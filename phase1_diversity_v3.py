"""
Phase 1: Local CPU Diversity Experiments - SIMPLIFIED & FAST
Goal: Find models that add +0.0015 CV to current best blend (0.76203)
Focus: Core 4 GBMs + Fast diversity models (no nested CV)
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.calibration import CalibratedClassifierCV
import time, os, gc, json, warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. LOAD RAW DATA & FEATURE ENGINEERING
# ============================================================

INPUT_DIR = r"C:\Users\amanuel\OneDrive\Documents\deepseek-harness\default-workspace\competition_unzipped\04. Competition (due Nov 20)\Competition\input"

print("Loading raw data...")
train = pd.read_csv(os.path.join(INPUT_DIR, 'train.csv'))
test = pd.read_csv(os.path.join(INPUT_DIR, 'test.csv'))
sample_sub = pd.read_csv(os.path.join(INPUT_DIR, 'sample_submission.csv'))

y = train['TARGET'].values
train_ids = train['SK_ID_CURR'].values
test_ids = test['SK_ID_CURR'].values

print(f"Train: {train.shape}, Test: {test.shape}, Positive rate: {y.mean():.4f}")

# ---- Feature Engineering ----

def handle_anomalies(df):
    df['DAYS_EMPLOYED_ANOM'] = (df['DAYS_EMPLOYED'] == 365243).astype(np.int8)
    df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].replace(365243, np.nan)
    if 'OWN_CAR_AGE' in df.columns:
        df['OWN_CAR_AGE'] = df['OWN_CAR_AGE'].clip(lower=0)
    return df

def add_missing_indicators(df, cols):
    for c in cols:
        if c in df.columns:
            df[f'{c}_WAS_MISSING'] = df[c].isnull().astype(np.int8)
    return df

def add_domain_ratios(df):
    eps = 1e-5
    df['CREDIT_INCOME_RATIO'] = df['AMT_CREDIT'] / (df['AMT_INCOME_TOTAL'] + eps)
    df['ANNUITY_INCOME_RATIO'] = df['AMT_ANNUITY'] / (df['AMT_INCOME_TOTAL'] + eps)
    df['CREDIT_TERM'] = df['AMT_ANNUITY'] / (df['AMT_CREDIT'] + eps)
    df['DAYS_EMPLOYED_RATIO'] = df['DAYS_EMPLOYED'] / (df['DAYS_BIRTH'] + eps)
    df['GOODS_CREDIT_RATIO'] = df['AMT_GOODS_PRICE'] / (df['AMT_CREDIT'] + eps)
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / (df['CNT_FAM_MEMBERS'] + eps)
    df['CREDIT_TO_GOODS_RATIO'] = df['AMT_CREDIT'] / (df['AMT_GOODS_PRICE'] + eps)
    df['ANNUITY_TO_CREDIT_RATIO'] = df['AMT_ANNUITY'] / (df['AMT_CREDIT'] + eps)
    df['INCOME_TO_CREDIT_RATIO'] = df['AMT_INCOME_TOTAL'] / (df['AMT_CREDIT'] + eps)
    df['PHONE_CHANGE_TO_AGE'] = df['DAYS_LAST_PHONE_CHANGE'] / (df['DAYS_BIRTH'] + eps)
    df['EMPLOYED_TO_BIRTH_RATIO'] = df['DAYS_EMPLOYED'] / (df['DAYS_BIRTH'] + eps)
    df['REGISTRATION_TO_BIRTH'] = df['DAYS_REGISTRATION'] / (df['DAYS_BIRTH'] + eps)
    df['ID_PUBLISH_TO_BIRTH'] = df['DAYS_ID_PUBLISH'] / (df['DAYS_BIRTH'] + eps)
    df['CAR_AGE_TO_EMPLOYED'] = df['OWN_CAR_AGE'] / (df['DAYS_EMPLOYED'].abs() + eps)
    df['CAR_AGE_TO_BIRTH'] = df['OWN_CAR_AGE'] / (df['DAYS_BIRTH'].abs() + eps)
    return df

def add_ext_source_interactions(df):
    ext = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
    df['EXT_SOURCES_MEAN'] = df[ext].mean(axis=1)
    df['EXT_SOURCES_PROD'] = df['EXT_SOURCE_1'].fillna(0) * df['EXT_SOURCE_2'].fillna(0) * df['EXT_SOURCE_3'].fillna(0)
    df['EXT_SOURCES_STD'] = df[ext].std(axis=1)
    df['EXT_SOURCES_MAX'] = df[ext].max(axis=1)
    df['EXT_SOURCES_MIN'] = df[ext].min(axis=1)
    df['EXT_1_2_PROD'] = df['EXT_SOURCE_1'].fillna(0) * df['EXT_SOURCE_2'].fillna(0)
    df['EXT_1_3_PROD'] = df['EXT_SOURCE_1'].fillna(0) * df['EXT_SOURCE_3'].fillna(0)
    df['EXT_2_3_PROD'] = df['EXT_SOURCE_2'].fillna(0) * df['EXT_SOURCE_3'].fillna(0)
    df['EXT_1_2_DIFF'] = (df['EXT_SOURCE_1'] - df['EXT_SOURCE_2']).abs()
    df['EXT_1_3_DIFF'] = (df['EXT_SOURCE_1'] - df['EXT_SOURCE_3']).abs()
    df['EXT_2_3_DIFF'] = (df['EXT_SOURCE_2'] - df['EXT_SOURCE_3']).abs()
    df['EXT_1_2_RATIO'] = df['EXT_SOURCE_1'] / (df['EXT_SOURCE_2'] + 1e-5)
    df['EXT_1_3_RATIO'] = df['EXT_SOURCE_1'] / (df['EXT_SOURCE_3'] + 1e-5)
    df['EXT_2_3_RATIO'] = df['EXT_SOURCE_2'] / (df['EXT_SOURCE_3'] + 1e-5)
    df['EXT_SOURCES_COUNT'] = df[ext].notnull().sum(axis=1)
    df['EXT_SOURCES_WEIGHTED_MEAN'] = (df['EXT_SOURCE_1'].fillna(0) + df['EXT_SOURCE_2'].fillna(0) + df['EXT_SOURCE_3'].fillna(0)) / (df['EXT_SOURCES_COUNT'] + 1e-5)
    return df

def add_bureau_aggregations(df):
    b = ['AMT_REQ_CREDIT_BUREAU_HOUR','AMT_REQ_CREDIT_BUREAU_MON',
         'AMT_REQ_CREDIT_BUREAU_QRT','AMT_REQ_CREDIT_BUREAU_YEAR']
    df['BUREAU_QUERY_SUM'] = df[b].fillna(0).sum(axis=1)
    df['BUREAU_QUERY_MEAN'] = df[b].fillna(0).mean(axis=1)
    df['BUREAU_QUERY_MAX'] = df[b].fillna(0).max(axis=1)
    df['BUREAU_QUERY_COUNT'] = df[b].notnull().sum(axis=1)
    df['BUREAU_RECENT_TO_HIST_RATIO'] = (df['AMT_REQ_CREDIT_BUREAU_HOUR'].fillna(0) + df['AMT_REQ_CREDIT_BUREAU_MON'].fillna(0)) / (df['AMT_REQ_CREDIT_BUREAU_QRT'].fillna(0) + df['AMT_REQ_CREDIT_BUREAU_YEAR'].fillna(0) + 1e-5)
    return df

def add_binned_features(df, ref):
    binc = ['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3','AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY',
            'DAYS_BIRTH','DAYS_EMPLOYED','DAYS_REGISTRATION','DAYS_ID_PUBLISH','DAYS_LAST_PHONE_CHANGE',
            'CNT_FAM_MEMBERS','CNT_CHILDREN','REGION_POPULATION_RELATIVE','REGION_RATING_CLIENT']
    for col in binc:
        if col in df.columns:
            try:
                q = ref[col].quantile([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]).values
                df[f'{col}_QBIN'] = pd.cut(df[col], bins=[-np.inf]+list(q)+[np.inf], labels=False, duplicates='drop')
            except Exception:
                df[f'{col}_QBIN'] = pd.qcut(df[col], q=10, labels=False, duplicates='drop')
            df[f'{col}_QBIN'] = df[f'{col}_QBIN'].fillna(-1).astype(np.int8)
    return df

HIGH_MISSING = ['EXT_SOURCE_1','OWN_CAR_AGE','EXT_SOURCE_3','AMT_REQ_CREDIT_BUREAU_HOUR',
                'AMT_REQ_CREDIT_BUREAU_MON','AMT_REQ_CREDIT_BUREAU_QRT','AMT_REQ_CREDIT_BUREAU_YEAR',
                'EXT_SOURCE_2','CNT_FAM_MEMBERS']

# Apply FE
tr_fe = train.drop(columns=['TARGET','SK_ID_CURR']).copy()
te_fe = test.drop(columns=['SK_ID_CURR']).copy()

tr_fe = handle_anomalies(tr_fe); te_fe = handle_anomalies(te_fe)
tr_fe = add_missing_indicators(tr_fe, HIGH_MISSING); te_fe = add_missing_indicators(te_fe, HIGH_MISSING)
tr_fe = add_domain_ratios(tr_fe); te_fe = add_domain_ratios(te_fe)
tr_fe = add_ext_source_interactions(tr_fe); te_fe = add_ext_source_interactions(te_fe)
tr_fe = add_bureau_aggregations(tr_fe); te_fe = add_bureau_aggregations(te_fe)
tr_fe = add_binned_features(tr_fe, tr_fe); te_fe = add_binned_features(te_fe, tr_fe)

# Label + frequency encoding
cat_cols = tr_fe.select_dtypes(include=['object']).columns.tolist()
for col in cat_cols:
    combined = pd.concat([tr_fe[col], te_fe[col]], axis=0)
    le = LabelEncoder(); le.fit(combined.astype(str))
    tr_fe[col] = le.transform(tr_fe[col].astype(str))
    te_fe[col] = le.transform(te_fe[col].astype(str))
    freq = combined.value_counts(normalize=True).to_dict()
    tr_fe[f'{col}_FREQ'] = tr_fe[col].map(freq).astype(np.float32)
    te_fe[f'{col}_FREQ'] = te_fe[col].map(freq).astype(np.float32)

feature_cols = list(tr_fe.columns)
tr_fe[feature_cols] = tr_fe[feature_cols].fillna(-999)
te_fe[feature_cols] = te_fe[feature_cols].fillna(-999)
X0 = tr_fe.astype(np.float32)
X_test0 = te_fe.astype(np.float32)
print(f"Base engineered: {X0.shape}")

# ---- Enrichment + Dead Feature Removal ----

DEAD = ['FLAG_MOBIL','ORGANIZATION_TYPE','EXT_SOURCE_2_WAS_MISSING','CNT_FAM_MEMBERS_WAS_MISSING',
        'AMT_REQ_CREDIT_BUREAU_MON_WAS_MISSING','AMT_REQ_CREDIT_BUREAU_YEAR_WAS_MISSING',
        'EXT_SOURCE_1_WAS_MISSING','EXT_SOURCE_3_WAS_MISSING','FLAG_EMP_PHONE',
        'EXT_SOURCE_3_QBIN','OCCUPATION_TYPE_FREQ','NAME_HOUSING_TYPE_FREQ','NAME_FAMILY_STATUS_FREQ',
        'NAME_EDUCATION_TYPE_FREQ','NAME_INCOME_TYPE_FREQ','CODE_GENDER_FREQ','NAME_CONTRACT_TYPE_FREQ',
        'ORGANIZATION_TYPE_FREQ']

def enrich(df, base_cols):
    X = df.copy(); eps = 1e-5
    ext = ['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']
    filled = X[ext].replace(-999, np.nan)
    f = filled.fillna(filled.mean())
    age_years = (X['DAYS_BIRTH'].replace(-999, np.nan).abs() / 365.0).fillna(
        X['DAYS_BIRTH'].replace(-999, np.nan).abs().mean() / 365.0)
    for c in ext:
        X[f'{c}X_AGE'] = f[c] * age_years
    X['EXT_MEAN_X_AGE'] = f.mean(axis=1) * age_years
    X['EXT_2x3'] = f['EXT_SOURCE_2'] * f['EXT_SOURCE_3']
    X['EXT_1x2x3_NANSAFE'] = f['EXT_SOURCE_1'] * f['EXT_SOURCE_2'] * f['EXT_SOURCE_3']
    X['EXT_2_SQ'] = f['EXT_SOURCE_2'] ** 2
    X['EXT_3_SQ'] = f['EXT_SOURCE_3'] ** 2
    X['EXT_SUM'] = f.sum(axis=1)
    emp_years = X['DAYS_EMPLOYED'].replace(-999, np.nan).abs() / 365.0
    X['CREDIT_PER_INCOME_YEAR'] = X['AMT_CREDIT'] / ((X['AMT_INCOME_TOTAL'] + eps) * emp_years.fillna(emp_years.mean()) + eps)
    X['ANNUITY_PER_FAM'] = X['AMT_ANNUITY'] / (X['CNT_FAM_MEMBERS'].replace(0, 1) + eps)
    X['CREDIT_PER_FAM'] = X['AMT_CREDIT'] / (X['CNT_FAM_MEMBERS'].replace(0, 1) + eps)
    X['INCOME_PER_CHILD'] = X['AMT_INCOME_TOTAL'] / (X['CNT_CHILDREN'].clip(lower=0) + 1)
    X['NAN_COUNT'] = (df[base_cols] == -999).sum(axis=1).astype(np.int16)
    X['REGION_RATING_X_EXT_MEAN'] = X['REGION_RATING_CLIENT'] * f.mean(axis=1)
    return X

X = enrich(X0, feature_cols).drop(columns=[c for c in DEAD if c in X0.columns])
X_test = enrich(X_test0, feature_cols).drop(columns=[c for c in DEAD if c in X_test0.columns])
assert list(X.columns) == list(X_test.columns)
print(f"Final features: {X.shape[1]}")

# ---- CV Folds + Target Encoding ----

SEED = 42
N_SPLITS = 5
N_ESTIMATORS = 3000
EARLY_STOPPING = 200

HIGH_CARD = ['ORGANIZATION_TYPE','OCCUPATION_TYPE','NAME_INCOME_TYPE','NAME_EDUCATION_TYPE',
             'NAME_FAMILY_STATUS','NAME_HOUSING_TYPE','NAME_CONTRACT_TYPE']
TE_COLS = [c for c in HIGH_CARD if c in X.columns]
CATS = ['NAME_CONTRACT_TYPE','CODE_GENDER','NAME_INCOME_TYPE','NAME_EDUCATION_TYPE',
        'NAME_FAMILY_STATUS','NAME_HOUSING_TYPE','OCCUPATION_TYPE','FLAG_WORK_PHONE']

def add_te(X_tr, X_va, X_te, y_tr, smoothing=10):
    X_tr, X_va, X_te = X_tr.copy(), X_va.copy(), X_te.copy()
    gs = pd.Series(y_tr, index=X_tr.index); gm = gs.mean()
    for col in TE_COLS:
        means = gs.groupby(X_tr[col]).mean()
        counts = X_tr[col].value_counts()
        sm = (means * counts + gm * smoothing) / (counts + smoothing)
        X_tr[f'{col}_TE'] = X_tr[col].map(sm).fillna(gm)
        X_va[f'{col}_TE'] = X_va[col].map(sm).fillna(gm)
        X_te[f'{col}_TE'] = X_te[col].map(sm).fillna(gm)
    return X_tr, X_va, X_te

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
FOLDS = list(skf.split(X, y))

def cat_idx(Xc):
    return [list(Xc.columns).index(c) for c in CATS if c in Xc.columns]

print(f"Folds ready | TE cols: {TE_COLS}")

# ============================================================
# 2. TUED PARAMS (from Optuna)
# ============================================================

LGB_PARAMS = {"learning_rate": 0.04321176733666811, "num_leaves": 28, "max_depth": 3,
    "min_child_samples": 97, "subsample": 0.744137352044092, "colsample_bytree": 0.8258847596441867,
    "reg_alpha": 0.1853706492115699, "reg_lambda": 0.035504460820014246,
    "min_split_gain": 0.7618907678134024, "min_child_weight": 0.10567827282349425}

XGB_PARAMS = {"learning_rate": 0.023127311910052063, "max_depth": 3, "min_child_weight": 8.595010289429197,
    "subsample": 0.8803532825765757, "colsample_bytree": 0.5325711926493584,
    "colsample_bylevel": 0.6102966544921201, "reg_alpha": 0.0034216968534539678,
    "reg_lambda": 7.070529170721945, "gamma": 0.29349768963525397}

CAT_PARAMS = {"learning_rate": 0.033841402216264506, "depth": 3, "l2_leaf_reg": 4.5756057217815505,
    "bagging_temperature": 0.8546621517583892, "random_strength": 0.0028359528535650993,
    "border_count": 215}

# ============================================================
# 3. CV EVALUATION FUNCTIONS
# ============================================================

def convert_cats(df):
    """Convert categorical columns to int for model compatibility"""
    df = df.copy()
    for c in CATS:
        if c in df.columns:
            df[c] = df[c].astype(int)
    return df

def cv_score_lgb(X, y, name, seeds=[42], n_estimators=2000, early_stopping=100):
    """LGBM with optional seed averaging"""
    oof_all = np.zeros(len(X))
    for seed in seeds:
        oof_s = np.zeros(len(X))
        for fold, (tri, vai) in enumerate(FOLDS):
            X_tr, X_va, X_te = add_te(X.iloc[tri], X.iloc[vai], X_test, y[tri])
            X_tr = convert_cats(X_tr); X_va = convert_cats(X_va)
            m = lgb.LGBMClassifier(**LGB_PARAMS, n_estimators=n_estimators, random_state=seed, verbose=-1, n_jobs=2)
            m.fit(X_tr, y[tri], eval_set=[(X_va, y[vai])], eval_metric='auc',
                  categorical_feature=cat_idx(X_tr),
                  callbacks=[lgb.early_stopping(early_stopping, verbose=False)])
            oof_s[vai] = m.predict_proba(X_va)[:, 1]
        oof_all += oof_s / len(seeds)
    auc = roc_auc_score(y, oof_all)
    print(f"  {name}: CV AUC = {auc:.5f}")
    return oof_all

def cv_score_xgb(X, y, name, n_estimators=2000, early_stopping=100):
    oof = np.zeros(len(X))
    xp0 = dict(XGB_PARAMS); xp0.update({'objective':'binary:logistic','eval_metric':'auc','n_jobs':2,'verbosity':0,'random_state':42})
    for fold, (tri, vai) in enumerate(FOLDS):
        X_tr, X_va, X_te = add_te(X.iloc[tri], X.iloc[vai], X_test, y[tri])
        X_tr = convert_cats(X_tr); X_va = convert_cats(X_va)
        dtr = xgb.DMatrix(X_tr, label=y[tri], enable_categorical=True); dva = xgb.DMatrix(X_va, label=y[vai], enable_categorical=True)
        m = xgb.train({**xp0, 'tree_method':'hist'}, dtr, num_boost_round=n_estimators,
                      evals=[(dva,'val')], early_stopping_rounds=early_stopping, verbose_eval=False)
        oof[vai] = m.predict(dva, iteration_range=(0, m.best_iteration + 1))
    auc = roc_auc_score(y, oof)
    print(f"  {name}: CV AUC = {auc:.5f}")
    return oof

def cv_score_cat(X, y, name, n_estimators=2000, early_stopping=100):
    oof = np.zeros(len(X))
    cp0 = dict(CAT_PARAMS); cp0.update({'loss_function':'Logloss','eval_metric':'AUC','verbose':False,'allow_writing_files':False})
    for fold, (tri, vai) in enumerate(FOLDS):
        X_tr, X_va, X_te = add_te(X.iloc[tri], X.iloc[vai], X_test, y[tri])
        X_tr = convert_cats(X_tr); X_va = convert_cats(X_va)
        ci = cat_idx(X_tr)
        m = cb.CatBoostClassifier(**cp0, iterations=n_estimators, random_seed=42)
        m.fit(X_tr, y[tri], eval_set=(X_va, y[vai]), cat_features=ci, early_stopping_rounds=early_stopping)
        oof[vai] = m.predict_proba(X_va)[:, 1]
    auc = roc_auc_score(y, oof)
    print(f"  {name}: CV AUC = {auc:.5f}")
    return oof

def cv_score_hgb(X, y, name):
    oof = np.zeros(len(X))
    for fold, (tri, vai) in enumerate(FOLDS):
        X_tr, X_va, X_te = add_te(X.iloc[tri], X.iloc[vai], X_test, y[tri])
        X_tr = convert_cats(X_tr); X_va = convert_cats(X_va)
        m = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=600, max_leaf_nodes=31,
                                           min_samples_leaf=50, l2_regularization=1.0, random_state=SEED)
        m.fit(X_tr, y[tri])
        oof[vai] = m.predict_proba(X_va)[:, 1]
    auc = roc_auc_score(y, oof)
    print(f"  {name}: CV AUC = {auc:.5f}")
    return oof

def cv_score_sklearn(model_factory, X, y, name):
    """Generic CV for sklearn models"""
    oof = np.zeros(len(X))
    for fold, (tri, vai) in enumerate(FOLDS):
        X_tr, X_va, X_te = add_te(X.iloc[tri], X.iloc[vai], X_test, y[tri])
        X_tr = convert_cats(X_tr); X_va = convert_cats(X_va)
        m = model_factory()
        m.fit(X_tr, y[tri])
        oof[vai] = m.predict_proba(X_va)[:, 1]
    auc = roc_auc_score(y, oof)
    print(f"  {name}: CV AUC = {auc:.5f}")
    return oof

# ============================================================
# 4. REGENERATE BASELINE GBMS (THE CORE 4)
# ============================================================

print("\n" + "="*60)
print("REGENERATING BASELINE GBMS (Core 4 Models)")
print("="*60)

# LGB 3-seed
print("\n--- LGB 3-seed ---")
oof_lgb3 = cv_score_lgb(X, y, "LGB3", seeds=[42, 1337, 2026])

# XGB
print("\n--- XGB ---")
oof_xgb = cv_score_xgb(X, y, "XGB")

# CAT (CPU)
print("\n--- CAT (CPU) ---")
oof_cat = cv_score_cat(X, y, "CAT")

# HGB
print("\n--- HGB ---")
oof_hgb = cv_score_hgb(X, y, "HGB")

# ============================================================
# 5. DIVERSITY MODELS (FAST - no nested CV)
# ============================================================

print("\n" + "="*60)
print("DIVERSITY MODELS")
print("="*60)

# Linear models on standardized features
scaler = StandardScaler()
X_scaled_df = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

print("\n--- LogisticRegression (C=1.0) ---")
oof_lr = cv_score_sklearn(
    lambda: LogisticRegression(C=1.0, random_state=42, max_iter=500, n_jobs=4, solver='lbfgs'),
    X_scaled_df, y, "LogReg"
)

print("\n--- ExtraTrees ---")
oof_et = cv_score_sklearn(
    lambda: ExtraTreesClassifier(n_estimators=300, max_features='sqrt', max_depth=8,
                                 min_samples_leaf=20, random_state=42, n_jobs=4, verbose=0),
    X, y, "ExtraTrees"
)

print("\n--- RandomForest ---")
oof_rf = cv_score_sklearn(
    lambda: RandomForestClassifier(n_estimators=300, max_features='sqrt', max_depth=8,
                                   min_samples_leaf=20, random_state=42, n_jobs=4, verbose=0),
    X, y, "RandomForest"
)

# ============================================================
# 6. FEATURE SUBSPACE ENSEMBLE (3 LGBM on 70% features)
# ============================================================

print("\n--- Feature Subspace Ensemble (3 LGBM x 70% features) ---")
n_features = X.shape[1]
subspace_size = int(n_features * 0.7)
oof_subspace = []

# Ensure TE columns are always included
te_col_indices = [list(X.columns).index(c) for c in TE_COLS if c in X.columns]

for i in range(3):
    np.random.seed(42 + i)
    # Sample from non-TE columns
    non_te_indices = [j for j in range(n_features) if j not in te_col_indices]
    sampled_non_te = np.random.choice(non_te_indices, min(subspace_size - len(te_col_indices), len(non_te_indices)), replace=False)
    feat_idx = np.concatenate([te_col_indices, sampled_non_te])
    
    X_sub = X.iloc[:, feat_idx]
    X_test_sub = X_test.iloc[:, feat_idx]
    
    oof_s = np.zeros(len(X))
    for fold, (tri, vai) in enumerate(FOLDS):
        X_tr, X_va, X_te = add_te(X_sub.iloc[tri], X_sub.iloc[vai], X_test_sub, y[tri])
        X_tr = convert_cats(X_tr); X_va = convert_cats(X_va)
        m = lgb.LGBMClassifier(**LGB_PARAMS, n_estimators=2000, random_state=42, verbose=-1, n_jobs=2)
        m.fit(X_tr, y[tri], eval_set=[(X_va, y[vai])], eval_metric='auc',
              categorical_feature=cat_idx(X_tr),
              callbacks=[lgb.early_stopping(100, verbose=False)])
        oof_s[vai] = m.predict_proba(X_va)[:, 1]
    auc_s = roc_auc_score(y, oof_s)
    print(f"  Subspace {i+1}: {auc_s:.5f}")
    oof_subspace.append(oof_s)

# ============================================================
# 7. ALL OOFS COLLECTED - TEST BLENDS
# ============================================================

print("\n" + "="*60)
print("BLEND EXPERIMENTS")
print("="*60)

all_oofs = {
    'LGB3': oof_lgb3,
    'XGB': oof_xgb,
    'CAT': oof_cat,
    'HGB': oof_hgb,
    'LogReg': oof_lr,
    'ExtraTrees': oof_et,
    'RandomForest': oof_rf,
    'Subspace0': oof_subspace[0],
    'Subspace1': oof_subspace[1],
    'Subspace2': oof_subspace[2],
}

# Single model scores
print("\n--- Single Model OOF AUCs ---")
for k, v in all_oofs.items():
    print(f"  {k:15s}: {roc_auc_score(y, v):.5f}")

# Simple blends
avg_blend = np.mean(list(all_oofs.values()), axis=0)
print(f"\nSimple average (all): {roc_auc_score(y, avg_blend):.5f}")

# Core 4 only
core4 = ['LGB3', 'XGB', 'CAT', 'HGB']
avg_core4 = np.mean([all_oofs[k] for k in core4], axis=0)
print(f"Core 4 average: {roc_auc_score(y, avg_core4):.5f}")

# Rank average
from scipy.stats import rankdata
rank_blend = np.mean([rankdata(p) for p in all_oofs.values()], axis=0) / len(y)
print(f"Rank average (all): {roc_auc_score(y, rank_blend):.5f}")

rank_core4 = np.mean([rankdata(all_oofs[k]) for k in core4], axis=0) / len(y)
print(f"Rank average (core4): {roc_auc_score(y, rank_core4):.5f}")

# Geometric mean
geo_blend = np.exp(np.mean([np.log(p + 1e-8) for p in all_oofs.values()], axis=0))
print(f"Geometric mean (all): {roc_auc_score(y, geo_blend):.5f}")

geo_core4 = np.exp(np.mean([np.log(all_oofs[k] + 1e-8) for k in core4], axis=0))
print(f"Geometric mean (core4): {roc_auc_score(y, geo_core4):.5f}")

# Harmonic mean
harm_blend = len(all_oofs) / np.sum([1/(p + 1e-8) for p in all_oofs.values()], axis=0)
print(f"Harmonic mean (all): {roc_auc_score(y, harm_blend):.5f}")

# ============================================================
# 8. HILL-CLIMB BLEND SELECTION
# ============================================================

print("\n--- Hill-Climb Blend Selection ---")

OOF_M = np.column_stack(list(all_oofs.values()))
names = list(all_oofs.keys())

current = None
chosen = []
counts = {k: 0 for k in names}

for step in range(15):
    best_k, best_a = None, -1.0
    for i, k in enumerate(names):
        cand = OOF_M[:, i] if current is None else (current + OOF_M[:, i]) / 2
        a = roc_auc_score(y, cand)
        if a > best_a:
            best_k, best_a = k, a
    if current is not None and best_a <= roc_auc_score(y, current) + 1e-6:
        break
    current = OOF_M[:, names.index(best_k)] if current is None else (current + OOF_M[:, names.index(best_k)]) / 2
    counts[best_k] += 1
    chosen.append(best_k)
    print(f"  Step {step+1}: +{best_k} -> {best_a:.5f} | counts: {{k:v for k,v in counts.items() if v}}")

hill_auc = roc_auc_score(y, current)
print(f"\nHill-climb final: {hill_auc:.5f}")

# ============================================================
# 9. GREEDY WEIGHT OPTIMIZATION (on Core 4)
# ============================================================

print("\n--- Greedy Weight Optimization (Core 4) ---")
core_oofs = {k: all_oofs[k] for k in core4}
CORE_M = np.column_stack([core_oofs[k] for k in core4])
core_names = core4

best_weights = np.ones(len(core_names)) / len(core_names)
best_auc = roc_auc_score(y, best_weights @ CORE_M.T)

for iteration in range(100):
    improved = False
    for i in range(len(core_names)):
        for delta in [0.01, -0.01]:
            new_w = best_weights.copy()
            new_w[i] += delta
            if new_w.min() >= 0:
                new_w = new_w / new_w.sum()
                auc = roc_auc_score(y, new_w @ CORE_M.T)
                if auc > best_auc + 1e-6:
                    best_auc = auc
                    best_weights = new_w
                    improved = True
    if not improved:
        break

print(f"Greedy weights: {dict(zip(core_names, best_weights))}")
print(f"Greedy blend AUC: {best_auc:.5f}")

# ============================================================
# 10. RANK CORRELATIONS
# ============================================================

print("\n--- Rank Correlations vs XGB (strongest single) ---")
xgb_ranks = rankdata(all_oofs['XGB'])
for k, v in all_oofs.items():
    rho = np.corrcoef(rankdata(v), xgb_ranks)[0, 1]
    print(f"  {k:15s}: {rho:.4f}")

# ============================================================
# 11. SAVE RESULTS
# ============================================================

results = {
    'single_models': {k: float(roc_auc_score(y, v)) for k, v in all_oofs.items()},
    'blends': {
        'simple_avg_all': float(roc_auc_score(y, avg_blend)),
        'simple_avg_core4': float(roc_auc_score(y, avg_core4)),
        'rank_avg_all': float(roc_auc_score(y, rank_blend)),
        'rank_avg_core4': float(roc_auc_score(y, rank_core4)),
        'geo_mean_all': float(roc_auc_score(y, geo_blend)),
        'geo_mean_core4': float(roc_auc_score(y, geo_core4)),
        'harm_mean_all': float(roc_auc_score(y, harm_blend)),
        'hill_climb_all': float(hill_auc),
        'greedy_core4': float(best_auc),
    },
    'hill_climb_counts': counts,
    'greedy_core4_weights': {k: float(w) for k, w in zip(core_names, best_weights)},
    'rank_correlations_vs_xgb': {k: float(np.corrcoef(rankdata(v), xgb_ranks)[0, 1]) for k, v in all_oofs.items()},
}

with open('phase1_results_v3.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n" + "="*60)
print("PHASE 1 COMPLETE - Results saved to phase1_results_v3.json")
print("="*60)
print(json.dumps(results, indent=2))

# ============================================================
# 12. SAVE OOF PREDS FOR PHASE 2 (STACKING)
# ============================================================

np.save('phase1_oof_lgb3.npy', oof_lgb3)
np.save('phase1_oof_xgb.npy', oof_xgb)
np.save('phase1_oof_cat.npy', oof_cat)
np.save('phase1_oof_hgb.npy', oof_hgb)
np.save('phase1_oof_lr.npy', oof_lr)
np.save('phase1_oof_et.npy', oof_et)
np.save('phase1_oof_rf.npy', oof_rf)
np.save('phase1_oof_subspace0.npy', oof_subspace[0])
np.save('phase1_oof_subspace1.npy', oof_subspace[1])
np.save('phase1_oof_subspace2.npy', oof_subspace[2])

print("\nOOF predictions saved for Phase 2 (stacking)")