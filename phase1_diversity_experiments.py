"""
Phase 1: Local CPU Diversity Experiments
Goal: Find models that add +0.0015 CV to current best blend (0.76203)
Tests: Linear models, ExtraTrees, RF, Feature subspace, Rank/Geo/Harm blends
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV, RidgeCV, ElasticNetCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
import time, os, gc, json, warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. LOAD RAW DATA & FEATURE ENGINEERING (from honors notebook)
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

# ---- Feature Engineering (copied from gci_honors_reproduction.ipynb) ----

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
            # Handle NaN values before converting to int8
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

# ---- Enrichment + Dead Feature Removal (from NB3 v2) ----

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
# 2. LOAD BASELINE OOF PREDS (for rank/geo/harm blends)
# ============================================================

# We'll regenerate the 4 baseline models quickly, or load from Drive if available
# For now, let's regenerate LGB3, XGB, CAT, HGB (but this takes time)
# Let's first test the SIMPLE blends that need no new training

print("\n" + "="*60)
print("PHASE 1A: RANK / GEOMETRIC / HARMONIC BLENDS (Instant)")
print("="*60)

# We need baseline OOF preds. Let's check if we have them from the Colab run
# Since we don't have .npy files locally, we'll need to regenerate
# But first, let's try the SIMPLE approaches that use existing data

# Actually, let's just run the quick diversity models first
# and compute rank correlations

# ============================================================
# 3. QUICK DIVERSITY MODELS (Linear, ExtraTrees, RF)
# ============================================================

print("\n" + "="*60)
print("PHASE 1B: LINEAR MODELS ON STANDARDIZED FEATURES")
print("="*60)

# Standardize features for linear models
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# Get top 20 SHAP features from earlier (we know them)
# EXT_SOURCES_WEIGHTED_MEAN, EXT_SOURCES_MEAN, AMT_GOODS_PRICE, EXT_2_3_PROD,
# AMT_ANNUITY, CODE_GENDER, CREDIT_TERM, EXT_SOURCES_MAX, EXT_SOURCES_MIN,
# ANNUITY_TO_CREDIT_RATIO, OCCUPATION_TYPE_TE, NAME_EDUCATION_TYPE_TE,
# DAYS_EMPLOYED_QBIN, GOODS_CREDIT_RATIO, CAR_AGE_TO_EMPLOYED,
# CREDIT_TO_GOODS_RATIO, AMT_CREDIT, EXT_SOURCE_3, DAYS_ID_PUBLISH,
# EXT_1_2_PROD

# For linear models, let's use all features but with polynomial degree 2 on top features
# Actually, simpler: just use standardized features + degree-2 poly on top 10

# Let's get feature importances from a quick LGBM
print("Quick LGBM for feature importance...")
lgb_quick = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, max_depth=3,
                                num_leaves=28, random_state=42, n_jobs=4, verbose=-1)
lgb_quick.fit(X, y, categorical_feature=cat_idx(X))
importances = pd.DataFrame({'feature': X.columns, 'importance': lgb_quick.feature_importances_})
importances = importances.sort_values('importance', ascending=False)
top20 = importances.head(20)['feature'].tolist()
print(f"Top 20 features: {top20}")

# Create polynomial features on top 10
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_top10 = X[top20[:10]]
X_top10_poly = poly.fit_transform(X_top10)
print(f"Poly features shape: {X_top10_poly.shape}")

# Combine: standardized all + poly top10
X_linear = np.hstack([X_scaled, X_top10_poly])
X_test_linear = np.hstack([X_test_scaled, poly.transform(X_test[top20[:10]])])
print(f"Linear feature matrix: {X_linear.shape}")

# ============================================================
# 4. CV EVALUATION FUNCTION
# ============================================================

def cv_score(model_factory, X, y, name, n_splits=5, seed=42):
    """Run stratified K-fold CV, return OOF preds and AUC"""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(X))
    t0 = time.time()
    for fold, (tri, vai) in enumerate(skf.split(X, y)):
        model = model_factory()
        model.fit(X[tri], y[tri])
        oof[vai] = model.predict_proba(X[vai])[:, 1]
    auc = roc_auc_score(y, oof)
    print(f"  {name}: CV AUC = {auc:.5f} ({time.time()-t0:.0f}s)")
    return oof, auc

# ============================================================
# 5. TEST LINEAR MODELS
# ============================================================

print("\n--- Linear Models (Standardized + Poly Top 10) ---")

# LogisticRegression with CV
oof_lr, auc_lr = cv_score(
    lambda: LogisticRegressionCV(Cs=10, cv=3, random_state=42, max_iter=1000, n_jobs=4, scoring='roc_auc'),
    X_linear, y, "LogRegCV"
)

# RidgeCV
oof_ridge, auc_ridge = cv_score(
    lambda: RidgeCV(alphas=np.logspace(-3, 3, 13), cv=3, scoring='roc_auc'),
    X_linear, y, "RidgeCV"
)

# ElasticNetCV
oof_en, auc_en = cv_score(
    lambda: ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], alphas=np.logspace(-3, 1, 9), cv=3, random_state=42, max_iter=2000, n_jobs=4, scoring='roc_auc'),
    X_linear, y, "ElasticNetCV"
)

# ============================================================
# 6. TEST EXTRATREES & RANDOMFOREST
# ============================================================

print("\n--- ExtraTrees & RandomForest ---")

oof_et, auc_et = cv_score(
    lambda: ExtraTreesClassifier(n_estimators=500, max_features='sqrt', max_depth=10,
                                 min_samples_leaf=10, random_state=42, n_jobs=4, verbose=0),
    X, y, "ExtraTrees"
)

oof_rf, auc_rf = cv_score(
    lambda: RandomForestClassifier(n_estimators=500, max_features='sqrt', max_depth=10,
                                   min_samples_leaf=10, random_state=42, n_jobs=4, verbose=0),
    X, y, "RandomForest"
)

# ============================================================
# 7. FEATURE SUBSPACE ENSEMBLE (5 LGBM on 70% features)
# ============================================================

print("\n--- Feature Subspace Ensemble (5 LGBM × 70% features) ---")

n_features = X.shape[1]
subspace_size = int(n_features * 0.7)
oof_subspace = np.zeros((5, len(X)))

for i in range(5):
    np.random.seed(42 + i)
    feat_idx = np.random.choice(n_features, subspace_size, replace=False)
    X_sub = X.iloc[:, feat_idx]
    X_test_sub = X_test.iloc[:, feat_idx]
    
    oof_s = np.zeros(len(X))
    for fold, (tri, vai) in enumerate(FOLDS):
        X_tr, X_va, X_te = add_te(X_sub.iloc[tri], X_sub.iloc[vai], X_test_sub, y[tri])
        m = lgb.LGBMClassifier(n_estimators=2000, learning_rate=0.05, max_depth=3,
                               num_leaves=28, random_state=42, n_jobs=2, verbose=-1)
        m.fit(X_tr, y[tri], eval_set=[(X_va, y[vai])], eval_metric='auc',
              categorical_feature=cat_idx(X_tr),
              callbacks=[lgb.early_stopping(100, verbose=False)])
        oof_s[vai] = m.predict_proba(X_va)[:, 1]
    auc_s = roc_auc_score(y, oof_s)
    print(f"  Subspace {i+1}: {auc_s:.5f}")
    oof_subspace[i] = oof_s

# Best single subspace
best_sub_idx = np.argmax([roc_auc_score(y, oof_subspace[i]) for i in range(5)])
print(f"Best subspace: {best_sub_idx} (AUC={roc_auc_score(y, oof_subspace[best_sub_idx]):.5f})")

# ============================================================
# 8. RANK / GEOMETRIC / HARMONIC BLENDS (using all OOFs collected)
# ============================================================

print("\n--- Rank / Geometric / Harmonic Blends ---")

# Collect all OOF preds
all_oofs = {
    'LogReg': oof_lr,
    'Ridge': oof_ridge,
    'ElasticNet': oof_en,
    'ExtraTrees': oof_et,
    'RandomForest': oof_rf,
}
# Add best subspace
all_oofs[f'Subspace{best_sub_idx}'] = oof_subspace[best_sub_idx]

# Also include the baseline GBMs if we had them - for now use what we have
# Let's also do a quick LGBM 5-fold to have a strong baseline
print("\n--- Quick LGBM 5-fold (baseline) ---")
oof_lgb_base = np.zeros(len(X))
for fold, (tri, vai) in enumerate(FOLDS):
    X_tr, X_va, X_te = add_te(X.iloc[tri], X.iloc[vai], X_test, y[tri])
    m = lgb.LGBMClassifier(n_estimators=2000, learning_rate=0.05, max_depth=3,
                           num_leaves=28, random_state=42, n_jobs=2, verbose=-1)
    m.fit(X_tr, y[tri], eval_set=[(X_va, y[vai])], eval_metric='auc',
          categorical_feature=cat_idx(X_tr),
          callbacks=[lgb.early_stopping(100, verbose=False)])
    oof_lgb_base[vai] = m.predict_proba(X_va)[:, 1]
auc_lgb_base = roc_auc_score(y, oof_lgb_base)
print(f"  LGBM baseline: {auc_lgb_base:.5f}")
all_oofs['LGBM_base'] = oof_lgb_base

# Now test blends
print("\n--- Blend Experiments ---")

from scipy.stats import rankdata

# Simple average
avg_blend = np.mean(list(all_oofs.values()), axis=0)
print(f"Simple average: {roc_auc_score(y, avg_blend):.5f}")

# Rank average
rank_blend = np.mean([rankdata(p) for p in all_oofs.values()], axis=0) / len(y)
print(f"Rank average: {roc_auc_score(y, rank_blend):.5f}")

# Geometric mean
geo_blend = np.exp(np.mean([np.log(p + 1e-8) for p in all_oofs.values()], axis=0))
print(f"Geometric mean: {roc_auc_score(y, geo_blend):.5f}")

# Harmonic mean
harm_blend = len(all_oofs) / np.sum([1/(p + 1e-8) for p in all_oofs.values()], axis=0)
print(f"Harmonic mean: {roc_auc_score(y, harm_blend):.5f}")

# ============================================================
# 9. HILL-CLIMB BLEND SELECTION
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

# Weight grid (step 0.02 for 4-5 models, but let's do coarse for many)
# For many models, use differential evolution or just greedy
# Let's do a simple greedy weight optimization
print("\n--- Greedy Weight Optimization ---")
best_weights = np.ones(len(names)) / len(names)
best_auc = roc_auc_score(y, best_weights @ OOF_M.T)

for _ in range(100):
    improved = False
    for i in range(len(names)):
        for delta in [0.02, -0.02]:
            new_w = best_weights.copy()
            new_w[i] += delta
            # Renormalize
            if new_w.min() >= 0:
                new_w = new_w / new_w.sum()
                auc = roc_auc_score(y, new_w @ OOF_M.T)
                if auc > best_auc + 1e-6:
                    best_auc = auc
                    best_weights = new_w
                    improved = True
    if not improved:
        break

print(f"Greedy weights: {dict(zip(names, best_weights))}")
print(f"Greedy blend AUC: {best_auc:.5f}")

# ============================================================
# 10. RANK CORRELATIONS
# ============================================================

print("\n--- Rank Correlations vs LGBM_base ---")
for k, v in all_oofs.items():
    rho = np.corrcoef(rankdata(v), rankdata(all_oofs['LGBM_base']))[0, 1]
    print(f"  {k:15s}: {rho:.4f}")

# ============================================================
# 11. SAVE RESULTS
# ============================================================

results = {
    'single_models': {k: float(roc_auc_score(y, v)) for k, v in all_oofs.items()},
    'blends': {
        'simple_avg': float(roc_auc_score(y, avg_blend)),
        'rank_avg': float(roc_auc_score(y, rank_blend)),
        'geo_mean': float(roc_auc_score(y, geo_blend)),
        'harm_mean': float(roc_auc_score(y, harm_blend)),
        'hill_climb': float(hill_auc),
        'greedy_weights': float(best_auc),
    },
    'hill_climb_counts': counts,
    'greedy_weights': {k: float(w) for k, w in zip(names, best_weights)},
    'rank_correlations': {k: float(np.corrcoef(rankdata(v), rankdata(all_oofs['LGBM_base']))[0, 1]) for k, v in all_oofs.items()},
}

with open('phase1_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n" + "="*60)
print("PHASE 1 COMPLETE - Results saved to phase1_results.json")
print("="*60)
print(json.dumps(results, indent=2))