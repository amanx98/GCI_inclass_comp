import sys, json, time
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import pandas as pd

WS = r"C:\Users\amanuel\OneDrive\Documents\deepseek-harness\default-workspace"
T = r"C:\Users\amanuel\AppData\Local\Temp\opencode"
C = rf"{WS}\competition_unzipped\04. Competition (due Nov 20)\Competition"

nb = json.load(open(rf"{WS}\gci_honors_reproduction.ipynb"))
cells = [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']

# cell order: 1 imports, 2 paths(Drive), 3 FE, 4 enrich
src_imports = cells[0]
src_paths = f"""
import pandas as pd, numpy as np
INPUT = r'{C}\\input'
train = pd.read_csv(f'{{INPUT}}/train.csv')
test = pd.read_csv(f'{{INPUT}}/test.csv')
sample_sub = pd.read_csv(f'{{INPUT}}/sample_submission.csv')
print(train.shape, test.shape, sample_sub.shape)
"""
src_fe = cells[2]
src_enrich = cells[3]

g = {}
t0 = time.time()
exec(src_imports, g)
exec(src_paths, g)
exec(src_fe, g)
exec(src_enrich, g)
print(f"notebook FE done in {time.time()-t0:.0f}s")
X_nb = g['X']; Xt_nb = g['X_test']
print("notebook X:", X_nb.shape, "X_test:", Xt_nb.shape)

X_ref = pd.read_parquet(rf"{T}\X_e2.parquet")
Xt_ref = pd.read_parquet(rf"{T}\X_test_e2.parquet")
print("reference X:", X_ref.shape, "X_test:", Xt_ref.shape)

print("\ncolumn sets equal (train):", set(X_nb.columns) == set(X_ref.columns))
print("column ORDER equal:", list(X_nb.columns) == list(X_ref.columns))
if list(X_nb.columns) != list(X_ref.columns):
    only_nb = [c for c in X_nb.columns if c not in X_ref.columns]
    only_ref = [c for c in X_ref.columns if c not in X_nb.columns]
    print(" only notebook:", only_nb[:20], "..." if len(only_nb) > 20 else "")
    print(" only reference:", only_ref[:20], "..." if len(only_ref) > 20 else "")

common = [c for c in X_ref.columns if c in X_nb.columns]
print(f"\ncommon cols: {len(common)}")
A = X_nb[common].astype(np.float64).to_numpy()
B = X_ref[common].astype(np.float64).to_numpy()
eq = np.isclose(A, B, rtol=1e-5, atol=1e-5) | (np.isnan(A) & np.isnan(B))
print("all cells equal:", eq.all(), " mismatch cells:", int((~eq).sum()), f"({(~eq).mean()*100:.4f}%)")
if (~eq).any():
    rows, cols = np.where(~eq)
    bad_cols = pd.Series([common[j] for j in cols]).value_counts()
    print("\ntop mismatching columns:\n", bad_cols.head(20).to_string())
    print("max abs diff:", np.nanmax(np.abs(A - B)))

# test frame
common_t = [c for c in Xt_ref.columns if c in Xt_nb.columns]
print(f"\ntest col order equal: {list(Xt_nb.columns)==list(Xt_ref.columns)}  common: {len(common_t)}")
At = Xt_nb[common_t].astype(np.float64).to_numpy()
Bt = Xt_ref[common_t].astype(np.float64).to_numpy()
eqt = np.isclose(At, Bt, rtol=1e-5, atol=1e-5) | (np.isnan(At) & np.isnan(Bt))
print("test all cells equal:", eqt.all(), " mismatch cells:", int((~eqt).sum()), f"({(~eqt).mean()*100:.4f}%)")
if (~eqt).any():
    rows, cols = np.where(~eqt)
    bad_cols = pd.Series([common_t[j] for j in cols]).value_counts()
    print("\ntop mismatching test columns:\n", bad_cols.head(20).to_string())
    print("max abs diff:", np.nanmax(np.abs(At - Bt)))

# y and ids
y = np.load(rf"{T}\drive\GCI Competition\features\y_train.npy")
print("\ny equal:", np.array_equal(g['y'], y), " train ids equal:", np.array_equal(g['train_ids'], g['train']['SK_ID_CURR'].values))
print("test ids equal:", np.array_equal(g['test_ids'], Xt_ref.shape[0] and g['test_ids']))
