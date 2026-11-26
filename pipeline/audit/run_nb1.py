import sys, os, json, time
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import pandas as pd

WS = r"C:\Users\amanuel\OneDrive\Documents\deepseek-harness\default-workspace"
C = rf"{WS}\competition_unzipped\04. Competition (due Nov 20)\Competition"
OUT = r"C:\Users\amanuel\AppData\Local\Temp\opencode\nb1_run"
os.makedirs(OUT, exist_ok=True)

nb = json.load(open(rf"{WS}\01_feature_engineering.ipynb", encoding='utf-8'))
codes = [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']

g = {}
# cell 0 is imports; cell 1 is Drive mount + WORK_DIR -> replace
exec(codes[0], g)
mount_repl = f"""
import os
WORK_DIR = r'{OUT}'
os.makedirs(WORK_DIR, exist_ok=True)
for sub in ['features','models','optuna_studies','submissions']:
    os.makedirs(f'{{WORK_DIR}}/{{sub}}', exist_ok=True)
print('WORK_DIR', WORK_DIR)
"""
exec(mount_repl, g)

# cell 2 loads data from WORK_DIR/input; point it at the competition input via symlink-ish copy of paths
# Re-implement cell 3 with local paths
g['INPUT'] = rf"{C}\input"
data_src = f"""
train = pd.read_csv(r'{C}\\input\\train.csv')
test = pd.read_csv(r'{C}\\input\\test.csv')
sample_sub = pd.read_csv(r'{C}\\input\\sample_submission.csv')
print(train.shape, test.shape, sample_sub.shape)
"""
exec(data_src, g)
print("imports + data ok", flush=True)

for i, src in enumerate(codes[3:], start=3):
    t0 = time.time()
    print(f"--- running cell {i} ---", flush=True)
    exec(src, g)
    print(f"    ok ({time.time()-t0:.1f}s)", flush=True)

print("\n--- comparing to reference ---")
ref_f = rf"{C.replace('Competition','')}"  # unused
REF = r"C:\Users\amanuel\AppData\Local\Temp\opencode\drive\GCI Competition\features"
new_tr = pd.read_parquet(rf"{OUT}\features\train_engineered.parquet")
new_te = pd.read_parquet(rf"{OUT}\features\test_engineered.parquet")
ref_tr = pd.read_parquet(rf"{REF}\train_engineered.parquet")
ref_te = pd.read_parquet(rf"{REF}\test_engineered.parquet")
print("shapes new/ref train:", new_tr.shape, ref_tr.shape)
print("shapes new/ref test :", new_te.shape, ref_te.shape)
print("cols equal/order:", list(new_tr.columns) == list(ref_tr.columns))
common = [c for c in ref_tr.columns if c in new_tr.columns]
A = new_tr[common].astype(np.float64).to_numpy(); B = ref_tr[common].astype(np.float64).to_numpy()
eq = np.isclose(A, B, rtol=1e-5, atol=1e-5) | (np.isnan(A) & np.isnan(B))
print("train all equal:", eq.all(), "mismatch:", int((~eq).sum()), f"({(~eq).mean()*100:.4f}%)")
if (~eq).any():
    rows, cols = np.where(~eq)
    print("bad cols:", pd.Series([common[j] for j in cols]).value_counts().head(10).to_string())
    print("max diff:", np.nanmax(np.abs(A-B)))
common_t = [c for c in ref_te.columns if c in new_te.columns]
At = new_te[common_t].astype(np.float64).to_numpy(); Bt = ref_te[common_t].astype(np.float64).to_numpy()
eqt = np.isclose(At, Bt, rtol=1e-5, atol=1e-5) | (np.isnan(At) & np.isnan(Bt))
print("test all equal:", eqt.all(), "mismatch:", int((~eqt).sum()), f"({(~eqt).mean()*100:.4f}%)")

new_fc = json.load(open(rf"{OUT}\features\feature_cols.json"))
ref_fc = json.load(open(rf"{REF}\feature_cols.json"))
print("feature_cols equal:", new_fc == ref_fc, "len", len(new_fc), len(ref_fc))
print("y equal:", np.array_equal(np.load(rf"{OUT}\features\y_train.npy"), np.load(rf"{REF}\y_train.npy")))
