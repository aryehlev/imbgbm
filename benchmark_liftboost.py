#!/usr/bin/env python3
"""
LiftBoost vs CatBoost/LightGBM/XGBoost on a 99/1 imbalanced dataset.

Metrics focused on rare-event ranking quality:
  - PR-AUC          (area under precision-recall curve)
  - Precision@top1% (quality of the very top bucket)
  - Lift@top1%      (concentration of positives in top 1%)
  - Recall@FPR=1%   (recall when false positive rate = 1%)
  - AUC-ROC         (overall ranking)
  - LogLoss / ECE   (calibration)
"""

import os, sys, subprocess, time, textwrap
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score, log_loss, roc_curve
)
import catboost as cb
import lightgbm as lgb
import xgboost as xgb

SEED      = 42
BENCH_DIR = "/tmp/liftboost_bench"
IMBGBM    = "/home/user/imbgbm/target/release/imbgbm"
os.makedirs(BENCH_DIR, exist_ok=True)
rng = np.random.default_rng(SEED)

# ── 99/1 synthetic dataset ────────────────────────────────────────────────────
# Structured signal: 6 continuous features + 3 categorical (integer-encoded),
# cross-feature interactions, heteroscedastic noise.  Positive rate ~1%.

print("Generating 99/1 dataset …")
N = 80_000
np.random.seed(SEED)

# Continuous features
X_cont = rng.standard_normal((N, 6))
X_cont[:, 2] = np.abs(X_cont[:, 2])          # skewed
X_cont[:, 5] = X_cont[:, 0] * X_cont[:, 1]   # interaction

# Categorical features (label-encoded integers)
CARDS = [50, 30, 20]
X_cat = np.column_stack([rng.integers(0, c, N) for c in CARDS])

# Per-category effects
cat_effects = [rng.normal(0, 1.2, c) for c in CARDS]
cat_signal  = sum(cat_effects[j][X_cat[:, j]] for j in range(3))

# Latent score
lat  = 0.6*X_cont[:,0] - 0.5*X_cont[:,1] + 0.8*X_cont[:,2]
lat += 0.4*X_cont[:,3]*X_cont[:,4] + cat_signal
lat += rng.normal(0, 1.3, N)  # noise

# Calibrate intercept to hit ~1% positive rate
def sigmoid(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))
lo, hi = -30.0, 30.0
for _ in range(80):
    mid = (lo + hi) / 2.0
    if sigmoid(lat + mid).mean() > 0.01: hi = mid
    else:                                 lo = mid
intercept = (lo + hi) / 2.0
y = (rng.uniform(size=N) < sigmoid(lat + intercept)).astype(int)

X = np.c_[X_cont, X_cat.astype(np.float32)]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y)

print(f"  train n={len(y_tr)}  pos={y_tr.mean():.3%}")
print(f"  test  n={len(y_te)}  pos={y_te.mean():.3%}")
print(f"  total positives in test: {y_te.sum()}\n")
global_rate = y_tr.mean()

# Save CSVs for the imbgbm CLI
TR_CSV = f"{BENCH_DIR}/train.csv"
TE_CSV = f"{BENCH_DIR}/test.csv"
np.savetxt(TR_CSV, np.c_[X_tr, y_tr], delimiter=",", fmt="%.6f")
np.savetxt(TE_CSV, X_te,              delimiter=",", fmt="%.6f")

# ── Metric helpers ────────────────────────────────────────────────────────────
def precision_at_k(y_true, y_prob, k_frac=0.01):
    k = max(1, int(len(y_true) * k_frac))
    top_idx = np.argsort(-y_prob)[:k]
    return float(y_true[top_idx].mean())

def lift_at_k(y_true, y_prob, k_frac=0.01):
    return precision_at_k(y_true, y_prob, k_frac) / max(y_true.mean(), 1e-9)

def recall_at_fpr(y_true, y_prob, fpr_target=0.01):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    idx = max(np.searchsorted(fpr, fpr_target, "right") - 1, 0)
    return float(tpr[idx])

def ece(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    err  = 0.0
    for lo_, hi_ in zip(bins[:-1], bins[1:]):
        m = (y_prob >= lo_) & (y_prob < hi_)
        if m.sum() == 0: continue
        err += m.sum() * abs(y_true[m].mean() - y_prob[m].mean())
    return err / len(y_true)

rows = []
def evaluate(name, y_true, y_prob, elapsed):
    y_prob = np.clip(np.asarray(y_prob, float), 1e-7, 1 - 1e-7)
    out = dict(
        model   = name,
        time_s  = elapsed,
        auc     = roc_auc_score(y_true, y_prob),
        prauc   = average_precision_score(y_true, y_prob),
        p1pct   = precision_at_k(y_true, y_prob, 0.01),
        lift1   = lift_at_k(y_true, y_prob, 0.01),
        r_fpr1  = recall_at_fpr(y_true, y_prob, 0.01),
        logloss = log_loss(y_true, y_prob),
        ece     = ece(y_true, y_prob),
    )
    print(f"  AUC={out['auc']:.4f}  PR-AUC={out['prauc']:.4f}  "
          f"P@1%={out['p1pct']:.4f}  Lift@1%={out['lift1']:.2f}x  "
          f"R@FPR1%={out['r_fpr1']:.4f}  LogLoss={out['logloss']:.4f}  "
          f"ECE={out['ece']:.4f}  t={elapsed:.1f}s")
    rows.append(out)

def run_cli(args):
    r = subprocess.run([IMBGBM] + args, capture_output=True, text=True)
    if r.returncode != 0:
        print("CLI STDERR:", r.stderr[-2000:])
        sys.exit(1)
    return r

def read_probs(stdout):
    return np.array([float(x) for x in stdout.strip().splitlines()])

SEP = "=" * 70

# ── 1. CatBoost tuned ─────────────────────────────────────────────────────────
print(SEP); print("1. CatBoost (tuned, scale_pos_weight + ordered boosting)")
t0 = time.time()
cat_model = cb.CatBoostClassifier(
    iterations=500, learning_rate=0.05, depth=7,
    l2_leaf_reg=3.0, bagging_temperature=0.5,
    border_count=254, min_data_in_leaf=10,
    scale_pos_weight=int(1.0 / global_rate),
    random_seed=SEED, verbose=0,
)
cat_model.fit(X_tr, y_tr)
evaluate("CatBoost", y_te, cat_model.predict_proba(X_te)[:, 1], time.time() - t0)

# ── 2. LightGBM ───────────────────────────────────────────────────────────────
print(SEP); print("2. LightGBM (is_unbalance=True)")
t0 = time.time()
lgb_model = lgb.LGBMClassifier(
    n_estimators=500, learning_rate=0.05, max_depth=7,
    num_leaves=63, min_child_samples=20, subsample=0.8,
    colsample_bytree=0.8, is_unbalance=True,
    random_state=SEED, verbose=-1,
)
lgb_model.fit(X_tr, y_tr)
evaluate("LightGBM", y_te, lgb_model.predict_proba(X_te)[:, 1], time.time() - t0)

# ── 3. XGBoost ────────────────────────────────────────────────────────────────
print(SEP); print("3. XGBoost (scale_pos_weight)")
t0 = time.time()
xgb_model = xgb.XGBClassifier(
    n_estimators=500, learning_rate=0.05, max_depth=7,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=int(1.0 / global_rate),
    eval_metric="logloss", use_label_encoder=False,
    random_state=SEED, verbosity=0,
)
xgb_model.fit(X_tr, y_tr)
evaluate("XGBoost", y_te, xgb_model.predict_proba(X_te)[:, 1], time.time() - t0)

# ── 4. imbgbm standard (BCE + adaptive sampler) ───────────────────────────────
print(SEP); print("4. imbgbm Standard (BCE + adaptive sampler)")
t0 = time.time()
run_cli(["train", "--input", TR_CSV, "--output", f"{BENCH_DIR}/m_std.json",
         "--loss", "bce", "--n-rounds", "500", "--learning-rate", "0.05",
         "--max-depth", "7", "--sampler", "adaptive", "--subsample", "0.5",
         "--col-subsample", "0.8", "--calibrate", "--early-stopping-rounds", "30"])
r = run_cli(["predict", "--input", TE_CSV, "--model", f"{BENCH_DIR}/m_std.json",
             "--calibrated"])
evaluate("imbgbm Standard", y_te, read_probs(r.stdout), time.time() - t0)

# ── 5. imbgbm LiftBoost: positive-mass splitter only ─────────────────────────
print(SEP); print("5. imbgbm LiftBoost (positive-mass splitter, α=1.0)")
t0 = time.time()
run_cli(["train", "--input", TR_CSV, "--output", f"{BENCH_DIR}/m_pm.json",
         "--loss", "bce", "--n-rounds", "500", "--learning-rate", "0.05",
         "--max-depth", "7", "--sampler", "adaptive", "--subsample", "0.5",
         "--col-subsample", "0.8", "--calibrate", "--early-stopping-rounds", "30",
         "--splitter", "positive-mass", "--pm-alpha", "1.0", "--pm-min-pos-leaf", "10"])
r = run_cli(["predict", "--input", TE_CSV, "--model", f"{BENCH_DIR}/m_pm.json",
             "--calibrated"])
evaluate("imbgbm LiftBoost (PM)", y_te, read_probs(r.stdout), time.time() - t0)

# ── 6. imbgbm LiftBoost: positive-mass + tail weighting ─────────────────────
print(SEP); print("6. imbgbm LiftBoost (PM splitter + top-1% tail weighting from round 200)")
t0 = time.time()
run_cli(["train", "--input", TR_CSV, "--output", f"{BENCH_DIR}/m_liftboost.json",
         "--loss", "bce", "--n-rounds", "500", "--learning-rate", "0.05",
         "--max-depth", "7", "--sampler", "adaptive", "--subsample", "0.5",
         "--col-subsample", "0.8", "--calibrate", "--early-stopping-rounds", "30",
         "--splitter", "positive-mass", "--pm-alpha", "1.0", "--pm-min-pos-leaf", "10",
         "--tail-weight", "--tail-top-rate", "0.01", "--tail-start-round", "200"])
r = run_cli(["predict", "--input", TE_CSV, "--model", f"{BENCH_DIR}/m_liftboost.json",
             "--calibrated"])
evaluate("imbgbm LiftBoost (PM+Tail)", y_te, read_probs(r.stdout), time.time() - t0)

# ── 7. imbgbm LiftBoost + raw-isotonic calibration ──────────────────────────
print(SEP); print("7. imbgbm LiftBoost (PM+Tail) + raw-isotonic calibration")
t0 = time.time()
run_cli(["train", "--input", TR_CSV, "--output", f"{BENCH_DIR}/m_liftboost_iso.json",
         "--loss", "bce", "--n-rounds", "350", "--learning-rate", "0.05",
         "--max-depth", "7", "--sampler", "adaptive", "--subsample", "0.5",
         "--col-subsample", "0.8", "--calibrate", "--raw-isotonic",
         "--early-stopping-rounds", "0",
         "--splitter", "positive-mass", "--pm-alpha", "1.0", "--pm-min-pos-leaf", "10",
         "--tail-weight", "--tail-top-rate", "0.01", "--tail-start-round", "200"])
r = run_cli(["predict", "--input", TE_CSV, "--model", f"{BENCH_DIR}/m_liftboost_iso.json",
             "--raw-isotonic"])
evaluate("imbgbm LiftBoost (PM+Tail+Iso)", y_te, read_probs(r.stdout), time.time() - t0)

# ── Summary table ─────────────────────────────────────────────────────────────
print()
print("=" * 95)
print("RESULTS — 99/1 Rare-Event Classification Benchmark")
print("=" * 95)

METRICS = [
    ("auc",     "AUC-ROC",    True),
    ("prauc",   "PR-AUC",     True),
    ("p1pct",   "P@top1%",    True),
    ("lift1",   "Lift@top1%", True),
    ("r_fpr1",  "R@FPR=1%",   True),
    ("logloss", "LogLoss",    False),
    ("ece",     "ECE",        False),
]

hdr = f"{'Model':<38}" + "".join(f"{lab:>12}" for _, lab, _ in METRICS) + f"{'Time(s)':>8}"
print(hdr)
print("-" * len(hdr))
for row in rows:
    line = f"{row['model']:<38}"
    for k, _, _ in METRICS:
        line += f"{row[k]:12.4f}"
    line += f"{row['time_s']:8.0f}"
    print(line)

# Per-metric winner annotation
print()
cb_row = rows[0]
print(f"\n{'Metric':<12} {'CatBoost':>10}", end="")
for r in rows[1:]:
    print(f"  {r['model'][:16]:>16}", end="")
print()
print("-" * 100)

for k, label, higher_better in METRICS:
    all_vals = [r[k] for r in rows]
    best = max(all_vals) if higher_better else min(all_vals)
    print(f"  {label:<10} {cb_row[k]:9.4f}", end="")
    for r in rows[1:]:
        v    = r[k]
        mark = "◄" if abs(v - best) < 1e-6 else " "
        beats = (v > cb_row[k] + 1e-6) if higher_better else (v < cb_row[k] - 1e-6)
        flag  = "★" if beats else " "
        print(f"   {flag}{v:9.4f}{mark}", end="")
    print()

print()
print("★ = beats CatBoost  ◄ = best overall")
print()

for r in rows[1:]:
    n_win = sum(
        1 for k, _, hb in METRICS
        if ((r[k] > cb_row[k] + 1e-6) if hb else (r[k] < cb_row[k] - 1e-6))
    )
    details = []
    for k, lab, hb in METRICS:
        delta = (r[k] - cb_row[k]) / max(abs(cb_row[k]), 1e-9) * 100
        if not hb:
            delta = -delta
        sign = "+" if delta > 0 else ""
        details.append(f"{lab}:{sign}{delta:.1f}%")
    print(f"  {r['model']:<40} {n_win}/{len(METRICS)} wins  [{', '.join(details)}]")

print()
print(f"Global positive rate: {global_rate:.3%}")
print(f"Test set positives: {y_te.sum()} / {len(y_te)}")
print(f"LiftBoost Lift@top1% = {rows[-1]['lift1']:.2f}x  (CatBoost: {cb_row['lift1']:.2f}x)")
