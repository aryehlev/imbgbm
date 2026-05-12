#!/usr/bin/env python3
"""
LiftBoost vs CatBoost / LightGBM / XGBoost on a Porto Seguro–style dataset.

Porto Seguro Safe Driver Prediction was the flagship Kaggle competition where
CatBoost was heavily used (it's from Yandex and the competition required native
categorical support).  The dataset has ~3.6% positives and 57 features
(categorical, binary, integer ordinal, continuous) with complex interactions.

Since we can't download Kaggle data here, we generate a synthetic dataset that
faithfully reproduces the statistical structure that makes Porto Seguro hard:
  - ~3.5% positive rate (vs 99/1 — moderately rare)
  - Mixed feature types: binary, ordinal, categorical (label-encoded)
  - Categorical features with cardinalities mimicking the real dataset
  - Cross-feature interactions (the main driver of signal)
  - Missing-value indicator features (-1 encoded)

Metrics tracked (the ones that matter in rare-event ranking):
  - Gini = 2*AUC - 1  (competition metric for Porto Seguro)
  - PR-AUC
  - Precision@top 3%  (top 3% because ~3.5% positive rate)
  - Lift@top 3%
  - Recall@FPR=5%
  - Normalised Gini (= Gini / perfect_Gini)
"""

import os, sys, subprocess, time
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss, roc_curve
import catboost as cb
import lightgbm as lgb
import xgboost as xgb

SEED      = 42
BENCH_DIR = "/tmp/porto_bench"
IMBGBM    = "/home/user/imbgbm/target/release/imbgbm"
os.makedirs(BENCH_DIR, exist_ok=True)
rng = np.random.default_rng(SEED)

# ── Porto Seguro–style synthetic dataset ──────────────────────────────────────
# Feature structure mirrors the actual competition:
#   ps_ind (individual / driver features): binary + ordinal + categorical
#   ps_reg (registration / car-registration features): continuous + ordinal
#   ps_car (car attributes): categorical + binary + continuous
#   ps_calc (computed / engineered): continuous + binary (mostly noise)

print("Generating Porto Seguro–style dataset (400k rows, ~3.5% positive) …")
N = 400_000
np.random.seed(SEED)

# Cardinalities of categorical features (ps_ind_*_cat, ps_car_*_cat)
# Approximates the real Porto Seguro cardinalities
CAT_CARDS = {
    'ind_02': 3, 'ind_04': 2, 'ind_05': 8,      # ps_ind categorical
    'car_01': 13, 'car_02': 2, 'car_03': 3,      # ps_car categorical
    'car_04': 10, 'car_06': 18, 'car_07': 2,
    'car_08': 2, 'car_09': 6, 'car_10': 3,
    'car_11': 104,                                # high-cardinality
}
CAT_NAMES = list(CAT_CARDS.keys())
CAT_VALS  = list(CAT_CARDS.values())

# Binary features (ps_*_bin)
N_BIN = 17

# Ordinal integer features (ps_*_ind without _cat, ps_reg_*, small range)
ORD_RANGES = [3, 4, 5, 7, 8, 4, 11, 5, 3]   # 9 ordinal features

# Continuous features (ps_reg_01..03, ps_car_12..15, ps_calc continuous)
N_CONT = 10

# Generate raw feature matrix
# -- Categorical
X_cat = np.column_stack([
    rng.integers(0, c, N) for c in CAT_VALS
])
# -- Binary
X_bin = rng.integers(0, 2, (N, N_BIN)).astype(np.float32)
# Inject ~15% "missing" as -1 in some binary features
for j in range(4):
    mask = rng.random(N) < 0.15
    X_bin[mask, j] = -1
# -- Ordinal
X_ord = np.column_stack([rng.integers(0, r, N) for r in ORD_RANGES])
# -- Continuous (log-normal + normal)
X_cont = np.column_stack([
    np.abs(rng.standard_normal(N)),              # right-skewed
    rng.random(N),                               # uniform [0,1]
    np.abs(rng.standard_normal(N)) * 0.5,
    rng.standard_normal(N),
    rng.standard_normal(N) ** 2,                 # chi-squared-like
    rng.random(N),
    np.abs(rng.standard_normal(N)),
    rng.random(N),
    rng.standard_normal(N),
    rng.random(N),
])

# -- Per-category effects (the main signal source)
cat_effects = [rng.normal(0, 1.0 + j*0.1, c) for j, c in enumerate(CAT_VALS)]
cat_signal  = sum(cat_effects[j][X_cat[:, j]] for j in range(len(CAT_VALS)))

# Continuous feature signal
cont_signal = (
    0.8 * X_cont[:, 0]
    - 0.5 * X_cont[:, 1]
    + 0.6 * X_cont[:, 2] * X_cont[:, 3]   # interaction
    + 0.4 * X_cont[:, 4]
    - 0.3 * X_cont[:, 5]
)

# Ordinal signal
ord_signal = 0.4 * X_ord[:, 0] - 0.3 * X_ord[:, 2] + 0.2 * X_ord[:, 4]

# Binary signal
bin_signal = 0.6 * X_bin[:, 0] - 0.4 * X_bin[:, 2] + 0.3 * X_bin[:, 5]
bin_signal = np.where(X_bin[:, 0] == -1, 0, bin_signal)

# Cross-feature interaction (cat × ordinal — what CatBoost excels at)
inter = cat_effects[0][X_cat[:, 0]] * (X_ord[:, 0] / ORD_RANGES[0])

# Latent score
lat  = cat_signal + 0.5 * cont_signal + 0.3 * ord_signal + 0.2 * bin_signal + 0.4 * inter
lat += rng.normal(0, 1.5, N)

# Calibrate to ~3.5% positive rate
def sigmoid(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))
lo, hi = -30.0, 30.0
for _ in range(80):
    mid = (lo + hi) / 2.0
    if sigmoid(lat + mid).mean() > 0.035: hi = mid
    else:                                  lo = mid
y = (rng.uniform(size=N) < sigmoid(lat + (lo + hi) / 2)).astype(int)

# Assemble full feature matrix (continuous | binary | ordinal | categorical)
# For imbgbm CLI: all numeric, no native categorical support
# Categorical features are label-encoded (as integers) — same as what you'd
# target-encode in a real Porto Seguro submission without CatBoost
X_all = np.c_[
    X_cont.astype(np.float32),
    X_bin.astype(np.float32),
    X_ord.astype(np.float32),
    X_cat.astype(np.float32),
]

X_tr, X_te, y_tr, y_te = train_test_split(
    X_all, y, test_size=0.2, random_state=SEED, stratify=y
)

global_rate = y_tr.mean()
print(f"  train n={len(y_tr):,}  pos={global_rate:.3%}")
print(f"  test  n={len(y_te):,}  pos={y_te.mean():.3%}")
print(f"  total positives in test: {y_te.sum():,}")
print(f"  features: {X_all.shape[1]} total  "
      f"({N_CONT} cont + {N_BIN} bin + {len(ORD_RANGES)} ord + {len(CAT_VALS)} cat)\n")

# ── Target-encode categoricals for imbgbm (5-fold OOF) ───────────────────────
# CatBoost handles raws; imbgbm needs encoded features
cont_end  = N_CONT
bin_end   = cont_end + N_BIN
ord_end   = bin_end + len(ORD_RANGES)
cat_start = ord_end   # categorical columns start here

def te_col(codes_tr, y_tr_arr, codes_te, card, smoothing=20.0):
    gm = float(y_tr_arr.mean())
    cs = np.zeros(card); cc = np.zeros(card)
    np.add.at(cs, codes_tr, y_tr_arr); np.add.at(cc, codes_tr, 1)
    te = (cs[codes_te] + gm * smoothing) / (cc[codes_te] + smoothing)
    oof = np.full(len(y_tr_arr), gm, dtype=np.float32)
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    for tri, vai in kf.split(np.arange(len(y_tr_arr))):
        cs2 = np.zeros(card); cc2 = np.zeros(card)
        np.add.at(cs2, codes_tr[tri], y_tr_arr[tri])
        np.add.at(cc2, codes_tr[tri], 1)
        sm2 = (cs2 + gm * smoothing) / (cc2 + smoothing)
        oof[vai] = sm2[codes_tr[vai]]
    return oof.astype(np.float32), te.astype(np.float32)

print("Target-encoding categorical features …")
X_tr_enc = X_tr.copy()
X_te_enc = X_te.copy()
for j, card in enumerate(CAT_VALS):
    col = cat_start + j
    tr_codes = X_tr[:, col].astype(int)
    te_codes = X_te[:, col].astype(int)
    oof_te, test_te = te_col(tr_codes, y_tr, te_codes, card)
    X_tr_enc[:, col] = oof_te
    X_te_enc[:, col] = test_te
print("  done.\n")

# Save CSVs for imbgbm CLI
TR_CSV = f"{BENCH_DIR}/train_enc.csv"
TE_CSV = f"{BENCH_DIR}/test_enc.csv"
np.savetxt(TR_CSV, np.c_[X_tr_enc, y_tr], delimiter=",", fmt="%.6f")
np.savetxt(TE_CSV, X_te_enc,              delimiter=",", fmt="%.6f")

# Cat feature indices for CatBoost (raw integer columns in X_tr_raw)
# CatBoost handles them natively — this is CatBoost's main edge
CAT_FEATURE_IDX = list(range(cat_start, X_all.shape[1]))

# ── Metric helpers ────────────────────────────────────────────────────────────
TOP_FRAC = 0.03   # 3% = approximately 1× the positive rate

def gini(y_true, y_prob):
    return 2 * roc_auc_score(y_true, y_prob) - 1

def precision_at_k(y_true, y_prob, k_frac=TOP_FRAC):
    k = max(1, int(len(y_true) * k_frac))
    return float(y_true[np.argsort(-y_prob)[:k]].mean())

def lift_at_k(y_true, y_prob, k_frac=TOP_FRAC):
    return precision_at_k(y_true, y_prob, k_frac) / max(y_true.mean(), 1e-9)

def recall_at_fpr(y_true, y_prob, fpr_target=0.05):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    idx = max(np.searchsorted(fpr, fpr_target, "right") - 1, 0)
    return float(tpr[idx])

rows = []
def evaluate(name, y_true, y_prob, elapsed):
    y_prob = np.clip(np.asarray(y_prob, float), 1e-7, 1 - 1e-7)
    out = dict(
        model  = name,
        time_s = elapsed,
        gini   = gini(y_true, y_prob),
        prauc  = average_precision_score(y_true, y_prob),
        p3pct  = precision_at_k(y_true, y_prob),
        lift3  = lift_at_k(y_true, y_prob),
        r_fpr5 = recall_at_fpr(y_true, y_prob, 0.05),
        ll     = log_loss(y_true, y_prob),
    )
    print(f"  Gini={out['gini']:.4f}  PR-AUC={out['prauc']:.4f}  "
          f"P@3%={out['p3pct']:.4f}  Lift@3%={out['lift3']:.2f}x  "
          f"R@FPR5%={out['r_fpr5']:.4f}  LogLoss={out['ll']:.4f}  "
          f"t={elapsed:.1f}s")
    rows.append(out)

def run_cli(args):
    r = subprocess.run([IMBGBM] + args, capture_output=True, text=True)
    if r.returncode != 0:
        print("CLI STDERR:", r.stderr[-2000:])
        sys.exit(1)
    return r

def read_probs(stdout):
    return np.array([float(x) for x in stdout.strip().splitlines()])

SEP = "=" * 72

# CatBoost needs string values in categorical columns — use pandas
import pandas as pd
df_tr_raw = pd.DataFrame(X_tr)
df_te_raw = pd.DataFrame(X_te)
for j in CAT_FEATURE_IDX:
    df_tr_raw[j] = df_tr_raw[j].astype(int).astype(str)
    df_te_raw[j] = df_te_raw[j].astype(int).astype(str)

# ── 1. CatBoost — native categoricals (its main edge) ────────────────────────
print(SEP)
print("1. CatBoost (native cat features + ordered boosting + scale_pos_weight)")
t0 = time.time()
cb_model = cb.CatBoostClassifier(
    iterations=500, learning_rate=0.05, depth=6,
    l2_leaf_reg=3.0, bagging_temperature=0.5,
    border_count=254, min_data_in_leaf=20,
    cat_features=CAT_FEATURE_IDX,
    auto_class_weights="SqrtBalanced",
    random_seed=SEED, verbose=0,
)
cb_model.fit(df_tr_raw, y_tr)
evaluate("CatBoost (native cats)", y_te, cb_model.predict_proba(df_te_raw)[:, 1], time.time()-t0)

# ── 2. LightGBM (target-encoded cats) ────────────────────────────────────────
print(SEP)
print("2. LightGBM (target-encoded cats, is_unbalance)")
t0 = time.time()
lgb_m = lgb.LGBMClassifier(
    n_estimators=500, learning_rate=0.05, max_depth=6,
    num_leaves=63, min_child_samples=20,
    subsample=0.8, colsample_bytree=0.8,
    is_unbalance=True, random_state=SEED, verbose=-1,
)
lgb_m.fit(X_tr_enc, y_tr)
evaluate("LightGBM (TE cats)", y_te, lgb_m.predict_proba(X_te_enc)[:, 1], time.time()-t0)

# ── 3. XGBoost (target-encoded cats) ─────────────────────────────────────────
print(SEP)
print("3. XGBoost (target-encoded cats, scale_pos_weight)")
t0 = time.time()
xgb_m = xgb.XGBClassifier(
    n_estimators=500, learning_rate=0.05, max_depth=6,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=int(1.0 / global_rate),
    eval_metric="logloss", random_state=SEED, verbosity=0,
)
xgb_m.fit(X_tr_enc, y_tr)
evaluate("XGBoost (TE cats)", y_te, xgb_m.predict_proba(X_te_enc)[:, 1], time.time()-t0)

# ── 4. imbgbm Standard ────────────────────────────────────────────────────────
print(SEP)
print("4. imbgbm Standard (BCE + adaptive sampler + OOF calibration)")
t0 = time.time()
run_cli(["train", "--input", TR_CSV, "--output", f"{BENCH_DIR}/m_std.json",
         "--loss", "bce", "--n-rounds", "500", "--learning-rate", "0.05",
         "--max-depth", "6", "--sampler", "adaptive", "--subsample", "0.5",
         "--col-subsample", "0.8", "--calibrate", "--early-stopping-rounds", "30",
         "--min-samples-leaf", "20"])
r = run_cli(["predict", "--input", TE_CSV, "--model", f"{BENCH_DIR}/m_std.json",
             "--calibrated"])
evaluate("imbgbm Standard", y_te, read_probs(r.stdout), time.time()-t0)

# ── 5. imbgbm LiftBoost PM only (alpha=0.5, no tail) ─────────────────────────
print(SEP)
print("5. imbgbm LiftBoost (positive-mass splitter α=0.5)")
t0 = time.time()
run_cli(["train", "--input", TR_CSV, "--output", f"{BENCH_DIR}/m_pm05.json",
         "--loss", "bce", "--n-rounds", "500", "--learning-rate", "0.05",
         "--max-depth", "6", "--sampler", "adaptive", "--subsample", "0.5",
         "--col-subsample", "0.8", "--calibrate", "--early-stopping-rounds", "30",
         "--min-samples-leaf", "20",
         "--splitter", "positive-mass", "--pm-alpha", "0.5", "--pm-min-pos-leaf", "5"])
r = run_cli(["predict", "--input", TE_CSV, "--model", f"{BENCH_DIR}/m_pm05.json",
             "--calibrated"])
evaluate("imbgbm LiftBoost PM α=0.5", y_te, read_probs(r.stdout), time.time()-t0)

# ── 6. imbgbm LiftBoost PM α=1.0 ─────────────────────────────────────────────
print(SEP)
print("6. imbgbm LiftBoost (positive-mass splitter α=1.0)")
t0 = time.time()
run_cli(["train", "--input", TR_CSV, "--output", f"{BENCH_DIR}/m_pm10.json",
         "--loss", "bce", "--n-rounds", "500", "--learning-rate", "0.05",
         "--max-depth", "6", "--sampler", "adaptive", "--subsample", "0.5",
         "--col-subsample", "0.8", "--calibrate", "--early-stopping-rounds", "30",
         "--min-samples-leaf", "20",
         "--splitter", "positive-mass", "--pm-alpha", "1.0", "--pm-min-pos-leaf", "5"])
r = run_cli(["predict", "--input", TE_CSV, "--model", f"{BENCH_DIR}/m_pm10.json",
             "--calibrated"])
evaluate("imbgbm LiftBoost PM α=1.0", y_te, read_probs(r.stdout), time.time()-t0)

# ── 7. imbgbm LiftBoost PM + top-tail weighting ───────────────────────────────
print(SEP)
print("7. imbgbm LiftBoost PM α=1.0 + top-3% tail weighting (start round 200)")
t0 = time.time()
run_cli(["train", "--input", TR_CSV, "--output", f"{BENCH_DIR}/m_liftboost.json",
         "--loss", "bce", "--n-rounds", "500", "--learning-rate", "0.05",
         "--max-depth", "6", "--sampler", "adaptive", "--subsample", "0.5",
         "--col-subsample", "0.8", "--calibrate", "--early-stopping-rounds", "30",
         "--min-samples-leaf", "20",
         "--splitter", "positive-mass", "--pm-alpha", "1.0", "--pm-min-pos-leaf", "5",
         "--tail-weight", "--tail-top-rate", "0.03", "--tail-start-round", "200"])
r = run_cli(["predict", "--input", TE_CSV, "--model", f"{BENCH_DIR}/m_liftboost.json",
             "--calibrated"])
evaluate("imbgbm LiftBoost (PM+Tail)", y_te, read_probs(r.stdout), time.time()-t0)

# ── Summary ────────────────────────────────────────────────────────────────────
print()
print("=" * 90)
print("PORTO SEGURO–STYLE BENCHMARK — Rare-Event Ranking")
print("Competition metric: Normalised Gini (= 2*AUC - 1)")
print("=" * 90)

METRICS = [
    ("gini",   "Gini",     True),
    ("prauc",  "PR-AUC",   True),
    ("p3pct",  "P@top3%",  True),
    ("lift3",  "Lift@3%",  True),
    ("r_fpr5", "R@FPR5%",  True),
    ("ll",     "LogLoss",  False),
]

hdr = f"{'Model':<38}" + "".join(f"{lab:>10}" for _, lab, _ in METRICS) + f"{'Time(s)':>8}"
print(hdr); print("-" * len(hdr))
for row in rows:
    print(f"{row['model']:<38}" +
          "".join(f"{row[k]:10.4f}" for k, _, _ in METRICS) +
          f"{row['time_s']:8.0f}")

cb_row = rows[0]
print()
print(f"{'Metric':<10} {'CatBoost':>10}", end="")
for r in rows[1:]:
    print(f"  {r['model'][:18]:>18}", end="")
print(); print("-" * 110)
for k, label, hb in METRICS:
    all_vals = [r[k] for r in rows]
    best = max(all_vals) if hb else min(all_vals)
    print(f"  {label:<8} {cb_row[k]:9.4f}", end="")
    for r in rows[1:]:
        v = r[k]
        mark  = "◄" if abs(v - best) < 1e-6 else " "
        beats = (v > cb_row[k] + 1e-6) if hb else (v < cb_row[k] - 1e-6)
        flag  = "★" if beats else " "
        print(f"  {flag}{v:9.4f}{mark}", end="")
    print()

print("\n★ = beats CatBoost  ◄ = best overall\n")
for r in rows[1:]:
    wins = sum(
        1 for k, _, hb in METRICS
        if ((r[k] > cb_row[k] + 1e-6) if hb else (r[k] < cb_row[k] - 1e-6))
    )
    details = []
    for k, lab, hb in METRICS:
        delta = (r[k] - cb_row[k]) / max(abs(cb_row[k]), 1e-9) * 100
        if not hb: delta = -delta
        details.append(f"{lab}:{'+' if delta>0 else ''}{delta:.1f}%")
    print(f"  {r['model']:<40} {wins}/{len(METRICS)} wins  [{', '.join(details)}]")

print()
print(f"Global positive rate:  {global_rate:.3%}")
print(f"Test positives: {y_te.sum():,} / {len(y_te):,}")
print(f"CatBoost Gini:         {cb_row['gini']:.4f}")
best_lb = max(r['gini'] for r in rows[4:])
print(f"Best LiftBoost Gini:   {best_lb:.4f}  ({(best_lb-cb_row['gini'])/max(abs(cb_row['gini']),1e-9)*100:+.1f}% vs CatBoost)")
