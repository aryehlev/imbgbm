#!/usr/bin/env python3
"""
Benchmark imbgbm vs CatBoost on a synthetic Amazon-employee-access-like dataset.

Dataset mimics the CatBoost/Kaggle Amazon Access dataset:
  - ~33 k training rows, ~6 k test rows
  - 9 high-cardinality categorical features (RESOURCE, MGR_ID, ROLE_* hierarchy)
  - ~6 % positive rate (access approved)
  - Latent score drawn from a realistic hierarchical model

CatBoost:  native categorical handling (target statistics, CatBoost ordering).
imbgbm:    smoothed target encoding applied before fitting (no label leakage:
           encoding computed on train folds, applied to holdout / test).

Four conditions:
  1. CatBoost  — default
  2. CatBoost  — auto_class_weights=Balanced
  3. imbgbm    — BCE, uniform subsample=0.8
  4. imbgbm    — Focal (γ=2, α=0.25) + OOF calibration (Bayesian-smoothed)
"""

import os, sys, subprocess, time
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss, brier_score_loss
import catboost as cb

SEED = 42
rng  = np.random.default_rng(SEED)

IMBGBM    = "/home/user/imbgbm/target/release/imbgbm"
BENCH_DIR = "/tmp/imbgbm_amazon_bench"
os.makedirs(BENCH_DIR, exist_ok=True)

# ── Synthetic Amazon-like dataset ─────────────────────────────────────────────
print("Generating synthetic Amazon-employee-access dataset …")

N     = 39_000
# Cardinalities matching the real Amazon dataset
CARDS = [7518, 4243, 128, 177, 449, 343, 2358, 67, 343]
NAMES = ["RESOURCE", "MGR_ID", "ROLE_ROLLUP_1", "ROLE_ROLLUP_2",
         "ROLE_DEPTNAME", "ROLE_TITLE", "ROLE_FAMILY_DESC",
         "ROLE_FAMILY", "ROLE_CODE"]

# Assign each category a fixed latent effect drawn at construction time.
# The effect magnitude controls feature importance.
np.random.seed(SEED)
effects = []
scales  = [1.2, 0.9, 1.5, 1.3, 1.1, 1.0, 0.8, 1.4, 1.0]  # per-feature signal
for c, s in zip(CARDS, scales):
    e = np.random.normal(0, s, c)  # each category ID gets a fixed effect
    effects.append(e)

# Sample all 9 features uniformly
X_raw = np.column_stack([rng.integers(0, c, size=N) for c in CARDS])

# Latent score = sum of per-feature category effects + noise
lat = sum(effects[j][X_raw[:, j]] for j in range(len(CARDS)))
lat += rng.normal(0, 1.5, N)   # noise term

def sigmoid(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

# Find intercept so that mean(sigmoid(lat + b0)) ≈ 6% (binary search)
target_pos = 0.06
lo, hi = -30.0, 30.0
for _ in range(80):
    mid = (lo + hi) / 2.0
    if sigmoid(lat + mid).mean() > target_pos:
        hi = mid
    else:
        lo = mid
b0 = (lo + hi) / 2.0

p  = sigmoid(lat + b0)
y  = (rng.uniform(size=N) < p).astype(int)

print(f"  Positive rate : {y.mean():.2%}")

X_train_cat, X_test_cat, y_train, y_test = train_test_split(
    X_raw, y, test_size=6_000, random_state=SEED, stratify=y
)
print(f"  Train : {X_train_cat.shape}  pos={y_train.mean():.2%}")
print(f"  Test  : {X_test_cat.shape}   pos={y_test.mean():.2%}\n")

cat_features = list(range(9))

# ── Smoothed target encoding for imbgbm ──────────────────────────────────────
# 5-fold OOF target encoding prevents leakage on train; test uses global stats.

def target_encode(X_tr, y_tr, X_te, cards, smoothing=10.0):
    """Smoothed mean-target encoding.

    Train uses 5-fold OOF to prevent leakage.
    Test uses full-train statistics.
    Returns float32 arrays of the same shape.
    """
    n_tr, n_feat = X_tr.shape
    X_tr_enc = np.zeros_like(X_tr, dtype=np.float32)
    X_te_enc = np.zeros_like(X_te, dtype=np.float32)
    global_mean = float(y_tr.mean())

    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

    for j in range(n_feat):
        # Full-train encoding for test set
        cat_mean = np.zeros(cards[j])
        cat_cnt  = np.zeros(cards[j])
        np.add.at(cat_mean, X_tr[:, j], y_tr)
        np.add.at(cat_cnt,  X_tr[:, j], 1)
        smoothed = (cat_mean + global_mean * smoothing) / (cat_cnt + smoothing)
        X_te_enc[:, j] = smoothed[X_te[:, j]]

        # OOF encoding for train set
        oof = np.full(n_tr, global_mean, dtype=np.float32)
        for tr_idx, val_idx in kf.split(X_tr):
            cm  = np.zeros(cards[j])
            cc  = np.zeros(cards[j])
            np.add.at(cm, X_tr[tr_idx, j], y_tr[tr_idx])
            np.add.at(cc, X_tr[tr_idx, j], 1)
            sm = (cm + global_mean * smoothing) / (cc + smoothing)
            oof[val_idx] = sm[X_tr[val_idx, j]]
        X_tr_enc[:, j] = oof

    return X_tr_enc, X_te_enc

print("Computing OOF target encoding for imbgbm …")
t_enc0 = time.time()
X_train_enc, X_test_enc = target_encode(
    X_train_cat, y_train, X_test_cat, CARDS
)
t_enc = time.time() - t_enc0
print(f"  Encoding done in {t_enc:.1f}s\n")

# CSVs for imbgbm (target-encoded floats, label last)
TRAIN_CSV = f"{BENCH_DIR}/train.csv"
TEST_CSV  = f"{BENCH_DIR}/test.csv"
np.savetxt(TRAIN_CSV, np.c_[X_train_enc, y_train], delimiter=",", fmt="%.6f")
np.savetxt(TEST_CSV,  X_test_enc,                   delimiter=",", fmt="%.6f")

# ── Metric helpers ────────────────────────────────────────────────────────────

def ece(y_true, y_prob, n_bins=10):
    bins  = np.linspace(0, 1, n_bins + 1)
    total = len(y_true)
    err   = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        err += mask.sum() * abs(y_true[mask].mean() - y_prob[mask].mean())
    return err / total

def recall_at_fpr(y_true, y_prob, target_fpr=0.01):
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    idx = np.searchsorted(fpr, target_fpr, side="right") - 1
    return float(tpr[max(idx, 0)])

rows = []

def evaluate(name, y_true, y_prob, elapsed):
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float).clip(1e-7, 1 - 1e-7)
    auc     = roc_auc_score(y_true, y_prob)
    prauc   = average_precision_score(y_true, y_prob)
    ll      = log_loss(y_true, y_prob)
    brier   = brier_score_loss(y_true, y_prob)
    ece_val = ece(y_true, y_prob)
    r1pct   = recall_at_fpr(y_true, y_prob, 0.01)
    print(f"  AUC-ROC        : {auc:.4f}")
    print(f"  PR-AUC         : {prauc:.4f}")
    print(f"  Log-loss       : {ll:.4f}")
    print(f"  Brier score    : {brier:.4f}")
    print(f"  ECE (10 bins)  : {ece_val:.4f}")
    print(f"  Recall@FPR=1%  : {r1pct:.4f}")
    print(f"  Train time (s) : {elapsed:.1f}")
    rows.append(dict(model=name, auc=auc, prauc=prauc,
                     logloss=ll, brier=brier, ece=ece_val,
                     recall_1pct_fpr=r1pct, time_s=elapsed))

# ── 1. CatBoost default ───────────────────────────────────────────────────────

print("=" * 62)
print("1. CatBoost — default (native cat features, 300 rounds)")
t0 = time.time()
cb_default = cb.CatBoostClassifier(
    iterations=300, learning_rate=0.1, depth=6,
    cat_features=cat_features,
    random_seed=SEED, verbose=0,
)
cb_default.fit(X_train_cat, y_train)
elapsed = time.time() - t0
probs = cb_default.predict_proba(X_test_cat)[:, 1]
evaluate("CatBoost default", y_test, probs, elapsed)

# ── 2. CatBoost Balanced ──────────────────────────────────────────────────────

print()
print("=" * 62)
print("2. CatBoost — Balanced (native cat features, 300 rounds)")
t0 = time.time()
cb_bal = cb.CatBoostClassifier(
    iterations=300, learning_rate=0.1, depth=6,
    cat_features=cat_features,
    auto_class_weights="Balanced",
    random_seed=SEED, verbose=0,
)
cb_bal.fit(X_train_cat, y_train)
elapsed = time.time() - t0
probs = cb_bal.predict_proba(X_test_cat)[:, 1]
evaluate("CatBoost Balanced", y_test, probs, elapsed)

# ── 3. imbgbm BCE ─────────────────────────────────────────────────────────────

def run_cli(args):
    r = subprocess.run([IMBGBM] + args, capture_output=True, text=True)
    if r.returncode != 0:
        print("imbgbm stderr:", r.stderr[-2000:])
        sys.exit(1)
    return r

print()
print("=" * 62)
print("3. imbgbm — BCE, subsample=0.8 (target-encoded, 300 rounds)")
t0 = time.time()
run_cli([
    "train",
    "--input",                TRAIN_CSV,
    "--output",               f"{BENCH_DIR}/model_bce.json",
    "--loss",                 "bce",
    "--n-rounds",             "300",
    "--learning-rate",        "0.1",
    "--max-depth",            "6",
    "--subsample",            "0.8",
    "--early-stopping-rounds","0",
])
elapsed = time.time() - t0
r = run_cli(["predict", "--input", TEST_CSV,
             "--model", f"{BENCH_DIR}/model_bce.json"])
probs = np.array([float(x) for x in r.stdout.strip().splitlines()])
evaluate("imbgbm BCE", y_test, probs, elapsed)

# ── 4. imbgbm Focal + OOF calibration ────────────────────────────────────────

print()
print("=" * 62)
print("4. imbgbm — Focal (γ=2, α=0.25) + OOF cal. (Bayesian-smoothed, 300 rounds)")
t0 = time.time()
run_cli([
    "train",
    "--input",                TRAIN_CSV,
    "--output",               f"{BENCH_DIR}/model_focal.json",
    "--loss",                 "focal",
    "--n-rounds",             "300",
    "--learning-rate",        "0.05",
    "--max-depth",            "7",
    "--gamma",                "2.0",
    "--alpha",                "0.25",
    "--calibrate",
    "--early-stopping-rounds","0",
])
elapsed = time.time() - t0

r_raw = run_cli(["predict", "--input", TEST_CSV,
                 "--model", f"{BENCH_DIR}/model_focal.json"])
probs_raw = np.array([float(x) for x in r_raw.stdout.strip().splitlines()])
evaluate("imbgbm Focal (raw)", y_test, probs_raw, elapsed)

print()
print("  — calibrated probabilities (Bayesian-smoothed OOF leaf stats)")
r_cal = run_cli(["predict", "--input", TEST_CSV,
                 "--model", f"{BENCH_DIR}/model_focal.json",
                 "--calibrated"])
probs_cal = np.array([float(x) for x in r_cal.stdout.strip().splitlines()])
evaluate("imbgbm Focal+OOF", y_test, probs_cal, 0.0)

# ── Summary table ─────────────────────────────────────────────────────────────

print()
print("=" * 62)
print("SUMMARY  (Amazon-like: 33k train / 6k test, 9 cat features, 6% pos)")
print("=" * 62)
hdr = (f"{'Model':<26} {'AUC':>6} {'PR-AUC':>7} {'Logloss':>8} "
       f"{'Brier':>6} {'ECE':>6} {'R@1%':>6} {'t(s)':>5}")
print(hdr)
print("-" * len(hdr))
for r in rows:
    print(
        f"{r['model']:<26} "
        f"{r['auc']:6.4f} "
        f"{r['prauc']:7.4f} "
        f"{r['logloss']:8.4f} "
        f"{r['brier']:6.4f} "
        f"{r['ece']:6.4f} "
        f"{r['recall_1pct_fpr']:6.4f} "
        f"{r['time_s']:5.1f}"
    )
print()
print("Notes:")
print("  CatBoost: native ordered target statistics for categoricals.")
print("  imbgbm  : 5-fold OOF smoothed mean-target encoding (no leakage).")
print("  PR-AUC   higher = better discrimination of rare positives")
print("  ECE      lower  = better-calibrated probability outputs")
print("  R@1%FPR  recall at 1% FPR — key RTB operating point")
