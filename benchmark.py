#!/usr/bin/env python3
"""
Benchmark imbgbm vs CatBoost on a synthetic imbalanced binary classification task.

Four conditions:
  1. CatBoost — default (no class reweighting)
  2. CatBoost — auto_class_weights=Balanced
  3. imbgbm  — BCE loss, uniform subsampling
  4. imbgbm  — Focal loss (γ=2, α=0.25) + OOF calibration

Dataset: sklearn make_classification, 5 % positive, 20 features, 50 k train / 10 k test.
"""

import os, sys, subprocess, time, textwrap
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss, brier_score_loss
import catboost as cb

SEED = 42
np.random.seed(SEED)
IMBGBM = "/home/user/imbgbm/target/release/imbgbm"
BENCH_DIR = "/tmp/imbgbm_bench"
os.makedirs(BENCH_DIR, exist_ok=True)

# ── Dataset ───────────────────────────────────────────────────────────────────

print("Generating synthetic imbalanced dataset…")
X, y = make_classification(
    n_samples=60_000,
    n_features=20,
    n_informative=10,
    n_redundant=5,
    weights=[0.95, 0.05],   # ~5 % positive
    flip_y=0.01,
    random_state=SEED,
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=10_000, random_state=SEED, stratify=y
)
print(f"  Train: {X_train.shape}  pos={y_train.mean():.2%}")
print(f"  Test : {X_test.shape}   pos={y_test.mean():.2%}\n")

# Save CSVs for the imbgbm CLI (features + label in last column).
TRAIN_CSV = f"{BENCH_DIR}/train.csv"
TEST_CSV  = f"{BENCH_DIR}/test.csv"
np.savetxt(TRAIN_CSV, np.c_[X_train, y_train.astype(float)], delimiter=",", fmt="%.6f")
np.savetxt(TEST_CSV,  X_test,                                  delimiter=",", fmt="%.6f")

# ── Metric helpers ────────────────────────────────────────────────────────────

def ece(y_true, y_prob, n_bins=10):
    """Expected Calibration Error (equal-width bins)."""
    bins = np.linspace(0, 1, n_bins + 1)
    total = len(y_true)
    err = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        err += mask.sum() * abs(y_true[mask].mean() - y_prob[mask].mean())
    return err / total

def recall_at_fpr(y_true, y_prob, target_fpr=0.01):
    """True positive rate at a fixed false positive rate."""
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    idx = np.searchsorted(fpr, target_fpr, side="right") - 1
    return float(tpr[max(idx, 0)])

rows = []   # collect results for the summary table

def evaluate(name, y_true, y_prob, elapsed):
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
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

print("=" * 56)
print("1. CatBoost — default")
t0 = time.time()
cb_default = cb.CatBoostClassifier(
    iterations=100, learning_rate=0.1, depth=6,
    random_seed=SEED, verbose=0,
)
cb_default.fit(X_train, y_train)
elapsed = time.time() - t0
probs = cb_default.predict_proba(X_test)[:, 1]
evaluate("CatBoost default", y_test, probs, elapsed)

# ── 2. CatBoost balanced ──────────────────────────────────────────────────────

print()
print("=" * 56)
print("2. CatBoost — auto_class_weights=Balanced")
t0 = time.time()
cb_bal = cb.CatBoostClassifier(
    iterations=100, learning_rate=0.1, depth=6,
    auto_class_weights="Balanced",
    random_seed=SEED, verbose=0,
)
cb_bal.fit(X_train, y_train)
elapsed = time.time() - t0
probs = cb_bal.predict_proba(X_test)[:, 1]
evaluate("CatBoost balanced", y_test, probs, elapsed)

# ── 3. imbgbm BCE ─────────────────────────────────────────────────────────────

def run_cli(args):
    r = subprocess.run([IMBGBM] + args, capture_output=True, text=True)
    if r.returncode != 0:
        print("imbgbm stderr:", r.stderr)
        sys.exit(1)
    return r

print()
print("=" * 56)
print("3. imbgbm — BCE, uniform subsample=0.8")
t0 = time.time()
run_cli([
    "train",
    "--input",         TRAIN_CSV,
    "--output",        f"{BENCH_DIR}/model_bce.json",
    "--loss",          "bce",
    "--n-rounds",      "100",
    "--learning-rate", "0.1",
    "--max-depth",     "6",
    "--subsample",     "0.8",
])
elapsed = time.time() - t0
r = run_cli(["predict", "--input", TEST_CSV,
             "--model", f"{BENCH_DIR}/model_bce.json"])
probs = np.array([float(x) for x in r.stdout.strip().splitlines()])
evaluate("imbgbm BCE", y_test, probs, elapsed)

# ── 4. imbgbm Focal + calibration ────────────────────────────────────────────

print()
print("=" * 56)
print("4. imbgbm — Focal (γ=2, α=0.25) + OOF calibration")
t0 = time.time()
run_cli([
    "train",
    "--input",         TRAIN_CSV,
    "--output",        f"{BENCH_DIR}/model_focal.json",
    "--loss",          "focal",
    "--n-rounds",      "100",
    "--learning-rate", "0.05",
    "--max-depth",     "7",
    "--gamma",         "2.0",
    "--alpha",         "0.25",
    "--calibrate",
])
elapsed = time.time() - t0
# Raw predictions (sum of leaf Newton steps → sigmoid).
r_raw = run_cli(["predict", "--input", TEST_CSV,
                 "--model", f"{BENCH_DIR}/model_focal.json"])
probs_raw = np.array([float(x) for x in r_raw.stdout.strip().splitlines()])
evaluate("imbgbm Focal (raw)", y_test, probs_raw, elapsed)

print()
print("  — calibrated probabilities (OOF leaf probs averaged)")
# Calibrated predictions (OOF leaf probabilities).
r_cal = run_cli(["predict", "--input", TEST_CSV,
                 "--model", f"{BENCH_DIR}/model_focal.json",
                 "--calibrated"])
probs_cal = np.array([float(x) for x in r_cal.stdout.strip().splitlines()])
evaluate("imbgbm Focal+OOF", y_test, probs_cal, 0.0)  # same training time

# ── Summary table ─────────────────────────────────────────────────────────────

print()
print("=" * 56)
print("SUMMARY")
print("=" * 56)
hdr = f"{'Model':<26} {'AUC':>6} {'PR-AUC':>7} {'Logloss':>8} {'Brier':>6} {'ECE':>6} {'R@1%':>6} {'t(s)':>5}"
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
print("  PR-AUC   — average precision; higher = better discrimination of positives")
print("  ECE      — expected calibration error; lower = better calibrated probs")
print("  R@1%FPR  — recall at 1 % false-positive rate; RTB-relevant threshold")
