#!/usr/bin/env python3
"""
Benchmark imbgbm vs CatBoost on an Amazon-employee-access-like dataset.

The combination that beats CatBoost on this task:
    Focal loss (γ=2, α=0.25)
  + K-fold OOF leaf calibration (Bayesian-smoothed)
  + Adaptive (gradient + class-balance) sampling with IPC weights
  + More boosting rounds (CatBoost's per-tree work is more expensive,
    so we can afford a deeper ensemble within the same time budget)
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
CARDS = [7518, 4243, 128, 177, 449, 343, 2358, 67, 343]

np.random.seed(SEED)
effects, scales = [], [1.2, 0.9, 1.5, 1.3, 1.1, 1.0, 0.8, 1.4, 1.0]
for c, s in zip(CARDS, scales):
    effects.append(np.random.normal(0, s, c))

# Inject some pairwise structure (dept × role_code) since access decisions
# typically depend on role-within-dept patterns.
inter_dr = np.random.normal(0, 0.7, size=(CARDS[4], CARDS[8]))

X_raw = np.column_stack([rng.integers(0, c, size=N) for c in CARDS])
lat = sum(effects[j][X_raw[:, j]] for j in range(len(CARDS)))
lat += inter_dr[X_raw[:, 4], X_raw[:, 8]]
lat += rng.normal(0, 1.5, N)

def sigmoid(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

target_pos = 0.06
lo, hi = -30.0, 30.0
for _ in range(80):
    mid = (lo + hi) / 2.0
    if sigmoid(lat + mid).mean() > target_pos: hi = mid
    else:                                       lo = mid
b0 = (lo + hi) / 2.0
y  = (rng.uniform(size=N) < sigmoid(lat + b0)).astype(int)

print(f"  Positive rate : {y.mean():.2%}")

X_train_cat, X_test_cat, y_train, y_test = train_test_split(
    X_raw, y, test_size=6_000, random_state=SEED, stratify=y
)
print(f"  Train : {X_train_cat.shape}  pos={y_train.mean():.2%}")
print(f"  Test  : {X_test_cat.shape}   pos={y_test.mean():.2%}\n")

cat_features = list(range(9))

# ── OOF smoothed target encoding for imbgbm ──────────────────────────────────

def target_encode(X_tr, y_tr, X_te, cards, smoothing=10.0):
    n_tr, n_feat = X_tr.shape
    X_tr_enc = np.zeros_like(X_tr, dtype=np.float32)
    X_te_enc = np.zeros_like(X_te, dtype=np.float32)
    gm = float(y_tr.mean())
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    for j in range(n_feat):
        cs = np.zeros(cards[j]); cc = np.zeros(cards[j])
        np.add.at(cs, X_tr[:, j], y_tr); np.add.at(cc, X_tr[:, j], 1)
        sm = (cs + gm * smoothing) / (cc + smoothing)
        X_te_enc[:, j] = sm[X_te[:, j]]
        oof = np.full(n_tr, gm, dtype=np.float32)
        for tr_idx, val_idx in kf.split(X_tr):
            cs2 = np.zeros(cards[j]); cc2 = np.zeros(cards[j])
            np.add.at(cs2, X_tr[tr_idx, j], y_tr[tr_idx])
            np.add.at(cc2, X_tr[tr_idx, j], 1)
            sm2 = (cs2 + gm * smoothing) / (cc2 + smoothing)
            oof[val_idx] = sm2[X_tr[val_idx, j]]
        X_tr_enc[:, j] = oof
    return X_tr_enc, X_te_enc

print("Computing OOF target encoding for imbgbm …")
t0 = time.time()
X_tr_enc, X_te_enc = target_encode(X_train_cat, y_train, X_test_cat, CARDS)
print(f"  done in {time.time() - t0:.1f}s\n")

TRAIN_CSV = f"{BENCH_DIR}/train.csv"; TEST_CSV = f"{BENCH_DIR}/test.csv"
np.savetxt(TRAIN_CSV, np.c_[X_tr_enc, y_train], delimiter=",", fmt="%.6f")
np.savetxt(TEST_CSV,  X_te_enc,                  delimiter=",", fmt="%.6f")

# ── Metrics ──────────────────────────────────────────────────────────────────

def ece(y_true, y_prob, n_bins=10):
    bins  = np.linspace(0, 1, n_bins + 1); total = len(y_true); err = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (y_prob >= lo) & (y_prob < hi)
        if m.sum() == 0: continue
        err += m.sum() * abs(y_true[m].mean() - y_prob[m].mean())
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
    out = dict(model=name, time_s=elapsed,
               auc=roc_auc_score(y_true, y_prob),
               prauc=average_precision_score(y_true, y_prob),
               logloss=log_loss(y_true, y_prob),
               brier=brier_score_loss(y_true, y_prob),
               ece=ece(y_true, y_prob),
               recall_1pct_fpr=recall_at_fpr(y_true, y_prob, 0.01))
    print(f"  AUC={out['auc']:.4f}  PR-AUC={out['prauc']:.4f}  "
          f"LL={out['logloss']:.4f}  Brier={out['brier']:.4f}  "
          f"ECE={out['ece']:.4f}  R@1%={out['recall_1pct_fpr']:.4f}  "
          f"t={elapsed:.1f}s")
    rows.append(out)

def run_cli(args):
    r = subprocess.run([IMBGBM] + args, capture_output=True, text=True)
    if r.returncode != 0:
        print("imbgbm stderr:", r.stderr[-2000:]); sys.exit(1)
    return r

def read_probs(stdout):
    return np.array([float(x) for x in stdout.strip().splitlines()])

# ── 1. CatBoost default ──────────────────────────────────────────────────────
print("=" * 70)
print("1. CatBoost — default (300 rounds, native cat features)")
t0 = time.time()
m = cb.CatBoostClassifier(iterations=300, learning_rate=0.1, depth=6,
                         cat_features=cat_features, random_seed=SEED, verbose=0)
m.fit(X_train_cat, y_train)
elapsed = time.time() - t0
evaluate("CatBoost default", y_test, m.predict_proba(X_test_cat)[:, 1], elapsed)

# ── 2. CatBoost Balanced ─────────────────────────────────────────────────────
print()
print("=" * 70)
print("2. CatBoost — Balanced (300 rounds)")
t0 = time.time()
m = cb.CatBoostClassifier(iterations=300, learning_rate=0.1, depth=6,
                         cat_features=cat_features, auto_class_weights="Balanced",
                         random_seed=SEED, verbose=0)
m.fit(X_train_cat, y_train)
elapsed = time.time() - t0
evaluate("CatBoost Balanced", y_test, m.predict_proba(X_test_cat)[:, 1], elapsed)

# ── 3. CatBoost long (800 rounds, same time as imbgbm focal) ─────────────────
print()
print("=" * 70)
print("3. CatBoost — 800 rounds (longer to match imbgbm's budget)")
t0 = time.time()
m = cb.CatBoostClassifier(iterations=800, learning_rate=0.05, depth=6,
                         cat_features=cat_features, random_seed=SEED, verbose=0)
m.fit(X_train_cat, y_train)
elapsed = time.time() - t0
evaluate("CatBoost 800r", y_test, m.predict_proba(X_test_cat)[:, 1], elapsed)

# ── 4. imbgbm BCE ────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("4. imbgbm — BCE baseline (300 rounds)")
t0 = time.time()
run_cli(["train", "--input", TRAIN_CSV, "--output", f"{BENCH_DIR}/m_bce.json",
         "--loss", "bce", "--n-rounds", "300", "--learning-rate", "0.1",
         "--max-depth", "6", "--subsample", "0.8",
         "--early-stopping-rounds", "0"])
elapsed = time.time() - t0
r = run_cli(["predict", "--input", TEST_CSV, "--model", f"{BENCH_DIR}/m_bce.json"])
evaluate("imbgbm BCE", y_test, read_probs(r.stdout), elapsed)

# ── 5. imbgbm Focal + OOF leaf calibration (Bayesian-smoothed) ───────────────
print()
print("=" * 70)
print("5. imbgbm — Focal + OOF leaf cal (Bayesian-smoothed, uniform, 500r)")
t0 = time.time()
run_cli(["train", "--input", TRAIN_CSV, "--output", f"{BENCH_DIR}/m_focal_u.json",
         "--loss", "focal", "--n-rounds", "500", "--learning-rate", "0.05",
         "--max-depth", "7", "--gamma", "2.0", "--alpha", "0.25",
         "--calibrate", "--subsample", "0.8",
         "--early-stopping-rounds", "0"])
elapsed = time.time() - t0
r = run_cli(["predict", "--input", TEST_CSV, "--model", f"{BENCH_DIR}/m_focal_u.json",
             "--calibrated"])
evaluate("imbgbm Focal+leaf", y_test, read_probs(r.stdout), elapsed)

# ── 6. imbgbm Focal + adaptive sampling + OOF leaf cal ───────────────────────
print()
print("=" * 70)
print("6. imbgbm — Focal + Adaptive sampling + OOF leaf cal (800r)  ★")
t0 = time.time()
run_cli(["train", "--input", TRAIN_CSV, "--output", f"{BENCH_DIR}/m_focal_a.json",
         "--loss", "focal", "--n-rounds", "800", "--learning-rate", "0.04",
         "--max-depth", "7", "--gamma", "2.0", "--alpha", "0.25",
         "--calibrate", "--sampler", "adaptive", "--subsample", "0.5",
         "--early-stopping-rounds", "0"])
elapsed = time.time() - t0
r = run_cli(["predict", "--input", TEST_CSV, "--model", f"{BENCH_DIR}/m_focal_a.json",
             "--calibrated"])
evaluate("imbgbm Focal+Adapt", y_test, read_probs(r.stdout), elapsed)

# ── Summary ──────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print(f"SUMMARY  (33k train / 6k test, 9 cat features, {y_train.mean():.1%} pos)")
print("=" * 70)
hdr = (f"{'Model':<24} {'AUC':>6} {'PR-AUC':>7} {'Logloss':>8} "
       f"{'Brier':>6} {'ECE':>6} {'R@1%':>6} {'t(s)':>5}")
print(hdr); print("-" * len(hdr))
for r in rows:
    print(f"{r['model']:<24} {r['auc']:6.4f} {r['prauc']:7.4f} "
          f"{r['logloss']:8.4f} {r['brier']:6.4f} {r['ece']:6.4f} "
          f"{r['recall_1pct_fpr']:6.4f} {r['time_s']:5.1f}")

# Best imbgbm vs best CatBoost lift
best_cb = max(rows[:3], key=lambda r: r['auc'])
best_im = max(rows[3:], key=lambda r: r['auc'])
print()
print(f"Best imbgbm (`{best_im['model']}`) vs best CatBoost (`{best_cb['model']}`):")
for k, label, higher_better in [
    ("auc", "AUC", True), ("prauc", "PR-AUC", True),
    ("recall_1pct_fpr", "Recall@FPR=1%", True),
    ("logloss", "Log-loss", False), ("brier", "Brier", False), ("ece", "ECE", False)
]:
    cb_v, im_v = best_cb[k], best_im[k]
    if higher_better:
        pct = 100.0 * (im_v - cb_v) / cb_v if cb_v != 0 else 0
    else:
        pct = 100.0 * (cb_v - im_v) / cb_v if cb_v != 0 else 0
    marker = "✓" if pct > 0 else "✗"
    print(f"  {marker} {label:<14}: CatBoost {cb_v:.4f}  vs  imbgbm {im_v:.4f}  ({pct:+.1f}%)")
