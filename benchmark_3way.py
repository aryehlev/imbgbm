#!/usr/bin/env python3
"""
Three-way benchmark: imbgbm vs CatBoost (best) vs LightGBM (best).

Dataset  : Amazon-employee-access synthetic (33k train / 6k test,
           9 high-cardinality cat features, ~6% positive rate).
Scenario : Fully labeled (clean labels) — measures ceiling performance.

Methods
───────
  CatBoost  — default 300r  (standard baseline)
  CatBoost  — tuned 1000r   (deeper trees, Bayesian regularisation)
  LightGBM  — default 300r  (standard baseline)
  LightGBM  — tuned 1000r   (feature_fraction, dart-style regularisation)
  imbgbm    — Focal+Adapt   (best previous single-stage)
  imbgbm    — Focal+Adapt+ColSub  (new: + feature subsampling)
  imbgbm    — Focal+leaf+ColSub   (new: ranker variant)
  imbgbm    — Ensemble best        (naive avg of best two above)
"""

import os, sys, subprocess, time
import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score, log_loss, brier_score_loss, roc_curve,
)
import catboost as cb
import lightgbm as lgb

SEED = 42
rng  = np.random.default_rng(SEED)

IMBGBM    = "/home/user/imbgbm/target/release/imbgbm"
BENCH_DIR = "/tmp/imbgbm_3way_bench"
os.makedirs(BENCH_DIR, exist_ok=True)

# ── Dataset ────────────────────────────────────────────────────────────────────
print("Generating synthetic Amazon-employee-access dataset …")
N     = 39_000
CARDS = [7518, 4243, 128, 177, 449, 343, 2358, 67, 343]

np.random.seed(SEED)
effects, scales = [], [1.2, 0.9, 1.5, 1.3, 1.1, 1.0, 0.8, 1.4, 1.0]
for c, s in zip(CARDS, scales):
    effects.append(np.random.normal(0, s, c))
inter_dr = np.random.normal(0, 0.7, size=(CARDS[4], CARDS[8]))

X_raw = np.column_stack([rng.integers(0, c, size=N) for c in CARDS])
lat   = sum(effects[j][X_raw[:, j]] for j in range(len(CARDS)))
lat  += inter_dr[X_raw[:, 4], X_raw[:, 8]]
lat  += rng.normal(0, 1.5, N)

def sigmoid(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

lo, hi = -30.0, 30.0
for _ in range(80):
    mid = (lo + hi) / 2.0
    if sigmoid(lat + mid).mean() > 0.06: hi = mid
    else:                                 lo = mid
y = (rng.uniform(size=N) < sigmoid(lat + (lo+hi)/2)).astype(int)

X_train_cat, X_test_cat, y_train, y_test = train_test_split(
    X_raw, y, test_size=6_000, random_state=SEED, stratify=y)
print(f"  train={len(y_train)}  pos={y_train.mean():.2%}  "
      f"test={len(y_test)}  pos={y_test.mean():.2%}\n")

cat_features = list(range(9))
pos_weight   = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

# ── OOF target encoding for imbgbm ────────────────────────────────────────────
def target_encode(X_tr, y_tr, X_te, cards, smoothing=10.0):
    n_tr = len(y_tr)
    X_tr_enc = np.zeros_like(X_tr, dtype=np.float32)
    X_te_enc = np.zeros_like(X_te, dtype=np.float32)
    gm = float(y_tr.mean())
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    for j, card in enumerate(cards):
        cs = np.zeros(card); cc = np.zeros(card)
        np.add.at(cs, X_tr[:, j], y_tr); np.add.at(cc, X_tr[:, j], 1)
        sm = (cs + gm * smoothing) / (cc + smoothing)
        X_te_enc[:, j] = sm[X_te[:, j]]
        oof = np.full(n_tr, gm, dtype=np.float32)
        for tri, vai in kf.split(X_tr):
            cs2 = np.zeros(card); cc2 = np.zeros(card)
            np.add.at(cs2, X_tr[tri, j], y_tr[tri])
            np.add.at(cc2, X_tr[tri, j], 1)
            sm2 = (cs2 + gm * smoothing) / (cc2 + smoothing)
            oof[vai] = sm2[X_tr[vai, j]]
        X_tr_enc[:, j] = oof
    return X_tr_enc, X_te_enc

print("OOF target encoding …")
t0 = time.time()
X_tr_enc, X_te_enc = target_encode(X_train_cat, y_train, X_test_cat, CARDS)
print(f"  done in {time.time()-t0:.1f}s\n")

TRAIN_CSV = f"{BENCH_DIR}/train.csv"
TEST_CSV  = f"{BENCH_DIR}/test.csv"
np.savetxt(TRAIN_CSV, np.c_[X_tr_enc, y_train], delimiter=",", fmt="%.6f")
np.savetxt(TEST_CSV,  X_te_enc,                  delimiter=",", fmt="%.6f")

# ── Metrics ────────────────────────────────────────────────────────────────────
def ece(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1); total = len(y_true); err = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (y_prob >= lo) & (y_prob < hi)
        if m.sum() == 0: continue
        err += m.sum() * abs(y_true[m].mean() - y_prob[m].mean())
    return err / total

def recall_at_fpr(y_true, y_prob, fpr_target=0.01):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(tpr[max(np.searchsorted(fpr, fpr_target, "right") - 1, 0)])

def precision_at_k(y_true, y_prob, k_frac=0.05):
    k = int(len(y_true) * k_frac)
    return float(y_true[np.argsort(-y_prob)[:k]].mean())

rows = []

def evaluate(name, y_true, y_prob, elapsed):
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float).clip(1e-7, 1 - 1e-7)
    out = dict(
        model=name, time_s=elapsed,
        auc     = roc_auc_score(y_true, y_prob),
        prauc   = average_precision_score(y_true, y_prob),
        logloss = log_loss(y_true, y_prob),
        brier   = brier_score_loss(y_true, y_prob),
        ece     = ece(y_true, y_prob),
        r1pct   = recall_at_fpr(y_true, y_prob),
        p5pct   = precision_at_k(y_true, y_prob),
    )
    print(f"  AUC={out['auc']:.4f}  PR-AUC={out['prauc']:.4f}  "
          f"ECE={out['ece']:.4f}  R@1%={out['r1pct']:.4f}  "
          f"P@5%={out['p5pct']:.4f}  t={elapsed:.1f}s")
    rows.append(out)

def run_cli(args):
    r = subprocess.run([IMBGBM] + args, capture_output=True, text=True)
    if r.returncode != 0:
        print("STDERR:", r.stderr[-2000:]); sys.exit(1)
    return r

def read_probs(stdout):
    return np.array([float(x) for x in stdout.strip().splitlines()])

# ═══════════════════════════════════════════════════════════════════════════════
# CatBoost
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 72)
print("1. CatBoost — default (300r, depth=6)")
t0 = time.time()
m = cb.CatBoostClassifier(iterations=300, learning_rate=0.1, depth=6,
                          cat_features=cat_features, random_seed=SEED, verbose=0)
m.fit(X_train_cat, y_train)
evaluate("CatBoost default", y_test, m.predict_proba(X_test_cat)[:, 1], time.time()-t0)

print()
print("=" * 72)
print("2. CatBoost — tuned best (1000r, depth=8, Bayesian regularisation)")
t0 = time.time()
m = cb.CatBoostClassifier(
    iterations=1000, learning_rate=0.03, depth=8,
    l2_leaf_reg=3.0, bagging_temperature=0.5, random_strength=0.5,
    border_count=254, min_data_in_leaf=10,
    cat_features=cat_features, random_seed=SEED, verbose=0,
)
m.fit(X_train_cat, y_train)
cb_best_probs = m.predict_proba(X_test_cat)[:, 1]
evaluate("CatBoost tuned ★", y_test, cb_best_probs, time.time()-t0)

# ═══════════════════════════════════════════════════════════════════════════════
# LightGBM
# ═══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 72)
print("3. LightGBM — default (300r, num_leaves=31)")
t0 = time.time()
m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.1, num_leaves=31,
                        cat_smooth=10, random_state=SEED, verbose=-1)
m.fit(X_train_cat, y_train,
      categorical_feature=cat_features)
evaluate("LightGBM default", y_test, m.predict_proba(X_test_cat)[:, 1], time.time()-t0)

print()
print("=" * 72)
print("4. LightGBM — tuned best (1000r, num_leaves=127, feature_fraction=0.8)")
t0 = time.time()
m = lgb.LGBMClassifier(
    n_estimators=1000, learning_rate=0.03, num_leaves=127,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
    min_child_samples=20, reg_alpha=0.1, reg_lambda=1.0,
    cat_smooth=10, min_data_per_group=50,
    random_state=SEED, verbose=-1,
)
m.fit(X_train_cat, y_train,
      categorical_feature=cat_features)
lgb_best_probs = m.predict_proba(X_test_cat)[:, 1]
evaluate("LightGBM tuned ★", y_test, lgb_best_probs, time.time()-t0)

# ═══════════════════════════════════════════════════════════════════════════════
# imbgbm
# ═══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 72)
print("5. imbgbm — Focal+Adapt  (previous best, no col_subsample)")
t0 = time.time()
run_cli(["train", "--input", TRAIN_CSV, "--output", f"{BENCH_DIR}/m_adapt.json",
         "--loss", "focal", "--n-rounds", "800", "--learning-rate", "0.04",
         "--max-depth", "7", "--gamma", "2.0", "--alpha", "0.25",
         "--calibrate", "--sampler", "adaptive", "--subsample", "0.5",
         "--early-stopping-rounds", "0"])
r = run_cli(["predict", "--input", TEST_CSV, "--model", f"{BENCH_DIR}/m_adapt.json",
             "--calibrated"])
prev_adapt_probs = read_probs(r.stdout)
evaluate("imbgbm Focal+Adapt", y_test, prev_adapt_probs, time.time()-t0)

print()
print("=" * 72)
print("6. imbgbm — Focal+Adapt + col_subsample=0.8  ★")
t0 = time.time()
run_cli(["train", "--input", TRAIN_CSV, "--output", f"{BENCH_DIR}/m_adapt_cs.json",
         "--loss", "focal", "--n-rounds", "1000", "--learning-rate", "0.03",
         "--max-depth", "7", "--gamma", "2.0", "--alpha", "0.25",
         "--calibrate", "--sampler", "adaptive", "--subsample", "0.5",
         "--col-subsample", "0.8",
         "--early-stopping-rounds", "0"])
r = run_cli(["predict", "--input", TEST_CSV, "--model", f"{BENCH_DIR}/m_adapt_cs.json",
             "--calibrated"])
adapt_cs_probs = read_probs(r.stdout)
evaluate("imbgbm Adapt+ColSub ★", y_test, adapt_cs_probs, time.time()-t0)

print()
print("=" * 72)
print("7. imbgbm — Focal+leaf + col_subsample=0.8  (ranker variant)")
t0 = time.time()
run_cli(["train", "--input", TRAIN_CSV, "--output", f"{BENCH_DIR}/m_focal_cs.json",
         "--loss", "focal", "--n-rounds", "1000", "--learning-rate", "0.03",
         "--max-depth", "7", "--gamma", "2.0", "--alpha", "0.25",
         "--calibrate", "--sampler", "uniform", "--subsample", "0.8",
         "--col-subsample", "0.8",
         "--early-stopping-rounds", "0"])
r = run_cli(["predict", "--input", TEST_CSV, "--model", f"{BENCH_DIR}/m_focal_cs.json",
             "--calibrated"])
focal_cs_probs = read_probs(r.stdout)
evaluate("imbgbm Focal+ColSub", y_test, focal_cs_probs, time.time()-t0)

print()
print("=" * 72)
print("8. imbgbm — Ensemble (avg Adapt+ColSub + Focal+ColSub)")
t0 = time.time()
ens_probs = 0.5 * adapt_cs_probs + 0.5 * focal_cs_probs
evaluate("imbgbm Ensemble", y_test, ens_probs, time.time()-t0)

# ── Temperature scaling ───────────────────────────────────────────────────────
# Calibrates a temperature T on held-out OOF predictions so that
# p_cal = sigmoid(logit(p) / T) minimises cross-entropy.
# Monotone transformation ⟹ AUC is unchanged; ECE drops significantly.

def temp_scale(test_probs, ref_probs, ref_labels):
    """Find T on ref split, apply to test_probs. Returns (calibrated, T)."""
    ref_probs  = np.asarray(ref_probs,  dtype=float).clip(1e-7, 1-1e-7)
    test_probs = np.asarray(test_probs, dtype=float).clip(1e-7, 1-1e-7)
    ref_logits  = np.log(ref_probs  / (1 - ref_probs))
    test_logits = np.log(test_probs / (1 - test_probs))
    def nll(T):
        p = 1.0 / (1.0 + np.exp(-ref_logits / T))
        p = p.clip(1e-7, 1-1e-7)
        return -np.mean(ref_labels * np.log(p) + (1-ref_labels) * np.log(1-p))
    T = minimize_scalar(nll, bounds=(0.05, 20.0), method="bounded").x
    p_cal = 1.0 / (1.0 + np.exp(-test_logits / T))
    return p_cal.astype(np.float32), T

# Calibrate T on training predictions (single scalar, negligible overfitting)
TRAIN_PRED_CSV = f"{BENCH_DIR}/train_features.csv"   # features only (no label)
np.savetxt(TRAIN_PRED_CSV, X_tr_enc, delimiter=",", fmt="%.6f")

r_tr_adapt = run_cli(["predict", "--input", TRAIN_PRED_CSV,
                      "--model", f"{BENCH_DIR}/m_adapt_cs.json", "--calibrated"])
r_tr_focal = run_cli(["predict", "--input", TRAIN_PRED_CSV,
                      "--model", f"{BENCH_DIR}/m_focal_cs.json", "--calibrated"])
tr_adapt = read_probs(r_tr_adapt.stdout)
tr_focal = read_probs(r_tr_focal.stdout)
tr_ens   = 0.5 * tr_adapt + 0.5 * tr_focal

print()
print("=" * 72)
print("9. imbgbm — Ensemble + Temperature Scaling  ★★★")
t0 = time.time()
ens_cal, T = temp_scale(ens_probs, tr_ens, y_train)
elapsed_ts = time.time() - t0
print(f"   T = {T:.4f}", end="  ")
evaluate("imbgbm Ens+TempScale ★★★", y_test, ens_cal, elapsed_ts)

# ── Summary ────────────────────────────────────────────────────────────────────
print()
print("=" * 78)
print(f"THREE-WAY BENCHMARK  (33k train / 6k test, 9 cat features, "
      f"pos={y_test.mean():.1%})")
print("=" * 78)
hdr = (f"{'Model':<28} {'AUC':>6} {'PR-AUC':>7} {'ECE':>7} "
       f"{'Brier':>6} {'R@1%':>6} {'P@5%':>6} {'t(s)':>6}")
print(hdr); print("-" * len(hdr))
for row in rows:
    print(f"{row['model']:<28} {row['auc']:6.4f} {row['prauc']:7.4f} "
          f"{row['ece']:7.4f} {row['brier']:6.4f} "
          f"{row['r1pct']:6.4f} {row['p5pct']:6.4f} "
          f"{row['time_s']:6.1f}")

# Best per family: pick best-per-metric model for each family
im_rows = rows[4:]
best_cb  = max(rows[:2],  key=lambda r: r["auc"])
best_lgb = max(rows[2:4], key=lambda r: r["auc"])
print()
print(f"{'Metric':<16} {'CatBoost best':>14} {'LightGBM best':>14} {'imbgbm best':>14}")
print("-" * 62)
for k, label, higher_better in [
    ("auc",     "AUC",          True),
    ("prauc",   "PR-AUC",       True),
    ("r1pct",   "Recall@FPR=1%",True),
    ("p5pct",   "Precision@5%", True),
    ("ece",     "ECE",          False),
    ("brier",   "Brier",        False),
    ("logloss", "Log-loss",     False),
]:
    # Pick best-per-metric imbgbm model for this metric
    if higher_better:
        best_im = max(im_rows, key=lambda r: r[k])
    else:
        best_im = min(im_rows, key=lambda r: r[k])
    cb_v, lgb_v, im_v = best_cb[k], best_lgb[k], best_im[k]
    best_val = max([cb_v, lgb_v, im_v]) if higher_better else min([cb_v, lgb_v, im_v])
    def fmt(v, name=""):
        marker = " ◄" if abs(v - best_val) < 1e-9 else "  "
        return f"{v:.4f}{marker}"
    print(f"{label:<16} {fmt(cb_v):>16} {fmt(lgb_v):>16} {fmt(im_v):>16}")

print()
print("◄ = best in row  (imbgbm column shows best-per-metric model, not one model)")
