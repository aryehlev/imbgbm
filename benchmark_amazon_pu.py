#!/usr/bin/env python3
"""
PU-boosting benchmark on Amazon-employee-access-like data.

We take the same Amazon-style synthetic dataset (33 k train / 6 k test, 9 high-
cardinality categorical features, ~6 % true positives) and simulate a realistic
positive-unlabeled scenario:

  - Take a labeling rate `c` (e.g. 0.30) and only label that fraction of true
    positives. All others become "unlabeled" (s = 0), mixed in with the true
    negatives.
  - Train each method on the *observed* (s) labels.
  - Evaluate against the *true* (y) labels.

This is exactly the advertiser-seed scenario: the advertiser sends 30 % of its
real converters; the rest are buried in the unlabeled pool and we must find
them. CatBoost has no PU support and will be misled by the hidden positives —
the question is whether the imbgbm PU research-mode features close that gap.
"""

import os, sys, subprocess, time
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss, brier_score_loss
import catboost as cb

SEED = 42
rng  = np.random.default_rng(SEED)

IMBGBM    = "/home/user/imbgbm/target/release/imbgbm"
BENCH_DIR = "/tmp/imbgbm_amazon_pu_bench"
os.makedirs(BENCH_DIR, exist_ok=True)

# Labeling rate: fraction of TRUE positives that keep their label.
C_RATE = 0.30

# ── Synthetic Amazon-like dataset (same as benchmark_amazon.py) ──────────────
print("Generating synthetic Amazon-employee-access dataset …")

N     = 39_000
CARDS = [7518, 4243, 128, 177, 449, 343, 2358, 67, 343]
NAMES = ["RESOURCE", "MGR_ID", "ROLE_ROLLUP_1", "ROLE_ROLLUP_2",
         "ROLE_DEPTNAME", "ROLE_TITLE", "ROLE_FAMILY_DESC",
         "ROLE_FAMILY", "ROLE_CODE"]

np.random.seed(SEED)
effects, scales = [], [1.2, 0.9, 1.5, 1.3, 1.1, 1.0, 0.8, 1.4, 1.0]
for c, s in zip(CARDS, scales):
    effects.append(np.random.normal(0, s, c))
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
y_true = (rng.uniform(size=N) < sigmoid(lat + b0)).astype(int)

# ── Simulate biased PU labeling ──────────────────────────────────────────────
# Each true positive is labeled (s=1) independently with probability C_RATE.
# This is the SCAR assumption.
s_obs = y_true.copy()
keep = rng.uniform(size=N) < C_RATE
s_obs = (y_true & keep).astype(int)   # only kept positives stay labeled

n_true_pos = int(y_true.sum())
n_seeds    = int(s_obs.sum())
print(f"  N total           : {N}")
print(f"  True positives    : {n_true_pos}  ({y_true.mean():.2%})")
print(f"  Labeled seeds     : {n_seeds}  ({s_obs.mean():.2%})  "
      f"[c = {n_seeds / max(n_true_pos, 1):.2f}]")
print(f"  Hidden positives  : {n_true_pos - n_seeds}  "
      f"(buried in 'unlabeled' pool of {N - n_seeds})")
print()

# Train / test split (stratified on TRUE labels so each split has positives).
idx_tr, idx_te = train_test_split(
    np.arange(N), test_size=6_000, random_state=SEED, stratify=y_true
)
X_train_cat = X_raw[idx_tr]; X_test_cat = X_raw[idx_te]
y_train_true = y_true[idx_tr]; y_test_true = y_true[idx_te]
s_train      = s_obs[idx_tr];  s_test_true  = y_true[idx_te]  # evaluate on truth
print(f"  Train : {X_train_cat.shape}  seeds={s_train.mean():.2%}  "
      f"true_pos={y_train_true.mean():.2%}")
print(f"  Test  : {X_test_cat.shape}   true_pos={y_test_true.mean():.2%}\n")

cat_features = list(range(9))

# ── OOF smoothed target encoding for imbgbm (using observed PU labels!) ──────
# Important: encoding is done on s (the observed labels), not y_true. This is
# what would happen in production — we never see the hidden positives.

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

print("Computing OOF target encoding using observed PU labels …")
t0 = time.time()
X_tr_enc, X_te_enc = target_encode(X_train_cat, s_train, X_test_cat, CARDS)
print(f"  done in {time.time() - t0:.1f}s\n")

TRAIN_CSV = f"{BENCH_DIR}/train.csv"; TEST_CSV = f"{BENCH_DIR}/test.csv"
np.savetxt(TRAIN_CSV, np.c_[X_tr_enc, s_train], delimiter=",", fmt="%.6f")
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

def precision_at_k(y_true, y_prob, k_frac=0.05):
    n = len(y_true); k = int(n * k_frac)
    order = np.argsort(-y_prob)[:k]
    return float(y_true[order].mean())

rows = []

def evaluate(name, y_true, y_prob, elapsed):
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float).clip(1e-7, 1 - 1e-7)
    out = dict(
        model=name, time_s=elapsed,
        auc   = roc_auc_score(y_true, y_prob),
        prauc = average_precision_score(y_true, y_prob),
        logloss = log_loss(y_true, y_prob),
        brier   = brier_score_loss(y_true, y_prob),
        ece     = ece(y_true, y_prob),
        recall_1pct_fpr = recall_at_fpr(y_true, y_prob, 0.01),
        p_at_5pct       = precision_at_k(y_true, y_prob, 0.05),
    )
    print(f"  AUC={out['auc']:.4f}  PR-AUC={out['prauc']:.4f}  "
          f"LL={out['logloss']:.4f}  Brier={out['brier']:.4f}  "
          f"ECE={out['ece']:.4f}  R@1%={out['recall_1pct_fpr']:.4f}  "
          f"P@5%={out['p_at_5pct']:.4f}  t={elapsed:.1f}s")
    rows.append(out)

def run_cli(args):
    r = subprocess.run([IMBGBM] + args, capture_output=True, text=True)
    if r.returncode != 0:
        print("imbgbm stderr:", r.stderr[-2000:]); sys.exit(1)
    return r

def read_probs(stdout):
    return np.array([float(x) for x in stdout.strip().splitlines()])

# ── 1. CatBoost default (observed labels) ────────────────────────────────────
print("=" * 70)
print("1. CatBoost — default on observed PU labels (no PU awareness)")
t0 = time.time()
m = cb.CatBoostClassifier(iterations=300, learning_rate=0.1, depth=6,
                         cat_features=cat_features, random_seed=SEED, verbose=0)
m.fit(X_train_cat, s_train)
elapsed = time.time() - t0
evaluate("CatBoost default", y_test_true, m.predict_proba(X_test_cat)[:, 1], elapsed)

# ── 2. CatBoost Balanced (the "scale_pos_weight" trap) ──────────────────────
print()
print("=" * 70)
print("2. CatBoost — Balanced (the scale_pos_weight trap)")
t0 = time.time()
m = cb.CatBoostClassifier(iterations=300, learning_rate=0.1, depth=6,
                         cat_features=cat_features, auto_class_weights="Balanced",
                         random_seed=SEED, verbose=0)
m.fit(X_train_cat, s_train)
elapsed = time.time() - t0
evaluate("CatBoost Balanced", y_test_true, m.predict_proba(X_test_cat)[:, 1], elapsed)

# ── 3. imbgbm BCE baseline (treats unlabeled as negative) ────────────────────
print()
print("=" * 70)
print("3. imbgbm — BCE baseline (treats unlabeled as negative)")
t0 = time.time()
run_cli(["train", "--input", TRAIN_CSV, "--output", f"{BENCH_DIR}/m_bce.json",
         "--loss", "bce", "--n-rounds", "400", "--learning-rate", "0.08",
         "--max-depth", "6", "--subsample", "0.8",
         "--early-stopping-rounds", "0"])
elapsed = time.time() - t0
r = run_cli(["predict", "--input", TEST_CSV, "--model", f"{BENCH_DIR}/m_bce.json"])
evaluate("imbgbm BCE", y_test_true, read_probs(r.stdout), elapsed)

# ── 4. imbgbm Focal + OOF leaf calibration (last branch's strongest baseline)
print()
print("=" * 70)
print("4. imbgbm — Focal + OOF leaf cal (strongest non-PU baseline)")
t0 = time.time()
run_cli(["train", "--input", TRAIN_CSV, "--output", f"{BENCH_DIR}/m_focal.json",
         "--loss", "focal", "--n-rounds", "500", "--learning-rate", "0.05",
         "--max-depth", "7", "--gamma", "2.0", "--alpha", "0.25",
         "--calibrate", "--subsample", "0.8", "--early-stopping-rounds", "0"])
elapsed = time.time() - t0
r = run_cli(["predict", "--input", TEST_CSV, "--model", f"{BENCH_DIR}/m_focal.json",
             "--calibrated"])
evaluate("imbgbm Focal+leaf", y_test_true, read_probs(r.stdout), elapsed)

# ── 5. imbgbm PU loss ────────────────────────────────────────────────────────
print()
print("=" * 70)
print("5. imbgbm — PULoss (du Plessis et al.) with seed-rate-aware prior")
# True prior in this dataset is 6 %; with c=0.3, observed positive rate ≈ 1.8 %.
t0 = time.time()
run_cli(["train", "--input", TRAIN_CSV, "--output", f"{BENCH_DIR}/m_pu.json",
         "--loss", "pu", "--pu-prior", "0.06",
         "--n-rounds", "500", "--learning-rate", "0.05", "--max-depth", "6",
         "--subsample", "0.8", "--early-stopping-rounds", "0"])
elapsed = time.time() - t0
r = run_cli(["predict", "--input", TEST_CSV, "--model", f"{BENCH_DIR}/m_pu.json"])
evaluate("imbgbm PULoss", y_test_true, read_probs(r.stdout), elapsed)

# ── 6. imbgbm Density-Ratio (idea 9) ────────────────────────────────────────
print()
print("=" * 70)
print("6. imbgbm — Density-ratio loss (idea 9, symmetric P/U log-ratio)")
t0 = time.time()
run_cli(["train", "--input", TRAIN_CSV, "--output", f"{BENCH_DIR}/m_dr.json",
         "--loss", "density_ratio", "--pu-prior", "0.5",
         "--n-rounds", "500", "--learning-rate", "0.05", "--max-depth", "6",
         "--subsample", "0.8", "--early-stopping-rounds", "0"])
elapsed = time.time() - t0
r = run_cli(["predict", "--input", TEST_CSV, "--model", f"{BENCH_DIR}/m_dr.json"])
evaluate("imbgbm DensityRatio", y_test_true, read_probs(r.stdout), elapsed)

# ── 7. imbgbm PU-GOSS (idea 1) ──────────────────────────────────────────────
print()
print("=" * 70)
print("7. imbgbm — Focal + PU-GOSS sampler (idea 1, hard-neg mining)")
t0 = time.time()
run_cli(["train", "--input", TRAIN_CSV, "--output", f"{BENCH_DIR}/m_pugoss.json",
         "--loss", "focal", "--sampler", "pu_goss",
         "--n-rounds", "500", "--learning-rate", "0.05", "--max-depth", "7",
         "--gamma", "2.0", "--alpha", "0.25", "--subsample", "0.30",
         "--early-stopping-rounds", "0"])
elapsed = time.time() - t0
r = run_cli(["predict", "--input", TEST_CSV, "--model", f"{BENCH_DIR}/m_pugoss.json"])
evaluate("imbgbm Focal+PU-GOSS", y_test_true, read_probs(r.stdout), elapsed)

# ── 8. imbgbm PU-aware splitter (idea 2) ────────────────────────────────────
print()
print("=" * 70)
print("8. imbgbm — Focal + PU-aware splitter (idea 2)")
t0 = time.time()
run_cli(["train", "--input", TRAIN_CSV, "--output", f"{BENCH_DIR}/m_pusplit.json",
         "--loss", "focal", "--splitter", "pu",
         "--n-rounds", "500", "--learning-rate", "0.05", "--max-depth", "7",
         "--gamma", "2.0", "--alpha", "0.25", "--subsample", "0.8",
         "--early-stopping-rounds", "0"])
elapsed = time.time() - t0
r = run_cli(["predict", "--input", TEST_CSV, "--model", f"{BENCH_DIR}/m_pusplit.json"])
evaluate("imbgbm Focal+PUSplit", y_test_true, read_probs(r.stdout), elapsed)

# ── 9. imbgbm Focal + Elkan-Noto PU correction (idea 7) ──────────────────────
print()
print("=" * 70)
print("9. imbgbm — Focal + Elkan-Noto PU correction (idea 7)")
t0 = time.time()
run_cli(["train", "--input", TRAIN_CSV, "--output", f"{BENCH_DIR}/m_en.json",
         "--loss", "focal", "--n-rounds", "500", "--learning-rate", "0.05",
         "--max-depth", "7", "--gamma", "2.0", "--alpha", "0.25",
         "--calibrate", "--platt", "--estimate-pu-rate",
         "--subsample", "0.8", "--early-stopping-rounds", "0"])
elapsed = time.time() - t0
r = run_cli(["predict", "--input", TEST_CSV, "--model", f"{BENCH_DIR}/m_en.json",
             "--pu"])
evaluate("imbgbm Focal+ElkanNoto", y_test_true, read_probs(r.stdout), elapsed)

# ── 10. imbgbm Seed expansion (idea 6) ──────────────────────────────────────
print()
print("=" * 70)
print("10. imbgbm — Focal + seed expansion 3% (idea 6)")
t0 = time.time()
run_cli(["train", "--input", TRAIN_CSV, "--output", f"{BENCH_DIR}/m_se.json",
         "--loss", "focal", "--n-rounds", "400", "--learning-rate", "0.05",
         "--max-depth", "7", "--gamma", "2.0", "--alpha", "0.25",
         "--seed-expansion", "0.03",
         "--subsample", "0.8", "--early-stopping-rounds", "0"])
elapsed = time.time() - t0
r = run_cli(["predict", "--input", TEST_CSV, "--model", f"{BENCH_DIR}/m_se.json"])
evaluate("imbgbm Focal+SeedExpand", y_test_true, read_probs(r.stdout), elapsed)

# ── 11. imbgbm combined PU recipe (1 + 6 + 7) ────────────────────────────────
print()
print("=" * 70)
print("11. imbgbm — SeedPU-GBDT ★  (focal + PU-GOSS + seed-exp + Elkan-Noto)")
t0 = time.time()
run_cli(["train", "--input", TRAIN_CSV, "--output", f"{BENCH_DIR}/m_combo.json",
         "--loss", "focal", "--sampler", "pu_goss",
         "--n-rounds", "500", "--learning-rate", "0.04", "--max-depth", "7",
         "--gamma", "2.0", "--alpha", "0.25", "--subsample", "0.30",
         "--seed-expansion", "0.03", "--calibrate", "--platt", "--estimate-pu-rate",
         "--early-stopping-rounds", "0"])
elapsed = time.time() - t0
r = run_cli(["predict", "--input", TEST_CSV, "--model", f"{BENCH_DIR}/m_combo.json",
             "--pu"])
evaluate("imbgbm SeedPU-GBDT ★", y_test_true, read_probs(r.stdout), elapsed)

# ── Summary ──────────────────────────────────────────────────────────────────
print()
print("=" * 78)
print(f"SUMMARY  (Amazon-like PU: 33k train / 6k test, 9 cat features, c={C_RATE:.0%}, "
      f"true pos rate={y_test_true.mean():.1%})")
print("=" * 78)
hdr = (f"{'Model':<26} {'AUC':>6} {'PR-AUC':>7} {'Logloss':>8} "
       f"{'Brier':>6} {'ECE':>6} {'R@1%':>6} {'P@5%':>6} {'t(s)':>5}")
print(hdr); print("-" * len(hdr))
for r in rows:
    print(f"{r['model']:<26} {r['auc']:6.4f} {r['prauc']:7.4f} "
          f"{r['logloss']:8.4f} {r['brier']:6.4f} {r['ece']:6.4f} "
          f"{r['recall_1pct_fpr']:6.4f} {r['p_at_5pct']:6.4f} "
          f"{r['time_s']:5.1f}")

# Highlight: best imbgbm PU result vs best CatBoost
best_cb = max(rows[:2], key=lambda r: r['auc'])
best_im = max(rows[2:], key=lambda r: r['auc'])
print()
print(f"Best imbgbm (`{best_im['model']}`) vs best CatBoost (`{best_cb['model']}`):")
for k, label, higher_better in [
    ("auc", "AUC", True), ("prauc", "PR-AUC", True),
    ("recall_1pct_fpr", "Recall@FPR=1%", True),
    ("p_at_5pct", "Precision@5%", True),
    ("logloss", "Log-loss", False),
    ("brier", "Brier", False), ("ece", "ECE", False),
]:
    cb_v, im_v = best_cb[k], best_im[k]
    if higher_better:
        pct = 100.0 * (im_v - cb_v) / cb_v if cb_v != 0 else 0
    else:
        pct = 100.0 * (cb_v - im_v) / cb_v if cb_v != 0 else 0
    marker = "✓" if pct > 0 else "✗"
    print(f"  {marker} {label:<14}: CatBoost {cb_v:.4f}  vs  imbgbm {im_v:.4f}  ({pct:+.1f}%)")
