#!/usr/bin/env python3
"""
RankCal-PUGBDT on the fully-labeled CatBoost benchmark.

Same Amazon-like dataset as benchmark_amazon.py (all positives labeled, no
PU contamination). Tests whether the two-stage stacked architecture adds
value even with complete labels.

  M_rank = Focal+leaf         → rank_score
  M_cal  = Focal+Adaptive     → calibrated probability (complete-label version)
"""

import os, sys, subprocess, time
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score, log_loss, brier_score_loss,
    roc_curve,
)
import catboost as cb

SEED = 42
rng  = np.random.default_rng(SEED)

IMBGBM    = "/home/user/imbgbm/target/release/imbgbm"
BENCH_DIR = "/tmp/imbgbm_rankcal_full_bench"
os.makedirs(BENCH_DIR, exist_ok=True)

# ── Dataset (identical to benchmark_amazon.py) ─────────────────────────────
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

target_pos = 0.06
lo, hi = -30.0, 30.0
for _ in range(80):
    mid = (lo + hi) / 2.0
    if sigmoid(lat + mid).mean() > target_pos: hi = mid
    else:                                       lo = mid
b0 = (lo + hi) / 2.0
y  = (rng.uniform(size=N) < sigmoid(lat + b0)).astype(int)

X_train_cat, X_test_cat, y_train, y_test = train_test_split(
    X_raw, y, test_size=6_000, random_state=SEED, stratify=y)
print(f"  Train={len(y_train)}  pos={y_train.mean():.2%}  "
      f"Test={len(y_test)}  pos={y_test.mean():.2%}\n")

cat_features = list(range(9))

# ── OOF target encoding ────────────────────────────────────────────────────
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

print("OOF target encoding …")
t0 = time.time()
X_tr_enc, X_te_enc = target_encode(X_train_cat, y_train, X_test_cat, CARDS)
print(f"  done in {time.time()-t0:.1f}s\n")

TRAIN_CSV = f"{BENCH_DIR}/train.csv"
TEST_CSV  = f"{BENCH_DIR}/test.csv"
np.savetxt(TRAIN_CSV, np.c_[X_tr_enc, y_train], delimiter=",", fmt="%.6f")
np.savetxt(TEST_CSV,  X_te_enc,                  delimiter=",", fmt="%.6f")

# ── Utilities ──────────────────────────────────────────────────────────────
def ece(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1); total = len(y_true); err = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (y_prob >= lo) & (y_prob < hi)
        if m.sum() == 0: continue
        err += m.sum() * abs(y_true[m].mean() - y_prob[m].mean())
    return err / total

def recall_at_fpr(y_true, y_prob, target_fpr=0.01):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    idx = np.searchsorted(fpr, target_fpr, side="right") - 1
    return float(tpr[max(idx, 0)])

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
        r1pct   = recall_at_fpr(y_true, y_prob, 0.01),
        p5pct   = precision_at_k(y_true, y_prob, 0.05),
    )
    print(f"  AUC={out['auc']:.4f}  PR-AUC={out['prauc']:.4f}  "
          f"ECE={out['ece']:.4f}  R@1%={out['r1pct']:.4f}  "
          f"P@5%={out['p5pct']:.4f}  t={elapsed:.1f}s")
    rows.append(out)

def run_cli(args):
    r = subprocess.run([IMBGBM] + args, capture_output=True, text=True)
    if r.returncode != 0:
        print("imbgbm stderr:", r.stderr[-3000:]); sys.exit(1)
    return r

def read_probs(stdout):
    return np.array([float(x) for x in stdout.strip().splitlines()])

# ── A. CatBoost default ────────────────────────────────────────────────────
print("=" * 72)
print("A. CatBoost — default (300 rounds, native cat features)")
t0 = time.time()
cb_default = cb.CatBoostClassifier(
    iterations=300, learning_rate=0.1, depth=6,
    cat_features=cat_features, random_seed=SEED, verbose=0)
cb_default.fit(X_train_cat, y_train)
probs_a = cb_default.predict_proba(X_test_cat)[:, 1]
evaluate("A CatBoost default", y_test, probs_a, time.time() - t0)

# ── B. CatBoost 800r ───────────────────────────────────────────────────────
print()
print("=" * 72)
print("B. CatBoost — 800 rounds (matched time budget)")
t0 = time.time()
cb_long = cb.CatBoostClassifier(
    iterations=800, learning_rate=0.05, depth=6,
    cat_features=cat_features, random_seed=SEED, verbose=0)
cb_long.fit(X_train_cat, y_train)
probs_b = cb_long.predict_proba(X_test_cat)[:, 1]
evaluate("B CatBoost 800r", y_test, probs_b, time.time() - t0)

# ── C. imbgbm Focal+leaf ───────────────────────────────────────────────────
print()
print("=" * 72)
print("C. imbgbm — Focal + OOF leaf cal  (best single-stage ranker)")
t0 = time.time()
run_cli(["train", "--input", TRAIN_CSV, "--output", f"{BENCH_DIR}/m_focal.json",
         "--loss", "focal", "--n-rounds", "500", "--learning-rate", "0.05",
         "--max-depth", "7", "--gamma", "2.0", "--alpha", "0.25",
         "--calibrate", "--subsample", "0.8", "--early-stopping-rounds", "0"])
elapsed_c = time.time() - t0
r_c = run_cli(["predict", "--input", TEST_CSV, "--model", f"{BENCH_DIR}/m_focal.json",
               "--calibrated"])
probs_c = read_probs(r_c.stdout)
evaluate("C imbgbm Focal+leaf", y_test, probs_c, elapsed_c)

# ── D. imbgbm Focal+Adaptive ───────────────────────────────────────────────
print()
print("=" * 72)
print("D. imbgbm — Focal + Adaptive sampling + OOF leaf cal  (800r)")
t0 = time.time()
run_cli(["train", "--input", TRAIN_CSV, "--output", f"{BENCH_DIR}/m_adapt.json",
         "--loss", "focal", "--n-rounds", "800", "--learning-rate", "0.04",
         "--max-depth", "7", "--gamma", "2.0", "--alpha", "0.25",
         "--calibrate", "--sampler", "adaptive", "--subsample", "0.5",
         "--early-stopping-rounds", "0"])
elapsed_d = time.time() - t0
r_d = run_cli(["predict", "--input", TEST_CSV, "--model", f"{BENCH_DIR}/m_adapt.json",
               "--calibrated"])
probs_d = read_probs(r_d.stdout)
evaluate("D imbgbm Focal+Adapt", y_test, probs_d, elapsed_d)

# ── OOF rank score generation ──────────────────────────────────────────────
print()
print("=" * 72)
print("Generating 5-fold OOF rank scores (no leakage) …")

def oof_rank_scores(X_tr, y_tr, n_folds=5):
    scores = np.zeros(len(X_tr), dtype=np.float32)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    for k, (tr_idx, va_idx) in enumerate(kf.split(X_tr)):
        fold_tr  = f"{BENCH_DIR}/rc_fold{k}_tr.csv"
        fold_va  = f"{BENCH_DIR}/rc_fold{k}_va.csv"
        fold_mdl = f"{BENCH_DIR}/rc_fold{k}_rank.json"
        np.savetxt(fold_tr, np.c_[X_tr[tr_idx], y_tr[tr_idx]], delimiter=",", fmt="%.6f")
        np.savetxt(fold_va, X_tr[va_idx],                        delimiter=",", fmt="%.6f")
        run_cli(["train", "--input", fold_tr, "--output", fold_mdl,
                 "--loss", "focal", "--n-rounds", "400", "--learning-rate", "0.05",
                 "--max-depth", "7", "--gamma", "2.0", "--alpha", "0.25",
                 "--calibrate", "--subsample", "0.8", "--early-stopping-rounds", "0"])
        r = run_cli(["predict", "--input", fold_va, "--model", fold_mdl, "--calibrated"])
        scores[va_idx] = read_probs(r.stdout)
    return scores

t0 = time.time()
oof_scores = oof_rank_scores(X_tr_enc, y_train)
print(f"  done in {time.time()-t0:.1f}s  "
      f"OOF AUC={roc_auc_score(y_train, oof_scores):.4f}")

# Full M_rank for test-time predictions
print("Training full M_rank on all training data …")
t0 = time.time()
run_cli(["train", "--input", TRAIN_CSV, "--output", f"{BENCH_DIR}/m_rank_full.json",
         "--loss", "focal", "--n-rounds", "500", "--learning-rate", "0.05",
         "--max-depth", "7", "--gamma", "2.0", "--alpha", "0.25",
         "--calibrate", "--subsample", "0.8", "--early-stopping-rounds", "0"])
r_full = run_cli(["predict", "--input", TEST_CSV, "--model", f"{BENCH_DIR}/m_rank_full.json",
                  "--calibrated"])
test_rank_scores = read_probs(r_full.stdout)
print(f"  done in {time.time()-t0:.1f}s")

def rank_features(scores, ref_scores):
    p   = scores.clip(1e-7, 1 - 1e-7).astype(np.float32)
    lg  = np.log(p / (1.0 - p)).astype(np.float32)
    pct = (np.searchsorted(np.sort(ref_scores), scores, side="left")
           / len(ref_scores)).astype(np.float32)
    return np.column_stack([p, lg, pct])

RF_TR = rank_features(oof_scores,       oof_scores)
RF_TE = rank_features(test_rank_scores, oof_scores)

X_tr_aug = np.c_[X_tr_enc, RF_TR]
X_te_aug = np.c_[X_te_enc, RF_TE]

TRAIN_AUG_CSV = f"{BENCH_DIR}/train_aug.csv"
TEST_AUG_CSV  = f"{BENCH_DIR}/test_aug.csv"
np.savetxt(TRAIN_AUG_CSV, np.c_[X_tr_aug, y_train], delimiter=",", fmt="%.6f")
np.savetxt(TEST_AUG_CSV,  X_te_aug,                  delimiter=",", fmt="%.6f")
print()

# ── E. RankCal-Naive (ensemble C+D) ───────────────────────────────────────
print("=" * 72)
print("E. RankCal-Naive — weighted ensemble of C (rank) + D (calibrated)")
t0_e = time.time()
probs_e = 0.6 * probs_c + 0.4 * probs_d
evaluate("E RankCal-Naive", y_test, probs_e, time.time() - t0_e)

# ── F. RankCal-Stacked ★ (M_cal trained on augmented features) ────────────
print()
print("=" * 72)
print("F. RankCal-Stacked ★  — M_cal = Focal+Adapt on [features | OOF rank]")
t0 = time.time()
run_cli(["train", "--input", TRAIN_AUG_CSV, "--output", f"{BENCH_DIR}/m_cal.json",
         "--loss", "focal", "--n-rounds", "800", "--learning-rate", "0.04",
         "--max-depth", "7", "--gamma", "2.0", "--alpha", "0.25",
         "--calibrate", "--sampler", "adaptive", "--subsample", "0.5",
         "--early-stopping-rounds", "0"])
elapsed_f = time.time() - t0
r_f = run_cli(["predict", "--input", TEST_AUG_CSV, "--model", f"{BENCH_DIR}/m_cal.json",
               "--calibrated"])
probs_f = read_probs(r_f.stdout)
evaluate("F RankCal-Stacked ★", y_test, probs_f, elapsed_f)

# ── G. RankCal-Filter (top-25% retrieval → M_cal rerank) ──────────────────
print()
print("=" * 72)
print("G. RankCal-Filter — top-25% by rank_score → stacked M_cal rerank")
t0_g = time.time()
K_FRAC = 0.25
k_cut  = int(len(y_test) * K_FRAC)
order  = np.argsort(-test_rank_scores)
top_k  = order[:k_cut]
probs_g = test_rank_scores.copy()
probs_g[top_k] = probs_f[top_k]
evaluate("G RankCal-Filter", y_test, probs_g, time.time() - t0_g)

# ── H. Two-output policy ───────────────────────────────────────────────────
print()
print("=" * 72)
print("H. Two-output policy — rank_score (C) for AUC | p_bid (F) for ECE")
t0_h = time.time()
y_t    = np.asarray(y_test, dtype=float)
p_rank = probs_c.clip(1e-7, 1 - 1e-7)
p_bid  = probs_f.clip(1e-7, 1 - 1e-7)
out_h = dict(
    model="H Two-output policy", time_s=time.time() - t0_h,
    auc     = roc_auc_score(y_t, p_rank),
    prauc   = average_precision_score(y_t, p_rank),
    logloss = log_loss(y_t, p_bid),
    brier   = brier_score_loss(y_t, p_bid),
    ece     = ece(y_t, p_bid),
    r1pct   = recall_at_fpr(y_t, p_rank, 0.01),
    p5pct   = precision_at_k(y_t, p_rank, 0.05),
)
print(f"  Rank: AUC={out_h['auc']:.4f} R@1%={out_h['r1pct']:.4f} "
      f"P@5%={out_h['p5pct']:.4f}")
print(f"  Bid:  ECE={out_h['ece']:.4f} Brier={out_h['brier']:.4f} "
      f"LL={out_h['logloss']:.4f}")
rows.append(out_h)

# ── I. RankCal-Blend ───────────────────────────────────────────────────────
print()
print("=" * 72)
print("I. RankCal-Blend — p_bid × (1 + α × norm_rank_score)")
t0_i = time.time()
norm_rank = (test_rank_scores - test_rank_scores.min()) / (
    test_rank_scores.max() - test_rank_scores.min() + 1e-9)
probs_i = (probs_f * (1.0 + 0.10 * norm_rank)).clip(0.0, 1.0)
evaluate("I RankCal-Blend", y_test, probs_i, time.time() - t0_i)

# ── Summary ────────────────────────────────────────────────────────────────
print()
print("=" * 78)
print(f"RANKCAL-PUGBDT (fully labeled)  — {len(y_train)} train / {len(y_test)} test, "
      f"pos={y_test.mean():.1%}")
print("=" * 78)
hdr = (f"{'Cond':<26} {'AUC':>6} {'PR-AUC':>7} {'ECE':>6} "
       f"{'Brier':>6} {'R@1%':>6} {'P@5%':>6} {'t(s)':>5}")
print(hdr)
print("-" * len(hdr))
for row in rows:
    print(f"{row['model']:<26} {row['auc']:6.4f} {row['prauc']:7.4f} "
          f"{row['ece']:6.4f} {row['brier']:6.4f} "
          f"{row['r1pct']:6.4f} {row['p5pct']:6.4f} "
          f"{row['time_s']:5.1f}")

print()
cb_row = rows[0]   # CatBoost default
best_im = max(rows[2:], key=lambda r: r["auc"])
print(f"Best imbgbm (`{best_im['model']}`) vs CatBoost default:")
for k, label, higher_better in [
    ("auc",     "AUC",          True),
    ("prauc",   "PR-AUC",       True),
    ("r1pct",   "Recall@FPR=1%",True),
    ("p5pct",   "Precision@5%", True),
    ("logloss", "Log-loss",     False),
    ("brier",   "Brier",        False),
    ("ece",     "ECE",          False),
]:
    cb_v, im_v = cb_row[k], best_im[k]
    pct = 100.0 * (im_v - cb_v) / max(abs(cb_v), 1e-9) * (1 if higher_better else -1)
    print(f"  {'✓' if pct > 0 else '✗'} {label:<15}: "
          f"CatBoost {cb_v:.4f}  imbgbm {im_v:.4f}  ({pct:+.1f}%)")

print()
print("Stacked vs single-stage (full-label setting):")
c_row = rows[2]   # C: Focal+leaf
d_row = rows[3]   # D: Focal+Adapt
f_row = rows[5]   # F: stacked
print(f"  Focal+leaf  AUC={c_row['auc']:.4f}  ECE={c_row['ece']:.4f}")
print(f"  Focal+Adapt AUC={d_row['auc']:.4f}  ECE={d_row['ece']:.4f}")
print(f"  Stacked     AUC={f_row['auc']:.4f}  ECE={f_row['ece']:.4f}")
