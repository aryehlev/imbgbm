#!/usr/bin/env python3
"""
RankCal-PUGBDT: Two-stage stacked ranker-calibrator for PU learning.

Architecture
────────────
  M_rank = Focal+leaf       → rank_score  (audience building, top-N retrieval)
  M_cal  = Focal+PU-GOSS    → p_bid       (RTB bidding probability)

Training procedure (no leakage)
  1. 5-fold OOF on M_rank (Focal+leaf) → per-row OOF rank_score for every
     training row.
  2. Augment training features: [original_features | rank_score, rank_logit,
     rank_percentile] and train M_cal (Focal+PU-GOSS) on augmented features.
  3. Train a full M_rank on all training data (for test-time inference).

Inference
  rank_score = M_rank(x)                       # audience building
  p_bid      = M_cal( x ‖ rank_features(x) )   # RTB bid price

Conditions benchmarked
  A  CatBoost default        (PU-blind baseline)
  B  imbgbm Focal+leaf       (best-AUC baseline)
  C  imbgbm Focal+PU-GOSS    (best-ECE baseline)
  D  RankCal-Naive           weighted ensemble of B + C
  E  RankCal-Stacked ★       two-stage stacked model (this paper)
  F  RankCal-Filter          top-25% retrieval by rank_score → PU-GOSS rerank
  G  Two-output policy       rank_score for AUC / p_bid for ECE (best of E)
  H  RankCal-Blend           p_bid × (1 + α × norm_rank_score)
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
BENCH_DIR = "/tmp/imbgbm_rankcal_bench"
os.makedirs(BENCH_DIR, exist_ok=True)

C_RATE = 0.30

# ── Synthetic Amazon-like dataset ─────────────────────────────────────────────
print("Generating synthetic Amazon-employee-access dataset …")

N     = 39_000
CARDS = [7518, 4243, 128, 177, 449, 343, 2358, 67, 343]
NAMES = ["RESOURCE","MGR_ID","ROLE_ROLLUP_1","ROLE_ROLLUP_2","ROLE_DEPTNAME",
         "ROLE_TITLE","ROLE_FAMILY_DESC","ROLE_FAMILY","ROLE_CODE"]

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
b0     = (lo + hi) / 2.0
y_true = (rng.uniform(size=N) < sigmoid(lat + b0)).astype(int)

s_obs = y_true.copy()
keep  = rng.uniform(size=N) < C_RATE
s_obs = (y_true & keep).astype(int)

n_true_pos = int(y_true.sum())
n_seeds    = int(s_obs.sum())
print(f"  N={N}  true_pos={n_true_pos}({y_true.mean():.2%})  "
      f"seeds={n_seeds}({s_obs.mean():.2%})  c≈{n_seeds/max(n_true_pos,1):.2f}")

idx_tr, idx_te = train_test_split(
    np.arange(N), test_size=6_000, random_state=SEED, stratify=y_true)
X_train_cat  = X_raw[idx_tr];   X_test_cat  = X_raw[idx_te]
y_train_true = y_true[idx_tr];  y_test_true = y_true[idx_te]
s_train      = s_obs[idx_tr]
print(f"  train={len(idx_tr)}  test={len(idx_te)}\n")

cat_features = list(range(9))

# ── OOF target encoding (on observed PU labels) ───────────────────────────────
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
X_tr_enc, X_te_enc = target_encode(X_train_cat, s_train, X_test_cat, CARDS)
print(f"  done in {time.time()-t0:.1f}s\n")

TRAIN_CSV = f"{BENCH_DIR}/train.csv"
TEST_CSV  = f"{BENCH_DIR}/test.csv"
np.savetxt(TRAIN_CSV, np.c_[X_tr_enc, s_train], delimiter=",", fmt="%.6f")
np.savetxt(TEST_CSV,  X_te_enc,                  delimiter=",", fmt="%.6f")

# ── Utilities ─────────────────────────────────────────────────────────────────

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
    order = np.argsort(-y_prob)[:k]
    return float(y_true[order].mean())

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

# ── Stage 1: OOF rank scores for M_rank (Focal+leaf) ─────────────────────────

def oof_rank_scores(X_tr, s_tr, n_folds=5):
    """Train Focal+leaf via k-fold CV; return OOF probability for each row."""
    scores = np.zeros(len(X_tr), dtype=np.float32)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    for k, (tr_idx, va_idx) in enumerate(kf.split(X_tr)):
        fold_tr  = f"{BENCH_DIR}/rc_fold{k}_tr.csv"
        fold_va  = f"{BENCH_DIR}/rc_fold{k}_va.csv"
        fold_mdl = f"{BENCH_DIR}/rc_fold{k}_rank.json"
        np.savetxt(fold_tr, np.c_[X_tr[tr_idx], s_tr[tr_idx]], delimiter=",", fmt="%.6f")
        np.savetxt(fold_va, X_tr[va_idx],                        delimiter=",", fmt="%.6f")
        run_cli(["train", "--input", fold_tr, "--output", fold_mdl,
                 "--loss", "focal", "--n-rounds", "400", "--learning-rate", "0.05",
                 "--max-depth", "7", "--gamma", "2.0", "--alpha", "0.25",
                 "--calibrate", "--subsample", "0.8", "--early-stopping-rounds", "0"])
        r = run_cli(["predict", "--input", fold_va, "--model", fold_mdl, "--calibrated"])
        scores[va_idx] = read_probs(r.stdout)
    return scores

def rank_features(scores, ref_scores):
    """
    Build [rank_score, rank_logit, rank_percentile] from raw probabilities.
    ref_scores is the training OOF distribution for percentile lookup.
    """
    p   = scores.clip(1e-7, 1 - 1e-7).astype(np.float32)
    lg  = np.log(p / (1.0 - p)).astype(np.float32)
    pct = (np.searchsorted(np.sort(ref_scores), scores, side="left")
           / len(ref_scores)).astype(np.float32)
    return np.column_stack([p, lg, pct])

# ── A. CatBoost default ────────────────────────────────────────────────────────
print("=" * 72)
print("A. CatBoost — default (PU-blind baseline)")
t0 = time.time()
cb_default = cb.CatBoostClassifier(
    iterations=300, learning_rate=0.1, depth=6,
    cat_features=cat_features, random_seed=SEED, verbose=0)
cb_default.fit(X_train_cat, s_train)
evaluate("A CatBoost default", y_test_true, cb_default.predict_proba(X_test_cat)[:, 1],
         time.time() - t0)

# ── B. imbgbm Focal+leaf (best-AUC single-stage baseline) ────────────────────
print()
print("=" * 72)
print("B. imbgbm — Focal + OOF leaf cal  (best single-stage AUC)")
t0 = time.time()
run_cli(["train", "--input", TRAIN_CSV, "--output", f"{BENCH_DIR}/m_focal.json",
         "--loss", "focal", "--n-rounds", "500", "--learning-rate", "0.05",
         "--max-depth", "7", "--gamma", "2.0", "--alpha", "0.25",
         "--calibrate", "--subsample", "0.8", "--early-stopping-rounds", "0"])
elapsed_b = time.time() - t0
r_b = run_cli(["predict", "--input", TEST_CSV, "--model", f"{BENCH_DIR}/m_focal.json",
               "--calibrated"])
probs_b = read_probs(r_b.stdout)
evaluate("B imbgbm Focal+leaf", y_test_true, probs_b, elapsed_b)

# ── C. imbgbm Focal+PU-GOSS (best-ECE single-stage baseline) ─────────────────
print()
print("=" * 72)
print("C. imbgbm — Focal + PU-GOSS  (best single-stage ECE)")
t0 = time.time()
run_cli(["train", "--input", TRAIN_CSV, "--output", f"{BENCH_DIR}/m_pugoss.json",
         "--loss", "focal", "--sampler", "pu_goss",
         "--n-rounds", "500", "--learning-rate", "0.05", "--max-depth", "7",
         "--gamma", "2.0", "--alpha", "0.25", "--subsample", "0.30",
         "--early-stopping-rounds", "0"])
elapsed_c = time.time() - t0
r_c = run_cli(["predict", "--input", TEST_CSV, "--model", f"{BENCH_DIR}/m_pugoss.json"])
probs_c = read_probs(r_c.stdout)
evaluate("C imbgbm Focal+PU-GOSS", y_test_true, probs_c, elapsed_c)

# ── Generate OOF rank scores for training set ─────────────────────────────────
print()
print("=" * 72)
print("Generating 5-fold OOF rank scores for stacking (no leakage) …")
t0 = time.time()
oof_scores = oof_rank_scores(X_tr_enc, s_train, n_folds=5)
print(f"  OOF done in {time.time()-t0:.1f}s  "
      f"mean={oof_scores.mean():.4f}  max={oof_scores.max():.4f}")

# Reference distribution for test-time percentile lookup
sorted_oof = np.sort(oof_scores)

# Train full M_rank on all training data (for test-time predictions)
print("Training full M_rank on all training data …")
t0 = time.time()
run_cli(["train", "--input", TRAIN_CSV, "--output", f"{BENCH_DIR}/m_rank_full.json",
         "--loss", "focal", "--n-rounds", "500", "--learning-rate", "0.05",
         "--max-depth", "7", "--gamma", "2.0", "--alpha", "0.25",
         "--calibrate", "--subsample", "0.8", "--early-stopping-rounds", "0"])
r_full = run_cli(["predict", "--input", TEST_CSV, "--model", f"{BENCH_DIR}/m_rank_full.json",
                  "--calibrated"])
test_rank_scores = read_probs(r_full.stdout)
print(f"  full M_rank done in {time.time()-t0:.1f}s")

# Build augmented datasets
RF_TR = rank_features(oof_scores,       sorted_oof)   # training: OOF (no leak)
RF_TE = rank_features(test_rank_scores, sorted_oof)   # test: full-model scores

X_tr_aug = np.c_[X_tr_enc, RF_TR]
X_te_aug = np.c_[X_te_enc, RF_TE]

TRAIN_AUG_CSV = f"{BENCH_DIR}/train_aug.csv"
TEST_AUG_CSV  = f"{BENCH_DIR}/test_aug.csv"
np.savetxt(TRAIN_AUG_CSV, np.c_[X_tr_aug, s_train], delimiter=",", fmt="%.6f")
np.savetxt(TEST_AUG_CSV,  X_te_aug,                  delimiter=",", fmt="%.6f")
print()

# ── D. RankCal-Naive (weighted ensemble B + C, no stacking) ──────────────────
print("=" * 72)
print("D. RankCal-Naive — weighted ensemble of B (rank) and C (calibration)")
# α=0.7 from rank, 0.3 from calibrator (tune-free heuristic)
t0_d = time.time()
alpha_d = 0.7
probs_d = alpha_d * probs_b + (1.0 - alpha_d) * probs_c
evaluate("D RankCal-Naive", y_test_true, probs_d, time.time() - t0_d)

# ── E. RankCal-Stacked ★  (M_cal trained on augmented features) ───────────────
print()
print("=" * 72)
print("E. RankCal-Stacked ★  — M_cal = Focal+PU-GOSS on [features | OOF rank]")
t0 = time.time()
run_cli(["train", "--input", TRAIN_AUG_CSV, "--output", f"{BENCH_DIR}/m_cal.json",
         "--loss", "focal", "--sampler", "pu_goss",
         "--n-rounds", "500", "--learning-rate", "0.05", "--max-depth", "7",
         "--gamma", "2.0", "--alpha", "0.25", "--subsample", "0.30",
         "--early-stopping-rounds", "0"])
elapsed_e = time.time() - t0
r_e = run_cli(["predict", "--input", TEST_AUG_CSV, "--model", f"{BENCH_DIR}/m_cal.json"])
probs_e = read_probs(r_e.stdout)
evaluate("E RankCal-Stacked ★", y_test_true, probs_e, elapsed_e)

# ── F. RankCal-Filter  (top 25% retrieval → PU-GOSS rerank on subset) ─────────
print()
print("=" * 72)
print("F. RankCal-Filter — top-25% retrieval by rank_score → M_cal rerank")
t0_f = time.time()
# Retrieve top 25% candidates from the test set by rank_score
K_FRAC = 0.25
k_cut  = int(len(y_test_true) * K_FRAC)
order  = np.argsort(-test_rank_scores)
top_k  = order[:k_cut]

# Score all test rows: candidates get M_cal score, rest keep rank_score (low)
probs_f = test_rank_scores.copy()
probs_f[top_k] = probs_e[top_k]   # replace with M_cal for shortlisted rows
evaluate("F RankCal-Filter", y_test_true, probs_f, time.time() - t0_f)

# ── G. Two-output policy (rank_score for AUC / p_bid for ECE) ─────────────────
print()
print("=" * 72)
print("G. Two-output policy — rank_score for ranking | p_bid for bidding")
# We evaluate each output on its intended objective:
#   - M_rank (probs_b) for AUC/P@5%
#   - M_cal  (probs_e) for ECE/calibration
# Here we report M_rank since AUC is the primary ranking metric.
# Below we print a side-by-side comparison.
t0_g = time.time()
print("   Ranker output (rank_score):")
print("  ", end="")
# evaluate in a sub-scope so we can capture both
y_t = np.asarray(y_test_true, dtype=float)
p_rank = probs_b.clip(1e-7, 1-1e-7)
p_bid  = probs_e.clip(1e-7, 1-1e-7)
out_g = dict(
    model="G Two-output policy", time_s=time.time() - t0_g,
    auc     = roc_auc_score(y_t, p_rank),
    prauc   = average_precision_score(y_t, p_rank),
    logloss = log_loss(y_t, p_bid),
    brier   = brier_score_loss(y_t, p_bid),
    ece     = ece(y_t, p_bid),
    r1pct   = recall_at_fpr(y_t, p_rank, 0.01),
    p5pct   = precision_at_k(y_t, p_rank, 0.05),
)
print(f"  Rank: AUC={out_g['auc']:.4f} R@1%={out_g['r1pct']:.4f} P@5%={out_g['p5pct']:.4f}")
print(f"  Bid:  ECE={out_g['ece']:.4f} Brier={out_g['brier']:.4f} LL={out_g['logloss']:.4f}")
rows.append(out_g)

# ── H. RankCal-Blend  p_bid × (1 + α × norm_rank) ───────────────────────────
print()
print("=" * 72)
print("H. RankCal-Blend — p_bid × (1 + α × norm_rank_score)")
t0_h = time.time()
norm_rank = (test_rank_scores - test_rank_scores.min()) / (
    test_rank_scores.max() - test_rank_scores.min() + 1e-9)
alpha_h  = 0.15
probs_h  = probs_e * (1.0 + alpha_h * norm_rank)
probs_h  = probs_h.clip(0.0, 1.0)
evaluate("H RankCal-Blend", y_test_true, probs_h, time.time() - t0_h)

# ── Summary ───────────────────────────────────────────────────────────────────
print()
print("=" * 78)
print(f"RANKCAL-PUGBDT SUMMARY  (Amazon-like PU: c={C_RATE:.0%}, "
      f"true_pos={y_test_true.mean():.1%})")
print()
print("Two objectives:")
print("  rank_score → audience building (maximise AUC / Precision@5%)")
print("  p_bid      → RTB bidding       (minimise ECE / Brier)")
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
best_e = rows[4]   # E: stacked

print("RankCal-Stacked ★  vs  CatBoost default:")
for k, label, higher_better in [
    ("auc",     "AUC",          True),
    ("prauc",   "PR-AUC",       True),
    ("r1pct",   "Recall@FPR=1%",True),
    ("p5pct",   "Precision@5%", True),
    ("logloss", "Log-loss",     False),
    ("brier",   "Brier",        False),
    ("ece",     "ECE",          False),
]:
    cb_v, im_v = cb_row[k], best_e[k]
    if higher_better:
        pct = 100.0 * (im_v - cb_v) / max(abs(cb_v), 1e-9)
    else:
        pct = 100.0 * (cb_v - im_v) / max(abs(cb_v), 1e-9)
    marker = "✓" if pct > 0 else "✗"
    print(f"  {marker} {label:<15}: CatBoost {cb_v:.4f}  RankCal {im_v:.4f}  ({pct:+.1f}%)")

print()
print("Key result: Does stacking preserve ranking power AND fix calibration?")
b_auc  = rows[1]["auc"];  b_ece  = rows[1]["ece"]   # Focal+leaf
c_auc  = rows[2]["auc"];  c_ece  = rows[2]["ece"]   # Focal+PU-GOSS
e_auc  = rows[4]["auc"];  e_ece  = rows[4]["ece"]   # Stacked
print(f"  Focal+leaf AUC={b_auc:.4f}  ECE={b_ece:.4f}")
print(f"  Focal+PUGOSS AUC={c_auc:.4f}  ECE={c_ece:.4f}")
print(f"  RankCal-Stacked AUC={e_auc:.4f}  ECE={e_ece:.4f}")
auc_vs_leaf  = 100*(e_auc  - b_auc)  / max(b_auc,  1e-9)
ece_vs_pugos = 100*(c_ece  - e_ece)  / max(c_ece,  1e-9)
print(f"  → AUC vs Focal+leaf:    {auc_vs_leaf:+.1f}%")
print(f"  → ECE vs Focal+PU-GOSS: {ece_vs_pugos:+.1f}% (positive = better)")
