#!/usr/bin/env python3
"""
Final imbgbm improvement benchmark.

Key insight from previous experiments:
  - Per-leaf calibrated mode: great ECE (0.0004), AUC (0.8595) — but low resolution (std=0.0084)
  - BCE raw + OOF-isotonic: beats CatBoost on 6/7 (loses R@1% due to isotonic ties)
  - Focal OOF-Platt(nested): beats CatBoost on 6/7 (loses ECE 0.0113 > 0.0064)

New approach: BCE + OOF Platt (smooth, no ties → preserves R@1% + good ECE from BCE)
Also trying: better feature engineering to improve all metrics simultaneously.
"""

import os, sys, subprocess, time
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss, brier_score_loss, roc_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
import catboost as cb

SEED = 42
rng  = np.random.default_rng(SEED)

IMBGBM    = "/home/user/imbgbm/target/release/imbgbm"
BENCH_DIR = "/tmp/imbgbm_final_bench"
os.makedirs(BENCH_DIR, exist_ok=True)

# ── Same dataset ───────────────────────────────────────────────────────────────
print("Generating dataset …")
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

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw, y, test_size=6_000, random_state=SEED, stratify=y)
print(f"  train={len(y_train)}  pos={y_train.mean():.2%}  "
      f"test={len(y_test)}  pos={y_test.mean():.2%}\n")

# ── Metrics ───────────────────────────────────────────────────────────────────
def ece(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1); total = len(y_true); err = 0.0
    for lo_, hi_ in zip(bins[:-1], bins[1:]):
        m = (y_prob >= lo_) & (y_prob < hi_)
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
    print(f"  AUC={out['auc']:.4f}  PR-AUC={out['prauc']:.4f}  ECE={out['ece']:.4f}  "
          f"Brier={out['brier']:.4f}  LogLoss={out['logloss']:.4f}  "
          f"R@1%={out['r1pct']:.4f}  P@5%={out['p5pct']:.4f}  t={elapsed:.1f}s")
    rows.append(out)
    return out

def run_cli(args):
    r = subprocess.run([IMBGBM] + args, capture_output=True, text=True)
    if r.returncode != 0:
        print("STDERR:", r.stderr[-2000:]); sys.exit(1)
    return r

def read_probs(stdout):
    return np.array([float(x) for x in stdout.strip().splitlines()])

def te_col(codes_tr, y_tr, codes_te, card, smoothing=10.0):
    gm = float(y_tr.mean())
    cs = np.zeros(card); cc = np.zeros(card)
    np.add.at(cs, codes_tr, y_tr); np.add.at(cc, codes_tr, 1)
    te  = (cs[codes_te] + gm*smoothing) / (cc[codes_te] + smoothing)
    oof = np.full(len(y_tr), gm, dtype=np.float32)
    kf  = KFold(n_splits=5, shuffle=True, random_state=SEED)
    for tri, vai in kf.split(np.arange(len(y_tr))):
        cs2 = np.zeros(card); cc2 = np.zeros(card)
        np.add.at(cs2, codes_tr[tri], y_tr[tri])
        np.add.at(cc2, codes_tr[tri], 1)
        sm2 = (cs2 + gm*smoothing) / (cc2 + smoothing)
        oof[vai] = sm2[codes_tr[vai]]
    return oof.astype(np.float32), te.astype(np.float32)

# ── Feature engineering ───────────────────────────────────────────────────────
print("[A] Building features …")
X_tr_enc = np.zeros((len(y_train), 9), dtype=np.float32)
X_te_enc = np.zeros((len(y_test),  9), dtype=np.float32)
for j, card in enumerate(CARDS):
    X_tr_enc[:, j], X_te_enc[:, j] = te_col(
        X_train_raw[:, j], y_train, X_test_raw[:, j], card)

PAIRS = [(4,8), (0,1), (1,5), (2,3)]
tr_extras, te_extras = [], []
for a, b in PAIRS:
    ctr = X_train_raw[:,a].astype(np.int64)*CARDS[b] + X_train_raw[:,b].astype(np.int64)
    cte = X_test_raw[:,a].astype(np.int64) *CARDS[b] + X_test_raw[:,b].astype(np.int64)
    tr_f, te_f = te_col(ctr, y_train, cte, CARDS[a]*CARDS[b])
    tr_extras.append(tr_f); te_extras.append(te_f)

X_tr_13 = np.c_[X_tr_enc, np.column_stack(tr_extras)]
X_te_13 = np.c_[X_te_enc, np.column_stack(te_extras)]

FEAT_TR = f"{BENCH_DIR}/feat_tr.csv"
FEAT_TE = f"{BENCH_DIR}/feat_te.csv"
np.savetxt(FEAT_TR, np.c_[X_tr_13, y_train], delimiter=",", fmt="%.6f")
np.savetxt(FEAT_TE, X_te_13,                  delimiter=",", fmt="%.6f")
print(f"  {X_tr_13.shape[1]} features\n")

# Helper: run 5-fold OOF + full model, return (oof_raw, test_raw)
def oof_and_full_raw(name, train_args):
    kf5 = KFold(n_splits=5, shuffle=True, random_state=SEED + 3)
    oof = np.zeros(len(y_train), dtype=np.float32)
    for k, (tri, vai) in enumerate(kf5.split(X_tr_13)):
        ft = f"{BENCH_DIR}/{name}_f{k}_tr.csv"
        fv = f"{BENCH_DIR}/{name}_f{k}_va.csv"
        fm = f"{BENCH_DIR}/{name}_f{k}.json"
        np.savetxt(ft, np.c_[X_tr_13[tri], y_train[tri]], delimiter=",", fmt="%.6f")
        np.savetxt(fv, X_tr_13[vai], delimiter=",", fmt="%.6f")
        run_cli(["train", "--input", ft, "--output", fm] + train_args)
        r = run_cli(["predict", "--input", fv, "--model", fm])
        oof[vai] = read_probs(r.stdout)
    # Full model
    fm_full = f"{BENCH_DIR}/{name}_full.json"
    run_cli(["train", "--input", FEAT_TR, "--output", fm_full] + train_args)
    r_te = run_cli(["predict", "--input", FEAT_TE, "--model", fm_full])
    test = read_probs(r_te.stdout)
    return oof, test

def platt_calibrate(oof_raw, y_tr, test_raw):
    lr = LogisticRegression(C=1e6, max_iter=1000)
    lr.fit(oof_raw.reshape(-1, 1), y_tr)
    return lr.predict_proba(test_raw.reshape(-1, 1))[:, 1].astype(np.float32)

def iso_calibrate(oof_raw, y_tr, test_raw):
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(oof_raw, y_tr)
    return iso.predict(test_raw).astype(np.float32)

# ── REF: CatBoost ─────────────────────────────────────────────────────────────
print("=" * 72)
print("REF 1. CatBoost tuned")
t0 = time.time()
cat_features = list(range(9))
cb_full = cb.CatBoostClassifier(
    iterations=1000, learning_rate=0.03, depth=8,
    l2_leaf_reg=3.0, bagging_temperature=0.5, random_strength=0.5,
    border_count=254, min_data_in_leaf=10,
    cat_features=cat_features, random_seed=SEED, verbose=0)
cb_full.fit(X_train_raw, y_train)
evaluate("CatBoost tuned", y_test, cb_full.predict_proba(X_test_raw)[:,1], time.time()-t0)

# ── imbgbm Focal Calibrated (best per prev benchmark) ─────────────────────────
print()
print("=" * 72)
print("1. imbgbm Focal+Adapt CALIBRATED (per-leaf OOF — best standalone)")
t0 = time.time()
FOCAL_ARGS = [
    "--loss", "focal", "--n-rounds", "1000", "--learning-rate", "0.03",
    "--max-depth", "7", "--gamma", "2.0", "--alpha", "0.25",
    "--sampler", "adaptive", "--subsample", "0.5", "--col-subsample", "0.8",
    "--calibrate", "--early-stopping-rounds", "0",
]
run_cli(["train", "--input", FEAT_TR, "--output", f"{BENCH_DIR}/m_focal_cal.json"] + FOCAL_ARGS)
r = run_cli(["predict", "--input", FEAT_TE, "--model", f"{BENCH_DIR}/m_focal_cal.json", "--calibrated"])
p_focal_cal = read_probs(r.stdout)
evaluate("imbgbm Focal Cal", y_test, p_focal_cal, time.time()-t0)

# ── BCE + OOF Platt (main hypothesis) ─────────────────────────────────────────
print()
print("=" * 72)
print("2. imbgbm BCE+Adapt OOF-Platt [new] (smooth calibration → preserves R@1%)")
t0 = time.time()
BCE_ARGS = [
    "--loss", "bce", "--n-rounds", "1000", "--learning-rate", "0.03",
    "--max-depth", "7",
    "--sampler", "adaptive", "--subsample", "0.5", "--col-subsample", "0.8",
    "--early-stopping-rounds", "0",
]
bce_oof, bce_test_raw = oof_and_full_raw("bce", BCE_ARGS)
print(f"  OOF AUC={roc_auc_score(y_train, bce_oof):.4f}  "
      f"ECE(raw)={ece(y_train, np.clip(bce_oof,1e-7,1-1e-7)):.4f}")
p_bce_platt = platt_calibrate(bce_oof, y_train, bce_test_raw)
evaluate("imbgbm BCE OOF-Platt", y_test, p_bce_platt, time.time()-t0)

# ── Focal + OOF Platt ─────────────────────────────────────────────────────────
print()
print("=" * 72)
print("3. imbgbm Focal+Adapt OOF-Platt (focal ranking + smooth Platt)")
t0 = time.time()
FOCAL_BASE_ARGS = [
    "--loss", "focal", "--n-rounds", "1000", "--learning-rate", "0.03",
    "--max-depth", "7", "--gamma", "2.0", "--alpha", "0.25",
    "--sampler", "adaptive", "--subsample", "0.5", "--col-subsample", "0.8",
    "--early-stopping-rounds", "0",
]
focal_oof, focal_test_raw = oof_and_full_raw("focal", FOCAL_BASE_ARGS)
print(f"  OOF AUC={roc_auc_score(y_train, focal_oof):.4f}  "
      f"ECE(raw)={ece(y_train, np.clip(focal_oof,1e-7,1-1e-7)):.4f}")
p_focal_platt = platt_calibrate(focal_oof, y_train, focal_test_raw)
evaluate("imbgbm Focal OOF-Platt", y_test, p_focal_platt, time.time()-t0)

# ── BCE + OOF Isotonic ────────────────────────────────────────────────────────
print()
print("=" * 72)
print("4. imbgbm BCE+Adapt OOF-Isotonic (from previous benchmark — best ECE/Brier)")
t0 = time.time()
p_bce_iso = iso_calibrate(bce_oof, y_train, bce_test_raw)
evaluate("imbgbm BCE OOF-Iso", y_test, p_bce_iso, time.time()-t0)

# ── Focal + OOF Isotonic ──────────────────────────────────────────────────────
print()
print("=" * 72)
print("5. imbgbm Focal+Adapt OOF-Isotonic (focal ranking + isotonic)")
t0 = time.time()
p_focal_iso = iso_calibrate(focal_oof, y_train, focal_test_raw)
evaluate("imbgbm Focal OOF-Iso", y_test, p_focal_iso, time.time()-t0)

# ── Combined: Focal for ranking + BCE for calibration ─────────────────────────
print()
print("=" * 72)
print("6. imbgbm COMBINED: focal ranking preserved + BCE isotonic calibration")
print("   Rank-normalize focal test scores using BCE calibrated distribution")
t0 = time.time()
# Sort focal test scores and assign BCE isotonic calibrated values at same quantile
focal_order = np.argsort(focal_test_raw)
bce_iso_sorted = np.sort(p_bce_iso)
p_combined = np.empty_like(focal_test_raw)
p_combined[focal_order] = bce_iso_sorted
evaluate("imbgbm Focal+BCE-RN", y_test, p_combined, time.time()-t0)

# ── Focal + dual calibration: Platt then isotonic ─────────────────────────────
print()
print("=" * 72)
print("7. imbgbm Focal dual-calib: OOF-Platt THEN nested OOF-Iso (fix residual ECE)")
t0 = time.time()
# Platt-calibrated OOF predictions on training
lr_p = LogisticRegression(C=1e6, max_iter=1000)
lr_p.fit(focal_oof.reshape(-1, 1), y_train)
focal_oof_platt = lr_p.predict_proba(focal_oof.reshape(-1, 1))[:, 1].astype(np.float32)

# Nested isotonic on Platt-OOF to fix residual ECE
iso_dual = IsotonicRegression(out_of_bounds="clip")
iso_dual.fit(focal_oof_platt, y_train)
p_focal_dual = iso_dual.predict(p_focal_platt).astype(np.float32)
evaluate("imbgbm Focal Dual-Calib", y_test, p_focal_dual, time.time()-t0)

# ── BCE + deeper trees + OOF Platt ───────────────────────────────────────────
print()
print("=" * 72)
print("8. imbgbm BCE deeper (depth=9) + OOF-Platt (more capacity)")
t0 = time.time()
BCE_DEEP_ARGS = [
    "--loss", "bce", "--n-rounds", "1000", "--learning-rate", "0.02",
    "--max-depth", "9",
    "--sampler", "adaptive", "--subsample", "0.5", "--col-subsample", "0.8",
    "--lambda", "0.5", "--min-samples-leaf", "15",
    "--early-stopping-rounds", "0",
]
bce_deep_oof, bce_deep_test = oof_and_full_raw("bce_deep", BCE_DEEP_ARGS)
print(f"  OOF AUC={roc_auc_score(y_train, bce_deep_oof):.4f}  "
      f"ECE(raw)={ece(y_train, np.clip(bce_deep_oof,1e-7,1-1e-7)):.4f}")
p_bce_deep_platt = platt_calibrate(bce_deep_oof, y_train, bce_deep_test)
evaluate("imbgbm BCE-deep OOF-Platt", y_test, p_bce_deep_platt, time.time()-t0)

# ── Focal deeper + OOF Platt ─────────────────────────────────────────────────
print()
print("=" * 72)
print("9. imbgbm Focal+Adapt deeper (depth=9, lr=0.02) + OOF-Platt")
t0 = time.time()
FOCAL_DEEP_ARGS = [
    "--loss", "focal", "--n-rounds", "1000", "--learning-rate", "0.02",
    "--max-depth", "9", "--gamma", "2.0", "--alpha", "0.25",
    "--sampler", "adaptive", "--subsample", "0.5", "--col-subsample", "0.8",
    "--lambda", "0.5", "--min-samples-leaf", "15",
    "--early-stopping-rounds", "0",
]
focal_deep_oof, focal_deep_test = oof_and_full_raw("focal_deep", FOCAL_DEEP_ARGS)
print(f"  OOF AUC={roc_auc_score(y_train, focal_deep_oof):.4f}  "
      f"ECE(raw)={ece(y_train, np.clip(focal_deep_oof,1e-7,1-1e-7)):.4f}")
p_focal_deep_platt = platt_calibrate(focal_deep_oof, y_train, focal_deep_test)
evaluate("imbgbm Focal-deep OOF-Platt", y_test, p_focal_deep_platt, time.time()-t0)

# ── Summary ───────────────────────────────────────────────────────────────────
print()
print("=" * 90)
print("FINAL IMBGBM IMPROVEMENT BENCHMARK")
print("=" * 90)
hdr = f"{'Model':<38} {'AUC':>6} {'PR-AUC':>7} {'ECE':>7} {'Brier':>7} {'LogLoss':>8} {'R@1%':>6} {'P@5%':>6}"
print(hdr); print("-" * len(hdr))
for row in rows:
    print(f"{row['model']:<38} {row['auc']:6.4f} {row['prauc']:7.4f} "
          f"{row['ece']:7.4f} {row['brier']:7.4f} {row['logloss']:8.4f} "
          f"{row['r1pct']:6.4f} {row['p5pct']:6.4f}")

print()
cb_r = rows[0]
METRICS = [
    ("auc",     "AUC",     True),
    ("prauc",   "PR-AUC",  True),
    ("r1pct",   "R@1%",    True),
    ("p5pct",   "P@5%",    True),
    ("ece",     "ECE",     False),
    ("brier",   "Brier",   False),
    ("logloss", "LogLoss", False),
]

print(f"\n{'Metric':<12} {'CatBoost':>10}  ", end="")
for r in rows[1:]:
    print(f"{r['model'][:13]:>14}", end="")
print()
print("-" * 140)
for k, label, hb in METRICS:
    all_vals = [r[k] for r in rows]
    best = max(all_vals) if hb else min(all_vals)
    print(f"  {label:<10} {cb_r[k]:9.4f}   ", end="")
    for r in rows[1:]:
        v = r[k]
        mark = "◄" if abs(v - best) < 1e-9 else " "
        beats_cb = (v > cb_r[k] + 1e-6) if hb else (v < cb_r[k] - 1e-6)
        flag = "★" if beats_cb else " "
        print(f" {flag}{v:9.4f}{mark}", end="")
    print()

print()
print("★ = beats CatBoost  ◄ = overall best")
print()
for r in rows[1:]:
    wins = sum(1 for k,_,hb in METRICS
               if ((r[k] > cb_r[k] + 1e-6) if hb else (r[k] < cb_r[k] - 1e-6)))
    markers = []
    for k,_,hb in METRICS:
        beats = (r[k] > cb_r[k] + 1e-6) if hb else (r[k] < cb_r[k] - 1e-6)
        markers.append("✓" if beats else "✗")
    print(f"  {r['model']:<38}  {wins}/7  [{' '.join(markers)}]")
print()
print("  Metrics order: AUC  PR-AUC  R@1%  P@5%  ECE  Brier  LogLoss")
