#!/usr/bin/env python3
"""
Test OOF isotonic calibration of imbgbm raw scores.

Hypothesis: per-leaf averaging has low resolution (CLT collapses predictions).
Fix: use raw boosted scores + OOF isotonic calibration to preserve resolution
while achieving good ECE.
"""

import os, sys, subprocess, time
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss, brier_score_loss, roc_curve
from sklearn.isotonic import IsotonicRegression
import catboost as cb

SEED = 42
rng  = np.random.default_rng(SEED)

IMBGBM    = "/home/user/imbgbm/target/release/imbgbm"
BENCH_DIR = "/tmp/imbgbm_rawiso_bench"
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

def run_cli(args, check=True):
    r = subprocess.run([IMBGBM] + args, capture_output=True, text=True)
    if check and r.returncode != 0:
        print("STDERR:", r.stderr[-2000:]); sys.exit(1)
    return r

def read_probs(stdout):
    return np.array([float(x) for x in stdout.strip().splitlines()])

# ── Feature engineering ───────────────────────────────────────────────────────
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

# ── REF: CatBoost ─────────────────────────────────────────────────────────────
print("=" * 72)
print("REF. CatBoost tuned")
t0 = time.time()
cat_features = list(range(9))
cb_full = cb.CatBoostClassifier(
    iterations=1000, learning_rate=0.03, depth=8,
    l2_leaf_reg=3.0, bagging_temperature=0.5, random_strength=0.5,
    border_count=254, min_data_in_leaf=10,
    cat_features=cat_features, random_seed=SEED, verbose=0)
cb_full.fit(X_train_raw, y_train)
evaluate("CatBoost tuned", y_test, cb_full.predict_proba(X_test_raw)[:,1], time.time()-t0)

# ── imbgbm baseline: per-leaf calibrated ──────────────────────────────────────
print()
print("=" * 72)
print("1. imbgbm Focal+Adapt CALIBRATED (per-leaf OOF avg) — baseline")
t0 = time.time()
TRAIN_ARGS = [
    "--loss", "focal", "--n-rounds", "1000", "--learning-rate", "0.03",
    "--max-depth", "7", "--gamma", "2.0", "--alpha", "0.25",
    "--sampler", "adaptive", "--subsample", "0.5", "--col-subsample", "0.8",
    "--calibrate", "--early-stopping-rounds", "0",
]
run_cli(["train", "--input", FEAT_TR, "--output", f"{BENCH_DIR}/m_cal.json"] + TRAIN_ARGS)
r = run_cli(["predict", "--input", FEAT_TE, "--model", f"{BENCH_DIR}/m_cal.json", "--calibrated"])
p_cal = read_probs(r.stdout)
t_cal = time.time() - t0
evaluate("imbgbm Focal Cal", y_test, p_cal, t_cal)

# ── imbgbm raw sigmoid (no calibration) ──────────────────────────────────────
print()
print("=" * 72)
print("2. imbgbm Focal+Adapt RAW sigmoid (no calibration)")
t0 = time.time()
run_cli(["train", "--input", FEAT_TR, "--output", f"{BENCH_DIR}/m_raw.json",
         "--loss", "focal", "--n-rounds", "1000", "--learning-rate", "0.03",
         "--max-depth", "7", "--gamma", "2.0", "--alpha", "0.25",
         "--sampler", "adaptive", "--subsample", "0.5", "--col-subsample", "0.8",
         "--early-stopping-rounds", "0"])
r = run_cli(["predict", "--input", FEAT_TE, "--model", f"{BENCH_DIR}/m_raw.json"])
p_raw = read_probs(r.stdout)
evaluate("imbgbm Focal Raw", y_test, p_raw, time.time() - t0)

# ── imbgbm OOF isotonic calibration on raw scores ─────────────────────────────
print()
print("=" * 72)
print("3. imbgbm Focal+Adapt OOF-Isotonic on RAW scores (new approach)")
print("   Train 5-fold OOF → get raw scores → isotonic → apply to full-model raw")
t0 = time.time()

OOF_ARGS = [
    "--loss", "focal", "--n-rounds", "1000", "--learning-rate", "0.03",
    "--max-depth", "7", "--gamma", "2.0", "--alpha", "0.25",
    "--sampler", "adaptive", "--subsample", "0.5", "--col-subsample", "0.8",
    "--early-stopping-rounds", "0",
]

kf5 = KFold(n_splits=5, shuffle=True, random_state=SEED)
oof_raw = np.zeros(len(y_train), dtype=np.float32)

for k, (tri, vai) in enumerate(kf5.split(X_tr_13)):
    ft = f"{BENCH_DIR}/oof_f{k}_tr.csv"
    fv = f"{BENCH_DIR}/oof_f{k}_va.csv"
    fm = f"{BENCH_DIR}/oof_f{k}.json"
    np.savetxt(ft, np.c_[X_tr_13[tri], y_train[tri]], delimiter=",", fmt="%.6f")
    np.savetxt(fv, X_tr_13[vai], delimiter=",", fmt="%.6f")
    run_cli(["train", "--input", ft, "--output", fm] + OOF_ARGS)
    r = run_cli(["predict", "--input", fv, "--model", fm])
    oof_raw[vai] = read_probs(r.stdout)
    print(f"  Fold {k+1}/5 OOF AUC={roc_auc_score(y_train[vai], oof_raw[vai]):.4f}", flush=True)

print(f"  Full OOF AUC={roc_auc_score(y_train, oof_raw):.4f}  "
      f"ECE(raw)={ece(y_train, np.clip(oof_raw,1e-7,1-1e-7)):.4f}")

# Fit isotonic on OOF raw sigmoid predictions vs y
iso = IsotonicRegression(out_of_bounds="clip")
iso.fit(oof_raw, y_train)
oof_iso_preds = iso.predict(oof_raw)
print(f"  OOF after isotonic: ECE={ece(y_train, np.clip(oof_iso_preds,1e-7,1-1e-7)):.4f}  "
      f"AUC={roc_auc_score(y_train, oof_iso_preds):.4f}")

# Apply full-model raw to isotonic calibrator
# Full model (trained on all data) → raw predictions on test
r_full = run_cli(["predict", "--input", FEAT_TE, "--model", f"{BENCH_DIR}/m_raw.json"])
p_full_raw = read_probs(r_full.stdout)

# Apply isotonic calibration fit on OOF raw scores
p_raw_iso = iso.predict(p_full_raw).astype(np.float32)
t_rawiso = time.time() - t0
evaluate("imbgbm Focal Raw+OOF-Iso", y_test, p_raw_iso, t_rawiso)

# ── BCE + OOF isotonic on raw scores ─────────────────────────────────────────
print()
print("=" * 72)
print("4. imbgbm BCE+Adapt OOF-Isotonic on RAW scores")
t0 = time.time()

BCE_OOF_ARGS = [
    "--loss", "bce", "--n-rounds", "1000", "--learning-rate", "0.03",
    "--max-depth", "7",
    "--sampler", "adaptive", "--subsample", "0.5", "--col-subsample", "0.8",
    "--early-stopping-rounds", "0",
]

oof_raw_bce = np.zeros(len(y_train), dtype=np.float32)
for k, (tri, vai) in enumerate(kf5.split(X_tr_13)):
    ft = f"{BENCH_DIR}/oof_bce_f{k}_tr.csv"
    fv = f"{BENCH_DIR}/oof_bce_f{k}_va.csv"
    fm = f"{BENCH_DIR}/oof_bce_f{k}.json"
    np.savetxt(ft, np.c_[X_tr_13[tri], y_train[tri]], delimiter=",", fmt="%.6f")
    np.savetxt(fv, X_tr_13[vai], delimiter=",", fmt="%.6f")
    run_cli(["train", "--input", ft, "--output", fm] + BCE_OOF_ARGS)
    r = run_cli(["predict", "--input", fv, "--model", fm])
    oof_raw_bce[vai] = read_probs(r.stdout)

print(f"  Full OOF AUC={roc_auc_score(y_train, oof_raw_bce):.4f}  "
      f"ECE(raw)={ece(y_train, np.clip(oof_raw_bce,1e-7,1-1e-7)):.4f}")

iso_bce = IsotonicRegression(out_of_bounds="clip")
iso_bce.fit(oof_raw_bce, y_train)

# Full BCE model on test
run_cli(["train", "--input", FEAT_TR, "--output", f"{BENCH_DIR}/m_bce_raw.json"] + BCE_OOF_ARGS)
r_bce = run_cli(["predict", "--input", FEAT_TE, "--model", f"{BENCH_DIR}/m_bce_raw.json"])
p_bce_raw = read_probs(r_bce.stdout)
p_bce_iso = iso_bce.predict(p_bce_raw).astype(np.float32)
evaluate("imbgbm BCE Raw+OOF-Iso", y_test, p_bce_iso, time.time() - t0)

# ── Focal OOF-isotonic on raw + nested OOF (no leakage) ────────────────────
print()
print("=" * 72)
print("5. imbgbm Focal NESTED OOF-Iso (train isotonic calibrator on outer OOF)")
print("   Solves train/test raw score distribution shift with nested cross-val")
t0 = time.time()

# Nested: outer 5 folds for calibration, inner OOF for training isotonic
outer_kf = KFold(n_splits=5, shuffle=True, random_state=SEED + 7)
oof_iso_nested = np.zeros(len(y_train), dtype=np.float32)

for k, (tri, vai) in enumerate(outer_kf.split(X_tr_13)):
    # Train on tri, get raw predictions on vai
    ft = f"{BENCH_DIR}/nest_f{k}_tr.csv"
    fv = f"{BENCH_DIR}/nest_f{k}_va.csv"
    fm = f"{BENCH_DIR}/nest_f{k}.json"
    np.savetxt(ft, np.c_[X_tr_13[tri], y_train[tri]], delimiter=",", fmt="%.6f")
    np.savetxt(fv, X_tr_13[vai], delimiter=",", fmt="%.6f")
    run_cli(["train", "--input", ft, "--output", fm] + OOF_ARGS)
    r = run_cli(["predict", "--input", fv, "--model", fm])
    oof_iso_nested[vai] = read_probs(r.stdout)

# Fit isotonic on these nested OOF raw predictions
iso_nested = IsotonicRegression(out_of_bounds="clip")
iso_nested.fit(oof_iso_nested, y_train)

# Apply to full-model test predictions
p_nested_iso = iso_nested.predict(p_full_raw).astype(np.float32)
evaluate("imbgbm Focal Nested-OOF-Iso", y_test, p_nested_iso, time.time() - t0)

# ── Focal: raw score + OOF Platt (nested) ─────────────────────────────────
print()
print("=" * 72)
print("6. imbgbm Focal OOF-Platt on RAW score (nested, no leakage)")
t0 = time.time()
from sklearn.linear_model import LogisticRegression

lr_nested = LogisticRegression(C=1e6, max_iter=1000)
lr_nested.fit(oof_iso_nested.reshape(-1, 1), y_train)  # reuse nested OOF raw
p_oof_platt = lr_nested.predict_proba(p_full_raw.reshape(-1, 1))[:, 1].astype(np.float32)
evaluate("imbgbm Focal OOF-Platt(nested)", y_test, p_oof_platt, time.time() - t0)

# ── Compare: prediction spread (resolution proxy) ─────────────────────────────
print()
print("=" * 72)
print("Resolution analysis (std of predictions = proxy for resolution):")
print(f"  CatBoost raw proba:        std={cb_full.predict_proba(X_test_raw)[:,1].std():.4f}")
print(f"  imbgbm Focal Cal:          std={p_cal.std():.4f}")
print(f"  imbgbm Focal Raw:          std={p_raw.std():.4f}")
print(f"  imbgbm Focal Raw+OOF-Iso:  std={p_raw_iso.std():.4f}")
print(f"  imbgbm BCE Raw+OOF-Iso:    std={p_bce_iso.std():.4f}")
print(f"  imbgbm Focal Nested-Iso:   std={p_nested_iso.std():.4f}")
print(f"  imbgbm Focal OOF-Platt:    std={p_oof_platt.std():.4f}")

# ── Summary ───────────────────────────────────────────────────────────────────
print()
print("=" * 90)
print("RAW-SCORE CALIBRATION BENCHMARK")
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
for r in rows[1:]:
    wins = sum(1 for k,_,hb in METRICS
               if ((r[k] > cb_r[k] + 1e-6) if hb else (r[k] < cb_r[k] - 1e-6)))
    print(f"  {r['model']:<38}  beats CatBoost on {wins}/7 metrics")
