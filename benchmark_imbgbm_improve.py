#!/usr/bin/env python3
"""
Focused benchmark: improve imbgbm as a standalone algorithm vs CatBoost.

Axes to explore:
  1. Prediction mode: --calibrated (per-leaf avg) vs --platt (cumulative score)
  2. Loss: focal vs bce
  3. Hyperparameters: lambda, min-samples-leaf, max-depth, n-rounds
  4. Features: more interactions, count encoding, lower smoothing
"""

import os, sys, subprocess, time
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss, brier_score_loss, roc_curve
import catboost as cb

SEED = 42
rng  = np.random.default_rng(SEED)

IMBGBM    = "/home/user/imbgbm/target/release/imbgbm"
BENCH_DIR = "/tmp/imbgbm_improve_bench"
os.makedirs(BENCH_DIR, exist_ok=True)

# ── Same dataset as benchmark_ultimate.py ─────────────────────────────────────
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

# 9 base OOF target-encoded
X_tr_enc = np.zeros((len(y_train), 9), dtype=np.float32)
X_te_enc = np.zeros((len(y_test),  9), dtype=np.float32)
for j, card in enumerate(CARDS):
    X_tr_enc[:, j], X_te_enc[:, j] = te_col(
        X_train_raw[:, j], y_train, X_test_raw[:, j], card)

# 4 interaction pairs (planted is 4×8)
PAIRS = [(4,8), (0,1), (1,5), (2,3)]
tr_extras, te_extras = [], []
for a, b in PAIRS:
    ctr = X_train_raw[:,a].astype(np.int64)*CARDS[b] + X_train_raw[:,b].astype(np.int64)
    cte = X_test_raw[:,a].astype(np.int64) *CARDS[b] + X_test_raw[:,b].astype(np.int64)
    tr_f, te_f = te_col(ctr, y_train, cte, CARDS[a]*CARDS[b])
    tr_extras.append(tr_f); te_extras.append(te_f)

# Extended features: 4 more pairs + count encoding (log(count+1) per category)
EXTRA_PAIRS = [(0,4), (1,8), (3,6), (5,7)]
for a, b in EXTRA_PAIRS:
    ctr = X_train_raw[:,a].astype(np.int64)*CARDS[b] + X_train_raw[:,b].astype(np.int64)
    cte = X_test_raw[:,a].astype(np.int64) *CARDS[b] + X_test_raw[:,b].astype(np.int64)
    tr_f, te_f = te_col(ctr, y_train, cte, CARDS[a]*CARDS[b])
    tr_extras.append(tr_f); te_extras.append(te_f)

# Count (frequency) encoding for 5 most cardinality-differentiating features
def count_encode(codes_tr, codes_te, card):
    cnt = np.bincount(codes_tr, minlength=card).astype(np.float32)
    total = len(codes_tr)
    return np.log1p(cnt[codes_tr] / total), np.log1p(cnt[codes_te] / total)

for j in [0, 1, 4, 6, 8]:  # high-cardinality cols
    tr_f, te_f = count_encode(X_train_raw[:,j], X_test_raw[:,j], CARDS[j])
    tr_extras.append(tr_f.astype(np.float32))
    te_extras.append(te_f.astype(np.float32))

# Lower smoothing base encoding (2.0 instead of 10.0)
X_tr_enc2 = np.zeros((len(y_train), 9), dtype=np.float32)
X_te_enc2 = np.zeros((len(y_test),  9), dtype=np.float32)
for j, card in enumerate(CARDS):
    X_tr_enc2[:, j], X_te_enc2[:, j] = te_col(
        X_train_raw[:, j], y_train, X_test_raw[:, j], card, smoothing=2.0)

X_tr_13  = np.c_[X_tr_enc,  np.column_stack(tr_extras[:4])]   # original 13 feats
X_te_13  = np.c_[X_te_enc,  np.column_stack(te_extras[:4])]

X_tr_ext = np.c_[X_tr_enc2, np.column_stack(tr_extras)]       # 9+8+5 = 22 feats (low-smooth TE + 8 pairs + 5 count)
X_te_ext = np.c_[X_te_enc2, np.column_stack(te_extras)]

print(f"  Base (13 feat): {X_tr_13.shape[1]} cols")
print(f"  Extended (22 feat): {X_tr_ext.shape[1]} cols\n")

FEAT_TR_13  = f"{BENCH_DIR}/feat13_tr.csv"
FEAT_TE_13  = f"{BENCH_DIR}/feat13_te.csv"
FEAT_TR_EXT = f"{BENCH_DIR}/featext_tr.csv"
FEAT_TE_EXT = f"{BENCH_DIR}/featext_te.csv"
np.savetxt(FEAT_TR_13,  np.c_[X_tr_13,  y_train], delimiter=",", fmt="%.6f")
np.savetxt(FEAT_TE_13,  X_te_13,                   delimiter=",", fmt="%.6f")
np.savetxt(FEAT_TR_EXT, np.c_[X_tr_ext, y_train], delimiter=",", fmt="%.6f")
np.savetxt(FEAT_TE_EXT, X_te_ext,                  delimiter=",", fmt="%.6f")

# ── REF: CatBoost ─────────────────────────────────────────────────────────────
print("=" * 72)
print("REF 1. CatBoost tuned (original cat features)")
t0 = time.time()
cat_features = list(range(9))
cb_full = cb.CatBoostClassifier(
    iterations=1000, learning_rate=0.03, depth=8,
    l2_leaf_reg=3.0, bagging_temperature=0.5, random_strength=0.5,
    border_count=254, min_data_in_leaf=10,
    cat_features=cat_features, random_seed=SEED, verbose=0)
cb_full.fit(X_train_raw, y_train)
evaluate("CatBoost tuned", y_test, cb_full.predict_proba(X_test_raw)[:,1], time.time()-t0)

# ── imbgbm configs ────────────────────────────────────────────────────────────
def train_predict(name, feat_tr, feat_te, train_args, predict_flag):
    mpath = f"{BENCH_DIR}/{name.replace(' ','_')}.json"
    t0 = time.time()
    run_cli(["train", "--input", feat_tr, "--output", mpath] + train_args)
    r = run_cli(["predict", "--input", feat_te, "--model", mpath, predict_flag])
    return read_probs(r.stdout), time.time() - t0

def focal_args(depth=7, lam=1.0, min_leaf=20, lr=0.03, extra_flags=None):
    args = [
        "--loss", "focal", "--n-rounds", "1000", "--learning-rate", str(lr),
        "--max-depth", str(depth), "--gamma", "2.0", "--alpha", "0.25",
        "--sampler", "adaptive", "--subsample", "0.5", "--col-subsample", "0.8",
        "--lambda", str(lam), "--min-samples-leaf", str(min_leaf),
        "--early-stopping-rounds", "0",
    ]
    if extra_flags:
        args.extend(extra_flags)
    return args

def bce_args(depth=7, lam=1.0, min_leaf=20, lr=0.03, extra_flags=None):
    args = [
        "--loss", "bce", "--n-rounds", "1000", "--learning-rate", str(lr),
        "--max-depth", str(depth),
        "--sampler", "adaptive", "--subsample", "0.5", "--col-subsample", "0.8",
        "--lambda", str(lam), "--min-samples-leaf", str(min_leaf),
        "--early-stopping-rounds", "0",
    ]
    if extra_flags:
        args.extend(extra_flags)
    return args

print()
print("=" * 72)
print("1. imbgbm Focal+Adapt CALIBRATED (baseline — per-leaf avg)")
p, t = train_predict("im_base_cal", FEAT_TR_13, FEAT_TE_13,
                     focal_args(extra_flags=["--calibrate"]), "--calibrated")
evaluate("imbgbm Focal Calibrated", y_test, p, t)

print()
print("=" * 72)
print("2. imbgbm Focal+Adapt PLATT (cumulative boosted score)")
p, t = train_predict("im_base_platt", FEAT_TR_13, FEAT_TE_13,
                     focal_args(extra_flags=["--calibrate", "--platt"]), "--platt")
evaluate("imbgbm Focal Platt", y_test, p, t)

print()
print("=" * 72)
print("3. imbgbm BCE+Adapt PLATT (bce loss, platt calibration)")
p, t = train_predict("im_bce_platt", FEAT_TR_13, FEAT_TE_13,
                     bce_args(extra_flags=["--calibrate", "--platt"]), "--platt")
evaluate("imbgbm BCE Platt", y_test, p, t)

print()
print("=" * 72)
print("4. imbgbm Focal+Adapt PLATT + tuned (lambda=0.3, min-leaf=10, depth=9)")
p, t = train_predict("im_tuned_platt", FEAT_TR_13, FEAT_TE_13,
                     focal_args(depth=9, lam=0.3, min_leaf=10,
                                extra_flags=["--calibrate", "--platt"]), "--platt")
evaluate("imbgbm Focal Platt+Tuned", y_test, p, t)

print()
print("=" * 72)
print("5. imbgbm BCE+Adapt PLATT + tuned (lambda=0.3, min-leaf=10, depth=9)")
p, t = train_predict("im_bce_tuned_platt", FEAT_TR_13, FEAT_TE_13,
                     bce_args(depth=9, lam=0.3, min_leaf=10,
                              extra_flags=["--calibrate", "--platt"]), "--platt")
evaluate("imbgbm BCE Platt+Tuned", y_test, p, t)

print()
print("=" * 72)
print("6. imbgbm Focal+Adapt PLATT + extended features (22 feats)")
p, t = train_predict("im_ext_platt", FEAT_TR_EXT, FEAT_TE_EXT,
                     focal_args(extra_flags=["--calibrate", "--platt"]), "--platt")
evaluate("imbgbm Focal Platt+Ext22", y_test, p, t)

print()
print("=" * 72)
print("7. imbgbm Focal+Adapt PLATT + extended features + tuned")
p, t = train_predict("im_ext_tuned_platt", FEAT_TR_EXT, FEAT_TE_EXT,
                     focal_args(depth=9, lam=0.3, min_leaf=10,
                                extra_flags=["--calibrate", "--platt"]), "--platt")
evaluate("imbgbm Focal Platt+Ext22+Tuned", y_test, p, t)

print()
print("=" * 72)
print("8. imbgbm BCE+Adapt PLATT + extended features + tuned")
p, t = train_predict("im_bce_ext_tuned_platt", FEAT_TR_EXT, FEAT_TE_EXT,
                     bce_args(depth=9, lam=0.3, min_leaf=10,
                              extra_flags=["--calibrate", "--platt"]), "--platt")
evaluate("imbgbm BCE Platt+Ext22+Tuned", y_test, p, t)

print()
print("=" * 72)
print("9. imbgbm Focal+Adapt CALIBRATED + extended features + tuned")
p, t = train_predict("im_ext_tuned_cal", FEAT_TR_EXT, FEAT_TE_EXT,
                     focal_args(depth=9, lam=0.3, min_leaf=10,
                                extra_flags=["--calibrate"]), "--calibrated")
evaluate("imbgbm Focal Cal+Ext22+Tuned", y_test, p, t)

print()
print("=" * 72)
print("10. imbgbm BCE+Adapt PLATT + ext feats + tuned + deeper lr=0.02")
p, t = train_predict("im_bce_ext_lr02", FEAT_TR_EXT, FEAT_TE_EXT,
                     bce_args(depth=9, lam=0.3, min_leaf=10, lr=0.02,
                              extra_flags=["--calibrate", "--platt"]), "--platt")
evaluate("imbgbm BCE Platt+Ext+lr0.02", y_test, p, t)

# ── Summary ───────────────────────────────────────────────────────────────────
print()
print("=" * 90)
print("IMBGBM IMPROVEMENT BENCHMARK")
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

print(f"{'Metric':<12} {'CatBoost':>10} ", end="")
for r in rows[1:]:
    print(f" {r['model'][:12]:>12}", end="")
print()
print("-" * 120)
for k, label, hb in METRICS:
    print(f"  {label:<10} {cb_r[k]:9.4f}  ", end="")
    for r in rows[1:]:
        v = r[k]
        best_excl_cb = max(rows[1:], key=lambda x: x[k] if hb else -x[k])
        mark = "◄" if abs(v - best_excl_cb[k]) < 1e-9 else " "
        beats_cb = (v > cb_r[k] + 1e-6) if hb else (v < cb_r[k] - 1e-6)
        flag = "★" if beats_cb else " "
        print(f" {flag}{v:10.4f}{mark}", end="")
    print()

print()
print("Key: ◄ = best among imbgbm variants  ★ = beats CatBoost")
print()
for r in rows[1:]:
    wins_vs_cb = sum(1 for k,_,hb in METRICS
                     if ((r[k] > cb_r[k] + 1e-6) if hb else (r[k] < cb_r[k] - 1e-6)))
    print(f"  {r['model']:<38}  beats CatBoost on {wins_vs_cb}/7 metrics")
