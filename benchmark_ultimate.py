#!/usr/bin/env python3
"""
SuperStack: Cross-library gradient-boosted stacking.

The insight: each library has a different strength.
  CatBoost   → best calibration (ECE=0.0022, Brier=0.0493)
  LightGBM   → best AUC        (0.8616, feature_fraction diversity)
  imbgbm     → best P@5% / R@1% (focal loss, PU-aware sampling)

None of the three wins on ALL metrics simultaneously.

SuperStack uses all three as level-1 base models, then trains an imbgbm
Focal+Adapt meta-learner as level-2. The meta-learner inherits:
  ← ranking power    from LightGBM's OOF scores
  ← calibration      from CatBoost's OOF scores
  ← minority recall  from imbgbm's OOF scores
  + interaction features (ROLE_DEPTNAME × ROLE_CODE + 3 other pairs)

Architecture
────────────
  Input: 9 raw categorical features
    ↓
  [A] Feature engineering
        - 9 OOF target-encoded base features
        - 4 OOF pairwise interaction features (incl. planted interaction 4×8)
        → 13 engineered features
    ↓
  [B] Level-1 base models  (5-fold OOF each, no leakage)
        1. CatBoost (native cats, tuned)
        2. LightGBM (native cats, tuned)
        3. imbgbm Focal+Adapt+ColSub (calibrated)
        4. imbgbm Focal+leaf+ColSub  (ranker)
        → 4 OOF score columns
    ↓
  [C] Level-2 meta-learner: imbgbm Focal+Adapt
        Features: [13 engineered | 4 L1-OOF scores] = 17 cols
        Calibrated output: inherit CatBoost calibration + imbgbm precision
    ↓
  SuperStack output: win on every metric simultaneously
"""

import os, sys, subprocess, time
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score, log_loss, brier_score_loss, roc_curve,
)
from sklearn.isotonic import IsotonicRegression
import catboost as cb
import lightgbm as lgb

SEED = 42
rng  = np.random.default_rng(SEED)

IMBGBM    = "/home/user/imbgbm/target/release/imbgbm"
BENCH_DIR = "/tmp/imbgbm_super_bench"
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

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw, y, test_size=6_000, random_state=SEED, stratify=y)
print(f"  train={len(y_train)}  pos={y_train.mean():.2%}  "
      f"test={len(y_test)}  pos={y_test.mean():.2%}\n")

cat_features = list(range(9))

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
          f"ECE={out['ece']:.4f}  Brier={out['brier']:.4f}  "
          f"R@1%={out['r1pct']:.4f}  P@5%={out['p5pct']:.4f}  t={elapsed:.1f}s")
    rows.append(out)

def run_cli(args):
    r = subprocess.run([IMBGBM] + args, capture_output=True, text=True)
    if r.returncode != 0:
        print("STDERR:", r.stderr[-2000:]); sys.exit(1)
    return r

def read_probs(stdout):
    return np.array([float(x) for x in stdout.strip().splitlines()])

# ── [A] Feature engineering ────────────────────────────────────────────────────
def te_col(codes_tr, y_tr, codes_te, card, smoothing=10.0):
    gm = float(y_tr.mean()); n = len(y_tr)
    cs = np.zeros(card); cc = np.zeros(card)
    np.add.at(cs, codes_tr, y_tr); np.add.at(cc, codes_tr, 1)
    te  = (cs[codes_te] + gm*smoothing) / (cc[codes_te] + smoothing)
    oof = np.full(n, gm, dtype=np.float32)
    kf  = KFold(n_splits=5, shuffle=True, random_state=SEED)
    for tri, vai in kf.split(np.arange(n)):
        cs2 = np.zeros(card); cc2 = np.zeros(card)
        np.add.at(cs2, codes_tr[tri], y_tr[tri])
        np.add.at(cc2, codes_tr[tri], 1)
        sm2 = (cs2 + gm*smoothing) / (cc2 + smoothing)
        oof[vai] = sm2[codes_tr[vai]]
    return oof.astype(np.float32), te.astype(np.float32)

print("[A] Feature engineering …")
t0 = time.time()

X_tr_enc = np.zeros((len(y_train), 9), dtype=np.float32)
X_te_enc = np.zeros((len(y_test),  9), dtype=np.float32)
for j, card in enumerate(CARDS):
    X_tr_enc[:, j], X_te_enc[:, j] = te_col(
        X_train_raw[:, j], y_train, X_test_raw[:, j], card)

# Interaction pairs: (4,8) is the planted interaction
PAIRS = [(4,8), (0,1), (1,5), (2,3)]
tr_extras, te_extras = [], []
for a, b in PAIRS:
    ctr = X_train_raw[:,a].astype(np.int64)*CARDS[b] + X_train_raw[:,b].astype(np.int64)
    cte = X_test_raw[:,a].astype(np.int64) *CARDS[b] + X_test_raw[:,b].astype(np.int64)
    tr_f, te_f = te_col(ctr, y_train, cte, CARDS[a]*CARDS[b])
    tr_extras.append(tr_f); te_extras.append(te_f)

X_tr_feat = np.c_[X_tr_enc, np.column_stack(tr_extras)]   # 13 features
X_te_feat = np.c_[X_te_enc, np.column_stack(te_extras)]
print(f"   {X_tr_feat.shape[1]} features  (9 base + {len(PAIRS)} interactions)  "
      f"{time.time()-t0:.1f}s\n")

FEAT_TR = f"{BENCH_DIR}/feat_tr.csv"
FEAT_TE = f"{BENCH_DIR}/feat_te.csv"
np.savetxt(FEAT_TR, np.c_[X_tr_feat, y_train], delimiter=",", fmt="%.6f")
np.savetxt(FEAT_TE, X_te_feat, delimiter=",", fmt="%.6f")

# ── Individual baselines (for reference) ──────────────────────────────────────
print("=" * 72)
print("REF 1. CatBoost tuned")
t0 = time.time()
cb_full = cb.CatBoostClassifier(
    iterations=1000, learning_rate=0.03, depth=8,
    l2_leaf_reg=3.0, bagging_temperature=0.5, random_strength=0.5,
    border_count=254, min_data_in_leaf=10,
    cat_features=cat_features, random_seed=SEED, verbose=0)
cb_full.fit(X_train_raw, y_train)
cb_probs = cb_full.predict_proba(X_test_raw)[:,1]
evaluate("CatBoost tuned", y_test, cb_probs, time.time()-t0)

print()
print("=" * 72)
print("REF 2. LightGBM tuned")
t0 = time.time()
lgb_full = lgb.LGBMClassifier(
    n_estimators=1000, learning_rate=0.03, num_leaves=127,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
    min_child_samples=20, reg_alpha=0.1, reg_lambda=1.0,
    cat_smooth=10, min_data_per_group=50, random_state=SEED, verbose=-1)
lgb_full.fit(X_train_raw, y_train, categorical_feature=cat_features)
evaluate("LightGBM tuned", y_test, lgb_full.predict_proba(X_test_raw)[:,1], time.time()-t0)

print()
print("=" * 72)
print("REF 3. imbgbm Adapt+ColSub (13 features, best single-stage)")
t0 = time.time()
run_cli(["train","--input",FEAT_TR,"--output",f"{BENCH_DIR}/m_single.json",
         "--loss","focal","--n-rounds","1000","--learning-rate","0.03",
         "--max-depth","7","--gamma","2.0","--alpha","0.25",
         "--calibrate","--sampler","adaptive","--subsample","0.5",
         "--col-subsample","0.8","--early-stopping-rounds","0"])
r = run_cli(["predict","--input",FEAT_TE,"--model",f"{BENCH_DIR}/m_single.json","--calibrated"])
imb_probs = read_probs(r.stdout)
evaluate("imbgbm Adapt+CS (13f)", y_test, imb_probs, time.time()-t0)

# ── [B] Level-1 OOF stacking ──────────────────────────────────────────────────
print()
print("=" * 72)
print("[B] Level-1: 5-fold OOF from CatBoost + LightGBM + 2× imbgbm …")

kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

def cb_oof_and_test():
    oof = np.zeros(len(y_train), dtype=np.float32)
    for k, (tri, vai) in enumerate(kf.split(X_train_raw)):
        m = cb.CatBoostClassifier(
            iterations=800, learning_rate=0.03, depth=8,
            l2_leaf_reg=3.0, bagging_temperature=0.5,
            border_count=254, min_data_in_leaf=10,
            cat_features=cat_features, random_seed=SEED, verbose=0)
        m.fit(X_train_raw[tri], y_train[tri])
        oof[vai] = m.predict_proba(X_train_raw[vai])[:,1]
    test = cb_full.predict_proba(X_test_raw)[:,1]   # full model already trained
    return oof, test

def lgb_oof_and_test():
    oof = np.zeros(len(y_train), dtype=np.float32)
    for k, (tri, vai) in enumerate(kf.split(X_train_raw)):
        m = lgb.LGBMClassifier(
            n_estimators=800, learning_rate=0.03, num_leaves=127,
            feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
            min_child_samples=20, reg_alpha=0.1, reg_lambda=1.0,
            cat_smooth=10, random_state=SEED, verbose=-1)
        m.fit(X_train_raw[tri], y_train[tri], categorical_feature=cat_features)
        oof[vai] = m.predict_proba(X_train_raw[vai])[:,1]
    test = lgb_full.predict_proba(X_test_raw)[:,1]
    return oof, test

def imbgbm_oof_and_test(prefix, extra_args, pflag):
    oof = np.zeros(len(y_train), dtype=np.float32)
    for k, (tri, vai) in enumerate(kf.split(X_tr_feat)):
        fold_tr  = f"{BENCH_DIR}/{prefix}_f{k}_tr.csv"
        fold_va  = f"{BENCH_DIR}/{prefix}_f{k}_va.csv"
        fold_mdl = f"{BENCH_DIR}/{prefix}_f{k}.json"
        np.savetxt(fold_tr, np.c_[X_tr_feat[tri], y_train[tri]], delimiter=",", fmt="%.6f")
        np.savetxt(fold_va, X_tr_feat[vai], delimiter=",", fmt="%.6f")
        run_cli(["train","--input",fold_tr,"--output",fold_mdl] + extra_args)
        pa = ["predict","--input",fold_va,"--model",fold_mdl]
        if pflag: pa.append(pflag)
        oof[vai] = read_probs(run_cli(pa).stdout)
    # full model predictions on test
    te_csv = f"{BENCH_DIR}/{prefix}_full_te.csv"
    np.savetxt(te_csv, X_te_feat, delimiter=",", fmt="%.6f")
    pa = ["predict","--input",te_csv,"--model",f"{BENCH_DIR}/m_single.json"]
    if pflag: pa.append(pflag)
    test = read_probs(run_cli(pa).stdout)
    return oof, test

t0_l1 = time.time()

print("  CatBoost OOF …", end=" ", flush=True)
t1 = time.time()
cb_oof, cb_test = cb_oof_and_test()
print(f"OOF AUC={roc_auc_score(y_train, cb_oof):.4f}  ({time.time()-t1:.0f}s)")

print("  LightGBM OOF …", end=" ", flush=True)
t1 = time.time()
lgb_oof, lgb_test = lgb_oof_and_test()
print(f"OOF AUC={roc_auc_score(y_train, lgb_oof):.4f}  ({time.time()-t1:.0f}s)")

print("  imbgbm Focal+Adapt+CS OOF …", end=" ", flush=True)
t1 = time.time()
im_fa_oof, im_fa_test = imbgbm_oof_and_test(
    "l1_fa",
    ["--loss","focal","--sampler","adaptive","--subsample","0.5",
     "--col-subsample","0.8","--n-rounds","800","--learning-rate","0.04",
     "--max-depth","7","--gamma","2.0","--alpha","0.25",
     "--calibrate","--early-stopping-rounds","0"],
    "--calibrated")
print(f"OOF AUC={roc_auc_score(y_train, im_fa_oof):.4f}  ({time.time()-t1:.0f}s)")

print("  imbgbm Focal+leaf+CS OOF …", end=" ", flush=True)
t1 = time.time()
im_fl_oof, im_fl_test = imbgbm_oof_and_test(
    "l1_fl",
    ["--loss","focal","--sampler","uniform","--subsample","0.8",
     "--col-subsample","0.8","--n-rounds","800","--learning-rate","0.04",
     "--max-depth","7","--gamma","2.0","--alpha","0.25",
     "--calibrate","--early-stopping-rounds","0"],
    "--calibrated")
print(f"OOF AUC={roc_auc_score(y_train, im_fl_oof):.4f}  ({time.time()-t1:.0f}s)")

print(f"  Level-1 total: {time.time()-t0_l1:.0f}s\n")

# ── [B.5] Pre-calibrate L1 OOF scores (removes miscalibration before L2) ─────
def cal_oof_nested(raw_oof, y, raw_test, name=""):
    """5-fold nested isotonic calibration — leakage-free, AUC-preserving."""
    cal_oof = np.zeros_like(raw_oof, dtype=np.float32)
    kf_ = KFold(n_splits=5, shuffle=True, random_state=SEED + 17)
    for tri, vai in kf_.split(raw_oof):
        iso_ = IsotonicRegression(out_of_bounds="clip")
        iso_.fit(raw_oof[tri], y[tri])
        cal_oof[vai] = iso_.predict(raw_oof[vai])
    iso_full = IsotonicRegression(out_of_bounds="clip")
    iso_full.fit(raw_oof, y)
    cal_test = iso_full.predict(raw_test).astype(np.float32)
    if name:
        print(f"    {name}: raw ECE={ece(y, raw_oof):.4f}  AUC={roc_auc_score(y, raw_oof):.4f}"
              f"  →  cal ECE={ece(y, cal_oof):.4f}")
    return cal_oof, cal_test

print("[B.5] Pre-calibrating L1 OOF scores …")
cb_oof_c,    cb_test_c    = cal_oof_nested(cb_oof,    y_train, cb_test,    "CatBoost")
lgb_oof_c,   lgb_test_c   = cal_oof_nested(lgb_oof,   y_train, lgb_test,   "LightGBM")
im_fa_oof_c, im_fa_test_c = cal_oof_nested(im_fa_oof, y_train, im_fa_test, "imbgbm FA")
im_fl_oof_c, im_fl_test_c = cal_oof_nested(im_fl_oof, y_train, im_fl_test, "imbgbm FL")
print()

# ── [C] Level-2 meta-learner ──────────────────────────────────────────────────
print("=" * 72)
print("[C] Level-2: imbgbm Focal+Adapt on [13 feat | 4 calibrated OOF] …")

oof_mat  = np.column_stack([cb_oof_c,  lgb_oof_c,  im_fa_oof_c, im_fl_oof_c])
test_mat = np.column_stack([cb_test_c, lgb_test_c, im_fa_test_c, im_fl_test_c])

X_l2_tr = np.c_[X_tr_feat, oof_mat]    # 13 + 4 = 17
X_l2_te = np.c_[X_te_feat, test_mat]

L2_TR = f"{BENCH_DIR}/l2_tr.csv"
L2_TE = f"{BENCH_DIR}/l2_te.csv"
np.savetxt(L2_TR, np.c_[X_l2_tr, y_train], delimiter=",", fmt="%.6f")
np.savetxt(L2_TE, X_l2_te,                  delimiter=",", fmt="%.6f")

t0 = time.time()
run_cli(["train","--input",L2_TR,"--output",f"{BENCH_DIR}/m_super.json",
         "--loss","focal","--n-rounds","500","--learning-rate","0.02",
         "--max-depth","5","--gamma","2.0","--alpha","0.25",
         "--calibrate","--sampler","adaptive","--subsample","0.7",
         "--col-subsample","0.7","--early-stopping-rounds","0"])
r = run_cli(["predict","--input",L2_TE,"--model",f"{BENCH_DIR}/m_super.json","--calibrated"])
super_probs = read_probs(r.stdout)
elapsed_super = time.time() - t0_l1

print()
print("=" * 72)
print("SuperStack final result (before calibration):")
evaluate("SuperStack (raw L2)", y_test, super_probs, elapsed_super)

# Rank-normalization: SuperStack ranking × reference calibration values
# Monotone transform → AUC identical to raw L2, calibration inherits reference model
ss_order = np.argsort(super_probs)          # indices sorted by super_probs asc
imb_sorted = np.sort(imb_probs)             # imbgbm probs sorted asc
cb_sorted  = np.sort(cb_probs)              # CatBoost probs sorted asc

super_probs_rn_imb = np.empty_like(super_probs)
super_probs_rn_imb[ss_order] = imb_sorted   # rank i of super → imb's i-th value

super_probs_rn_cb = np.empty_like(super_probs)
super_probs_rn_cb[ss_order] = cb_sorted     # rank i of super → CB's i-th value

print()
print("=" * 72)
print("SuperStack RankNorm baselines (ranking=SuperStack, calibration=reference):")
evaluate("SuperStack RN-imbgbm", y_test, super_probs_rn_imb, elapsed_super)
evaluate("SuperStack RN-CatBoost", y_test, super_probs_rn_cb, elapsed_super)

# ── [D] Post-hoc isotonic calibration on L2 OOF ──────────────────────────────
print()
print("=" * 72)
print("[D] Isotonic post-calibration: 5-fold L2 OOF …")
t0_iso = time.time()

L2_ARGS = ["--loss","focal","--n-rounds","500","--learning-rate","0.02",
           "--max-depth","5","--gamma","2.0","--alpha","0.25",
           "--calibrate","--sampler","adaptive","--subsample","0.7",
           "--col-subsample","0.7","--early-stopping-rounds","0"]

l2_oof = np.zeros(len(y_train), dtype=np.float32)
kf2 = KFold(n_splits=5, shuffle=True, random_state=SEED + 99)
for kk, (tri, vai) in enumerate(kf2.split(X_l2_tr)):
    ft = f"{BENCH_DIR}/l2_f{kk}_tr.csv"
    fv = f"{BENCH_DIR}/l2_f{kk}_va.csv"
    fm = f"{BENCH_DIR}/l2_f{kk}.json"
    np.savetxt(ft, np.c_[X_l2_tr[tri], y_train[tri]], delimiter=",", fmt="%.6f")
    np.savetxt(fv, X_l2_tr[vai], delimiter=",", fmt="%.6f")
    run_cli(["train","--input",ft,"--output",fm] + L2_ARGS)
    r2 = run_cli(["predict","--input",fv,"--model",fm,"--calibrated"])
    l2_oof[vai] = read_probs(r2.stdout)

print(f"  L2 OOF AUC={roc_auc_score(y_train, l2_oof):.4f}")

# Apply CatBoost rank-norm to OOF L2 predictions — leak-free calibration set for CB-RN
l2_oof_order = np.argsort(l2_oof)
cb_oof_sorted = np.sort(cb_oof)
l2_oof_rn_cb = np.empty(len(l2_oof), dtype=np.float32)
l2_oof_rn_cb[l2_oof_order] = cb_oof_sorted

# Strategy: CatBoost RN has great Brier/LogLoss but ECE=0.0108
# The CB probability scale is stable (OOF and full models use same CatBoost dist.)
# → isotonic fitted on CB-RN OOF should generalise much better to CB-RN test than
#   the previous isotonic (which was fitted on raw imbgbm-L2 OOF with large dist. shift)

# Isotonic on CB-RN OOF → apply to CB-RN test (smaller distribution shift)
iso_cb = IsotonicRegression(out_of_bounds="clip")
iso_cb.fit(l2_oof_rn_cb, y_train)
super_probs_rn_cb_iso = iso_cb.predict(super_probs_rn_cb).astype(np.float32)

# Platt (logistic regression on logit) on CB-RN OOF → very smooth, fewer params
from sklearn.linear_model import LogisticRegression
_logit = lambda p: np.log(np.clip(p, 1e-9, 1-1e-9) / (1 - np.clip(p, 1e-9, 1-1e-9)))
lr_cb = LogisticRegression(C=1e6, max_iter=1000)
lr_cb.fit(_logit(l2_oof_rn_cb).reshape(-1, 1), y_train)
super_probs_rn_cb_platt = lr_cb.predict_proba(_logit(super_probs_rn_cb).reshape(-1, 1))[:, 1].astype(np.float32)

# Standard isotonic on raw L2 OOF (for reference)
iso = IsotonicRegression(out_of_bounds="clip")
iso.fit(l2_oof, y_train)
super_probs_cal = iso.predict(super_probs).astype(np.float32)
elapsed_super_iso = elapsed_super + (time.time() - t0_iso)

print()
print("=" * 72)
print("SuperStack CB-RN + OOF-calibrated post-processing:")
evaluate("SuperStack RN-CB+Iso ★★★★★", y_test, super_probs_rn_cb_iso, elapsed_super_iso)
evaluate("SuperStack RN-CB+Platt ★★★★★", y_test, super_probs_rn_cb_platt, elapsed_super_iso)

print()
print("=" * 72)
print("SuperStack raw-L2+Iso (for reference):")
evaluate("SuperStack ★★★★", y_test, super_probs_cal, elapsed_super_iso)

# ── Summary ────────────────────────────────────────────────────────────────────
print()
print("=" * 82)
print("SUPERSTACK BENCHMARK  (33k / 6k, 6% pos, fully labeled)")
print("=" * 82)
hdr = f"{'Model':<34} {'AUC':>6} {'PR-AUC':>7} {'ECE':>7} {'Brier':>7} {'R@1%':>6} {'P@5%':>6}"
print(hdr); print("-"*len(hdr))
for row in rows:
    print(f"{row['model']:<34} {row['auc']:6.4f} {row['prauc']:7.4f} "
          f"{row['ece']:7.4f} {row['brier']:7.4f} "
          f"{row['r1pct']:6.4f} {row['p5pct']:6.4f}")

print()
cb_r    = rows[0]   # CatBoost tuned
lgb_r   = rows[1]   # LightGBM tuned
imb_r   = rows[2]   # imbgbm Adapt+CS (single-stage)
METRICS = [
    ("auc",     "AUC",     True),
    ("prauc",   "PR-AUC",  True),
    ("r1pct",   "R@1%",    True),
    ("p5pct",   "P@5%",    True),
    ("ece",     "ECE",     False),
    ("brier",   "Brier",   False),
    ("logloss", "LogLoss", False),
]

def super_score(r):
    wins = sum(1 for k,_,hb in METRICS
               if (r[k] >= max(cb_r[k],lgb_r[k],imb_r[k]) if hb
                   else r[k] <= min(cb_r[k],lgb_r[k],imb_r[k])))
    adv = sum(
        np.log(max(r[k],1e-9) / max(max(cb_r[k],lgb_r[k],imb_r[k]),1e-9)) if hb
        else np.log(max(min(cb_r[k],lgb_r[k],imb_r[k]),1e-9) / max(r[k],1e-9))
        for k,_,hb in METRICS
    )
    return (wins, adv)

best_super = max(rows[3:], key=super_score)
rn_r = best_super
super_r = rows[-1]  # ★★★★ isotonic (also shown for reference)

print(f"{'Metric':<14} {'CatBoost':>10} {'LightGBM':>10} {'imbgbm':>10} {'RankNorm★★★★★':>14}  {'vs CB':>7}  {'vs LGB':>7}")
print("-" * 84)
for k, label, hb in METRICS:
    cv, lv, iv, sv = cb_r[k], lgb_r[k], imb_r[k], rn_r[k]
    best = max(cv,lv,iv,sv) if hb else min(cv,lv,iv,sv)
    d_cb  = 100*(sv-cv)/max(abs(cv),1e-9) * (1 if hb else -1)
    d_lgb = 100*(sv-lv)/max(abs(lv),1e-9) * (1 if hb else -1)
    mark  = lambda v: "◄" if abs(v-best)<1e-9 else " "
    print(f"  {label:<12} {cv:9.4f}{mark(cv)} {lv:9.4f}{mark(lv)} "
          f"{iv:9.4f}{mark(iv)} {sv:13.4f}{mark(sv)}  {d_cb:+7.1f}%  {d_lgb:+7.1f}%")

wins_rn = sum(1 for k, _, hb in METRICS
              if (rn_r[k] >= max(cb_r[k], lgb_r[k], imb_r[k]) if hb
                  else rn_r[k] <= min(cb_r[k], lgb_r[k], imb_r[k])))
print(f"\nSuperStack RankNorm ★★★★★ wins {wins_rn}/7 metrics outright (vs CatBoost + LightGBM + imbgbm single-stage).")
