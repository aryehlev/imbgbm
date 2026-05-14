#!/usr/bin/env python3
"""
Fair benchmark: imbgbm vs Perpetual vs CatBoost vs LightGBM.

All models receive the same 13 OOF target-encoded features.
CatBoost is also run on its native raw categoricals as a reference.
Results are averaged over 5 seeds.
"""

import os, sys, subprocess, time
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss, brier_score_loss, roc_curve
import catboost as cb
import lightgbm as lgb
import perpetual

SEEDS     = [42, 7, 13, 99, 2025]
N         = 39_000
CARDS     = [7518, 4243, 128, 177, 449, 343, 2358, 67, 343]
BENCH_DIR = "/tmp/imbgbm_perpetual_bench"
IMBGBM    = "/home/user/imbgbm/target/release/imbgbm"
os.makedirs(BENCH_DIR, exist_ok=True)

# ── Metrics ───────────────────────────────────────────────────────────────────

def ece(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    err = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (y_prob >= lo) & (y_prob < hi)
        if m.sum() == 0:
            continue
        err += m.sum() * abs(y_true[m].mean() - y_prob[m].mean())
    return err / len(y_true)

def recall_at_fpr(y_true, y_prob, fpr_target=0.01):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(tpr[max(np.searchsorted(fpr, fpr_target, "right") - 1, 0)])

def precision_at_k(y_true, y_prob, k_frac=0.05):
    k = int(len(y_true) * k_frac)
    return float(y_true[np.argsort(-y_prob)[:k]].mean())

def metrics(y_true, y_prob):
    y_prob = np.clip(y_prob, 1e-7, 1 - 1e-7)
    return dict(
        auc     = roc_auc_score(y_true, y_prob),
        prauc   = average_precision_score(y_true, y_prob),
        logloss = log_loss(y_true, y_prob),
        brier   = brier_score_loss(y_true, y_prob),
        ece     = ece(y_true, y_prob),
        r1pct   = recall_at_fpr(y_true, y_prob),
        p5pct   = precision_at_k(y_true, y_prob),
    )

# ── Feature engineering ───────────────────────────────────────────────────────

def te_col(codes_tr, y_tr, codes_te, card, smoothing=10.0):
    gm = float(y_tr.mean())
    cs = np.zeros(card); cc = np.zeros(card)
    np.add.at(cs, codes_tr, y_tr); np.add.at(cc, codes_tr, 1)
    te  = (cs[codes_te] + gm * smoothing) / (cc[codes_te] + smoothing)
    oof = np.full(len(y_tr), gm, dtype=np.float32)
    kf  = KFold(n_splits=5, shuffle=True, random_state=42)
    for tri, vai in kf.split(np.arange(len(y_tr))):
        cs2 = np.zeros(card); cc2 = np.zeros(card)
        np.add.at(cs2, codes_tr[tri], y_tr[tri])
        np.add.at(cc2, codes_tr[tri], 1)
        sm2 = (cs2 + gm * smoothing) / (cc2 + smoothing)
        oof[vai] = sm2[codes_tr[vai]]
    return oof.astype(np.float32), te.astype(np.float32)

PAIRS = [(4, 8), (0, 1), (1, 5), (2, 3)]

def build_features(X_train_raw, y_train, X_test_raw):
    X_tr = np.zeros((len(y_train), 9), dtype=np.float32)
    X_te = np.zeros((len(X_test_raw), 9), dtype=np.float32)
    for j, card in enumerate(CARDS):
        X_tr[:, j], X_te[:, j] = te_col(
            X_train_raw[:, j], y_train, X_test_raw[:, j], card)
    tr_extras, te_extras = [], []
    for a, b in PAIRS:
        ctr = X_train_raw[:, a].astype(np.int64) * CARDS[b] + X_train_raw[:, b].astype(np.int64)
        cte = X_test_raw[:, a].astype(np.int64)  * CARDS[b] + X_test_raw[:, b].astype(np.int64)
        tr_f, te_f = te_col(ctr, y_train, cte, CARDS[a] * CARDS[b])
        tr_extras.append(tr_f); te_extras.append(te_f)
    X_tr13 = np.c_[X_tr, np.column_stack(tr_extras)]
    X_te13 = np.c_[X_te, np.column_stack(te_extras)]
    return X_tr13, X_te13

# ── CLI helpers ───────────────────────────────────────────────────────────────

def run_cli(args):
    r = subprocess.run([IMBGBM] + args, capture_output=True, text=True)
    if r.returncode != 0:
        print("STDERR:", r.stderr[-2000:]); sys.exit(1)
    return r

def read_probs(stdout):
    return np.array([float(x) for x in stdout.strip().splitlines()])

# ── Per-seed run ──────────────────────────────────────────────────────────────

MODELS = [
    "CatBoost (raw cat)",
    "CatBoost (13 feat)",
    "LightGBM (13 feat)",
    "Perpetual (13 feat)",
    "imbgbm BCE+calibrate (13 feat)",
    "imbgbm BCE+raw_iso (13 feat)",
]
METRICS_KEYS = ["auc", "prauc", "r1pct", "p5pct", "ece", "brier", "logloss"]
HIGHER_BETTER = [True, True, True, True, False, False, False]

all_results = {m: {k: [] for k in METRICS_KEYS} for m in MODELS}
seed_times  = {m: [] for m in MODELS}

for seed_idx, SEED in enumerate(SEEDS):
    print(f"\n{'='*72}")
    print(f"SEED {SEED}  ({seed_idx+1}/{len(SEEDS)})")
    print('='*72)

    rng = np.random.default_rng(SEED)
    np.random.seed(SEED)

    # Generate dataset
    effects, scales = [], [1.2, 0.9, 1.5, 1.3, 1.1, 1.0, 0.8, 1.4, 1.0]
    for c, s in zip(CARDS, scales):
        effects.append(np.random.normal(0, s, c))
    inter_dr = np.random.normal(0, 0.7, size=(CARDS[4], CARDS[8]))

    X_raw = np.column_stack([rng.integers(0, c, size=N) for c in CARDS])
    lat   = sum(effects[j][X_raw[:, j]] for j in range(len(CARDS)))
    lat  += inter_dr[X_raw[:, 4], X_raw[:, 8]]
    lat  += rng.normal(0, 1.5, N)

    sigmoid = lambda x: 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))
    lo, hi = -30.0, 30.0
    for _ in range(80):
        mid = (lo + hi) / 2.0
        if sigmoid(lat + mid).mean() > 0.06: hi = mid
        else:                                 lo = mid
    y = (rng.uniform(size=N) < sigmoid(lat + (lo + hi) / 2)).astype(int)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=6_000, random_state=SEED, stratify=y)
    print(f"  pos rate train={y_train.mean():.2%}  test={y_test.mean():.2%}")

    X_tr13, X_te13 = build_features(X_train_raw, y_train, X_test_raw)
    feat_tr = f"{BENCH_DIR}/tr_{SEED}.csv"
    feat_te = f"{BENCH_DIR}/te_{SEED}.csv"
    np.savetxt(feat_tr, np.c_[X_tr13, y_train], delimiter=",", fmt="%.6f")
    np.savetxt(feat_te, X_te13,                  delimiter=",", fmt="%.6f")

    def record(name, y_prob, t):
        m = metrics(y_test, y_prob)
        for k in METRICS_KEYS:
            all_results[name][k].append(m[k])
        seed_times[name].append(t)
        print(f"  {name:<35}  AUC={m['auc']:.4f}  PR-AUC={m['prauc']:.4f}  "
              f"ECE={m['ece']:.4f}  Brier={m['brier']:.4f}  "
              f"R@1%={m['r1pct']:.4f}  t={t:.1f}s")

    # 1. CatBoost on raw categoricals
    print("\n[1] CatBoost (raw categoricals)")
    t0 = time.time()
    cb_raw = cb.CatBoostClassifier(
        iterations=1000, learning_rate=0.03, depth=8,
        l2_leaf_reg=3.0, bagging_temperature=0.5, random_strength=0.5,
        border_count=254, min_data_in_leaf=10,
        cat_features=list(range(9)), random_seed=SEED, verbose=0)
    cb_raw.fit(X_train_raw, y_train)
    record("CatBoost (raw cat)", cb_raw.predict_proba(X_test_raw)[:, 1], time.time() - t0)

    # 2. CatBoost on same 13 engineered features
    print("[2] CatBoost (13 feat)")
    t0 = time.time()
    cb_13 = cb.CatBoostClassifier(
        iterations=1000, learning_rate=0.03, depth=8,
        l2_leaf_reg=3.0, bagging_temperature=0.5, random_strength=0.5,
        border_count=254, min_data_in_leaf=10,
        random_seed=SEED, verbose=0)
    cb_13.fit(X_tr13, y_train)
    record("CatBoost (13 feat)", cb_13.predict_proba(X_te13)[:, 1], time.time() - t0)

    # 3. LightGBM on 13 features
    print("[3] LightGBM (13 feat)")
    t0 = time.time()
    lgb_tr, lgb_va, lgb_ytr, lgb_yva = train_test_split(
        X_tr13, y_train, test_size=0.15, random_state=SEED, stratify=y_train)
    lgb_ds  = lgb.Dataset(lgb_tr, label=lgb_ytr)
    lgb_val = lgb.Dataset(lgb_va, label=lgb_yva, reference=lgb_ds)
    lgb_params = dict(
        objective="binary", metric="binary_logloss",
        num_leaves=63, learning_rate=0.03,
        min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
        reg_lambda=1.0, seed=SEED, verbose=-1,
    )
    lgb_m = lgb.train(lgb_params, lgb_ds,
                      num_boost_round=1000,
                      valid_sets=[lgb_val],
                      callbacks=[lgb.early_stopping(20, verbose=False),
                                 lgb.log_evaluation(-1)])
    record("LightGBM (13 feat)", lgb_m.predict(X_te13), time.time() - t0)

    # 4. Perpetual on 13 features
    print("[4] Perpetual (13 feat)")
    t0 = time.time()
    perp = perpetual.PerpetualBooster(objective="LogLoss", budget=1.0, seed=SEED)
    perp.fit(X_tr13, y_train)
    record("Perpetual (13 feat)", perp.predict_proba(X_te13)[:, 1], time.time() - t0)

    # 5. imbgbm BCE + calibrate (per-leaf OOF)
    print("[5] imbgbm BCE+calibrate")
    t0 = time.time()
    m_path = f"{BENCH_DIR}/imbgbm_bce_{SEED}.json"
    run_cli(["train", "--input", feat_tr, "--output", m_path,
             "--loss", "bce", "--n-rounds", "300", "--learning-rate", "0.05",
             "--max-depth", "6", "--sampler", "adaptive", "--subsample", "0.5",
             "--col-subsample", "0.8", "--calibrate", "--early-stopping-rounds", "20",
             "--seed", str(SEED)])
    r = run_cli(["predict", "--input", feat_te, "--model", m_path, "--calibrated"])
    record("imbgbm BCE+calibrate (13 feat)", read_probs(r.stdout), time.time() - t0)

    # 6. imbgbm BCE + raw isotonic (K-fold OOF)
    print("[6] imbgbm BCE+raw_iso")
    t0 = time.time()
    m_iso = f"{BENCH_DIR}/imbgbm_iso_{SEED}.json"
    run_cli(["train", "--input", feat_tr, "--output", m_iso,
             "--loss", "bce", "--n-rounds", "300", "--learning-rate", "0.05",
             "--max-depth", "6", "--sampler", "adaptive", "--subsample", "0.5",
             "--col-subsample", "0.8", "--calibrate", "--raw-isotonic",
             "--early-stopping-rounds", "20", "--seed", str(SEED)])
    r = run_cli(["predict", "--input", feat_te, "--model", m_iso, "--raw-isotonic"])
    record("imbgbm BCE+raw_iso (13 feat)", read_probs(r.stdout), time.time() - t0)

# ── Summary ───────────────────────────────────────────────────────────────────

print(f"\n{'='*90}")
print(f"RESULTS AVERAGED OVER {len(SEEDS)} SEEDS: {SEEDS}")
print('='*90)

METRIC_LABELS = ["AUC", "PR-AUC", "R@1%", "P@5%", "ECE", "Brier", "LogLoss"]
col_w = 34

hdr = f"{'Model':<{col_w}}" + "".join(f"  {l:>9}" for l in METRIC_LABELS) + "   time(s)"
print(hdr)
print("-" * len(hdr))

mean_results = {}
for name in MODELS:
    means = {k: float(np.mean(all_results[name][k])) for k in METRICS_KEYS}
    stds  = {k: float(np.std(all_results[name][k]))  for k in METRICS_KEYS}
    mean_results[name] = means
    t_mean = float(np.mean(seed_times[name]))
    row = f"{name:<{col_w}}"
    for k, hb in zip(METRICS_KEYS, HIGHER_BETTER):
        row += f"  {means[k]:9.4f}"
    row += f"   {t_mean:6.1f}"
    print(row)

print()
print("Win counts vs CatBoost (raw cat):")
ref = mean_results["CatBoost (raw cat)"]
for name in MODELS:
    if name == "CatBoost (raw cat)":
        continue
    wins = 0
    marks = []
    for k, hb in zip(METRICS_KEYS, HIGHER_BETTER):
        v, r = mean_results[name][k], ref[k]
        beats = (v > r + 1e-5) if hb else (v < r - 1e-5)
        wins += beats
        marks.append("✓" if beats else "✗")
    print(f"  {name:<{col_w}}  {wins}/7  [{' '.join(marks)}]")

print()
print("  Metrics: AUC  PR-AUC  R@1%  P@5%  ECE  Brier  LogLoss")
