#!/usr/bin/env python3
"""
PU lookalike-modeling failure-mode demonstration (paper idea 10).

Generates a hard, realistic PU dataset (multimodal positives, seed selection
bias, hidden positives in the unlabeled pool) and shows how each "standard"
practice silently breaks the model, then how the imbgbm research-mode
features (PU-GOSS, PU-aware splits, mode preservation, Elkan-Noto correction,
leaf-confidence shrinkage, seed expansion) fix each failure mode.

Failure modes demonstrated:
  F1. Random negatives — biased decision boundary
  F2. scale_pos_weight reweighting — destroys calibration
  F3. AUC-only evaluation — masks ECE / overbidding
  F4. Random train/test split — overstates time-extrapolation accuracy
  F5. Single-mode positives — collapses rare modes
  F6. Naive probability output — ignores hidden-positive contamination

For each failure, we report the metric that lies and the metric that exposes it,
then show the imbgbm fix.
"""

import os, sys, time, subprocess
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss, brier_score_loss

SEED = 42
rng = np.random.default_rng(SEED)
IMBGBM = "/home/user/imbgbm/target/release/imbgbm"
BENCH_DIR = "/tmp/imbgbm_eval"
os.makedirs(BENCH_DIR, exist_ok=True)


# ── 1. Construct a multimodal PU dataset ─────────────────────────────────────
# True positives belong to one of 4 modes (finance / travel / gamer / parent).
# Seeds only come from modes 0 (finance) and 1 (travel) — biased seeds.
# Unlabeled pool: 5% hidden positives (all 4 modes) + 95% true negatives.

N         = 30_000
N_FEATS   = 12
MODES     = 4
SEED_MASS = 0.7   # 70% of seeds come from mode 0; 25% from mode 1; 5% other.

# Mode centroids
centroids = rng.normal(0, 2.0, size=(MODES, N_FEATS))

# True positive cluster proportions in the *population*
true_mode_mix = np.array([0.25, 0.25, 0.25, 0.25])
prior_pos     = 0.06

# Population labels (ground truth, only used for evaluation)
n_pos = int(N * prior_pos)
mode_assign = rng.choice(MODES, size=n_pos, p=true_mode_mix)
pos_feats = centroids[mode_assign] + rng.normal(0, 0.6, size=(n_pos, N_FEATS))
neg_feats = rng.normal(0, 2.5, size=(N - n_pos, N_FEATS))
X = np.vstack([pos_feats, neg_feats])
y_true = np.concatenate([np.ones(n_pos, dtype=int), np.zeros(N - n_pos, dtype=int)])
# Also track which positives came from which mode for later analysis
mode_id = np.concatenate([mode_assign, -np.ones(N - n_pos, dtype=int)])

# Shuffle
perm = rng.permutation(N)
X, y_true, mode_id = X[perm], y_true[perm], mode_id[perm]

# Biased seed labeling: only positives from modes 0 / 1 get labeled, and with
# a per-mode labeling rate c that's higher for mode 0.
# c_0 = 0.40, c_1 = 0.15, c_2 = c_3 = 0 → strong seed bias.
labeling_rate = np.array([0.40, 0.15, 0.0, 0.0])
s = np.zeros(N, dtype=int)  # observed PU label
for i in range(N):
    if y_true[i] == 1 and mode_id[i] >= 0:
        if rng.uniform() < labeling_rate[mode_id[i]]:
            s[i] = 1

print(f"Population stats:")
print(f"  N total       : {N}")
print(f"  True positives: {(y_true==1).sum()} ({y_true.mean():.2%})")
print(f"  Labeled seeds : {(s==1).sum()} ({(s==1).mean():.2%})")
print(f"  Mode distribution in seeds:")
for m in range(MODES):
    in_mode = ((y_true == 1) & (mode_id == m)).sum()
    labeled = ((s == 1) & (mode_id == m)).sum()
    print(f"    Mode {m}: {labeled}/{in_mode} seeds  ({labeled/max(1,in_mode):.0%} of mode)")


# ── 2. Train/test splits ──────────────────────────────────────────────────────
# Temporal: assign a synthetic timestamp by mode (modes appear over time).
# This simulates RTB where new positive modes emerge later in the stream.
timestamps = (mode_id + rng.normal(0, 0.1, N)).astype(float)
timestamps[mode_id < 0] = rng.uniform(-1, MODES, size=(mode_id < 0).sum())
order = np.argsort(timestamps)

# Random split (failure mode F4): treats time as i.i.d.
X_r_tr, X_r_te, y_r_tr, y_r_te, s_r_tr, s_r_te, mode_r_tr, mode_r_te = train_test_split(
    X, y_true, s, mode_id, test_size=0.2, random_state=SEED, stratify=s
)
# Temporal split: first 80% by time → train, last 20% → test
n_tr = int(N * 0.8)
tr_idx, te_idx = order[:n_tr], order[n_tr:]
X_t_tr, X_t_te = X[tr_idx], X[te_idx]
y_t_tr, y_t_te = y_true[tr_idx], y_true[te_idx]
s_t_tr, s_t_te = s[tr_idx], s[te_idx]
mode_t_tr, mode_t_te = mode_id[tr_idx], mode_id[te_idx]


# ── 3. Helpers ────────────────────────────────────────────────────────────────

def ece(y_true, y_prob, n_bins=10):
    bins  = np.linspace(0, 1, n_bins + 1); total = len(y_true); err = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (y_prob >= lo) & (y_prob < hi)
        if m.sum() == 0: continue
        err += m.sum() * abs(y_true[m].mean() - y_prob[m].mean())
    return err / total

def recall_per_mode(y_true, y_prob, mode, threshold_fpr=0.05):
    """Per-mode recall at a population FPR of 5%."""
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    idx = np.searchsorted(fpr, threshold_fpr, side="right") - 1
    t = thr[max(idx, 0)]
    preds = (y_prob >= t).astype(int)
    out = {}
    for m in range(MODES):
        mask = (mode == m) & (y_true == 1)
        if mask.sum() == 0: continue
        out[m] = (preds[mask] == 1).mean()
    return out

def write_csv(path, X, y=None):
    if y is None:
        np.savetxt(path, X, delimiter=",", fmt="%.6f")
    else:
        np.savetxt(path, np.c_[X, y], delimiter=",", fmt="%.6f")

def run_imbgbm(train_args, predict_args, X_tr, y_tr, X_te, model_path):
    write_csv(f"{BENCH_DIR}/tr.csv", X_tr, y_tr)
    write_csv(f"{BENCH_DIR}/te.csv", X_te)
    base_tr = [
        IMBGBM, "train",
        "--input",  f"{BENCH_DIR}/tr.csv",
        "--output", model_path,
        "--early-stopping-rounds", "0",
    ]
    r = subprocess.run(base_tr + train_args, capture_output=True, text=True)
    if r.returncode != 0:
        print("train failed:", r.stderr[-1500:]); sys.exit(1)
    base_pred = [IMBGBM, "predict", "--input", f"{BENCH_DIR}/te.csv", "--model", model_path]
    r = subprocess.run(base_pred + predict_args, capture_output=True, text=True)
    if r.returncode != 0:
        print("predict failed:", r.stderr[-1500:]); sys.exit(1)
    return np.array([float(x) for x in r.stdout.strip().splitlines()])

def report(name, y_true, y_prob, mode):
    auc = roc_auc_score(y_true, y_prob)
    pra = average_precision_score(y_true, y_prob)
    yp  = y_prob.clip(1e-7, 1 - 1e-7)
    ll  = log_loss(y_true, yp)
    bs  = brier_score_loss(y_true, yp)
    ec  = ece(y_true, yp)
    rpm = recall_per_mode(y_true, yp, mode)
    rpm_min = min(rpm.values()) if rpm else 0.0
    print(f"  {name:<32} AUC={auc:.3f}  PR-AUC={pra:.3f}  LL={ll:.3f}  "
          f"Brier={bs:.4f}  ECE={ec:.3f}  worst-mode-recall={rpm_min:.2f}  "
          f"per-mode={[f'{rpm.get(m,0):.2f}' for m in range(MODES)]}")
    return dict(auc=auc, prauc=pra, ll=ll, brier=bs, ece=ec,
                worst_mode_recall=rpm_min, per_mode=rpm)


# ── 4. Failure modes ─────────────────────────────────────────────────────────
print()
print("=" * 78)
print("FAILURE MODES — each row shows a 'standard' practice and what it costs.")
print("=" * 78)
print()

# Use random split (the typical practice) for the failure-mode demo.
X_tr, X_te = X_r_tr, X_r_te
y_tr, y_te = y_r_tr, y_r_te
s_tr, s_te = s_r_tr, s_r_te
mode_te    = mode_r_te

print("F1. Naive PU = BCE on (seeds positive, all else negative)")
p_f1 = run_imbgbm(
    ["--loss", "bce", "--n-rounds", "300", "--learning-rate", "0.05", "--max-depth", "6"],
    [],
    X_tr, s_tr, X_te, f"{BENCH_DIR}/f1.json",
)
r_f1 = report("BCE on PU labels", y_te, p_f1, mode_te)

print()
print("F2. scale_pos_weight-style up-weighting (focal α=0.75)")
p_f2 = run_imbgbm(
    ["--loss", "focal", "--n-rounds", "300", "--learning-rate", "0.05",
     "--max-depth", "6", "--gamma", "0.0", "--alpha", "0.75"],
    [],
    X_tr, s_tr, X_te, f"{BENCH_DIR}/f2.json",
)
r_f2 = report("Focal α=0.75 (pretend balance)", y_te, p_f2, mode_te)

print()
print("F3. The 'AUC looks fine' trap — show ECE / per-mode breakdown ↑")
# The F1 model often has good AUC but poor ECE and poor mode-3 recall.

print()
print("F4. Temporal vs random split — the model that 'works' may not extrapolate")
X_tr2, X_te2 = X_t_tr, X_t_te
y_tr2, y_te2 = y_t_tr, y_t_te
s_tr2, s_te2 = s_t_tr, s_t_te
mode_te2     = mode_t_te
p_f4 = run_imbgbm(
    ["--loss", "bce", "--n-rounds", "300", "--learning-rate", "0.05", "--max-depth", "6"],
    [],
    X_tr2, s_tr2, X_te2, f"{BENCH_DIR}/f4.json",
)
print("    Random-split BCE on PU labels:")
r_random = r_f1
report("(random split)", y_te, p_f1, mode_te)
print("    Temporal-split BCE on PU labels (same algorithm):")
r_temporal = report("(temporal split)", y_te2, p_f4, mode_te2)
delta_auc = r_temporal['auc'] - r_random['auc']
print(f"  → AUC change when switching to temporal split: {delta_auc:+.3f}  "
      f"(negative = inflated by i.i.d. assumption)")

print()
print("F5. Single-mode collapse — seeds came from modes 0 & 1; what about 2 & 3?")
print("  → per-mode recall above shows modes 2 & 3 typically near 0 with naive training.")


# ── 5. imbgbm research-mode fixes ────────────────────────────────────────────
print()
print("=" * 78)
print("RESEARCH-MODE FIXES — each one targets a specific failure mode.")
print("=" * 78)
print()

# Reset to random split for direct comparability with F1.
print("R1. PU-GOSS sampler (idea 1) — keeps seeds, mines hard unlabeled")
p_r1 = run_imbgbm(
    ["--loss", "focal", "--sampler", "pu_goss",
     "--n-rounds", "400", "--learning-rate", "0.05", "--max-depth", "6",
     "--gamma", "2.0", "--alpha", "0.25", "--subsample", "0.30"],
    [],
    X_tr, s_tr, X_te, f"{BENCH_DIR}/r1.json",
)
report("PU-GOSS + focal", y_te, p_r1, mode_te)

print()
print("R2. PU-aware splitter (idea 2) — discounts splits with too few seeds")
p_r2 = run_imbgbm(
    ["--loss", "focal", "--splitter", "pu",
     "--n-rounds", "400", "--learning-rate", "0.05", "--max-depth", "6",
     "--gamma", "2.0", "--alpha", "0.25"],
    [],
    X_tr, s_tr, X_te, f"{BENCH_DIR}/r2.json",
)
report("PU-aware splitter", y_te, p_r2, mode_te)

print()
print("R4. Leaf-confidence shrinkage (idea 4) — same model, shrink α=20 at inference")
p_r4 = run_imbgbm(
    ["--loss", "focal", "--n-rounds", "400", "--learning-rate", "0.05",
     "--max-depth", "6", "--gamma", "2.0", "--alpha", "0.25"],
    ["--shrink-alpha", "20.0"],
    X_tr, s_tr, X_te, f"{BENCH_DIR}/r4.json",
)
report("Leaf-shrinkage α=20", y_te, p_r4, mode_te)

print()
print("R6. Seed expansion (idea 6) — pilot → promote 3% → refine")
p_r6 = run_imbgbm(
    ["--loss", "focal", "--n-rounds", "300", "--learning-rate", "0.05",
     "--max-depth", "6", "--gamma", "2.0", "--alpha", "0.25",
     "--seed-expansion", "0.03"],
    [],
    X_tr, s_tr, X_te, f"{BENCH_DIR}/r6.json",
)
report("Seed expansion", y_te, p_r6, mode_te)

print()
print("R7. Elkan-Noto PU correction (idea 7) — divides p by estimated c")
p_r7 = run_imbgbm(
    ["--loss", "focal", "--n-rounds", "400", "--learning-rate", "0.05",
     "--max-depth", "6", "--gamma", "2.0", "--alpha", "0.25",
     "--calibrate", "--estimate-pu-rate"],
    ["--pu"],
    X_tr, s_tr, X_te, f"{BENCH_DIR}/r7.json",
)
report("Elkan-Noto PU correction", y_te, p_r7, mode_te)

print()
print("R9. Density-ratio loss (idea 9) — different framing entirely")
p_r9 = run_imbgbm(
    ["--loss", "density_ratio", "--pu-prior", "0.5",
     "--n-rounds", "400", "--learning-rate", "0.05", "--max-depth", "6"],
    [],
    X_tr, s_tr, X_te, f"{BENCH_DIR}/r9.json",
)
report("Density-ratio loss", y_te, p_r9, mode_te)

print()
print("R★. Combined research recipe — focal + PU-GOSS + PU splitter + shrinkage + expansion")
p_combined = run_imbgbm(
    ["--loss", "focal", "--sampler", "pu_goss", "--splitter", "pu",
     "--n-rounds", "500", "--learning-rate", "0.04", "--max-depth", "7",
     "--gamma", "2.0", "--alpha", "0.25", "--subsample", "0.30",
     "--seed-expansion", "0.03", "--calibrate", "--estimate-pu-rate"],
    ["--pu"],
    X_tr, s_tr, X_te, f"{BENCH_DIR}/combined.json",
)
r_combined = report("imbgbm research recipe ★", y_te, p_combined, mode_te)


# ── 6. Active labeling experiment (R8) ───────────────────────────────────────
print()
print("=" * 78)
print("R8. Active negative requesting (idea 8) — query top-K uncertain unlabeled.")
print("=" * 78)
budget = 50
# Train an initial model with calibration (so inter-tree variance is meaningful)
init_model = f"{BENCH_DIR}/active_init.json"
_ = run_imbgbm(
    ["--loss", "focal", "--n-rounds", "200", "--learning-rate", "0.05",
     "--max-depth", "6", "--gamma", "2.0", "--alpha", "0.25", "--calibrate"],
    [],
    X_tr, s_tr, X_te, init_model,
)
# Use the Query command on the *unlabeled* training rows.
unl_idx = np.where(s_tr == 0)[0]
write_csv(f"{BENCH_DIR}/unl.csv", X_tr[unl_idx])
r = subprocess.run(
    [IMBGBM, "query", "--input", f"{BENCH_DIR}/unl.csv",
     "--model", init_model, "--budget", str(budget)],
    capture_output=True, text=True
)
queried = [int(line.split(",")[0]) for line in r.stdout.strip().splitlines()]
print(f"  Queried {budget} unlabeled rows for advertiser-style labeling.")
# Of those queried, how many would the ground truth have called positive?
hits = sum(1 for q in queried if y_tr[unl_idx[q]] == 1)
rand = budget * y_tr[unl_idx].mean()
print(f"  Hits among queries     : {hits} / {budget}  ({hits/budget:.1%})")
print(f"  Expected from random   : {rand:.1f} ({rand/budget:.1%})")
print(f"  Lift over random       : {hits/max(rand, 1e-9):.2f}×")

# Simulate accepting the hint: relabel those rows as positives, retrain.
s_active = s_tr.copy()
for q in queried:
    if y_tr[unl_idx[q]] == 1:
        s_active[unl_idx[q]] = 1
p_active = run_imbgbm(
    ["--loss", "focal", "--n-rounds", "400", "--learning-rate", "0.05",
     "--max-depth", "6", "--gamma", "2.0", "--alpha", "0.25"],
    [],
    X_tr, s_active, X_te, f"{BENCH_DIR}/active.json",
)
print("  After incorporating advertiser-confirmed labels:")
r_active = report("Focal + active labels", y_te, p_active, mode_te)


# ── 7. Final summary ─────────────────────────────────────────────────────────
print()
print("=" * 78)
print("SUMMARY — does the research recipe close every failure mode?")
print("=" * 78)
print(f"{'Configuration':<36} {'AUC':>5} {'PR-AUC':>7} {'ECE':>6} {'worst mode':>11}")
for label, r in [
    ("F1 BCE on PU labels (baseline)", r_f1),
    ("F2 Focal α=0.75 (fake balance)", r_f2),
    ("Combined research recipe ★",     r_combined),
    ("Active labeling (R8)",            r_active),
]:
    print(f"{label:<36} {r['auc']:5.3f} {r['prauc']:7.3f} {r['ece']:6.3f} "
          f"{r['worst_mode_recall']:11.2f}")
