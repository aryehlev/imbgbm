use imbgbm_core::{BinnedDataset, CalibratedTree, NodeKind, RowMetadata};

// ── Isotonic regression (Pool Adjacent Violators) ────────────────────────────

/// Fit an isotonic (monotone non-decreasing) regression to `(x, y, weight)` pairs.
///
/// Input must be sorted by `x` ascending.  Returns calibrated `y` values in
/// the same order as input.
pub fn isotonic_regression(pairs: &[(f32, f32, f32)]) -> Vec<f32> {
    if pairs.is_empty() {
        return vec![];
    }

    struct Block {
        mean: f32,
        total_weight: f32,
        start: usize,
        end: usize,
    }

    let mut blocks: Vec<Block> = Vec::new();
    for (i, &(_, y, w)) in pairs.iter().enumerate() {
        let mut new_block = Block { mean: y, total_weight: w, start: i, end: i + 1 };
        while let Some(prev) = blocks.last() {
            if prev.mean <= new_block.mean {
                break;
            }
            let mut merged = blocks.pop().unwrap();
            let total_w = merged.total_weight + new_block.total_weight;
            merged.mean = (merged.mean * merged.total_weight
                + new_block.mean * new_block.total_weight)
                / total_w;
            merged.total_weight = total_w;
            merged.end = new_block.end;
            new_block = merged;
        }
        blocks.push(new_block);
    }

    let mut out = vec![0.0f32; pairs.len()];
    for block in blocks {
        for i in block.start..block.end {
            out[i] = block.mean.clamp(0.0, 1.0);
        }
    }
    out
}

// ── Fold assignment ───────────────────────────────────────────────────────────

/// Per-row fold index (0 .. k_folds - 1).
pub type FoldAssignment = Vec<usize>;

/// How to split training data into K folds for OOF calibration.
#[derive(Clone, Debug)]
pub enum FoldStrategy {
    /// Stratified random K-fold (default).  Preserves positive class fraction
    /// in each fold.
    Random { seed: u64 },
    /// Chronological K-fold based on per-row Unix timestamps in `RowMetadata`.
    ///
    /// Fold 0 = earliest examples, fold K-1 = latest.  This mirrors the
    /// real deployment scenario where the model is trained on historical data
    /// and calibrated on more-recent traffic.
    Temporal,
}

impl Default for FoldStrategy {
    fn default() -> Self {
        FoldStrategy::Random { seed: 42 }
    }
}

/// Assign rows to folds according to `strategy`.
pub fn assign_folds(
    labels: &[f32],
    metadata: &RowMetadata,
    k_folds: usize,
    strategy: &FoldStrategy,
) -> FoldAssignment {
    assert!(k_folds >= 2, "need at least 2 folds");
    match strategy {
        FoldStrategy::Random { seed } => assign_folds_stratified(labels, k_folds, *seed),
        FoldStrategy::Temporal => {
            let ts = metadata
                .timestamps
                .as_deref()
                .expect("FoldStrategy::Temporal requires RowMetadata::timestamps");
            assign_folds_temporal(ts, k_folds)
        }
    }
}

/// Stratified random K-fold: each fold has approximately the same positive rate.
pub fn assign_folds_stratified(labels: &[f32], k_folds: usize, seed: u64) -> FoldAssignment {
    let n = labels.len();
    let mut pos_indices: Vec<usize> =
        labels.iter().enumerate().filter(|(_, &y)| y > 0.5).map(|(i, _)| i).collect();
    let mut neg_indices: Vec<usize> =
        labels.iter().enumerate().filter(|(_, &y)| y <= 0.5).map(|(i, _)| i).collect();

    let shuffle = |v: &mut Vec<usize>, s: u64| {
        let mut rng = s;
        for i in (1..v.len()).rev() {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let j = (rng >> 33) as usize % (i + 1);
            v.swap(i, j);
        }
    };
    shuffle(&mut pos_indices, seed);
    shuffle(&mut neg_indices, seed ^ 0xdead_beef);

    let mut assignment = vec![0usize; n];
    for (slot, &idx) in pos_indices.iter().enumerate() {
        assignment[idx] = slot % k_folds;
    }
    for (slot, &idx) in neg_indices.iter().enumerate() {
        assignment[idx] = slot % k_folds;
    }
    assignment
}

/// Chronological K-fold: sorts rows by timestamp and assigns folds in order.
///
/// Rows with equal timestamps are assigned to the same fold in input order.
/// Fold 0 contains the oldest examples; fold K-1 the newest.
pub fn assign_folds_temporal(timestamps: &[i64], k_folds: usize) -> FoldAssignment {
    let n = timestamps.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by_key(|&i| timestamps[i]);

    let mut assignment = vec![0usize; n];
    for (slot, &row) in order.iter().enumerate() {
        // Map slot (0..n) to fold (0..k_folds) proportionally.
        assignment[row] = (slot * k_folds) / n.max(1);
    }
    assignment
}

// ── OOF leaf calibration ──────────────────────────────────────────────────────

/// Calibrate the leaf values of a tree using K-fold OOF statistics.
///
/// ## What it does
/// For each fold k the tree structure (splits) is held fixed — it was learned
/// on all data.  We recompute leaf values using only the folds ≠ k, then
/// record the resulting leaf value and the empirical positive rate for fold k's
/// examples in that leaf.
///
/// Final `leaf_values[l]`       = weighted mean of K OOF Newton-step values.
/// Final `leaf_probabilities[l]` = isotonic-calibrated P(y=1 | route→leaf l),
///                                  derived from the K (leaf_value, pos_rate)
///                                  pairs via Pool-Adjacent-Violators regression.
/// Calibrate a tree's leaf values via K-fold OOF statistics and **also** return
/// the per-row OOF leaf value (i.e. the leaf value computed from the K-1 folds
/// that did not include that row). Summing the OOF leaf value across all trees
/// gives an honest OOF boosted score, suitable for fitting post-hoc Platt
/// scaling without label leakage.
pub fn calibrate_tree_with_oof(
    tree: &mut CalibratedTree,
    data: &BinnedDataset,
    indices: &[u32],
    gradients: &[f32],
    hessians: &[f32],
    folds: &FoldAssignment,
    k_folds: usize,
    lambda: f32,
) -> Vec<f32> {
    let oof = calibrate_tree_inner(tree, data, indices, gradients, hessians,
                                    folds, k_folds, lambda, true);
    oof.expect("returned OOF leaf values requested")
}

pub fn calibrate_tree(
    tree: &mut CalibratedTree,
    data: &BinnedDataset,
    indices: &[u32],
    gradients: &[f32],
    hessians: &[f32],
    folds: &FoldAssignment,
    k_folds: usize,
    lambda: f32,
) {
    let _ = calibrate_tree_inner(tree, data, indices, gradients, hessians,
                                  folds, k_folds, lambda, false);
}

fn calibrate_tree_inner(
    tree: &mut CalibratedTree,
    data: &BinnedDataset,
    indices: &[u32],
    gradients: &[f32],
    hessians: &[f32],
    folds: &FoldAssignment,
    k_folds: usize,
    lambda: f32,
    return_oof: bool,
) -> Option<Vec<f32>> {
    let n_leaves = tree.structure.n_leaves;

    // Global positive rate — used as a Bayesian prior to prevent per-leaf
    // empirical rates from collapsing to 0 when a fold sees only 1–2 positives.
    let n_pos: f32 = data.labels.iter().filter(|&&y| y > 0.5).count() as f32;
    let global_prior = (n_pos / data.labels.len() as f32).clamp(1e-4, 1.0 - 1e-4);
    // Pseudo-count: equivalent to having seen `smoothing` extra examples with
    // the base rate. Larger values shrink aggressively toward the prior.
    let smoothing = 10.0_f32;

    // Per-leaf, per-fold accumulators: [fold][leaf].
    let mut fold_sum_g   = vec![vec![0.0f32; n_leaves]; k_folds];
    let mut fold_sum_h   = vec![vec![0.0f32; n_leaves]; k_folds];
    let mut fold_pos     = vec![vec![0u32;  n_leaves]; k_folds];
    let mut fold_total   = vec![vec![0u32;  n_leaves]; k_folds];

    for &row_u in indices {
        let row = row_u as usize;
        let leaf = route_to_leaf(tree, data, row);
        let fold_k = folds[row];
        let g = gradients[row];
        let h = hessians[row];
        let is_pos = data.labels[row] > 0.5;

        for k in 0..k_folds {
            if k != fold_k {
                fold_sum_g[k][leaf] += g;
                fold_sum_h[k][leaf] += h;
            } else {
                fold_total[k][leaf] += 1;
                if is_pos { fold_pos[k][leaf] += 1; }
            }
        }
    }

    // Per-(fold, leaf) OOF leaf value, used when returning per-row OOF scores.
    let mut fold_leaf_val = vec![vec![0.0f32; n_leaves]; k_folds];

    // Aggregate per-leaf.
    for l in 0..n_leaves {
        let mut oof_pairs: Vec<(f32, f32, f32)> = Vec::new(); // (leaf_value, pos_rate, weight)

        for k in 0..k_folds {
            if fold_sum_h[k][l] > 0.0 && fold_total[k][l] > 0 {
                let val = -fold_sum_g[k][l] / (fold_sum_h[k][l] + lambda);
                fold_leaf_val[k][l] = val;
                // Beta-prior smoothing: prevents collapse to 0 when the OOF
                // fold has few positives (common with 5% imbalance + K=5 folds).
                let count_pos   = fold_pos[k][l] as f32;
                let count_total = fold_total[k][l] as f32;
                let rate = (count_pos + global_prior * smoothing)
                    / (count_total + smoothing);
                let w    = count_total;
                oof_pairs.push((val, rate, w));
            }
        }

        if oof_pairs.is_empty() {
            continue; // Leaf not observed OOF — keep in-sample value.
        }

        // Weighted-mean OOF leaf value.
        let total_w: f32 = oof_pairs.iter().map(|(_, _, w)| w).sum();
        tree.leaf_values[l] =
            oof_pairs.iter().map(|(v, _, w)| v * w).sum::<f32>() / total_w;

        // Isotonic calibration: sort by leaf_value, map → positive rate.
        oof_pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let calibrated = isotonic_regression(&oof_pairs);

        // Select the calibrated probability closest to the mean leaf value.
        let mean_v = tree.leaf_values[l];
        let best_prob = oof_pairs
            .iter()
            .zip(calibrated.iter())
            .min_by(|(a, _), (b, _)| {
                (a.0 - mean_v).abs().partial_cmp(&(b.0 - mean_v).abs()).unwrap()
            })
            .map(|(_, &p)| p)
            .unwrap_or(0.5);

        tree.leaf_probabilities[l] = best_prob;
    }

    if return_oof {
        // For each row, look up its (fold, leaf) and emit the OOF leaf value.
        let n = data.labels.len();
        let mut per_row = vec![0.0f32; n];
        for row in 0..n {
            let leaf = route_to_leaf(tree, data, row);
            let k    = folds[row];
            // If the (fold, leaf) slot was empty (row's leaf was unseen OOF),
            // fall back to the in-sample leaf value.
            per_row[row] = if fold_total[k][leaf] > 0 {
                fold_leaf_val[k][leaf]
            } else {
                tree.leaf_values[leaf]
            };
        }
        Some(per_row)
    } else {
        None
    }
}

// ── PU label-frequency estimation (Elkan & Noto 2008) ────────────────────────

/// Estimator type for the PU labeling rate `c = P(s=1 | y=1)`.
#[derive(Clone, Copy, Debug)]
pub enum PuPriorEstimator {
    /// e1: mean predicted P(s=1|x) over held-out labeled positives. Most stable.
    MeanOverPositives,
    /// e2: 1 over the maximum predicted P(s=1|x) — sensitive to outliers.
    MaxOverPositives,
    /// e3: median predicted P(s=1|x) over labeled positives. Robust alternative.
    MedianOverPositives,
}

/// Estimate the PU labeling rate `c = P(s=1 | y=1)` from out-of-fold predictions.
///
/// Inputs:
///   `oof_scores` – out-of-fold P(s=1|x) for every training row.
///   `labels`      – the observed labels (1 = labeled positive, 0 = unlabeled).
///   `estimator`   – which of Elkan & Noto's three estimators to use.
///
/// Returns `c ∈ (0, 1]`. Under SCAR (selected completely at random), the true
/// positive probability is `P(y=1|x) = P(s=1|x) / c`. Lower `c` → larger
/// upward correction at inference.
pub fn estimate_pu_label_rate(
    oof_scores: &[f32],
    labels: &[f32],
    estimator: PuPriorEstimator,
) -> f32 {
    assert_eq!(oof_scores.len(), labels.len());
    let pos_scores: Vec<f32> = oof_scores
        .iter()
        .zip(labels)
        .filter(|(_, &y)| y > 0.5)
        .map(|(&s, _)| s)
        .collect();
    if pos_scores.is_empty() {
        return 1.0;
    }
    let c = match estimator {
        PuPriorEstimator::MeanOverPositives => {
            pos_scores.iter().sum::<f32>() / pos_scores.len() as f32
        }
        PuPriorEstimator::MaxOverPositives => {
            pos_scores.iter().copied().fold(0.0f32, f32::max)
        }
        PuPriorEstimator::MedianOverPositives => {
            let mut s = pos_scores.clone();
            s.sort_by(|a, b| a.partial_cmp(b).unwrap());
            s[s.len() / 2]
        }
    };
    c.clamp(1e-3, 1.0)
}

/// Apply the Elkan-Noto class-prior correction to a predicted probability.
///
/// `P(y=1|x) = clip(P(s=1|x) / c, 0, 1)`
///
/// In practice the linear rescaling can push values above 1 in the tail —
/// callers should `clamp(0, 1)` after applying.
#[inline]
pub fn pu_correct_probability(p_s_given_x: f32, c: f32) -> f32 {
    (p_s_given_x / c.max(1e-3)).clamp(0.0, 1.0)
}

// ── Platt scaling on OOF boosted scores ──────────────────────────────────────

/// Fit Platt scaling `(a, b)` minimising binary log-loss of
/// `sigmoid(a * score + b)` against `labels`.
///
/// Uses Newton-Raphson with full Hessian. Converges in ~5–10 iterations.
pub fn fit_platt(scores: &[f32], labels: &[f32]) -> (f32, f32) {
    assert_eq!(scores.len(), labels.len(), "scores/labels length mismatch");
    let n = scores.len() as f32;
    if n == 0.0 { return (1.0, 0.0); }

    // Smoothed targets (Platt 1999 §5) — prevents log(0) for perfect splits.
    let n_pos: f32 = labels.iter().filter(|&&y| y > 0.5).count() as f32;
    let n_neg: f32 = n - n_pos;
    let hi = (n_pos + 1.0) / (n_pos + 2.0);
    let lo = 1.0 / (n_neg + 2.0);
    let targets: Vec<f32> = labels
        .iter()
        .map(|&y| if y > 0.5 { hi } else { lo })
        .collect();

    let mut a = 1.0_f32;
    let mut b = 0.0_f32;
    for _ in 0..50 {
        let mut g_a = 0.0f64; let mut g_b = 0.0f64;
        let mut h_aa = 0.0f64; let mut h_ab = 0.0f64; let mut h_bb = 0.0f64;
        for (&s, &t) in scores.iter().zip(targets.iter()) {
            let z = (a * s + b) as f64;
            let p = 1.0 / (1.0 + (-z).exp());
            let err = p - t as f64;
            let pq  = p * (1.0 - p);
            g_a += err * s as f64;
            g_b += err;
            h_aa += pq * (s as f64) * (s as f64);
            h_ab += pq * s as f64;
            h_bb += pq;
        }
        // Solve 2x2 Hessian system H · Δ = −g, with tiny ridge for stability.
        let lambda = 1e-9;
        h_aa += lambda;
        h_bb += lambda;
        let det = h_aa * h_bb - h_ab * h_ab;
        if det.abs() < 1e-18 { break; }
        let inv = 1.0 / det;
        let da = -(h_bb * g_a - h_ab * g_b) * inv;
        let db = -(-h_ab * g_a + h_aa * g_b) * inv;
        a += da as f32;
        b += db as f32;
        if da.abs() < 1e-7 && db.abs() < 1e-7 { break; }
    }
    (a, b)
}

// ── Internal helpers ──────────────────────────────────────────────────────────

fn route_to_leaf(tree: &CalibratedTree, data: &BinnedDataset, row: usize) -> usize {
    let mut node_idx = 0usize;
    loop {
        match &tree.structure.nodes[node_idx].kind {
            NodeKind::Leaf { leaf_idx } => return *leaf_idx as usize,
            NodeKind::Internal(n) => {
                let bin = data.bins[n.feature as usize][row];
                node_idx = if bin <= n.split_bin {
                    n.left as usize
                } else {
                    n.right as usize
                };
            }
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn temporal_folds_are_ordered() {
        // 10 examples with strictly increasing timestamps → fold 0 = earliest.
        let ts: Vec<i64> = (0..10).map(|i| i as i64 * 100).collect();
        let folds = assign_folds_temporal(&ts, 5);
        // Folds should be non-decreasing as timestamp increases.
        let fold_by_time: Vec<usize> = (0..10usize).map(|i| folds[i]).collect();
        for w in fold_by_time.windows(2) {
            assert!(w[0] <= w[1], "temporal folds are not monotone: {fold_by_time:?}");
        }
    }

    #[test]
    fn stratified_folds_cover_all_rows() {
        let labels: Vec<f32> = (0..100).map(|i| if i % 10 == 0 { 1.0 } else { 0.0 }).collect();
        let folds = assign_folds_stratified(&labels, 5, 42);
        assert_eq!(folds.len(), 100);
        for k in 0..5 {
            let count = folds.iter().filter(|&&f| f == k).count();
            assert!(count > 0, "fold {k} is empty");
        }
    }

    #[test]
    fn isotonic_regression_is_monotone() {
        // Noisy non-monotone input → output should be non-decreasing.
        let pairs: Vec<(f32, f32, f32)> =
            vec![(0.0, 0.9, 1.0), (1.0, 0.1, 1.0), (2.0, 0.5, 1.0), (3.0, 0.8, 1.0)];
        let out = isotonic_regression(&pairs);
        for w in out.windows(2) {
            assert!(w[0] <= w[1] + 1e-6, "isotonic output not monotone: {out:?}");
        }
    }
}
