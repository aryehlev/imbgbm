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
    let n_leaves = tree.structure.n_leaves;

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

    // Aggregate per-leaf.
    for l in 0..n_leaves {
        let mut oof_pairs: Vec<(f32, f32, f32)> = Vec::new(); // (leaf_value, pos_rate, weight)

        for k in 0..k_folds {
            if fold_sum_h[k][l] > 0.0 && fold_total[k][l] > 0 {
                let val = -fold_sum_g[k][l] / (fold_sum_h[k][l] + lambda);
                let rate = fold_pos[k][l] as f32 / fold_total[k][l] as f32;
                let w    = fold_total[k][l] as f32;
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
