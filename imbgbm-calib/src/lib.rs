use imbgbm_core::{BinnedDataset, CalibratedTree, NodeKind};

// ── Isotonic regression (Pool Adjacent Violators) ────────────────────────────

/// Fit an isotonic (monotone non-decreasing) regression to `y` values paired
/// with monotonically increasing `x` values.  Returns calibrated y-values.
///
/// Input `pairs` must be sorted by x ascending.  Each element is (x, y, weight).
/// Returns a vec of calibrated y values in the same order as input.
pub fn isotonic_regression(pairs: &[(f32, f32, f32)]) -> Vec<f32> {
    if pairs.is_empty() {
        return vec![];
    }
    // Pool Adjacent Violators (PAV) on weighted means.
    // Each "block" has a weighted mean y value.
    struct Block {
        mean: f32,
        total_weight: f32,
        start: usize,
        end: usize,
    }

    let mut blocks: Vec<Block> = Vec::new();
    for (i, &(_, y, w)) in pairs.iter().enumerate() {
        let mut new_block = Block { mean: y, total_weight: w, start: i, end: i + 1 };
        // Merge with previous blocks while monotonicity is violated.
        while let Some(prev) = blocks.last() {
            if prev.mean <= new_block.mean {
                break;
            }
            let mut merged = blocks.pop().unwrap();
            let total_w = merged.total_weight + new_block.total_weight;
            merged.mean =
                (merged.mean * merged.total_weight + new_block.mean * new_block.total_weight)
                    / total_w;
            merged.total_weight = total_w;
            merged.end = new_block.end;
            new_block = merged;
        }
        blocks.push(new_block);
    }

    // Expand blocks back to per-example values.
    let mut out = vec![0.0f32; pairs.len()];
    for block in blocks {
        for i in block.start..block.end {
            out[i] = block.mean.clamp(0.0, 1.0);
        }
    }
    out
}

// ── OOF leaf calibration ──────────────────────────────────────────────────────

/// Fold assignment: which fold each training example belongs to (0..K).
pub type FoldAssignment = Vec<usize>;

/// Assign n examples to k folds in a stratified manner.
pub fn assign_folds(labels: &[f32], k_folds: usize, seed: u64) -> FoldAssignment {
    assert!(k_folds >= 2, "need at least 2 folds");
    let n = labels.len();
    // Stratified by label: keep positive fraction consistent across folds.
    let mut pos_indices: Vec<usize> = labels
        .iter()
        .enumerate()
        .filter(|(_, &y)| y > 0.5)
        .map(|(i, _)| i)
        .collect();
    let mut neg_indices: Vec<usize> = labels
        .iter()
        .enumerate()
        .filter(|(_, &y)| y <= 0.5)
        .map(|(i, _)| i)
        .collect();

    // Deterministic shuffle using a simple LCG.
    let shuffle = |v: &mut Vec<usize>, seed: u64| {
        let mut s = seed;
        for i in (1..v.len()).rev() {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let j = (s >> 33) as usize % (i + 1);
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

/// Calibrate the leaf values of a tree using K-fold OOF statistics.
///
/// For each fold k:
///   1. Compute leaf gradient/hessian sums excluding fold k.
///   2. Derive Newton-step leaf values from those OOF sums.
///   3. Record the leaf value and the empirical positive rate for fold k's
///      examples in each leaf.
///
/// Final `leaf_probabilities[l]` = isotonic-calibrated estimate of P(y=1|leaf l)
/// derived from the K (leaf_value, positive_rate) pairs.
///
/// Final `leaf_values[l]` = average of K OOF Newton-step values (lower variance
/// than the in-sample estimate).
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

    // Per-leaf, per-fold accumulators.
    // dim: [k_folds][n_leaves]
    let mut fold_sum_g = vec![vec![0.0f32; n_leaves]; k_folds];
    let mut fold_sum_h = vec![vec![0.0f32; n_leaves]; k_folds];
    let mut fold_pos_count = vec![vec![0u32; n_leaves]; k_folds];
    let mut fold_total_count = vec![vec![0u32; n_leaves]; k_folds];

    let n_cols = data.n_cols;

    // Route each example to its leaf using the fixed tree structure.
    for &row_u in indices {
        let row = row_u as usize;
        // Build per-row bin slice on-the-fly (column-major → row slice).
        let leaf = {
            let mut node_idx = 0usize;
            loop {
                match &tree.structure.nodes[node_idx].kind {
                    NodeKind::Leaf { leaf_idx } => break *leaf_idx as usize,
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
        };
        let fold_k = folds[row];
        let g = gradients[row];
        let h = hessians[row];
        let is_pos = data.labels[row] > 0.5;

        for k in 0..k_folds {
            if k != fold_k {
                // This example is in the training set for fold k.
                fold_sum_g[k][leaf] += g;
                fold_sum_h[k][leaf] += h;
            } else {
                // This example is the test set for fold k.
                fold_total_count[k][leaf] += 1;
                if is_pos {
                    fold_pos_count[k][leaf] += 1;
                }
            }
        }
        let _ = n_cols;
    }

    // For each leaf, aggregate OOF (leaf_value, positive_rate) pairs.
    let mut leaf_oof_values = vec![Vec::<f32>::new(); n_leaves];
    let mut leaf_oof_pos_rates = vec![Vec::<f32>::new(); n_leaves];
    let mut leaf_oof_weights = vec![Vec::<f32>::new(); n_leaves];

    for k in 0..k_folds {
        for l in 0..n_leaves {
            if fold_sum_h[k][l] > 0.0 && fold_total_count[k][l] > 0 {
                let leaf_val = -fold_sum_g[k][l] / (fold_sum_h[k][l] + lambda);
                let pos_rate = fold_pos_count[k][l] as f32 / fold_total_count[k][l] as f32;
                let weight = fold_total_count[k][l] as f32;
                leaf_oof_values[l].push(leaf_val);
                leaf_oof_pos_rates[l].push(pos_rate);
                leaf_oof_weights[l].push(weight);
            }
        }
    }

    // Compute final leaf values and calibrated probabilities.
    for l in 0..n_leaves {
        let vals = &leaf_oof_values[l];
        let rates = &leaf_oof_pos_rates[l];
        let weights = &leaf_oof_weights[l];

        if vals.is_empty() {
            // Leaf not visited in OOF — keep the in-sample Newton step.
            continue;
        }

        // Weighted mean of OOF leaf values.
        let total_w: f32 = weights.iter().sum();
        tree.leaf_values[l] = vals
            .iter()
            .zip(weights.iter())
            .map(|(&v, &w)| v * w)
            .sum::<f32>()
            / total_w;

        // Isotonic calibration: map OOF leaf_value → OOF positive_rate.
        // Sort by leaf_value ascending, fit isotonic regression.
        let mut pairs: Vec<(f32, f32, f32)> = vals
            .iter()
            .zip(rates.iter())
            .zip(weights.iter())
            .map(|((&v, &r), &w)| (v, r, w))
            .collect();
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let calibrated = isotonic_regression(&pairs);
        // With K=5 we get at most 5 calibrated values.  Use the one
        // closest to the weighted-mean leaf_value.
        let mean_v = tree.leaf_values[l];
        let best = pairs
            .iter()
            .zip(calibrated.iter())
            .min_by(|(a, _), (b, _)| {
                (a.0 - mean_v).abs().partial_cmp(&(b.0 - mean_v).abs()).unwrap()
            })
            .map(|(_, &c)| c)
            .unwrap_or(0.5);

        tree.leaf_probabilities[l] = best;
    }
}
