use rand::{seq::SliceRandom, SeedableRng};

use imbgbm_core::{
    histogram::{build_histograms, Histogram},
    BinnedDataset, CalibratedTree, InternalNode, Node, NodeKind, TreeStructure,
};
use imbgbm_split::Splitter;

/// Grow a single regression tree on the given `indices` subset.
///
/// `ipc_weights[j]` is the inverse-probability correction weight for
/// `indices[j]`.  Pass an empty slice or all-ones to skip correction.
///
/// `col_subsample` (0, 1] controls what fraction of features are eligible
/// for splitting in this tree (feature_fraction / colsample_bytree).
/// `seed` is used to draw the random column subset reproducibly.
pub fn grow_tree(
    data: &BinnedDataset,
    indices: &[u32],
    ipc_weights: &[f32],
    gradients: &[f32],
    hessians: &[f32],
    splitter: &dyn Splitter,
    max_depth: usize,
    min_child_weight: f32,
    min_samples_leaf: usize,
    lambda: f32,
    prior: Option<f32>,
    col_subsample: f32,
    seed: u64,
) -> CalibratedTree {
    let mut nodes: Vec<Node> = Vec::new();
    let mut leaf_values: Vec<f32> = Vec::new();
    let mut leaf_counts: Vec<u32> = Vec::new();

    // Root gradient/hessian sums (IPC-weighted).
    let (root_g, root_h) = weighted_sum(indices, ipc_weights, gradients, hessians);

    let n_bins_per_col: Vec<usize> = (0..data.n_cols).map(|c| data.n_bins_for_col(c)).collect();

    // Column subsampling: pick a random subset of features once per tree.
    // Sorted for cache-friendly histogram access.
    let active_cols: Vec<usize> = if col_subsample >= 1.0 || data.n_cols == 0 {
        (0..data.n_cols).collect()
    } else {
        let n_keep = ((data.n_cols as f32 * col_subsample).ceil() as usize).max(1);
        let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);
        let mut order: Vec<usize> = (0..data.n_cols).collect();
        order.shuffle(&mut rng);
        let mut chosen = order[..n_keep].to_vec();
        chosen.sort_unstable();
        chosen
    };
    let full_cols = active_cols.len() == data.n_cols;

    // For the subsampled case, clone selected column bin data once per tree
    // so the stack loop can reuse it across every node.
    let sub_bins: Option<Vec<Vec<u8>>> = if full_cols {
        None
    } else {
        Some(active_cols.iter().map(|&c| data.bins[c].clone()).collect())
    };
    let sub_n_bins: Option<Vec<usize>> = if full_cols {
        None
    } else {
        Some(active_cols.iter().map(|&c| n_bins_per_col[c]).collect())
    };

    // Allocate root node slot.
    nodes.push(Node { kind: NodeKind::Leaf { leaf_idx: 0 } });

    // Work-stack: (node_slot, row_indices, ipc_weights_for_those_rows, depth, sum_g, sum_h)
    let mut stack: Vec<(usize, Vec<u32>, Vec<f32>, usize, f32, f32)> = vec![(
        0,
        indices.to_vec(),
        ipc_weights.to_vec(),
        0,
        root_g,
        root_h,
    )];

    while let Some((slot, idx, weights, depth, sum_g, sum_h)) = stack.pop() {
        let make_leaf = depth >= max_depth
            || idx.len() < min_samples_leaf.max(2)
            || sum_h < min_child_weight;

        if make_leaf {
            let leaf_idx = leaf_values.len() as u32;
            leaf_values.push(-sum_g / (sum_h + lambda));
            leaf_counts.push(idx.len() as u32);
            nodes[slot] = Node { kind: NodeKind::Leaf { leaf_idx } };
            continue;
        }

        let histograms: Vec<Histogram> = if full_cols {
            build_histograms(
                &data.bins,
                &n_bins_per_col,
                &data.labels,
                gradients,
                hessians,
                &idx,
                &weights,
            )
        } else {
            build_histograms(
                sub_bins.as_ref().unwrap(),
                sub_n_bins.as_ref().unwrap(),
                &data.labels,
                gradients,
                hessians,
                &idx,
                &weights,
            )
        };

        match splitter.find_best_split(&histograms, prior, lambda, min_child_weight) {
            None => {
                let leaf_idx = leaf_values.len() as u32;
                leaf_values.push(-sum_g / (sum_h + lambda));
                leaf_counts.push(idx.len() as u32);
                nodes[slot] = Node { kind: NodeKind::Leaf { leaf_idx } };
            }
            Some(split) => {
                // Map histogram index → original column index.
                let feat = active_cols[split.feature];
                let split_bin = split.split_bin;

                // Partition uses original column data (not the cloned subset).
                let (left_idx, left_w, right_idx, right_w) =
                    partition_by_bin(&idx, &weights, &data.bins[feat], split_bin);

                if left_idx.len() < min_samples_leaf || right_idx.len() < min_samples_leaf {
                    let leaf_idx = leaf_values.len() as u32;
                    leaf_values.push(-sum_g / (sum_h + lambda));
                    leaf_counts.push(idx.len() as u32);
                    nodes[slot] = Node { kind: NodeKind::Leaf { leaf_idx } };
                    continue;
                }

                let left_slot = nodes.len();
                nodes.push(Node { kind: NodeKind::Leaf { leaf_idx: 0 } });
                let right_slot = nodes.len();
                nodes.push(Node { kind: NodeKind::Leaf { leaf_idx: 0 } });

                let split_threshold = data.bin_thresholds[feat][split_bin as usize];
                nodes[slot] = Node {
                    kind: NodeKind::Internal(InternalNode {
                        feature: feat as u32,
                        split_bin,
                        split_threshold,
                        left: left_slot as u32,
                        right: right_slot as u32,
                    }),
                };

                stack.push((
                    right_slot,
                    right_idx,
                    right_w,
                    depth + 1,
                    split.right_sum_g,
                    split.right_sum_h,
                ));
                stack.push((
                    left_slot,
                    left_idx,
                    left_w,
                    depth + 1,
                    split.left_sum_g,
                    split.left_sum_h,
                ));
            }
        }
    }

    let n_leaves = leaf_values.len();
    let structure = TreeStructure { nodes, n_leaves };
    let leaf_probabilities = vec![0.0f32; n_leaves];

    CalibratedTree { structure, leaf_values, leaf_probabilities, leaf_counts }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// IPC-weighted sum of gradients and hessians for the given index subset.
fn weighted_sum(
    indices: &[u32],
    weights: &[f32],
    gradients: &[f32],
    hessians: &[f32],
) -> (f32, f32) {
    let use_w = weights.len() == indices.len();
    indices.iter().enumerate().fold((0.0_f32, 0.0_f32), |(ag, ah), (j, &r)| {
        let w = if use_w { weights[j] } else { 1.0 };
        (ag + gradients[r as usize] * w, ah + hessians[r as usize] * w)
    })
}

/// Partition (indices, weights) by whether `bins_col[row] <= split_bin`.
fn partition_by_bin(
    indices: &[u32],
    weights: &[f32],
    bins_col: &[u8],
    split_bin: u8,
) -> (Vec<u32>, Vec<f32>, Vec<u32>, Vec<f32>) {
    let use_w = weights.len() == indices.len();
    let (mut li, mut lw, mut ri, mut rw) = (vec![], vec![], vec![], vec![]);
    for (j, &row) in indices.iter().enumerate() {
        let w = if use_w { weights[j] } else { 1.0 };
        if bins_col[row as usize] <= split_bin {
            li.push(row);
            lw.push(w);
        } else {
            ri.push(row);
            rw.push(w);
        }
    }
    (li, lw, ri, rw)
}
