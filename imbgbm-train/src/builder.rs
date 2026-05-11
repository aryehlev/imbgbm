use imbgbm_core::{
    histogram::{build_histograms, Histogram},
    BinnedDataset, CalibratedTree, InternalNode, Node, NodeKind, TreeStructure,
};
use imbgbm_split::Splitter;

/// Grow a single regression tree on the given `indices` subset.
///
/// Returns the tree structure with:
/// - `leaf_values` set to in-sample Newton steps
/// - `leaf_probabilities` zeroed (filled later by OOF calibration)
pub fn grow_tree(
    data: &BinnedDataset,
    indices: &[u32],
    gradients: &[f32],
    hessians: &[f32],
    splitter: &dyn Splitter,
    max_depth: usize,
    min_child_weight: f32,
    min_samples_leaf: usize,
    lambda: f32,
    prior: Option<f32>,
) -> CalibratedTree {
    let mut nodes: Vec<Node> = Vec::new();
    let mut leaf_values: Vec<f32> = Vec::new();
    let mut leaf_counts: Vec<u32> = Vec::new();

    // Pre-compute total gradient/hessian for the root.
    let (root_g, root_h): (f32, f32) = indices
        .iter()
        .map(|&r| (gradients[r as usize], hessians[r as usize]))
        .fold((0.0, 0.0), |(ag, ah), (g, h)| (ag + g, ah + h));

    let n_bins_per_col: Vec<usize> = (0..data.n_cols)
        .map(|c| data.n_bins_for_col(c))
        .collect();

    // Recursive builder using a work-stack to avoid call-stack overflow on deep trees.

    // We need to allocate node slots before recursing into children.
    // We use a two-phase approach:
    //   Phase 1: allocate the root node slot.
    //   Phase 2: process a stack of (node_slot, indices, depth, sum_g, sum_h).

    nodes.push(Node { kind: NodeKind::Leaf { leaf_idx: 0 } }); // placeholder
    let root_slot = 0usize;

    let mut stack: Vec<(usize, Vec<u32>, usize, f32, f32)> =
        vec![(root_slot, indices.to_vec(), 0, root_g, root_h)];

    while let Some((slot, idx, depth, sum_g, sum_h)) = stack.pop() {
        // Decide: split or leaf?
        let make_leaf = depth >= max_depth
            || idx.len() < min_samples_leaf.max(2)
            || sum_h < min_child_weight;

        if make_leaf {
            let leaf_idx = leaf_values.len() as u32;
            let val = -sum_g / (sum_h + lambda);
            leaf_values.push(val);
            leaf_counts.push(idx.len() as u32);
            nodes[slot] = Node { kind: NodeKind::Leaf { leaf_idx } };
            continue;
        }

        // Build histograms for this node.
        let histograms = build_histograms(
            &data.bins,
            &n_bins_per_col,
            &data.labels,
            gradients,
            hessians,
            &idx,
        );

        let split = splitter.find_best_split(&histograms, prior, lambda, min_child_weight);

        if split.is_none() {
            // No valid split found; make this a leaf.
            let leaf_idx = leaf_values.len() as u32;
            leaf_values.push(-sum_g / (sum_h + lambda));
            leaf_counts.push(idx.len() as u32);
            nodes[slot] = Node { kind: NodeKind::Leaf { leaf_idx } };
            continue;
        }
        let split = split.unwrap();

        // Partition indices.
        let feat = split.feature;
        let split_bin = split.split_bin;
        let (left_idx, right_idx): (Vec<u32>, Vec<u32>) =
            idx.iter().partition(|&&r| data.bins[feat][r as usize] <= split_bin);

        // Check minimum samples on each side.
        if left_idx.len() < min_samples_leaf || right_idx.len() < min_samples_leaf {
            let leaf_idx = leaf_values.len() as u32;
            leaf_values.push(-sum_g / (sum_h + lambda));
            leaf_counts.push(idx.len() as u32);
            nodes[slot] = Node { kind: NodeKind::Leaf { leaf_idx } };
            continue;
        }

        // Allocate child node slots.
        let left_slot = nodes.len();
        nodes.push(Node { kind: NodeKind::Leaf { leaf_idx: 0 } }); // placeholder
        let right_slot = nodes.len();
        nodes.push(Node { kind: NodeKind::Leaf { leaf_idx: 0 } }); // placeholder

        // Look up the actual threshold value from the bin boundaries.
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

        // Push children onto the stack.
        stack.push((right_slot, right_idx, depth + 1, split.right_sum_g, split.right_sum_h));
        stack.push((left_slot, left_idx, depth + 1, split.left_sum_g, split.left_sum_h));
    }

    let n_leaves = leaf_values.len();
    let structure = TreeStructure { nodes, n_leaves };
    let leaf_probabilities = vec![0.0f32; n_leaves];

    CalibratedTree { structure, leaf_values, leaf_probabilities, leaf_counts }
}
