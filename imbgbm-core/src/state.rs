/// Mutable per-round training state shared between all components.
pub struct BoostingState {
    /// Accumulated raw predictions (sum of learning_rate * leaf_value over all trees).
    pub predictions: Vec<f32>,
    /// Current gradient vector (dL/dpred_i).
    pub gradients: Vec<f32>,
    /// Current Hessian vector (d²L/dpred_i²), clamped positive.
    pub hessians: Vec<f32>,
    /// Constant initial score (log-odds of class prior).
    pub init_score: f32,
}

impl BoostingState {
    pub fn new(n_rows: usize, init_score: f32) -> Self {
        BoostingState {
            predictions: vec![init_score; n_rows],
            gradients: vec![0.0; n_rows],
            hessians: vec![1.0; n_rows],
            init_score,
        }
    }

    /// Update cumulative predictions using the latest tree's leaf values.
    pub fn apply_tree(
        &mut self,
        tree: &crate::CalibratedTree,
        bins_col_major: &[Vec<u8>],
        learning_rate: f32,
    ) {
        let n_rows = self.predictions.len();
        // Collect per-row bin vectors for routing.
        let n_cols = bins_col_major.len();
        for r in 0..n_rows {
            // Build a temporary per-row bin slice via the column-major data.
            // For large datasets rayon would help here; kept simple for now.
            let leaf = {
                let mut idx = 0usize;
                loop {
                    match &tree.structure.nodes[idx].kind {
                        crate::NodeKind::Leaf { leaf_idx } => break *leaf_idx as usize,
                        crate::NodeKind::Internal(n) => {
                            let bin = bins_col_major[n.feature as usize][r];
                            idx = if bin <= n.split_bin {
                                n.left as usize
                            } else {
                                n.right as usize
                            };
                        }
                    }
                }
            };
            self.predictions[r] += learning_rate * tree.leaf_values[leaf];
            let _ = n_cols; // suppress unused warning
        }
    }
}
