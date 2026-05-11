use serde::{Deserialize, Serialize};

/// An internal split node.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InternalNode {
    pub feature: u32,
    /// Upper boundary of the left bin range (used during training routing).
    pub split_bin: u8,
    /// Actual feature value threshold (used during inference).
    pub split_threshold: f32,
    /// Index into `TreeStructure::nodes` for the left child.
    pub left: u32,
    /// Index into `TreeStructure::nodes` for the right child.
    pub right: u32,
}

/// A node in the tree: either an internal split or a terminal leaf.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum NodeKind {
    Internal(InternalNode),
    Leaf { leaf_idx: u32 },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Node {
    pub kind: NodeKind,
}

/// Tree topology (splits only, no leaf values).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TreeStructure {
    /// Arena of nodes; root is always at index 0.
    pub nodes: Vec<Node>,
    pub n_leaves: usize,
}

impl TreeStructure {
    /// Route a raw feature row to a leaf index (for inference).
    pub fn route(&self, features: &[f32]) -> usize {
        let mut idx = 0usize;
        loop {
            match &self.nodes[idx].kind {
                NodeKind::Leaf { leaf_idx } => return *leaf_idx as usize,
                NodeKind::Internal(n) => {
                    let val = features[n.feature as usize];
                    idx = if val <= n.split_threshold {
                        n.left as usize
                    } else {
                        n.right as usize
                    };
                }
            }
        }
    }

    /// Route a binned row (u8 bins) to a leaf index (for training).
    pub fn route_binned(&self, bins: &[u8]) -> usize {
        let mut idx = 0usize;
        loop {
            match &self.nodes[idx].kind {
                NodeKind::Leaf { leaf_idx } => return *leaf_idx as usize,
                NodeKind::Internal(n) => {
                    let bin = bins[n.feature as usize];
                    idx = if bin <= n.split_bin {
                        n.left as usize
                    } else {
                        n.right as usize
                    };
                }
            }
        }
    }
}

/// A fully built tree with Newton-step leaf values and optional OOF-calibrated probabilities.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CalibratedTree {
    pub structure: TreeStructure,
    /// Raw Newton-step leaf values (log-odds contributions).
    pub leaf_values: Vec<f32>,
    /// OOF-calibrated per-leaf probabilities (same length as leaf_values).
    /// Empty until calibration is run.
    pub leaf_probabilities: Vec<f32>,
    /// Per-leaf example count from the training pass.
    pub leaf_counts: Vec<u32>,
}

impl CalibratedTree {
    pub fn new(structure: TreeStructure) -> Self {
        let n = structure.n_leaves;
        CalibratedTree {
            structure,
            leaf_values: vec![0.0; n],
            leaf_probabilities: vec![0.0; n],
            leaf_counts: vec![0; n],
        }
    }

    /// Predict the raw leaf value for a raw feature row.
    #[inline]
    pub fn predict_raw(&self, features: &[f32]) -> f32 {
        self.leaf_values[self.structure.route(features)]
    }

    /// Predict the calibrated probability for a raw feature row.
    #[inline]
    pub fn predict_prob(&self, features: &[f32]) -> f32 {
        self.leaf_probabilities[self.structure.route(features)]
    }
}
