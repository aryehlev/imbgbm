use imbgbm_core::{CalibratedTree, NodeKind};
use serde::{Deserialize, Serialize};

// ── Model ────────────────────────────────────────────────────────────────────

/// A fully-trained imbgbm model ready for inference.
///
/// Two prediction modes are available:
/// - **Raw**: sum Newton-step leaf values across all trees, apply sigmoid.
///   Equivalent to a standard GBDT output.
/// - **Calibrated**: average the per-leaf OOF-calibrated probabilities across
///   trees.  More reliable under class imbalance.
#[derive(Clone, Serialize, Deserialize)]
pub struct Model {
    pub trees: Vec<CalibratedTree>,
    pub learning_rate: f32,
    pub init_score: f32,
}

impl Model {
    pub fn new(trees: Vec<CalibratedTree>, learning_rate: f32, init_score: f32) -> Self {
        Model { trees, learning_rate, init_score }
    }

    /// Predict the raw log-odds sum for a single example.
    pub fn predict_raw(&self, features: &[f32]) -> f32 {
        self.trees
            .iter()
            .map(|t| t.predict_raw(features))
            .sum::<f32>()
            * self.learning_rate
            + self.init_score
    }

    /// Predict probability via sigmoid of the raw score.
    pub fn predict_proba_raw(&self, features: &[f32]) -> f32 {
        sigmoid(self.predict_raw(features))
    }

    /// Predict probability using per-leaf OOF-calibrated leaf probabilities.
    ///
    /// Trees without calibration (empty `leaf_probabilities`) fall back to the
    /// raw sigmoid path for that tree.
    pub fn predict_proba_calibrated(&self, features: &[f32]) -> f32 {
        if self.trees.is_empty() {
            return 0.5;
        }
        let sum: f32 = self.trees.iter().map(|t| t.predict_prob(features)).sum();
        sum / self.trees.len() as f32
    }

    /// Batch predict (raw mode) for a matrix stored row-major.
    pub fn predict_batch_raw(&self, rows: &[&[f32]]) -> Vec<f32> {
        rows.iter().map(|r| self.predict_proba_raw(r)).collect()
    }

    /// Batch predict (calibrated mode) for a matrix stored row-major.
    pub fn predict_batch_calibrated(&self, rows: &[&[f32]]) -> Vec<f32> {
        rows.iter().map(|r| self.predict_proba_calibrated(r)).collect()
    }

    /// Serialize to JSON.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }

    /// Deserialize from JSON.
    pub fn from_json(s: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(s)
    }
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

// ── Leaf routing (column-major binned data for batch inference at training time)

/// Route a column-major binned matrix through a `CalibratedTree` and return
/// per-example leaf indices.  Used internally during training.
pub fn route_all(tree: &CalibratedTree, bins_col_major: &[Vec<u8>], n_rows: usize) -> Vec<usize> {
    (0..n_rows)
        .map(|row| {
            let mut idx = 0usize;
            loop {
                match &tree.structure.nodes[idx].kind {
                    NodeKind::Leaf { leaf_idx } => break *leaf_idx as usize,
                    NodeKind::Internal(n) => {
                        let bin = bins_col_major[n.feature as usize][row];
                        idx = if bin <= n.split_bin {
                            n.left as usize
                        } else {
                            n.right as usize
                        };
                    }
                }
            }
        })
        .collect()
}
