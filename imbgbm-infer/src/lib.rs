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
    /// Optional Platt scaling parameters `(a, b)` fit on OOF boosted scores.
    /// When set, `predict_proba_platt(x) = sigmoid(a * raw_score + b)`.
    /// Preserves the additive boosting structure (unlike per-leaf averaging).
    #[serde(default)]
    pub platt: Option<(f32, f32)>,
    /// PU label rate `c = P(s=1|y=1)` estimated via Elkan-Noto. When set, the
    /// model can output `P(y=1|x) = P(s=1|x) / c` instead of the labeled-class
    /// probability, recovering the true positive probability under SCAR.
    #[serde(default)]
    pub pu_label_rate: Option<f32>,
}

impl Model {
    pub fn new(trees: Vec<CalibratedTree>, learning_rate: f32, init_score: f32) -> Self {
        Model { trees, learning_rate, init_score, platt: None, pu_label_rate: None }
    }

    pub fn with_platt(mut self, a: f32, b: f32) -> Self {
        self.platt = Some((a, b));
        self
    }

    pub fn with_pu_label_rate(mut self, c: f32) -> Self {
        self.pu_label_rate = Some(c.clamp(1e-3, 1.0));
        self
    }

    /// Predict the *true* positive probability under SCAR by dividing the
    /// Platt-or-raw probability by the estimated PU label rate `c`. Falls back
    /// to `predict_proba_platt` (or raw) if `c` is not set.
    pub fn predict_proba_pu(&self, features: &[f32]) -> f32 {
        let p_s = self.predict_proba_platt(features);
        match self.pu_label_rate {
            Some(c) => (p_s / c.max(1e-3)).clamp(0.0, 1.0),
            None => p_s,
        }
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

    /// Predict probability through the Platt-scaled boosted score.
    /// Falls back to `predict_proba_raw` if Platt parameters are not present.
    pub fn predict_proba_platt(&self, features: &[f32]) -> f32 {
        match self.platt {
            Some((a, b)) => sigmoid(a * self.predict_raw(features) + b),
            None => self.predict_proba_raw(features),
        }
    }

    /// Predict probability after **leaf-confidence shrinkage**: each tree's
    /// leaf value is multiplied by `n_eff / (n_eff + alpha)` before summing.
    /// Tiny leaves with high apparent gain but few effective examples are
    /// pulled toward zero, preventing the model from overbidding on noisy
    /// micro-clusters.
    ///
    /// `alpha` controls the strength of the prior (10–50 typical). Pass
    /// alpha = 0 to recover the unshrunk raw prediction.
    pub fn predict_proba_shrunk(&self, features: &[f32], alpha: f32) -> f32 {
        let raw_sum: f32 = self.trees.iter().map(|t| t.predict_shrunk(features, alpha)).sum();
        sigmoid(raw_sum * self.learning_rate + self.init_score)
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

// ── Active-learning query selector (idea 8) ──────────────────────────────────

/// Information-theoretic scoring for selecting the most informative unlabeled
/// rows to ask a domain expert (e.g. advertiser) to label.
///
/// For a tree ensemble, we use a simple proxy for BALD: the average per-tree
/// leaf-positive-rate variance. Rows that route to leaves where individual
/// trees disagree (high inter-tree variance over `leaf_probabilities`) carry
/// the most signal under a fixed labeling budget.
///
/// Returns a vector of `(row_index_in_pool, info_score)` sorted by descending
/// score. Pick the top `budget` rows.
pub fn rank_query_candidates(
    model: &Model,
    pool: &[&[f32]],
) -> Vec<(usize, f32)> {
    let has_calib = !model.trees.is_empty()
        && model.trees.iter().any(|t| !t.leaf_probabilities.is_empty());
    let mut scored: Vec<(usize, f32)> = pool
        .iter()
        .enumerate()
        .map(|(i, row)| {
            let info = if has_calib {
                inter_tree_variance(model, row)
            } else {
                // Fall back to predictive-entropy = p(1-p) on the raw output.
                let p = model.predict_proba_raw(row);
                p * (1.0 - p)
            };
            (i, info)
        })
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    scored
}

/// Variance of per-tree leaf probabilities for a single example.
fn inter_tree_variance(model: &Model, features: &[f32]) -> f32 {
    if model.trees.is_empty() { return 0.0; }
    let probs: Vec<f32> = model
        .trees
        .iter()
        .filter(|t| !t.leaf_probabilities.is_empty())
        .map(|t| t.predict_prob(features))
        .collect();
    if probs.is_empty() { return 0.0; }
    let mean = probs.iter().sum::<f32>() / probs.len() as f32;
    probs.iter().map(|&p| (p - mean).powi(2)).sum::<f32>() / probs.len() as f32
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
