use imbgbm_core::{CalibratedTree, NodeKind};
use serde::{Deserialize, Serialize};

// ── Raw isotonic calibration lookup table ────────────────────────────────────

/// Non-parametric calibration mapping: OOF raw boosted score → probability.
///
/// Fitted by sorting OOF (score, label) pairs, running Pool-Adjacent-Violators
/// isotonic regression, and compressing the resulting step function into a
/// compact set of breakpoints. At inference, linear interpolation between
/// adjacent breakpoints gives smooth, monotone probabilities.
///
/// Because fold models are trained on K-1/K of the data with identical
/// hyper-parameters, their raw score distribution matches the final model's
/// test distribution closely enough that no scale/offset alignment is needed.
#[derive(Clone, Serialize, Deserialize, Default)]
pub struct RawIsoCal {
    /// Sorted OOF raw score breakpoints (compressed block midpoints).
    pub scores: Vec<f32>,
    /// Corresponding calibrated probabilities from isotonic regression.
    pub probs: Vec<f32>,
}

impl RawIsoCal {
    pub fn is_empty(&self) -> bool { self.scores.is_empty() }

    /// Apply isotonic calibration to a raw boosted score via linear interpolation.
    pub fn predict(&self, raw: f32) -> f32 {
        let n = self.scores.len();
        if n == 0 { return sigmoid(raw); }
        if raw <= self.scores[0] { return self.probs[0]; }
        if raw >= self.scores[n - 1] { return self.probs[n - 1]; }
        let pos = self.scores.partition_point(|&x| x < raw);
        if pos == 0 { return self.probs[0]; }
        if pos >= n { return self.probs[n - 1]; }
        let lo = pos - 1;
        let t = (raw - self.scores[lo]) / (self.scores[pos] - self.scores[lo] + 1e-9);
        (self.probs[lo] + t * (self.probs[pos] - self.probs[lo])).clamp(0.0, 1.0)
    }
}

// ── Model ────────────────────────────────────────────────────────────────────

/// A fully-trained imbgbm model ready for inference.
///
/// Prediction modes:
/// - **Raw**: sigmoid of cumulative Newton-step leaf values.
/// - **Calibrated**: average per-leaf OOF positive rates across trees.
/// - **Platt**: sigmoid(a * raw_score + b), preserves additive structure.
/// - **RawIso**: isotonic-calibrated raw score via OOF PAV mapping.
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
    /// OOF isotonic calibration on raw scores. When non-empty,
    /// `predict_proba_raw_iso()` uses this mapping instead of sigmoid.
    #[serde(default)]
    pub raw_iso_cal: RawIsoCal,
}

impl Model {
    pub fn new(trees: Vec<CalibratedTree>, learning_rate: f32, init_score: f32) -> Self {
        Model {
            trees, learning_rate, init_score,
            platt: None, pu_label_rate: None, raw_iso_cal: RawIsoCal::default(),
        }
    }

    pub fn with_platt(mut self, a: f32, b: f32) -> Self {
        self.platt = Some((a, b));
        self
    }

    pub fn with_pu_label_rate(mut self, c: f32) -> Self {
        self.pu_label_rate = Some(c.clamp(1e-3, 1.0));
        self
    }

    pub fn with_raw_iso_cal(mut self, scores: Vec<f32>, probs: Vec<f32>) -> Self {
        self.raw_iso_cal = RawIsoCal { scores, probs };
        self
    }

    /// Predict the *true* positive probability under SCAR by dividing the
    /// best available probability by the estimated PU label rate `c`.
    /// Priority: raw_iso > platt > raw. Falls back to best available if `c` is not set.
    pub fn predict_proba_pu(&self, features: &[f32]) -> f32 {
        let p_s = if !self.raw_iso_cal.is_empty() {
            self.predict_proba_raw_iso(features)
        } else if self.platt.is_some() {
            self.predict_proba_platt(features)
        } else {
            self.predict_proba_raw(features)
        };
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

    /// Predict probability using the OOF isotonic calibration of the raw score.
    /// Falls back to `predict_proba_raw` if no calibration mapping is stored.
    pub fn predict_proba_raw_iso(&self, features: &[f32]) -> f32 {
        if self.raw_iso_cal.is_empty() {
            return self.predict_proba_raw(features);
        }
        self.raw_iso_cal.predict(self.predict_raw(features))
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
