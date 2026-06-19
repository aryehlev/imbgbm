use std::sync::Arc;
use imbgbm_calib::FoldStrategy;
use imbgbm_core::RowMetadata;
use imbgbm_loss::Objective;
use imbgbm_sample::Sampler;
use imbgbm_split::Splitter;

/// Dynamic tail-weighting applied to gradients/hessians each boosting round.
///
/// Rows near the deployment threshold (top-`top_rate` bucket boundary) receive
/// high weights so the tree focuses its structure on the ranking boundary that
/// matters at inference.  Easy negatives far below the threshold are suppressed.
///
/// Applied by multiplying each row's gradient and hessian by its weight before
/// histogram accumulation.  This is distinct from focal loss (per-row hardness)
/// and from sampling (which biases which rows are seen at all).
#[derive(Clone)]
pub struct TailWeightConfig {
    /// Top fraction of rows to target, e.g. 0.01 = top 1%.
    pub top_rate: f64,
    /// Weight for positive examples scored below the threshold (missed positives).
    pub weight_missed_pos: f32,
    /// Weight for negative examples scored above the threshold (false positives in top bucket).
    pub weight_false_pos: f32,
    /// Weight for examples within `boundary_width` of the threshold on either side.
    pub weight_boundary: f32,
    /// Raw-score half-width around the threshold that counts as "boundary".
    pub boundary_width: f32,
    /// Weight for easy negatives (far below threshold, correctly ranked).
    pub weight_easy_neg: f32,
    /// Weight for easy positives (already above threshold, correctly ranked).
    pub weight_easy_pos: f32,
    /// First round (0-indexed) at which tail weighting activates.
    /// Set to 0 to apply from the very first round.
    pub start_round: usize,
}

impl TailWeightConfig {
    /// Defaults from the LiftBoost design: strong focus on the top-1% bucket.
    pub fn top1pct() -> Self {
        TailWeightConfig {
            top_rate: 0.01,
            weight_missed_pos: 20.0,
            weight_false_pos: 20.0,
            weight_boundary: 8.0,
            boundary_width: 0.5,
            weight_easy_neg: 0.05,
            weight_easy_pos: 0.5,
            start_round: 0,
        }
    }

    /// Start tail weighting after `round` rounds of normal training.
    pub fn starting_at(mut self, round: usize) -> Self {
        self.start_round = round;
        self
    }
}

/// Full training configuration.
pub struct Config {
    // ── Boosting ──────────────────────────────────────────────────────────────
    pub n_rounds: usize,
    pub learning_rate: f32,

    // ── Tree topology ─────────────────────────────────────────────────────────
    pub max_depth: usize,
    pub min_child_weight: f32,
    pub min_samples_leaf: usize,
    pub lambda: f32,

    // ── Binning ───────────────────────────────────────────────────────────────
    pub n_bins: usize,

    // ── Column subsampling ────────────────────────────────────────────────────
    /// Fraction of features randomly selected for each tree (0, 1].
    /// 0.8 adds diversity across trees and reduces variance.  1.0 = all features.
    pub col_subsample: f32,

    // ── OOF calibration ───────────────────────────────────────────────────────
    pub k_folds: usize,
    pub calibrate: bool,
    /// How to split examples into K folds.  Use `FoldStrategy::Temporal` when
    /// `metadata.timestamps` is populated for RTB / time-sensitive settings.
    pub fold_strategy: FoldStrategy,

    // ── Per-row metadata (segments + timestamps) ──────────────────────────────
    /// Optional contextual metadata used by the sampler and the calibrator.
    /// Must have `n_rows` entries if provided; set to `None` to use defaults.
    pub metadata: Option<Arc<RowMetadata>>,

    // ── Components ────────────────────────────────────────────────────────────
    pub objective: Arc<dyn Objective>,
    pub sampler: Arc<dyn Sampler>,
    pub splitter: Arc<dyn Splitter>,

    // ── Early stopping ────────────────────────────────────────────────────────
    pub early_stopping_rounds: Option<usize>,

    // ── Post-hoc calibration on OOF boosted scores ────────────────────────────
    /// When true and `calibrate` is also true, fit Platt scaling `(a, b)` after
    /// training and store it on the model. Preserves additive boosting structure
    /// (unlike per-leaf probability averaging).
    pub platt_scale: bool,
    /// When true and `calibrate` is also true, fit an isotonic calibration mapping
    /// on OOF raw scores. Trains K fold models and fits PAV isotonic regression on
    /// their held-out raw scores. Mutually exclusive with `platt_scale`; if both
    /// are true, isotonic takes precedence.
    pub raw_isotonic: bool,

    // ── Top-tail gradient weighting ───────────────────────────────────────────
    /// When set, multiply each row's gradient and hessian by a weight that
    /// focuses learning on the deployment threshold (top-K boundary).
    /// See `TailWeightConfig` for the weighting scheme.
    pub tail_weight: Option<TailWeightConfig>,

    // ── Categorical features ──────────────────────────────────────────────────
    /// Column indices (0-based) that contain integer-encoded categorical values.
    /// These receive OOF Bayesian target encoding during training (no leakage)
    /// and full-dataset encoding stored on the model for inference.
    pub cat_features: Vec<usize>,

    pub seed: u64,
}

impl Config {
    /// Default configuration: BCE loss + adaptive sampler + per-leaf OOF calibration.
    /// For post-hoc isotonic calibration on raw scores, also set `raw_isotonic: true`
    /// (requires training K fold models; ~K× slower).
    pub fn default_bce() -> Self {
        use imbgbm_loss::BCELoss;
        use imbgbm_sample::AdaptiveSampler;
        use imbgbm_split::StandardSplitter;
        Config {
            n_rounds: 300,
            learning_rate: 0.05,
            max_depth: 6,
            min_child_weight: 1.0,
            min_samples_leaf: 20,
            lambda: 1.0,
            n_bins: 255,
            col_subsample: 0.8,
            k_folds: 5,
            calibrate: true,
            fold_strategy: FoldStrategy::Random { seed: 42 },
            metadata: None,
            objective: Arc::new(BCELoss),
            sampler: Arc::new(AdaptiveSampler::new(0.2, 0.1, Some(0.5), 0.0, 42)),
            splitter: Arc::new(StandardSplitter),
            early_stopping_rounds: Some(20),
            platt_scale: false,
            raw_isotonic: false,
            tail_weight: None,
            cat_features: vec![],
            seed: 42,
        }
    }

    pub fn focal_calibrated(gamma: f32, alpha: f32) -> Self {
        use imbgbm_loss::FocalLoss;
        use imbgbm_sample::AdaptiveSampler;
        use imbgbm_split::VarianceAwareSplitter;
        Config {
            n_rounds: 200,
            learning_rate: 0.05,
            max_depth: 7,
            min_child_weight: 5.0,
            min_samples_leaf: 20,
            lambda: 1.0,
            n_bins: 255,
            col_subsample: 0.8,
            k_folds: 5,
            calibrate: true,
            fold_strategy: FoldStrategy::Random { seed: 42 },
            metadata: None,
            objective: Arc::new(FocalLoss::new(gamma, alpha)),
            sampler: Arc::new(AdaptiveSampler::new(0.2, 0.1, Some(0.5), 0.0, 42)),
            splitter: Arc::new(VarianceAwareSplitter::new(0.1, 5)),
            early_stopping_rounds: Some(20),
            platt_scale: false,
            raw_isotonic: false,
            tail_weight: None,
            cat_features: vec![],
            seed: 42,
        }
    }

    /// LiftBoost preset for 99/1 imbalanced classification.
    ///
    /// Uses the `PositiveMassSplitter` which adds a statistically-regularised
    /// lift gain term to the standard Newton gain, rewarding splits that create
    /// reliable positive concentration above the global base rate.
    ///
    /// The phased training schedule:
    /// - Rounds 0–99:   normal split gain; broad structure from BCE gradients.
    /// - Rounds 100+:   positive-mass split gain active via the splitter.
    /// - Rounds 200+:   top-tail gradient weighting kicks in (top-1% boundary).
    ///
    /// `prior` is the expected positive rate (e.g. 0.01 for 1 % positives).
    /// It is used to size `min_pos_leaf` and initialise the positive-mass splitter.
    pub fn lift_boost(prior: f32) -> Self {
        use imbgbm_loss::BCELoss;
        use imbgbm_sample::AdaptiveSampler;
        use imbgbm_split::PositiveMassSplitter;

        // min_pos_leaf: at least 5 positives per leaf, but scale with dataset
        // size.  Caller can override by reconstructing the splitter.
        let min_pos_leaf = 20.0_f32;

        Config {
            n_rounds: 350,
            learning_rate: 0.05,
            max_depth: 6,
            min_child_weight: 5.0,
            min_samples_leaf: 20,
            lambda: 1.0,
            n_bins: 255,
            col_subsample: 0.8,
            k_folds: 5,
            calibrate: true,
            fold_strategy: FoldStrategy::Random { seed: 42 },
            metadata: None,
            objective: Arc::new(BCELoss),
            sampler: Arc::new(AdaptiveSampler::new(0.2, 0.1, Some(prior), 0.0, 42)),
            splitter: Arc::new(
                PositiveMassSplitter::new(1.0, min_pos_leaf)
                    .with_penalty(0.1),
            ),
            early_stopping_rounds: Some(30),
            platt_scale: false,
            raw_isotonic: false,
            tail_weight: Some(TailWeightConfig::top1pct().starting_at(200)),
            cat_features: vec![],
            seed: 42,
        }
    }
}
