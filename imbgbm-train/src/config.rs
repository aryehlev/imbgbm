use std::sync::Arc;
use imbgbm_loss::Objective;
use imbgbm_sample::Sampler;
use imbgbm_split::Splitter;

/// Full training configuration.
pub struct Config {
    // ── Boosting ──────────────────────────────────────────────────────────────
    pub n_rounds: usize,
    pub learning_rate: f32,

    // ── Tree topology ─────────────────────────────────────────────────────────
    pub max_depth: usize,
    /// Minimum sum of hessians required in each leaf (like min_child_weight).
    pub min_child_weight: f32,
    /// Minimum number of examples per leaf.
    pub min_samples_leaf: usize,
    /// L2 regularisation for leaf values.
    pub lambda: f32,

    // ── Binning ───────────────────────────────────────────────────────────────
    pub n_bins: usize,

    // ── OOF calibration ───────────────────────────────────────────────────────
    pub k_folds: usize,
    pub calibrate: bool,

    // ── Components ────────────────────────────────────────────────────────────
    pub objective: Arc<dyn Objective>,
    pub sampler: Arc<dyn Sampler>,
    pub splitter: Arc<dyn Splitter>,

    // ── Early stopping ────────────────────────────────────────────────────────
    /// Stop if validation metric does not improve for this many rounds.
    pub early_stopping_rounds: Option<usize>,

    pub seed: u64,
}

impl Config {
    /// Construct a default BCE config with sensible hyperparameters.
    pub fn default_bce() -> Self {
        use imbgbm_loss::BCELoss;
        use imbgbm_sample::UniformSampler;
        use imbgbm_split::StandardSplitter;
        Config {
            n_rounds: 100,
            learning_rate: 0.1,
            max_depth: 6,
            min_child_weight: 1.0,
            min_samples_leaf: 20,
            lambda: 1.0,
            n_bins: 255,
            k_folds: 5,
            calibrate: false,
            objective: Arc::new(BCELoss),
            sampler: Arc::new(UniformSampler::new(0.8, 42)),
            splitter: Arc::new(StandardSplitter),
            early_stopping_rounds: Some(10),
            seed: 42,
        }
    }

    /// Construct a focal-loss config with OOF calibration enabled.
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
            k_folds: 5,
            calibrate: true,
            objective: Arc::new(FocalLoss::new(gamma, alpha)),
            sampler: Arc::new(AdaptiveSampler::new(0.2, 0.1, Some(0.5), 0.0, 42)),
            splitter: Arc::new(VarianceAwareSplitter::new(0.1, 5)),
            early_stopping_rounds: Some(20),
            seed: 42,
        }
    }
}
