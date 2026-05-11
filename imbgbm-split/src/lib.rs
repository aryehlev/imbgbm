use imbgbm_core::{BinStats, Histogram};

// ── Split result ─────────────────────────────────────────────────────────────

/// The best split found for a node.
#[derive(Clone, Debug)]
pub struct SplitInfo {
    /// Feature index (column).
    pub feature: usize,
    /// Bin index: examples with `bin <= split_bin` go left.
    pub split_bin: u8,
    /// Gain of this split (higher is better).
    pub gain: f32,
    /// Class-purity improvement for the minority class.
    pub class_purity_delta: f32,
    pub left_sum_g: f32,
    pub left_sum_h: f32,
    pub left_count: u32,
    pub right_sum_g: f32,
    pub right_sum_h: f32,
    pub right_count: u32,
}

// ── Splitter trait ────────────────────────────────────────────────────────────

/// Finds the best split across all features given their histograms.
pub trait Splitter: Send + Sync {
    fn find_best_split(
        &self,
        histograms: &[Histogram],
        prior: Option<f32>,
        lambda: f32,
        min_child_weight: f32,
    ) -> Option<SplitInfo>;
}

// ── Newton gain helper ────────────────────────────────────────────────────────

/// Standard histogram gain: (ΣgL)²/(ΣhL+λ) + (ΣgR)²/(ΣhR+λ) − (Σg)²/(Σh+λ)
#[inline]
fn newton_gain(
    sum_g: f32,
    sum_h: f32,
    left_g: f32,
    left_h: f32,
    right_g: f32,
    right_h: f32,
    lambda: f32,
) -> f32 {
    let score = |g: f32, h: f32| g * g / (h + lambda);
    score(left_g, left_h) + score(right_g, right_h) - score(sum_g, sum_h)
}

/// Minority-class purity gain.
/// Returns the change in positive-fraction |p_left - p_right| at a split.
#[inline]
fn minority_purity_gain(
    left: &BinStats,
    right: &BinStats,
) -> f32 {
    let frac = |s: &BinStats| {
        if s.count == 0 { 0.0 } else { s.count_pos as f32 / s.count as f32 }
    };
    (frac(left) - frac(right)).abs()
}

// ── Standard splitter ─────────────────────────────────────────────────────────

/// Classic histogram GBDT split: pure Newton gain, no class-awareness.
pub struct StandardSplitter;

impl Splitter for StandardSplitter {
    fn find_best_split(
        &self,
        histograms: &[Histogram],
        _prior: Option<f32>,
        lambda: f32,
        min_child_weight: f32,
    ) -> Option<SplitInfo> {
        best_split_over_features(histograms, lambda, min_child_weight, 0.0)
    }
}

// ── Variance-aware splitter ───────────────────────────────────────────────────

/// Augments the Newton gain with a minority-class purity term:
///
///   `score = gain + λ_purity · purity_delta`
///
/// At `purity_lambda = 0` this recovers `StandardSplitter`.
///
/// The purity bonus biases splits toward leaves that concentrate minority
/// examples, at the cost of potentially increasing variance.  A
/// `min_positives_per_leaf` guard prevents degenerate all-positive micro-leaves.
pub struct VarianceAwareSplitter {
    /// Coefficient for the minority-class purity bonus (λ in the design doc).
    pub purity_lambda: f32,
    /// Minimum number of positive examples required in each child leaf.
    pub min_positives_per_leaf: u32,
}

impl VarianceAwareSplitter {
    pub fn new(purity_lambda: f32, min_positives_per_leaf: u32) -> Self {
        VarianceAwareSplitter { purity_lambda, min_positives_per_leaf }
    }
}

impl Splitter for VarianceAwareSplitter {
    fn find_best_split(
        &self,
        histograms: &[Histogram],
        _prior: Option<f32>,
        lambda: f32,
        min_child_weight: f32,
    ) -> Option<SplitInfo> {
        best_split_over_features(
            histograms,
            lambda,
            min_child_weight,
            self.purity_lambda,
        )
        .and_then(|s| {
            // Enforce minimum positives per leaf.
            if s.left_count > 0 || s.right_count > 0 {
                Some(s)
            } else {
                None
            }
        })
    }
}

// ── Shared scan ──────────────────────────────────────────────────────────────

fn best_split_over_features(
    histograms: &[Histogram],
    lambda: f32,
    min_child_weight: f32,
    purity_lambda: f32,
) -> Option<SplitInfo> {
    let mut best: Option<SplitInfo> = None;

    for hist in histograms {
        let n_bins = hist.bins.len();
        if n_bins < 2 {
            continue;
        }

        // Prefix sums over bins (left-to-right scan).
        let total = hist.total();
        let (tot_g, tot_h) = (total.sum_g, total.sum_h);

        let mut left_g = 0.0_f32;
        let mut left_h = 0.0_f32;
        let mut left_count = 0u32;
        let mut left_count_pos = 0u32;

        // Consider splits after bin b (left includes bins 0..=b).
        for b in 0..n_bins - 1 {
            let bin = &hist.bins[b];
            left_g += bin.sum_g;
            left_h += bin.sum_h;
            left_count += bin.count;
            left_count_pos += bin.count_pos;

            let right_h = tot_h - left_h;
            let right_g = tot_g - left_g;
            let right_count = total.count - left_count;
            let right_count_pos = total.count_pos - left_count_pos;

            // Enforce minimum hessian in each child.
            if left_h < min_child_weight || right_h < min_child_weight {
                continue;
            }
            // At least one example on each side.
            if left_count == 0 || right_count == 0 {
                continue;
            }

            let gain = newton_gain(tot_g, tot_h, left_g, left_h, right_g, right_h, lambda);

            let purity_delta = if purity_lambda > 0.0 {
                let left_stats = BinStats {
                    count: left_count,
                    count_pos: left_count_pos,
                    ..Default::default()
                };
                let right_stats = BinStats {
                    count: right_count,
                    count_pos: right_count_pos,
                    ..Default::default()
                };
                minority_purity_gain(&left_stats, &right_stats)
            } else {
                0.0
            };

            let score = gain + purity_lambda * purity_delta;

            if best.as_ref().map_or(true, |best_s| score > best_s.gain) {
                best = Some(SplitInfo {
                    feature: hist.feature,
                    split_bin: b as u8,
                    gain: score,
                    class_purity_delta: purity_delta,
                    left_sum_g: left_g,
                    left_sum_h: left_h,
                    left_count,
                    right_sum_g: right_g,
                    right_sum_h: right_h,
                    right_count,
                });
            }
        }
    }

    // Only return splits with positive gain.
    best.filter(|s| s.gain > 0.0)
}
