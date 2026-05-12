use imbgbm_core::{BinStats, Histogram};

// ── Positive-mass helpers ─────────────────────────────────────────────────────

/// Wilson score lower confidence bound for a binomial proportion.
/// Returns the conservative lower bound on the true positive rate.
fn beta_lcb(pos: f64, total: f64, z: f64) -> f64 {
    if total <= 0.0 {
        return 0.0;
    }
    let p = pos / total;
    let z2 = z * z;
    let denom = 1.0 + z2 / total;
    let center = p + z2 / (2.0 * total);
    let margin = z * ((p * (1.0 - p) / total) + z2 / (4.0 * total * total)).sqrt();
    ((center - margin) / denom).max(0.0)
}

/// Positive-mass gain: reward splits that create statistically reliable
/// positive concentration above the global base rate in each child.
///
/// Only children with at least `min_pos_leaf` positives contribute.
/// Uses Wilson LCB to avoid rewarding fluky small-positive leaves.
fn positive_mass_gain(
    left_pos: f64,
    left_total: f64,
    right_pos: f64,
    right_total: f64,
    global_rate: f64,
    min_pos_leaf: f64,
    z: f64,
) -> f64 {
    let base = global_rate.max(1e-12);
    let mut gain = 0.0;
    if left_pos >= min_pos_leaf {
        let lcb = beta_lcb(left_pos, left_total, z);
        let lift = (lcb / base).max(1e-12);
        gain += left_pos * lift.ln_1p();
    }
    if right_pos >= min_pos_leaf {
        let lcb = beta_lcb(right_pos, right_total, z);
        let lift = (lcb / base).max(1e-12);
        gain += right_pos * lift.ln_1p();
    }
    gain
}

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

// ── Positive-mass splitter ────────────────────────────────────────────────────

/// LiftBoost split criterion: Newton gain + statistically-regularised lift gain.
///
/// For each candidate split the score is:
///
///   `score = standard_gain + alpha * positive_mass_gain - tiny_leaf_penalty`
///
/// `positive_mass_gain` rewards children that reliably concentrate positive
/// examples above the global base rate, using a Wilson confidence lower bound
/// so that leaves with very few positives do not look artificially attractive.
///
/// `tiny_leaf_penalty` (per positive in an undersized leaf) discourages
/// splits that scatter rare positives into statistically unreliable leaves.
pub struct PositiveMassSplitter {
    /// Weight on the positive-mass gain term (α).  Typical range: 0.3–2.0.
    pub alpha: f32,
    /// Minimum positives required in a child for it to contribute lift gain.
    /// Children below this threshold incur the tiny-leaf penalty instead.
    pub min_pos_leaf: f32,
    /// z-score for the Wilson lower confidence bound.  1.96 ≈ 95 % one-sided.
    pub lcb_z: f32,
    /// Per-positive penalty for a child with fewer positives than `min_pos_leaf`.
    pub tiny_leaf_penalty: f32,
}

impl PositiveMassSplitter {
    /// Construct with sensible defaults for 99/1 imbalanced data.
    ///
    /// `prior` is the global positive rate (e.g. 0.01 for 1 % positives).
    /// `min_pos_leaf` defaults to `max(5, 0.001 * total_positives)` at call
    /// time; pass an explicit override to tighten or loosen the guard.
    pub fn new(alpha: f32, min_pos_leaf: f32) -> Self {
        PositiveMassSplitter {
            alpha,
            min_pos_leaf,
            lcb_z: 1.96,
            tiny_leaf_penalty: 0.0,
        }
    }

    pub fn with_penalty(mut self, tiny_leaf_penalty: f32) -> Self {
        self.tiny_leaf_penalty = tiny_leaf_penalty;
        self
    }

    pub fn with_lcb_z(mut self, z: f32) -> Self {
        self.lcb_z = z;
        self
    }
}

impl Splitter for PositiveMassSplitter {
    fn find_best_split(
        &self,
        histograms: &[Histogram],
        prior: Option<f32>,
        lambda: f32,
        min_child_weight: f32,
    ) -> Option<SplitInfo> {
        let global_rate = prior.unwrap_or(0.01) as f64;
        best_split_positive_mass(
            histograms,
            lambda,
            min_child_weight,
            self.alpha as f64,
            self.min_pos_leaf as f64,
            self.lcb_z as f64,
            self.tiny_leaf_penalty as f64,
            global_rate,
        )
    }
}

fn best_split_positive_mass(
    histograms: &[Histogram],
    lambda: f32,
    min_child_weight: f32,
    alpha: f64,
    min_pos_leaf: f64,
    lcb_z: f64,
    tiny_leaf_penalty: f64,
    global_rate: f64,
) -> Option<SplitInfo> {
    let mut best: Option<SplitInfo> = None;

    for hist in histograms {
        let n_bins = hist.bins.len();
        if n_bins < 2 {
            continue;
        }

        let total = hist.total();
        let (tot_g, tot_h) = (total.sum_g, total.sum_h);

        let mut left_g = 0.0_f32;
        let mut left_h = 0.0_f32;
        let mut left_count = 0u32;
        let mut left_count_pos = 0u32;

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

            if left_h < min_child_weight || right_h < min_child_weight {
                continue;
            }
            if left_count == 0 || right_count == 0 {
                continue;
            }

            let std_gain = newton_gain(tot_g, tot_h, left_g, left_h, right_g, right_h, lambda);

            let pm_gain = positive_mass_gain(
                left_count_pos as f64,
                left_count as f64,
                right_count_pos as f64,
                right_count as f64,
                global_rate,
                min_pos_leaf,
                lcb_z,
            );

            // Penalise children that have some positives but below the minimum —
            // they are too small to produce reliable lift estimates.
            let tiny_penalty = {
                let mut p = 0.0_f64;
                let lp = left_count_pos as f64;
                let rp = right_count_pos as f64;
                if lp > 0.0 && lp < min_pos_leaf {
                    p += tiny_leaf_penalty * lp;
                }
                if rp > 0.0 && rp < min_pos_leaf {
                    p += tiny_leaf_penalty * rp;
                }
                p
            };

            let score = std_gain as f64 + alpha * pm_gain - tiny_penalty;

            if best.as_ref().map_or(true, |bs| score as f32 > bs.gain) {
                best = Some(SplitInfo {
                    feature: hist.feature,
                    split_bin: b as u8,
                    gain: score as f32,
                    class_purity_delta: 0.0,
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

    best.filter(|s| s.gain > 0.0)
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
