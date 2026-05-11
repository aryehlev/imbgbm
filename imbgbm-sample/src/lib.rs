use rand::{Rng, SeedableRng};
use rand::rngs::SmallRng;

// ── Sampler trait ────────────────────────────────────────────────────────────

/// Decides which row indices each tree will see.
///
/// The key invariant: returned indices are a multiset of `[0, n)`.  The
/// gradient/hessian arrays keep their original indexing; only the subset of
/// rows is changed.
pub trait Sampler: Send + Sync {
    fn sample_indices(
        &self,
        gradients: &[f32],
        hessians: &[f32],
        labels: &[f32],
        round: usize,
    ) -> Vec<u32>;
}

// ── Uniform sampler ──────────────────────────────────────────────────────────

/// Bernoulli row sub-sampling — identical to LightGBM's `bagging_fraction`.
pub struct UniformSampler {
    /// Fraction of rows to keep (0, 1].
    pub subsample: f32,
    pub seed: u64,
}

impl UniformSampler {
    pub fn new(subsample: f32, seed: u64) -> Self {
        assert!(subsample > 0.0 && subsample <= 1.0);
        UniformSampler { subsample, seed }
    }
}

impl Sampler for UniformSampler {
    fn sample_indices(
        &self,
        gradients: &[f32],
        _hessians: &[f32],
        _labels: &[f32],
        round: usize,
    ) -> Vec<u32> {
        let n = gradients.len();
        if self.subsample >= 1.0 {
            return (0..n as u32).collect();
        }
        let mut rng = SmallRng::seed_from_u64(self.seed ^ (round as u64 * 0x9e37_79b9));
        (0..n as u32).filter(|_| rng.gen::<f32>() < self.subsample).collect()
    }
}

// ── GOSS sampler ─────────────────────────────────────────────────────────────

/// Gradient-One-Side Sampling (Ke et al. 2017).
///
/// Keeps all examples in the top `top_rate` fraction of |gradient|, then
/// randomly samples `other_rate` of the rest, re-weighting the sampled tail
/// so the histogram statistics remain unbiased.
///
/// Note: re-weighting is stored externally (via the gradient array) — the
/// index list returned here does NOT include per-example weights.  The caller
/// must scale tail-gradients by `(1-top_rate)/other_rate` before building
/// histograms if bias-correction is desired.  For simplicity in this
/// implementation we return a plain index list and document the correction.
pub struct GossSampler {
    /// Fraction of large-gradient examples to always keep.
    pub top_rate: f32,
    /// Fraction of the remaining examples to randomly sample.
    pub other_rate: f32,
    pub seed: u64,
}

impl GossSampler {
    pub fn new(top_rate: f32, other_rate: f32, seed: u64) -> Self {
        assert!(top_rate > 0.0 && top_rate < 1.0);
        assert!(other_rate > 0.0 && other_rate <= 1.0);
        GossSampler { top_rate, other_rate, seed }
    }
}

impl Sampler for GossSampler {
    fn sample_indices(
        &self,
        gradients: &[f32],
        _hessians: &[f32],
        _labels: &[f32],
        round: usize,
    ) -> Vec<u32> {
        let n = gradients.len();
        let top_k = ((n as f32 * self.top_rate).ceil() as usize).min(n);

        // Sort indices by |gradient| descending.
        let mut order: Vec<u32> = (0..n as u32).collect();
        order.sort_unstable_by(|&a, &b| {
            gradients[b as usize]
                .abs()
                .partial_cmp(&gradients[a as usize].abs())
                .unwrap()
        });

        let top: Vec<u32> = order[..top_k].to_vec();
        let rest = &order[top_k..];

        let mut rng = SmallRng::seed_from_u64(self.seed ^ (round as u64 * 0x517c_c1b7));
        let sampled_rest: Vec<u32> =
            rest.iter().copied().filter(|_| rng.gen::<f32>() < self.other_rate).collect();

        let mut out = top;
        out.extend_from_slice(&sampled_rest);
        out
    }
}

// ── Adaptive sampler ─────────────────────────────────────────────────────────

/// Class- and gradient-aware sampler.
///
/// Algorithm:
///   1. Partition examples into (large-gradient, small-gradient) by percentile.
///   2. Within each partition, stratify by class label.
///   3. Sample to approach `class_balance_target` positive fraction while
///      preserving the high-gradient signal.
///   4. Optionally up-weight examples near the decision boundary
///      (small |gradient| AND |p - 0.5| small, approximated by small h).
pub struct AdaptiveSampler {
    /// Keep all examples with |grad| in the top this fraction.
    pub keep_top_grad_frac: f32,
    /// Sample this fraction of the remaining examples.
    pub sample_rest_frac: f32,
    /// Target positive fraction after sampling.  `None` = natural distribution.
    pub class_balance_target: Option<f32>,
    /// Extra weight multiplier for boundary examples (heuristic, set 0 to disable).
    pub uncertainty_weight: f32,
    pub seed: u64,
}

impl AdaptiveSampler {
    pub fn new(
        keep_top_grad_frac: f32,
        sample_rest_frac: f32,
        class_balance_target: Option<f32>,
        uncertainty_weight: f32,
        seed: u64,
    ) -> Self {
        AdaptiveSampler {
            keep_top_grad_frac,
            sample_rest_frac,
            class_balance_target,
            uncertainty_weight,
            seed,
        }
    }
}

impl Sampler for AdaptiveSampler {
    fn sample_indices(
        &self,
        gradients: &[f32],
        hessians: &[f32],
        labels: &[f32],
        round: usize,
    ) -> Vec<u32> {
        let n = gradients.len();
        let mut rng = SmallRng::seed_from_u64(self.seed ^ (round as u64 * 0x6c62_272e));

        // Compute |gradient| magnitudes.
        let abs_grads: Vec<f32> = gradients.iter().map(|g| g.abs()).collect();

        // Find the percentile threshold for the top fraction.
        let top_k = ((n as f32 * self.keep_top_grad_frac).ceil() as usize).min(n);
        let threshold = {
            let mut sorted = abs_grads.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            if top_k == 0 { f32::INFINITY } else { sorted[n - top_k] }
        };

        let (mut pos_top, mut neg_top, mut pos_rest, mut neg_rest) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());

        for i in 0..n {
            let is_pos = labels[i] > 0.5;
            if abs_grads[i] >= threshold {
                if is_pos { pos_top.push(i as u32) } else { neg_top.push(i as u32) }
            } else {
                if is_pos { pos_rest.push(i as u32) } else { neg_rest.push(i as u32) }
            }
        }

        let mut out: Vec<u32> = Vec::with_capacity(n);
        // Always keep the top-gradient examples.
        out.extend_from_slice(&pos_top);
        out.extend_from_slice(&neg_top);

        // Sample the rest, optionally balancing classes.
        let target_pos_frac = self.class_balance_target.unwrap_or_else(|| {
            let total_pos = (pos_top.len() + pos_rest.len()) as f32;
            total_pos / n as f32
        });

        let n_rest_to_sample = ((pos_rest.len() + neg_rest.len()) as f32
            * self.sample_rest_frac)
            .ceil() as usize;

        // Split n_rest_to_sample into positive and negative quotas.
        let n_pos_quota = (n_rest_to_sample as f32 * target_pos_frac).round() as usize;
        let n_neg_quota = n_rest_to_sample - n_pos_quota;

        out.extend(sample_without_replacement(&pos_rest, n_pos_quota, &mut rng));
        out.extend(sample_without_replacement(&neg_rest, n_neg_quota, &mut rng));

        // Optional: also include high-uncertainty examples from the rest
        // (small |gradient| = close to decision boundary means uncertain).
        if self.uncertainty_weight > 0.0 {
            let h_threshold = {
                let mut sh: Vec<f32> = hessians.iter().copied().collect();
                sh.sort_by(|a, b| b.partial_cmp(a).unwrap()); // descending
                let k = (n as f32 * 0.05).ceil() as usize;
                sh.get(k).copied().unwrap_or(0.0)
            };
            for i in 0..n {
                if hessians[i] >= h_threshold && abs_grads[i] < threshold {
                    if rng.gen::<f32>() < self.uncertainty_weight {
                        out.push(i as u32);
                    }
                }
            }
        }

        out
    }
}

fn sample_without_replacement(
    pool: &[u32],
    n: usize,
    rng: &mut SmallRng,
) -> Vec<u32> {
    if n == 0 || pool.is_empty() {
        return vec![];
    }
    if n >= pool.len() {
        return pool.to_vec();
    }
    // Fisher-Yates partial shuffle on a copy.
    let mut v = pool.to_vec();
    let take = n.min(v.len());
    for i in 0..take {
        let j = i + rng.gen_range(0..(v.len() - i));
        v.swap(i, j);
    }
    v[..take].to_vec()
}
