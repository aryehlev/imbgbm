use imbgbm_core::RowMetadata;
use rand::{Rng, SeedableRng};
use rand::rngs::SmallRng;
use std::collections::HashMap;

pub mod modes;

// ── SampleSet ────────────────────────────────────────────────────────────────

/// The output of a single sampling step: which rows to use and their IPC weights.
///
/// The `ipc_weights[j]` is 1/π_{indices[j]}, where π_i is the probability
/// that row i was included.  Multiplying gradients and hessians by these
/// weights in histogram accumulation corrects for the sampling distribution
/// shift, preserving unbiased gradient/hessian estimates.
pub struct SampleSet {
    /// Row indices (subset of 0..n_rows).
    pub indices: Vec<u32>,
    /// Inverse-probability correction weights, parallel to `indices`.
    /// All-ones means no correction (uniform sampling probability).
    pub ipc_weights: Vec<f32>,
}

impl SampleSet {
    /// Construct a SampleSet with no IPC correction (all weights = 1).
    pub fn uniform_weights(indices: Vec<u32>) -> Self {
        let n = indices.len();
        SampleSet { indices, ipc_weights: vec![1.0; n] }
    }
}

// ── Sampler trait ────────────────────────────────────────────────────────────

/// Decides which row indices each tree will see, and with what IPC weights.
pub trait Sampler: Send + Sync {
    fn sample(
        &self,
        gradients: &[f32],
        hessians: &[f32],
        labels: &[f32],
        metadata: &RowMetadata,
        round: usize,
    ) -> SampleSet;
}

// ── Uniform sampler ──────────────────────────────────────────────────────────

/// Bernoulli row sub-sampling at a fixed rate.
///
/// IPC weight = 1/subsample for every sampled row, so histograms remain
/// unbiased estimates of the full-data gradients.
pub struct UniformSampler {
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
    fn sample(
        &self,
        gradients: &[f32],
        _hessians: &[f32],
        _labels: &[f32],
        _metadata: &RowMetadata,
        round: usize,
    ) -> SampleSet {
        let n = gradients.len();
        if self.subsample >= 1.0 {
            return SampleSet::uniform_weights((0..n as u32).collect());
        }
        let mut rng = SmallRng::seed_from_u64(self.seed ^ (round as u64 * 0x9e37_79b9));
        let ipc = 1.0 / self.subsample;
        let mut indices = Vec::new();
        let mut weights = Vec::new();
        for i in 0..n as u32 {
            if rng.gen::<f32>() < self.subsample {
                indices.push(i);
                weights.push(ipc);
            }
        }
        SampleSet { indices, ipc_weights: weights }
    }
}

// ── GOSS sampler ─────────────────────────────────────────────────────────────

/// Gradient-One-Side Sampling (Ke et al. 2017) with true IPC correction.
///
/// - High-gradient examples (top `top_rate` fraction by |grad|): always kept,
///   IPC weight = 1.0.
/// - Remaining examples: sampled at rate `other_rate`, IPC weight = 1/other_rate.
///
/// With these weights the histogram gradient/hessian sums are unbiased
/// estimates of the full-data sums, unlike the original GOSS paper which uses
/// a heuristic `(1-top_rate)/other_rate` scaling.
pub struct GossSampler {
    pub top_rate: f32,
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
    fn sample(
        &self,
        gradients: &[f32],
        _hessians: &[f32],
        _labels: &[f32],
        _metadata: &RowMetadata,
        round: usize,
    ) -> SampleSet {
        let n = gradients.len();
        let top_k = ((n as f32 * self.top_rate).ceil() as usize).min(n);

        let mut order: Vec<u32> = (0..n as u32).collect();
        order.sort_unstable_by(|&a, &b| {
            gradients[b as usize]
                .abs()
                .partial_cmp(&gradients[a as usize].abs())
                .unwrap()
        });

        let top_indices = &order[..top_k];
        let rest = &order[top_k..];

        let mut rng = SmallRng::seed_from_u64(self.seed ^ (round as u64 * 0x517c_c1b7));
        let tail_ipc = 1.0 / self.other_rate;

        let mut indices = Vec::with_capacity(top_k + (rest.len() as f32 * self.other_rate) as usize + 1);
        let mut weights = Vec::with_capacity(indices.capacity());

        for &i in top_indices {
            indices.push(i);
            weights.push(1.0_f32);
        }
        for &i in rest {
            if rng.gen::<f32>() < self.other_rate {
                indices.push(i);
                weights.push(tail_ipc);
            }
        }

        SampleSet { indices, ipc_weights: weights }
    }
}

// ── Adaptive sampler ─────────────────────────────────────────────────────────

/// Class- and gradient-aware sampler with optional per-segment quotas.
///
/// ## Sampling algorithm
/// 1. Partition rows into (high-gradient, low-gradient) by |grad| percentile.
/// 2. Stratify each partition by class label.
/// 3. If `class_balance_target` is set, bias sampling toward that positive
///    fraction; otherwise preserve the natural distribution.
/// 4. Optionally boost examples near the decision boundary (large hessian,
///    small |gradient|).
/// 5. If `segment_quotas` is non-empty, ensure each unique segment value
///    contributes at least `min_per_segment` examples.
///
/// ## IPC weights
/// Each returned row carries 1/π_i, where π_i is the stratum-specific
/// sampling probability computed from the above scheme.
pub struct AdaptiveSampler {
    pub keep_top_grad_frac: f32,
    pub sample_rest_frac: f32,
    pub class_balance_target: Option<f32>,
    pub uncertainty_weight: f32,
    /// Per-segment-dimension quota constraints.
    pub segment_quotas: Vec<SegmentQuota>,
    pub seed: u64,
}

/// Quota constraint for one segment dimension.
#[derive(Clone)]
pub struct SegmentQuota {
    /// Index into `RowMetadata::segments`.
    pub segment_dim: usize,
    /// Minimum number of examples from every non-empty segment value per tree.
    pub min_per_segment: usize,
    /// Maximum fraction of the total sample budget from any single segment.
    pub max_fraction: f32,
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
            segment_quotas: vec![],
            seed,
        }
    }

    pub fn with_quota(mut self, quota: SegmentQuota) -> Self {
        self.segment_quotas.push(quota);
        self
    }
}

impl Sampler for AdaptiveSampler {
    fn sample(
        &self,
        gradients: &[f32],
        hessians: &[f32],
        labels: &[f32],
        metadata: &RowMetadata,
        round: usize,
    ) -> SampleSet {
        let n = gradients.len();
        let mut rng = SmallRng::seed_from_u64(self.seed ^ (round as u64 * 0x6c62_272e));

        let abs_grads: Vec<f32> = gradients.iter().map(|g| g.abs()).collect();

        // Threshold for "high gradient" stratum.
        let top_k = ((n as f32 * self.keep_top_grad_frac).ceil() as usize).min(n);
        let grad_threshold = {
            let mut sorted = abs_grads.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            if top_k == 0 { f32::INFINITY } else { sorted[n - top_k] }
        };

        // Partition into four strata.
        let (mut pos_top, mut neg_top, mut pos_rest, mut neg_rest) =
            (Vec::<u32>::new(), Vec::<u32>::new(), Vec::<u32>::new(), Vec::<u32>::new());

        for i in 0..n {
            let is_pos = labels[i] > 0.5;
            let is_high = abs_grads[i] >= grad_threshold;
            match (is_pos, is_high) {
                (true,  true)  => pos_top.push(i as u32),
                (false, true)  => neg_top.push(i as u32),
                (true,  false) => pos_rest.push(i as u32),
                (false, false) => neg_rest.push(i as u32),
            }
        }

        // Natural positive fraction.
        let pos_frac_natural = (pos_top.len() + pos_rest.len()) as f32 / n as f32;
        let target_pos_frac = self.class_balance_target.unwrap_or(pos_frac_natural);

        let n_rest_budget = ((pos_rest.len() + neg_rest.len()) as f32 * self.sample_rest_frac)
            .ceil() as usize;
        let n_pos_quota = (n_rest_budget as f32 * target_pos_frac).round() as usize;
        let n_neg_quota = n_rest_budget.saturating_sub(n_pos_quota);

        // Sampling probabilities for IPC.
        let pi_pos_rest = if pos_rest.is_empty() {
            1.0
        } else {
            (n_pos_quota as f32 / pos_rest.len() as f32).min(1.0).max(f32::EPSILON)
        };
        let pi_neg_rest = if neg_rest.is_empty() {
            1.0
        } else {
            (n_neg_quota as f32 / neg_rest.len() as f32).min(1.0).max(f32::EPSILON)
        };

        let mut indices: Vec<u32> = Vec::new();
        let mut weights: Vec<f32> = Vec::new();

        // High-gradient examples: π = 1.
        for &i in pos_top.iter().chain(neg_top.iter()) {
            indices.push(i);
            weights.push(1.0);
        }

        // Low-gradient positives.
        let sampled_pos = sample_without_replacement(&pos_rest, n_pos_quota, &mut rng);
        for i in sampled_pos {
            indices.push(i);
            weights.push(1.0 / pi_pos_rest);
        }

        // Low-gradient negatives.
        let sampled_neg = sample_without_replacement(&neg_rest, n_neg_quota, &mut rng);
        for i in sampled_neg {
            indices.push(i);
            weights.push(1.0 / pi_neg_rest);
        }

        // Uncertainty bonus: near-boundary examples (high h, low |g|).
        if self.uncertainty_weight > 0.0 {
            let h_thresh = {
                let mut sh: Vec<f32> = hessians.iter().copied().collect();
                sh.sort_by(|a, b| b.partial_cmp(a).unwrap());
                let k = (n as f32 * 0.05).ceil() as usize;
                sh.get(k).copied().unwrap_or(0.0)
            };
            for i in 0..n {
                if hessians[i] >= h_thresh && abs_grads[i] < grad_threshold {
                    if rng.gen::<f32>() < self.uncertainty_weight {
                        // These are additional samples; π treated as uncertainty_weight.
                        indices.push(i as u32);
                        weights.push(1.0 / self.uncertainty_weight.max(f32::EPSILON));
                    }
                }
            }
        }

        // Segment quota enforcement: ensure each segment value has at least
        // min_per_segment examples in the final sample set.
        for quota in &self.segment_quotas {
            let dim = quota.segment_dim;
            if dim >= metadata.n_segment_dims() {
                continue;
            }
            let seg_ids = &metadata.segments[dim];

            // Count current samples per segment value.
            let mut seg_counts: HashMap<u32, usize> = HashMap::new();
            for &row in &indices {
                *seg_counts.entry(seg_ids[row as usize]).or_insert(0) += 1;
            }

            // For each under-quota segment, pull in additional random examples.
            let mut by_segment: HashMap<u32, Vec<u32>> = HashMap::new();
            for i in 0..n as u32 {
                by_segment.entry(seg_ids[i as usize]).or_default().push(i);
            }

            for (seg_val, pool) in &by_segment {
                // Segments with the NO_CLUSTER sentinel (e.g. unlabeled rows in
                // a positive-mode segment) are not enforced — only real modes.
                if *seg_val == crate::modes::NO_CLUSTER { continue; }
                let current = *seg_counts.get(seg_val).unwrap_or(&0);
                let needed = quota.min_per_segment.saturating_sub(current);
                if needed == 0 {
                    continue;
                }
                // Already-included rows filtered out; π for these extras = needed/pool_size.
                let included: std::collections::HashSet<u32> =
                    indices.iter().copied().collect();
                let extras: Vec<u32> =
                    pool.iter().copied().filter(|r| !included.contains(r)).collect();
                let pi_extra = (needed as f32 / extras.len().max(1) as f32)
                    .min(1.0)
                    .max(f32::EPSILON);
                let drawn = sample_without_replacement(&extras, needed, &mut rng);
                for row in drawn {
                    indices.push(row);
                    weights.push(1.0 / pi_extra);
                }
            }

            // Cap any segment that exceeds max_fraction of the budget.
            let budget = indices.len();
            let max_count = (budget as f32 * quota.max_fraction).ceil() as usize;
            let mut new_indices = Vec::with_capacity(indices.len());
            let mut new_weights = Vec::with_capacity(weights.len());
            let mut seg_counts: HashMap<u32, usize> = HashMap::new();
            for (j, (&row, &w)) in indices.iter().zip(weights.iter()).enumerate() {
                let count = seg_counts.entry(seg_ids[row as usize]).or_insert(0);
                if *count < max_count {
                    new_indices.push(row);
                    new_weights.push(w);
                    *count += 1;
                }
                let _ = j;
            }
            indices = new_indices;
            weights = new_weights;
        }

        SampleSet { indices, ipc_weights: weights }
    }
}

// ── PU-GOSS sampler ──────────────────────────────────────────────────────────

/// Positive-Unlabeled Gradient-Based One-Side Sampling.
///
/// A PU-specialised refinement of LightGBM's GOSS. GOSS keeps the
/// largest-|gradient| examples and subsamples the rest. PU-GOSS recognises that
/// in a PU setup the unlabeled pool contains hidden positives, and that the
/// most informative training rows are not just high-gradient: they are also
/// (a) suspiciously-high-scoring unlabeled (likely hidden positives), and
/// (b) unlabeled close to the decision boundary (likely lookalikes).
///
/// ## Strata (each tree's sample is the union of)
///   1. **All labeled positives** (no subsampling — they are precious).
///   2. **High-|gradient| unlabeled** ("hard" — currently mis-fit).
///   3. **High-score unlabeled** ("suspicious" — likely hidden positives).
///   4. **High-uncertainty unlabeled** (large hessian, |grad| near zero —
///      boundary lookalikes; relevant for hard-negative mining).
///   5. **Reliable-negative unlabeled** (very low score, low gradient) —
///      subsampled at `reliable_neg_rate` to anchor the negative class.
///
/// Score for unlabeled examples is recovered from `|gradient|`, which equals
/// `p` for BCE/focal/PU losses when `y = 0`.
///
/// ## IPC weights
/// Strata 2–5 each carry `1 / sampling_probability` so the histogram remains an
/// unbiased estimator of the full-data gradient. Labeled positives have weight
/// 1 since they are always included.
pub struct PuGossSampler {
    pub top_grad_rate: f32,
    pub top_score_rate: f32,
    pub uncertainty_rate: f32,
    pub reliable_neg_rate: f32,
    /// Hard-negative mining boost: oversample stratum 4 by this factor
    /// (IPC adjusted accordingly so it stays unbiased).
    pub hard_neg_boost: f32,
    pub seed: u64,
}

impl PuGossSampler {
    pub fn new(
        top_grad_rate: f32,
        top_score_rate: f32,
        uncertainty_rate: f32,
        reliable_neg_rate: f32,
        seed: u64,
    ) -> Self {
        for &r in &[top_grad_rate, top_score_rate, uncertainty_rate, reliable_neg_rate] {
            assert!((0.0..=1.0).contains(&r), "PU-GOSS rates must lie in [0, 1]");
        }
        PuGossSampler {
            top_grad_rate,
            top_score_rate,
            uncertainty_rate,
            reliable_neg_rate,
            hard_neg_boost: 1.0,
            seed,
        }
    }

    pub fn with_hard_neg_boost(mut self, boost: f32) -> Self {
        assert!(boost >= 1.0, "boost must be >= 1.0");
        self.hard_neg_boost = boost;
        self
    }
}

impl Sampler for PuGossSampler {
    fn sample(
        &self,
        gradients: &[f32],
        hessians: &[f32],
        labels: &[f32],
        _metadata: &RowMetadata,
        round: usize,
    ) -> SampleSet {
        let n = gradients.len();
        let mut rng = SmallRng::seed_from_u64(self.seed ^ (round as u64 * 0xa511_7f8b));

        // Partition rows by label.
        let mut pos: Vec<u32> = Vec::new();
        let mut unl: Vec<u32> = Vec::with_capacity(n);
        for i in 0..n {
            if labels[i] > 0.5 { pos.push(i as u32) } else { unl.push(i as u32) };
        }

        let abs_grads: Vec<f32> = gradients.iter().map(|g| g.abs()).collect();
        // "Boundary uncertainty" = hessian (high for p≈0.5, low for p≈0 or 1).
        // We rank unlabeled by hessian * (1 - 2*|grad|/(|grad|+EPS)) — i.e.
        // high hessian + low |grad|. Equivalent to ranking by p*(1-p) when
        // (1 - 2|y - p|) is large.
        let uncertainty: Vec<f32> = (0..n)
            .map(|i| hessians[i] * (1.0 - abs_grads[i]).max(0.0))
            .collect();

        // Build sorted unlabeled index pools by each criterion (descending).
        let mut by_grad: Vec<u32> = unl.clone();
        by_grad.sort_unstable_by(|&a, &b| {
            abs_grads[b as usize].partial_cmp(&abs_grads[a as usize]).unwrap()
        });
        let mut by_score: Vec<u32> = unl.clone();
        // For unlabeled (y=0), |gradient| = p (BCE/focal). For other losses,
        // |gradient| is still monotone in the score, so this remains a valid
        // ranking.
        by_score.sort_unstable_by(|&a, &b| {
            abs_grads[b as usize].partial_cmp(&abs_grads[a as usize]).unwrap()
        });
        let mut by_unc: Vec<u32> = unl.clone();
        by_unc.sort_unstable_by(|&a, &b| {
            uncertainty[b as usize].partial_cmp(&uncertainty[a as usize]).unwrap()
        });
        // Reliable negatives: lowest score AND low |grad|.
        let mut by_reliable: Vec<u32> = unl.clone();
        by_reliable.sort_unstable_by(|&a, &b| {
            abs_grads[a as usize].partial_cmp(&abs_grads[b as usize]).unwrap()
        });

        let nu = unl.len() as f32;
        let take = |frac: f32| -> usize {
            ((nu * frac).ceil() as usize).min(unl.len())
        };
        let k_grad   = take(self.top_grad_rate);
        let k_score  = take(self.top_score_rate);
        // Hard-negative mining: take more uncertainty examples but mark the
        // sampling probability so the IPC correction stays honest.
        let k_unc    = take((self.uncertainty_rate * self.hard_neg_boost).min(1.0));
        let k_rel    = take(self.reliable_neg_rate);

        let mut taken = std::collections::HashSet::new();
        let mut indices: Vec<u32> = Vec::with_capacity(pos.len() + k_grad + k_score + k_unc + k_rel);
        let mut weights: Vec<f32> = Vec::with_capacity(indices.capacity());

        // Stratum 1: all positives (no subsampling).
        for &p in &pos {
            indices.push(p);
            weights.push(1.0);
            taken.insert(p);
        }

        let push_stratum =
            |pool: &[u32], k: usize, stratum_size: usize, taken: &mut std::collections::HashSet<u32>,
             indices: &mut Vec<u32>, weights: &mut Vec<f32>| {
                if k == 0 || stratum_size == 0 { return; }
                let pi = (k as f32 / stratum_size as f32).clamp(f32::EPSILON, 1.0);
                let w = 1.0 / pi;
                let mut added = 0usize;
                for &row in pool {
                    if added >= k { break; }
                    if taken.insert(row) {
                        indices.push(row);
                        weights.push(w);
                        added += 1;
                    }
                }
            };

        // Stratum 2: top |gradient|.
        push_stratum(&by_grad, k_grad, unl.len(), &mut taken, &mut indices, &mut weights);
        // Stratum 3: top score (suspicious unlabeled).
        push_stratum(&by_score, k_score, unl.len(), &mut taken, &mut indices, &mut weights);
        // Stratum 4: high uncertainty / hard-negative lookalikes.
        push_stratum(&by_unc, k_unc, unl.len(), &mut taken, &mut indices, &mut weights);
        // Stratum 5: reliable negatives — sample uniformly from the bottom
        // half by |gradient| so we do not always pick the same anchors.
        let bottom_half_end = (unl.len() / 2).max(k_rel);
        let bottom_pool: Vec<u32> = by_reliable.iter().take(bottom_half_end).copied().collect();
        if !bottom_pool.is_empty() && k_rel > 0 {
            let drawn = sample_without_replacement(&bottom_pool, k_rel, &mut rng);
            let pi = (k_rel as f32 / unl.len() as f32).clamp(f32::EPSILON, 1.0);
            let w = 1.0 / pi;
            for row in drawn {
                if taken.insert(row) {
                    indices.push(row);
                    weights.push(w);
                }
            }
        }

        SampleSet { indices, ipc_weights: weights }
    }
}

fn sample_without_replacement(pool: &[u32], n: usize, rng: &mut SmallRng) -> Vec<u32> {
    if n == 0 || pool.is_empty() {
        return vec![];
    }
    if n >= pool.len() {
        return pool.to_vec();
    }
    let mut v = pool.to_vec();
    let take = n.min(v.len());
    for i in 0..take {
        let j = i + rng.gen_range(0..(v.len() - i));
        v.swap(i, j);
    }
    v[..take].to_vec()
}
