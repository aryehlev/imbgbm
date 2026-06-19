use std::collections::HashMap;
use serde::{Deserialize, Serialize};

/// Target encoding for one categorical column.
///
/// Maps integer category codes to Bayesian-smoothed mean target values.
/// Two variants are computed during training:
///
/// - **OOF variant** (K-fold): used as the binned feature during training to
///   prevent target leakage (categories are encoded from held-out folds only).
/// - **Full-dataset variant** (stored in `Model`): used at inference; slightly
///   biased but lower variance because it uses all training examples.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct CatEncoding {
    /// Feature column index in the raw input vector.
    pub col_idx: usize,
    /// Category code (integer) → smoothed mean target value.
    // serde_json serialises i64 map keys as strings automatically.
    pub codes: HashMap<i64, f32>,
    /// Fallback value for unseen categories at inference time.
    pub global_mean: f32,
}

impl CatEncoding {
    /// Encode a raw feature value (integer category code) to its target mean.
    #[inline]
    pub fn encode(&self, raw: f32) -> f32 {
        self.codes.get(&(raw as i64)).copied().unwrap_or(self.global_mean)
    }
}

/// Compute OOF target encoding for a single categorical column.
///
/// Returns `(oof_values, full_encoding)`:
/// - `oof_values[i]` is the TE value for row `i` computed from the *other*
///   K-1 folds — safe to use as a training feature without leakage.
/// - `full_encoding` uses all rows and should be stored in the model for
///   inference (lower variance, acceptable bias).
///
/// Bayesian smoothing: `TE(cat) = (sum_y + α * μ) / (n + α)` where
/// `α = smoothing` and `μ = global positive rate`.
pub fn fit_target_encoding(
    codes: &[f32],
    labels: &[f32],
    k_folds: usize,
    smoothing: f32,
    col_idx: usize,
    seed: u64,
) -> (Vec<f32>, CatEncoding) {
    let n = codes.len();
    assert_eq!(n, labels.len());
    let global_mean: f32 = if n > 0 { labels.iter().sum::<f32>() / n as f32 } else { 0.5 };

    // ── Full-dataset encoding (inference) ────────────────────────────────────
    let mut sum_y: HashMap<i64, f32> = HashMap::new();
    let mut cnt:   HashMap<i64, u32> = HashMap::new();
    for (&c, &y) in codes.iter().zip(labels.iter()) {
        let code = c as i64;
        *sum_y.entry(code).or_insert(0.0) += y;
        *cnt.entry(code).or_insert(0) += 1;
    }
    let full_codes: HashMap<i64, f32> = sum_y
        .iter()
        .map(|(&code, &s)| {
            let n_cat = *cnt.get(&code).unwrap() as f32;
            (code, (s + smoothing * global_mean) / (n_cat + smoothing))
        })
        .collect();
    let inference_enc = CatEncoding { col_idx, codes: full_codes, global_mean };

    // ── OOF encoding (training features) ────────────────────────────────────
    let fold_assign = stratified_fold_assign(labels, k_folds, seed);
    let mut oof_values = vec![global_mean; n];

    for k in 0..k_folds {
        let mut fold_sum_y: HashMap<i64, f32> = HashMap::new();
        let mut fold_cnt:   HashMap<i64, u32> = HashMap::new();
        for i in 0..n {
            if fold_assign[i] != k {
                let code = codes[i] as i64;
                *fold_sum_y.entry(code).or_insert(0.0) += labels[i];
                *fold_cnt.entry(code).or_insert(0) += 1;
            }
        }
        for i in 0..n {
            if fold_assign[i] == k {
                let code = codes[i] as i64;
                let s = fold_sum_y.get(&code).copied().unwrap_or(0.0);
                let c = fold_cnt.get(&code).copied().unwrap_or(0) as f32;
                oof_values[i] = (s + smoothing * global_mean) / (c + smoothing);
            }
        }
    }

    (oof_values, inference_enc)
}

/// Stratified K-fold assignment for target encoding.
/// Positives and negatives are shuffled independently then interleaved across
/// folds to preserve class balance within each fold.
fn stratified_fold_assign(labels: &[f32], k_folds: usize, seed: u64) -> Vec<usize> {
    let n = labels.len();
    let mut pos: Vec<usize> = labels.iter().enumerate()
        .filter(|(_, &y)| y > 0.5).map(|(i, _)| i).collect();
    let mut neg: Vec<usize> = labels.iter().enumerate()
        .filter(|(_, &y)| y <= 0.5).map(|(i, _)| i).collect();

    let mut rng = seed;
    let shuffle = |v: &mut Vec<usize>, s: u64| {
        let mut r = s;
        for i in (1..v.len()).rev() {
            r = r.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let j = (r >> 33) as usize % (i + 1);
            v.swap(i, j);
        }
    };
    shuffle(&mut pos, rng);
    rng ^= 0xdead_beef_cafe_0000;
    shuffle(&mut neg, rng);

    let mut assignment = vec![0usize; n];
    for (slot, &i) in pos.iter().enumerate() { assignment[i] = slot % k_folds; }
    for (slot, &i) in neg.iter().enumerate() { assignment[i] = slot % k_folds; }
    assignment
}
