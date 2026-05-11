//! Two-step boosting from biased positive seeds (idea 6).
//!
//! Real-world advertiser "good user" seeds are not a representative sample of
//! the true positive distribution — they are only the positives the advertiser
//! has already found, often dominated by a single acquisition channel
//! (e.g. Facebook converters). A vanilla classifier trained on these seeds
//! cannot recognise good users from other channels (search, news, gaming)
//! because they look unlabeled.
//!
//! `train_with_seed_expansion` runs two boosting passes:
//!   1. **Pilot pass**: train on `(seeds=positive, rest=unlabeled)` using the
//!      caller-supplied config. Score the unlabeled rows.
//!   2. **Promotion**: rank unlabeled rows by score, take the top
//!      `expansion_fraction`, and promote them to "soft positives" with a
//!      reliability weight `promotion_weight ∈ (0, 1]` applied via the IPC
//!      mechanism.
//!   3. **Refinement pass**: retrain on the union of the original seeds and
//!      the promoted positives, using the same config. The final model can
//!      assign mass to positive modes the original seeds did not cover.
//!
//! The promotion step is *gated*: if the promoted set's average pilot score
//! is below `min_promotion_score`, we skip expansion and return the pilot
//! model unchanged (under the assumption that the unlabeled pool genuinely
//! contains few hidden positives and expansion would harm calibration).
//!
//! Combine with `imbgbm_calib::estimate_pu_label_rate` to also output
//! properly-corrected probabilities.

use std::sync::Arc;

use imbgbm_core::{Dataset, RowMetadata};
use imbgbm_infer::Model;

use crate::{config::Config, loop_::train};

/// Controls for biased-positive expansion.
#[derive(Clone, Debug)]
pub struct SeedExpansionConfig {
    /// Top fraction of unlabeled rows (by pilot-pass score) to promote to
    /// soft positives in the refinement pass. Typical: 0.01 – 0.10.
    pub expansion_fraction: f32,
    /// Soft-label weight assigned to the promoted positives. Lower values
    /// keep the model conservative about expansion. Typical: 0.3 – 0.7.
    pub promotion_weight: f32,
    /// Minimum mean pilot score among promoted rows for the expansion to be
    /// retained. If the top quantile is still scoring near baseline, skip
    /// expansion entirely.
    pub min_promotion_score: f32,
}

impl Default for SeedExpansionConfig {
    fn default() -> Self {
        SeedExpansionConfig {
            expansion_fraction: 0.03,
            promotion_weight: 0.5,
            min_promotion_score: 0.3,
        }
    }
}

/// Outcome of the seed-expansion procedure.
pub struct SeedExpansionResult {
    /// Final trained model.
    pub model: Model,
    /// Number of unlabeled rows promoted to soft positives. Zero if expansion
    /// was skipped (`mean_promoted_score < min_promotion_score`).
    pub n_promoted: usize,
    /// Mean pilot score across the promoted set (or the would-be promoted set
    /// when expansion is skipped). Useful for diagnostics.
    pub mean_promoted_score: f32,
}

/// Run pilot → promote → refine.
///
/// The dataset is the original one with seed positives encoded as `label = 1`
/// and everything else as `label = 0`. The refinement pass mutates a *copy*
/// of the labels — the original `dataset` is not modified.
pub fn train_with_seed_expansion(
    dataset: &Dataset,
    config: &Config,
    expansion: &SeedExpansionConfig,
) -> SeedExpansionResult {
    // ── 1. Pilot pass ─────────────────────────────────────────────────────────
    let pilot = train(dataset, config);

    // ── 2. Score the unlabeled rows ───────────────────────────────────────────
    let n = dataset.labels.len();
    let mut scored: Vec<(usize, f32)> = Vec::with_capacity(n);
    for i in 0..n {
        if dataset.labels[i] <= 0.5 {
            let row = dataset.row_copy(i);
            scored.push((i, pilot.predict_proba_raw(&row)));
        }
    }
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let n_promote = ((scored.len() as f32) * expansion.expansion_fraction).ceil() as usize;
    let promoted_slice = &scored[..n_promote.min(scored.len())];
    let mean_promoted_score = if promoted_slice.is_empty() {
        0.0
    } else {
        promoted_slice.iter().map(|(_, s)| s).sum::<f32>() / promoted_slice.len() as f32
    };

    // Gate: if the pilot does not assign high enough probability to the
    // top-quantile unlabeled rows, expansion is unlikely to help.
    if mean_promoted_score < expansion.min_promotion_score || promoted_slice.is_empty() {
        return SeedExpansionResult { model: pilot, n_promoted: 0, mean_promoted_score };
    }

    // ── 3. Build expanded labels ──────────────────────────────────────────────
    // Promoted rows get a *soft label* equal to `promotion_weight`. Hard labels
    // (true seeds) keep label = 1.0. This is interpreted by the loss as a
    // weighted target: e.g. for BCE, `g = p - 0.5` for a half-labelled row.
    let mut new_labels = dataset.labels.clone();
    for &(idx, _) in promoted_slice {
        new_labels[idx] = expansion.promotion_weight;
    }
    let new_dataset = with_labels(dataset, new_labels);

    // ── 4. Refinement pass ───────────────────────────────────────────────────
    // Reuse the same Config — Sampler and Splitter are stateless, and the
    // `metadata` reference is row-aligned (we only changed labels).
    let refined = train(&new_dataset, config);

    SeedExpansionResult { model: refined, n_promoted: promoted_slice.len(), mean_promoted_score }
}

fn with_labels(dataset: &Dataset, new_labels: Vec<f32>) -> Dataset {
    Dataset {
        features: dataset.features.clone(),
        labels: new_labels,
        n_rows: dataset.n_rows,
        n_cols: dataset.n_cols,
    }
}

/// Helper that lets callers supply a custom metadata when re-training (e.g.
/// to plug in cluster IDs computed from the pilot model's positive predictions).
pub fn train_with_seed_expansion_with_metadata(
    dataset: &Dataset,
    base_config: &Config,
    expansion: &SeedExpansionConfig,
    refine_metadata: Option<Arc<RowMetadata>>,
) -> SeedExpansionResult {
    let pilot = train(dataset, base_config);

    let n = dataset.labels.len();
    let mut scored: Vec<(usize, f32)> = Vec::with_capacity(n);
    for i in 0..n {
        if dataset.labels[i] <= 0.5 {
            let row = dataset.row_copy(i);
            scored.push((i, pilot.predict_proba_raw(&row)));
        }
    }
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let n_promote = ((scored.len() as f32) * expansion.expansion_fraction).ceil() as usize;
    let promoted_slice = &scored[..n_promote.min(scored.len())];
    let mean_promoted_score = if promoted_slice.is_empty() { 0.0 } else {
        promoted_slice.iter().map(|(_, s)| s).sum::<f32>() / promoted_slice.len() as f32
    };

    if mean_promoted_score < expansion.min_promotion_score || promoted_slice.is_empty() {
        return SeedExpansionResult { model: pilot, n_promoted: 0, mean_promoted_score };
    }

    let mut new_labels = dataset.labels.clone();
    for &(idx, _) in promoted_slice {
        new_labels[idx] = expansion.promotion_weight;
    }
    let new_dataset = with_labels(dataset, new_labels);

    let mut refine_config = clone_config_with_metadata(base_config, refine_metadata);
    let _ = &mut refine_config; // suppress unused-mut warning when metadata is None.
    let refined = train(&new_dataset, &refine_config);

    SeedExpansionResult { model: refined, n_promoted: promoted_slice.len(), mean_promoted_score }
}

fn clone_config_with_metadata(c: &Config, md: Option<Arc<RowMetadata>>) -> Config {
    Config {
        n_rounds: c.n_rounds,
        learning_rate: c.learning_rate,
        max_depth: c.max_depth,
        min_child_weight: c.min_child_weight,
        min_samples_leaf: c.min_samples_leaf,
        lambda: c.lambda,
        n_bins: c.n_bins,
        k_folds: c.k_folds,
        calibrate: c.calibrate,
        fold_strategy: c.fold_strategy.clone(),
        metadata: md.or_else(|| c.metadata.clone()),
        objective: c.objective.clone(),
        sampler: c.sampler.clone(),
        splitter: c.splitter.clone(),
        early_stopping_rounds: c.early_stopping_rounds,
        col_subsample: c.col_subsample,
        platt_scale: c.platt_scale,
        seed: c.seed,
    }
}
