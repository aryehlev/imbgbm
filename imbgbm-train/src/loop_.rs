use imbgbm_calib::{assign_folds, calibrate_tree, calibrate_tree_with_oof, fit_platt, fit_raw_isotonic};
use imbgbm_core::{BinnedDataset, BoostingState, Dataset, RowMetadata};
use imbgbm_infer::{route_all, Model, RawIsoCal};

use crate::{builder::grow_tree, config::{Config, TailWeightConfig}};

/// Train an imbgbm model.
pub fn train(dataset: &Dataset, config: &Config) -> Model {
    let binned = BinnedDataset::from_dataset(dataset, config.n_bins);
    let n_rows = binned.n_rows;

    // Resolve metadata: use the config-supplied one or an empty default.
    let empty_meta = RowMetadata::empty();
    let metadata: &RowMetadata = config
        .metadata
        .as_deref()
        .unwrap_or(&empty_meta);

    // Initial score: log-odds of the class prior.
    let prior = config
        .objective
        .class_prior()
        .unwrap_or_else(|| binned.class_prior());
    let init_score = if prior > 0.0 && prior < 1.0 {
        (prior / (1.0 - prior)).ln()
    } else {
        0.0
    };

    let mut state = BoostingState::new(n_rows, init_score);

    let folds = if config.calibrate {
        Some(assign_folds(&binned.labels, metadata, config.k_folds, &config.fold_strategy))
    } else {
        None
    };

    let all_indices: Vec<u32> = (0..n_rows as u32).collect();
    let mut trees = Vec::with_capacity(config.n_rounds);
    let mut best_loss = f32::INFINITY;
    let mut rounds_without_improvement = 0usize;

    // Per-row OOF boosted score (used to fit Platt or isotonic calibration).
    let mut oof_predictions: Vec<f32> = vec![init_score; n_rows];
    let want_platt    = config.calibrate && config.platt_scale && !config.raw_isotonic;
    let want_raw_iso  = config.calibrate && config.raw_isotonic;

    for round in 0..config.n_rounds {
        // ── 1. Gradients ────────────────────────────────────────────────────
        let (g, h) = config.objective.grad_hess(&binned.labels, &state.predictions);
        state.gradients = g;
        state.hessians = h;

        // ── 1b. Top-tail gradient weighting ─────────────────────────────────
        // Upweight examples near the deployment threshold (top-K boundary) and
        // suppress easy negatives.  Applied by scaling gradients and hessians
        // so the histogram-building step naturally focuses on the ranking margin.
        if let Some(ref tw) = config.tail_weight {
            if round >= tw.start_round {
                let weights = tail_weights(&binned.labels, &state.predictions, tw);
                for i in 0..n_rows {
                    state.gradients[i] *= weights[i];
                    state.hessians[i] *= weights[i];
                }
            }
        }

        // ── 2. Sampling (returns indices + IPC weights) ──────────────────────
        let sample = config.sampler.sample(
            &state.gradients,
            &state.hessians,
            &binned.labels,
            metadata,
            round,
        );

        // ── 3. Tree growing (uses IPC weights in histogram building) ─────────
        let mut tree = grow_tree(
            &binned,
            &sample.indices,
            &sample.ipc_weights,
            &state.gradients,
            &state.hessians,
            config.splitter.as_ref(),
            config.max_depth,
            config.min_child_weight,
            config.min_samples_leaf,
            config.lambda,
            config.objective.class_prior(),
            config.col_subsample,
            config.seed.wrapping_add(round as u64 * 2654435761),
        );

        // ── 4. OOF calibration ───────────────────────────────────────────────
        if config.calibrate {
            if let Some(ref fold_assign) = folds {
                if want_platt {
                    // Platt scaling needs per-row OOF accumulated scores.
                    let per_row_oof = calibrate_tree_with_oof(
                        &mut tree,
                        &binned,
                        &all_indices,
                        &state.gradients,
                        &state.hessians,
                        fold_assign,
                        config.k_folds,
                        config.lambda,
                    );
                    for row in 0..n_rows {
                        oof_predictions[row] += config.learning_rate * per_row_oof[row];
                    }
                } else if !want_raw_iso {
                    // Per-leaf OOF positive-rate calibration.
                    // Skipped when raw_isotonic=true: the isotonic path trains K
                    // separate fold models with calibrate=false, so the main model
                    // must also stay raw — otherwise the score distributions diverge
                    // and the isotonic mapping is applied out-of-distribution.
                    calibrate_tree(
                        &mut tree,
                        &binned,
                        &all_indices,
                        &state.gradients,
                        &state.hessians,
                        fold_assign,
                        config.k_folds,
                        config.lambda,
                    );
                }
            }
        }

        // ── 5. Update predictions ─────────────────────────────────────────────
        let leaf_assignments = route_all(&tree, &binned.bins, n_rows);
        for row in 0..n_rows {
            state.predictions[row] +=
                config.learning_rate * tree.leaf_values[leaf_assignments[row]];
        }

        trees.push(tree);

        // ── Early stopping ────────────────────────────────────────────────────
        if let Some(patience) = config.early_stopping_rounds {
            let loss = mean_log_loss(&binned.labels, &state.predictions);
            if loss < best_loss - 1e-6 {
                best_loss = loss;
                rounds_without_improvement = 0;
            } else {
                rounds_without_improvement += 1;
                if rounds_without_improvement >= patience {
                    break;
                }
            }
        }
    }

    let mut model = Model::new(trees, config.learning_rate, init_score);

    // Fit Platt scaling on OOF boosted scores (preserves additive structure).
    if want_platt {
        let (a, b) = fit_platt(&oof_predictions, &binned.labels);
        model = model.with_platt(a, b);
    }

    // Fit isotonic calibration by training K held-out fold models and collecting
    // their predictions on the rows they never saw.  These are genuine
    // out-of-sample scores with the same distribution as full-model test scores,
    // so no scale/offset correction is needed at inference.
    if want_raw_iso {
        let true_oof = collect_kfold_raw_oof_scores(dataset, config);
        let (scores, probs) = fit_raw_isotonic(&true_oof, &binned.labels);
        model.raw_iso_cal = RawIsoCal { scores, probs };
    }

    model
}

/// Train K fold models on K-1 folds each; return per-row raw scores on the
/// held-out fold.  These are genuine out-of-sample predictions whose score
/// distribution matches the full-model test distribution, so they can feed
/// isotonic calibration without any scale/offset correction.
fn collect_kfold_raw_oof_scores(dataset: &Dataset, config: &Config) -> Vec<f32> {
    let n = dataset.n_rows;
    let binned_tmp = BinnedDataset::from_dataset(dataset, config.n_bins);
    let empty_meta = RowMetadata::empty();
    let metadata = config.metadata.as_deref().unwrap_or(&empty_meta);
    let fold_assign = assign_folds(
        &binned_tmp.labels, metadata, config.k_folds, &config.fold_strategy,
    );

    let mut oof_scores = vec![0.0f32; n];

    for fold in 0..config.k_folds {
        // Build fold training/validation splits.
        let train_indices: Vec<u32> = (0..n as u32)
            .filter(|&i| fold_assign[i as usize] != fold).collect();
        let val_rows: Vec<usize> = (0..n)
            .filter(|&i| fold_assign[i] == fold).collect();

        let fold_dataset = dataset.subset(&train_indices);

        // Fold model config: same hyper-params, no calibration, no isotonic.
        let fold_config = Config {
            n_rounds:               config.n_rounds,
            learning_rate:          config.learning_rate,
            max_depth:              config.max_depth,
            min_child_weight:       config.min_child_weight,
            min_samples_leaf:       config.min_samples_leaf,
            lambda:                 config.lambda,
            n_bins:                 config.n_bins,
            k_folds:                config.k_folds,
            calibrate:              false,
            fold_strategy:          config.fold_strategy.clone(),
            metadata:               None,
            objective:              config.objective.clone(),
            sampler:                config.sampler.clone(),
            splitter:               config.splitter.clone(),
            early_stopping_rounds:  config.early_stopping_rounds,
            col_subsample:          config.col_subsample,
            platt_scale:            false,
            raw_isotonic:           false,
            tail_weight:            config.tail_weight.clone(),
            seed:                   config.seed.wrapping_add(fold as u64 * 0x9e3779b9u64),
        };

        let fold_model = train(&fold_dataset, &fold_config);

        // Score held-out rows using the fold model.
        // Dataset is column-major; assemble each row on the fly.
        for &i in &val_rows {
            let row: Vec<f32> = dataset.features.iter().map(|col| col[i]).collect();
            oof_scores[i] = fold_model.predict_raw(&row);
        }
    }

    oof_scores
}

/// Compute per-row weights for top-tail gradient focusing.
///
/// Finds the raw-score threshold τ for the top `cfg.top_rate` fraction, then
/// assigns weights based on each row's class and position relative to τ:
///
/// - Positive below τ (missed positive):  `weight_missed_pos`
/// - Negative above τ (false positive):   `weight_false_pos`
/// - Either class within `boundary_width` of τ: `weight_boundary`
/// - Negative far below τ (easy negative): `weight_easy_neg`
/// - Positive far above τ (easy positive): `weight_easy_pos`
fn tail_weights(labels: &[f32], scores: &[f32], cfg: &TailWeightConfig) -> Vec<f32> {
    let n = scores.len();
    let mut sorted = scores.to_vec();
    sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let threshold_idx = ((1.0 - cfg.top_rate) * n as f64) as usize;
    let tau = sorted[threshold_idx.min(n.saturating_sub(1))];

    labels
        .iter()
        .zip(scores.iter())
        .map(|(&y, &s)| {
            let is_pos = y > 0.5;
            let in_top = s >= tau;
            let near_boundary = (s - tau).abs() < cfg.boundary_width;

            if is_pos && !in_top {
                cfg.weight_missed_pos
            } else if !is_pos && in_top {
                cfg.weight_false_pos
            } else if near_boundary {
                cfg.weight_boundary
            } else if !is_pos {
                cfg.weight_easy_neg
            } else {
                cfg.weight_easy_pos
            }
        })
        .collect()
}

fn mean_log_loss(labels: &[f32], preds: &[f32]) -> f32 {
    let eps = 1e-7_f32;
    let n = labels.len() as f32;
    labels
        .iter()
        .zip(preds)
        .map(|(&y, &x)| {
            let p = sigmoid(x).clamp(eps, 1.0 - eps);
            -(y * p.ln() + (1.0 - y) * (1.0 - p).ln())
        })
        .sum::<f32>()
        / n
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 { 1.0 / (1.0 + (-x).exp()) } else { let e = x.exp(); e / (1.0 + e) }
}
