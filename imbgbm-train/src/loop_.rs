use imbgbm_calib::{assign_folds, calibrate_tree};
use imbgbm_core::{BinnedDataset, BoostingState, Dataset, RowMetadata};
use imbgbm_infer::{route_all, Model};

use crate::{builder::grow_tree, config::Config};

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

    for round in 0..config.n_rounds {
        // ── 1. Gradients ────────────────────────────────────────────────────
        let (g, h) = config.objective.grad_hess(&binned.labels, &state.predictions);
        state.gradients = g;
        state.hessians = h;

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
        );

        // ── 4. OOF calibration ───────────────────────────────────────────────
        if config.calibrate {
            if let Some(ref fold_assign) = folds {
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

    Model::new(trees, config.learning_rate, init_score)
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
