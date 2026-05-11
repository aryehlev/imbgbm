use clap::{Parser, Subcommand};
use std::sync::Arc;

use imbgbm_calib::FoldStrategy;
use imbgbm_core::Dataset;
use imbgbm_infer::Model;
use imbgbm_calib::{estimate_pu_label_rate, PuPriorEstimator};
use imbgbm_loss::{BCELoss, DensityRatioLoss, FocalLoss, PULoss};
use imbgbm_sample::{AdaptiveSampler, GossSampler, PuGossSampler, Sampler, UniformSampler};
use imbgbm_split::{PuAwareSplitter, StandardSplitter, VarianceAwareSplitter};
use imbgbm_train::{
    train, train_with_seed_expansion, Config, SeedExpansionConfig,
};
use imbgbm_infer::rank_query_candidates;

#[derive(Parser)]
#[command(name = "imbgbm", about = "Imbalance-aware gradient boosted trees")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Train a model on a CSV file (last column = binary label, no header).
    Train {
        #[arg(short, long)]
        input: String,
        #[arg(short, long)]
        output: String,
        /// Loss function: bce | focal | pu | density_ratio
        #[arg(long, default_value = "bce")]
        loss: String,
        /// PU class prior (for --loss pu and --loss density_ratio)
        #[arg(long, default_value_t = 0.05)]
        pu_prior: f32,
        /// Splitter: standard | variance | pu
        #[arg(long, default_value = "standard")]
        splitter: String,
        #[arg(long, default_value_t = 100)]
        n_rounds: usize,
        #[arg(long, default_value_t = 0.1)]
        learning_rate: f32,
        #[arg(long, default_value_t = 6)]
        max_depth: usize,
        #[arg(long, default_value_t = 0.8)]
        subsample: f32,
        /// Sampler: uniform | goss | adaptive
        #[arg(long, default_value = "uniform")]
        sampler: String,
        #[arg(long, default_value_t = 2.0)]
        gamma: f32,
        #[arg(long, default_value_t = 0.25)]
        alpha: f32,
        /// Enable OOF leaf calibration
        #[arg(long)]
        calibrate: bool,
        /// Fit Platt scaling on OOF boosted scores (requires --calibrate).
        /// Preserves additive boosting structure; superior to per-leaf averaging
        /// for fixing focal-loss miscalibration.
        #[arg(long)]
        platt: bool,
        /// Fold strategy for OOF calibration: random | temporal
        /// (temporal requires a timestamp column before the label)
        #[arg(long, default_value = "random")]
        fold_strategy: String,
        #[arg(long, default_value_t = 42)]
        seed: u64,
        /// Early-stopping patience in rounds (0 = disabled)
        #[arg(long, default_value_t = 20)]
        early_stopping_rounds: usize,
        /// Run biased-positive seed expansion: pilot → promote → refine.
        /// Expansion fraction is the top-quantile of unlabeled rows to promote.
        #[arg(long, default_value_t = 0.0)]
        seed_expansion: f32,
        /// Estimate the Elkan-Noto PU label rate from OOF positives and store
        /// it on the model. Requires --calibrate. At inference, use
        /// `predict --pu` to apply the correction.
        #[arg(long)]
        estimate_pu_rate: bool,
        /// Fraction of features sampled per tree (column subsampling).
        /// 1.0 = all features; 0.8 is a good starting point.
        #[arg(long, default_value_t = 1.0)]
        col_subsample: f32,
    },
    /// Predict probabilities for a feature CSV (no label column, no header).
    Predict {
        #[arg(short, long)]
        input: String,
        #[arg(short, long)]
        model: String,
        /// Use OOF-leaf-calibrated probabilities (requires --calibrate at train time)
        #[arg(long)]
        calibrated: bool,
        /// Use Platt-scaled probabilities (requires --platt at train time)
        #[arg(long)]
        platt: bool,
        /// Apply Elkan-Noto PU correction: P(y=1|x) = p / c (requires
        /// --estimate-pu-rate at train time).
        #[arg(long)]
        pu: bool,
        /// Apply leaf-confidence shrinkage with the given alpha (0 = off).
        /// Tiny leaves get pulled toward zero; reduces overbidding noise.
        #[arg(long, default_value_t = 0.0)]
        shrink_alpha: f32,
    },
    /// Rank an unlabeled pool by information gain for active labeling.
    /// Output: one `row_index,score` per line, descending by score.
    Query {
        #[arg(short, long)]
        input: String,
        #[arg(short, long)]
        model: String,
        /// Number of rows to output (0 = all).
        #[arg(long, default_value_t = 0)]
        budget: usize,
    },
}

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Commands::Train {
            input, output, loss, pu_prior, splitter, n_rounds, learning_rate, max_depth,
            subsample, sampler, gamma, alpha, calibrate, platt, fold_strategy, seed,
            early_stopping_rounds, seed_expansion, estimate_pu_rate, col_subsample,
        } => {
            let (features, labels) = load_csv_with_label(&input);
            let rows: Vec<&[f32]> = features.iter().map(|r| r.as_slice()).collect();
            let dataset = Dataset::from_rows(&rows, labels);

            let objective: Arc<dyn imbgbm_loss::Objective> = match loss.as_str() {
                "focal"         => Arc::new(FocalLoss::new(gamma, alpha)),
                "pu"            => Arc::new(PULoss::new(pu_prior)),
                "density_ratio" => Arc::new(DensityRatioLoss::new(pu_prior.clamp(0.01, 0.99))),
                _               => Arc::new(BCELoss),
            };

            let split_impl: Arc<dyn imbgbm_split::Splitter> = match splitter.as_str() {
                "variance" => Arc::new(VarianceAwareSplitter::new(0.1, 5)),
                "pu"       => Arc::new(PuAwareSplitter::new(0.1, 5.0, 20.0, 2)),
                _          => Arc::new(StandardSplitter),
            };

            let fs = match fold_strategy.as_str() {
                "temporal" => FoldStrategy::Temporal,
                _          => FoldStrategy::Random { seed },
            };

            let config = Config {
                n_rounds,
                learning_rate,
                max_depth,
                min_child_weight: 1.0,
                min_samples_leaf: 20,
                lambda: 1.0,
                n_bins: 255,
                k_folds: 5,
                calibrate,
                fold_strategy: fs,
                metadata: None,
                objective,
                sampler: build_sampler(&sampler, subsample, seed),
                splitter: split_impl,
                early_stopping_rounds: if early_stopping_rounds == 0 { None } else { Some(early_stopping_rounds) },
                col_subsample: col_subsample.clamp(0.01, 1.0),
                platt_scale: platt,
                seed,
            };

            eprintln!(
                "Training {} rounds on {} examples × {} features …",
                n_rounds, dataset.n_rows, dataset.n_cols
            );
            let mut model = if seed_expansion > 0.0 {
                let ex_cfg = SeedExpansionConfig {
                    expansion_fraction: seed_expansion,
                    ..SeedExpansionConfig::default()
                };
                let res = train_with_seed_expansion(&dataset, &config, &ex_cfg);
                eprintln!(
                    "Seed expansion: promoted {} unlabeled rows (mean pilot score {:.3}).",
                    res.n_promoted, res.mean_promoted_score
                );
                res.model
            } else {
                train(&dataset, &config)
            };
            eprintln!("Done — {} trees.", model.trees.len());

            if estimate_pu_rate {
                // Estimate c = P(s=1 | y=1) using in-sample predictions on the
                // labeled positives. For best accuracy this should be OOF; we
                // approximate with the model's own predictions on its training
                // labeled positives, then store c on the model.
                let mut scores = Vec::with_capacity(dataset.n_rows);
                for i in 0..dataset.n_rows {
                    let row = dataset.row_copy(i);
                    scores.push(model.predict_proba_raw(&row));
                }
                let c = estimate_pu_label_rate(
                    &scores, &dataset.labels, PuPriorEstimator::MedianOverPositives,
                );
                eprintln!("Estimated PU label rate c = {c:.4}");
                model = model.with_pu_label_rate(c);
            }

            let json = model.to_json().expect("serialisation failed");
            std::fs::write(&output, &json).expect("could not write model file");
            eprintln!("Model → {output}");
        }

        Commands::Predict { input, model, calibrated, platt, pu, shrink_alpha } => {
            let rows = load_csv_features(&input);
            let json = std::fs::read_to_string(&model).expect("could not read model");
            let m = Model::from_json(&json).expect("model parse error");

            for row in &rows {
                let p = if pu {
                    m.predict_proba_pu(row)
                } else if platt {
                    m.predict_proba_platt(row)
                } else if calibrated {
                    m.predict_proba_calibrated(row)
                } else if shrink_alpha > 0.0 {
                    m.predict_proba_shrunk(row, shrink_alpha)
                } else {
                    m.predict_proba_raw(row)
                };
                println!("{p:.6}");
            }
        }

        Commands::Query { input, model, budget } => {
            let rows = load_csv_features(&input);
            let json = std::fs::read_to_string(&model).expect("could not read model");
            let m = Model::from_json(&json).expect("model parse error");
            let row_refs: Vec<&[f32]> = rows.iter().map(|r| r.as_slice()).collect();
            let scored = rank_query_candidates(&m, &row_refs);
            let limit = if budget == 0 { scored.len() } else { budget.min(scored.len()) };
            for (idx, score) in scored.into_iter().take(limit) {
                println!("{idx},{score:.6}");
            }
        }
    }
}

fn build_sampler(name: &str, subsample: f32, seed: u64) -> Arc<dyn Sampler> {
    match name {
        "goss"     => Arc::new(GossSampler::new(0.2, subsample.max(0.1), seed)),
        "adaptive" => Arc::new(AdaptiveSampler::new(
            0.2,                      // keep top-20% by |gradient|
            subsample.max(0.1),       // sample tail at given rate
            Some(0.5),                // re-balance toward 50/50 classes
            0.1,                      // light uncertainty boost
            seed,
        )),
        "pu_goss" => Arc::new(
            PuGossSampler::new(
                0.20,                 // top |gradient|
                0.10,                 // suspicious unlabeled (high score)
                0.10,                 // boundary uncertainty
                subsample.max(0.05),  // reliable negatives
                seed,
            )
            .with_hard_neg_boost(1.5),
        ),
        _ => Arc::new(UniformSampler::new(subsample, seed)),
    }
}

fn load_csv_with_label(path: &str) -> (Vec<Vec<f32>>, Vec<f32>) {
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_path(path)
        .unwrap_or_else(|e| panic!("cannot open {path}: {e}"));
    let mut features = Vec::new();
    let mut labels = Vec::new();
    for rec in rdr.records() {
        let rec = rec.expect("CSV parse error");
        let vals: Vec<f32> = rec.iter()
            .map(|s| s.trim().parse::<f32>().expect("non-numeric"))
            .collect();
        if vals.is_empty() { continue; }
        labels.push(*vals.last().unwrap());
        features.push(vals[..vals.len() - 1].to_vec());
    }
    (features, labels)
}

fn load_csv_features(path: &str) -> Vec<Vec<f32>> {
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_path(path)
        .unwrap_or_else(|e| panic!("cannot open {path}: {e}"));
    rdr.records()
        .map(|r| {
            r.expect("CSV parse error").iter()
                .map(|s| s.trim().parse::<f32>().expect("non-numeric"))
                .collect()
        })
        .collect()
}
