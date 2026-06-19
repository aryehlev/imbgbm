use clap::{Parser, Subcommand};
use std::sync::Arc;

use imbgbm_calib::FoldStrategy;
use imbgbm_core::Dataset;
use imbgbm_infer::Model;
use imbgbm_loss::{BCELoss, FocalLoss};
use imbgbm_sample::{AdaptiveSampler, GossSampler, Sampler, UniformSampler};
use imbgbm_split::{PositiveMassSplitter, Splitter, StandardSplitter, VarianceAwareSplitter};
use imbgbm_train::{train, Config, TailWeightConfig};

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
        /// Loss function: bce | focal
        #[arg(long, default_value = "bce")]
        loss: String,
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
        /// Preserves additive boosting structure.
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
        /// Fraction of features randomly selected per tree (column subsampling).
        /// 0.8 adds diversity across trees; 1.0 uses all features.
        #[arg(long, default_value_t = 0.8)]
        col_subsample: f32,
        /// L2 regularisation strength for leaf values. Lower = more aggressive splits.
        #[arg(long, default_value_t = 1.0)]
        lambda: f32,
        /// Minimum number of training samples required in each leaf node.
        #[arg(long, default_value_t = 20)]
        min_samples_leaf: usize,
        /// Minimum sum of hessians (effective sample count) for a child to be valid.
        #[arg(long, default_value_t = 1.0)]
        min_child_weight: f32,
        /// Fit OOF isotonic calibration on raw boosted scores (requires --calibrate).
        /// Trains K fold models and fits PAV isotonic regression on held-out scores.
        /// Mutually exclusive with --platt; isotonic takes precedence when both set.
        #[arg(long)]
        raw_isotonic: bool,
        /// Splitter: standard | variance-aware | positive-mass
        #[arg(long, default_value = "standard")]
        splitter: String,
        /// Purity bonus coefficient for variance-aware splitter.
        #[arg(long, default_value_t = 0.1_f32)]
        purity_lambda: f32,
        /// Alpha coefficient for positive-mass splitter (positive-mass gain weight).
        #[arg(long, default_value_t = 1.0_f32)]
        pm_alpha: f32,
        /// Minimum positives per leaf for positive-mass splitter.
        #[arg(long, default_value_t = 20_f32)]
        pm_min_pos_leaf: f32,
        /// Enable top-K tail gradient weighting (focuses learning on top-K boundary).
        #[arg(long)]
        tail_weight: bool,
        /// Top fraction of rows to target for tail weighting (e.g. 0.01 = top 1%).
        #[arg(long, default_value_t = 0.01_f64)]
        tail_top_rate: f64,
        /// Round at which tail weighting activates (0 = from round 1).
        #[arg(long, default_value_t = 0_usize)]
        tail_start_round: usize,
        /// Comma-separated 0-based column indices for integer-encoded categorical features.
        /// Example: "6,7,8" for the last three columns.  These receive OOF Bayesian
        /// target encoding so no leakage occurs during training.
        #[arg(long, default_value = "")]
        cat_features: String,
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
        /// Use OOF isotonic calibrated probabilities (requires --raw-isotonic at train time).
        #[arg(long)]
        raw_isotonic: bool,
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
            input, output, loss, n_rounds, learning_rate, max_depth,
            subsample, sampler, gamma, alpha, calibrate, platt, fold_strategy, seed,
            early_stopping_rounds, col_subsample, lambda, min_samples_leaf, min_child_weight,
            raw_isotonic, splitter, purity_lambda, pm_alpha, pm_min_pos_leaf,
            tail_weight, tail_top_rate, tail_start_round, cat_features,
        } => {
            let (features, labels) = load_csv_with_label(&input);
            let rows: Vec<&[f32]> = features.iter().map(|r| r.as_slice()).collect();
            let dataset = Dataset::from_rows(&rows, labels);

            let objective: Arc<dyn imbgbm_loss::Objective> = match loss.as_str() {
                "focal" => Arc::new(FocalLoss::new(gamma, alpha)),
                _       => Arc::new(BCELoss),
            };

            let fs = match fold_strategy.as_str() {
                "temporal" => {
                    eprintln!("error: --fold-strategy temporal requires timestamp metadata which the CLI does not yet parse.");
                    std::process::exit(1);
                }
                _ => FoldStrategy::Random { seed },
            };

            let splitter_arc: Arc<dyn Splitter> = match splitter.as_str() {
                "variance-aware" => Arc::new(VarianceAwareSplitter::new(purity_lambda, 5)),
                "positive-mass"  => Arc::new(
                    PositiveMassSplitter::new(pm_alpha, pm_min_pos_leaf)
                ),
                _ => Arc::new(StandardSplitter),
            };

            let cat_feature_indices: Vec<usize> = if cat_features.is_empty() {
                vec![]
            } else {
                cat_features.split(',')
                    .map(|s| s.trim().parse::<usize>().expect("invalid --cat-features index"))
                    .collect()
            };

            let tail_weight_cfg = if tail_weight {
                let mut tw = TailWeightConfig::top1pct();
                tw.top_rate    = tail_top_rate;
                tw.start_round = tail_start_round;
                Some(tw)
            } else {
                None
            };

            let config = Config {
                n_rounds,
                learning_rate,
                max_depth,
                min_child_weight,
                min_samples_leaf,
                lambda,
                n_bins: 255,
                col_subsample,
                k_folds: 5,
                calibrate,
                fold_strategy: fs,
                metadata: None,
                objective,
                sampler: build_sampler(&sampler, subsample, seed),
                splitter: splitter_arc,
                early_stopping_rounds: if early_stopping_rounds == 0 { None } else { Some(early_stopping_rounds) },
                platt_scale: platt,
                raw_isotonic,
                tail_weight: tail_weight_cfg,
                cat_features: cat_feature_indices,
                seed,
            };

            eprintln!(
                "Training {} rounds on {} examples × {} features …",
                n_rounds, dataset.n_rows, dataset.n_cols
            );
            let model = train(&dataset, &config);
            eprintln!("Done — {} trees.", model.trees.len());

            let json = model.to_json().expect("serialisation failed");
            std::fs::write(&output, &json).expect("could not write model file");
            eprintln!("Model → {output}");
        }

        Commands::Predict { input, model, calibrated, platt, pu, raw_isotonic } => {
            let rows = load_csv_features(&input);
            let json = std::fs::read_to_string(&model).expect("could not read model");
            let m = Model::from_json(&json).expect("model parse error");

            for row in &rows {
                let p = if pu {
                    m.predict_proba_pu(row)
                } else if raw_isotonic {
                    m.predict_proba_raw_iso(row)
                } else if platt {
                    m.predict_proba_platt(row)
                } else if calibrated {
                    m.predict_proba_calibrated(row)
                } else {
                    m.predict_proba_raw(row)
                };
                println!("{p:.6}");
            }
        }

        Commands::Query { input: _, model: _, budget: _ } => {
            eprintln!("Query command not yet implemented.");
            std::process::exit(1);
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
