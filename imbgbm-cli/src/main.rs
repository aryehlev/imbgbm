use clap::{Parser, Subcommand};
use std::sync::Arc;

use imbgbm_core::Dataset;
use imbgbm_infer::Model;
use imbgbm_loss::{BCELoss, FocalLoss};
use imbgbm_sample::UniformSampler;
use imbgbm_split::StandardSplitter;
use imbgbm_train::{train, Config};

// ── CLI definition ────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(name = "imbgbm", about = "Imbalance-aware gradient boosted trees")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Train a model on a CSV file and save it to JSON.
    Train {
        /// Path to training CSV (last column = label, no header).
        #[arg(short, long)]
        input: String,
        /// Where to write the trained model (JSON).
        #[arg(short, long)]
        output: String,
        /// Loss function: bce | focal.
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
        /// Focal loss gamma (ignored for bce).
        #[arg(long, default_value_t = 2.0)]
        gamma: f32,
        /// Focal loss alpha (ignored for bce).
        #[arg(long, default_value_t = 0.25)]
        alpha: f32,
        /// Enable OOF leaf calibration.
        #[arg(long)]
        calibrate: bool,
        #[arg(long, default_value_t = 42)]
        seed: u64,
    },
    /// Apply a saved model to a CSV and print probabilities.
    Predict {
        /// Path to feature CSV (no label column, no header).
        #[arg(short, long)]
        input: String,
        /// Path to model JSON produced by `train`.
        #[arg(short, long)]
        model: String,
        /// Use calibrated probabilities (requires --calibrate at train time).
        #[arg(long)]
        calibrated: bool,
    },
}

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Commands::Train {
            input,
            output,
            loss,
            n_rounds,
            learning_rate,
            max_depth,
            subsample,
            gamma,
            alpha,
            calibrate,
            seed,
        } => {
            let (features, labels) = load_csv_with_label(&input);
            let rows: Vec<&[f32]> = features.iter().map(|r| r.as_slice()).collect();
            let dataset = Dataset::from_rows(&rows, labels);

            let objective: Arc<dyn imbgbm_loss::Objective> = match loss.as_str() {
                "focal" => Arc::new(FocalLoss::new(gamma, alpha)),
                _ => Arc::new(BCELoss),
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
                objective,
                sampler: Arc::new(UniformSampler::new(subsample, seed)),
                splitter: Arc::new(StandardSplitter),
                early_stopping_rounds: Some(10),
                seed,
            };

            eprintln!(
                "Training {} rounds on {} examples, {} features …",
                n_rounds,
                dataset.n_rows,
                dataset.n_cols
            );
            let model = train(&dataset, &config);
            eprintln!("Trained {} trees.", model.trees.len());

            let json = model.to_json().expect("serialisation failed");
            std::fs::write(&output, &json).expect("could not write model file");
            eprintln!("Model written to {output}");
        }

        Commands::Predict { input, model, calibrated } => {
            let rows = load_csv_features(&input);
            let json = std::fs::read_to_string(&model).expect("could not read model file");
            let m = Model::from_json(&json).expect("model parse error");

            for row in &rows {
                let p = if calibrated {
                    m.predict_proba_calibrated(row)
                } else {
                    m.predict_proba_raw(row)
                };
                println!("{p:.6}");
            }
        }
    }
}

// ── CSV helpers ───────────────────────────────────────────────────────────────

/// Load CSV where the LAST column is the binary label.
fn load_csv_with_label(path: &str) -> (Vec<Vec<f32>>, Vec<f32>) {
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_path(path)
        .unwrap_or_else(|e| panic!("cannot open {path}: {e}"));

    let mut features: Vec<Vec<f32>> = Vec::new();
    let mut labels: Vec<f32> = Vec::new();

    for result in rdr.records() {
        let rec = result.expect("CSV parse error");
        let vals: Vec<f32> = rec
            .iter()
            .map(|s| s.trim().parse::<f32>().expect("non-numeric value in CSV"))
            .collect();
        if vals.is_empty() {
            continue;
        }
        labels.push(*vals.last().unwrap());
        features.push(vals[..vals.len() - 1].to_vec());
    }
    (features, labels)
}

/// Load CSV of features only (no label column).
fn load_csv_features(path: &str) -> Vec<Vec<f32>> {
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_path(path)
        .unwrap_or_else(|e| panic!("cannot open {path}: {e}"));

    rdr.records()
        .map(|r| {
            r.expect("CSV parse error")
                .iter()
                .map(|s| s.trim().parse::<f32>().expect("non-numeric"))
                .collect()
        })
        .collect()
}
