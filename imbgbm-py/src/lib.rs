use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use std::sync::Arc;

use imbgbm_calib::FoldStrategy;
use imbgbm_core::Dataset;
use imbgbm_infer::Model;
use imbgbm_loss::{BCELoss, FocalLoss};
use imbgbm_sample::{AdaptiveSampler, UniformSampler};
use imbgbm_split::StandardSplitter;
use imbgbm_train::{train, Config};

// ── Python-visible model ──────────────────────────────────────────────────────

#[pyclass(name = "GradientBoostedTree")]
struct PyModel {
    inner: Model,
}

#[pymethods]
impl PyModel {
    /// Predict calibrated probabilities.  Uses isotonic calibration when
    /// the model was trained with raw_isotonic=True (the default), otherwise
    /// falls back to per-leaf OOF averaging or raw sigmoid.
    fn predict_proba(&self, x: Vec<Vec<f32>>) -> PyResult<Vec<f32>> {
        Ok(x.iter()
            .map(|row| {
                if !self.inner.raw_iso_cal.is_empty() {
                    self.inner.predict_proba_raw_iso(row)
                } else if self.inner.trees.iter().any(|t| t.leaf_probabilities.iter().any(|&p| p > 0.0)) {
                    self.inner.predict_proba_calibrated(row)
                } else {
                    self.inner.predict_proba_raw(row)
                }
            })
            .collect())
    }

    fn predict_raw(&self, x: Vec<Vec<f32>>) -> PyResult<Vec<f32>> {
        Ok(x.iter().map(|row| self.inner.predict_raw(row)).collect())
    }

    fn to_json(&self) -> PyResult<String> {
        self.inner.to_json().map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn n_trees(&self) -> usize {
        self.inner.trees.len()
    }
}

// ── Training entry point ──────────────────────────────────────────────────────

#[pyfunction]
#[pyo3(signature = (
    x, y,
    loss = "bce",
    n_rounds = 300,
    learning_rate = 0.05,
    max_depth = 6,
    subsample = 0.8,
    col_subsample = 0.8,
    gamma = 2.0,
    alpha = 0.25,
    calibrate = true,
    raw_isotonic = false,
    fold_strategy = "random",
    seed = 42
))]
fn fit(
    x: Vec<Vec<f32>>,
    y: Vec<f32>,
    loss: &str,
    n_rounds: usize,
    learning_rate: f32,
    max_depth: usize,
    subsample: f32,
    col_subsample: f32,
    gamma: f32,
    alpha: f32,
    calibrate: bool,
    raw_isotonic: bool,
    fold_strategy: &str,
    seed: u64,
) -> PyResult<PyModel> {
    if x.is_empty() {
        return Err(PyValueError::new_err("empty training set"));
    }
    if x.len() != y.len() {
        return Err(PyValueError::new_err("x and y length mismatch"));
    }

    let rows: Vec<&[f32]> = x.iter().map(|r| r.as_slice()).collect();
    let dataset = Dataset::from_rows(&rows, y);

    let objective: Arc<dyn imbgbm_loss::Objective> = match loss {
        "bce"   => Arc::new(BCELoss),
        "focal" => Arc::new(FocalLoss::new(gamma, alpha)),
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown loss '{other}': choose 'bce' or 'focal'"
            )))
        }
    };

    let fs = match fold_strategy {
        "temporal" => FoldStrategy::Temporal,
        _ => FoldStrategy::Random { seed },
    };

    // Default to adaptive sampler (better for imbalanced data).
    // If subsample == 1.0 the user explicitly wants uniform sampling.
    let sampler: Arc<dyn imbgbm_sample::Sampler> = if subsample >= 1.0 {
        Arc::new(UniformSampler::new(1.0, seed))
    } else {
        Arc::new(AdaptiveSampler::new(0.2, subsample.max(0.05), Some(0.5), 0.0, seed))
    };

    let config = Config {
        n_rounds,
        learning_rate,
        max_depth,
        min_child_weight: 1.0,
        min_samples_leaf: 20,
        lambda: 1.0,
        n_bins: 255,
        col_subsample,
        k_folds: 5,
        calibrate,
        fold_strategy: fs,
        metadata: None,
        objective,
        sampler,
        splitter: Arc::new(StandardSplitter),
        early_stopping_rounds: None,
        platt_scale: false,
        raw_isotonic,
        tail_weight: None,
        seed,
    };

    Ok(PyModel { inner: train(&dataset, &config) })
}

#[pyfunction]
fn load_model(json: &str) -> PyResult<PyModel> {
    Model::from_json(json)
        .map(|m| PyModel { inner: m })
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pymodule]
fn imbgbm(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyModel>()?;
    m.add_function(wrap_pyfunction!(fit, m)?)?;
    m.add_function(wrap_pyfunction!(load_model, m)?)?;
    Ok(())
}
