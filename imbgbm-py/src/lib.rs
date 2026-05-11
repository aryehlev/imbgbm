use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use std::sync::Arc;

use imbgbm_core::Dataset;
use imbgbm_infer::Model;
use imbgbm_loss::{BCELoss, FocalLoss};
use imbgbm_sample::UniformSampler;
use imbgbm_split::StandardSplitter;
use imbgbm_train::{train, Config};

// ── Python-visible model wrapper ─────────────────────────────────────────────

/// imbgbm gradient boosted model for imbalanced binary classification.
#[pyclass(name = "GradientBoostedTree")]
struct PyModel {
    inner: Model,
}

#[pymethods]
impl PyModel {
    /// Predict probabilities for a 2-D list of features (list-of-rows).
    ///
    /// Returns a list of f32 probabilities in [0, 1].
    /// Uses OOF-calibrated probabilities when available, otherwise raw sigmoid.
    fn predict_proba(&self, x: Vec<Vec<f32>>) -> PyResult<Vec<f32>> {
        Ok(x.iter()
            .map(|row| {
                if self.inner.trees.iter().any(|t| !t.leaf_probabilities.is_empty()
                    && t.leaf_probabilities.iter().any(|&p| p > 0.0)) {
                    self.inner.predict_proba_calibrated(row)
                } else {
                    self.inner.predict_proba_raw(row)
                }
            })
            .collect())
    }

    /// Predict raw log-odds scores (before sigmoid).
    fn predict_raw(&self, x: Vec<Vec<f32>>) -> PyResult<Vec<f32>> {
        Ok(x.iter().map(|row| self.inner.predict_raw(row)).collect())
    }

    /// Serialize the model to a JSON string.
    fn to_json(&self) -> PyResult<String> {
        self.inner.to_json().map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Number of trees in the ensemble.
    fn n_trees(&self) -> usize {
        self.inner.trees.len()
    }
}

// ── Training entry point ─────────────────────────────────────────────────────

/// Train a gradient boosted tree model.
///
/// Args:
///     x: list of rows, each row is a list of floats.
///     y: list of binary labels (0.0 or 1.0).
///     loss: "bce" or "focal".
///     n_rounds: number of boosting rounds.
///     learning_rate: step size.
///     max_depth: maximum tree depth.
///     subsample: row sub-sampling fraction.
///     gamma: focal loss gamma parameter (ignored for bce).
///     alpha: focal loss alpha parameter (ignored for bce).
///     calibrate: enable OOF leaf calibration.
///     seed: random seed.
#[pyfunction]
#[pyo3(signature = (
    x, y,
    loss = "bce",
    n_rounds = 100,
    learning_rate = 0.1,
    max_depth = 6,
    subsample = 0.8,
    gamma = 2.0,
    alpha = 0.25,
    calibrate = false,
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
    gamma: f32,
    alpha: f32,
    calibrate: bool,
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
        "bce" => Arc::new(BCELoss),
        "focal" => Arc::new(FocalLoss::new(gamma, alpha)),
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown loss '{}': choose 'bce' or 'focal'",
                other
            )))
        }
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
        early_stopping_rounds: None,
        seed,
    };

    let model = train(&dataset, &config);
    Ok(PyModel { inner: model })
}

/// Load a model from a JSON string produced by `GradientBoostedTree.to_json()`.
#[pyfunction]
fn load_model(json: &str) -> PyResult<PyModel> {
    Model::from_json(json)
        .map(|m| PyModel { inner: m })
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

// ── Module registration ───────────────────────────────────────────────────────

#[pymodule]
fn imbgbm(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyModel>()?;
    m.add_function(wrap_pyfunction!(fit, m)?)?;
    m.add_function(wrap_pyfunction!(load_model, m)?)?;
    Ok(())
}
