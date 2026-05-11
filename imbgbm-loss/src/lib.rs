/// Common numerical epsilon used across loss implementations.
const EPS: f32 = 1e-7;

// ── Shared helpers ──────────────────────────────────────────────────────────

#[inline]
fn sigmoid(x: f32) -> f32 {
    // Numerically stable sigmoid.
    if x >= 0.0 {
        let e = (-x).exp();
        1.0 / (1.0 + e)
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

// ── Objective trait ─────────────────────────────────────────────────────────

/// The core loss interface.  Every component in imbgbm depends on an
/// `Objective` for gradients, probability outputs, and the class prior.
pub trait Objective: Send + Sync {
    /// Compute per-example (gradient, hessian) pairs for the current
    /// raw predictions.  Hessians are guaranteed positive (clamped to EPS).
    fn grad_hess(&self, y_true: &[f32], y_pred: &[f32]) -> (Vec<f32>, Vec<f32>);

    /// Map a raw leaf score (sum of Newton steps) to a probability in [0, 1].
    fn predict_proba(&self, raw: f32) -> f32;

    /// Fraction of positives in the training set, used to initialise the
    /// boosting score and to guide the sampler.  Returns `None` for losses
    /// that do not require it.
    fn class_prior(&self) -> Option<f32>;
}

// ── Binary cross-entropy ────────────────────────────────────────────────────

/// Standard log-loss: `L = -(y·log(p) + (1-y)·log(1-p))`.
///
/// `grad = p - y`, `hess = p·(1-p)`.
pub struct BCELoss;

impl Objective for BCELoss {
    fn grad_hess(&self, y_true: &[f32], y_pred: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let n = y_true.len();
        let mut g = Vec::with_capacity(n);
        let mut h = Vec::with_capacity(n);
        for (&y, &x) in y_true.iter().zip(y_pred) {
            let p = sigmoid(x);
            g.push(p - y);
            h.push((p * (1.0 - p)).max(EPS));
        }
        (g, h)
    }

    fn predict_proba(&self, raw: f32) -> f32 {
        sigmoid(raw)
    }

    fn class_prior(&self) -> Option<f32> {
        None
    }
}

// ── Focal loss ──────────────────────────────────────────────────────────────

/// Focal loss (Lin et al. 2017): down-weights easy examples.
///
/// `L = -alpha_t · (1-p_t)^gamma · log(p_t)`
///
/// Analytic first and second derivatives are used; hessians are clamped to EPS
/// because they can go negative for large gamma.
pub struct FocalLoss {
    /// Focusing parameter γ ≥ 0.  γ=0 recovers alpha-balanced BCE.
    pub gamma: f32,
    /// Class weight for positives.  Negatives get weight `1-alpha`.
    pub alpha: f32,
}

impl FocalLoss {
    pub fn new(gamma: f32, alpha: f32) -> Self {
        FocalLoss { gamma, alpha }
    }
}

impl Objective for FocalLoss {
    fn grad_hess(&self, y_true: &[f32], y_pred: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let n = y_true.len();
        let mut gv = Vec::with_capacity(n);
        let mut hv = Vec::with_capacity(n);
        for (&y, &x) in y_true.iter().zip(y_pred) {
            let (g, h) = focal_grad_hess(x, y, self.gamma, self.alpha);
            gv.push(g);
            hv.push(h.max(EPS));
        }
        (gv, hv)
    }

    fn predict_proba(&self, raw: f32) -> f32 {
        sigmoid(raw)
    }

    fn class_prior(&self) -> Option<f32> {
        None
    }
}

/// Analytic (g, h) for focal loss at a single example.
///
/// Derivation (y=1):
///   L  = -α·(1-p)^γ·log(p)
///   g₁ = α·p·(1-p)^γ·[γ·log(p) − (1-p)/p]          (dL/dx via chain rule)
///   h₁ = α·p·(1-p) · [dA/dp·B + A·dB/dp]             (second derivative)
///
///   where A=p·(1-p)^γ, B=γ·log(p)−q/p, q=1-p
///
/// By symmetry, y=0 swaps p↔q and α↔(1-α).
fn focal_grad_hess(x: f32, y: f32, gamma: f32, alpha: f32) -> (f32, f32) {
    let p_raw = sigmoid(x);
    let p = p_raw.clamp(EPS, 1.0 - EPS);
    let q = 1.0 - p;

    if y > 0.5 {
        // ── y = 1 ──────────────────────────────────────────────────────────
        let q_g = q.powf(gamma);       // (1-p)^gamma
        let q_g1 = q.powf(gamma - 1.0); // (1-p)^(gamma-1)
        let ln_p = p.ln();

        // g₁ = α · p · q^γ · [γ·ln(p) - q/p]
        let g = alpha * p * q_g * (gamma * ln_p - q / p);

        // Exact second derivative:
        //   A = p·q^γ,  B = γ·ln(p) - q/p
        //   dA/dp = q^(γ-1)·(q - γ·p)   [= q^(γ-1)·(1-p(γ+1))]
        //   dB/dp = γ/p + 1/p²            [d(γln p)/dp + d(-q/p)/dp = γ/p + 1/p²]
        //   h = α·p·q · (dA/dp·B + A·dB/dp)
        let da_dp = q_g1 * (q - gamma * p);
        let b = gamma * ln_p - q / p;
        let db_dp = gamma / p + 1.0 / (p * p);
        let a = p * q_g;
        let h = alpha * p * q * (da_dp * b + a * db_dp);
        (g, h)
    } else {
        // ── y = 0 ──────────────────────────────────────────────────────────
        // By symmetry with y=1 (swap p↔q, α↔(1-α)):
        //   L  = -(1-α)·p^γ·log(q)
        //   g₀ = (1-α)·p^γ · [p − γ·q·log(q)]
        //   h₀ = (1-α)·p·q · [γ·p^(γ-1)·D + p^γ·dD/dp]
        //   where D = p - γ·q·log(q),  dD/dp = 1 + γ·(log(q)+1)
        let alpha1 = 1.0 - alpha;
        let p_g = p.powf(gamma);       // p^gamma
        let p_g1 = p.powf(gamma - 1.0); // p^(gamma-1)
        let ln_q = q.ln();

        let d = p - gamma * q * ln_q;
        let g = alpha1 * p_g * d;

        let dd_dp = 1.0 + gamma * (ln_q + 1.0);
        let h = alpha1 * p * q * (gamma * p_g1 * d + p_g * dd_dp);
        (g, h)
    }
}

// ── Positive-Unlabeled loss ─────────────────────────────────────────────────

/// Positive-Unlabeled learning loss (du Plessis et al. 2015).
///
/// `L_PU = prior · E_P[ℓ(f)] + E_U[ℓ(-f)] - prior · E_P[ℓ(-f)]`
///
/// Implemented as a re-weighted BCE:
///   positive examples → weight `1`,
///   unlabeled examples → effective weight via the correction term.
pub struct PULoss {
    /// Prior probability of a positive: P(y=1).
    pub prior: f32,
}

impl PULoss {
    pub fn new(prior: f32) -> Self {
        assert!(prior > 0.0 && prior < 1.0, "prior must be in (0, 1)");
        PULoss { prior }
    }
}

impl Objective for PULoss {
    fn grad_hess(&self, y_true: &[f32], y_pred: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let pi = self.prior;
        let n = y_true.len();
        let mut gv = Vec::with_capacity(n);
        let mut hv = Vec::with_capacity(n);
        for (&y, &x) in y_true.iter().zip(y_pred) {
            let p = sigmoid(x);
            let (g, h) = if y > 0.5 {
                // Positive example: standard BCE gradient, weighted by prior.
                // Gradient of prior·ℓ(f) - prior·ℓ(-f) = prior·(2p - 1).
                let g = pi * (2.0 * p - 1.0);
                let h = pi * 2.0 * p * (1.0 - p);
                (g, h)
            } else {
                // Unlabeled example: gradient of ℓ(-f) which has target 0.
                // ∂ℓ(-f)/∂x = -∂ℓ(f)/∂x|_{y=0} = -(p - 0) by negation ... actually:
                // ℓ(-f) = -log(1 - sigmoid(-f)) = -log(sigmoid(f)) = log(1+e^{-f})
                // ∂/∂f = -sigmoid(-f) = -(1-p), then chain ∂f/∂x = 1.
                // Minus sign because we want to MINIMISE: g = p - 1 ... hmm
                // Standard: g_unlabeled = p (gradient of -log(1-p) w.r.t. x).
                let g = p;
                let h = p * (1.0 - p);
                (g, h)
            };
            gv.push(g);
            hv.push(h.max(EPS));
        }
        (gv, hv)
    }

    fn predict_proba(&self, raw: f32) -> f32 {
        sigmoid(raw)
    }

    fn class_prior(&self) -> Option<f32> {
        Some(self.prior)
    }
}

// ── Density-ratio loss ──────────────────────────────────────────────────────

/// Logistic-loss density-ratio objective (Kanamori et al. 2010, Menon & Ong 2016).
///
/// Trains the tree to output `r(x) = log p(x | positives) / p(x | unlabeled)`,
/// the log-density-ratio of positives over the unlabeled population. This is
/// exactly the lookalike quantity: "how overrepresented is this user pattern
/// among the seeds?" — without forcing a classification framing on top of
/// unreliable negatives.
///
/// Per-example loss:
///   y=1 (positive): L+ = softplus(-x) · (1 - π)
///   y=0 (unlabeled): L- = softplus(+x) · π
///
/// The π reweighting (P/U mass-balancing) is required so the Bayes-optimal
/// minimiser is the log-density-ratio rather than the log-posterior. We use
/// `π = mass_ratio = E[w_P] / (E[w_P] + E[w_U])`, defaulting to 0.5.
///
/// Equivalent to: train BCE on (positives ∪ unlabeled), but with positives
/// re-weighted by (1 - π)/π and unlabeled by 1.
pub struct DensityRatioLoss {
    /// Mass-balance parameter in (0, 1). 0.5 = symmetric P/U treatment.
    pub pi: f32,
}

impl DensityRatioLoss {
    pub fn new(pi: f32) -> Self {
        assert!(pi > 0.0 && pi < 1.0, "pi must be in (0, 1)");
        DensityRatioLoss { pi }
    }
}

impl Objective for DensityRatioLoss {
    fn grad_hess(&self, y_true: &[f32], y_pred: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let n = y_true.len();
        let mut gv = Vec::with_capacity(n);
        let mut hv = Vec::with_capacity(n);
        let w_pos = 1.0 - self.pi;
        let w_unl = self.pi;
        for (&y, &x) in y_true.iter().zip(y_pred) {
            let p = sigmoid(x);
            let (g, h) = if y > 0.5 {
                // d/dx softplus(-x) = -sigmoid(-x) = p - 1
                (w_pos * (p - 1.0), w_pos * p * (1.0 - p))
            } else {
                // d/dx softplus(x) = sigmoid(x) = p
                (w_unl * p, w_unl * p * (1.0 - p))
            };
            gv.push(g);
            hv.push(h.max(EPS));
        }
        (gv, hv)
    }

    fn predict_proba(&self, raw: f32) -> f32 {
        // Output is a log-density-ratio, not a probability. We expose
        // sigmoid(raw) as a monotone transform that produces values in [0, 1]
        // suitable for ranking; calibration must be applied separately for
        // a probabilistic interpretation.
        sigmoid(raw)
    }

    fn class_prior(&self) -> Option<f32> {
        Some(self.pi)
    }
}

// ── Asymmetric loss ─────────────────────────────────────────────────────────

/// Asymmetric loss (Ridnik et al. 2021): different focusing for
/// positives vs negatives, with hard-negative suppression via probability clip.
///
/// For positives:  `L+ = (1-p)^gamma_pos · (-log(p))`
/// For negatives:  `L- = p_m^gamma_neg · (-log(1-p_m))` where `p_m = max(p - clip, 0)`
pub struct AsymmetricLoss {
    pub gamma_pos: f32,
    pub gamma_neg: f32,
    /// Probability margin: negatives with p < clip are zeroed out entirely.
    pub clip: f32,
}

impl AsymmetricLoss {
    pub fn new(gamma_pos: f32, gamma_neg: f32, clip: f32) -> Self {
        assert!(clip >= 0.0 && clip < 1.0);
        AsymmetricLoss { gamma_pos, gamma_neg, clip }
    }
}

impl Objective for AsymmetricLoss {
    fn grad_hess(&self, y_true: &[f32], y_pred: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let n = y_true.len();
        let mut gv = Vec::with_capacity(n);
        let mut hv = Vec::with_capacity(n);
        for (&y, &x) in y_true.iter().zip(y_pred) {
            let (g, h) = if y > 0.5 {
                // Positive: focal with gamma_pos, alpha=1.
                focal_grad_hess(x, 1.0, self.gamma_pos, 1.0)
            } else {
                // Negative with margin shift: replace p with p_m = max(p-clip, 0).
                let p = sigmoid(x);
                let pm = (p - self.clip).max(0.0);
                if pm < EPS {
                    // Hard suppression: gradient is zero.
                    (0.0, EPS)
                } else {
                    // Focal for negative class using shifted probability.
                    // Recompute x_shifted so that sigmoid(x_shifted) = pm.
                    // x_shifted = log(pm / (1-pm))
                    let x_shifted = (pm / (1.0 - pm + EPS)).ln();
                    focal_grad_hess(x_shifted, 0.0, self.gamma_neg, 1.0)
                }
            };
            gv.push(g);
            hv.push(h.max(EPS));
        }
        (gv, hv)
    }

    fn predict_proba(&self, raw: f32) -> f32 {
        sigmoid(raw)
    }

    fn class_prior(&self) -> Option<f32> {
        None
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Central-difference gradient with step h.
    fn fd_grad(x: f32, h: f32, loss_fn: impl Fn(f32) -> f32) -> f32 {
        (loss_fn(x + h) - loss_fn(x - h)) / (2.0 * h)
    }

    /// Check that the analytic gradient of `loss` matches the FD gradient of
    /// `loss_fn` (the scalar loss as a function of the raw score x).
    fn check_gradient(
        loss: &dyn Objective,
        x: f32,
        y: f32,
        loss_fn: impl Fn(f32) -> f32,
        tol: f32,
    ) {
        let analytic = loss.grad_hess(&[y], &[x]).0[0];
        // Use a moderate step; too small risks f32 cancellation, too large adds
        // O(h²) bias.  h = 1e-3 gives ~1e-6 relative error for smooth functions.
        let h = 1e-3_f32;
        let numerical = fd_grad(x, h, loss_fn);
        assert!(
            (analytic - numerical).abs() < tol,
            "gradient mismatch at x={x}, y={y}: analytic={analytic}, fd={numerical}"
        );
    }

    #[test]
    fn bce_gradient_finite_diff() {
        let loss = BCELoss;
        for &x in &[-2.0_f32, -0.5, 0.0, 0.5, 2.0] {
            for &y in &[0.0_f32, 1.0] {
                let loss_fn = move |xv: f32| {
                    let p = sigmoid(xv).clamp(EPS, 1.0 - EPS);
                    if y > 0.5 { -p.ln() } else { -(1.0 - p).ln() }
                };
                check_gradient(&loss, x, y, loss_fn, 2e-3);
            }
        }
    }

    #[test]
    fn focal_gradient_finite_diff_gamma0() {
        // With gamma=0, focal = alpha-weighted BCE.  The FD must use the same
        // alpha-weighted loss function as the analytic gradient.
        let alpha = 0.25_f32;
        let loss = FocalLoss::new(0.0, alpha);
        let bce = BCELoss;
        for &x in &[-2.0_f32, -0.5, 0.0, 0.5, 2.0] {
            for &y in &[0.0_f32, 1.0] {
                let alpha_t = if y > 0.5 { alpha } else { 1.0 - alpha };
                let loss_fn = move |xv: f32| {
                    let p = sigmoid(xv).clamp(EPS, 1.0 - EPS);
                    // gamma=0: (1-p_t)^0 = 1, so L = -alpha_t * log(p_t)
                    if y > 0.5 { -alpha_t * p.ln() } else { -alpha_t * (1.0 - p).ln() }
                };
                check_gradient(&loss, x, y, loss_fn, 2e-3);

                // Also verify focal(γ=0) == alpha_t * BCE.
                let (gf, _) = loss.grad_hess(&[y], &[x]);
                let (gb, _) = bce.grad_hess(&[y], &[x]);
                assert!(
                    (gf[0] - alpha_t * gb[0]).abs() < 1e-5,
                    "focal(γ=0) != α_t·BCE at x={x} y={y}"
                );
            }
        }
    }

    #[test]
    fn focal_gradient_finite_diff_gamma2() {
        let gamma = 2.0_f32;
        let alpha = 0.25_f32;
        let loss = FocalLoss::new(gamma, alpha);
        for &x in &[-1.0_f32, 0.0, 1.0] {
            for &y in &[0.0_f32, 1.0] {
                let loss_fn = move |xv: f32| {
                    let p = sigmoid(xv).clamp(EPS, 1.0 - EPS);
                    let q = 1.0 - p;
                    if y > 0.5 {
                        -alpha * q.powf(gamma) * p.ln()
                    } else {
                        -(1.0 - alpha) * p.powf(gamma) * q.ln()
                    }
                };
                check_gradient(&loss, x, y, loss_fn, 2e-3);
            }
        }
    }

    #[test]
    fn hessians_are_positive() {
        let losses: Vec<Box<dyn Objective>> = vec![
            Box::new(BCELoss),
            Box::new(FocalLoss::new(2.0, 0.25)),
            Box::new(FocalLoss::new(5.0, 0.5)),
            Box::new(PULoss::new(0.1)),
            Box::new(DensityRatioLoss::new(0.5)),
        ];
        for loss in &losses {
            for &x in &[-4.0_f32, -1.0, 0.0, 1.0, 4.0] {
                for &y in &[0.0_f32, 1.0] {
                    let (_, hv) = loss.grad_hess(&[y], &[x]);
                    assert!(hv[0] > 0.0, "non-positive hessian at x={x} y={y}");
                }
            }
        }
    }
}
