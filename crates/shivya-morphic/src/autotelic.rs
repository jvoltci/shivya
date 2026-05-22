use shivya_flux::{SubstrateError, RIDGE_EPSILON};

/// Reports the most-recent inversion failure encountered by
/// `dyn_mat_inv_stable`. Resets on every successful plain inversion. Exposed
/// for diagnostics; the math path itself always returns a usable matrix.
pub fn last_stabilization_event() -> Option<SubstrateError> {
    LAST_STAB_EVENT.with(|cell| cell.borrow().clone())
}

thread_local! {
    static LAST_STAB_EVENT: std::cell::RefCell<Option<SubstrateError>> =
        const { std::cell::RefCell::new(None) };
}

fn record_stab_event(evt: Option<SubstrateError>) {
    LAST_STAB_EVENT.with(|cell| *cell.borrow_mut() = evt);
}

/// Inverts `matrix`, with a ridge-regularised fallback for singular inputs.
///
/// - Plain inversion first.
/// - On singularity, adds `RIDGE_EPSILON` to the diagonal and retries.
/// - If still singular, returns the identity (equivalent to "no information"
///   on this update step) instead of panicking.
///
/// Each step's outcome is recorded in `last_stabilization_event()` so the
/// daemon can surface degeneracy in telemetry without crashing.
pub fn dyn_mat_inv_stable(matrix: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let n = matrix.len();
    match dyn_mat_inv(matrix) {
        Ok(inv) => {
            record_stab_event(None);
            return inv;
        }
        Err(_) => {
            record_stab_event(Some(SubstrateError::SingularMatrix {
                size: n,
                det: dyn_mat_det(matrix),
            }));
        }
    }
    // Ridge regularisation: M + epsilon * I breaks colinearity without
    // meaningfully changing the well-conditioned subspace.
    let mut stabilised = matrix.clone();
    for i in 0..n {
        stabilised[i][i] += RIDGE_EPSILON;
    }
    if let Ok(inv) = dyn_mat_inv(&stabilised) {
        return inv;
    }
    record_stab_event(Some(SubstrateError::StabilizationFailed {
        size: n,
        ridge: RIDGE_EPSILON,
    }));
    let mut identity = vec![vec![0.0; n]; n];
    for i in 0..n {
        identity[i][i] = 1.0;
    }
    identity
}

// Dynamic matrix and vector helper functions
pub fn dyn_mat_inv(matrix: &Vec<Vec<f64>>) -> Result<Vec<Vec<f64>>, String> {
    let n = matrix.len();
    let mut aug = vec![vec![0.0; 2 * n]; n];
    for i in 0..n {
        for j in 0..n {
            aug[i][j] = matrix[i][j];
        }
        aug[i][n + i] = 1.0;
    }

    for i in 0..n {
        let mut pivot_row = i;
        for j in (i + 1)..n {
            if aug[j][i].abs() > aug[pivot_row][i].abs() {
                pivot_row = j;
            }
        }
        if aug[pivot_row][i].abs() < 1e-12 {
            return Err("Matrix is singular".to_string());
        }
        if pivot_row != i {
            aug.swap(i, pivot_row);
        }

        let pivot = aug[i][i];
        for j in 0..(2 * n) {
            aug[i][j] /= pivot;
        }

        for j in 0..n {
            if j != i {
                let factor = aug[j][i];
                for k in 0..(2 * n) {
                    aug[j][k] -= factor * aug[i][k];
                }
            }
        }
    }

    let mut inv = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            inv[i][j] = aug[i][n + j];
        }
    }
    Ok(inv)
}

pub fn dyn_mat_det(matrix: &Vec<Vec<f64>>) -> f64 {
    let n = matrix.len();
    let mut m = matrix.clone();
    let mut det = 1.0;

    for i in 0..n {
        let mut pivot_row = i;
        for j in (i + 1)..n {
            if m[j][i].abs() > m[pivot_row][i].abs() {
                pivot_row = j;
            }
        }
        if m[pivot_row][i].abs() < 1e-15 {
            return 0.0;
        }
        if pivot_row != i {
            m.swap(i, pivot_row);
            det = -det;
        }
        det *= m[i][i];

        for j in (i + 1)..n {
            let factor = m[j][i] / m[i][i];
            for k in i..n {
                m[j][k] -= factor * m[i][k];
            }
        }
    }
    det
}

pub fn dyn_mat_mul_vec(mat: &Vec<Vec<f64>>, vec: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; mat.len()];
    for r in 0..mat.len() {
        let mut sum = 0.0;
        for c in 0..mat[r].len() {
            sum += mat[r][c] * vec[c];
        }
        out[r] = sum;
    }
    out
}

pub fn dyn_mat_mul_mat(mat_a: &Vec<Vec<f64>>, mat_b: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let rows = mat_a.len();
    let cols = mat_b[0].len();
    let inner = mat_a[0].len();
    let mut out = vec![vec![0.0; cols]; rows];
    for r in 0..rows {
        for p in 0..cols {
            let mut sum = 0.0;
            for c in 0..inner {
                sum += mat_a[r][c] * mat_b[c][p];
            }
            out[r][p] = sum;
        }
    }
    out
}

pub fn dyn_mat_transpose(mat: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let rows = mat.len();
    let cols = mat[0].len();
    let mut out = vec![vec![0.0; rows]; cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c][r] = mat[r][c];
        }
    }
    out
}

pub fn dyn_vec_dot(u: &[f64], v: &[f64]) -> f64 {
    u.iter().zip(v.iter()).map(|(&ui, &vi)| ui * vi).sum()
}

pub fn dyn_vec_sub(u: &[f64], v: &[f64]) -> Vec<f64> {
    u.iter().zip(v.iter()).map(|(&ui, &vi)| ui - vi).collect()
}

pub fn dyn_gaussian_kl(
    mu1: &[f64],
    sigma1: &Vec<Vec<f64>>,
    mu2: &[f64],
    sigma2: &Vec<Vec<f64>>,
) -> f64 {
    let n = mu1.len();
    let sigma2_inv = dyn_mat_inv_stable(sigma2);

    let prod = dyn_mat_mul_mat(&sigma2_inv, sigma1);
    let mut tr = 0.0;
    for i in 0..n {
        tr += prod[i][i];
    }

    let diff = dyn_vec_sub(mu2, mu1);
    let inv_diff = dyn_mat_mul_vec(&sigma2_inv, &diff);
    let quadratic = dyn_vec_dot(&diff, &inv_diff);

    let det1 = dyn_mat_det(sigma1);
    let det2 = dyn_mat_det(sigma2);

    let log_det_ratio = if det1 <= 0.0 || det2 <= 0.0 {
        0.0
    } else {
        (det2 / det1).ln()
    };

    0.5 * (tr + quadratic - (n as f64) + log_det_ratio)
}

pub struct DynamicGibbsAgent {
    pub s_dim: usize,
    pub a_dim: usize,
    pub i_dim: usize,

    pub mu_prior: Vec<f64>,
    pub sigma_prior: Vec<Vec<f64>>,
    pub g_s: Vec<Vec<f64>>, // s_dim x i_dim
    pub sigma_s_0: Vec<Vec<f64>>,
    pub w: Vec<Vec<f64>>,
    pub m: Vec<Vec<f64>>,

    pub mu_pref: Vec<f64>,
    pub sigma_pref: Vec<Vec<f64>>,

    pub mu_q: Vec<f64>,
    pub sigma_q: Vec<Vec<f64>>,

    pub active_state: Vec<f64>,
    pub f_history: Vec<f64>,
    pub tau_novelty: f64,
}

impl DynamicGibbsAgent {
    pub fn new(
        s_dim: usize,
        a_dim: usize,
        i_dim: usize,
        mu_prior: Vec<f64>,
        sigma_prior: Vec<Vec<f64>>,
        g_s: Vec<Vec<f64>>,
        sigma_s_0: Vec<Vec<f64>>,
        w: Vec<Vec<f64>>,
        m: Vec<Vec<f64>>,
        mu_pref: Vec<f64>,
        sigma_pref: Vec<Vec<f64>>,
        tau_novelty: f64,
    ) -> Self {
        let mu_q = mu_prior.clone();
        let active_state = vec![0.0; a_dim];
        let sigma_q = vec![vec![0.0; i_dim]; i_dim];
        
        let mut agent = Self {
            s_dim,
            a_dim,
            i_dim,
            mu_prior,
            sigma_prior,
            g_s,
            sigma_s_0,
            w,
            m,
            mu_pref,
            sigma_pref,
            mu_q,
            sigma_q,
            active_state,
            f_history: Vec::new(),
            tau_novelty,
        };
        agent.sigma_q = agent.compute_optimal_sigma_q(&agent.active_state);
        agent
    }

    pub fn compute_sigma_s(&self, active: &[f64]) -> Vec<Vec<f64>> {
        let mut sigma_s = self.sigma_s_0.clone();
        let w_a = dyn_mat_mul_vec(&self.w, active);
        for i in 0..self.s_dim {
            sigma_s[i][i] += (-w_a[i]).exp();
        }
        sigma_s
    }

    pub fn compute_optimal_sigma_q(&self, active: &[f64]) -> Vec<Vec<f64>> {
        let sigma_s = self.compute_sigma_s(active);
        let sigma_s_inv = dyn_mat_inv_stable(&sigma_s);

        let g_s_t = dyn_mat_transpose(&self.g_s);
        let temp = dyn_mat_mul_mat(&g_s_t, &sigma_s_inv);
        let precision_contrib = dyn_mat_mul_mat(&temp, &self.g_s);

        let sigma_prior_inv = dyn_mat_inv_stable(&self.sigma_prior);

        let mut total_precision = vec![vec![0.0; self.i_dim]; self.i_dim];
        for i in 0..self.i_dim {
            for j in 0..self.i_dim {
                total_precision[i][j] = sigma_prior_inv[i][j] + precision_contrib[i][j];
            }
        }

        dyn_mat_inv_stable(&total_precision)
    }

    pub fn compute_free_energy(&self, s: &[f64], mu_q: &[f64], sigma_q: &Vec<Vec<f64>>) -> f64 {
        let kl = dyn_gaussian_kl(mu_q, sigma_q, &self.mu_prior, &self.sigma_prior);

        let sigma_s = self.compute_sigma_s(&self.active_state);
        let sigma_s_inv = dyn_mat_inv_stable(&sigma_s);
        let det_s = dyn_mat_det(&sigma_s);

        let pred_obs = dyn_mat_mul_vec(&self.g_s, mu_q);
        let diff = dyn_vec_sub(s, &pred_obs);
        let inv_diff = dyn_mat_mul_vec(&sigma_s_inv, &diff);
        let quad = dyn_vec_dot(&diff, &inv_diff);

        let g_s_t = dyn_mat_transpose(&self.g_s);
        let temp = dyn_mat_mul_mat(&g_s_t, &sigma_s_inv);
        let quad_prec = dyn_mat_mul_mat(&temp, &self.g_s);

        let mut trace_term = 0.0;
        for i in 0..self.i_dim {
            for j in 0..self.i_dim {
                trace_term += quad_prec[i][j] * sigma_q[j][i];
            }
        }

        let safe_det_s = if det_s > 0.0 { det_s } else { RIDGE_EPSILON };
        let nll = 0.5 * ((self.s_dim as f64) * (2.0 * std::f64::consts::PI).ln() + safe_det_s.ln() + quad + trace_term);
        kl + nll
    }

    pub fn update_beliefs(&mut self, observation: &[f64], lr: f64, max_iters: usize, tol: f64) -> f64 {
        self.sigma_q = self.compute_optimal_sigma_q(&self.active_state);

        let sigma_s = self.compute_sigma_s(&self.active_state);
        let sigma_s_inv = dyn_mat_inv_stable(&sigma_s);
        let g_s_t = dyn_mat_transpose(&self.g_s);
        let sigma_prior_inv = dyn_mat_inv_stable(&self.sigma_prior);

        let mut final_f = 0.0;

        for _ in 0..max_iters {
            final_f = self.compute_free_energy(observation, &self.mu_q, &self.sigma_q);

            // Gradient: prior_term - likelihood_term
            let mu_diff = dyn_vec_sub(&self.mu_q, &self.mu_prior);
            let prior_term = dyn_mat_mul_vec(&sigma_prior_inv, &mu_diff);

            let pred_obs = dyn_mat_mul_vec(&self.g_s, &self.mu_q);
            let diff = dyn_vec_sub(observation, &pred_obs);
            let inv_diff = dyn_mat_mul_vec(&sigma_s_inv, &diff);
            let likelihood_term = dyn_mat_mul_vec(&g_s_t, &inv_diff);

            let grad = dyn_vec_sub(&prior_term, &likelihood_term);

            let grad_norm = grad.iter().map(|&x| x * x).sum::<f64>().sqrt();
            if grad_norm < tol {
                break;
            }

            for i in 0..self.i_dim {
                self.mu_q[i] -= lr * grad[i];
            }
        }

        self.f_history.push(final_f);
        
        // Autotelic tracking: if moving average of F exceeds threshold, trigger state expansion
        if self.f_history.len() >= 3 {
            let last_3_avg = self.f_history.iter().rev().take(3).sum::<f64>() / 3.0;
            if last_3_avg > self.tau_novelty {
                self.expand_state_space();
            }
        }

        final_f
    }

    pub fn expand_state_space(&mut self) {
        // Expand the dimensionality of internal belief representations by 1
        let old_dim = self.i_dim;
        self.i_dim += 1;

        self.mu_prior.push(0.0);
        
        // Expand prior covariance with wide uncertainty (variance = 10.0)
        let mut new_sigma_prior = vec![vec![0.0; self.i_dim]; self.i_dim];
        for i in 0..old_dim {
            for j in 0..old_dim {
                new_sigma_prior[i][j] = self.sigma_prior[i][j];
            }
        }
        new_sigma_prior[old_dim][old_dim] = 10.0;
        self.sigma_prior = new_sigma_prior;

        // Expand g_s: add a column to each row
        for r in 0..self.s_dim {
            self.g_s[r].push(0.05); // Small coupling factor for the new dimension
        }

        // Expand m: add a row for the new internal dimension
        self.m.push(vec![0.0; self.a_dim]);

        self.mu_q.push(0.0);
        self.sigma_q = self.compute_optimal_sigma_q(&self.active_state);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_autotelic_state_expansion() {
        // Initialize dynamic agent with 2 sensory, 1 active, 2 internal dimensions
        let mut agent = DynamicGibbsAgent::new(
            2, 1, 2,
            vec![0.0, 0.0], // mu_prior
            vec![vec![10.0, 0.0], vec![0.0, 10.0]], // sigma_prior
            vec![vec![2.0, 0.5], vec![0.5, 1.5]], // g_s
            vec![vec![0.1, 0.0], vec![0.0, 0.1]], // sigma_s_0
            vec![vec![0.0], vec![0.0]], // w
            vec![vec![0.0], vec![0.0]], // m
            vec![0.0, 0.0], // mu_pref
            vec![vec![1.0, 0.0], vec![0.0, 1.0]], // sigma_pref
            5.0, // tau_novelty threshold
        );

        assert_eq!(agent.i_dim, 2);

        // Inject severe overload observations three times to force moving average over 5.0
        let obs = [10.0, 10.0];
        
        agent.update_beliefs(&obs, 0.1, 10, 1e-4);
        agent.update_beliefs(&obs, 0.1, 10, 1e-4);
        agent.update_beliefs(&obs, 0.1, 10, 1e-4);

        // Moving average of last 3 free energy steps should breach tau_novelty, trigger expansion
        assert_eq!(agent.i_dim, 3);
        assert_eq!(agent.mu_prior.len(), 3);
        assert_eq!(agent.sigma_prior.len(), 3);
        assert_eq!(agent.g_s[0].len(), 3);
    }
}
