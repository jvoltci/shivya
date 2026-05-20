use crate::blanket::MarkovBlanket;

pub trait MatrixMath<const N: usize> {
    fn det(&self) -> f64;
    fn inv(&self) -> [[f64; N]; N];
}

impl MatrixMath<1> for [[f64; 1]; 1] {
    fn det(&self) -> f64 {
        self[0][0]
    }
    fn inv(&self) -> [[f64; 1]; 1] {
        if self[0][0].abs() < 1e-15 {
            panic!("Singular 1x1 matrix");
        }
        [[1.0 / self[0][0]]]
    }
}

impl MatrixMath<2> for [[f64; 2]; 2] {
    fn det(&self) -> f64 {
        self[0][0] * self[1][1] - self[0][1] * self[1][0]
    }
    fn inv(&self) -> [[f64; 2]; 2] {
        let d = self.det();
        if d.abs() < 1e-15 {
            panic!("Singular 2x2 matrix");
        }
        [
            [self[1][1] / d, -self[0][1] / d],
            [-self[1][0] / d, self[0][0] / d],
        ]
    }
}

impl MatrixMath<3> for [[f64; 3]; 3] {
    fn det(&self) -> f64 {
        self[0][0] * (self[1][1] * self[2][2] - self[1][2] * self[2][1])
            - self[0][1] * (self[1][0] * self[2][2] - self[1][2] * self[2][0])
            + self[0][2] * (self[1][0] * self[2][1] - self[1][1] * self[2][0])
    }
    fn inv(&self) -> [[f64; 3]; 3] {
        let d = self.det();
        if d.abs() < 1e-15 {
            panic!("Singular 3x3 matrix");
        }
        let c00 = self[1][1] * self[2][2] - self[1][2] * self[2][1];
        let c01 = -(self[1][0] * self[2][2] - self[1][2] * self[2][0]);
        let c02 = self[1][0] * self[2][1] - self[1][1] * self[2][0];

        let c10 = -(self[0][1] * self[2][2] - self[0][2] * self[2][1]);
        let c11 = self[0][0] * self[2][2] - self[0][2] * self[2][0];
        let c12 = -(self[0][0] * self[2][1] - self[0][1] * self[2][0]);

        let c20 = self[0][1] * self[1][2] - self[0][2] * self[1][1];
        let c21 = -(self[0][0] * self[1][2] - self[0][2] * self[1][0]);
        let c22 = self[0][0] * self[1][1] - self[0][1] * self[1][0];

        [
            [c00 / d, c10 / d, c20 / d],
            [c01 / d, c11 / d, c21 / d],
            [c02 / d, c12 / d, c22 / d],
        ]
    }
}

// Matrix-vector multiply
pub fn mat_mul_vec<const R: usize, const C: usize>(mat: &[[f64; C]; R], vec: &[f64; C]) -> [f64; R] {
    let mut out = [0.0; R];
    for r in 0..R {
        let mut sum = 0.0;
        for c in 0..C {
            sum += mat[r][c] * vec[c];
        }
        out[r] = sum;
    }
    out
}

// Matrix-matrix multiply
pub fn mat_mul_mat<const R: usize, const C: usize, const P: usize>(
    mat_a: &[[f64; C]; R],
    mat_b: &[[f64; P]; C],
) -> [[f64; P]; R] {
    let mut out = [[0.0; P]; R];
    for r in 0..R {
        for p in 0..P {
            let mut sum = 0.0;
            for c in 0..C {
                sum += mat_a[r][c] * mat_b[c][p];
            }
            out[r][p] = sum;
        }
    }
    out
}

// Transpose of a matrix
pub fn mat_transpose<const R: usize, const C: usize>(mat: &[[f64; C]; R]) -> [[f64; R]; C] {
    let mut out = [[0.0; R]; C];
    for r in 0..R {
        for c in 0..C {
            out[c][r] = mat[r][c];
        }
    }
    out
}

pub fn vec_dot<const N: usize>(u: &[f64; N], v: &[f64; N]) -> f64 {
    u.iter().zip(v.iter()).map(|(&ui, &vi)| ui * vi).sum()
}

pub fn vec_sub<const N: usize>(u: &[f64; N], v: &[f64; N]) -> [f64; N] {
    let mut out = [0.0; N];
    for i in 0..N {
        out[i] = u[i] - v[i];
    }
    out
}

pub fn vec_add<const N: usize>(u: &[f64; N], v: &[f64; N]) -> [f64; N] {
    let mut out = [0.0; N];
    for i in 0..N {
        out[i] = u[i] + v[i];
    }
    out
}

pub fn vec_scale<const N: usize>(u: &[f64; N], s: f64) -> [f64; N] {
    let mut out = [0.0; N];
    for i in 0..N {
        out[i] = u[i] * s;
    }
    out
}

pub fn vec_norm_sq<const N: usize>(u: &[f64; N]) -> f64 {
    u.iter().map(|&x| x * x).sum()
}

pub fn gaussian_kl<const N: usize>(
    mu1: &[f64; N],
    sigma1: &[[f64; N]; N],
    mu2: &[f64; N],
    sigma2: &[[f64; N]; N],
) -> f64
where
    [[f64; N]; N]: MatrixMath<N>,
{
    let sigma2_inv = sigma2.inv();

    let prod = mat_mul_mat(&sigma2_inv, sigma1);
    let mut tr = 0.0;
    for i in 0..N {
        tr += prod[i][i];
    }

    let diff = vec_sub(mu2, mu1);
    let inv_diff = mat_mul_vec(&sigma2_inv, &diff);
    let quadratic = vec_dot(&diff, &inv_diff);

    let det1 = sigma1.det();
    let det2 = sigma2.det();

    let log_det_ratio = (det2 / det1).ln();

    0.5 * (tr + quadratic - (N as f64) + log_det_ratio)
}

pub struct GibbsFluxAgent<const S_DIM: usize, const A_DIM: usize, const I_DIM: usize>
where
    [[f64; S_DIM]; S_DIM]: MatrixMath<S_DIM>,
    [[f64; I_DIM]; I_DIM]: MatrixMath<I_DIM>,
{
    pub blanket: MarkovBlanket<S_DIM, A_DIM, I_DIM>,

    // Generative Model Parameters
    pub mu_prior: [f64; I_DIM],
    pub sigma_prior: [[f64; I_DIM]; I_DIM],
    pub g_s: [[f64; I_DIM]; S_DIM],
    pub sigma_s_0: [[f64; S_DIM]; S_DIM],
    pub w: [[f64; A_DIM]; S_DIM],
    pub m: [[f64; A_DIM]; I_DIM],

    // Preferences
    pub mu_pref: [f64; S_DIM],
    pub sigma_pref: [[f64; S_DIM]; S_DIM],

    pub sigma_prior_inv: [[f64; I_DIM]; I_DIM],

    // Internal representation beliefs
    pub mu_q: [f64; I_DIM],
    pub sigma_q: [[f64; I_DIM]; I_DIM],
}

impl<const S_DIM: usize, const A_DIM: usize, const I_DIM: usize> GibbsFluxAgent<S_DIM, A_DIM, I_DIM>
where
    [[f64; S_DIM]; S_DIM]: MatrixMath<S_DIM>,
    [[f64; I_DIM]; I_DIM]: MatrixMath<I_DIM>,
{
    pub fn new(
        mu_prior: [f64; I_DIM],
        sigma_prior: [[f64; I_DIM]; I_DIM],
        g_s: [[f64; I_DIM]; S_DIM],
        sigma_s_0: [[f64; S_DIM]; S_DIM],
        w: [[f64; A_DIM]; S_DIM],
        m: [[f64; A_DIM]; I_DIM],
        mu_pref: [f64; S_DIM],
        sigma_pref: [[f64; S_DIM]; S_DIM],
    ) -> Self {
        let sigma_prior_inv = sigma_prior.inv();
        let blanket = MarkovBlanket::new();
        let mu_q = mu_prior;
        
        let mut agent = Self {
            blanket,
            mu_prior,
            sigma_prior,
            g_s,
            sigma_s_0,
            w,
            m,
            mu_pref,
            sigma_pref,
            sigma_prior_inv,
            mu_q,
            sigma_q: [[0.0; I_DIM]; I_DIM],
        };
        agent.sigma_q = agent.compute_optimal_sigma_q(&agent.blanket.active);
        agent
    }

    pub fn compute_sigma_s(&self, active_state: &[f64; A_DIM]) -> [[f64; S_DIM]; S_DIM] {
        let mut sigma_s = self.sigma_s_0;
        let w_a = mat_mul_vec(&self.w, active_state);
        for i in 0..S_DIM {
            sigma_s[i][i] += (-w_a[i]).exp();
        }
        sigma_s
    }

    pub fn compute_optimal_sigma_q(&self, active_state: &[f64; A_DIM]) -> [[f64; I_DIM]; I_DIM] {
        let sigma_s = self.compute_sigma_s(active_state);
        let sigma_s_inv = sigma_s.inv();

        let g_s_t = mat_transpose(&self.g_s);
        let temp = mat_mul_mat(&g_s_t, &sigma_s_inv);
        let precision_contrib = mat_mul_mat(&temp, &self.g_s);

        let mut total_precision = [[0.0; I_DIM]; I_DIM];
        for i in 0..I_DIM {
            for j in 0..I_DIM {
                total_precision[i][j] = self.sigma_prior_inv[i][j] + precision_contrib[i][j];
            }
        }

        total_precision.inv()
    }

    pub fn compute_free_energy(&self, s: &[f64; S_DIM], mu_q: &[f64; I_DIM], sigma_q: &[[f64; I_DIM]; I_DIM]) -> f64 {
        let kl = gaussian_kl(mu_q, sigma_q, &self.mu_prior, &self.sigma_prior);

        let sigma_s = self.compute_sigma_s(&self.blanket.active);
        let sigma_s_inv = sigma_s.inv();
        let det_s = sigma_s.det();

        let pred_obs = mat_mul_vec(&self.g_s, mu_q);
        let diff = vec_sub(s, &pred_obs);
        let inv_diff = mat_mul_vec(&sigma_s_inv, &diff);
        let quad = vec_dot(&diff, &inv_diff);

        let g_s_t = mat_transpose(&self.g_s);
        let temp = mat_mul_mat(&g_s_t, &sigma_s_inv);
        let quad_prec = mat_mul_mat(&temp, &self.g_s);

        let mut trace_term = 0.0;
        for i in 0..I_DIM {
            for j in 0..I_DIM {
                trace_term += quad_prec[i][j] * sigma_q[j][i];
            }
        }

        let nll = 0.5 * ((S_DIM as f64) * (2.0 * std::f64::consts::PI).ln() + det_s.ln() + quad + trace_term);
        kl + nll
    }

    pub fn update_beliefs(&mut self, observation: &[f64; S_DIM], lr: f64, max_iters: usize, tol: f64) -> Vec<f64> {
        self.blanket.update_sensory(observation);
        self.sigma_q = self.compute_optimal_sigma_q(&self.blanket.active);

        let sigma_s = self.compute_sigma_s(&self.blanket.active);
        let sigma_s_inv = sigma_s.inv();
        let g_s_t = mat_transpose(&self.g_s);

        let mut f_history = Vec::new();

        for _ in 0..max_iters {
            let f_val = self.compute_free_energy(observation, &self.mu_q, &self.sigma_q);
            f_history.push(f_val);

            // Compute gradient
            // prior_term = sigma_prior_inv * (mu_q - mu_prior)
            let mu_diff = vec_sub(&self.mu_q, &self.mu_prior);
            let prior_term = mat_mul_vec(&self.sigma_prior_inv, &mu_diff);

            // likelihood_term = G_s^T * sigma_s_inv * (observation - G_s * mu_q)
            let pred_obs = mat_mul_vec(&self.g_s, &self.mu_q);
            let diff = vec_sub(observation, &pred_obs);
            let inv_diff = mat_mul_vec(&sigma_s_inv, &diff);
            let likelihood_term = mat_mul_vec(&g_s_t, &inv_diff);

            let grad = vec_sub(&prior_term, &likelihood_term);

            if grad.iter().map(|&x| x * x).sum::<f64>().sqrt() < tol {
                break;
            }

            self.mu_q = vec_sub(&self.mu_q, &vec_scale(&grad, lr));
        }

        self.blanket.update_internal(&self.mu_q);
        f_history
    }

    pub fn evaluate_policies(&self, policies: &[[f64; A_DIM]]) -> Vec<f64> {
        let mut results = Vec::new();
        for policy in policies {
            // theta_tau = mu_q + m * policy
            let delta_theta = mat_mul_vec(&self.m, policy);
            let pred_theta = vec_add(&self.mu_q, &delta_theta);

            // s_tau = g_s * theta_tau
            let pred_obs = mat_mul_vec(&self.g_s, &pred_theta);

            let sigma_s_pi = self.compute_sigma_s(policy);
            let sigma_q_pi = self.compute_optimal_sigma_q(policy);

            // pred_obs_cov = g_s * sigma_q * g_s^t + sigma_s
            let g_s_sigma_q = mat_mul_mat(&self.g_s, &sigma_q_pi);
            let g_s_t = mat_transpose(&self.g_s);
            let mut pred_obs_cov = mat_mul_mat(&g_s_sigma_q, &g_s_t);
            for i in 0..S_DIM {
                for j in 0..S_DIM {
                    pred_obs_cov[i][j] += sigma_s_pi[i][j];
                }
            }

            // Pragmatic: KL( N(pred_obs, pred_obs_cov) || N(mu_pref, sigma_pref) )
            let pragmatic = gaussian_kl(&pred_obs, &pred_obs_cov, &self.mu_pref, &self.sigma_pref);

            // Epistemic: H(sigma_q) = 0.5 * ln( (2*pi*e)^k * det(sigma_q) )
            let det_q = sigma_q_pi.det();
            let safe_det_q = if det_q <= 0.0 { 1e-15 } else { det_q };
            let epistemic = 0.5 * (((2.0 * std::f64::consts::PI * std::f64::consts::E).powi(I_DIM as i32) * safe_det_q).ln());

            results.push(pragmatic + epistemic);
        }
        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_math_2d() {
        let a = [
            [4.0, 7.0],
            [2.0, 6.0]
        ];
        assert!((a.det() - 10.0).abs() < 1e-9);
        let a_inv = a.inv();
        let i_check = mat_mul_mat(&a, &a_inv);
        assert!((i_check[0][0] - 1.0).abs() < 1e-9);
        assert!((i_check[0][1] - 0.0).abs() < 1e-9);
        assert!((i_check[1][0] - 0.0).abs() < 1e-9);
        assert!((i_check[1][1] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_belief_update_convergence() {
        let mut agent = GibbsFluxAgent::<2, 1, 2>::new(
            [0.0, 0.0],
            [[10.0, 0.0], [0.0, 10.0]],
            [[2.0, 0.5], [0.5, 1.5]],
            [[0.1, 0.0], [0.0, 0.1]],
            [[0.0], [0.0]],
            [[0.0], [0.0]],
            [0.0, 0.0],
            [[1.0, 0.0], [0.0, 1.0]],
        );

        let obs = [2.5, 1.0];
        let f_history = agent.update_beliefs(&obs, 0.1, 100, 1e-6);

        assert!(f_history.len() > 1);
        assert!(f_history[f_history.len() - 1] < f_history[0]);

        let pred_obs = mat_mul_vec(&agent.g_s, &agent.mu_q);
        assert!((pred_obs[0] - obs[0]).abs() < 0.2);
        assert!((pred_obs[1] - obs[1]).abs() < 0.2);
    }

    #[test]
    fn test_epistemic_policy_prioritization() {
        let agent = GibbsFluxAgent::<1, 1, 1>::new(
            [0.0],
            [[10.0]],
            [[1.0]],
            [[1.0]],
            [[2.0]], // W matrix (action sensitivity)
            [[1.0]],
            [1.0],
            [[10.0]], // Very broad target preference
        );

        let policies = vec![[0.0], [1.0], [2.0]];
        let g_scores = agent.evaluate_policies(&policies);

        // Action 2.0 (epistemic, noise reducing) should score lower than 0.0 or 1.0
        assert!(g_scores[2] < g_scores[0]);
    }
}
