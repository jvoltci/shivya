use crate::complex::SimplicialStateComplex;
use crate::solver::conjugate_gradient;

pub fn reconcile_state_delta(complex: &SimplicialStateComplex, delta_s: &[f64]) -> Vec<f64> {
    let t_count = complex.triangles.len();
    if t_count == 0 {
        // No triangles means no 2-simplices, hence zero curl.
        // The flow is trivially curl-free.
        return delta_s.to_vec();
    }

    let d1 = complex.d1();
    let d1_t = d1.transpose();

    // 1. Compute curl discrepancy: b2 = d1 * delta_s
    let b2 = d1.mul_vec(delta_s);

    // 2. Compute coexact Laplacian: L2 = d1 * d1_t
    let l2 = d1.mul_mat(&d1_t);

    // 3. Solve L2 * beta = b2
    let x0 = vec![0.0; t_count];
    let beta = match conjugate_gradient(&l2, &b2, &x0, 1e-8, 1000) {
        Ok(sol) => sol,
        Err(_) => x0, // Fallback to zero potential if solver diverges
    };

    // 4. Compute coexact conflict flow: S_coexact = d1_t * beta
    let s_coexact = d1_t.mul_vec(&beta);

    // 5. Reconcile: delta_s_reconciled = delta_s - S_coexact
    let mut reconciled = vec![0.0; delta_s.len()];
    for i in 0..delta_s.len() {
        reconciled[i] = delta_s[i] - s_coexact[i];
    }

    reconciled
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::complex::SimplicialStateComplex;

    #[test]
    fn test_topological_reconciliation() {
        let mut complex = SimplicialStateComplex::new();
        complex.add_vertex("A", 10.0);
        complex.add_vertex("B", 10.0);
        complex.add_vertex("C", 10.0);
        complex.add_edge("A", "B", 1.0);
        complex.add_edge("B", "C", 1.0);
        complex.add_edge("A", "C", 1.5); // forms triangle A-B-C

        // Non-zero curl: flow of 1.0, 1.0, 0.0
        let delta_s = vec![1.0, 1.0, 0.0];
        let reconciled = reconcile_state_delta(&complex, &delta_s);

        // Reconciled flows should be curl-free
        // d1 * reconciled should be zero (or close to 0)
        let d1 = complex.d1();
        let curl = d1.mul_vec(&reconciled);
        assert!(curl.len() > 0);
        for &val in &curl {
            assert!(val.abs() < 1e-7, "Curl should be projected out, got {}", val);
        }
    }
}
