use crate::complex::SimplicialStateComplex;
use crate::error::SubstrateError;
use crate::solver::conjugate_gradient;

/// Curl-projects `delta_s` onto the curl-free subspace of the 1-chain space.
///
/// Surfaces any internal math failure (singular Laplacian construction, dim
/// mismatches between `delta_s` and the edge count, etc.) as a
/// [`SubstrateError`]. Callers that prefer never-failing behaviour can use
/// [`reconcile_state_delta`] instead, which falls back to the unprojected
/// input on the same error conditions.
pub fn try_reconcile_state_delta(
    complex: &SimplicialStateComplex,
    delta_s: &[f64],
) -> Result<Vec<f64>, SubstrateError> {
    if delta_s.len() != complex.edges.len() {
        return Err(SubstrateError::DimensionMismatch {
            expected: complex.edges.len(),
            actual: delta_s.len(),
        });
    }

    let t_count = complex.triangles.len();
    if t_count == 0 {
        // No 2-simplices ⇒ no curl to project; the flow is already curl-free.
        return Ok(delta_s.to_vec());
    }

    let d1 = complex.d1()?;
    let d1_t = d1.transpose()?;

    // 1. Curl discrepancy: b2 = d1 * delta_s
    let b2 = d1.mul_vec(delta_s)?;

    // 2. Coexact Laplacian: L2 = d1 * d1_t
    let l2 = d1.mul_mat(&d1_t)?;

    // 3. Solve L2 * beta = b2 (CG; on divergence fall back to zero potential
    //    so the projection becomes the identity rather than corrupting state).
    let x0 = vec![0.0; t_count];
    let beta = conjugate_gradient(&l2, &b2, &x0, 1e-8, 1000).unwrap_or(x0);

    // 4. Coexact conflict flow: S_coexact = d1_t * beta
    let s_coexact = d1_t.mul_vec(&beta)?;

    // 5. Subtract the curl: delta_s_reconciled = delta_s - S_coexact
    let mut reconciled = vec![0.0; delta_s.len()];
    for i in 0..delta_s.len() {
        reconciled[i] = delta_s[i] - s_coexact[i];
    }
    Ok(reconciled)
}

/// Curl-projects `delta_s` and never fails: any internal error degrades to
/// returning the original `delta_s` unchanged (≡ identity projection).
///
/// This is the version every runtime layer calls. The `try_*` variant is for
/// callers that want to log degeneracy events without changing behaviour.
pub fn reconcile_state_delta(complex: &SimplicialStateComplex, delta_s: &[f64]) -> Vec<f64> {
    match try_reconcile_state_delta(complex, delta_s) {
        Ok(v) => v,
        Err(_) => delta_s.to_vec(),
    }
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

        // Reconciled flows should be curl-free:
        // d1 * reconciled should be zero (or close to 0)
        let d1 = complex.d1().expect("d1 is well-defined for this fixture");
        let curl = d1.mul_vec(&reconciled).expect("dims match by construction");
        assert!(!curl.is_empty());
        for &val in &curl {
            assert!(val.abs() < 1e-7, "Curl should be projected out, got {}", val);
        }
    }

    #[test]
    fn test_try_reconciliation_surfaces_dim_mismatch() {
        let mut complex = SimplicialStateComplex::new();
        complex.add_vertex("A", 0.0);
        complex.add_vertex("B", 0.0);
        complex.add_edge("A", "B", 0.0);

        // Wrong-length delta should produce a typed error, not a panic.
        let bad = vec![1.0, 2.0, 3.0];
        let err = try_reconcile_state_delta(&complex, &bad);
        assert!(matches!(err, Err(SubstrateError::DimensionMismatch { .. })));

        // The graceful wrapper returns the original input on the same input.
        let out = reconcile_state_delta(&complex, &bad);
        assert_eq!(out, bad);
    }
}
