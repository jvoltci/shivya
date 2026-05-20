use crate::operators::SparseMatrix;

pub fn conjugate_gradient(
    a_mat: &SparseMatrix,
    b: &[f64],
    x0: &[f64],
    tol: f64,
    max_iters: usize,
) -> Result<Vec<f64>, String> {
    let n = b.len();
    assert_eq!(a_mat.rows, n, "Matrix rows must match b length");
    assert_eq!(a_mat.cols, n, "Matrix must be square");
    assert_eq!(x0.len(), n, "x0 must match b length");

    let mut x = x0.to_vec();
    let ax = a_mat.mul_vec(&x);
    let mut r = vec![0.0; n];
    for i in 0..n {
        r[i] = b[i] - ax[i];
    }

    let mut r_sq_norm = r.iter().map(|&val| val * val).sum::<f64>();
    if r_sq_norm < tol {
        return Ok(x);
    }

    let mut p = r.clone();

    for _ in 0..max_iters {
        let ap = a_mat.mul_vec(&p);
        let p_ap = p.iter().zip(ap.iter()).map(|(&pi, &api)| pi * api).sum::<f64>();
        
        if p_ap.abs() < 1e-14 {
            break;
        }

        let alpha = r_sq_norm / p_ap;

        for i in 0..n {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }

        let next_r_sq_norm = r.iter().map(|&val| val * val).sum::<f64>();
        if next_r_sq_norm < tol {
            return Ok(x);
        }

        let beta = next_r_sq_norm / r_sq_norm;
        for i in 0..n {
            p[i] = r[i] + beta * p[i];
        }

        r_sq_norm = next_r_sq_norm;
    }

    Ok(x)
}
