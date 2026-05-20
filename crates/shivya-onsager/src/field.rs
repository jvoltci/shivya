pub struct OnsagerFlowRegulator {
    pub num_nodes: usize,
    // Symmetric phenomenological coupling matrix L_ij = L_ji
    pub l_matrix: Vec<Vec<f64>>,
}

impl OnsagerFlowRegulator {
    pub fn new(num_nodes: usize, base_coupling: f64) -> Self {
        let mut l_matrix = vec![vec![0.0; num_nodes]; num_nodes];
        // Initialize with base coupling symmetric coefficients
        for i in 0..num_nodes {
            for j in 0..num_nodes {
                if i != j {
                    l_matrix[i][j] = base_coupling;
                }
            }
        }
        Self { num_nodes, l_matrix }
    }

    // Calculates parameter transport flow J_p(i -> j) = L_ij * (mu_i - mu_j)
    pub fn compute_parameter_flows(&self, beliefs: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let mut flows = vec![vec![0.0; self.num_nodes]; self.num_nodes];
        for i in 0..self.num_nodes {
            for j in (i + 1)..self.num_nodes {
                // Ensure nodes have compatible dimensionality
                let len = beliefs[i].len().min(beliefs[j].len());
                let mut force = 0.0;
                for k in 0..len {
                    force += beliefs[i][k] - beliefs[j][k];
                }
                // Reciprocal coupling
                let l_ij = self.l_matrix[i][j];
                let flow = l_ij * force;
                flows[i][j] = flow;
                flows[j][i] = -flow; // Onsager Reciprocity constraint
            }
        }
        flows
    }

    // Calculates workload/pressure flow J_w(i -> j) = L_ij * (F_i - F_j)
    pub fn compute_workload_flows(&self, free_energies: &[f64]) -> Vec<Vec<f64>> {
        let mut flows = vec![vec![0.0; self.num_nodes]; self.num_nodes];
        for i in 0..self.num_nodes {
            for j in (i + 1)..self.num_nodes {
                let force = free_energies[i] - free_energies[j];
                let l_ij = self.l_matrix[i][j];
                let flow = l_ij * force;
                flows[i][j] = flow;
                flows[j][i] = -flow; // Reciprocal flow
            }
        }
        flows
    }

    // Applies flows to balance parameters in-place
    pub fn apply_parameter_migration(&self, beliefs: &mut [Vec<f64>], flows: &[Vec<f64>], rate: f64) {
        for i in 0..self.num_nodes {
            for j in 0..self.num_nodes {
                if i != j {
                    let flow = flows[i][j];
                    let len = beliefs[i].len();
                    for k in 0..len {
                        beliefs[i][k] -= rate * flow / (len as f64);
                    }
                }
            }
        }
    }
}
