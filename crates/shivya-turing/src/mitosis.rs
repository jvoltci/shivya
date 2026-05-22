use crate::morphogen::MorphogenSystem;

pub struct MitosisEngine {
    pub theta_mitosis: f64,
    pub epsilon: f64,
}

impl MitosisEngine {
    pub fn new(theta_mitosis: f64, epsilon: f64) -> Self {
        Self {
            theta_mitosis,
            epsilon,
        }
    }

    // Executes vertex split on any active node exceeding the mitosis threshold
    // Returns the ID of the new activated node if split occurred
    pub fn evaluate_and_split(
        &self,
        system: &mut MorphogenSystem,
        beliefs: &mut [Vec<f64>],
        adjacent_nodes: &mut [Vec<usize>],
    ) -> Option<(usize, usize)> { // Returns Some((parent_id, child_id))
        for i in 0..system.max_nodes {
            if system.active[i] && system.u[i] > self.theta_mitosis {
                // Find a dormant slot in our pre-allocated index pool
                if let Some(d) = system.active.iter().position(|&active| !active) {
                    // Activate the dormant slot (Mitosis)
                    system.active[d] = true;
                    // Inherit morphogen values with small symmetry-breaking perturbation
                    system.u[d] = system.u[i] - self.epsilon;
                    system.u[i] += self.epsilon;

                    system.v[d] = system.v[i];

                    // Copy and perturb the internal belief vectors
                    if !beliefs[i].is_empty() {
                        beliefs[d] = beliefs[i].clone();
                        beliefs[i][0] -= self.epsilon;
                        beliefs[d][0] += self.epsilon;
                    }

                    // Rewrite adjacency pointers in O(1) time
                    // 1. Establish strong mother-daughter coupling link
                    system.set_edge(i, d, 1.0);
                    
                    // 2. Clone adjacent neighborhood connectivity
                    for j in 0..system.max_nodes {
                        if j != d && j != i && system.active[j] && system.adj[i][j] > 0.0 {
                            let weight = system.adj[i][j];
                            system.set_edge(d, j, weight);
                        }
                    }

                    // 3. Update the global adjacent index cache
                    adjacent_nodes[d].clear();
                    adjacent_nodes[d].push(i);
                    adjacent_nodes[i].push(d);

                    for j in 0..system.max_nodes {
                        if j != d && j != i && system.active[j] && system.adj[i][j] > 0.0 {
                            adjacent_nodes[d].push(j);
                            adjacent_nodes[j].push(d);
                        }
                    }

                    return Some((i, d));
                }
            }
        }
        None
    }
}
