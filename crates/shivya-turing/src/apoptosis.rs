use crate::morphogen::MorphogenSystem;

pub struct ApoptosisEngine {
    pub theta_apoptosis: f64,
}

impl ApoptosisEngine {
    pub fn new(theta_apoptosis: f64) -> Self {
        Self { theta_apoptosis }
    }

    // Prunes node index slots falling below utility thresholds to reclaim slot space
    pub fn evaluate_and_prune(
        &self,
        system: &mut MorphogenSystem,
        beliefs: &mut [Vec<f64>],
        adjacent_nodes: &mut [Vec<usize>],
        free_energies: &[f64],
        utility_threshold: f64,
    ) -> Option<usize> { // Returns Some(pruned_id)
        for i in 0..system.max_nodes {
            // We require minimum 3 nodes to maintain basic triangle topology integrity
            let active_count = system.active.iter().filter(|&&act| act).count();
            if active_count <= 3 {
                break;
            }

            if system.active[i] && system.u[i] < self.theta_apoptosis {
                // Check if utility is negative (which we map to high free energy above threshold)
                let local_f = free_energies.get(i).cloned().unwrap_or(0.0);
                if local_f > utility_threshold {
                    // 1. Decouple adjacent edges from matrix
                    for j in 0..system.max_nodes {
                        system.adj[i][j] = 0.0;
                        system.adj[j][i] = 0.0;
                    }

                    // 2. Mark index slot as dormant to reclaim memory space
                    system.active[i] = false;
                    system.u[i] = 0.0;
                    system.v[i] = 0.0;
                    beliefs[i].clear();

                    // 3. Update the global adjacent index cache
                    adjacent_nodes[i].clear();
                    for j in 0..system.max_nodes {
                        if let Some(pos) = adjacent_nodes[j].iter().position(|&id| id == i) {
                            adjacent_nodes[j].remove(pos);
                        }
                    }

                    return Some(i);
                }
            }
        }
        None
    }
}
