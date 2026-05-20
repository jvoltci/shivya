use shivya_morphic::DynamicGibbsAgent;
use crate::harsanyi::LocalCoalitionSolver;
use crate::field::OnsagerFlowRegulator;

pub struct OnsagerCollectiveEnsemble {
    pub agents: Vec<DynamicGibbsAgent>,
    // Adjacency lists: adjacent_nodes[i] lists neighbor indexes of agent i
    pub adjacent_nodes: Vec<Vec<usize>>,
    pub regulator: OnsagerFlowRegulator,
}

impl OnsagerCollectiveEnsemble {
    pub fn new(
        agents: Vec<DynamicGibbsAgent>,
        adjacent_nodes: Vec<Vec<usize>>,
        base_coupling: f64,
    ) -> Self {
        let num_nodes = agents.len();
        let regulator = OnsagerFlowRegulator::new(num_nodes, base_coupling);
        Self {
            agents,
            adjacent_nodes,
            regulator,
        }
    }

    // Exec one step of concurrent belief updating, parameter migration, and return F_collective
    pub fn step(
        &mut self,
        observations: &[Vec<f64>],
        lr: f64,
        max_iters: usize,
        tol: f64,
        flow_rate: f64,
    ) -> f64 {
        let n = self.agents.len();
        let mut free_energies = vec![0.0; n];

        // 1. Concurrently update agent beliefs based on local sensory observations
        for i in 0..n {
            free_energies[i] = self.agents[i].update_beliefs(&observations[i], lr, max_iters, tol);
        }

        // 2. Perform Onsager reciprocal parameter migration
        let mut beliefs: Vec<Vec<f64>> = self.agents.iter().map(|a| a.mu_q.clone()).collect();
        let flows = self.regulator.compute_parameter_flows(&beliefs);
        self.regulator.apply_parameter_migration(&mut beliefs, &flows, flow_rate);

        // Write the updated/migrated beliefs back to agents
        for i in 0..n {
            self.agents[i].mu_q = beliefs[i].clone();
        }

        // 3. Compute Collective Free Energy incorporating Harsanyi Dividends
        let total_f = free_energies.iter().sum::<f64>();
        let mut sum_dividends = 0.0;

        for i in 0..n {
            let solver = LocalCoalitionSolver::new(i, self.adjacent_nodes[i].clone());
            
            // Define local coalitional value V(C)
            // If nodes inside coalition are synergistic (close beliefs), they yield positive value.
            // If they are antagonistic (divergent beliefs), they yield negative value (repulsion).
            let val_fn = |mask: u8| -> f64 {
                let nodes = solver.mask_to_nodes(mask);
                if nodes.len() < 2 {
                    return 0.0;
                }
                
                let mut synergy = 0.0;
                for idx_a in 0..nodes.len() {
                    for idx_b in (idx_a + 1)..nodes.len() {
                        let node_a = nodes[idx_a];
                        let node_b = nodes[idx_b];
                        
                        let b_a = &beliefs[node_a];
                        let b_b = &beliefs[node_b];
                        
                        // Compute squared distance between belief coordinates
                        let len = b_a.len().min(b_b.len());
                        let dist_sq: f64 = b_a.iter().zip(b_b.iter())
                            .take(len)
                            .map(|(&x, &y)| (x - y).powi(2))
                            .sum();
                        
                        // Synergistic threshold of 1.0. If dist < 1.0, positive synergy. Else, negative.
                        if dist_sq < 1.0 {
                            synergy += 2.0 * (1.0 - dist_sq);
                        } else {
                            // Antagonistic repulsion scaling
                            synergy -= 4.0 * (dist_sq - 1.0);
                        }
                    }
                }
                synergy
            };

            let dividends = solver.calculate_dividends(val_fn);
            for (mask, div) in dividends {
                // Focus on cooperative interaction coalitions of size >= 2
                if mask.count_ones() >= 2 {
                    sum_dividends += div;
                }
            }
        }

        // F_collective = Sum(F_i) - Sum(Cooperative Dividends)
        // Highly synergistic systems subtract dividends (lower F_collective).
        // Antagonistic systems add penalty (raise F_collective).
        total_f - sum_dividends
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use shivya_morphic::DynamicGibbsAgent;

    fn create_agent(initial_belief: f64) -> DynamicGibbsAgent {
        DynamicGibbsAgent::new(
            2, 1, 2,
            vec![initial_belief, 0.0], // mu_prior
            vec![vec![10.0, 0.0], vec![0.0, 10.0]], // sigma_prior
            vec![vec![1.0, 0.0], vec![0.0, 1.0]], // g_s
            vec![vec![0.1, 0.0], vec![0.0, 0.1]], // sigma_s_0
            vec![vec![0.0], vec![0.0]], // w
            vec![vec![0.0], vec![0.0]], // m
            vec![0.0, 0.0], // mu_pref
            vec![vec![1.0, 0.0], vec![0.0, 1.0]], // sigma_pref
            5.0, // tau_novelty threshold
        )
    }

    #[test]
    fn test_synergy_vs_antagonism_energy_dynamics() {
        // --- CASE 1: Synergistic Coalition (agents have identical/close beliefs) ---
        let mut synergistic_ensemble = OnsagerCollectiveEnsemble::new(
            vec![create_agent(0.1), create_agent(0.1)],
            vec![vec![1], vec![0]], // Bidirectional adjacent connection
            0.5,
        );

        let obs = vec![vec![0.1, 0.0], vec![0.1, 0.0]];
        let f_synergy = synergistic_ensemble.step(&obs, 0.1, 10, 1e-4, 0.1);

        // --- CASE 2: Antagonistic Coalition (agents have highly divergent beliefs) ---
        let mut antagonistic_ensemble = OnsagerCollectiveEnsemble::new(
            vec![create_agent(0.1), create_agent(5.0)],
            vec![vec![1], vec![0]], // Connection
            0.5,
        );

        let obs_antag = vec![vec![0.1, 0.0], vec![5.0, 0.0]];
        let f_antag = antagonistic_ensemble.step(&obs_antag, 0.1, 10, 1e-4, 0.1);

        // Collective Free Energy for Case 2 should be significantly higher due to negative Harsanyi dividends / repulsion penalty!
        println!("F_synergy: {}, F_antagonism: {}", f_synergy, f_antag);
        assert!(f_antag > f_synergy, "Antagonistic system interaction must increase collective energy (repulsion), while synergistic interaction minimizes it.");
    }
}
