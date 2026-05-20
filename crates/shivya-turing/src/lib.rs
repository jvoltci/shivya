pub mod morphogen;
pub mod mitosis;
pub mod apoptosis;

pub use morphogen::MorphogenSystem;
pub use mitosis::MitosisEngine;
pub use apoptosis::ApoptosisEngine;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_morphogen_cfl_guard() {
        let mut sys = MorphogenSystem::new(8, 0.1, 1.0);
        sys.activate_node(0, 1.0, 1.0);
        sys.activate_node(1, 1.0, 1.0);
        sys.set_edge(0, 1, 5.0); // High degree edge

        let dt_orig = 0.1;
        let dt_clamped = sys.get_cfl_dt(dt_orig);
        assert!(dt_clamped < dt_orig, "CFL Stability Guard must clamp time-step for high-degree systems to avoid NaN drift.");
    }

    #[test]
    fn test_mitosis_zero_allocation_split() {
        let max_nodes = 5;
        let mut system = MorphogenSystem::new(max_nodes, 0.01, 0.1);
        
        // Activate 3 initial nodes (triangle)
        system.activate_node(0, 0.5, 1.0);
        system.activate_node(1, 0.2, 1.0);
        system.activate_node(2, 0.3, 1.0);
        system.set_edge(0, 1, 1.0);
        system.set_edge(1, 2, 1.0);
        system.set_edge(0, 2, 1.0);

        // Pre-allocated beliefs and adjacent list cache
        let mut beliefs = vec![
            vec![0.1, 0.2],
            vec![0.5, -0.1],
            vec![0.3, 0.0],
            vec![], // dormant slot 3
            vec![], // dormant slot 4
        ];

        let mut adjacent_nodes = vec![
            vec![1, 2],
            vec![0, 2],
            vec![0, 1],
            vec![], // dormant
            vec![], // dormant
        ];

        // Trigger stress threshold on node 0
        system.u[0] = 2.5; // High stress

        let engine = MitosisEngine::new(2.0, 0.01);
        let res = engine.evaluate_and_split(&mut system, &mut beliefs, &mut adjacent_nodes);

        assert!(res.is_some(), "Mitosis should trigger when activator concentration exceeds theta_mitosis.");
        let (parent, child) = res.unwrap();
        assert_eq!(parent, 0);
        assert_eq!(child, 3); // Woke up first dormant slot
        assert!(system.active[3], "Mitosis must activate the dormant index slot.");
        
        // Assert beliefs inherited and perturbed by epsilon noise
        assert!((beliefs[3][0] - 0.11).abs() < 1e-6);
        assert!((beliefs[0][0] - 0.09).abs() < 1e-6);

        // Verify adjacency matrix updated in O(1)
        assert_eq!(system.adj[0][3], 1.0, "Mother and daughter nodes must be strongly coupled.");
        assert_eq!(system.adj[3][1], 1.0, "Daughter must inherit connections of the mother node.");
    }

    #[test]
    fn test_apoptosis_pruning() {
        let max_nodes = 5;
        let mut system = MorphogenSystem::new(max_nodes, 0.01, 0.1);
        
        system.activate_node(0, 1.0, 1.0);
        system.activate_node(1, 1.0, 1.0);
        system.activate_node(2, 1.0, 1.0);
        system.activate_node(3, 0.01, 1.0); // Low activator / low utility node
        
        system.set_edge(0, 1, 1.0);
        system.set_edge(1, 2, 1.0);
        system.set_edge(0, 2, 1.0);
        system.set_edge(0, 3, 0.5);

        let mut beliefs = vec![vec![0.0], vec![0.0], vec![0.0], vec![0.5], vec![]];
        let mut adjacent_nodes = vec![vec![1, 2, 3], vec![0, 2], vec![0, 1], vec![0], vec![]];
        let free_energies = vec![0.1, 0.1, 0.1, 8.5, 0.0]; // Node 3 has very high free energy (negative utility)

        let engine = ApoptosisEngine::new(0.05);
        let res = engine.evaluate_and_prune(&mut system, &mut beliefs, &mut adjacent_nodes, &free_energies, 5.0);

        assert_eq!(res, Some(3), "Apoptosis engine must cull node 3 due to low activator stress and negative utility.");
        assert!(!system.active[3], "Pruned index slot must be marked as dormant.");
        assert_eq!(system.adj[0][3], 0.0, "Adjacent connections to pruned node must be severed.");
    }
}
