use wasm_bindgen::prelude::*;
use shivya_hodge::complex::SimplicialStateComplex;

/// Physics-derived strain proxy used when no live telemetry is supplied.
///
/// In a browser the host's CPU is sandboxed away, so we fall back to a
/// deterministic Gierer–Meinhardt-coupled signal: strain rises with the local
/// activator `u` and is damped by the inhibitor `v`. Native (rlib) builds use
/// the exact same formula so unit tests remain reproducible.
#[cfg(target_arch = "wasm32")]
fn derived_strain(step: usize, node: usize, u: f64, v: f64) -> Vec<f64> {
    // Bounded, deterministic, monotone in u and inverse in v.
    let phase = (step as f64) * 0.05 + (node as f64) * 0.3;
    let activator_strain = (u / (1.0 + v.abs())).clamp(0.0, 1.0);
    let drift = 0.5 + 0.25 * phase.sin();
    let primary = 0.4 * activator_strain + 0.6 * drift;
    vec![primary.clamp(0.0, 1.0), (primary * 0.85).clamp(0.0, 1.0)]
}

#[cfg(not(target_arch = "wasm32"))]
fn derived_strain(step: usize, node: usize, u: f64, v: f64) -> Vec<f64> {
    let phase = (step as f64) * 0.05 + (node as f64) * 0.3;
    let activator_strain = (u / (1.0 + v.abs())).clamp(0.0, 1.0);
    let drift = 0.5 + 0.25 * phase.sin();
    let primary = 0.4 * activator_strain + 0.6 * drift;
    vec![primary.clamp(0.0, 1.0), (primary * 0.85).clamp(0.0, 1.0)]
}
use shivya_hodge::reconciler::reconcile_state_delta;
use shivya_flux::model::GibbsFluxAgent;
use shivya_morphic::{DynamicGibbsAgent, Expr, MorphicHotSwapper, compile};
use shivya_onsager::OnsagerCollectiveEnsemble;
use shivya_turing::{MorphogenSystem, MitosisEngine, ApoptosisEngine};

mod mind;
use crate::mind::MindCore;

// Keeps ShivyaSimulation backward-compatible for safety
#[wasm_bindgen]
pub struct ShivyaSimulation {
    complex: SimplicialStateComplex,
    agent: GibbsFluxAgent<2, 1, 2>,
}

impl Default for ShivyaSimulation {
    fn default() -> Self {
        Self::new()
    }
}

#[wasm_bindgen]
impl ShivyaSimulation {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        let mut complex = SimplicialStateComplex::new();
        complex.add_vertex("A", 10.0);
        complex.add_vertex("B", 10.0);
        complex.add_vertex("C", 10.0);
        complex.add_edge("A", "B", 1.0);
        complex.add_edge("B", "C", 1.0);
        complex.add_edge("A", "C", 1.5);

        let agent = GibbsFluxAgent::new(
            [0.0, 0.0],
            [[10.0, 0.0], [0.0, 10.0]],
            [[2.0, 0.5], [0.5, 1.5]],
            [[0.1, 0.0], [0.0, 0.1]],
            [[1.0], [0.5]],
            [[1.0], [0.5]],
            [0.0, 0.0],
            [[1.0, 0.0], [0.0, 1.0]],
        );

        Self { complex, agent }
    }

    pub fn add_vertex(&mut self, label: &str, initial_state: f64) -> usize {
        self.complex.add_vertex(label, initial_state)
    }

    pub fn add_edge(&mut self, u_label: &str, v_label: &str, initial_state: f64) {
        self.complex.add_edge(u_label, v_label, initial_state);
    }

    pub fn get_vertices_count(&self) -> usize {
        self.complex.vertices.len()
    }

    pub fn get_edges_count(&self) -> usize {
        self.complex.edges.len()
    }

    pub fn get_triangles_count(&self) -> usize {
        self.complex.triangles.len()
    }

    pub fn get_vertex_label(&self, idx: usize) -> String {
        self.complex.vertices.get(idx).cloned().unwrap_or_default()
    }

    pub fn get_vertex_state(&self, idx: usize) -> f64 {
        self.complex.vertex_states.get(idx).cloned().unwrap_or(0.0)
    }

    pub fn get_edge_u(&self, idx: usize) -> usize {
        self.complex.edges.get(idx).map(|&(u, _)| u).unwrap_or(0)
    }

    pub fn get_edge_v(&self, idx: usize) -> usize {
        self.complex.edges.get(idx).map(|&(_, v)| v).unwrap_or(0)
    }

    pub fn get_edge_state(&self, idx: usize) -> f64 {
        self.complex.edge_states.get(idx).cloned().unwrap_or(0.0)
    }

    pub fn reconcile_flows(&self, delta_s: Vec<f64>) -> Vec<f64> {
        reconcile_state_delta(&self.complex, &delta_s)
    }

    pub fn agent_update_beliefs(&mut self, obs_0: f64, obs_1: f64) -> Vec<f64> {
        self.agent.update_beliefs(&[obs_0, obs_1], 0.1, 100, 1e-6)
    }

    pub fn agent_free_energy(&self, obs_0: f64, obs_1: f64) -> f64 {
        self.agent.compute_free_energy(&[obs_0, obs_1], &self.agent.mu_q, &self.agent.sigma_q)
    }

    pub fn get_agent_beliefs(&self) -> Vec<f64> {
        self.agent.mu_q.to_vec()
    }
}

// Full 5-Layer browser WebAssembly orchestrator
#[wasm_bindgen]
pub struct SubstrateOrchestrator {
    max_nodes: usize,
    complex: SimplicialStateComplex,
    ensemble: OnsagerCollectiveEnsemble,
    swappers: Vec<MorphicHotSwapper>,
    turing: MorphogenSystem,
    mitosis: MitosisEngine,
    apoptosis: ApoptosisEngine,
    step_count: usize,
    mind: MindCore,
}

impl Default for SubstrateOrchestrator {
    fn default() -> Self {
        Self::new()
    }
}

#[wasm_bindgen]
impl SubstrateOrchestrator {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        let max_nodes = 10;
        let mut complex = SimplicialStateComplex::new();
        // Setup initial simplicial mesh
        complex.add_vertex("Node0", 1.0);
        complex.add_vertex("Node1", 1.2);
        complex.add_vertex("Node2", 0.9);
        complex.add_edge("Node0", "Node1", 0.5);
        complex.add_edge("Node1", "Node2", 0.6);
        complex.add_edge("Node0", "Node2", 0.8);

        let adjacent_nodes = vec![
            vec![1, 2],
            vec![0, 2],
            vec![0, 1],
            vec![], // Slot 3 dormant
            vec![], // Slot 4 dormant
            vec![], // Slot 5 dormant
            vec![], // Slot 6 dormant
            vec![], // Slot 7 dormant
            vec![], // Slot 8 dormant
            vec![], // Slot 9 dormant
        ];

        let create_agent = |mu_prior_val: f64| {
            DynamicGibbsAgent::new(
                2, 1, 2,
                vec![mu_prior_val, 0.0],
                vec![vec![10.0, 0.0], vec![0.0, 10.0]],
                vec![vec![1.5, 0.2], vec![0.2, 1.2]],
                vec![vec![0.1, 0.0], vec![0.0, 0.1]],
                vec![vec![0.0], vec![0.0]],
                vec![vec![0.0], vec![0.0]],
                vec![0.0, 0.0],
                vec![vec![1.0, 0.0], vec![0.0, 1.0]],
                5.0,
            )
        };

        let mut agents = Vec::new();
        for i in 0..max_nodes {
            agents.push(create_agent(0.1 + (i as f64) * 0.05));
        }

        let base_coupling = 0.5;
        let ensemble = OnsagerCollectiveEnsemble::new(agents, adjacent_nodes, base_coupling);

        let create_swapper = || {
            MorphicHotSwapper::new(Expr::Mul(
                Box::new(Expr::Const(1.0)),
                Box::new(Expr::Var(0)),
            ))
        };
        let mut swappers = Vec::new();
        for _ in 0..max_nodes {
            swappers.push(create_swapper());
        }

        let mut turing = MorphogenSystem::new(max_nodes, 0.01, 0.1);
        turing.activate_node(0, 0.5, 1.0);
        turing.activate_node(1, 0.2, 1.0);
        turing.activate_node(2, 0.3, 1.0);
        turing.set_edge(0, 1, 1.0);
        turing.set_edge(1, 2, 1.0);
        turing.set_edge(0, 2, 1.0);

        let mitosis = MitosisEngine::new(2.0, 0.01);
        let apoptosis = ApoptosisEngine::new(0.05);

        Self {
            max_nodes,
            complex,
            ensemble,
            swappers,
            turing,
            mitosis,
            apoptosis,
            step_count: 0,
            mind: MindCore::new(),
        }
    }

    pub fn reset(&mut self) {
        let mut complex = SimplicialStateComplex::new();
        complex.add_vertex("Node0", 1.0);
        complex.add_vertex("Node1", 1.2);
        complex.add_vertex("Node2", 0.9);
        complex.add_edge("Node0", "Node1", 0.5);
        complex.add_edge("Node1", "Node2", 0.6);
        complex.add_edge("Node0", "Node2", 0.8);

        let adjacent_nodes = vec![
            vec![1, 2],
            vec![0, 2],
            vec![0, 1],
            vec![], vec![], vec![], vec![], vec![], vec![], vec![],
        ];

        let create_agent = |mu_prior_val: f64| {
            DynamicGibbsAgent::new(
                2, 1, 2,
                vec![mu_prior_val, 0.0],
                vec![vec![10.0, 0.0], vec![0.0, 10.0]],
                vec![vec![1.5, 0.2], vec![0.2, 1.2]],
                vec![vec![0.1, 0.0], vec![0.0, 0.1]],
                vec![vec![0.0], vec![0.0]],
                vec![vec![0.0], vec![0.0]],
                vec![0.0, 0.0],
                vec![vec![1.0, 0.0], vec![0.0, 1.0]],
                5.0,
            )
        };

        let mut agents = Vec::new();
        for i in 0..self.max_nodes {
            agents.push(create_agent(0.1 + (i as f64) * 0.05));
        }

        let ensemble = OnsagerCollectiveEnsemble::new(agents, adjacent_nodes, 0.5);

        let create_swapper = || {
            MorphicHotSwapper::new(Expr::Mul(
                Box::new(Expr::Const(1.0)),
                Box::new(Expr::Var(0)),
            ))
        };
        let mut swappers = Vec::new();
        for _ in 0..self.max_nodes {
            swappers.push(create_swapper());
        }

        let mut turing = MorphogenSystem::new(self.max_nodes, 0.01, 0.1);
        turing.activate_node(0, 0.5, 1.0);
        turing.activate_node(1, 0.2, 1.0);
        turing.activate_node(2, 0.3, 1.0);
        turing.set_edge(0, 1, 1.0);
        turing.set_edge(1, 2, 1.0);
        turing.set_edge(0, 2, 1.0);

        self.complex = complex;
        self.ensemble = ensemble;
        self.swappers = swappers;
        self.turing = turing;
        self.step_count = 0;
        self.mind = MindCore::new();
    }

    pub fn inject_stress(&mut self, node_id: usize) -> bool {
        if node_id < self.max_nodes && self.turing.active[node_id] {
            // Force morphogen u high above mitosis threshold (2.0)
            self.turing.u[node_id] = 2.5;
            true
        } else {
            false
        }
    }

    pub fn trigger_apoptosis(&mut self, node_id: usize) -> bool {
        if node_id < self.max_nodes && self.turing.active[node_id] {
            // Force morphogen u very low (0.01) and spike the agent's free energy in history
            self.turing.u[node_id] = 0.01;
            self.turing.v[node_id] = 2.0;
            self.ensemble.agents[node_id].f_history.push(8.5); // Spike free energy
            true
        } else {
            false
        }
    }

    pub fn step(&mut self, inputs: &[f64]) -> String {
        self.step_count += 1;

        // 1. Gather active indices
        let mut active_indices = Vec::new();
        for i in 0..self.max_nodes {
            if self.turing.active[i] {
                active_indices.push(i);
            }
        }

        // 2. Scale coupling coefficients based on network parameters
        let base_coupling = 0.5;
        for i in 0..self.max_nodes {
            for j in 0..self.max_nodes {
                if i != j {
                    self.ensemble.regulator.l_matrix[i][j] = base_coupling;
                }
            }
        }

        // CPU observations come from one of two sources:
        //   - real `inputs` (callers wired to actual JS-side telemetry), or
        //   - a deterministic physics-derived strain model when no inputs are
        //     supplied. The browser sandbox forbids reading host CPU, so a
        //     reproducible diffusion-driven proxy stands in.
        let mut obs = vec![vec![0.0, 0.0]; self.max_nodes];
        for &i in &active_indices {
            if 2 * i + 1 < inputs.len() {
                obs[i][0] = inputs[2 * i];
                obs[i][1] = inputs[2 * i + 1];
            } else {
                obs[i] = derived_strain(self.step_count, i, self.turing.u[i], self.turing.v[i]);
            }
        }

        // Step Onsager Collective Ensemble (Layer 3 & 1)
        let collective_f = self.ensemble.step(&obs, 0.1, 10, 1e-4, 0.1);

        // Cognitive core: bucket the collective free energy into a small
        // categorical and feed it as a (collective, free_energy, bucket_b<k>)
        // event. Eight buckets cover the dynamic range we see in the demo
        // without inflating the codebook vocabulary every step.
        let bucket = (collective_f.abs().min(8.0)).floor() as i32;
        let bucket_label = format!("bucket_b{}", bucket);
        self.mind.observe("collective", "free_energy", &bucket_label);
        let mind_self_sim = self.mind.self_similarity("collective", "free_energy", &bucket_label);
        let mind_events = self.mind.events_ingested();
        let mind_in_episode = self.mind.event_count_in_episode();
        let mind_signature = self.mind.signature_hex();

        // 4. Morphic Hot-swapping VM updates (Layer 2)
        for &i in &active_indices {
            let dataset = vec![
                (vec![obs[i][0]], obs[i][0] * 1.5),
                (vec![obs[i][1]], obs[i][1] * 1.5),
            ];
            let seed = (self.step_count + i * 99) as u32;
            self.swappers[i].run_metamorphic_step(&dataset, seed);
        }

        // 5. Reconcile topological states (Layer 0)
        let mut delta_s = vec![0.0; self.complex.edges.len()];
        // Fill delta_s with small perturbations from observations
        for i in 0..delta_s.len() {
            let u_idx = self.complex.edges[i].0;
            let val = if u_idx < obs.len() { obs[u_idx][0] } else { 0.5 };
            delta_s[i] = val * 0.15;
        }
        let reconciled = reconcile_state_delta(&self.complex, &delta_s);
        let curl_deviation: f64 = reconciled.iter().zip(delta_s.iter())
            .map(|(&r, &d)| (r - d).powi(2))
            .sum::<f64>().sqrt();

        // 6. Gierer-Meinhardt reaction diffusion step (Layer 4)
        self.turing.step_rk4(0.05);

        // Gather beliefs & adjacency for Mitosis/Apoptosis
        let mut beliefs: Vec<Vec<f64>> = self.ensemble.agents.iter().map(|a| a.mu_q.clone()).collect();
        let mut adjacent_nodes = self.ensemble.adjacent_nodes.clone();

        // 7. Mitosis Engine split evaluation
        let mut hotswap_status = vec![false; self.max_nodes];
        if let Some((parent, child)) = self.mitosis.evaluate_and_split(&mut self.turing, &mut beliefs, &mut adjacent_nodes) {
            // Apply new beliefs to child agent
            self.ensemble.agents[child].mu_q = beliefs[child].clone();
            self.ensemble.agents[child].i_dim = beliefs[child].len();
            // Sync adjacency list
            self.ensemble.adjacent_nodes = adjacent_nodes.clone();
            // Update Hodge Mesh topology
            let child_label = format!("Node{}", child);
            self.complex.add_vertex(&child_label, 1.0);
            let parent_label = format!("Node{}", parent);
            self.complex.add_edge(&parent_label, &child_label, 1.0);
            hotswap_status[child] = true; // Mark as newly hotswapped/split
        }

        // 8. Apoptosis Engine pruning evaluation
        let mut free_energies = vec![0.0; self.max_nodes];
        for i in 0..self.max_nodes {
            free_energies[i] = self.ensemble.agents[i].f_history.last().cloned().unwrap_or(0.0);
        }
        if let Some(_pruned_node) = self.apoptosis.evaluate_and_prune(&mut self.turing, &mut beliefs, &mut adjacent_nodes, &free_energies, 5.0) {
            // Sync changes back
            self.ensemble.adjacent_nodes = adjacent_nodes;
        }

        // 9. Re-collect updated active indices
        let mut updated_active = Vec::new();
        for i in 0..self.max_nodes {
            if self.turing.active[i] {
                updated_active.push(i);
            }
        }

        // 10. Compile equations and build status JSON
        let mut equations = vec![String::new(); self.max_nodes];
        let mut inst_counts = vec![0; self.max_nodes];
        for i in 0..self.max_nodes {
            let expr = &self.swappers[i].current_expr;
            equations[i] = format!("{:?}", expr);
            let (insts, _) = compile(expr);
            inst_counts[i] = insts.len();
        }

        let mut edges = Vec::new();
        for &u in &updated_active {
            if u < self.ensemble.adjacent_nodes.len() {
                for &v in &self.ensemble.adjacent_nodes[u] {
                    if u < v && updated_active.contains(&v) {
                        edges.push((u, v));
                    }
                }
            }
        }

        let mut json = String::new();
        json.push_str("{\n");
        json.push_str(&format!("  \"collective_free_energy\": {:.6},\n", collective_f));
        json.push_str(&format!("  \"curl_deviation\": {:.6},\n", curl_deviation));
        json.push_str(&format!("  \"step_count\": {},\n", self.step_count));
        json.push_str(&format!("  \"active_nodes_count\": {},\n", updated_active.len()));
        json.push_str(&format!("  \"mind_events_ingested\": {},\n", mind_events));
        json.push_str(&format!("  \"mind_event_count_in_episode\": {},\n", mind_in_episode));
        json.push_str(&format!("  \"mind_self_similarity\": {:.6},\n", mind_self_sim));
        json.push_str(&format!("  \"mind_signature_hex\": \"{}\",\n", mind_signature));

        json.push_str("  \"active_pool\": [");
        let active_pool_str = updated_active.iter().map(|idx| idx.to_string()).collect::<Vec<String>>().join(", ");
        json.push_str(&active_pool_str);
        json.push_str("],\n");

        json.push_str("  \"edges\": [");
        let edges_str = edges.iter().map(|&(u, v)| format!("[{}, {}]", u, v)).collect::<Vec<String>>().join(", ");
        json.push_str(&edges_str);
        json.push_str("],\n");

        json.push_str("  \"nodes\": [\n");
        for (i, &idx) in updated_active.iter().enumerate() {
            let agent = &self.ensemble.agents[idx];
            let beliefs_str = agent.mu_q.iter()
                .map(|val| format!("{:.6}", val))
                .collect::<Vec<String>>()
                .join(", ");

            json.push_str("    {\n");
            json.push_str(&format!("      \"id\": {},\n", idx));
            json.push_str("      \"active\": true,\n");
            json.push_str(&format!("      \"free_energy\": {:.6},\n", agent.f_history.last().cloned().unwrap_or(0.0)));
            json.push_str(&format!("      \"belief_dim\": {},\n", agent.i_dim));
            json.push_str(&format!("      \"beliefs\": [{}],\n", beliefs_str));
            json.push_str(&format!("      \"morphic_equation\": \"{}\",\n", equations[idx].replace("\"", "\\\"")));
            json.push_str(&format!("      \"instruction_count\": {},\n", inst_counts[idx]));
            json.push_str(&format!("      \"morphogen_u\": {:.6},\n", self.turing.u[idx]));
            json.push_str(&format!("      \"morphogen_v\": {:.6},\n", self.turing.v[idx]));
            json.push_str(&format!("      \"hotswapped\": {}\n", hotswap_status[idx]));

            if i < updated_active.len() - 1 {
                json.push_str("    },\n");
            } else {
                json.push_str("    }\n");
            }
        }
        json.push_str("  ]\n");
        json.push('}');

        json
    }
}
