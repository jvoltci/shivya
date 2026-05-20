use wasm_bindgen::prelude::*;
use shivya_hodge::complex::SimplicialStateComplex;
use shivya_hodge::reconciler::reconcile_state_delta;
use shivya_flux::model::GibbsFluxAgent;
use shivya_morphic::{DynamicGibbsAgent, Expr, MorphicHotSwapper};
use shivya_onsager::OnsagerCollectiveEnsemble;

#[wasm_bindgen]
pub struct ShivyaSimulation {
    complex: SimplicialStateComplex,
    agent: GibbsFluxAgent<2, 1, 2>, // 2D sensory, 1D active, 2D internal
}

#[wasm_bindgen]
impl ShivyaSimulation {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        let mut complex = SimplicialStateComplex::new();
        // Set up initial simple topology
        complex.add_vertex("A", 10.0);
        complex.add_vertex("B", 10.0);
        complex.add_vertex("C", 10.0);
        complex.add_edge("A", "B", 1.0);
        complex.add_edge("B", "C", 1.0);
        complex.add_edge("A", "C", 1.5); // forms a triangle A-B-C

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

#[wasm_bindgen]
pub struct SubstrateOrchestrator {
    complex: SimplicialStateComplex,
    ensemble: OnsagerCollectiveEnsemble,
    swappers: Vec<MorphicHotSwapper>,
    step_count: usize,
}

#[wasm_bindgen]
impl SubstrateOrchestrator {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
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

        let agents = vec![
            create_agent(0.1),
            create_agent(0.2),
            create_agent(0.15),
        ];

        let ensemble = OnsagerCollectiveEnsemble::new(agents, adjacent_nodes, 0.5);

        let create_swapper = || {
            MorphicHotSwapper::new(Expr::Mul(
                Box::new(Expr::Const(1.0)),
                Box::new(Expr::Var(0)),
            ))
        };
        let swappers = vec![create_swapper(), create_swapper(), create_swapper()];

        Self {
            complex,
            ensemble,
            swappers,
            step_count: 0,
        }
    }

    pub fn step(&mut self, inputs: &[f64]) -> String {
        self.step_count += 1;
        let num_nodes = self.ensemble.agents.len();

        let mut obs = vec![vec![0.0, 0.0]; num_nodes];
        for i in 0..num_nodes {
            if 2 * i + 1 < inputs.len() {
                obs[i][0] = inputs[2 * i];
                obs[i][1] = inputs[2 * i + 1];
            }
        }

        let mut delta_s = vec![0.0; self.complex.edges.len()];
        for i in 0..delta_s.len() {
            if i < inputs.len() {
                delta_s[i] = inputs[i];
            }
        }
        let reconciled = reconcile_state_delta(&self.complex, &delta_s);
        let curl_deviation: f64 = reconciled.iter().zip(delta_s.iter())
            .map(|(&r, &d)| (r - d).powi(2))
            .sum::<f64>().sqrt();

        let collective_f = self.ensemble.step(&obs, 0.1, 10, 1e-4, 0.1);

        let mut hotswap_status = vec![false; num_nodes];
        let mut equations = vec![String::new(); num_nodes];

        for i in 0..num_nodes {
            let dataset = vec![
                (vec![obs[i][0]], obs[i][0] * 1.5),
                (vec![obs[i][1]], obs[i][1] * 1.5),
            ];
            let seed = (self.step_count + i * 99) as u32;
            let swapped = self.swappers[i].run_metamorphic_step(&dataset, seed);
            hotswap_status[i] = swapped;
            equations[i] = format!("{:?}", self.swappers[i].current_expr);
        }

        let mut json = String::new();
        json.push_str("{\n");
        json.push_str(&format!("  \"collective_free_energy\": {:.4},\n", collective_f));
        json.push_str(&format!("  \"curl_deviation\": {:.4},\n", curl_deviation));
        json.push_str("  \"nodes\": [\n");

        for i in 0..num_nodes {
            let agent = &self.ensemble.agents[i];
            let beliefs_str = agent.mu_q.iter()
                .map(|val| format!("{:.4}", val))
                .collect::<Vec<String>>()
                .join(", ");

            json.push_str("    {\n");
            json.push_str(&format!("      \"id\": {},\n", i));
            json.push_str(&format!("      \"free_energy\": {:.4},\n", agent.f_history.last().cloned().unwrap_or(0.0)));
            json.push_str(&format!("      \"belief_dim\": {},\n", agent.i_dim));
            json.push_str(&format!("      \"beliefs\": [{}],\n", beliefs_str));
            json.push_str(&format!("      \"morphic_equation\": \"{}\",\n", equations[i].replace("\"", "\\\"")));
            json.push_str(&format!("      \"hotswapped\": {}\n", hotswap_status[i]));

            if i < num_nodes - 1 {
                json.push_str("    },\n");
            } else {
                json.push_str("    }\n");
            }
        }
        json.push_str("  ]\n");
        json.push_str("}");

        json
    }
}
