use shivya::hodge::complex::SimplicialStateComplex;
use shivya::hodge::reconciler::reconcile_state_delta;
use shivya::morphic::{DynamicGibbsAgent, Expr, MorphicHotSwapper, compile};
use shivya::onsager::OnsagerCollectiveEnsemble;
use shivya::turing::{MorphogenSystem, MitosisEngine, ApoptosisEngine};
use serde::{Serialize, Deserialize};
use std::sync::Arc;
use tokio::sync::Mutex;
use shivya_p2p::routing::{NodeId, KBucketTable};
use shivya_p2p::transport::UdpTransport;
use shivya_p2p::protocol::{Frame, FramePayload};

fn lead_zeros_bytes(dist: &[u8; 20]) -> usize {
    let mut count = 0;
    for &byte in dist {
        if byte == 0 {
            count += 8;
        } else {
            count += byte.leading_zeros() as usize;
            break;
        }
    }
    count
}

#[derive(Serialize, Deserialize, Clone)]
pub struct NodeStatus {
    pub id: usize,
    pub active: bool,
    pub free_energy: f64,
    pub belief_dim: usize,
    pub beliefs: Vec<f64>,
    pub morphic_equation: String,
    pub instruction_count: usize,
    pub morphogen_u: f64,
    pub morphogen_v: f64,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct SystemStatus {
    pub collective_free_energy: f64,
    pub curl_deviation: f64,
    pub active_nodes_count: usize,
    pub active_pool: Vec<usize>,
    pub nodes: Vec<NodeStatus>,
    pub step_count: usize,
    pub edges: Vec<(usize, usize)>,
}

pub struct NativeOrchestrator {
    pub max_nodes: usize,
    pub complex: SimplicialStateComplex,
    pub ensemble: OnsagerCollectiveEnsemble,
    pub swappers: Vec<MorphicHotSwapper>,
    pub turing: MorphogenSystem,
    pub mitosis: MitosisEngine,
    pub apoptosis: ApoptosisEngine,
    pub step_count: usize,
    pub last_status: SystemStatus,
    
    // Phase 11 P2P Fields
    pub self_id: NodeId,
    pub p2p_table: Option<Arc<Mutex<KBucketTable>>>,
    pub p2p_transport: Option<Arc<UdpTransport>>,
}

impl NativeOrchestrator {
    pub fn new(max_nodes: usize) -> Self {
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

        let last_status = SystemStatus {
            collective_free_energy: 0.0,
            curl_deviation: 0.0,
            active_nodes_count: 3,
            active_pool: vec![0, 1, 2],
            nodes: Vec::new(),
            step_count: 0,
            edges: vec![(0, 1), (1, 2), (0, 2)],
        };

        let self_id = NodeId::random();

        Self {
            max_nodes,
            complex,
            ensemble,
            swappers,
            turing,
            mitosis,
            apoptosis,
            step_count: 0,
            last_status,
            self_id,
            p2p_table: None,
            p2p_transport: None,
        }
    }

    pub fn set_p2p(
        &mut self,
        self_id: NodeId,
        p2p_table: Arc<Mutex<KBucketTable>>,
        p2p_transport: Arc<UdpTransport>,
    ) {
        self.self_id = self_id;
        self.p2p_table = Some(p2p_table);
        self.p2p_transport = Some(p2p_transport);
    }

    pub fn step(&mut self, cpu_load: f64, net_rate: f64) {
        self.step_count += 1;

        // 0. Sync K-bucket peers to Layer 0 Hodge Simplicial Complex and Onsager connections
        if let Some(ref table) = self.p2p_table {
            if let Ok(table_lock) = table.try_lock() {
                let peers = table_lock.all_peers();
                for peer in peers {
                    let mut peer_label = String::new();
                    for &b in &peer.id.0[0..4] {
                        peer_label.push_str(&format!("{:02x}", b));
                    }
                    peer_label = format!("Peer_{}", peer_label);

                    // Convert XOR distance to edge state in DEC
                    let dist = self.self_id.xor_distance(&peer.id);
                    let lead_zeros = lead_zeros_bytes(&dist);
                    let scaled_dist = 1.0 - (lead_zeros as f64 / 160.0);
                    
                    self.complex.add_vertex(&peer_label, 1.0);
                    let _ = self.complex.add_edge("Node0", &peer_label, scaled_dist);
                }
            }
        }

        // 1. Gather active status
        let mut active_indices = Vec::new();
        for i in 0..self.max_nodes {
            if self.turing.active[i] {
                active_indices.push(i);
            }
        }

        // Scale coupling coefficients dynamically based on network bit-rate
        let net_rate_scaled = (net_rate / 1_000_000.0).min(5.0); // max 5.0 scale
        let base_coupling = 0.5;
        for i in 0..self.max_nodes {
            for j in 0..self.max_nodes {
                if i != j {
                    self.ensemble.regulator.l_matrix[i][j] = base_coupling * (1.0 + net_rate_scaled);
                }
            }
        }

        // 2. Prepare CPU observations for active agents
        let obs_val = cpu_load / 100.0;
        let mut obs = vec![vec![0.0, 0.0]; self.max_nodes];
        for &i in &active_indices {
            obs[i] = vec![obs_val, obs_val * 0.9];
        }

        // Step Onsager Collective Ensemble for active nodes
        let collective_f = self.ensemble.step(&obs, 0.1, 10, 1e-4, 0.1);

        // 3. Morphic Hot-swapping VM updates
        for &i in &active_indices {
            let dataset = vec![
                (vec![obs[i][0]], obs[i][0] * 1.5),
                (vec![obs[i][1]], obs[i][1] * 1.5),
            ];
            let seed = (self.step_count + i * 99) as u32;
            self.swappers[i].run_metamorphic_step(&dataset, seed);
        }

        // 4. Reconcile topological states (Layer 0)
        let mut delta_s = vec![0.0; self.complex.edges.len()];
        // Fill delta_s with tiny perturbation from CPU stress
        for i in 0..delta_s.len() {
            delta_s[i] = obs_val * 0.1;
        }
        let reconciled = reconcile_state_delta(&self.complex, &delta_s);
        let curl_deviation: f64 = reconciled.iter().zip(delta_s.iter())
            .map(|(&r, &d)| (r - d).powi(2))
            .sum::<f64>().sqrt();

        // 5. Gierer-Meinhardt reaction diffusion step (Layer 4)
        self.turing.step_rk4(0.05);

        // Extract beliefs and adjacency lists for mitosis/apoptosis
        let mut beliefs: Vec<Vec<f64>> = self.ensemble.agents.iter().map(|a| a.mu_q.clone()).collect();
        let mut adjacent_nodes = self.ensemble.adjacent_nodes.clone();

        // 6. Mitosis Engine split evaluation
        if let Some((parent, child)) = self.mitosis.evaluate_and_split(&mut self.turing, &mut beliefs, &mut adjacent_nodes) {
            // Apply new belief dimension to child agent
            self.ensemble.agents[child].mu_q = beliefs[child].clone();
            self.ensemble.agents[child].i_dim = beliefs[child].len();
            // Sync adjacency list
            self.ensemble.adjacent_nodes = adjacent_nodes.clone();
            // Update Hodge Mesh topology
            let child_label = format!("Node{}", child);
            self.complex.add_vertex(&child_label, 1.0);
            let parent_label = format!("Node{}", parent);
            self.complex.add_edge(&parent_label, &child_label, 1.0);
        }

        // 7. Apoptosis Engine pruning evaluation
        let mut free_energies = vec![0.0; self.max_nodes];
        for i in 0..self.max_nodes {
            free_energies[i] = self.ensemble.agents[i].f_history.last().cloned().unwrap_or(0.0);
        }
        if let Some(_pruned_node) = self.apoptosis.evaluate_and_prune(&mut self.turing, &mut beliefs, &mut adjacent_nodes, &free_energies, 5.0) {
            // Sync changes back
            self.ensemble.adjacent_nodes = adjacent_nodes;
        }

        // 8. Capture updated status state
        let mut nodes_status = Vec::new();
        for i in 0..self.max_nodes {
            let active = self.turing.active[i];
            let agent = &self.ensemble.agents[i];
            let (insts, _) = compile(&self.swappers[i].current_expr);
            nodes_status.push(NodeStatus {
                id: i,
                active,
                free_energy: agent.f_history.last().cloned().unwrap_or(0.0),
                belief_dim: agent.i_dim,
                beliefs: agent.mu_q.clone(),
                morphic_equation: format!("{:?}", self.swappers[i].current_expr),
                instruction_count: insts.len(),
                morphogen_u: self.turing.u[i],
                morphogen_v: self.turing.v[i],
            });
        }

        let mut edges = Vec::new();
        for &u in &active_indices {
            if u < self.ensemble.adjacent_nodes.len() {
                for &v in &self.ensemble.adjacent_nodes[u] {
                    if u < v && active_indices.contains(&v) {
                        edges.push((u, v));
                    }
                }
            }
        }

        self.last_status = SystemStatus {
            collective_free_energy: collective_f,
            curl_deviation,
            active_nodes_count: active_indices.len(),
            active_pool: active_indices,
            nodes: nodes_status,
            step_count: self.step_count,
            edges,
        };

        // 9. Broadcast thermodynamic state to all discovered peers
        if let Some(ref transport) = self.p2p_transport {
            if let Some(ref table) = self.p2p_table {
                if let Ok(table_lock) = table.try_lock() {
                    let peers = table_lock.all_peers();
                    let fe = self.last_status.collective_free_energy;
                    let pr = self.last_status.active_nodes_count as f64;
                    let push_frame = Frame {
                        sender: self.self_id,
                        payload: FramePayload::ThermodynamicPush {
                            free_energy: fe,
                            pressure: pr,
                        },
                    };
                    for peer in peers {
                        let tx = Arc::clone(&transport.socket);
                        let mut buf = [0u8; 100];
                        if let Ok(size) = push_frame.serialize(&mut buf) {
                            tokio::spawn(async move {
                                let _ = tx.send_to(&buf[..size], peer.address).await;
                            });
                        }
                    }
                }
            }
        }
    }

    pub fn handle_p2p_frame(&mut self, frame: Frame) {
        match frame.payload {
            FramePayload::ThermodynamicPush { free_energy, pressure } => {
                println!("[P2P Sync] Received ThermodynamicPush from {:?}: Free Energy = {:.4}, Pressure = {:.4}", frame.sender, free_energy, pressure);
                // Perturb beliefs slightly to represent incoming peer pressure
                if !self.ensemble.agents.is_empty() {
                    self.ensemble.agents[0].mu_q[0] += free_energy * 0.01;
                }
            }
            FramePayload::GradientDiff { target_id: _, coefficient, flow } => {
                println!("[P2P Sync] Received GradientDiff from {:?}: Coeff = {:.4}, Flow = {:.4}", frame.sender, coefficient, flow);
                if !self.ensemble.agents.is_empty() {
                    self.ensemble.agents[0].mu_q[0] += flow * coefficient * 0.1;
                }
            }
            _ => {}
        }
    }

    pub fn get_status_json(&self) -> String {
        serde_json::to_string_pretty(&self.last_status).unwrap_or_default()
    }
}
