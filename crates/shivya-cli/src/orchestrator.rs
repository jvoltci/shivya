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

use crate::bridge::{BridgeError, EdgeRecommendation, WorkloadMeshProxy, WorkloadSnapshot};

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
    /// L2 norm of the curl bled off the workload-bridge edge flux on the
    /// most recent settle(). 0.0 when the input was already curl-free.
    pub workload_curl_norm: f64,
    /// Curl-free per-edge offload-rate recommendations from the
    /// `WorkloadMeshProxy` driving the daemon's 1 Hz settle loop.
    pub workload_recommendations: Vec<EdgeRecommendation>,
    /// Full bridge snapshot (vertex masses + edge reported/recommended
    /// rates) for the most recent tick.
    pub workload_snapshot: WorkloadSnapshot,
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

    /// Application-facing bridge: maps queue/offload signals to the local
    /// simplicial complex, runs the curl projector, returns reconciled
    /// per-edge rates. Driven from `step_with_telemetry` every tick and
    /// reachable from outside the daemon via the UDS command protocol.
    pub workload_bridge: WorkloadMeshProxy,
    pub last_recommendations: Vec<EdgeRecommendation>,
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
            workload_curl_norm: 0.0,
            workload_recommendations: Vec::new(),
            workload_snapshot: WorkloadSnapshot::default(),
        };

        let self_id = NodeId::random();

        // Application-facing bridge mirrors the orchestrator's bootstrap
        // triangle (Node0-Node1-Node2). Inbound UDS clients can mutate this
        // proxy with queue/offload signals; the 1 Hz loop and the UDS
        // SETTLE command both run the curl projector against it and feed
        // the reconciled rates back into the Onsager coupling matrix.
        let workload_bridge = WorkloadMeshProxy::new(
            vec!["Node0".into(), "Node1".into(), "Node2".into()],
            vec![
                ("Node0".into(), "Node1".into()),
                ("Node1".into(), "Node2".into()),
                ("Node0".into(), "Node2".into()),
            ],
        )
        .expect("workload bridge bootstrap (default 3-node mesh)");

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
            workload_bridge,
            last_recommendations: Vec::new(),
        }
    }

    /// External recorder for application queue lengths (entry point used
    /// by the UDS `Q <node> <q>` command).
    pub fn record_workload_queue(&mut self, node: &str, q: usize) -> Result<(), BridgeError> {
        self.workload_bridge.record_queue_len(node, q)
    }

    /// External recorder for application offload rates (entry point used
    /// by the UDS `O <src> <dst> <rate>` command).
    pub fn record_workload_offload(
        &mut self,
        src: &str,
        dst: &str,
        rate: f64,
    ) -> Result<(), BridgeError> {
        self.workload_bridge.record_offload(src, dst, rate)
    }

    /// Runs the curl projector on the bridge's current state and applies
    /// the reconciled rates back into the active mesh: writes the per-edge
    /// recommendation into `complex.edge_states`, biases the Onsager
    /// `l_matrix` on that pair, and caches the recommendation set for the
    /// next status JSON dump.
    pub fn settle_and_apply(&mut self) -> Vec<EdgeRecommendation> {
        let recs = self.workload_bridge.settle();
        for rec in &recs {
            let u = rec.from.strip_prefix("Node").and_then(|s| s.parse::<usize>().ok());
            let v = rec.to.strip_prefix("Node").and_then(|s| s.parse::<usize>().ok());
            if let (Some(u), Some(v)) = (u, v) {
                if let Some((idx, sign)) = self.complex.find_edge_index(u, v) {
                    self.complex.edge_states[idx] = sign * rec.recommended_rate;
                }
                if u < self.max_nodes && v < self.max_nodes {
                    let scale = 1.0 + rec.recommended_rate.abs().min(2.0);
                    self.ensemble.regulator.l_matrix[u][v] *= scale;
                    self.ensemble.regulator.l_matrix[v][u] *= scale;
                }
            }
        }
        self.last_recommendations = recs.clone();
        recs
    }

    /// Read-only view of the cached recommendations from the most recent
    /// `settle_and_apply()`. Cheap enough to call per UDS connection.
    pub fn workload_recommendations(&self) -> &[EdgeRecommendation] {
        &self.last_recommendations
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

    pub fn step_with_telemetry(&mut self, cpu_load: f64, net_rate: f64, memory_used_ratio: f64) {
        self.step_count += 1;
        // Memory pressure modulates the Onsager base coupling: heavier RAM
        // contention => stronger inter-node migration pressure.
        let memory_pressure_scale = 1.0 + memory_used_ratio.clamp(0.0, 1.0);

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
                    self.complex.add_edge("Node0", &peer_label, scaled_dist);
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

        // Coupling reflects real host pressure: net bit-rate × RAM pressure.
        let net_rate_scaled = (net_rate / 1_000_000.0).min(5.0);
        let base_coupling = 0.5;
        for i in 0..self.max_nodes {
            for j in 0..self.max_nodes {
                if i != j {
                    self.ensemble.regulator.l_matrix[i][j] =
                        base_coupling * (1.0 + net_rate_scaled) * memory_pressure_scale;
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
        delta_s.fill(obs_val * 0.1);
        let reconciled = reconcile_state_delta(&self.complex, &delta_s);
        let curl_deviation: f64 = reconciled.iter().zip(delta_s.iter())
            .map(|(&r, &d)| (r - d).powi(2))
            .sum::<f64>().sqrt();

        // 4b. Drive the application-facing workload bridge from the
        // substrate's current state: queue length is read off the Turing
        // activator field (saturated activator ⇒ heavier in-flight work);
        // offload rate is the Onsager coupling × belief differential, the
        // same signal the ensemble itself uses to migrate parameters. Then
        // settle() runs the curl projector on the bridge's edge flux and
        // settle_and_apply() writes the reconciled per-edge rates back
        // into `complex.edge_states` and biases the Onsager L matrix.
        for i in 0..self.workload_bridge.node_names().len() {
            if i < self.turing.u.len() && self.turing.active[i] {
                let q = (self.turing.u[i].max(0.0)
                    * self.workload_bridge.queue_scale)
                    .round() as usize;
                let label = format!("Node{}", i);
                let _ = self.workload_bridge.record_queue_len(&label, q);
            }
        }
        let edge_labels = self.workload_bridge.edge_labels().to_vec();
        for (u_label, v_label) in &edge_labels {
            let u = u_label.strip_prefix("Node").and_then(|s| s.parse::<usize>().ok());
            let v = v_label.strip_prefix("Node").and_then(|s| s.parse::<usize>().ok());
            if let (Some(u), Some(v)) = (u, v) {
                if u < self.max_nodes && v < self.max_nodes {
                    let coupling = self.ensemble.regulator.l_matrix[u][v];
                    let belief_diff = if !self.ensemble.agents[u].mu_q.is_empty()
                        && !self.ensemble.agents[v].mu_q.is_empty()
                    {
                        self.ensemble.agents[u].mu_q[0] - self.ensemble.agents[v].mu_q[0]
                    } else {
                        0.0
                    };
                    let _ = self
                        .workload_bridge
                        .record_offload(u_label, v_label, coupling * belief_diff);
                }
            }
        }
        let workload_recs = self.settle_and_apply();
        let workload_curl_norm = self.workload_bridge.last_curl_norm();
        let workload_snapshot = self.workload_bridge.snapshot();

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
            workload_curl_norm,
            workload_recommendations: workload_recs,
            workload_snapshot,
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
