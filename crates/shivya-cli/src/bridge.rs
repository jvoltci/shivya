//! Workload bridge from application-level signals to the simplicial state
//! complex consumed by Layers 0-4.
//!
//! The internal substrate operates on 0-chains (vertex masses) and 1-chains
//! (signed edge flows). Most applications don't think that way — they think
//! "queue length on node X" and "I'm shipping Y requests/sec from A to B".
//! This module wires the two together so developers can deploy on Shivya
//! without first learning Discrete Exterior Calculus.
//!
//! Lifecycle per tick:
//!   1. `record_queue_len(node, q)` -> 0-simplex mass at vertex `node`.
//!   2. `record_offload(src, dst, rate)` -> oriented 1-chain entry on the
//!      `src—dst` edge.
//!   3. `settle()` runs the Hodge curl-projection across the local
//!      simplicial complex and returns curl-free routing recommendations.
//!
//! Anything that the substrate cannot reconcile (network split-brain,
//! singular telemetry) surfaces as a finite recommendation rather than a
//! panic; the underlying math path is already ridge-stabilised in
//! `shivya-flux`/`shivya-morphic`.

use shivya::hodge::complex::SimplicialStateComplex;
use shivya::hodge::reconciler::reconcile_state_delta;

#[derive(Debug, Clone, PartialEq)]
pub enum BridgeError {
    UnknownNode(String),
    NoSuchEdge { src: String, dst: String },
    DuplicateNode(String),
}

impl std::fmt::Display for BridgeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BridgeError::UnknownNode(n) => write!(f, "unknown node `{}`", n),
            BridgeError::NoSuchEdge { src, dst } => {
                write!(f, "no offload edge between `{}` and `{}`", src, dst)
            }
            BridgeError::DuplicateNode(n) => write!(f, "duplicate node `{}`", n),
        }
    }
}

impl std::error::Error for BridgeError {}

#[derive(Debug, Clone, PartialEq)]
pub struct EdgeRecommendation {
    pub from: String,
    pub to: String,
    /// Curl-free target offload rate. Positive => flow from `from` to `to`,
    /// matching the orientation the edge was declared with.
    pub recommended_rate: f64,
}

#[derive(Debug, Clone)]
pub struct NodeStat {
    pub name: String,
    pub queue_len: usize,
    /// 0-simplex mass = 1.0 + queue_len / queue_scale.
    pub mass: f64,
}

#[derive(Debug, Clone)]
pub struct EdgeStat {
    pub from: String,
    pub to: String,
    pub reported_rate: f64,
    pub recommended_rate: f64,
}

#[derive(Debug, Clone)]
pub struct WorkloadSnapshot {
    pub nodes: Vec<NodeStat>,
    pub edges: Vec<EdgeStat>,
    /// L2 norm of (reported - recommended) flux from the last `settle()`.
    /// Reports how much rotational disagreement the substrate had to absorb.
    pub last_curl_norm: f64,
}

/// Idiomatic facade over the simplicial substrate. Holds an internal
/// `SimplicialStateComplex` and translates application signals to/from it.
#[derive(Debug, Clone)]
pub struct WorkloadMeshProxy {
    node_names: Vec<String>,
    edge_labels: Vec<(String, String)>,
    queue_lens: Vec<usize>,
    /// Flux aligned with `complex.edges` order, signed in the orientation
    /// the edge was registered with (declaration order).
    flux_canonical: Vec<f64>,
    pub queue_scale: f64,
    complex: SimplicialStateComplex,
    last_curl_norm: f64,
    last_recommendation_canonical: Vec<f64>,
}

impl WorkloadMeshProxy {
    /// Builds a proxy over `node_names` (vertices, in deployment order) and
    /// `edges` (allowed offload paths, declaration order). The declaration
    /// orientation `(src, dst)` is the positive direction for flux reports.
    pub fn new(
        node_names: Vec<String>,
        edges: Vec<(String, String)>,
    ) -> Result<Self, BridgeError> {
        for i in 0..node_names.len() {
            for j in (i + 1)..node_names.len() {
                if node_names[i] == node_names[j] {
                    return Err(BridgeError::DuplicateNode(node_names[i].clone()));
                }
            }
        }
        let mut complex = SimplicialStateComplex::new();
        for n in &node_names {
            complex.add_vertex(n, 1.0);
        }
        for (u, v) in &edges {
            if !node_names.iter().any(|n| n == u) {
                return Err(BridgeError::UnknownNode(u.clone()));
            }
            if !node_names.iter().any(|n| n == v) {
                return Err(BridgeError::UnknownNode(v.clone()));
            }
            complex.add_edge(u, v, 0.0);
        }
        let queue_lens = vec![0; node_names.len()];
        let flux_canonical = vec![0.0; complex.edges.len()];
        let last_recommendation_canonical = flux_canonical.clone();
        Ok(Self {
            node_names,
            edge_labels: edges,
            queue_lens,
            flux_canonical,
            queue_scale: 32.0,
            complex,
            last_curl_norm: 0.0,
            last_recommendation_canonical,
        })
    }

    /// Sets a custom queue-to-mass scale. Default is 32.0 (a queue of length
    /// 32 maps to mass 2.0, doubling the empty-node mass of 1.0).
    pub fn set_queue_scale(&mut self, scale: f64) {
        self.queue_scale = scale.max(1e-6);
        // Re-project existing queue lengths through the new scale.
        for (i, &q) in self.queue_lens.clone().iter().enumerate() {
            self.complex.vertex_states[i] = 1.0 + q as f64 / self.queue_scale;
        }
    }

    /// Records the current incoming request queue length (or any non-negative
    /// integer "amount of pending work") at `node`. Maps to the 0-simplex
    /// mass on that vertex.
    pub fn record_queue_len(&mut self, node: &str, q: usize) -> Result<(), BridgeError> {
        let idx = self
            .node_names
            .iter()
            .position(|n| n == node)
            .ok_or_else(|| BridgeError::UnknownNode(node.into()))?;
        self.queue_lens[idx] = q;
        self.complex.vertex_states[idx] = 1.0 + q as f64 / self.queue_scale;
        Ok(())
    }

    /// Convenience alias for "vector array length" or "in-flight job count".
    /// Same semantics as `record_queue_len`.
    pub fn record_vector_load(&mut self, node: &str, items: usize) -> Result<(), BridgeError> {
        self.record_queue_len(node, items)
    }

    /// Reports the current offload rate (requests/sec, items/sec, watts —
    /// units are the caller's choice) flowing `src -> dst`. Maps to a signed
    /// entry on the 1-chain for that edge.
    pub fn record_offload(
        &mut self,
        src: &str,
        dst: &str,
        rate_per_sec: f64,
    ) -> Result<(), BridgeError> {
        let u = self
            .node_names
            .iter()
            .position(|n| n == src)
            .ok_or_else(|| BridgeError::UnknownNode(src.into()))?;
        let v = self
            .node_names
            .iter()
            .position(|n| n == dst)
            .ok_or_else(|| BridgeError::UnknownNode(dst.into()))?;
        let (edge_idx, sign) =
            self.complex
                .find_edge_index(u, v)
                .ok_or_else(|| BridgeError::NoSuchEdge {
                    src: src.into(),
                    dst: dst.into(),
                })?;
        self.flux_canonical[edge_idx] = sign * rate_per_sec;
        self.complex.edge_states[edge_idx] = self.flux_canonical[edge_idx];
        Ok(())
    }

    /// Project the reported flux into the curl-free subspace and return one
    /// `EdgeRecommendation` per declared edge, in declaration order. The
    /// recommended rate is what the application should drive the offload
    /// stream toward in order to settle the mesh into a non-rotational
    /// (consistent) state — without any central consensus round.
    pub fn settle(&mut self) -> Vec<EdgeRecommendation> {
        let reconciled = reconcile_state_delta(&self.complex, &self.flux_canonical);
        let mut residual_sq = 0.0;
        for (i, &r) in reconciled.iter().enumerate() {
            let d = r - self.flux_canonical[i];
            residual_sq += d * d;
        }
        self.last_curl_norm = residual_sq.sqrt();
        self.last_recommendation_canonical = reconciled.clone();

        let mut recs = Vec::with_capacity(self.edge_labels.len());
        for (u_label, v_label) in &self.edge_labels {
            let u = match self.node_names.iter().position(|n| n == u_label) {
                Some(i) => i,
                None => continue,
            };
            let v = match self.node_names.iter().position(|n| n == v_label) {
                Some(i) => i,
                None => continue,
            };
            if let Some((idx, sign)) = self.complex.find_edge_index(u, v) {
                recs.push(EdgeRecommendation {
                    from: u_label.clone(),
                    to: v_label.clone(),
                    recommended_rate: sign * reconciled[idx],
                });
            }
        }
        recs
    }

    /// Read-only snapshot of the most recent state. Cheap to call repeatedly.
    pub fn snapshot(&self) -> WorkloadSnapshot {
        let nodes = self
            .node_names
            .iter()
            .enumerate()
            .map(|(i, name)| NodeStat {
                name: name.clone(),
                queue_len: self.queue_lens[i],
                mass: self.complex.vertex_states[i],
            })
            .collect();

        let mut edges = Vec::new();
        for (u_label, v_label) in &self.edge_labels {
            let u = match self.node_names.iter().position(|n| n == u_label) {
                Some(i) => i,
                None => continue,
            };
            let v = match self.node_names.iter().position(|n| n == v_label) {
                Some(i) => i,
                None => continue,
            };
            if let Some((idx, sign)) = self.complex.find_edge_index(u, v) {
                edges.push(EdgeStat {
                    from: u_label.clone(),
                    to: v_label.clone(),
                    reported_rate: sign * self.flux_canonical[idx],
                    recommended_rate: sign * self.last_recommendation_canonical[idx],
                });
            }
        }

        WorkloadSnapshot {
            nodes,
            edges,
            last_curl_norm: self.last_curl_norm,
        }
    }

    pub fn node_names(&self) -> &[String] {
        &self.node_names
    }

    pub fn edge_labels(&self) -> &[(String, String)] {
        &self.edge_labels
    }

    pub fn last_curl_norm(&self) -> f64 {
        self.last_curl_norm
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(items: &[&str]) -> Vec<String> {
        items.iter().map(|s| (*s).to_string()).collect()
    }

    fn pairs(items: &[(&str, &str)]) -> Vec<(String, String)> {
        items.iter().map(|(a, b)| (a.to_string(), b.to_string())).collect()
    }

    #[test]
    fn queue_length_maps_to_vertex_mass() {
        let mut proxy = WorkloadMeshProxy::new(
            names(&["A", "B"]),
            pairs(&[("A", "B")]),
        )
        .expect("construct proxy");
        proxy.record_queue_len("A", 64).unwrap();
        let snap = proxy.snapshot();
        assert_eq!(snap.nodes[0].queue_len, 64);
        // 64 / 32.0 = 2.0; total mass = 1.0 + 2.0 = 3.0
        assert!((snap.nodes[0].mass - 3.0).abs() < 1e-9);
        // Untouched node stays at baseline mass 1.0.
        assert!((snap.nodes[1].mass - 1.0).abs() < 1e-9);
    }

    #[test]
    fn triangle_curl_is_projected_out() {
        // A-B-C triangle. Inject a rotational flow that pumps unit flux
        // around the cycle. The substrate must zero out the curl entirely.
        let mut proxy = WorkloadMeshProxy::new(
            names(&["A", "B", "C"]),
            pairs(&[("A", "B"), ("B", "C"), ("A", "C")]),
        )
        .expect("construct triangle");
        proxy.record_offload("A", "B", 1.0).unwrap();
        proxy.record_offload("B", "C", 1.0).unwrap();
        // Closing the loop in the OPPOSITE direction (C -> A == -A -> C)
        // creates a pure rotational flow with no source/sink.
        proxy.record_offload("A", "C", -1.0).unwrap();

        let recs = proxy.settle();
        assert_eq!(recs.len(), 3);
        // Curl norm > 0 because the input was non-trivially rotational.
        assert!(proxy.last_curl_norm() > 1e-3);

        // After projection: every recommended rate must be smaller in
        // magnitude than the corresponding input (the projection bleeds
        // off the rotational component).
        let snap = proxy.snapshot();
        for e in &snap.edges {
            assert!(
                e.recommended_rate.abs() <= e.reported_rate.abs() + 1e-9,
                "rec {:.6} should not exceed |input| {:.6}",
                e.recommended_rate,
                e.reported_rate
            );
        }
    }

    #[test]
    fn settle_is_idempotent_on_curl_free_input() {
        // A linear chain A-B has no triangles, so any flow is already
        // curl-free. settle() must return the input untouched and report
        // zero residual.
        let mut proxy =
            WorkloadMeshProxy::new(names(&["A", "B"]), pairs(&[("A", "B")])).unwrap();
        proxy.record_offload("A", "B", 4.2).unwrap();
        let recs = proxy.settle();
        assert_eq!(recs.len(), 1);
        assert!((recs[0].recommended_rate - 4.2).abs() < 1e-9);
        assert!(proxy.last_curl_norm() < 1e-12);
    }

    #[test]
    fn unknown_node_is_a_clean_error() {
        let mut proxy =
            WorkloadMeshProxy::new(names(&["A", "B"]), pairs(&[("A", "B")])).unwrap();
        let err = proxy.record_queue_len("ghost", 1).unwrap_err();
        assert_eq!(err, BridgeError::UnknownNode("ghost".into()));
        let err = proxy.record_offload("A", "ghost", 1.0).unwrap_err();
        assert_eq!(err, BridgeError::UnknownNode("ghost".into()));
    }

    #[test]
    fn missing_edge_is_a_clean_error() {
        let mut proxy =
            WorkloadMeshProxy::new(names(&["A", "B", "C"]), pairs(&[("A", "B")])).unwrap();
        let err = proxy.record_offload("A", "C", 1.0).unwrap_err();
        assert_eq!(
            err,
            BridgeError::NoSuchEdge {
                src: "A".into(),
                dst: "C".into()
            }
        );
    }

    #[test]
    fn duplicate_node_is_rejected_at_construction() {
        let err = WorkloadMeshProxy::new(
            names(&["A", "A"]),
            pairs(&[("A", "A")]),
        )
        .unwrap_err();
        assert_eq!(err, BridgeError::DuplicateNode("A".into()));
    }

    #[test]
    fn reverse_orientation_offload_is_signed_correctly() {
        // Edge declared (A, B). Reporting B -> A at +1.0 must round-trip
        // as a -1.0 in the recommendation for (A, B), because the canonical
        // orientation is opposite to the report.
        let mut proxy =
            WorkloadMeshProxy::new(names(&["A", "B"]), pairs(&[("A", "B")])).unwrap();
        proxy.record_offload("B", "A", 1.0).unwrap();
        let recs = proxy.settle();
        assert!((recs[0].recommended_rate - (-1.0)).abs() < 1e-9);
    }
}
