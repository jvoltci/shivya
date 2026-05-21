# Shivya: Design Philosophy

## 1. From global consensus to local flow reconciliation

Classical distributed systems treat state as a single shared variable and require replicas to agree on a global sequence of updates to that variable. That model fits applications where linearizable history is the load-bearing requirement: ledgers, transactional databases, kernel-style coordination services. The price is real — extra round trips, leader-failover machinery, the operational surface area of running Paxos/Raft at scale.

Many edge-resource problems do not need linearizable history. They need *eventually identical state* across nodes that are independently adjusting to local pressure (queue length, CPU load, bandwidth headroom). For that class of problem, the *flow* between nodes is the natural primitive, not the *value* sitting at any individual node.

Shivya is built for that class. It is a **consensus-free localized resource balancer**: each edge can keep mutating its local state, and when partitions heal, a Hodge curl-projector cancels the rotational disagreement that concurrent operations introduced. This is not equivalent to Paxos or Raft, and it does not provide linearizable global ordering. See [CITATIONS.md](../CITATIONS.md) for the underlying theorems and the workloads this regime suits.

State is modelled as a *state potential* defined on a directed simplicial complex:

- **0-simplices (vertices):** local state mass at a node.
- **1-simplices (edges):** directed flow between two nodes.
- **2-simplices (triangles):** concurrent-context boundaries, capable of carrying rotational tension.

Reconciliation is not a vote. It is a linear-algebra projection onto the curl-free subspace.

---

## 2. The Hodge decomposition of causal flow

Any discrete 1-chain `ΔS` on a simplicial complex decomposes uniquely into three orthogonal components:

$$\Delta S = d_0 \alpha + d_1^T \beta + \gamma$$

1. **`d_0 α` (exact / gradient):** the irrotational, conflict-free component. Represents local mutations that accumulate cleanly across the complex.
2. **`d_1^T β` (coexact / curl):** the rotational component. Represents race conditions and concurrent contradictions — closed-loop cycles where competing updates disagree (the classic diamond topology).
3. **`γ` (harmonic):** divergence-free *and* curl-free. Represents non-trivial topological loops (non-contractible homology cycles).

### The reconciliation step

When concurrent branches merge, the disagreement appears as non-zero curl: `d₁ ΔS ≠ 0`. The reconciler isolates this component by solving the coboundary Laplacian system:

$$L_2 \beta = d_1 \Delta S \quad \text{where} \quad L_2 = d_1 d_1^T$$

A stack-allocated Conjugate Gradient solver (`crates/shivya-hodge/src/solver.rs`, tolerance 1e-8) converges on `β`. The curl component is then projected out:

$$\Delta S_{\text{reconciled}} = \Delta S - d_1^T \beta$$

The remaining flow is curl-free. Every node observing the same complex reaches the same `ΔS_reconciled`, with no sequence numbers exchanged. The projector is idempotent — running it twice gives the same result as running it once.

This is what the Layer-0 integration test (`tests/jepsen_partitions.rs`) asserts. After a programmatic partition splits a 5-node bowtie complex and conflicting flows are injected on each side, the post-heal reconciled flow's curl norm is below 1e-7 — and a second projection introduces no further drift.

---

## 3. What the upper layers add

The Hodge projector is the floor, not the ceiling. The upper layers are what make this system useful for actual workloads:

- **Layer 1 (active inference)** lets each node maintain an explicit posterior over its observations. When telemetry collapses (e.g., singular covariance during sustained colinear load), the math path falls back to ridge regularisation (1e-6) and ultimately identity-matrix inversion. No `panic!` on the main path.

- **Layer 2 (self-optimising register core)** is a real bytecode VM with a 500-cycle budget, plus a generative model that can grow its own latent dimension at runtime. The "metamorphic" component is genetic programming with a free-energy fitness — useful, but labelled honestly.

- **Layer 3 (Onsager ensemble)** couples nodes with a symmetric flow matrix `L_ij = L_ji` and computes Harsanyi dividends over local coalitions (Möbius recursion over neighbourhood subsets). The collective free-energy functional rewards synergistic configurations.

- **Layer 4 (reaction-diffusion topology)** runs RK4-integrated Gierer-Meinhardt kinetics over the node graph, with an explicit CFL bound. Hot-spot activator peaks trigger pre-allocated mitosis; underutilised nodes get culled (with a ≥ 3-node integrity floor).

None of these layers require a global clock. None of them require an authoritative leader. All of them survive the chaos test (`tests/chaos_ensemble.rs`): 7 nodes, 15% UDP packet loss, random per-node isolation, plus a hard programmatic partition layered on top, with the assertion that collective free energy still trends down.

---

## 4. Where this fits

Use Shivya when:

- Your fleet is **edge-shaped**: many nodes, intermittent connectivity, no central coordinator.
- The workload is **flow-shaped**: load to balance, work to offload, resources to diffuse — not transactions to serialise.
- "Eventually identical" is acceptable. Most monitoring, telemetry-driven scheduling, and resource-rebalancing problems satisfy this.

Don't use Shivya when:

- You need **linearizable per-key reads**.
- You need an **authoritative transaction log** with strict commit semantics.
- You need **bounded staleness guarantees**. Shivya's convergence is asymptotic, not bounded-time.

Pick the right tool. Consensus is the right tool for many problems. Shivya is the right tool for some of the problems that consensus is overkill for.
