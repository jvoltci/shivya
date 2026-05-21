# Shivya: Design Manifesto

## 1. Why consensus is not always the right tool

Distributed systems have spent decades treating *total order over events* as the load-bearing primitive. Paxos, Raft, and atomic-clock coordinators are excellent at it, and the price they pay is real: extra round trips, increased tail latency, complex leader-failover dances, and operational surface area that doesn't shrink as the fleet grows.

That cost is warranted when the application genuinely needs linearizable history — a payment ledger, a coordination kernel, a database with serialisable transactions. It is wasted when the application only needs **eventually identical state across an edge fleet that's reconciling local resource pressure**.

Shivya targets the second case. It does not replace Raft; it provides a different primitive for a different problem.

---

## 2. The premise

Many edge-resource problems look like this:

- Each node has its own queue length, CPU pressure, available bandwidth.
- Nodes can offload work to neighbours, but they can't pause the world to ask a coordinator how to split the work.
- Concurrent decisions made during a partition produce conflicting deltas. Those deltas must reconcile to *a* consistent state when the partition heals — but the application doesn't care *which* consistent state, as long as everyone agrees.

This is a graph-theoretic problem with a well-understood answer: the Hodge decomposition. Any 1-chain over a simplicial complex (here: flows along edges between nodes) splits uniquely into a gradient component, a curl component, and a harmonic component. The curl component is exactly the part that captures "concurrent contradiction"; everything else is consistent on its face.

The contribution of Shivya's Layer 0 is just this: implement the curl projector concretely in code, ride it on real UDP, and assert in a test suite (`tests/jepsen_partitions.rs`, `tests/chaos_ensemble.rs`) that the projection actually cancels the curl introduced by a partition.

---

## 3. What stacks on top

The reconciler alone is not enough to run real workloads. So:

- **Layer 1** gives each node an active-inference agent that maintains an explicit posterior over its sensory inputs (CPU/network/RAM telemetry). This is variational free-energy minimisation, applied here as the workhorse for "what does this node believe about its current state."
- **Layer 2** is a register-machine VM with a 500-cycle budget per program. It evaluates and, when the running free-energy average gets too high, mutates the symbolic update law that governs the agent. The renamed "Self-Optimising Register Core" name is the honest one — this is genetic programming with a free-energy fitness function, not anything more.
- **Layer 3** couples agents together with a symmetric Onsager flow matrix `L_ij = L_ji`. Adjacent nodes diffuse belief parameters toward each other at a rate proportional to disagreement. The collective free-energy functional rewards synergistic neighbourhoods and penalises antagonistic ones via Möbius-recursion Harsanyi dividends.
- **Layer 4** runs a reaction-diffusion process over the node mesh, splitting overloaded nodes (mitosis from a pre-allocated object pool — no runtime heap resize) and culling underused ones (apoptosis with a ≥ 3-node integrity floor).

The five layers are independent enough to be useful on their own and dependent enough that they compose. None of them require a global clock.

---

## 4. What Shivya is *not*

- **Not a CP system.** No linearizable reads, no transaction log, no leader election. If your workload needs those, use Raft.
- **Not a CRDT framework.** CRDTs solve "concurrent writes that must merge"; Shivya solves "concurrent flows that must reconcile to a curl-free total." The two are related but not the same.
- **Not a production system at v0.2.** No third-party deployments, no benchmarks against existing systems, no SLA. Read the test suite, read the math, decide for yourself if the regime fits.
- **Not magic.** The math is correct, the tests pass, the daemon stays up under chaos. None of that makes it the right tool for every distributed problem.

---

## 5. Engineering principles

- **No panic on the main path.** Singular telemetry data goes through ridge regularisation (1e-6 diagonal), then identity-matrix fallback. Failure modes surface as `SubstrateError` for diagnostic visibility — never as process death.
- **Mathematical claims must have tests.** The Hodge projector is asserted idempotent (1e-7). The Onsager matrix is asserted symmetric. The RK4 step is asserted to obey its CFL bound. The chaos test asserts free-energy minimisation under 15% UDP loss + random node isolation + a hard partition.
- **No theatrical naming for naming's sake.** Every layer's name maps to a concrete engineering object: a simplicial complex, a register VM, a flow matrix, a reaction-diffusion solver. "Non-Dual" rhetoric is replaced with "Consensus-Free Distributed Resource-Sharing Mesh" — a longer name that's actually descriptive.
- **The application bridge is first-class.** The `WorkloadMeshProxy` API in `shivya-cli/src/bridge.rs` lets developers feed in queue lengths and offload rates and read back curl-free routing recommendations, without needing to know what a 1-chain is.

The substrate is a tool, not a worldview. Use it where it fits.
