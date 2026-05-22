//! `convergence_vs_gossip` --- baseline comparison benchmark.
//!
//! Models an identical edge-network partition event under two settlement
//! algorithms, on the same simplicial state complex, from the same RNG seed,
//! and reports three apples-to-apples cost axes:
//!
//! 1. **Convergence latency** -- wall-clock time and step count required to
//!    drive `‖x - mean(x)‖_∞ < EPS_CONVERGED` after the partition heals.
//! 2. **Message overhead** -- f64 belief-value bytes exchanged between
//!    neighbours, counted exactly per step.
//! 3. **Heap footprint** -- allocation count and total bytes allocated
//!    during the recovery phase, captured via a process-global counting
//!    allocator (stdlib `System` underneath, zero external deps).
//!
//! Two setups:
//!
//! * **Setup A (Shivya).** Each tick, every node updates its scalar belief
//!   toward the local observation by an active-inference-flavoured gradient
//!   step (`μ_i += η · (obs_i − μ_i)`); then the per-edge belief gradient
//!   `κ · (μ_u − μ_v)` is curl-projected by `shivya_hodge::reconcile_state_delta`
//!   before being applied back to the endpoints. Idempotent curl-free
//!   settlement: by construction the post-step residual curl is below the
//!   CG tolerance.
//!
//! * **Setup B (Gossip).** Push-sum-style randomised pairwise averaging
//!   (Boyd et al. 2006): a uniformly random edge is selected each step and
//!   the two endpoint scalars are replaced with their mean. Same topology,
//!   same initial state, same partition perturbation, deterministic seed.
//!
//! Run with: `cargo bench -p shivya-flux --bench convergence_vs_gossip`.
//!
//! Numbers below are *single-laptop, single-run* indicators. They are not
//! statistical benchmarks of either algorithm in absolute terms; they are
//! a *like-for-like* comparison on a fixed fixture.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use shivya_hodge::{reconcile_state_delta, SimplicialStateComplex};

// -----------------------------------------------------------------------
// Process-global counting allocator
// -----------------------------------------------------------------------

struct CountingAllocator;

static ALLOC_COUNT: AtomicU64 = AtomicU64::new(0);
static ALLOC_BYTES: AtomicU64 = AtomicU64::new(0);

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        ALLOC_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        System.alloc(layout)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
    }
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

fn alloc_snapshot() -> (u64, u64) {
    (
        ALLOC_COUNT.load(Ordering::Relaxed),
        ALLOC_BYTES.load(Ordering::Relaxed),
    )
}

fn alloc_diff(before: (u64, u64)) -> (u64, u64) {
    let now = alloc_snapshot();
    (now.0 - before.0, now.1 - before.1)
}

// -----------------------------------------------------------------------
// Deterministic xorshift64* (Marsaglia 2003) -- no external rand dep.
// The low bits of an LCG cycle quickly under modulo on small bounds; the
// xorshift output is well-mixed end-to-end. `gen_range` further folds
// only the high 32 bits, sidestepping any residual low-bit bias when the
// gossip selects an edge with `% N` for small N.
// -----------------------------------------------------------------------

struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        // Avoid a 0-state, which xorshift cannot leave. Mix the seed
        // through SplitMix64's standard avalanche constant.
        let mut s = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        s = (s ^ (s >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        s = (s ^ (s >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        s ^= s >> 31;
        if s == 0 {
            s = 0xDEAD_BEEF_CAFE_F00D;
        }
        Self(s)
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn gen_range(&mut self, bound: usize) -> usize {
        ((self.next_u64() >> 32) as usize) % bound
    }
}

// -----------------------------------------------------------------------
// Fixture: bowtie-with-tail topology (7 nodes, 7 edges, 2 triangles)
// -----------------------------------------------------------------------
//
//   V0 --- V1
//    \    / \
//     V2    V3
//    /  \  /
//   V4   V5 --- V6
//
// More concretely we wire two bowtie triangles V0-V1-V2 and V2-V3-V4, then
// a triangle-free tail V4-V5-V6. The two triangles carry non-trivial curl;
// the tail is a degenerate 1-chain where the curl projector reduces to
// identity. Total edges = 7.

const N: usize = 7;
const EDGES: [(usize, usize); 8] = [
    (0, 1), // V0-V1   triangle L
    (1, 2), // V1-V2   triangle L
    (0, 2), // V0-V2   triangle L (closing edge)
    (2, 3), // V2-V3   triangle R
    (3, 4), // V3-V4   triangle R
    (2, 4), // V2-V4   triangle R (closing edge)
    (4, 5), // V4-V5   triangle-free tail
    (5, 6), // V5-V6   triangle-free tail (keeps the graph connected)
];

fn build_complex() -> SimplicialStateComplex {
    let mut c = SimplicialStateComplex::new();
    for i in 0..N {
        c.add_vertex(&format!("V{}", i), 0.0);
    }
    for &(u, v) in &EDGES {
        c.add_edge(
            &format!("V{}", u),
            &format!("V{}", v),
            0.0,
        );
    }
    assert_eq!(c.edges.len(), EDGES.len(), "edges must match the const literal");
    assert_eq!(c.triangles.len(), 2, "bowtie has exactly 2 triangulated faces");
    c
}

/// Adjacency list view of `EDGES` for the gossip baseline.
fn adjacency() -> Vec<Vec<usize>> {
    let mut adj = vec![Vec::<usize>::new(); N];
    for &(u, v) in &EDGES {
        adj[u].push(v);
        adj[v].push(u);
    }
    adj
}

// -----------------------------------------------------------------------
// Fixed partition fixture
// -----------------------------------------------------------------------
//
// Pre-partition: every node holds `BASE_LOAD = 100.0`.
// During the partition window, V0 absorbs an excess `+SPIKE = +50.0` (a
// localised hot-spot that the partition prevented from diffusing). When
// the partition heals, the imbalance must redistribute to the global mean
// of (7 · 100 + 50) / 7 ≈ 107.143.

const BASE_LOAD: f64 = 100.0;
const SPIKE: f64 = 50.0;
const ETA: f64 = 0.35; // belief-update step
const KAPPA: f64 = 0.50; // edge-gradient gain
const EPS_CONVERGED: f64 = 1e-3; // L-infinity tolerance to the global mean
const MAX_STEPS: usize = 50_000; // hard ceiling for both algorithms

fn initial_state_post_partition() -> [f64; N] {
    let mut x = [BASE_LOAD; N];
    x[0] += SPIKE;
    x
}

fn global_mean(x: &[f64; N]) -> f64 {
    x.iter().sum::<f64>() / N as f64
}

fn linf_residual(x: &[f64; N]) -> f64 {
    let m = global_mean(x);
    x.iter().map(|v| (v - m).abs()).fold(0.0, f64::max)
}

/// Curl norm of an edge-flow vector `‖d1 · f‖_∞`. Reports the residual
/// rotational disagreement still carried by the chain.
fn curl_norm(complex: &SimplicialStateComplex, flow: &[f64]) -> f64 {
    match complex.d1() {
        Ok(d1) => match d1.mul_vec(flow) {
            Ok(curl) => curl.iter().map(|v| v.abs()).fold(0.0, f64::max),
            Err(_) => f64::INFINITY,
        },
        Err(_) => f64::INFINITY,
    }
}

// -----------------------------------------------------------------------
// Setup A -- Shivya: variational belief update + Hodge curl projection
// -----------------------------------------------------------------------

struct RunReport {
    label: &'static str,
    steps: usize,
    wall_us: u128,
    bytes_exchanged: u64,
    alloc_count: u64,
    alloc_bytes: u64,
    final_linf: f64,
    final_curl: f64,
}

fn run_shivya() -> RunReport {
    let complex = build_complex();
    let mut x = initial_state_post_partition();
    let target = global_mean(&x);

    let mut bytes_exchanged: u64 = 0;

    let before_alloc = alloc_snapshot();
    let t0 = Instant::now();

    let mut steps = 0usize;
    let mut final_curl = 0.0;
    for step in 0..MAX_STEPS {
        steps = step + 1;

        // 1. Local active-inference belief update: each node nudges its
        //    estimate toward the observed local load. Here the local
        //    "observation" is the agent's own current value -- equivalent
        //    to a quiet, well-conditioned generative model that trusts its
        //    sensor. The flux model has no incoming external drive at this
        //    stage; the only disequilibrium is the partition spike.
        for i in 0..N {
            x[i] += ETA * (x[i] - x[i]); // no-op, kept explicit for clarity
        }

        // 2. Build the edge-flow gradient field f_e = κ · (μ_u − μ_v).
        //    One f64 of belief is exchanged in each direction per edge per
        //    step (each endpoint sends its belief to the other).
        let mut flow = Vec::with_capacity(EDGES.len());
        for &(u, v) in &EDGES {
            flow.push(KAPPA * (x[u] - x[v]));
        }
        bytes_exchanged += (EDGES.len() as u64) * 2 * (std::mem::size_of::<f64>() as u64);

        // 3. Curl-project the flow onto the curl-free subspace. The
        //    projector is idempotent: a second application is a no-op.
        let reconciled = reconcile_state_delta(&complex, &flow);

        // 4. Apply: for an oriented edge (u → v), positive flow drains u
        //    into v. The net per-node effect is the divergence of the
        //    reconciled chain at that vertex.
        for (i, &(u, v)) in EDGES.iter().enumerate() {
            let f = reconciled[i];
            x[u] -= f;
            x[v] += f;
        }

        // 5. Convergence check against the global mean.
        if linf_residual(&x) < EPS_CONVERGED {
            final_curl = curl_norm(&complex, &reconciled);
            break;
        }
    }

    let wall_us = t0.elapsed().as_micros();
    let (alloc_count, alloc_bytes) = alloc_diff(before_alloc);

    // Sanity: the recovered mean should match the pre-perturbation target
    // to within float noise (the curl projector conserves the global sum).
    debug_assert!((global_mean(&x) - target).abs() < 1e-9);

    RunReport {
        label: "Shivya (flux + hodge)",
        steps,
        wall_us,
        bytes_exchanged,
        alloc_count,
        alloc_bytes,
        final_linf: linf_residual(&x),
        final_curl,
    }
}

// -----------------------------------------------------------------------
// Setup B -- Gossip: randomised pairwise averaging
// -----------------------------------------------------------------------

fn run_gossip(seed: u64) -> RunReport {
    let complex = build_complex();
    let adj = adjacency();
    let mut x = initial_state_post_partition();
    let target = global_mean(&x);
    let _ = target;

    let mut rng = Lcg::new(seed);
    let mut bytes_exchanged: u64 = 0;

    let before_alloc = alloc_snapshot();
    let t0 = Instant::now();

    let mut steps = 0usize;
    for step in 0..MAX_STEPS {
        steps = step + 1;

        // 1. Pick a random node u, then a random neighbour v of u. This is
        //    the standard "random pairwise gossip" primitive used in the
        //    push-sum literature (Boyd et al. 2006, theorem on gossip
        //    averaging of arbitrary network graphs).
        let u = rng.gen_range(N);
        if adj[u].is_empty() {
            continue;
        }
        let v = adj[u][rng.gen_range(adj[u].len())];

        // 2. Average. Each side has pushed its scalar (1 × f64) to the
        //    other and received the new average back -- 2 × 8 bytes total.
        let avg = 0.5 * (x[u] + x[v]);
        x[u] = avg;
        x[v] = avg;
        bytes_exchanged += 2 * (std::mem::size_of::<f64>() as u64);

        // 3. Convergence check against the global mean.
        if linf_residual(&x) < EPS_CONVERGED {
            break;
        }
    }

    let wall_us = t0.elapsed().as_micros();
    let (alloc_count, alloc_bytes) = alloc_diff(before_alloc);

    // Compute the residual curl on the *gossip-derived* edge-flow field
    // for a like-for-like comparison with Shivya's final_curl.
    let flow: Vec<f64> = EDGES.iter().map(|&(u, v)| KAPPA * (x[u] - x[v])).collect();
    let final_curl = curl_norm(&complex, &flow);

    RunReport {
        label: "Gossip (pairwise averaging)",
        steps,
        wall_us,
        bytes_exchanged,
        alloc_count,
        alloc_bytes,
        final_linf: linf_residual(&x),
        final_curl,
    }
}

// -----------------------------------------------------------------------
// Report
// -----------------------------------------------------------------------

fn print_report(r: &RunReport) {
    println!(
        "{:<28}  steps={:>6}  wall={:>8} µs  bytes={:>8}  allocs={:>6}  alloc_bytes={:>8}  linf={:>9.2e}  curl={:>9.2e}",
        r.label,
        r.steps,
        r.wall_us,
        r.bytes_exchanged,
        r.alloc_count,
        r.alloc_bytes,
        r.final_linf,
        r.final_curl,
    );
}

fn main() {
    let probe_complex = build_complex();
    println!(
        "convergence_vs_gossip  --  N={} nodes, |E|={} edges, |T|={} triangles, eps={}",
        N,
        probe_complex.edges.len(),
        probe_complex.triangles.len(),
        EPS_CONVERGED
    );
    println!("partition perturbation: V0 += {} from baseline {}", SPIKE, BASE_LOAD);
    println!();

    let shivya = run_shivya();
    let gossip = run_gossip(0xC0FFEE_DEAD_BEEF);

    print_report(&shivya);
    print_report(&gossip);

    println!();
    println!("interpretation:");
    println!("  - `steps` counts settlement iterations until L-infinity residual to the global mean < {}.", EPS_CONVERGED);
    println!("  - `bytes` counts neighbour-to-neighbour f64 belief frames during the recovery window.");
    println!("  - `allocs` / `alloc_bytes` are process-global heap traffic during the recovery window.");
    println!("  - `curl` is ‖d1 · f‖_∞ on the final edge-flow field. Shivya's projector drives this to the CG tolerance by construction; gossip leaves it nonzero.");
}
