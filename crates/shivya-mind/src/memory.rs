//! Tri-tier VSA memory with power-law decay and exact retraction.
//!
//! Three real-valued tally buffers accumulate sign-evidence from events:
//!
//! * `E` -- open episode (one episode worth of role-bound events),
//! * `D` -- open day (sealed episode beads waiting to consolidate),
//! * `M` -- long-term memory (decayed bundle of past days).
//!
//! The blueprint update rule is
//!
//! ```text
//!     alpha(tau) = (1 + beta * tau) ^ (-psi)
//!     M_{n+1}    = alpha_n * M_n + sign(D_n)
//! ```
//!
//! `M`, `D`, `E` are held as `[f32; D]` so we keep fine-grained gradients
//! during decay; bipolar projection happens only when callers actually
//! need the bit-packed form (`working_memory`, sealing). Bit-packed
//! evidence is materialised lazily via the codebook.

use crate::codebook::{Codebook, Role};
use crate::vsa::{
    accumulate_into, bind, bind_into, bundle, permute, sign_with_tiebreak, similarity,
    Hypervector, Pcg32, D,
};
use std::collections::BTreeSet;
use std::sync::Arc;

/// One real-valued accumulator. 40 KB. Boxed inside [`Memory`] so the
/// struct itself stays small enough to live anywhere.
pub type TallyBuffer = [f32; D];

/// A subject-predicate-object triple with optional context and time.
#[derive(Clone, Debug, PartialEq)]
pub struct Event {
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub ctx: Option<String>,
    pub t: Option<f64>,
}

impl Event {
    pub fn new(
        subject: impl Into<String>,
        predicate: impl Into<String>,
        object: impl Into<String>,
    ) -> Self {
        Self {
            subject: subject.into(),
            predicate: predicate.into(),
            object: object.into(),
            ctx: None,
            t: None,
        }
    }

    pub fn with_ctx(mut self, ctx: impl Into<String>) -> Self {
        self.ctx = Some(ctx.into());
        self
    }

    pub fn with_time(mut self, t: f64) -> Self {
        self.t = Some(t);
        self
    }
}

/// Three-tier VSA memory.
pub struct Memory {
    pub codebook: Arc<Codebook>,
    pub beta: f32,
    pub psi: f32,

    m_tally: Box<TallyBuffer>,
    d_tally: Box<TallyBuffer>,
    e_tally: Box<TallyBuffer>,

    rng: Pcg32,
    last_m_update_t: f64,
    event_count_in_episode: usize,
    episode_count_in_day: usize,
    day_count: u64,

    seen_labels: BTreeSet<String>,
    /// Clock callback returning epoch seconds. Defaults to a steady
    /// monotonic-ish counter; tests inject a mock.
    clock: ClockFn,
    /// Frozen "now" for tests; if `Some`, overrides `clock`.
    clock_override: Option<f64>,
}

type ClockFn = fn() -> f64;

fn default_clock() -> f64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

impl Memory {
    /// Construct a memory with the blueprint defaults: `beta = 1 / 1 day`,
    /// `psi = 0.5`.
    pub fn new(codebook: Arc<Codebook>) -> Self {
        let rng = codebook.bundle_rng();
        Self {
            codebook,
            beta: 1.0 / 86_400.0,
            psi: 0.5,
            m_tally: Box::new([0.0; D]),
            d_tally: Box::new([0.0; D]),
            e_tally: Box::new([0.0; D]),
            rng,
            last_m_update_t: default_clock(),
            event_count_in_episode: 0,
            episode_count_in_day: 0,
            day_count: 0,
            seen_labels: BTreeSet::new(),
            clock: default_clock,
            clock_override: None,
        }
    }

    /// Pin the clock to a fixed value. Used by deterministic tests.
    pub fn freeze_clock(&mut self, t: f64) {
        self.clock_override = Some(t);
        self.last_m_update_t = t;
    }

    /// Advance the frozen clock by `seconds`. No effect if the clock is
    /// not currently frozen.
    pub fn advance_clock(&mut self, seconds: f64) {
        if let Some(t) = self.clock_override {
            self.clock_override = Some(t + seconds);
        }
    }

    fn now(&self) -> f64 {
        self.clock_override.unwrap_or_else(|| (self.clock)())
    }

    /// Return the pinned clock value if one is set. Allows the
    /// segmenter to share the memory's notion of `now` in tests.
    pub fn frozen_now(&self) -> Option<f64> {
        self.clock_override
    }

    pub fn beta(&self) -> f32 {
        self.beta
    }

    pub fn psi(&self) -> f32 {
        self.psi
    }

    pub fn with_beta(mut self, beta: f32) -> Self {
        self.beta = beta;
        self
    }

    pub fn with_psi(mut self, psi: f32) -> Self {
        self.psi = psi;
        self
    }

    /// `alpha(tau) = (1 + beta * tau) ^ (-psi)`.
    pub fn alpha_effective(&self, tau_seconds: f64) -> f32 {
        let tau = tau_seconds.max(0.0);
        ((1.0 + self.beta as f64 * tau).powf(-self.psi as f64)) as f32
    }

    fn filler(&mut self, label: &str) -> Hypervector {
        if !self.seen_labels.contains(label) {
            self.seen_labels.insert(label.to_string());
        }
        self.codebook.get(label)
    }

    /// `F_event = bundle(filler ⊛ role for each role)`.
    pub fn encode_event(&mut self, event: &Event) -> Hypervector {
        let cb = self.codebook.clone();
        let mut parts: Vec<Hypervector> = Vec::with_capacity(5);
        let r_subj = cb.role(Role::Subj);
        let f_subj = self.filler(&event.subject);
        parts.push(bind(&f_subj, &r_subj));

        let r_pred = cb.role(Role::Pred);
        let f_pred = self.filler(&event.predicate);
        parts.push(bind(&f_pred, &r_pred));

        let r_obj = cb.role(Role::Obj);
        let f_obj = self.filler(&event.object);
        parts.push(bind(&f_obj, &r_obj));

        if let Some(ctx) = &event.ctx {
            let r_ctx = cb.role(Role::AppCtx);
            let f_ctx = self.filler(ctx);
            parts.push(bind(&f_ctx, &r_ctx));
        }
        if let Some(t) = event.t {
            let t_idx = (t as i64).rem_euclid(D as i64);
            let t_vec = cb.time_anchor(t_idx);
            let r_when = cb.role(Role::When);
            parts.push(bind(&t_vec, &r_when));
        }
        let refs: Vec<&Hypervector> = parts.iter().collect();
        bundle(&refs, &mut self.rng)
    }

    /// Ingest one event into the open episode buffer.
    pub fn update(&mut self, event: &Event) {
        let f = self.encode_event(event);
        accumulate_into(&mut self.e_tally, &f, 1.0);
        self.event_count_in_episode += 1;
    }

    /// Close the open episode `E` and fold it into the day buffer `D`.
    /// Returns the sealed bipolar episode bead, or `None` if the buffer
    /// was empty.
    pub fn seal_episode(&mut self) -> Option<Hypervector> {
        if self.event_count_in_episode == 0 {
            return None;
        }
        let e_k = sign_with_tiebreak(&self.e_tally, &mut self.rng);
        accumulate_into(&mut self.d_tally, &e_k, 1.0);
        for slot in self.e_tally.iter_mut() {
            *slot = 0.0;
        }
        self.event_count_in_episode = 0;
        self.episode_count_in_day += 1;
        Some(e_k)
    }

    /// Decay `M` and fold the day buffer `D` into it.
    pub fn consolidate_day(&mut self) {
        if self.event_count_in_episode > 0 {
            self.seal_episode();
        }
        if self.episode_count_in_day == 0 {
            return;
        }
        let now = self.now();
        let tau = (now - self.last_m_update_t).max(0.0);
        let alpha = self.alpha_effective(tau);
        let d_j = sign_with_tiebreak(&self.d_tally, &mut self.rng);
        for i in 0..D {
            self.m_tally[i] *= alpha;
        }
        accumulate_into(&mut self.m_tally, &d_j, 1.0);
        for slot in self.d_tally.iter_mut() {
            *slot = 0.0;
        }
        self.episode_count_in_day = 0;
        self.day_count += 1;
        self.last_m_update_t = now;
    }

    /// Bipolar projection of `M + D + E`. This is the query surface.
    pub fn working_memory(&mut self) -> Hypervector {
        let mut combined: Box<TallyBuffer> = Box::new([0.0; D]);
        for i in 0..D {
            combined[i] = self.m_tally[i] + self.d_tally[i] + self.e_tally[i];
        }
        sign_with_tiebreak(&combined, &mut self.rng)
    }

    /// Normalised similarity of `event` to working memory in [-1, 1].
    pub fn fact_strength(&mut self, event: &Event) -> f32 {
        let f = self.encode_event(event);
        let wm = self.working_memory();
        similarity(&f, &wm)
    }

    /// Cleanup the filler in a given role against a candidate set.
    /// When `candidates` is `None`, every label seen so far is scored.
    pub fn query(
        &mut self,
        role: Role,
        candidates: Option<&[String]>,
        top_k: usize,
    ) -> Vec<(String, f32)> {
        if top_k == 0 {
            return Vec::new();
        }
        let wm = self.working_memory();
        let r = self.codebook.role(role);
        let mut unbound = wm;
        bind_into(&mut unbound, &r);
        let labels: Vec<String> = match candidates {
            Some(cs) => cs.to_vec(),
            None => self.seen_labels.iter().cloned().collect(),
        };
        let mut scored: Vec<(String, f32)> = labels
            .into_iter()
            .map(|l| {
                let v = self.codebook.get(&l);
                let s = similarity(&v, &unbound);
                (l, s)
            })
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);
        scored
    }

    pub fn entity_labels(&self) -> Vec<String> {
        self.seen_labels
            .iter()
            .filter(|l| !l.starts_with("__"))
            .cloned()
            .collect()
    }

    /// Counter-active subtraction across all active tiers.
    ///
    /// `forget(event, weight)` is the exact inverse of `weight` prior
    /// `update(event)` calls when the identity-permutation fold-up is in
    /// use, so over-forgetting drives the fact below noise into
    /// negative territory.
    pub fn forget(&mut self, event: &Event, weight: f32) {
        let f = self.encode_event(event);
        accumulate_into(&mut self.e_tally, &f, -weight);
        accumulate_into(&mut self.d_tally, &f, -weight);
        accumulate_into(&mut self.m_tally, &f, -weight);
    }

    /// Chronological step-back across episode beads. Unbinds the
    /// `WHEN` role from `bead` and applies the inverse permutation.
    pub fn step_back(&self, bead: &Hypervector, k: i64) -> Hypervector {
        let r_when = self.codebook.role(Role::When);
        let unbound = bind(bead, &r_when);
        permute(&unbound, -k)
    }

    pub fn event_count_in_episode(&self) -> usize {
        self.event_count_in_episode
    }

    pub fn episode_count_in_day(&self) -> usize {
        self.episode_count_in_day
    }

    pub fn day_count(&self) -> u64 {
        self.day_count
    }

    /// Read-only access to the long-term tally (useful for tests).
    pub fn m_tally(&self) -> &TallyBuffer {
        &self.m_tally
    }
}
