//! Predictive-surprise loop and episodic segmentation.
//!
//! Per turn:
//!
//! 1. Score the incoming event against the predictor's prior state,
//!    `s_t = -log p(x_t | h_{t-1})`.
//! 2. Test the boundary rules against the EMA computed up to step `t-1`:
//!    spike (`s_t > mu + k * sigma`), drift (`sum_s > S_cap`), or
//!    time-cap (`now - t_open > T_max`).
//! 3. If any rule fires, seal the episode and emit a bead. The boundary
//!    event becomes the first event of the new episode (Zacks & Tversky).
//! 4. Fold `s_t` into the EMA and ingest the event into the open episode.

use crate::codebook::Role;
use crate::memory::{Event, Memory};
use crate::vsa::{bind, similarity, Hypervector, D};
use std::collections::HashMap;

/// A scorer for one incoming event against its own past.
pub trait Predictor {
    /// Surprise in nats. Must NOT mutate the predictor's state.
    fn surprise(&self, event: &Event) -> f32;
    /// Fold this event into the predictor's running state.
    fn observe(&mut self, event: &Event);
}

// ---------------------------------------------------------------- N-gram

/// Incremental bigram tracker over a single event atom (default: the
/// predicate). Laplace-smoothed; falls back to a unigram for the very
/// first event.
pub struct NgramPredictor {
    transitions: HashMap<String, HashMap<String, u32>>,
    unigrams: HashMap<String, u32>,
    prev_key: Option<String>,
    smoothing: f32,
    key_fn: fn(&Event) -> String,
}

impl NgramPredictor {
    pub fn new() -> Self {
        Self::with_key_fn(predicate_key)
    }

    pub fn with_key_fn(f: fn(&Event) -> String) -> Self {
        Self {
            transitions: HashMap::new(),
            unigrams: HashMap::new(),
            prev_key: None,
            smoothing: 0.5,
            key_fn: f,
        }
    }

    pub fn smoothing(mut self, s: f32) -> Self {
        self.smoothing = s;
        self
    }
}

impl Default for NgramPredictor {
    fn default() -> Self {
        Self::new()
    }
}

fn predicate_key(e: &Event) -> String {
    e.predicate.clone()
}

impl Predictor for NgramPredictor {
    fn surprise(&self, event: &Event) -> f32 {
        let k = (self.key_fn)(event);
        let vocab = self.unigrams.len().max(1) as f32;
        let (total, count) = match &self.prev_key {
            None => {
                let total: u32 = self.unigrams.values().sum();
                let count = self.unigrams.get(&k).copied().unwrap_or(0);
                (
                    total as f32 + self.smoothing * (vocab + 1.0),
                    count as f32 + self.smoothing,
                )
            }
            Some(prev) => match self.transitions.get(prev) {
                None => (self.smoothing * (vocab + 1.0), self.smoothing),
                Some(trans) => {
                    let total: u32 = trans.values().sum();
                    let count = trans.get(&k).copied().unwrap_or(0);
                    (
                        total as f32 + self.smoothing * (vocab + 1.0),
                        count as f32 + self.smoothing,
                    )
                }
            },
        };
        let p = if total > 0.0 {
            (count / total).max(1e-30)
        } else {
            1e-30
        };
        -p.ln()
    }

    fn observe(&mut self, event: &Event) {
        let k = (self.key_fn)(event);
        if let Some(prev) = &self.prev_key {
            *self
                .transitions
                .entry(prev.clone())
                .or_default()
                .entry(k.clone())
                .or_insert(0) += 1;
        }
        *self.unigrams.entry(k.clone()).or_insert(0) += 1;
        self.prev_key = Some(k);
    }
}

// -------------------------------------------------------- VSA-expectedness

/// Surprise as "how unlike a decaying summary of recent events is this
/// event". Maintains a running float bundle `h_summary` of recent
/// encodings, scores cosine similarity to it, maps to a probability,
/// returns `-log p`.
pub struct VsaExpectednessPredictor {
    h_summary: Box<[f32; D]>,
    decay: f32,
    n: u64,
}

impl VsaExpectednessPredictor {
    pub fn new(decay: f32) -> Self {
        Self {
            h_summary: Box::new([0.0; D]),
            decay,
            n: 0,
        }
    }

    fn embed_into(&self, memory: &mut Memory, event: &Event, out: &mut [f32; D]) {
        let f = memory.encode_event(event);
        for i in 0..D {
            let w = i / 64;
            let bit_idx = i % 64;
            let mask = 1u64 << (63 - bit_idx);
            let sign = if f.data[w] & mask != 0 { -1.0 } else { 1.0 };
            out[i] = sign;
        }
    }

    /// Surprise of `event` against the current running summary. Needs
    /// `&mut Memory` so the encoder can lazily extend the codebook
    /// cache, but does NOT update the memory's tallies.
    pub fn surprise_with_memory(&self, memory: &mut Memory, event: &Event) -> f32 {
        if self.n == 0 {
            return 0.0;
        }
        let mut f = [0.0f32; D];
        self.embed_into(memory, event, &mut f);
        let mut dot = 0.0f32;
        let mut nf = 0.0f32;
        let mut nh = 0.0f32;
        for i in 0..D {
            dot += f[i] * self.h_summary[i];
            nf += f[i] * f[i];
            nh += self.h_summary[i] * self.h_summary[i];
        }
        let nf = nf.sqrt();
        let nh = nh.sqrt();
        if nf == 0.0 || nh == 0.0 {
            return 0.0;
        }
        let cos = dot / (nf * nh);
        let p = ((cos + 1.0) / 2.0).max(1e-30);
        -p.ln()
    }

    pub fn observe_with_memory(&mut self, memory: &mut Memory, event: &Event) {
        let mut f = [0.0f32; D];
        self.embed_into(memory, event, &mut f);
        for i in 0..D {
            self.h_summary[i] = self.decay * self.h_summary[i] + f[i];
        }
        self.n += 1;
    }

    pub fn decay(&self) -> f32 {
        self.decay
    }
}

// ---------------------------------------------------------------- Hybrid

/// `s = lambda * s_ngram + (1 - lambda) * s_embed`, mixed in log-space.
pub struct HybridPredictor {
    pub ngram: NgramPredictor,
    pub embed: VsaExpectednessPredictor,
    pub lambda: f32,
}

impl HybridPredictor {
    pub fn new(ngram: NgramPredictor, embed: VsaExpectednessPredictor, lambda: f32) -> Self {
        Self {
            ngram,
            embed,
            lambda,
        }
    }

    pub fn surprise_with_memory(&self, memory: &mut Memory, event: &Event) -> f32 {
        self.lambda * self.ngram.surprise(event)
            + (1.0 - self.lambda) * self.embed.surprise_with_memory(memory, event)
    }

    pub fn observe_with_memory(&mut self, memory: &mut Memory, event: &Event) {
        self.ngram.observe(event);
        self.embed.observe_with_memory(memory, event);
    }
}

// ---------------------------------------------------------------- EMA

/// Online exponential moving mean and variance with a sample-mean warm-up.
///
/// First `warmup_n` updates use direct sample mean / variance so the
/// EMA does not start with an inflated variance that hides early spikes;
/// after that we switch to the West (1981) recurrence:
///
/// ```text
///     mu_t  = (1 - alpha) * mu_{t-1} + alpha * x_t
///     var_t = (1 - alpha) * (var_{t-1} + alpha * (x_t - mu_{t-1})^2)
/// ```
pub struct SurpriseEma {
    alpha: f32,
    warmup_n: usize,
    buf: Vec<f32>,
    pub mu: f32,
    pub var: f32,
    pub n: u64,
}

impl SurpriseEma {
    pub fn new(half_life: f32, init_var: f32, warmup_n: usize) -> Self {
        assert!(half_life > 0.0, "half_life must be positive");
        let alpha = 1.0 - 0.5f32.powf(1.0 / half_life);
        Self {
            alpha,
            warmup_n,
            buf: Vec::with_capacity(warmup_n),
            mu: 0.0,
            var: init_var,
            n: 0,
        }
    }

    pub fn update(&mut self, s: f32) {
        self.n += 1;
        if (self.n as usize) <= self.warmup_n {
            self.buf.push(s);
            let n = self.buf.len() as f32;
            self.mu = self.buf.iter().sum::<f32>() / n;
            if self.buf.len() > 1 {
                let m = self.mu;
                self.var = self.buf.iter().map(|x| (x - m) * (x - m)).sum::<f32>() / n;
            }
            return;
        }
        let delta = s - self.mu;
        self.var = (1.0 - self.alpha) * (self.var + self.alpha * delta * delta);
        self.mu += self.alpha * delta;
    }

    pub fn sigma(&self) -> f32 {
        self.var.max(1e-12).sqrt()
    }
}

// ---------------------------------------------------------------- Beads

/// A tier-1 timeline atom emitted when an episode seals.
#[derive(Clone, Debug)]
pub struct EpisodeBead {
    pub id: String,
    pub t_start: f64,
    pub t_end: f64,
    pub n_events: usize,
    pub reason: SealReason,
    pub surprise_peak: f32,
    pub thumbnail: [Option<String>; 3],
    pub vector: Hypervector,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SealReason {
    Spike,
    Drift,
    Time,
    Flush,
    Continue,
}

impl SealReason {
    pub fn as_str(self) -> &'static str {
        match self {
            SealReason::Spike => "spike",
            SealReason::Drift => "drift",
            SealReason::Time => "time",
            SealReason::Flush => "flush",
            SealReason::Continue => "continue",
        }
    }
}

#[derive(Clone, Debug)]
pub struct SegmenterDecision {
    pub seal: bool,
    pub reason: SealReason,
    pub surprise: f32,
    pub bead: Option<EpisodeBead>,
}

// ----------------------------------------------------------- Segmenter

/// Boundary detector that triggers on spike / drift / time-cap.
pub struct Segmenter {
    pub memory: Memory,
    predictor: PredictorBox,
    pub k_threshold: f32,
    pub s_cap_nats: f32,
    pub t_max_seconds: f64,
    pub bootstrap_events: usize,
    pub ema: SurpriseEma,

    episode_start_t: f64,
    cumulative_surprise: f32,
    peak_in_episode: f32,
    events_since_seal: usize,
    total_events_seen: u64,
    beads: Vec<EpisodeBead>,
    bead_counter: u64,
}

/// We hand-roll the predictor enum so we can call into either the
/// stateless trait surface (NgramPredictor) or the `&mut Memory`-using
/// surfaces (VsaExpectednessPredictor, HybridPredictor) from a single
/// segmenter object.
pub enum PredictorBox {
    Ngram(NgramPredictor),
    Vsa(VsaExpectednessPredictor),
    Hybrid(HybridPredictor),
}

impl PredictorBox {
    fn surprise(&self, memory: &mut Memory, event: &Event) -> f32 {
        match self {
            PredictorBox::Ngram(p) => p.surprise(event),
            PredictorBox::Vsa(p) => p.surprise_with_memory(memory, event),
            PredictorBox::Hybrid(p) => p.surprise_with_memory(memory, event),
        }
    }

    fn observe(&mut self, memory: &mut Memory, event: &Event) {
        match self {
            PredictorBox::Ngram(p) => p.observe(event),
            PredictorBox::Vsa(p) => p.observe_with_memory(memory, event),
            PredictorBox::Hybrid(p) => p.observe_with_memory(memory, event),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SegmenterConfig {
    pub k_threshold: f32,
    pub s_cap_nats: f32,
    pub t_max_seconds: f64,
    pub ema_half_life: f32,
    pub ema_init_var: f32,
    pub bootstrap_events: usize,
}

impl Default for SegmenterConfig {
    fn default() -> Self {
        Self {
            k_threshold: 2.0,
            s_cap_nats: 50.0,
            t_max_seconds: 3600.0,
            ema_half_life: 20.0,
            ema_init_var: 0.1,
            bootstrap_events: 8,
        }
    }
}

impl Segmenter {
    pub fn new(memory: Memory, predictor: PredictorBox, cfg: SegmenterConfig) -> Self {
        let now0 = memory.frozen_now().unwrap_or_else(default_now);
        Self {
            memory,
            predictor,
            k_threshold: cfg.k_threshold,
            s_cap_nats: cfg.s_cap_nats,
            t_max_seconds: cfg.t_max_seconds,
            bootstrap_events: cfg.bootstrap_events,
            ema: SurpriseEma::new(cfg.ema_half_life, cfg.ema_init_var, cfg.bootstrap_events),
            episode_start_t: now0,
            cumulative_surprise: 0.0,
            peak_in_episode: 0.0,
            events_since_seal: 0,
            total_events_seen: 0,
            beads: Vec::new(),
            bead_counter: 0,
        }
    }

    fn now(&self) -> f64 {
        // Memory holds the clock; reuse it so deterministic tests work.
        // We exploit consolidate_day's last_m_update_t for fixtures.
        // For a clean API: just call default clock through Memory's
        // free-standing `now`. We expose a small helper here.
        memory_now(&self.memory)
    }

    pub fn beads(&self) -> &[EpisodeBead] {
        &self.beads
    }

    /// One predictive step.
    pub fn observe(&mut self, event: &Event) -> SegmenterDecision {
        // 1. Score against current predictor state (before update).
        let s = self.predictor.surprise(&mut self.memory, event);
        self.observe_step(event, s)
    }

    /// Variant of [`Segmenter::observe`] that uses an externally supplied
    /// surprise value (e.g. Variational Free Energy lifted out of an
    /// active-inference agent) in place of the internal predictor's
    /// score. The predictor is still updated so its own state stays
    /// coherent for any downstream callers that mix the two modes.
    pub fn observe_with_surprise(
        &mut self,
        event: &Event,
        external_surprise: f32,
    ) -> SegmenterDecision {
        // Keep the internal predictor warm so callers can mix internal /
        // external scoring in the same stream without divergence.
        let _ = self.predictor.surprise(&mut self.memory, event);
        self.observe_step(event, external_surprise)
    }

    fn observe_step(&mut self, event: &Event, s: f32) -> SegmenterDecision {
        self.predictor.observe(&mut self.memory, event);
        self.total_events_seen += 1;

        // 2. Evaluate seal rules against the PRIOR EMA so big spikes
        //    cannot hide inside their own freshly-inflated sigma.
        let now = self.now();
        let bootstrapped = self.total_events_seen as usize > self.bootstrap_events;
        let nonempty = self.events_since_seal > 0;

        let spike = bootstrapped
            && nonempty
            && s > self.ema.mu + self.k_threshold * self.ema.sigma();
        let drift = nonempty && self.cumulative_surprise > self.s_cap_nats;
        let timecap = nonempty && (now - self.episode_start_t) > self.t_max_seconds;

        let (seal, reason) = if spike {
            (true, SealReason::Spike)
        } else if drift {
            (true, SealReason::Drift)
        } else if timecap {
            (true, SealReason::Time)
        } else {
            (false, SealReason::Continue)
        };

        let mut bead = None;
        if seal {
            bead = Some(self.seal_and_emit(reason, now));
            self.cumulative_surprise = 0.0;
            self.peak_in_episode = 0.0;
            self.events_since_seal = 0;
            self.episode_start_t = now;
        }

        // 3. Fold s_t into the EMA (skip the very first event so the
        //    cold-start artifact doesn't bias the warm-up sample).
        if self.total_events_seen > 1 {
            self.ema.update(s);
        }

        // 4. Ingest the event into the (possibly fresh) episode.
        self.memory.update(event);
        self.cumulative_surprise += s;
        if s > self.peak_in_episode {
            self.peak_in_episode = s;
        }
        self.events_since_seal += 1;

        SegmenterDecision {
            seal,
            reason,
            surprise: s,
            bead,
        }
    }

    /// Force-seal whatever open episode remains.
    pub fn flush(&mut self) -> Option<EpisodeBead> {
        if self.events_since_seal == 0 {
            return None;
        }
        let now = self.now();
        let bead = self.seal_and_emit(SealReason::Flush, now);
        self.cumulative_surprise = 0.0;
        self.peak_in_episode = 0.0;
        self.events_since_seal = 0;
        self.episode_start_t = now;
        Some(bead)
    }

    fn seal_and_emit(&mut self, reason: SealReason, t_end: f64) -> EpisodeBead {
        let e_k = self.memory.seal_episode().unwrap_or_else(|| {
            // Shouldn't trigger in practice because seal_and_emit is
            // only called when events_since_seal > 0, but be defensive.
            crate::vsa::zero()
        });
        let thumbnail = self.thumbnail(&e_k);
        let id = self.next_bead_id();
        let bead = EpisodeBead {
            id,
            t_start: self.episode_start_t,
            t_end,
            n_events: self.events_since_seal,
            reason,
            surprise_peak: self.peak_in_episode,
            thumbnail,
            vector: e_k,
        };
        self.beads.push(bead.clone());
        bead
    }

    fn thumbnail(&mut self, e_k: &Hypervector) -> [Option<String>; 3] {
        let entity_labels = self.memory.entity_labels();
        let mut out: [Option<String>; 3] = [None, None, None];
        if entity_labels.is_empty() {
            return out;
        }
        for (slot, role) in [Role::Subj, Role::Pred, Role::Obj].iter().enumerate() {
            let r = self.memory.codebook.role(*role);
            let probe = bind(e_k, &r);
            let mut best: Option<(String, f32)> = None;
            for lbl in &entity_labels {
                let v = self.memory.codebook.get(lbl);
                let s = similarity(&v, &probe);
                if best.as_ref().map(|(_, bs)| s > *bs).unwrap_or(true) {
                    best = Some((lbl.clone(), s));
                }
            }
            out[slot] = best.map(|(l, _)| l);
        }
        out
    }

    fn next_bead_id(&mut self) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"bead/");
        hasher.update(&self.bead_counter.to_le_bytes());
        hasher.update(&self.episode_start_t.to_le_bytes());
        self.bead_counter += 1;
        let digest = hasher.finalize();
        // 16 hex chars from the first 8 bytes.
        let bytes = &digest.as_bytes()[0..8];
        let mut s = String::with_capacity(16);
        for b in bytes {
            use std::fmt::Write;
            let _ = write!(s, "{:02x}", b);
        }
        s
    }
}

// Memory does not expose `now()` publicly so the segmenter can keep
// using the same clock the memory uses. We replicate the resolution
// here from a frozen override (if any) or from `SystemTime`.
fn memory_now(memory: &Memory) -> f64 {
    memory.frozen_now().unwrap_or_else(default_now)
}

fn default_now() -> f64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}
