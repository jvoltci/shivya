//! shivya-mind: bit-packed VSA memory engine.
//!
//! The library exposes a deterministic [`Codebook`] that turns string
//! labels into 10_000-bit hypervectors, a tri-tier [`Memory`] that bundles
//! role-bound events into long-term tallies under a power-law decay
//! schedule, and a [`Segmenter`] that watches predictive surprise and
//! seals episode beads when the stream's information rate spikes,
//! drifts, or runs out the wall clock.
//!
//! The on-disk algebra is bit-packed (`BitArray<[u64; 157], Msb0>`) so
//! `bind`, `permute`, `similarity`, and `bundle` ride directly on the
//! CPU's word-level XOR / popcount / rotate paths.

pub mod codebook;
pub mod ffi;
pub mod memory;
pub mod surprise;
pub mod vsa;

pub use codebook::{Codebook, Role, DEFAULT_SALT};
pub use memory::{Event, Memory, TallyBuffer};
pub use surprise::{
    EpisodeBead, HybridPredictor, NgramPredictor, Predictor, Segmenter,
    SegmenterDecision, SurpriseEma, VsaExpectednessPredictor,
};
pub use vsa::{Hypervector, Pcg32, D, WORDS};
