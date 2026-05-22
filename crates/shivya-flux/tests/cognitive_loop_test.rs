//! Unified cognitive-clockwork integration test.
//!
//! Wires the `shivya-flux` active-inference agent to the `shivya-mind`
//! episodic segmenter via the agent's own state lifecycle. Asserts that
//! when the environment's behaviour rule flips mid-stream, the raw
//! variational free-energy spike inside `shivya-flux` propagates into
//! `shivya-mind` and seals an episode bead automatically -- no
//! manual sealing, no scripted boundary insertion.

use std::sync::Arc;

use shivya_flux::GibbsFluxAgent;
use shivya_mind::{
    surprise::{NgramPredictor, PredictorBox, SealReason, Segmenter, SegmenterConfig},
    Codebook, Memory,
};

fn build_segmenter() -> Segmenter {
    let codebook = Arc::new(Codebook::with_default_salt());
    let mut memory = Memory::new(codebook);
    memory.freeze_clock(1_700_000_000.0);

    let predictor = PredictorBox::Ngram(NgramPredictor::new());
    let cfg = SegmenterConfig {
        k_threshold: 2.0,
        // Make sure spike fires before drift / time-cap in this test;
        // we are validating the spike path specifically.
        s_cap_nats: 1.0e12,
        t_max_seconds: 1.0e12,
        ema_half_life: 8.0,
        ema_init_var: 0.01,
        bootstrap_events: 6,
    };
    Segmenter::new(memory, predictor, cfg)
}

fn build_agent() -> GibbsFluxAgent<2, 1, 2> {
    GibbsFluxAgent::<2, 1, 2>::new(
        [0.0, 0.0],
        [[10.0, 0.0], [0.0, 10.0]],
        [[2.0, 0.5], [0.5, 1.5]],
        [[0.1, 0.0], [0.0, 0.1]],
        [[0.0], [0.0]],
        [[0.0], [0.0]],
        [0.0, 0.0],
        [[1.0, 0.0], [0.0, 1.0]],
    )
}

#[test]
fn regime_shift_triggers_automatic_episode_seal() {
    let mut agent = build_agent().with_segmenter(build_segmenter());

    // Phase 1: stationary regime, observations cluster near (2.5, 1.0).
    let phase1 = [2.5, 1.0];
    for _ in 0..24 {
        let report = agent.step(&phase1, 0.1, 100, 1e-6);
        // No regime change yet: no spike-driven seal should fire while
        // we burn in. (A flush is not possible without manual call.)
        if let Some(d) = report.decision {
            assert!(
                !d.seal || d.reason != SealReason::Spike,
                "phase-1 must not produce a spike-driven seal; got {d:?}"
            );
        }
    }

    let beads_before_shift = agent
        .segmenter
        .as_ref()
        .expect("segmenter attached")
        .beads()
        .len();

    // Phase 2: environment behaviour rule flips. The agent's prior
    // beliefs are tuned to phase 1, so the very first phase-2 step
    // should generate a free-energy explosion that the segmenter
    // converts into a sealed episode bead.
    let phase2 = [-2.5, -1.0];

    let mut shift_seal_seen = false;
    let mut shift_reason: Option<SealReason> = None;
    let mut shift_surprise: f32 = 0.0;
    for _ in 0..5 {
        let report = agent.step(&phase2, 0.1, 100, 1e-6);
        if let Some(d) = report.decision {
            if d.seal {
                shift_seal_seen = true;
                shift_reason = Some(d.reason);
                shift_surprise = d.surprise;
                break;
            }
        }
    }

    let segmenter = agent.segmenter.as_ref().expect("segmenter attached");
    let beads_after_shift = segmenter.beads().len();

    assert!(
        shift_seal_seen,
        "regime shift must trigger an automatic seal via free-energy spike"
    );
    assert_eq!(
        shift_reason,
        Some(SealReason::Spike),
        "the seal must be a spike (not drift / time / flush)"
    );
    assert!(
        beads_after_shift > beads_before_shift,
        "a new structured timeline bead must materialise on the node"
    );
    let last_bead = segmenter.beads().last().expect("bead must exist");
    assert!(
        last_bead.surprise_peak > 0.0,
        "sealed bead must carry a positive surprise peak"
    );
    assert!(
        last_bead.n_events > 0,
        "sealed bead must summarise at least one observed event"
    );
    assert!(
        shift_surprise.is_finite() && shift_surprise > 0.0,
        "decision surprise must be a finite positive free-energy reading; got {shift_surprise}"
    );
}
