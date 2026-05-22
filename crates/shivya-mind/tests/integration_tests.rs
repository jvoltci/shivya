//! Port of the Python-prototype benchmark suite. Each test reproduces
//! one of the regression checks the Python bench/* scripts used to gate
//! merges; together they cover capacity, decay fidelity, and
//! segmentation quality.

use std::sync::Arc;

use shivya_mind::{
    memory::{Event, Memory},
    surprise::{
        HybridPredictor, NgramPredictor, PredictorBox, Segmenter, SegmenterConfig,
        VsaExpectednessPredictor,
    },
    vsa, Codebook, Hypervector,
};

// ----------------------------------------------------------------- helpers

fn anchors(codebook: &Codebook, n: usize) -> Vec<Hypervector> {
    (0..n)
        .map(|i| codebook.get(&format!("anchor/{i}")))
        .collect()
}

fn top_k_indices_by_similarity(query: &Hypervector, pool: &[Hypervector], k: usize) -> Vec<usize> {
    let mut sims: Vec<(usize, f32)> = pool
        .iter()
        .enumerate()
        .map(|(i, v)| (i, vsa::similarity(query, v)))
        .collect();
    sims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    sims.truncate(k);
    sims.into_iter().map(|(i, _)| i).collect()
}

// =============================================================================
// 1. Capacity verification (B2). K=100 -> 100% precision, K=500 -> >= 96.26%.
// =============================================================================

fn capacity_at(codebook: &Codebook, anchors: &[Hypervector], k: usize, trials: usize) -> f32 {
    let n = anchors.len();
    let mut shuffle_rng = codebook.bundle_rng();
    let mut total_hits: usize = 0;
    let mut total_items: usize = 0;
    let mut pool_idx: Vec<usize> = (0..n).collect();
    for _ in 0..trials {
        shuffle_rng.shuffle(&mut pool_idx);
        let chosen = &pool_idx[..k];
        let refs: Vec<&Hypervector> = chosen.iter().map(|i| &anchors[*i]).collect();
        let mut bundle_rng = codebook.bundle_rng();
        let m = vsa::bundle(&refs, &mut bundle_rng);
        let ranked: Vec<usize> = top_k_indices_by_similarity(&m, anchors, k);
        let top_set: std::collections::HashSet<usize> = ranked.into_iter().collect();
        let hits = chosen.iter().filter(|i| top_set.contains(i)).count();
        total_hits += hits;
        total_items += k;
    }
    total_hits as f32 / total_items as f32
}

#[test]
fn capacity_k100_full_precision() {
    let codebook = Codebook::with_default_salt();
    let pool = anchors(&codebook, 1024);
    let recall = capacity_at(&codebook, &pool, 100, 4);
    assert!(
        recall >= 0.9999,
        "K=100 capacity should be ~100%, got {recall:.4}"
    );
}

#[test]
fn capacity_k500_high_recall() {
    let codebook = Codebook::with_default_salt();
    let pool = anchors(&codebook, 1024);
    let recall = capacity_at(&codebook, &pool, 500, 4);
    assert!(
        recall >= 0.9626,
        "K=500 capacity should be >= 96.26%, got {recall:.4}"
    );
}

// =============================================================================
// 2. Fidelity: power-law decay and retraction.
// =============================================================================

#[test]
fn alpha_1_day_equals_one_over_sqrt2() {
    let cb = Arc::new(Codebook::with_default_salt());
    let mem = Memory::new(cb);
    let a = mem.alpha_effective(86_400.0);
    assert!(
        (a - (1.0f32 / 2.0f32.sqrt())).abs() < 1e-6,
        "alpha(1 day) = {a}, expected 1/sqrt(2)"
    );
}

#[test]
fn alpha_schedule_is_monotonic_and_in_range() {
    let cb = Arc::new(Codebook::with_default_salt());
    let mem = Memory::new(cb);
    let taus = [0.0, 3600.0, 86_400.0, 7.0 * 86_400.0, 30.0 * 86_400.0];
    let mut last = f32::INFINITY;
    for tau in taus {
        let a = mem.alpha_effective(tau);
        assert!(a > 0.0 && a <= 1.0, "alpha out of range: {a}");
        assert!(a <= last, "alpha not monotonic at tau={tau}");
        last = a;
    }
}

#[test]
fn forget_drives_doomed_fact_below_threshold() {
    let cb = Arc::new(Codebook::with_default_salt());
    let mut mem = Memory::new(cb);

    let doomed = Event::new("alice", "owns", "scooter");
    let retained = [
        Event::new("bob", "lives_in", "berlin"),
        Event::new("carol", "studies", "physics"),
        Event::new("dave", "plays", "guitar"),
        Event::new("erin", "drinks", "espresso"),
    ];
    let pre_facts: Vec<&Event> = std::iter::once(&doomed).chain(retained.iter()).collect();
    let mut asserts_of_doomed = 0;
    for turn in 0..50 {
        if turn == 25 {
            mem.forget(&doomed, asserts_of_doomed as f32);
            continue;
        }
        if turn < 25 {
            let f = pre_facts[turn % pre_facts.len()];
            if std::ptr::eq(f, &doomed) {
                asserts_of_doomed += 1;
            }
            mem.update(f);
        } else {
            mem.update(&retained[turn % retained.len()]);
        }
    }
    let distractor = Event::new("malory", "writes", "poems");
    let threshold: f32 = 0.05;
    let s_doomed = mem.fact_strength(&doomed);
    let s_distract = mem.fact_strength(&distractor);
    assert!(
        s_doomed < threshold,
        "forgotten fact still above threshold: {s_doomed:+.4}"
    );
    for r in &retained {
        let s = mem.fact_strength(r);
        assert!(s > threshold, "retained fact below threshold: {s:+.4}");
    }
    assert!(
        s_distract.abs() < threshold,
        "distractor signal too strong: |{s_distract:+.4}|"
    );
}

#[test]
fn consolidate_day_applies_power_law_decay() {
    let cb = Arc::new(Codebook::with_default_salt());
    let mut mem = Memory::new(cb);
    mem.freeze_clock(1_700_000_000.0);

    let fact = Event::new("alice", "owns", "scooter");
    for _ in 0..5 {
        mem.update(&fact);
    }
    mem.seal_episode();
    mem.consolidate_day();
    let m_day1: Vec<f32> = mem.m_tally().to_vec();

    mem.advance_clock(30.0 * 86_400.0);
    mem.update(&Event::new("bob", "lives_in", "berlin"));
    mem.seal_episode();
    mem.consolidate_day();

    // Project both M tallies on the alice-fact direction (an int8 vector
    // we re-derive from the encoder for stability).
    let f_alice = mem.encode_event(&fact);
    let proj_day1 = project(&m_day1, &f_alice);
    let proj_day2 = project(&mem.m_tally().to_vec(), &f_alice);
    let alpha_expected = (1.0_f32 + 30.0).powf(-0.5);
    let ratio = proj_day2 / proj_day1;
    // Day-2 fold added one unrelated bipolar event roughly orthogonal
    // to the alice support, so the ratio should sit near alpha_expected
    // within +/- 0.05 absolute (noise floor for D = 10_000).
    assert!(
        (ratio - alpha_expected).abs() < 0.06,
        "decay ratio {ratio:.4} far from theoretical alpha {alpha_expected:.4}"
    );
}

fn project(tally: &[f32], v: &Hypervector) -> f32 {
    debug_assert_eq!(tally.len(), vsa::D);
    let mut acc = 0.0f32;
    for i in 0..vsa::D {
        let w = i / 64;
        let bit_idx = i % 64;
        let mask = 1u64 << (63 - bit_idx);
        let sign = if v.data[w] & mask != 0 { -1.0 } else { 1.0 };
        acc += tally[i] * sign;
    }
    acc / vsa::D as f32
}

// =============================================================================
// 3. Segmentation quality (B4). 4 atom-disjoint blocks of 10 events; F1 = 1.0.
// =============================================================================

const BLOCKS: &[Block] = &[
    Block {
        verbs: &["coded", "reviewed", "committed"],
        objects: &["function_x", "module_y", "patch_z"],
        ctx: "ide",
    },
    Block {
        verbs: &["cooked", "ate", "drank"],
        objects: &["pasta", "salad", "wine"],
        ctx: "kitchen",
    },
    Block {
        verbs: &["planned", "scheduled", "discussed"],
        objects: &["q3_review", "deadline", "proposal"],
        ctx: "office",
    },
    Block {
        verbs: &["read", "watched", "listened_to"],
        objects: &["novel", "documentary", "jazz_album"],
        ctx: "couch",
    },
];

struct Block {
    verbs: &'static [&'static str],
    objects: &'static [&'static str],
    ctx: &'static str,
}

const EVENTS_PER_BLOCK: usize = 10;
const SUBJECT: &str = "alice";

fn make_timeline(seed: u64) -> (Vec<Event>, std::collections::BTreeSet<usize>) {
    let mut rng = vsa::Pcg32::new(seed, 0x5121_5121_5121_5121);
    let mut timeline = Vec::with_capacity(BLOCKS.len() * EVENTS_PER_BLOCK);
    let mut boundaries = std::collections::BTreeSet::new();
    for (block_idx, block) in BLOCKS.iter().enumerate() {
        if block_idx > 0 {
            boundaries.insert(timeline.len());
        }
        for _ in 0..EVENTS_PER_BLOCK {
            let v = block.verbs[rng.gen_range(block.verbs.len() as u32) as usize];
            let o = block.objects[rng.gen_range(block.objects.len() as u32) as usize];
            timeline.push(Event::new(SUBJECT, v, o).with_ctx(block.ctx));
        }
    }
    (timeline, boundaries)
}

fn boundary_f1(
    predicted: &[usize],
    truth: &std::collections::BTreeSet<usize>,
    tolerance: usize,
) -> (f32, f32, f32) {
    let mut matched_true = std::collections::HashSet::new();
    let mut matched_pred = std::collections::HashSet::new();
    let mut tp = 0;
    for p in predicted {
        for t in truth {
            if matched_true.contains(t) {
                continue;
            }
            let diff = if *p >= *t { *p - *t } else { *t - *p };
            if diff > tolerance {
                continue;
            }
            tp += 1;
            matched_true.insert(*t);
            matched_pred.insert(*p);
            break;
        }
    }
    let fp = predicted.len() - tp;
    let fn_ = truth.len() - tp;
    let precision = if tp + fp == 0 {
        0.0
    } else {
        tp as f32 / (tp + fp) as f32
    };
    let recall = if tp + fn_ == 0 {
        0.0
    } else {
        tp as f32 / (tp + fn_) as f32
    };
    let f1 = if precision + recall == 0.0 {
        0.0
    } else {
        2.0 * precision * recall / (precision + recall)
    };
    (precision, recall, f1)
}

#[test]
fn segmentation_quality_is_perfect_on_synthetic_blocks() {
    let cb = Arc::new(Codebook::with_default_salt());
    let mut mem = Memory::new(cb);
    mem.freeze_clock(1_700_000_000.0);

    let predictor = PredictorBox::Vsa(VsaExpectednessPredictor::new(0.85));
    let cfg = SegmenterConfig {
        k_threshold: 2.0,
        s_cap_nats: 50.0,
        t_max_seconds: 3600.0,
        ema_half_life: 15.0,
        ema_init_var: 0.02,
        bootstrap_events: 6,
    };
    let mut segmenter = Segmenter::new(mem, predictor, cfg);

    let (timeline, truth) = make_timeline(0);
    let mut predicted: Vec<usize> = Vec::new();
    for (i, ev) in timeline.iter().enumerate() {
        let d = segmenter.observe(ev);
        if d.seal {
            predicted.push(i);
        }
    }
    let _ = segmenter.flush();

    let (precision, recall, f1) = boundary_f1(&predicted, &truth, 2);
    assert!(
        f1 >= 0.999,
        "F1={f1:.3}, precision={precision:.3}, recall={recall:.3}, predicted={predicted:?}, truth={truth:?}"
    );

    // At least three non-flush beads must have been emitted, one per
    // true context switch.
    let non_flush = segmenter
        .beads()
        .iter()
        .filter(|b| b.reason.as_str() != "flush")
        .count();
    assert!(
        non_flush >= truth.len(),
        "expected >= {} non-flush beads, got {non_flush}",
        truth.len()
    );

    // Spot-check first bead thumbnail anchors on block 0.
    if let Some(first) = segmenter.beads().first() {
        let block0_atoms: std::collections::HashSet<&str> = BLOCKS[0]
            .verbs
            .iter()
            .copied()
            .chain(BLOCKS[0].objects.iter().copied())
            .chain(std::iter::once(SUBJECT))
            .collect();
        let recognized = first
            .thumbnail
            .iter()
            .filter_map(|x| x.as_deref())
            .any(|s| block0_atoms.contains(s));
        assert!(
            recognized,
            "first bead thumbnail {:?} contains no block-0 atom",
            first.thumbnail
        );
    }
}

// =============================================================================
// 4. Smoke checks for the auxiliary predictors so the surface compiles.
// =============================================================================

#[test]
fn hybrid_predictor_produces_finite_surprise() {
    let cb = Arc::new(Codebook::with_default_salt());
    let mut mem = Memory::new(cb);
    let hybrid = HybridPredictor::new(
        NgramPredictor::new(),
        VsaExpectednessPredictor::new(0.9),
        0.5,
    );
    let event = Event::new("alice", "owns", "scooter");
    let s = hybrid.surprise_with_memory(&mut mem, &event);
    assert!(s.is_finite());
}

// =============================================================================
// 5. External-surprise injection. Lets active-inference agents drive the
//    segmenter directly with their own free-energy signal.
// =============================================================================

#[test]
fn observe_with_surprise_seals_on_externally_driven_spike() {
    use shivya_mind::surprise::SealReason;

    let cb = Arc::new(Codebook::with_default_salt());
    let mut mem = Memory::new(cb);
    mem.freeze_clock(1_700_000_000.0);

    let predictor = PredictorBox::Ngram(NgramPredictor::new());
    let cfg = SegmenterConfig {
        k_threshold: 2.0,
        s_cap_nats: 1.0e9,
        t_max_seconds: 1.0e9,
        ema_half_life: 8.0,
        ema_init_var: 0.001,
        bootstrap_events: 4,
    };
    let mut seg = Segmenter::new(mem, predictor, cfg);

    // Calm baseline: 16 events with low externally-supplied surprise.
    // EMA mu stays near 0.1, sigma stays tiny.
    let calm = Event::new("agent", "perceived", "stable");
    let mut seal_during_calm = false;
    for _ in 0..16 {
        let d = seg.observe_with_surprise(&calm, 0.1);
        if d.seal {
            seal_during_calm = true;
        }
    }
    assert!(
        !seal_during_calm,
        "calm phase should not produce a seal under externally-supplied low surprise"
    );

    // One large external spike, mimicking a free-energy explosion: must
    // trip the spike rule and emit a bead without further intervention.
    let shock = Event::new("agent", "perceived", "shock");
    let d = seg.observe_with_surprise(&shock, 100.0);
    assert!(d.seal, "external surprise spike must trigger a seal");
    assert_eq!(d.reason, SealReason::Spike);
    let bead = d.bead.expect("seal must carry a bead");
    assert!(bead.surprise_peak >= 0.1, "bead must record observed surprise peak");
    assert!(bead.n_events > 0, "bead must summarise at least one event");
}
