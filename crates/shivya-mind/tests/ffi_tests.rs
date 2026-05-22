//! Black-box smoke test for the C ABI surface in `shivya_mind::ffi`.
//!
//! The test calls the `extern "C"` entry points exactly as a Swift or
//! Kotlin caller would: raw pointers, NUL-terminated strings, a
//! caller-allocated packed buffer. It exercises a full
//! construct -> ingest -> project -> compare -> destruct cycle so the
//! pairing of `*_new` / `*_free` is validated end-to-end.

use std::ffi::CString;
use std::ptr;

use shivya_mind::ffi::{
    sm_codebook_free, sm_codebook_new, sm_hypervector_similarity, sm_memory_free,
    sm_memory_new, sm_memory_update, sm_memory_working_memory, PACKED_LEN,
};

fn cstr(s: &str) -> CString {
    CString::new(s).expect("test inputs are NUL-free")
}

#[test]
fn remember_then_query_via_raw_pointers() {
    let salt = b"shivya-mind/ffi-test";
    let cb = sm_codebook_new(salt.as_ptr(), salt.len());
    assert!(!cb.is_null(), "codebook handle should not be null");

    let mem = sm_memory_new(cb);
    assert!(!mem.is_null(), "memory handle should not be null");

    // Ingest one fact through the FFI ingest path.
    let subj = cstr("alice");
    let pred = cstr("likes");
    let obj = cstr("coffee");
    sm_memory_update(mem, subj.as_ptr(), pred.as_ptr(), obj.as_ptr());

    // Project working memory into two independent caller buffers.
    let mut buf_a = vec![0u8; PACKED_LEN];
    let mut buf_b = vec![0u8; PACKED_LEN];
    sm_memory_working_memory(mem, buf_a.as_mut_ptr());
    sm_memory_working_memory(mem, buf_b.as_mut_ptr());

    // After a single deterministic ingest there are no ties in the
    // sign projection, so two consecutive snapshots must be identical.
    assert_eq!(
        buf_a, buf_b,
        "consecutive working-memory snapshots should be byte-identical with no ties"
    );

    // Buffers must not be left at the zero pattern (would indicate the
    // FFI silently no-op'd).
    assert!(
        buf_a.iter().any(|&b| b != 0),
        "packed buffer should contain non-zero data"
    );

    // Self-similarity is exactly 1.0.
    let self_sim = sm_hypervector_similarity(buf_a.as_ptr(), buf_b.as_ptr());
    assert!(
        (self_sim - 1.0).abs() < 1e-6,
        "self-similarity = {self_sim}, expected 1.0"
    );

    // A second memory bound to the same codebook and primed with a
    // completely different fact should be markedly less similar.
    let mem2 = sm_memory_new(cb);
    assert!(!mem2.is_null());
    let s2 = cstr("bob");
    let p2 = cstr("hates");
    let o2 = cstr("waiting");
    sm_memory_update(mem2, s2.as_ptr(), p2.as_ptr(), o2.as_ptr());
    let mut buf_c = vec![0u8; PACKED_LEN];
    sm_memory_working_memory(mem2, buf_c.as_mut_ptr());

    let cross_sim = sm_hypervector_similarity(buf_a.as_ptr(), buf_c.as_ptr());
    assert!(
        cross_sim.abs() < 0.5,
        "two unrelated facts should be roughly orthogonal, got {cross_sim}"
    );
    assert!(
        cross_sim < self_sim,
        "cross similarity {cross_sim} should be below self {self_sim}"
    );

    // Tear everything down. Each handle is released exactly once;
    // running under leak/sanitizer would flag any double-free here.
    sm_memory_free(mem2);
    sm_memory_free(mem);
    sm_codebook_free(cb);
}

#[test]
fn handles_survive_many_allocation_cycles() {
    // Hammer the allocator to flush out any one-shot leaks that a
    // single round-trip would miss.
    for _ in 0..64 {
        let cb = sm_codebook_new(ptr::null(), 0);
        assert!(!cb.is_null());
        let mem = sm_memory_new(cb);
        assert!(!mem.is_null());

        let s = cstr("device");
        let p = cstr("emits");
        let o = cstr("telemetry");
        sm_memory_update(mem, s.as_ptr(), p.as_ptr(), o.as_ptr());

        let mut buf = vec![0u8; PACKED_LEN];
        sm_memory_working_memory(mem, buf.as_mut_ptr());

        sm_memory_free(mem);
        sm_codebook_free(cb);
    }
}

#[test]
fn null_arguments_do_not_crash() {
    // Every public entry point must tolerate the lazy / defensive
    // mobile caller that hands us a null.
    sm_codebook_free(ptr::null_mut());
    sm_memory_free(ptr::null_mut());
    assert!(sm_memory_new(ptr::null_mut()).is_null());

    let cb = sm_codebook_new(ptr::null(), 0);
    assert!(!cb.is_null());
    let mem = sm_memory_new(cb);
    assert!(!mem.is_null());

    // Partial nulls on the ingest path are dropped silently.
    let valid = cstr("x");
    sm_memory_update(mem, ptr::null(), valid.as_ptr(), valid.as_ptr());
    sm_memory_update(mem, valid.as_ptr(), ptr::null(), valid.as_ptr());
    sm_memory_update(mem, valid.as_ptr(), valid.as_ptr(), ptr::null());
    sm_memory_update(ptr::null_mut(), valid.as_ptr(), valid.as_ptr(), valid.as_ptr());

    // Null output buffer or null memory: no-op, no panic.
    sm_memory_working_memory(mem, ptr::null_mut());
    sm_memory_working_memory(ptr::null_mut(), ptr::null_mut());

    assert_eq!(sm_hypervector_similarity(ptr::null(), ptr::null()), 0.0);

    sm_memory_free(mem);
    sm_codebook_free(cb);
}
