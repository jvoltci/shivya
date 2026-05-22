// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Kotlin companion mirror of the `shivya_mind::ffi` C ABI surface
// (`crates/shivya-mind/src/ffi.rs`). This file lives next to the
// Rust source for visibility — Android module builds are expected
// to copy or symlink it into their `src/main/kotlin/` source tree
// under the package directory below. Cargo silently ignores
// non-`.rs` files in `src/`, so leaving it here does not affect
// the Rust compilation.
//
// The Rust side exposes seven `#[no_mangle] unsafe extern "C"`
// entry points and one compile-time constant `PACKED_LEN = 1_250`.
// All entry points are wrapped in `std::panic::catch_unwind` so a
// panic in Rust cannot cross the ABI boundary into the JVM.
//
// BINDING MECHANISM
// -----------------
// `external fun` symbols are resolved by the JVM at first call.
// The JVM looks up two flavours of symbol in this order:
//   1. The fully-mangled JNI name
//      `Java_io_github_jvoltci_shivya_mind_ShivyaMindNative_sm_1codebook_1new`
//      (underscores in the Kotlin name become `_1` after the
//      `Java_<pkg>_<class>_` prefix). The Rust crate does not
//      ship these wrappers today, so this path is currently
//      not used.
//   2. The bare C symbol named via `RegisterNatives` at load
//      time. To wire the bindings below straight to the
//      `sm_codebook_new` / `sm_memory_*` / `sm_hypervector_*`
//      C symbols exported by `libshivya_mind.so`, the host
//      application's JNI_OnLoad must call `RegisterNatives`
//      with a method table that maps each Kotlin signature to
//      its `sm_*` symbol. (This is the canonical pattern used
//      by hot-path Android libraries; see e.g. SQLite's JNI
//      glue.)
//
// Until the `JNI_OnLoad` glue lands, this file is the
// authoritative declaration of the JVM-side contract: argument
// types, nullability, return types, and the zero-copy
// expectations on `java.nio.ByteBuffer` operands.
//
// ZERO-COPY CONTRACT
// ------------------
// Two functions exchange `ByteBuffer` operands:
//
//   * `sm_memory_working_memory(mem, outPackedBuffer)`
//   * `sm_hypervector_similarity(aPacked, bPacked)`
//
// On the Kotlin side these MUST be **direct** buffers, allocated
// with `java.nio.ByteBuffer.allocateDirect(PACKED_LEN)`. Direct
// buffers expose a stable native address that the JNI binding can
// hand to Rust as a `*mut u8` / `*const u8` of exactly
// `PACKED_LEN = 1_250` bytes — zero copy, zero intermediary heap
// allocation, zero GC pressure on the hot path.
//
// Passing a non-direct (heap-array-backed) `ByteBuffer` will not
// crash but will silently fall back to a per-call copy through
// the JNI bounce buffer, defeating the performance contract.
// The runtime registration MUST assert `buffer.isDirect()` and
// reject otherwise; do not relax this in downstream consumers.

@file:Suppress("FunctionName") // sm_* names mirror the C ABI exactly.

package io.github.jvoltci.shivya.mind

import java.nio.ByteBuffer

/**
 * Compile-time mirror of the Rust constant `shivya_mind::ffi::PACKED_LEN`.
 * Buffers exchanged with `sm_memory_working_memory` and
 * `sm_hypervector_similarity` MUST be allocated with exactly this many
 * bytes. Re-derive it from `D / 8` only after coordinating a major
 * version bump on both sides.
 */
internal const val PACKED_LEN: Int = 1_250

/**
 * 1:1 mirror of the `shivya_mind::ffi` C ABI. Single companion
 * `object` so the JVM loads `libshivya_mind.so` exactly once per
 * class loader and `RegisterNatives` runs exactly once.
 *
 * Handle lifetimes follow the standard opaque-pointer discipline:
 *   * `sm_codebook_new` / `sm_memory_new` produce a non-zero `Long`
 *     on success or `0L` on failure. The JVM MUST never dereference
 *     the value, compare it for ordering, or persist it across
 *     process restarts.
 *   * Each handle must be passed to its matching `_free` exactly
 *     once. Passing `0L` to a `_free` is always safe and is a
 *     no-op (matches the Rust-side null guard).
 */
internal object ShivyaMindNative {
    init {
        System.loadLibrary("shivya_mind")
    }

    /**
     * Construct a deterministic codebook bound to `salt[..saltLen]`.
     * Pass `(null, 0)` to select the engine's default salt.
     *
     * @return Opaque non-zero handle on success; `0L` on allocation
     *         failure or panic.
     */
    external fun sm_codebook_new(salt: ByteArray?, saltLen: Int): Long

    /**
     * Release a codebook handle. `0L` is a no-op. Memories still
     * holding an `Arc<Codebook>` clone keep working until they
     * too are freed.
     */
    external fun sm_codebook_free(cb: Long)

    /**
     * Construct a tri-tier (`E`, `D`, `M`) memory bound to `cb`.
     * Bumps the codebook's `Arc` refcount; the caller still owns `cb`.
     *
     * @return Opaque non-zero handle on success; `0L` on `cb == 0L`
     *         or allocation failure.
     */
    external fun sm_memory_new(cb: Long): Long

    /** Release a memory handle. `0L` is a no-op. */
    external fun sm_memory_free(mem: Long)

    /**
     * Ingest one (subject, predicate, object) triple into the open
     * episode buffer of `mem`. The Kotlin compiler reserves
     * `object` as a hard keyword; the parameter is backtick-escaped
     * to keep the binding line up with the C parameter name.
     *
     * Null `mem` or invalid UTF-8 in any string silently drops the
     * event (the Rust side returns without panicking).
     */
    external fun sm_memory_update(
        mem: Long,
        subject: String,
        predicate: String,
        `object`: String,
    )

    /**
     * Sign-project `M + D + E` and write the resulting hypervector
     * as exactly [PACKED_LEN] bytes into `outPackedBuffer`.
     *
     * `outPackedBuffer` MUST be a direct buffer of capacity
     * `>= PACKED_LEN` (the Rust side writes the first `PACKED_LEN`
     * bytes). The JNI registration enforces direct-buffer addressing
     * so the write hits caller memory with zero intermediate copy.
     *
     * Null `mem` or a null underlying address is a silent no-op.
     */
    external fun sm_memory_working_memory(mem: Long, outPackedBuffer: ByteBuffer)

    /**
     * Bipolar cosine similarity over two pre-packed buffers.
     * Returns `(D - 2 * popcount(a XOR b)) / D` in `[-1.0, 1.0]`.
     *
     * Both buffers MUST be direct, of capacity `>= PACKED_LEN`.
     * Either-or-both null buffers yield `0.0f`.
     */
    external fun sm_hypervector_similarity(aPacked: ByteBuffer, bPacked: ByteBuffer): Float
}
