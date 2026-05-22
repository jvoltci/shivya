//! C-compatible FFI surface for `shivya-mind`.
//!
//! The interface presents opaque pointer handles to two object lifetimes
//! (`Codebook`, `Memory`) plus three hot-path entry points used by the
//! mobile companion app: `update`, `working_memory`, and a free-standing
//! `similarity` over two pre-packed bipolar buffers.
//!
//! Conventions:
//!
//! * Every entry point is wrapped in [`catch_unwind`] so a Rust panic
//!   cannot cross the ABI boundary and crash the host process.
//! * Raw pointers are validated with [`<*const T>::as_ref`] /
//!   [`<*mut T>::as_mut`] in `if let Some(_)` form; null inputs are
//!   tolerated and produce either a null return or a silent no-op.
//! * The packed bipolar buffer the mobile side allocates is exactly
//!   [`PACKED_LEN`] = `ceil(D / 8) = 1_250` bytes. Bits are written in
//!   big-endian byte order within each 64-bit chunk so byte 0's MSB is
//!   logical bit 0.
//! * The hot-path functions allocate nothing on the FFI layer itself;
//!   any heap traffic is whatever the underlying engine already does.

use crate::codebook::{Codebook, DEFAULT_SALT};
use crate::memory::{Event, Memory};
use crate::vsa::{D, WORDS};
use std::ffi::CStr;
use std::os::raw::c_char;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr;
use std::sync::Arc;

/// Length in bytes of the packed bipolar buffer exchanged across FFI.
///
/// `D = 10_000` logical bits, packed eight-to-a-byte, with no padding.
pub const PACKED_LEN: usize = 1_250;

const FULL_U64_CHUNKS: usize = PACKED_LEN / 8;
const TAIL_BYTES: usize = PACKED_LEN - FULL_U64_CHUNKS * 8;

/// Construct a fresh codebook from `salt_ptr[..salt_len]`. If the pointer
/// is null or the length is zero the engine default salt is used. Returns
/// an opaque handle; callers must release it with [`sm_codebook_free`].
///
/// The handle is, internally, `Box<Arc<Codebook>>` cast to `*mut Codebook`
/// so that multiple `Memory` instances can share one codebook without
/// disturbing the C-visible pointer identity.
/// # Safety
///
/// If `salt_ptr` is non-null, the caller must guarantee that
/// `salt_ptr[..salt_len]` is a readable slice of initialised bytes for
/// the duration of the call. Passing `(null, _)` or `(_, 0)` is always
/// safe and selects the engine default salt.
#[no_mangle]
pub unsafe extern "C" fn sm_codebook_new(salt_ptr: *const u8, salt_len: usize) -> *mut Codebook {
    catch_unwind(AssertUnwindSafe(|| {
        let salt: &[u8] = if salt_ptr.is_null() || salt_len == 0 {
            DEFAULT_SALT
        } else {
            unsafe { std::slice::from_raw_parts(salt_ptr, salt_len) }
        };
        let handle: Box<Arc<Codebook>> = Box::new(Arc::new(Codebook::new(salt)));
        Box::into_raw(handle).cast::<Codebook>()
    }))
    .unwrap_or(ptr::null_mut())
}

/// Release a codebook handle previously produced by [`sm_codebook_new`].
/// Null input is a no-op. Memories created from this codebook keep their
/// own `Arc` clone and continue to function until they are freed too.
/// # Safety
///
/// `cb` must either be null or a handle previously returned by
/// [`sm_codebook_new`] that has not yet been freed. Each handle must be
/// passed to this function at most once.
#[no_mangle]
pub unsafe extern "C" fn sm_codebook_free(cb: *mut Codebook) {
    if cb.is_null() {
        return;
    }
    let _ = catch_unwind(AssertUnwindSafe(|| unsafe {
        drop(Box::from_raw(cb.cast::<Arc<Codebook>>()));
    }));
}

/// Construct a new tri-tier memory bound to the given codebook handle.
/// The codebook's reference count is bumped; the codebook handle remains
/// owned by the caller and must still be released with
/// [`sm_codebook_free`]. Returns null on a null input or panic.
/// # Safety
///
/// `cb` must either be null or a live handle previously returned by
/// [`sm_codebook_new`]. The codebook handle is borrowed immutably; the
/// caller retains ownership and must still free it independently.
#[no_mangle]
pub unsafe extern "C" fn sm_memory_new(cb: *mut Codebook) -> *mut Memory {
    catch_unwind(AssertUnwindSafe(|| {
        let handle = cb.cast::<Arc<Codebook>>();
        if let Some(arc) = unsafe { handle.as_ref() } {
            let mem = Memory::new(Arc::clone(arc));
            Box::into_raw(Box::new(mem))
        } else {
            ptr::null_mut()
        }
    }))
    .unwrap_or(ptr::null_mut())
}

/// Release a memory handle previously produced by [`sm_memory_new`].
/// Null input is a no-op. The bound codebook is unaffected.
/// # Safety
///
/// `mem` must either be null or a handle previously returned by
/// [`sm_memory_new`] that has not yet been freed. Each handle must be
/// passed to this function at most once.
#[no_mangle]
pub unsafe extern "C" fn sm_memory_free(mem: *mut Memory) {
    if mem.is_null() {
        return;
    }
    let _ = catch_unwind(AssertUnwindSafe(|| unsafe {
        drop(Box::from_raw(mem));
    }));
}

/// Ingest one (subject, predicate, object) triple into the open episode
/// buffer of `mem`. All three string pointers must be valid, null-
/// terminated UTF-8. Invalid UTF-8, null arguments, or a null memory
/// handle silently drop the event.
/// # Safety
///
/// `mem` must be null or a live `*mut Memory` from [`sm_memory_new`].
/// Each of `subject`, `predicate`, `object` must be null or point at a
/// NUL-terminated, valid-UTF-8 byte sequence that remains readable for
/// the duration of the call. Any null or non-UTF-8 argument drops the
/// event silently.
#[no_mangle]
pub unsafe extern "C" fn sm_memory_update(
    mem: *mut Memory,
    subject: *const c_char,
    predicate: *const c_char,
    object: *const c_char,
) {
    let _ = catch_unwind(AssertUnwindSafe(|| {
        if let Some(m) = unsafe { mem.as_mut() } {
            let s = match cstr_str(subject) {
                Some(s) => s,
                None => return,
            };
            let p = match cstr_str(predicate) {
                Some(p) => p,
                None => return,
            };
            let o = match cstr_str(object) {
                Some(o) => o,
                None => return,
            };
            m.update(&Event::new(s, p, o));
        }
    }));
}

/// Sign-project `M + D + E` and write the resulting hypervector as
/// [`PACKED_LEN`] bytes into the caller-supplied buffer at
/// `out_packed_ptr`. Byte `i` contains bits `8*i .. 8*i + 8`, with bit
/// `8*i` in the most significant position of the byte. The caller must
/// guarantee at least [`PACKED_LEN`] writable bytes at the target. Null
/// inputs are silent no-ops.
/// # Safety
///
/// `mem` must be null or a live handle from [`sm_memory_new`]. If
/// `out_packed_ptr` is non-null it must point at a writable buffer of
/// at least [`PACKED_LEN`] bytes; the function writes exactly that
/// many bytes. Passing null for either pointer is a silent no-op.
#[no_mangle]
pub unsafe extern "C" fn sm_memory_working_memory(mem: *mut Memory, out_packed_ptr: *mut u8) {
    let _ = catch_unwind(AssertUnwindSafe(|| {
        if out_packed_ptr.is_null() {
            return;
        }
        if let Some(m) = unsafe { mem.as_mut() } {
            let wm = m.working_memory();
            let out = unsafe { std::slice::from_raw_parts_mut(out_packed_ptr, PACKED_LEN) };
            pack_hypervector(&wm, out);
        }
    }));
}

/// Bipolar cosine similarity over two pre-packed buffers of length
/// [`PACKED_LEN`]. Returns `(D - 2 * popcount(a XOR b)) / D` in
/// `[-1.0, 1.0]`. Null inputs yield `0.0`.
///
/// The loop reads each operand as 156 little-endian-agnostic `u64`
/// chunks plus a 2-byte tail and folds `count_ones` over their XOR; on
/// x86-64 / aarch64 this lowers to the hardware popcount instruction.
/// # Safety
///
/// If `a_packed` (resp. `b_packed`) is non-null it must point at a
/// readable buffer of at least [`PACKED_LEN`] initialised bytes. Either
/// or both pointers may be null, in which case the function returns
/// `0.0` without dereferencing.
#[no_mangle]
pub unsafe extern "C" fn sm_hypervector_similarity(a_packed: *const u8, b_packed: *const u8) -> f32 {
    catch_unwind(AssertUnwindSafe(|| {
        if a_packed.is_null() || b_packed.is_null() {
            return 0.0_f32;
        }
        let a = unsafe { std::slice::from_raw_parts(a_packed, PACKED_LEN) };
        let b = unsafe { std::slice::from_raw_parts(b_packed, PACKED_LEN) };
        let mut diff: u32 = 0;
        for i in 0..FULL_U64_CHUNKS {
            let off = i * 8;
            let aw = u64::from_be_bytes([
                a[off], a[off + 1], a[off + 2], a[off + 3],
                a[off + 4], a[off + 5], a[off + 6], a[off + 7],
            ]);
            let bw = u64::from_be_bytes([
                b[off], b[off + 1], b[off + 2], b[off + 3],
                b[off + 4], b[off + 5], b[off + 6], b[off + 7],
            ]);
            diff += (aw ^ bw).count_ones();
        }
        let tail_off = FULL_U64_CHUNKS * 8;
        for i in 0..TAIL_BYTES {
            diff += (a[tail_off + i] ^ b[tail_off + i]).count_ones();
        }
        (D as f32 - 2.0 * diff as f32) / D as f32
    }))
    .unwrap_or(0.0)
}

fn cstr_str<'a>(p: *const c_char) -> Option<&'a str> {
    if p.is_null() {
        return None;
    }
    unsafe { CStr::from_ptr(p) }.to_str().ok()
}

/// Big-endian serialise the first `D` bits of `hv` into `out` (length
/// `PACKED_LEN`). With `u32` storage the trailing 16 padding bits of
/// the last word are dropped on the floor; only the top 16 bits of
/// word 312 land in the output as the final two bytes.
fn pack_hypervector(hv: &crate::vsa::Hypervector, out: &mut [u8]) {
    debug_assert_eq!(out.len(), PACKED_LEN);
    let mut cursor = 0usize;
    for w in 0..WORDS {
        let be = hv.data[w].to_be_bytes();
        let remaining = PACKED_LEN - cursor;
        let take = remaining.min(be.len());
        out[cursor..cursor + take].copy_from_slice(&be[..take]);
        cursor += take;
        if cursor == PACKED_LEN {
            break;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_round_trip_under_similarity() {
        // Packing a vector and comparing it to itself via the FFI
        // similarity function must yield exactly 1.0.
        use crate::vsa::{random_hypervector, Pcg32};
        let mut rng = Pcg32::new(7, 11);
        let v = random_hypervector(&mut rng);
        let mut buf = [0u8; PACKED_LEN];
        pack_hypervector(&v, &mut buf);
        let s = unsafe { sm_hypervector_similarity(buf.as_ptr(), buf.as_ptr()) };
        assert!((s - 1.0).abs() < 1e-6, "self-similarity = {s}");
    }

    #[test]
    fn null_inputs_are_safe() {
        unsafe {
            sm_codebook_free(ptr::null_mut());
            sm_memory_free(ptr::null_mut());
            assert!(sm_memory_new(ptr::null_mut()).is_null());
            sm_memory_update(
                ptr::null_mut(),
                ptr::null(),
                ptr::null(),
                ptr::null(),
            );
            sm_memory_working_memory(ptr::null_mut(), ptr::null_mut());
            assert_eq!(sm_hypervector_similarity(ptr::null(), ptr::null()), 0.0);
        }
    }
}
