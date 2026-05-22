# Android JNI Binding Contract for `shivya-mind`

This document specifies the contract between an Android host process and
the `shivya-mind` cognitive core via its `extern "C"` FFI surface
(`crates/shivya-mind/src/ffi.rs`). It is the authoritative reference for
JNI implementers; the Rust side will not change without a corresponding
update here.

The native library is expected to be packaged into the Android APK as
`libshivya_mind.so` (one per ABI: `arm64-v8a`, `armeabi-v7a`,
`x86_64`). The same `.so` exposes every entry point listed below; there
is no Android-specific shim crate.

---

## 1. The C ABI surface (recap)

The Rust side exposes five `#[no_mangle] extern "C"` entry points and
one compile-time constant. All entry points are wrapped in
`std::panic::catch_unwind` so a panic in Rust cannot cross the ABI
boundary into the JVM.

| Rust signature | Purpose | Allocates? |
|---|---|---|
| `sm_codebook_new(salt_ptr: *const u8, salt_len: usize) -> *mut Codebook` | Construct a deterministic codebook bound to `salt[..len]`. Pass `(null, 0)` for the engine default salt. | Yes (one `Box<Arc<Codebook>>`). |
| `sm_codebook_free(cb: *mut Codebook)` | Release a codebook handle. Null-safe no-op. Other `Memory`s sharing it via `Arc` keep working until they too are freed. | No. Drops a `Box`. |
| `sm_memory_new(cb: *mut Codebook) -> *mut Memory` | Construct a tri-tier (`E`, `D`, `M`) memory bound to `cb`. Bumps the codebook's `Arc` refcount; the caller still owns `cb`. Returns null on null input. | Yes (one `Box<Memory>`, three tally buffers ≈ 120 KB). |
| `sm_memory_free(mem: *mut Memory)` | Release a memory handle. Null-safe no-op. The bound codebook is unaffected. | No. Drops a `Box`. |
| `sm_memory_update(mem, subject, predicate, object)` | Ingest one `(s, p, o)` triple into the open episode buffer. All three are null-terminated UTF-8. Invalid UTF-8, null pointers, or a null `mem` silently drop the event. | Bounded; whatever the engine internally does per update. |
| `sm_memory_working_memory(mem, out_packed_ptr: *mut u8)` | Sign-project `M + D + E` and write `PACKED_LEN = 1_250` bytes (10,000 packed bits, big-endian byte order within each 64-bit chunk, MSB-first) into the caller's buffer. Null-safe no-op. | No (writes into caller-owned memory). |
| `sm_hypervector_similarity(a_packed, b_packed: *const u8) -> f32` | Bipolar cosine over two `PACKED_LEN`-byte buffers. Returns `(D − 2·popcount(a XOR b)) / D` in `[−1.0, 1.0]`. Null inputs yield `0.0`. | No. Stack-resident u32 accumulator over 156 `u64` chunks + 2-byte tail. |
| `PACKED_LEN: usize = 1_250` (compile-time constant) | Exactly `ceil(D / 8)` for `D = 10_000`. Mirror it as a Kotlin constant; do not size buffers dynamically. | n/a |

All `*mut Codebook` and `*mut Memory` handles are **opaque pointers**.
The JVM must never dereference them, alter them, or compare them for
ordering. The only valid operations are: pass to a `sm_…` entry point;
hold; eventually pass to the matching `…_free`.

---

## 2. Kotlin `external` mapping (`internal object ShivyaMindNative`)

The Kotlin side mirrors the C ABI 1:1. There is one object, all
`external`, all `@JvmStatic`-equivalent (objects already give this).
Handles are typed as `Long` (a JNI `jlong`, large enough for any 64-bit
pointer).

```kotlin
package io.github.jvoltci.shivya.mind

import java.nio.ByteBuffer

/**
 * Direct one-to-one mapping of the `sm_…` C ABI from
 * `crates/shivya-mind/src/ffi.rs`. Do not call any of these methods
 * from outside `ShivyaMind` — the safe wrapper class owns lifecycle.
 */
internal object ShivyaMindNative {

    init {
        // The .so is shipped under jniLibs/<ABI>/libshivya_mind.so.
        // `System.loadLibrary` is the one place JNI symbols are
        // resolved; if this throws, every `external` call below will
        // throw `UnsatisfiedLinkError` until process restart.
        System.loadLibrary("shivya_mind")
    }

    /** Mirrors `PACKED_LEN`. Hardcoded to match the Rust constant. */
    const val PACKED_LEN: Int = 1_250

    /**
     * Construct a codebook. `salt` may be empty; passing an empty
     * array routes through to `DEFAULT_SALT` on the Rust side.
     * Returns 0L on allocation failure (panic caught in Rust).
     */
    external fun codebookNew(salt: ByteArray): Long

    /** No-op on `0L`. After this call `handle` must not be reused. */
    external fun codebookFree(handle: Long)

    /**
     * Bind a fresh `Memory` to `codebookHandle`. The codebook handle
     * remains owned by the caller. Returns 0L if `codebookHandle`
     * is 0L or a panic was caught.
     */
    external fun memoryNew(codebookHandle: Long): Long

    /** No-op on `0L`. */
    external fun memoryFree(handle: Long)

    /**
     * Ingest one `(subject, predicate, object)` triple. Any of the
     * three strings being null is treated as a silent drop on the
     * Rust side, but Kotlin's type system already excludes that.
     * UTF-8 conversion happens in the JNI bridge; Kotlin `String`
     * encodes as UTF-16 internally so an explicit encode is implied.
     */
    external fun memoryUpdate(
        handle: Long,
        subject: String,
        predicate: String,
        `object`: String,
    )

    /**
     * Sign-project `M + D + E` into `outBuffer`. The buffer **must**
     * be a direct `ByteBuffer` with `remaining() >= PACKED_LEN`. The
     * JNI side calls `GetDirectBufferAddress` and writes through the
     * raw pointer — heap-backed `ByteBuffer`s will throw
     * `IllegalArgumentException`.
     */
    external fun memoryWorkingMemory(handle: Long, outBuffer: ByteBuffer)

    /**
     * Bipolar cosine over two direct `ByteBuffer`s, each of length
     * exactly `PACKED_LEN`. Same direct-buffer requirement as
     * `memoryWorkingMemory`. Returns 0.0f if either argument fails
     * the direct-buffer / length precondition (precondition is
     * checked on the Kotlin side before the JNI call).
     */
    external fun hypervectorSimilarity(a: ByteBuffer, b: ByteBuffer): Float
}
```

### 2.1 Why `ByteBuffer` and not `ByteArray`

`ByteArray` is a JVM-managed `byte[]`: the GC owns it, it may move, and
`GetByteArrayElements` returns either a copy (almost always, on Android)
or a pinned pointer that must be released via
`ReleaseByteArrayElements`. Every JNI call that touches a `ByteArray`
incurs a 1,250-byte copy in *each direction*. Over a step loop running
at 1 Hz that is 30 KB/s wasted memory traffic; over background mining
of foreground-package events it is much worse.

`ByteBuffer.allocateDirect(PACKED_LEN)` allocates 1,250 bytes
**outside** the JVM heap, in a region the GC will not relocate. The
JNI bridge uses `GetDirectBufferAddress` to recover a `u8*` and writes
through it without copy. The same buffer can be reused across calls;
allocate once at app start, write/read in place forever.

The Rust side already assumes "exactly `PACKED_LEN` writable bytes at
the target pointer" (`sm_memory_working_memory` doc comment). A direct
`ByteBuffer` is the only Kotlin allocation that delivers that contract
without a copy.

### 2.2 The packed-bit layout (do not invent your own)

The Rust packer (`pack_hypervector` in `ffi.rs`) writes:

- Byte `i` carries logical bits `8·i .. 8·i + 8` of the bipolar
  hypervector.
- Within each byte, bit `8·i` lands in the **most significant
  position**: `out[i] >> 7 & 1` is logical bit `8·i`, `out[i] & 1` is
  logical bit `8·i + 7`.
- The first `WORDS = 157` `u64` words of the hypervector are written
  big-endian: word `w`'s most significant byte is `out[8·w]`.
- The final word (word 156) only contributes its top 16 bits — the
  trailing `WORDS·64 − D = 48` padding bits are dropped on the floor
  and **must be ignored** by any Kotlin reader.

The Kotlin side should treat the buffer as opaque between
`memoryWorkingMemory` and `hypervectorSimilarity`. Any bit-twiddling on
the JVM side is a correctness hazard.

---

## 3. The safe Kotlin wrapper

The raw `ShivyaMindNative` object is `internal`. Public surface goes
through a `ShivyaMind` class that owns lifecycle and guarantees the
free-on-close contract.

```kotlin
package io.github.jvoltci.shivya.mind

import java.io.Closeable
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.atomic.AtomicLong

/**
 * Owning handle for a `Memory` + its `Codebook`. Single-thread
 * affinity: all calls must come from the same thread that constructed
 * the instance. If you need concurrent ingestion, gate updates through
 * a single-threaded executor (typical pattern for on-device telemetry).
 *
 * Always use as `ShivyaMind(salt).use { … }` or wire to a
 * `LifecycleObserver` so `close()` runs deterministically. The
 * finalizer is a backstop, not a primary cleanup path.
 */
class ShivyaMind(salt: ByteArray = byteArrayOf()) : Closeable {

    private val codebook: AtomicLong = AtomicLong(ShivyaMindNative.codebookNew(salt))
    private val memory: AtomicLong = AtomicLong(
        ShivyaMindNative.memoryNew(codebook.get())
            .also { check(it != 0L) { "memoryNew returned null; codebook handle was likely 0L" } }
    )

    private val workingMemoryBuffer: ByteBuffer =
        ByteBuffer.allocateDirect(ShivyaMindNative.PACKED_LEN)
            .order(ByteOrder.BIG_ENDIAN) // matches the Rust packer convention

    init {
        check(codebook.get() != 0L) { "codebookNew returned null; out of memory?" }
    }

    /** Push one `(s, p, o)` triple into the open episode buffer. */
    fun observe(subject: String, predicate: String, `object`: String) {
        val h = memory.get()
        if (h == 0L) return
        ShivyaMindNative.memoryUpdate(h, subject, predicate, `object`)
    }

    /**
     * Return a *snapshot* of the working-memory hypervector as a
     * read-only view onto the internal direct buffer. The view is
     * invalidated by the next call to `workingMemory()`; callers that
     * need to keep the bytes around must copy them out themselves.
     */
    fun workingMemory(): ByteBuffer {
        val h = memory.get()
        if (h == 0L) {
            workingMemoryBuffer.clear()
            // Zero-fill on the JVM side so a leaked handle never
            // surfaces stale bytes.
            for (i in 0 until ShivyaMindNative.PACKED_LEN) workingMemoryBuffer.put(i, 0)
            return workingMemoryBuffer.asReadOnlyBuffer()
        }
        workingMemoryBuffer.clear()
        ShivyaMindNative.memoryWorkingMemory(h, workingMemoryBuffer)
        workingMemoryBuffer.position(ShivyaMindNative.PACKED_LEN)
        workingMemoryBuffer.flip()
        return workingMemoryBuffer.asReadOnlyBuffer()
    }

    /**
     * Idempotent. Calling twice is safe; calling on a closed instance
     * is a silent no-op. Releases the `Memory` first (drops the
     * 120 KB tally arrays), then the `Codebook` (drops the Arc; if
     * other instances still hold it, only the refcount decrements).
     */
    override fun close() {
        val m = memory.getAndSet(0L)
        if (m != 0L) ShivyaMindNative.memoryFree(m)
        val c = codebook.getAndSet(0L)
        if (c != 0L) ShivyaMindNative.codebookFree(c)
    }

    @Deprecated("Backstop only; rely on close().", level = DeprecationLevel.WARNING)
    protected fun finalize() {
        close()
    }
}

/** Free-standing similarity over two snapshot buffers. */
fun cosineSimilarity(a: ByteBuffer, b: ByteBuffer): Float {
    require(a.isDirect && b.isDirect) {
        "Both buffers must be direct (ByteBuffer.allocateDirect)"
    }
    require(a.remaining() == ShivyaMindNative.PACKED_LEN) {
        "a.remaining() = ${a.remaining()}, expected ${ShivyaMindNative.PACKED_LEN}"
    }
    require(b.remaining() == ShivyaMindNative.PACKED_LEN) {
        "b.remaining() = ${b.remaining()}, expected ${ShivyaMindNative.PACKED_LEN}"
    }
    return ShivyaMindNative.hypervectorSimilarity(a, b)
}
```

---

## 4. Memory safety surface

The host process is a long-running Android service. Three concrete
constraints fall out of that:

### 4.1 `close()` is mandatory; the finalizer is a backstop

Every `ShivyaMind` instance owns ≈ 120 KB of native heap (three
10,000-bit tally buffers as `f32` accumulators in the long-term
memory's open episode / day / consolidated tiers). The JVM finalizer
fires at GC discretion, which on Android can be tens of seconds to
minutes after the last reference drops. A service that opens and
closes ShivyaMind instances under load (e.g., per-foreground-app
sessions) will accumulate native-heap pressure that the Dalvik /
ART heap pressure signal cannot see.

Anchor `close()` to a deterministic event:

* `Activity.onDestroy()` / `Fragment.onDestroyView()` for UI-bound
  instances.
* `Service.onDestroy()` for service-scoped instances.
* `LifecycleObserver.onStateChanged(ON_DESTROY)` if the consumer
  follows AndroidX Lifecycle.
* `kotlin.use { … }` for short-lived block-scoped usage.

The included `finalize()` is annotated `@Deprecated` to discourage
relying on it. It exists only so that a forgotten close in a debug
build does not silently leak forever.

### 4.2 Free order: memory first, codebook last

The Rust side holds the codebook by `Arc<Codebook>` and clones the
`Arc` into each `Memory`. Freeing the codebook *before* the memory
will not crash — the memory's own `Arc` clone keeps the codebook alive
— but it leaks the codebook from the JNI side's accounting because the
`Box<Arc<Codebook>>` allocated by `sm_codebook_new` is dropped, while
the `Memory` continues to hold (and eventually drop) its clone.

To avoid that confusion, the `close()` implementation above always
frees memory first, then codebook. Match this order if you write your
own wrapper.

### 4.3 Double-free protection

`sm_codebook_free` and `sm_memory_free` are documented null-safe but
**not** double-free-safe. Passing the same non-null pointer twice is
undefined behaviour. The `AtomicLong.getAndSet(0L)` pattern in
`close()` ensures the C side sees each handle exactly once: the second
caller of `close()` observes `0L` and skips the FFI call.

If you write the wrapper differently (e.g., with `Long` properties
guarded by a `Mutex`), preserve this property. The standard JVM
"already-closed" exception pattern is acceptable; silently re-freeing
is not.

### 4.4 Buffer-pointer lifetime under direct `ByteBuffer`s

`GetDirectBufferAddress` returns a pointer that is valid for the
**lifetime of the JVM `ByteBuffer` object**. The JVM cannot move
direct buffers, but it can free the underlying native allocation when
the buffer is GC'd. Hold a strong reference to the `ByteBuffer` in
Kotlin for the entire call window. The wrapper above does this
implicitly by storing `workingMemoryBuffer` as a `val` on the
`ShivyaMind` instance.

Do **not** call `memoryWorkingMemory` with a buffer that was sliced /
duplicated / re-wrapped: those produce new `ByteBuffer` objects whose
direct-address contract depends on the platform implementation.
Always pass the original direct buffer.

### 4.5 Reference counting across multiple `Memory` instances (advanced)

If your app needs multiple `ShivyaMind` instances that share a single
codebook (e.g., a "memory per active conversation, codebook per
user"), do not rebuild the codebook each time. Hold a single shared
codebook handle at the application scope and pass it to
`ShivyaMindNative.memoryNew(...)` repeatedly.

This requires bypassing the safe wrapper. Document that bypass
clearly: the shared codebook must outlive every memory it backs, and
must be freed exactly once at process shutdown. The Rust side's `Arc`
refcount will keep the codebook alive even if the JNI handle is freed
early, but the JNI accounting will leak the `Box<Arc<Codebook>>`
overhead (one allocation per leaked handle, on the order of 24 bytes).

---

## 5. On-device telemetry → `(subject, predicate, object)` pipeline

The Memory engine ingests strongly-typed triples; the Android host's
job is to convert raw OS events into a small, stable vocabulary that
keeps the codebook bounded.

### 5.1 Sources we expect to be parsed cleanly

| Android source | Permission / API | Event shape |
|---|---|---|
| Foreground app changes | `UsageStatsManager.queryEvents(beginTime, endTime)` filtering `ACTIVITY_RESUMED` / `ACTIVITY_PAUSED`. Requires `PACKAGE_USAGE_STATS` (Settings → Special access). | `("user", "opened", "<package_name>")` and `("user", "closed", "<package_name>")` |
| Notifications | `NotificationListenerService.onNotificationPosted(sbn)`. Requires user opt-in via Notification Access. | `("<source_package>", "notified", "<channel_id>")` for the *fact* that something arrived; **never** the notification text. |
| Screen state | `ACTION_SCREEN_ON` / `ACTION_SCREEN_OFF` broadcast (no permission). | `("device", "screen", "on" \| "off")` |
| Connectivity transitions | `ConnectivityManager.registerDefaultNetworkCallback`. | `("device", "network", "wifi" \| "cellular" \| "none")` |

Do not feed timestamps as objects — the segmenter already has access
to wall-clock time through its own scheduler. Time appears in the
algebra implicitly via decay, not as a discrete event.

### 5.2 Canonicalisation rules

The codebook is a deterministic blake3-keyed mapping from label
strings to 10,000-bit hypervectors. Two distinct labels collide at
roughly `2 / D ≈ 0.0002` (the bipolar similarity floor), so the
vocabulary can be large without interference — but the labels must be
**stable**:

1. **Lowercase package names verbatim.** `com.android.chrome`, not
   `Chrome` or the user-facing label.
2. **Channel IDs verbatim**, the `NotificationChannel.id` string. Do
   not collapse multiple channels into the channel *name*.
3. **No PII anywhere.** Subject, predicate, and object are all
   bound into the hypervector and can in principle be unbound by
   anyone with the codebook salt. Treat the salt as device-private
   and never feed raw text content from notifications, SMS, or
   foreground UI scraping.
4. **Stable predicate vocabulary.** Pick a small closed set —
   `opened`, `closed`, `notified`, `screen`, `network`, `charging` —
   and never extend it from the Kotlin side without a paired update
   here. The Rust segmenter's predictive surprise is calibrated to
   the empirical predicate distribution; extending it after
   deployment confuses the surprise baseline.

### 5.3 Sketch of the listener wiring

```kotlin
import android.app.usage.UsageStatsManager
import android.content.Context
import androidx.lifecycle.LifecycleService
import io.github.jvoltci.shivya.mind.ShivyaMind

class ShivyaMindIngestService : LifecycleService() {

    private lateinit var mind: ShivyaMind
    private lateinit var usage: UsageStatsManager

    override fun onCreate() {
        super.onCreate()
        mind = ShivyaMind() // default salt; replace with user-private salt for prod
        usage = getSystemService(Context.USAGE_STATS_SERVICE) as UsageStatsManager
        // schedule pollForegroundEvents() on the lifecycle scope at e.g. 1 Hz
    }

    private fun pollForegroundEvents(beginMs: Long, endMs: Long) {
        val events = usage.queryEvents(beginMs, endMs)
        val ev = android.app.usage.UsageEvents.Event()
        while (events.hasNextEvent()) {
            events.getNextEvent(ev)
            val pkg = ev.packageName ?: continue
            when (ev.eventType) {
                android.app.usage.UsageEvents.Event.ACTIVITY_RESUMED ->
                    mind.observe("user", "opened", pkg)
                android.app.usage.UsageEvents.Event.ACTIVITY_PAUSED ->
                    mind.observe("user", "closed", pkg)
                // ignore everything else; the segmenter benefits from
                // a *sparse* event stream
            }
        }
    }

    override fun onDestroy() {
        mind.close()
        super.onDestroy()
    }
}
```

The `NotificationListenerService` integration is structurally
identical — same `observe()` call, different event source. Keep both
behind explicit user opt-in.

### 5.4 What goes back out

The Kotlin side can ask the memory for its current working hypervector
at any time:

```kotlin
val snapshot: ByteBuffer = mind.workingMemory() // read-only direct slice
```

That snapshot is what gets passed to `cosineSimilarity(snapshot, probe)`
to ask questions like *"how similar is the current memory state to a
probe vector I have lying around?"* The probe vector might come from a
saved baseline, a different memory instance, or a server-pushed
hypervector. The probe is opaque without the codebook salt, so the
similarity score is the only signal that crosses the trust boundary.

---

## 6. Build / packaging checklist

1. Add the Rust target list to `cargo build` — `aarch64-linux-android`,
   `armv7-linux-androideabi`, `x86_64-linux-android`. Use a stable
   NDK r25 or newer.
2. Place the resulting `libshivya_mind.so` under
   `app/src/main/jniLibs/<abi>/libshivya_mind.so`. Do **not** rename;
   `System.loadLibrary("shivya_mind")` strips the `lib` prefix and
   the `.so` suffix.
3. Strip debug symbols in release builds (`strip --strip-unneeded`).
   The exported symbols are only the seven `sm_…` entry points; the
   strip will not affect them.
4. The Kotlin code above must be compiled with `freeCompilerArgs +=
   "-Xjvm-default=all"` to satisfy the `Closeable.use` interop.
5. Add a CI smoke test that constructs a `ShivyaMind`, ingests one
   triple, reads `workingMemory()`, and closes — guarding against
   the `.so` being misnamed or missing for any ABI.

---

## 7. Out of scope for this document

* Cross-device sync (the algebra is naturally CRDT-clean but the
  transport, codebook-salt distribution, and consent model are
  product decisions, not JNI decisions).
* Encrypted-at-rest persistence of the hypervector (Android's
  `EncryptedFile` is the recommended path; format is whatever the
  consuming code chooses, since the hypervector is itself opaque).
* UI scaffolding, settings screens, or Jetpack Compose previews.
  Strictly product-side; not part of the binding contract.

The binding contract is the seven C entry points, the direct-buffer
allocation rule, the free-order rule, and the canonical predicate
vocabulary. Everything else is a host concern.
