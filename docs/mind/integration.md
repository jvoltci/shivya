# Shivya Mind — Integration Handbook

Audience: mobile app engineers (SwiftUI / Kotlin Compose), daemon authors,
and FFI consumers porting the engine to Rust, Swift, or Kotlin. The
engine itself is in `core/` and is intentionally opinion-free about how
events arrive; this document specifies the contract.

This is a wire-format and ABI document, not a tutorial. For the math,
read [architecture.md](architecture.md); for the philosophy, read
[philosophy.md](philosophy.md).

---

## 1. Telemetry → VSA event tuples

The engine accepts events as `(subject, predicate, object [, ctx, t])`
tuples. The string atoms are encoded into the codebook on first sight
and then referenced as deterministic 10,000-D bipolar hypervectors for
the lifetime of the device (the codebook salt is the only thing that
ties them to a particular device's symbol table).

### 1.1 Canonical telemetry mappings

Three telemetry sources are first-class:

| Source              | subject       | predicate       | object              | ctx (optional)       |
|---------------------|---------------|-----------------|---------------------|----------------------|
| **App switch**      | `user`        | `opened`        | `<package_name>`    | `<app_category>`     |
| **Text keystroke**  | `user`        | `typed`         | `<token_or_hash>`   | `<app_package>`      |
| **Clipboard**       | `user`        | `copied`        | `<sha8_of_payload>` | `<app_package>`      |
| **Notification**    | `<app>`       | `notified`      | `<channel_id>`      | `<priority_label>`   |
| **Location class**  | `user`        | `entered`       | `<place_class>`     | `<device_state>`     |
| **Foreground app**  | `<app>`       | `is_foreground` | `user`              | `<app_category>`     |

Atoms MUST be UTF-8, MUST NOT start with the double-underscore prefix
`__` (reserved for internal codebook structure: role atoms, time base),
and SHOULD be lowercased and free of whitespace. Long free-text payloads
SHOULD be hashed (BLAKE2b-8 hex is the recommended canonical form) to
preserve the codebook's bounded vocabulary commitment of `N ≤ 8192`.

### 1.2 Vocabulary budget

The codebook is lazy and deterministic but its empirical size affects
two things:

- **N-gram predictor footprint** in `meta.json` scales with `V × K`
  where `K = 32` is the top-K transition cap (see
  [store/persistence.py](../store/persistence.py)). Above `V = 8192`,
  the calibration becomes wasteful per byte.
- **Cleanup cost** in `query()`, `resolve()`, and bead thumbnails scales
  with the number of candidate labels under consideration. For
  on-device use, keep your active vocabulary under 4096 atoms.

If a telemetry source generates unbounded text (chat content, URLs),
hash it to a fixed-vocabulary token before ingestion.

### 1.3 Time and context

- `t` is epoch seconds (float). When non-null, the engine encodes the
  WHEN role as a cyclic permutation of a single deterministic time-base
  vector. Wall-clock granularity is one second.
- `ctx` is one optional atom bound to the `APP_CTX` role. For richer
  context (location + device + app), pre-compose a composite token
  upstream rather than asking the engine for variadic context slots.

---

## 2. IPC architecture

Mobile apps and the on-device daemon communicate with the memory
process over a per-platform local channel:

| Platform | Transport               | Framing                              |
|----------|-------------------------|--------------------------------------|
| Linux / macOS / iOS sandbox | Unix domain socket | Length-prefixed JSON (4-byte big-endian length) |
| Android  | AIDL over Binder        | Length-prefixed JSON in a single byte buffer    |

### 2.1 Wire format

Every message is a JSON object with a `kind` discriminator. The full
request/response set is small enough to fit on one screen:

```jsonc
// Client -> Mind
{"kind":"observe", "event":{"subject":"user","predicate":"opened","object":"com.example.notes","ctx":"productivity","t":1716400000.0}}
{"kind":"query",   "role":"SUBJ", "top_k":5}
{"kind":"resolve", "subject":"alice", "predicate":"owns", "top_k":3}
{"kind":"forget",  "event":{"subject":"alice","predicate":"owns","object":"scooter"}, "weight":5.0}
{"kind":"timeline","limit":50}
{"kind":"save"}

// Mind -> Client
{"kind":"decision", "seal":true, "reason":"spike", "surprise":0.51, "bead":{...}}
{"kind":"candidates", "results":[["alice", 0.53], ["bob", 0.04]]}
{"kind":"stats", "M_mag":12.4, "n_beads":17, "EMA_mu":0.31, "EMA_sigma":0.08}
{"kind":"error",  "code":"unknown_atom", "detail":"..."}
```

### 2.2 EpisodeBead payload

Sealed beads cross the IPC boundary as compact JSON. The
[surprise.py](../core/surprise.py) `EpisodeBead` dataclass is canonical;
field names and types are stable across the boundary:

```json
{
  "id":            "63995dc40de4afa3",
  "t_start":       1716400000.0,
  "t_end":         1716400420.0,
  "n_events":      7,
  "reason":        "spike",
  "surprise_peak": 0.443,
  "thumbnail":     ["alice", "reviewed", "function_x"],
  "title":         null
}
```

`id` is the BLAKE2b-8 hex of `E_k` bytes concatenated with the seal
timestamp. It is stable across saves and load cycles: two clients
observing the same sealed bead independently will produce the same id.

### 2.3 Backpressure and ordering

- The client SHOULD send `observe` calls in arrival order. The engine
  does not reorder; the EMA and segmenter assume causal sequence.
- The engine never blocks a client thread on I/O; persistence is
  triggered explicitly via `save`. Streamed beads in `decision`
  messages are the only thing pushed to the client.
- If the socket buffer fills, dropped `observe` messages are recoverable
  (the surprise calibration will absorb the gap within the EMA half-
  life). Dropped `decision` messages are NOT recoverable client-side;
  treat the bead SQLite database (`episodes.db`) as the source of
  truth for the timeline.

---

## 3. FFI array layouts

Direct memory sharing between Python (NumPy) and native code is
supported for the three state arrays plus the bit-packed projection.
All arrays are little-endian on every supported target; the engine
asserts this on startup. Length-of-buffer in bytes is constant
per-build (a function of `D` only) and known statically to the
consumer.

### 3.1 Per-tier float tally (read/write)

The hot state — `M_tally`, `D_tally`, `E_tally` — is shared as
contiguous `float32` arrays of length `D`:

```c
typedef struct {
    const float* data;   // 4 * D bytes, row-major contiguous
    uint32_t     D;      // 10000 in default build
} sm_tally_view_t;
```

- **Byte order:** little-endian IEEE 754 binary32.
- **Stride:** `sizeof(float)`, dense.
- **Lifetime:** owned by the Python process; valid until the next
  `update`, `seal_episode`, `consolidate_day`, or `forget` call. The
  consumer MUST NOT cache the pointer across those mutating calls.
- **Mutation rules:** read-only from FFI. Writes from native code break
  the exact-inverse property of `forget()` and corrupt the EMA
  calibration; do not enable.

For `D = 10_000` the tally array is exactly **40,000 bytes** per tier.

### 3.2 Bit-packed read-only projection (1,250 bytes)

The shipping format for over-the-wire transfer and for
energy-constrained read paths is the sign-projected, bit-packed form
of `M`. Layout matches `core.packing` exactly:

```c
typedef struct {
    const uint8_t* data;  // ceil(D / 8) bytes; D=10000 -> 1250 bytes
    uint32_t       D;     // logical dimension count
    uint8_t        last_byte_mask;  // 0xFF or a left-aligned mask if D % 8 != 0
} sm_packed_view_t;
```

- **Bit order:** big-endian per byte (most-significant bit is dimension
  index `i = 8 * byte_index`).
- **Encoding:** `bit = 0` means bipolar `+1`; `bit = 1` means bipolar
  `-1`. The convention is fixed (see [core/packing.py](../core/packing.py)
  lines 8–9).
- **Padding:** when `D` is not a multiple of 8, padding bits in the
  final byte are always zero; `last_byte_mask` indicates which bits
  are payload. Native code performing popcount MUST AND with
  `last_byte_mask` first.
- **Stability:** byte-identical between any two builds with the same
  `D`, salt, and event history. This is the format suitable for
  CRDT-clean device sync (bundle is commutative and associative on the
  packed form via bit-majority).

### 3.3 Codebook regeneration

The codebook is NEVER shipped over FFI or IPC. Native code that needs
to compute its own probe vectors (e.g., a Rust port of `query()`)
regenerates them by hashing `salt || label` with BLAKE3 (preferred) or
BLAKE2b (fallback) and seeding a NumPy-compatible PCG64 stream. The
exact hash backend in use is recorded in `meta.json` under
`hash_backend`; consumers MUST honor it to get byte-identical vectors.

### 3.4 Role atom bit layout

Role atoms (`SUBJ`, `PRED`, `OBJ`, `WHEN`, `APP_CTX`, ...) are
generated by hashing the prefix `__ROLE__/<name>` rather than the bare
role name. The 16-role enum is fixed in
[core/codebook.py](../core/codebook.py); native consumers SHOULD
hardcode this list rather than enumerate dynamically.

---

## 4. Backwards compatibility

`SAVE_VERSION` in [store/persistence.py](../store/persistence.py) is
the engine's wire-format version. Consumers MUST refuse saves with a
version higher than their build supports; older versions are
forward-compatible (missing sections like the v2 segmenter block or
v3 N-gram block trigger a clean cold-start of that subsystem rather
than a load failure).

| Version | Adds                                                         |
|---------|--------------------------------------------------------------|
| 1       | Tally bins + codebook salt + decay parameters                |
| 2       | Segmenter EMA state, VSA predictor `h_summary`               |
| 3       | N-gram predictor sparse top-K transitions; Hybrid predictor  |

When the engine is ported to Rust, the on-disk format is the
synchronization contract. Native code reads and writes the same five
files (`M.bin`, `D.bin`, `E.bin`, `predictor_h.bin`, `meta.json`) and
the same SQLite `episodes.db`. No translation layer is necessary.
