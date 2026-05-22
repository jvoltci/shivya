# Architecture

Deep technical reference for the Shivya Mind engine — the memory and episodic-cognition module of the [Shivya](https://github.com/jvoltci/shivya) substrate. Every equation in this document corresponds to a function in `core/`; every diagram corresponds to a runnable code path. For where this crate sits in the larger Shivya stack, see [Position in the Shivya stack](../README.md#position-in-the-shivya-stack) in the project README.

## 1. The MAP-B algebra engine

Shivya Mind's hypervectors live in `{−1, +1}^D` with `D = 10,000`. The algebra is **MAP-B** (Multiply-Add-Permute, Binary) — chosen over Plate's HRR because all three primitive operations collapse to bit-parallel hardware instructions when the vectors are bit-packed.

### 1.1 Two equivalent surfaces

The same algebra has two storage representations that are isomorphic under a fixed bit convention. We hold both and verify cross-surface equivalence as part of the benchmark suite.

| Surface | dtype | Storage | Convention |
|---|---|---|---|
| **Bipolar** | `int8`, shape `(D,)` | D bytes = 10,000 B | values in `{−1, +1}` |
| **Bit-packed** | `uint8`, shape `(⌈D/8⌉,)` | 1,250 B | bit 0 ↔ bipolar +1, bit 1 ↔ bipolar −1 |

Conversion is `core.packing.pack` / `core.packing.unpack`. Padding bits (when `D mod 8 ≠ 0`) are always zeroed so they contribute nothing to popcount-based similarity.

### 1.2 The four primitives

#### Bind — pairs concepts; self-inverse

```
bipolar:   a ⊛ b   =   a * b                  (element-wise int8 multiply)
packed :   a ⊛ b   =   a XOR b                (bitwise xor)
```

`a ⊛ a = 1` in both surfaces (`(±1)² = +1`, `a XOR a = 0`). The self-inverse property is what makes unbinding trivial: to recover `S` from `S ⊛ rSUBJ`, just bind again by `rSUBJ`.

#### Bundle — superposition with random tie-break

```
bipolar:   bundle(v_1, …, v_K)  =  sign( Σ v_i )
                                   zeros resolved by fair coin
packed :   majority vote per bit position, ties (count == K/2) resolved by fair coin
```

The fair-coin tie-break is the only randomized part of the algebra; without it, even-cardinality bundles introduce a systematic +1 bias from the conventional `sign(0) = +1`.

#### Permute — cyclic shift; injects sequence order

```
bipolar:   Π^k(v)   =   np.roll(v, k)
packed :   bitwise cyclic rotation by k bits
```

Permutation is a fixed bijection — any permutation works algebraically; cyclic shift is the cheapest on hardware. The current PoC permutes through `np.roll` on the bipolar surface and through unpack/roll/pack on the packed surface; both produce identical results (verified in `bench/capacity.py`).

#### Similarity — normalized dot product

```
bipolar:   sim(a, b)   =   (a · b) / D            range: [−1, +1]
packed :   sim(a, b)   =   (D − 2 · popcount(a XOR b)) / D
```

The packed formula derives from `bipolar_dot = D − 2 · |disagreeing positions|` and `|disagreeing| = popcount(a XOR b)`. We use the LUT-based popcount path when NumPy < 2.0 (no `np.bitwise_count`).

### 1.3 Capacity — what you can store in one vector

The theoretical bound for MAP-B with cleanup memory (Kanerva 1988; Plate 2003) is

```
K_max  ≈  D / (2 · log₂ N)        at 95% per-item recall
```

For `D = 10,000` and a codebook of `N = 1,024` atoms, the bound is **K ≈ 500**. The empirically measured curve from `bench/capacity.py`:

```
K       recall
10      1.0000
25      1.0000
50      1.0000
100     1.0000
200     0.9945
300     0.9828
500     0.9626        ← K@95% bound; empirically still ≥ 95%
800     0.9587
```

The empirical curve is gentler than the bound (the Kanerva formula is a worst-case heuristic; real i.i.d. Rademacher vectors stay above 95% somewhat past the theoretical knee).

### 1.4 The deterministic codebook

Every label maps to a fixed bipolar vector via

```
seed(label, salt)  =  first 64 bits of BLAKE3(salt || label)     # or BLAKE2b fallback
C[label]           =  sign( RNG(seed).normal(D) ) cast to int8
```

Two consequences:

1. Two devices sharing the 16-byte salt reconstruct **byte-identical** vectors for every label — sync requires only the salt, never the codebook contents.
2. The codebook is **lazy**: vectors are generated on first access and memoized. Initializing a Codebook is essentially free.

Three classes of labels: **role atoms** (`SUBJ`, `PRED`, `OBJ`, `APP_CTX`, `WHEN`, …; 16 reserved), **anchor atoms** (a fixed multilingual lexicon), and **dynamic atoms** (any string the user has ever asserted).

### 1.5 Encoding an event

An event is a 4-tuple `(subject, predicate, object, ctx)` plus an optional `t`. The encoding is **Plate role binding** — bind each filler to its role atom, then bundle:

```
F_event = sign(
    C[subject]   ⊛ rSUBJ
  + C[predicate] ⊛ rPRED
  + C[object]    ⊛ rOBJ
  + C[ctx]       ⊛ rAPP_CTX        # if present
  + C_time(t)    ⊛ rWHEN           # if present
)
```

Recovering any filler is one bind + one cleanup:

```
F_event ⊛ rSUBJ   =   C[subject]   +   noise from other terms
                  →   argmax over codebook  =  subject
```

Cleanup succeeds with high probability while the event holds ≤ ~30 role-filler pairs, far beyond our 5-role schema.

## 2. The tri-tier memory hierarchy

A flat `M = Σ F_event` saturates after a few hundred events (§1.3). Shivya Mind uses a three-tier accumulator with **power-law decay** applied between consolidations.

### 2.1 The tiers

```
Tier 0  (events)     E_open        ─ open episode buffer; new events bundle here
Tier 1  (day)        D_open        ─ open day buffer; receives sealed episodes
Tier 2  (long-term)  M             ─ decayed long-term memory
```

All three are held as **float32 tallies** (not bipolar) so that magnitude information survives across consolidations. The bipolar form is recovered by `sign()` only at read time — `working_memory() = sign(M + D + E)`.

### 2.2 The fold-up

```
seal_episode()       :   D ← D + sign(E),         E ← 0
consolidate_day()    :   M ← α_n · M + sign(D),   D ← 0
```

where `α_n` is the power-law decay coefficient computed from real-time elapsed seconds since the last consolidation:

```
α(τ)  =  (1 + β · τ)^(−ψ)            β = 1/day = 1/86400 s⁻¹,   ψ = 0.5
```

The analytic check (asserted in `bench/memory_drift.py`):

```
τ            α
0            1.0000
1 hour       0.9798
1 day        0.7071     ← exactly 1/√2 by construction
1 week       0.3536
30 days      0.1796
1 year       0.0523
```

This is Wickelgren's law (Anderson & Schooler 1991 derive it as the rational analysis of retention) — **slower than exponential**, faster than uniform, and the only retention curve that matches empirically observed forgetting on timescales from seconds to years.

### 2.3 The full update equation

The Blueprint §1.6 equation is

```
M_{n+1}  =  α_n · M_n  +  Π^{seq(j_n)}(D_{j_n})
```

The current implementation specializes `Π^seq` to identity in the fold-up (order between events is already encoded by each `F_event`'s `WHEN` role binding to a time anchor `C_time(t) = Π^t(T_base)`). Re-enabling per-episode and per-day rotations is straightforward when temporal-navigation queries ("what was I doing the episode before this one?") arrive.

### 2.4 Querying

`working_memory()` is the read surface — bipolar projection of the full accumulator:

```
W   =   sign(M + D + E)
```

Three patterns over `W`:

- **Fact strength** — `sim(F_query, W)` returns the cosine similarity of a candidate fact to the combined memory.
- **Role recovery** — `W ⊛ rROLE` produces a noisy bundle of all fillers seen in that role position; cleanup against a candidate label set returns the dominant ones.
- **Concept presence** — `sim(W ⊛ rROLE, C[label])` measures how strongly a specific label appears in a specific role position.

### 2.5 Retraction

`Memory.forget(event, weight=λ)` is exact algebraic subtraction:

```
F   =   encode_event(event)
E ← E − λ·F
D ← D − λ·F
M ← M − λ·F
```

Subtracting from all three tiers simultaneously means the combined `working_memory()` sum loses `3λF`. The "extra factor of 3" makes forgets aggressive: with `λ = 5`, a fact that has been asserted five times is not just removed but driven to a strongly negative association (around −1.0). This is structured retraction — see `docs/philosophy.md §3` for the design rationale and its associative side effects.

## 3. The information-theoretic surprise loop

Episode boundaries are not clock-driven. They are placed where a new event is **maximally unpredictable** given the running history — Friston's free-energy principle applied to the conversational stream.

### 3.1 The surprise quantity

For each incoming event `x_t`:

```
s_t   =   −log p(x_t | h_{t-1})
```

`h_{t-1}` is whatever internal state the predictor maintains. Three concrete predictors are implemented (`core/surprise.py`):

| Predictor | `h` is | Cost |
|---|---|---|
| `NgramPredictor` | Laplace-smoothed bigram counts on a chosen atom (default: predicate) | counts table, O(V) |
| `VSAExpectednessPredictor` | decaying running bundle `h_summary` of recent `F_event` vectors | 1 hypervector, O(D) |
| `HybridPredictor` | weighted log-mix of both (Blueprint §2.1) | sum of the above |

A clean hook (`embedder` callable) replaces the VSA path with any external `event → ndarray` function — drop in a transformer embedder later without touching the segmenter.

### 3.2 Online surprise statistics

`SurpriseEMA` tracks `μ_s` and `σ_s` with a two-phase update:

```
Phase 1 (n ≤ warmup_n):     μ = mean(buf),  var = pop_var(buf)
Phase 2 (n  > warmup_n):    delta = s − μ
                            var  ← (1 − α)(var + α · delta²)
                            μ    ← μ + α · delta
```

The warm-up phase fixes a real and subtle problem: a pure-EMA `σ` is biased *high* during the transient where `μ` is climbing toward steady state — spikes hide inside their own freshly inflated σ. Direct-sample statistics over the first ~8 events give a clean baseline before the exponential kernel takes over. The very first event's surprise (cold-start artifact) is also skipped from the EMA.

### 3.3 The triple-constraint seal rule

Three rules compete in parallel; any one of them can fire a seal:

```
spike    :    s_t        >  μ_s + k · σ_s            (k default: 2.0)
drift    :    Σ s_t      >  S_cap                    (S_cap default: 50 nats)
time_cap :    now − t₀   >  T_max                    (T_max default: 3600 s)
```

Each rule covers a failure mode of the other two:

| Rule | Catches | Why the others miss it |
|---|---|---|
| **spike** | Sudden topic shift; abrupt context change | Drift hasn't crossed `S_cap`; wall-clock is short |
| **drift** | Slow, monotonic concept migration | No single event exceeds `μ + kσ` |
| **time_cap** | Long stretches of unsurprising activity | Surprise stays below `S_cap`; spike never fires |

The boundary event is the **first event of the new episode**, per the Zacks & Tversky event-segmentation convention. The seal closes the previous episode and resets the segmenter's per-episode counters; the current event's surprise is the new episode's first contribution.

### 3.4 Bead serialization

On seal, the segmenter:

1. Calls `Memory.seal_episode()`, which returns the bipolar `E_k` sealed into `D`.
2. Extracts a **thumbnail** — top filler per role — by computing `E_k ⊛ rROLE` and cleaning up against the seen-labels codebook for `ROLE ∈ {SUBJ, PRED, OBJ}`.
3. Constructs an `EpisodeBead(id, t_start, t_end, n_events, reason, surprise_peak, thumbnail, title=None)` and emits it to the caller.
4. Resets `cumulative_surprise`, `peak_in_episode`, `events_since_seal`, and `episode_start_t`.

The bead is the only thing that crosses the engine → UI boundary. Raw events, atoms, and hypervectors stay inside `core/` and `store/`.

## 4. Process topology

The complete live pipeline:

```
┌──────────────────────────────────────────────────────────────────────────┐
│  adb logcat  (subprocess via subprocess.Popen with explicit argv,        │
│  filtered: ActivityTaskManager:I  ActivityManager:I  *:S)                │
│  starts at tail (-T 1); never replays history                            │
└─────────────────────────────┬────────────────────────────────────────────┘
                              │  stdout pipe, line-buffered
                              │  + select.select() heartbeat every 0.25s
                              │    (lets the caller honour --max-seconds /
                              │     Ctrl-C even during long idle windows)
                              ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  StreamParser  (ingest/stream_parser.py)                                 │
│  five ordered rules; first match wins:                                   │
│      1.  AndroidActivityRule    "Displayed com.X.Y/...",                 │
│                                 "am_on_resume_called: [...]",            │
│                                 "cmp=com.X.Y/..."                        │
│      2.  NowPlayingRule         "Opened X - Now Playing: Y"              │
│      3.  NotificationRule       "Notification from X: A says B"          │
│      4.  AuthorshipRule         "I [just] [finished] writing X on Y"     │
│      5.  AppOpenRule            "Opened X"  (general fallback)           │
│                                                                          │
│  helpers:  normalize(label)     lowercase + collapse to underscore       │
│            summarize(phrase)    drop stop-words, take 2 content tokens   │
│            package_to_app(pkg)  com.spotify.music → spotify              │
│            app_to_ctx(app)      app → category via APP_CATEGORY map      │
└─────────────────────────────┬────────────────────────────────────────────┘
                              │  Event(subject, predicate, object, ctx)
                              ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  Segmenter  (core/surprise.py)                                           │
│  ┌───────────────────┐   ┌──────────────────────┐   ┌────────────────┐   │
│  │  Predictor        │   │  SurpriseEMA         │   │  Memory        │   │
│  │  surprise(x)  → s │ → │  update(s)           │ → │  encode_event  │   │
│  │  observe(x)       │   │  μ, σ  (warmup→EMA)  │   │  E ← E + F     │   │
│  └───────────────────┘   └──────────┬───────────┘   └──────────┬─────┘   │
│                                     │                          │         │
│  Rules:    spike      s   >  μ + kσ ┤                          │         │
│            drift      Σs  >  S_cap  ├──→  seal? ──────────────►│         │
│            time_cap   Δt  >  T_max  ┘     thumbnail = E_k ⊛ R  │         │
│                                                                ▼         │
└──────────────────────────────────────────────────────┬───────────────────┘
                                                       │ EpisodeBead
                                                       ▼
                       ┌─────────────────────────────────────────┐
                       │  EpisodeStore   (store/episodes.py)     │
                       │  SQLite beads(id PK, t_start, t_end,    │
                       │              n_events, reason, peak,    │
                       │              thumbnail_{0,1,2}, title)  │
                       │  + indexes on thumbnail_{0,1,2}         │
                       │                                         │
                       │  persistence.save / load                │
                       │  (store/persistence.py)                 │
                       │     M.bin              40 KB float32    │
                       │     D.bin              40 KB float32    │
                       │     E.bin              40 KB float32    │
                       │     predictor_h.bin    40 KB float32    │
                       │     meta.json          codebook salt,   │
                       │                        decay constants, │
                       │                        counters,        │
                       │                        seen-label list, │
                       │                        segmenter EMA,   │
                       │                        predictor state  │
                       └─────────────────────────────────────────┘
```

Three properties of this topology that matter:

1. **No shell.** Every subprocess invocation uses an explicit argv list. No `shell=True`, no string interpolation into a command line — there is no injection surface even when user-controlled strings end up in event objects.
2. **No network.** Every default code path is local. ADB itself is a USB or local-TCP protocol to the user's own device. The codebook is regenerated from a salt; no model weights are downloaded.
3. **No required cloud SDK.** Only NumPy is mandatory. `blake3`, `adb`, and the optional embedder callable are all swappable; the engine falls back to stdlib BLAKE2b, the `ReplayStream`, and the pure-VSA expectedness predictor respectively.

## 5. Where the math actually lives in the code

| Mathematical object | File | Function |
|---|---|---|
| `a ⊛ b` (bind) | `core/vsa.py` | `bind` / `bind_packed` |
| `Σ + sign` (bundle) | `core/vsa.py` | `bundle` / `bundle_packed` |
| `Π^k` (permute) | `core/vsa.py` | `permute` / `permute_packed` |
| `(a·b)/D` (similarity) | `core/vsa.py` | `similarity` / `similarity_packed` |
| cleanup against codebook | `core/vsa.py` | `cleanup` |
| `C[label] = sign(RNG(seed).normal(D))` | `core/codebook.py` | `_vector_from_seed`, `Codebook.__getitem__` |
| `C_time(t) = Π^t(T_base)` | `core/codebook.py` | `Codebook.time_anchor` |
| `F_event = sign(Σ filler ⊛ role)` | `core/memory.py` | `Memory.encode_event` |
| `α(τ) = (1 + βτ)^(−ψ)` | `core/memory.py` | `Memory.alpha_effective` |
| `M ← αM + sign(D)` (tier 2 fold) | `core/memory.py` | `Memory.consolidate_day` |
| `M ← M − λF` (forget) | `core/memory.py` | `Memory.forget` |
| `s = −log p(x | h)` | `core/surprise.py` | `*Predictor.surprise` |
| μ/σ warm-up + EMA | `core/surprise.py` | `SurpriseEMA.update` |
| spike / drift / time-cap | `core/surprise.py` | `Segmenter.observe` |
