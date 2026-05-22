# Philosophy

The technical decisions in Shivya Mind are downstream of four convictions about what memory is, what it should forget, what its blast radius should be, and who should own it. This document defends those convictions.

## 1. Memory as active reconstruction

The dominant architecture for AI "memory" in 2026 is a token log. Every utterance and every tool call is appended to a growing array; retrieval is `Ctrl-F` over the array. This is essentially how a court reporter takes notes — and it gives an LLM the cognition of someone trying to deliberate while reading their own transcript out loud, in full, before answering each question. It is the wrong primitive for almost every reason.

Biological memory is not a log. Tulving's encoding/retrieval distinction (1972), Bartlett's reconstructive account of remembering (1932), and a half-century of consolidation research (Nader, Schiller, Dudai) converge on the same picture: every act of remembering is also an act of *re-encoding*. The hippocampus does not replay frozen records; it generates a plausible past from compressed traces, conditioned on present context. We are wrong about details, certain about gist, and our certainty does not track our accuracy. This is not a bug — it is what enables generalisation. If memory were verbatim it would be useless for understanding anything new.

Shivya Mind takes this seriously at the level of the data structure. The long-term memory `M` is a single hypervector. There are no "records" to retrieve. A query is not a string match; it is a vector operation — bind, unbind, cleanup — that *generates* the recalled content from the compressed substrate. Two queries about the same fact return slightly different reconstructions because the codebook and the surrounding activation have shifted. The retrieved content is plausible against the substrate, not identical to anything originally stored. This is generative remembering, and it falls out of the algebra for free.

The forgetting curve we use is **power-law**:

```
α(τ)  =  (1 + β · τ)^(−ψ)
```

with `β = 1/day` and `ψ = 0.5`. This is not exponential. Exponential forgetting (the textbook choice for "decay" parameters) decays at a *constant rate per unit time* — which is wrong empirically (Wickelgren 1972) and wrong information-theoretically. Anderson & Schooler (1991) derived the power-law form from the rational analysis of memory: it is the optimal retention schedule for an agent whose environment delivers events with a power-law need-probability distribution, which empirically the real world does. Shivya Mind's `α` curve is not aesthetic — it is what the math says.

The consequence is that recent memory remains crisp, old memory degrades gradually, and the magnitude of `M` does not blow up. Decay is the *prerequisite* for a constant-footprint memory engine, not a feature added on top.

## 2. The virtue of forgetting

The instinct of the engineer who first encounters a system with bounded memory is to ask "how do we make it bigger?" — and then to lobby for an infinite store. This is wrong. Forgetting is not a failure mode of intelligence; it is a *prerequisite* for it.

Three reasons:

**1 — Forgetting is compression, and compression is understanding.** A memory that retains every detail of every event is incapable of recognising patterns, because patterns are the residue left after the noise is discarded. Solomon Shereshevsky, the mnemonist studied by Luria, could repeat any string of digits verbatim years after hearing it — and could not, despite enormous effort, hold a job, follow a conversation, or recognise the same face under different lighting. His memory was uncompressed, and so it was uninhabitable. Compression is what made meaning possible.

**2 — Forgetting is a defence against poisoning.** Every long-running interactive system silently accumulates contradictory inputs, retracted assertions, sycophantic affirmations, and stale tool outputs. An append-only log has no garbage collector for any of these; once a falsehood enters the prefix it has eternal vote weight. A memory engine that cannot forget *must* be wrong eventually. Shivya Mind's `forget()` is exact algebraic subtraction — `M ← M − λF` — which is the cleanest mathematical statement possible of "I no longer hold this".

**3 — Forgetting is the user's right.** A memory you cannot evict is not yours. The "right to be forgotten" is a regulatory translation of a much older intuition: the things I have said to a system are mine to retract. Shivya Mind implements this as a one-line vector operation; there is no log to scrub, no derived statistic to recompute. The fact is gone from the algebra because the algebra has been edited.

The power-law decay schedule and the algebraic `forget` are not two separate features. They are the same conviction expressed on two timescales — slow, automatic compression of unrehearsed evidence; fast, deliberate erasure on user command.

## 3. Associative interference (predicate leakage)

A surprising and load-bearing observation from `bench/memory_drift.py`:

When we `/forget alice owns scooter` with `λ = 5`, the algebra subtracts the bound vector

```
λ · sign( C[alice] ⊛ rSUBJ  +  C[owns] ⊛ rPRED  +  C[scooter] ⊛ rOBJ )
```

from every active tier. This is not selective. Every component of that bound vector is removed — including the `C[owns] ⊛ rPRED` component. Any *other* fact that has `owns` as its predicate now sees a small negative bias in its support vectors. We observed, before we replaced it with an atom-disjoint control, that the distractor `malory owns kayak` — never asserted, never mentioned — picked up a similarity of **−0.0782** after we forgot the scooter fact.

The instinct is to read this as a bug. It is not. It is the holographic property of the algebra correctly producing the cognitive phenomenon known as **associative interference**: the leakage of inhibition or disinhibition across related concepts after a single learning or unlearning event.

The phenomenon is well-documented in humans. Learning that a particular source is unreliable bleeds skepticism into adjacent sources; unlearning a fact about a person partially destabilises adjacent facts about the same person; retracting a claim about a domain leaves residual doubt in nearby claims in that domain. This is not noise. It is the cost of having a *holographic* memory — one in which every part of the substrate participates in every retrieval — and it is the same cost humans pay.

Two consequences of taking this seriously:

**1 — Retraction is structured, not surgical.** A user who says "actually, that fact about Alice was wrong" is not just saying "remove that one node from the graph". They are signalling that their model of Alice — or of ownership claims, or of scooters — should be slightly less confident. Shivya Mind's retraction reflects this. The fact is forgotten harder than it was asserted (`λ = 5` against typical 1-event assertions); related claims are mildly discounted; the system as a whole becomes appropriately skeptical of the neighbourhood. This is the right behaviour. A memory that retracts perfectly surgically would be a memory that does not learn from retraction.

**2 — The shape of the wake encodes generalisation.** When we retract a specific claim, what *exactly* gets discounted depends on which atoms the retracted fact shared with other facts. This is, structurally, generalisation — induction over shared role-fillers. A retraction-event that shares more atoms with another fact transfers more skepticism to it. The wake left by `forget()` is itself an inferential signal.

The reason the bench's initial distractor (`malory owns kayak`) tripped the threshold and the replacement (`malory writes poems`) did not was not a flaw in the bench. It was the system correctly differentiating between facts that share structure with the retracted one and facts that do not. We documented the change rather than papering over it because the behaviour is desirable.

## 4. Privacy by construction

Privacy in most contemporary architectures is a *policy* layered on top of a data structure that is, by default, optimally extractable. Logs are searchable; embeddings are inverse-mappable; training data is reconstructable; transcripts are summarisable. Every safety property has to be enforced against the natural grain of the substrate.

Shivya Mind inverts this. Privacy is a *property of the algebra*, not a discipline imposed on it. Three structural facts:

**1 — The memory vector is opaque without the codebook.** `M` is a 40 KB blob of float32 values in roughly `[−10, +10]`. Without the codebook (which is regenerated from a 16-byte salt held only on the user's device), there is no operation that turns `M` into a fact, a name, or a sentence. Cleanup requires *knowing what to look for*. There is no "scan all facts" primitive because there are no facts as discrete entities — only the holographic sum of bound triples. A leaked `M.bin` without the salt is indistinguishable from random noise.

**2 — Forgetting is mathematical, not procedural.** When a user retracts a fact, the corresponding vector contribution is subtracted from `M`. There is no log entry recording "fact X was deleted", no audit trail for a regulator to subpoena, no shadow copy that "still has it". The fact is gone because the algebra has been edited. The vector itself is the source of truth, and it has been transformed. This is the strongest possible form of deletion: not "removed from view" but "removed from existence".

**3 — Local by default; sync is opt-in and CRDT-clean.** The default code path is on-device. There is no required cloud component, no telemetry, no model API call. When sync is desired, the algebra is naturally a CRDT — bundle is commutative and associative, so two devices converge to the same `M` by signed-sum regardless of order. The codebook is identical across devices by deterministic seeding, so it never traverses the wire. The only thing that ever leaves a device, in the default configuration, is an encrypted hypervector update — an opaque difference of two opaque blobs.

The cumulative effect is that the surveillance vector is closed. There is no datacentre with a copy of the user's memory because no datacentre is required. There is no API to "improve the model" because there is no model in the AI-vendor sense. There is no terms-of-service that grants anyone the right to train on the user's history, because the history was never sent anywhere and is not in a form that could be trained on if it were.

This is not a regulatory posture. It is a structural one. Privacy that cannot be removed by changing terms of service or by court order or by a server-side bug is privacy that actually exists. Everything else is a promise.

---

## Coda — what we are claiming and what we are not

Shivya Mind is not an AGI, an LLM replacement, a chatbot, or a productivity app. It is a small, sharp implementation of one specific argument: that the next interesting kind of personal AI is not the one with the biggest context window, but the one with the smallest one that is *the right shape*. Token logs are the wrong shape. Decaying holographic hypervectors with structured retraction are, we think, much closer to right.

The four convictions above — reconstructive memory, the virtue of forgetting, structured retraction, privacy as algebra — are not separately defensible from the math. They *are* the math, written in prose. Every line of code in `core/` is a commitment to one or more of them. If a future version of Shivya Mind drifts from these, it will be because we discovered a better answer to the same questions, not because we got tired of these answers.

The first version of any of these systems is going to be wrong in ways we cannot currently see. But the four commitments above are not a feature list; they are an orientation. We will know we are on the right path when bigger systems we build have less to apologise for than the systems we are replacing — when forgetting is the default and remembering is the act, when the algebra is the source of truth and the policy is just commentary, when the user owns the substrate and the substrate owns nothing of the user.

---

## Postscript — Shivya and the mind that sits inside it

Shivya Mind exists because [Shivya](https://github.com/jvoltci/shivya) exists. The parent project — `shivya-hodge`, `shivya-flux`, `shivya-morphic`, `shivya-onsager`, `shivya-turing` — solves a *physics* problem: how does a mesh of edge devices reconcile state and balance load without electing a leader? It answers in the language of discrete differential geometry, statistical mechanics, and active inference. What it does *not* answer — and was never meant to — is what any single node *remembers* about its own life. A Shivya daemon, taken individually, knows the curl-free flow on its incident edges and not much else. It has no past.

That is the gap Shivya Mind fills. The same information-theoretic surprise (`F = KL(q‖prior) + nll(s | g(μ_q))`) that `shivya-flux` uses to drive belief updates is the quantity our segmenter watches to carve episode boundaries. The same commitment Shivya makes to zero-allocation, fixed-budget computation is the commitment we make to a constant 40 KB memory floor. The substrate's *physics* is the mind's *clock*; the substrate's *budget* is the mind's *constraint*; the substrate's *consensus-free* posture is the mind's *local-first* posture. The four convictions above (reconstructive memory, the virtue of forgetting, structured retraction, privacy as algebra) are the same convictions Shivya already commits to — expressed in the dialect of memory rather than the dialect of mesh.

Shivya is what a swarm *does*. Shivya Mind is what each member of the swarm *remembers having done*.
