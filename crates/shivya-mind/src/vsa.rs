//! MAP-B VSA algebra over bit-packed hypervectors.
//!
//! Convention (matches the Python prototype): bit `0` represents bipolar
//! `+1`, bit `1` represents bipolar `-1`. Under this mapping
//!
//! * `bind` collapses to bitwise XOR,
//! * `similarity` is `(D - 2 * popcount(a ^ b)) / D`,
//! * `bundle` is bit-majority with deterministic tie-breaking.
//!
//! Storage is a `BitArray<[u64; WORDS], Msb0>` so the algebra rides
//! directly on the CPU's word-level XOR and popcount paths. The trailing
//! `WORDS * 64 - D = 48` padding bits are kept at zero so they do not
//! perturb popcount-based similarity.

use bitvec::array::BitArray;
use bitvec::order::Msb0;

/// Logical hypervector dimension.
pub const D: usize = 10_000;

/// Number of `u64` storage words: `ceil(D / 64) = 157`.
pub const WORDS: usize = 157;

/// Number of unused padding bits in the final storage word.
pub const PADDING_BITS: usize = WORDS * 64 - D;

/// Bit-packed hypervector. The MSB of word 0 is logical bit 0.
pub type Hypervector = BitArray<[u64; WORDS], Msb0>;

/// Build a fresh zeroed hypervector.
#[inline]
pub fn zero() -> Hypervector {
    BitArray::ZERO
}

/// Mask the final storage word so the trailing 48 padding bits are zero.
#[inline]
pub fn mask_padding(h: &mut Hypervector) {
    if PADDING_BITS == 0 {
        return;
    }
    let keep = 64 - PADDING_BITS;
    let mask: u64 = if keep == 0 { 0 } else { !0u64 << PADDING_BITS };
    h.data[WORDS - 1] &= mask;
}

/// Bind: self-inverse pairing operator, XOR on the bit layout.
#[inline]
pub fn bind(a: &Hypervector, b: &Hypervector) -> Hypervector {
    let mut out = zero();
    for i in 0..WORDS {
        out.data[i] = a.data[i] ^ b.data[i];
    }
    out
}

/// Bind in place: `a ^= b`.
#[inline]
pub fn bind_into(a: &mut Hypervector, b: &Hypervector) {
    for i in 0..WORDS {
        a.data[i] ^= b.data[i];
    }
}

/// Normalised Hamming similarity in `[-1, 1]`.
///
/// `(D - 2 * popcount(a ^ b)) / D`. Padding bits are zero in both
/// operands so their XOR contributes nothing to the popcount.
#[inline]
pub fn similarity(a: &Hypervector, b: &Hypervector) -> f32 {
    let mut diff: u32 = 0;
    for i in 0..WORDS {
        diff += (a.data[i] ^ b.data[i]).count_ones();
    }
    (D as f32 - 2.0 * diff as f32) / D as f32
}

/// Cyclic right shift of the first `D` bits by `k`. Negative `k` shifts
/// left. The result is masked so padding bits stay zero.
pub fn permute(v: &Hypervector, k: i64) -> Hypervector {
    let k = (k.rem_euclid(D as i64)) as usize;
    if k == 0 {
        return *v;
    }
    let mut out = zero();
    for i in 0..D {
        let src = (i + D - k) % D;
        let bit = read_bit(v, src);
        write_bit(&mut out, i, bit);
    }
    out
}

/// PCG32 (XSH-RR) PRNG: small, deterministic, no allocations. Used for
/// codebook seeding and for bundle tie-breaks.
#[derive(Clone, Debug)]
pub struct Pcg32 {
    state: u64,
    inc: u64,
}

impl Pcg32 {
    /// Construct a stream from a 64-bit seed and a 64-bit stream id.
    /// The stream id is forced to be odd, as required by PCG.
    pub fn new(seed: u64, stream: u64) -> Self {
        let mut rng = Pcg32 {
            state: 0,
            inc: stream.wrapping_shl(1) | 1,
        };
        let _ = rng.next_u32();
        rng.state = rng.state.wrapping_add(seed);
        let _ = rng.next_u32();
        rng
    }

    /// Convenience constructor seeding from the first 16 bytes of a hash.
    pub fn from_bytes(bytes: &[u8]) -> Self {
        let mut seed_buf = [0u8; 8];
        let mut stream_buf = [0u8; 8];
        seed_buf.copy_from_slice(&bytes[0..8]);
        stream_buf.copy_from_slice(&bytes[8..16]);
        Pcg32::new(
            u64::from_le_bytes(seed_buf),
            u64::from_le_bytes(stream_buf),
        )
    }

    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        let old = self.state;
        self.state = old
            .wrapping_mul(6364136223846793005)
            .wrapping_add(self.inc);
        let xorshifted = (((old >> 18) ^ old) >> 27) as u32;
        let rot = (old >> 59) as u32;
        xorshifted.rotate_right(rot)
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        ((self.next_u32() as u64) << 32) | (self.next_u32() as u64)
    }

    /// Uniform integer in `[0, n)`. Uses rejection sampling to avoid
    /// modulo bias.
    pub fn gen_range(&mut self, n: u32) -> u32 {
        debug_assert!(n > 0);
        let threshold = (u32::MAX - n + 1) % n;
        loop {
            let r = self.next_u32();
            if r >= threshold {
                return r % n;
            }
        }
    }

    /// Fisher-Yates shuffle on a `&mut [usize]`.
    pub fn shuffle(&mut self, xs: &mut [usize]) {
        for i in (1..xs.len()).rev() {
            let j = self.gen_range((i + 1) as u32) as usize;
            xs.swap(i, j);
        }
    }
}

/// Fill a hypervector with random Rademacher bits from a PCG stream.
pub fn random_hypervector(rng: &mut Pcg32) -> Hypervector {
    let mut out = zero();
    for w in 0..WORDS {
        out.data[w] = rng.next_u64();
    }
    mask_padding(&mut out);
    out
}

/// Bit-majority bundle with deterministic tie-breaking.
///
/// For odd `K`, no ties are possible. For even `K`, exact ties (count
/// equal to `K / 2`) are broken by a single bit drawn from the supplied
/// PCG stream. Padding bits are zero in every input and so vote zero,
/// preserving the padding invariant in the output.
pub fn bundle(vecs: &[&Hypervector], rng: &mut Pcg32) -> Hypervector {
    let k = vecs.len();
    let mut out = zero();
    if k == 0 {
        return out;
    }
    let half = k / 2;
    let even = k % 2 == 0;
    // Column-wise majority over D logical bits.
    // 40 KB on the stack is acceptable for our 8 MB main-thread budget;
    // a u16 counter is enough up to K = 65_535.
    let mut counts = [0u16; D];
    for v in vecs {
        for w in 0..WORDS {
            let word = v.data[w];
            let base = w * 64;
            let end = ((w + 1) * 64).min(D);
            for bit_idx in 0..(end - base) {
                let mask = 1u64 << (63 - bit_idx);
                if word & mask != 0 {
                    counts[base + bit_idx] += 1;
                }
            }
        }
    }
    for i in 0..D {
        let c = counts[i] as usize;
        let bit_one = if c > half {
            true
        } else if c < half {
            false
        } else if even {
            (rng.next_u32() & 1) == 1
        } else {
            false
        };
        if bit_one {
            let w = i / 64;
            let bit_idx = i % 64;
            out.data[w] |= 1u64 << (63 - bit_idx);
        }
    }
    out
}

/// Sign-bundle a real-valued tally into a hypervector. Positive tallies
/// vote `+1` (bit 0), negative vote `-1` (bit 1), exact zeros are broken
/// by a coin flip from the supplied PCG stream.
pub fn sign_with_tiebreak(tally: &[f32; D], rng: &mut Pcg32) -> Hypervector {
    let mut out = zero();
    for i in 0..D {
        let v = tally[i];
        let bit_one = if v < 0.0 {
            true
        } else if v > 0.0 {
            false
        } else {
            (rng.next_u32() & 1) == 1
        };
        if bit_one {
            let w = i / 64;
            let bit_idx = i % 64;
            out.data[w] |= 1u64 << (63 - bit_idx);
        }
    }
    out
}

/// Add `+1` per zero bit and `-1` per one bit of `v` into a float tally.
#[inline]
pub fn accumulate_into(tally: &mut [f32; D], v: &Hypervector, weight: f32) {
    for w in 0..WORDS {
        let word = v.data[w];
        let base = w * 64;
        let end = ((w + 1) * 64).min(D);
        for bit_idx in 0..(end - base) {
            let mask = 1u64 << (63 - bit_idx);
            let sign = if word & mask != 0 { -1.0 } else { 1.0 };
            tally[base + bit_idx] += weight * sign;
        }
    }
}

#[inline]
fn read_bit(v: &Hypervector, i: usize) -> bool {
    let w = i / 64;
    let bit_idx = i % 64;
    let mask = 1u64 << (63 - bit_idx);
    v.data[w] & mask != 0
}

#[inline]
fn write_bit(v: &mut Hypervector, i: usize, bit_one: bool) {
    let w = i / 64;
    let bit_idx = i % 64;
    let mask = 1u64 << (63 - bit_idx);
    if bit_one {
        v.data[w] |= mask;
    } else {
        v.data[w] &= !mask;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bind_is_self_inverse() {
        let mut rng = Pcg32::new(1, 1);
        let a = random_hypervector(&mut rng);
        let b = random_hypervector(&mut rng);
        let bound = bind(&a, &b);
        let unbound = bind(&bound, &b);
        assert!((similarity(&unbound, &a) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn similarity_self_is_one() {
        let mut rng = Pcg32::new(2, 3);
        let a = random_hypervector(&mut rng);
        assert!((similarity(&a, &a) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn random_pair_similarity_is_near_zero() {
        let mut rng = Pcg32::new(7, 11);
        let a = random_hypervector(&mut rng);
        let b = random_hypervector(&mut rng);
        let s = similarity(&a, &b);
        // 5 sigma for D = 10_000 is 5 / sqrt(D) = 0.05.
        assert!(s.abs() < 0.05, "expected near-zero similarity, got {s}");
    }

    #[test]
    fn permute_is_invertible() {
        let mut rng = Pcg32::new(13, 17);
        let a = random_hypervector(&mut rng);
        let p = permute(&a, 17);
        let back = permute(&p, -17);
        assert!((similarity(&a, &back) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn padding_bits_are_zero_after_random_fill() {
        let mut rng = Pcg32::new(101, 103);
        let v = random_hypervector(&mut rng);
        let last = v.data[WORDS - 1];
        let padding_mask = (1u64 << PADDING_BITS) - 1;
        assert_eq!(last & padding_mask, 0);
    }

    #[test]
    fn bundle_recovers_constituent() {
        let mut rng = Pcg32::new(19, 23);
        let vs: Vec<Hypervector> = (0..7).map(|_| random_hypervector(&mut rng)).collect();
        let refs: Vec<&Hypervector> = vs.iter().collect();
        let bun = bundle(&refs, &mut rng);
        // Every constituent should be markedly more similar to the
        // bundle than a fresh random vector.
        let distractor = random_hypervector(&mut rng);
        let s_distract = similarity(&bun, &distractor);
        for v in &vs {
            let s = similarity(&bun, v);
            assert!(
                s > s_distract + 0.1,
                "constituent similarity {s} too close to distractor {s_distract}"
            );
        }
    }
}
