//! Cryptographic symbol table: label -> Hypervector via blake3-seeded PCG.

use crate::vsa::{permute, random_hypervector, Hypervector, Pcg32};
use std::cell::RefCell;
use std::collections::BTreeMap;

/// Default codebook salt. Two devices using the same salt and label
/// receive byte-identical hypervectors without any synchronisation.
pub const DEFAULT_SALT: &[u8] = b"shivya-mind/v1";

/// The sixteen reserved role atoms used by the memory engine.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum Role {
    Subj,
    Pred,
    Obj,
    Agent,
    When,
    Where,
    AppCtx,
    Device,
    Loc,
    Mood,
    Src,
    Instrument,
    Target,
    Quantity,
    Polarity,
    Session,
}

impl Role {
    pub fn as_str(self) -> &'static str {
        match self {
            Role::Subj => "SUBJ",
            Role::Pred => "PRED",
            Role::Obj => "OBJ",
            Role::Agent => "AGENT",
            Role::When => "WHEN",
            Role::Where => "WHERE",
            Role::AppCtx => "APP_CTX",
            Role::Device => "DEVICE",
            Role::Loc => "LOC",
            Role::Mood => "MOOD",
            Role::Src => "SRC",
            Role::Instrument => "INSTRUMENT",
            Role::Target => "TARGET",
            Role::Quantity => "QUANTITY",
            Role::Polarity => "POLARITY",
            Role::Session => "SESSION",
        }
    }
}

/// Lazy, deterministic mapping from labels to Rademacher hypervectors.
#[derive(Debug)]
pub struct Codebook {
    salt: Vec<u8>,
    cache: RefCell<BTreeMap<String, Hypervector>>,
    time_base: Hypervector,
    /// Lightweight stream id used for bundle tie-breaking. Derived
    /// deterministically from the salt so two devices share the same
    /// tie-break sequence.
    pub(crate) bundle_stream: u64,
}

impl Codebook {
    pub fn new(salt: &[u8]) -> Self {
        let time_base = vector_from_label(salt, "__TIME_BASE__");
        let bundle_stream = stream_from_salt(salt);
        Self {
            salt: salt.to_vec(),
            cache: RefCell::new(BTreeMap::new()),
            time_base,
            bundle_stream,
        }
    }

    pub fn with_default_salt() -> Self {
        Self::new(DEFAULT_SALT)
    }

    pub fn salt(&self) -> &[u8] {
        &self.salt
    }

    /// Build a fresh PCG stream seeded from the salt. Used by the
    /// memory engine for bundle tie-breaks so the same salt + same
    /// stream of writes yields the same output.
    pub fn bundle_rng(&self) -> Pcg32 {
        Pcg32::new(self.bundle_stream, 0xa5a5_a5a5_a5a5_a5a5)
    }

    /// Return the hypervector for `label`, generating it if missing.
    pub fn get(&self, label: &str) -> Hypervector {
        if let Some(v) = self.cache.borrow().get(label) {
            return *v;
        }
        let v = vector_from_label(&self.salt, label);
        self.cache.borrow_mut().insert(label.to_string(), v);
        v
    }

    pub fn role(&self, r: Role) -> Hypervector {
        let key = role_key(r);
        self.get(&key)
    }

    /// Rotate the time base by `t` positions for an integer time index.
    pub fn time_anchor(&self, t: i64) -> Hypervector {
        permute(&self.time_base, t)
    }

    /// Number of distinct labels currently in the cache.
    pub fn cache_len(&self) -> usize {
        self.cache.borrow().len()
    }

    /// Iterate (label, vector) pairs from the cache as a snapshot.
    pub fn snapshot(&self) -> Vec<(String, Hypervector)> {
        self.cache
            .borrow()
            .iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect()
    }
}

fn role_key(r: Role) -> String {
    let mut s = String::from("__ROLE__/");
    s.push_str(r.as_str());
    s
}

fn vector_from_label(salt: &[u8], label: &str) -> Hypervector {
    let mut hasher = blake3::Hasher::new();
    hasher.update(salt);
    hasher.update(label.as_bytes());
    let digest = hasher.finalize();
    let bytes = digest.as_bytes();
    let mut rng = Pcg32::from_bytes(&bytes[0..16]);
    random_hypervector(&mut rng)
}

fn stream_from_salt(salt: &[u8]) -> u64 {
    let digest = blake3::hash(salt);
    let mut buf = [0u8; 8];
    buf.copy_from_slice(&digest.as_bytes()[0..8]);
    u64::from_le_bytes(buf)
}
