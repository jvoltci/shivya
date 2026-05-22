use std::net::SocketAddr;
use std::time::SystemTime;

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NodeId(pub [u8; 20]);

impl NodeId {
    pub fn random() -> Self {
        let mut bytes = [0u8; 20];
        // Generate pseudo-random bytes using system time as seed
        let seed = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let mut r = seed;
        for i in 0..20 {
            r = r.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            bytes[i] = (r >> 32) as u8;
        }
        NodeId(bytes)
    }

    pub fn xor_distance(&self, other: &Self) -> [u8; 20] {
        let mut dist = [0u8; 20];
        for i in 0..20 {
            dist[i] = self.0[i] ^ other.0[i];
        }
        dist
    }

    pub fn leading_zeros(&self) -> usize {
        let mut count = 0;
        for &byte in &self.0 {
            if byte == 0 {
                count += 8;
            } else {
                count += byte.leading_zeros() as usize;
                break;
            }
        }
        count
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PeerInfo {
    pub id: NodeId,
    pub address: SocketAddr,
}

pub const K: usize = 4;
pub const BUCKET_COUNT: usize = 160;

#[derive(Clone, Copy, Debug)]
pub struct KBucket {
    pub peers: [Option<PeerInfo>; K],
    pub count: usize,
}

impl Default for KBucket {
    fn default() -> Self {
        Self::new()
    }
}

impl KBucket {
    pub fn new() -> Self {
        Self {
            peers: [None; K],
            count: 0,
        }
    }

    pub fn insert_or_update(&mut self, peer: PeerInfo) -> InsertResult {
        // 1. Search if peer already exists in this bucket
        for i in 0..self.count {
            if let Some(ref p) = self.peers[i] {
                if p.id == peer.id {
                    // Update address, and move to the end (LRU: most recently active)
                    let updated_peer = peer;
                    // Rotate elements [i..count] left to pull index `i` to the end
                    self.peers[i..self.count].rotate_left(1);
                    self.peers[self.count - 1] = Some(updated_peer);
                    return InsertResult::Updated;
                }
            }
        }

        // 2. If bucket is not full, insert at the end
        if self.count < K {
            self.peers[self.count] = Some(peer);
            self.count += 1;
            return InsertResult::Inserted;
        }

        // 3. Bucket is full. Trigger the LRU Eviction Guard.
        // Return the oldest peer (at index 0) for verification.
        if let Some(oldest) = self.peers[0] {
            InsertResult::BucketFullPendingEviction {
                oldest,
                candidate: peer,
            }
        } else {
            InsertResult::Inserted
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InsertResult {
    Inserted,
    Updated,
    BucketFullPendingEviction {
        oldest: PeerInfo,
        candidate: PeerInfo,
    },
}

pub struct KBucketTable {
    pub self_id: NodeId,
    pub buckets: [KBucket; BUCKET_COUNT],
}

impl KBucketTable {
    pub fn new(self_id: NodeId) -> Self {
        Self {
            self_id,
            buckets: [KBucket::new(); BUCKET_COUNT],
        }
    }

    pub fn insert(&mut self, peer: PeerInfo) -> InsertResult {
        if peer.id == self.self_id {
            return InsertResult::Updated;
        }
        let dist = self.self_id.xor_distance(&peer.id);
        let idx = self.get_bucket_index(&dist);
        self.buckets[idx].insert_or_update(peer)
    }

    pub fn evict_and_insert(&mut self, oldest_id: NodeId, candidate: PeerInfo) -> bool {
        let dist = self.self_id.xor_distance(&candidate.id);
        let idx = self.get_bucket_index(&dist);
        let bucket = &mut self.buckets[idx];

        if bucket.count == K {
            if let Some(ref oldest) = bucket.peers[0] {
                if oldest.id == oldest_id {
                    // Evict oldest! Shift remaining peers to the left
                    bucket.peers[0..K].rotate_left(1);
                    // Put candidate at the back (most-recently active)
                    bucket.peers[K - 1] = Some(candidate);
                    return true;
                }
            }
        }
        false
    }

    pub fn get_bucket_index(&self, dist: &[u8; 20]) -> usize {
        let mut lead_zeros = 0;
        for &byte in dist {
            if byte == 0 {
                lead_zeros += 8;
            } else {
                lead_zeros += byte.leading_zeros() as usize;
                break;
            }
        }
        if lead_zeros >= 160 {
            159
        } else {
            159 - lead_zeros
        }
    }

    pub fn closest_peers(&self, target: &NodeId, count: usize) -> Vec<PeerInfo> {
        let mut all_peers = Vec::new();
        for bucket in &self.buckets {
            for i in 0..bucket.count {
                if let Some(ref peer) = bucket.peers[i] {
                    all_peers.push(*peer);
                }
            }
        }

        all_peers.sort_by_key(|p| p.id.xor_distance(target));
        all_peers.truncate(count);
        all_peers
    }

    pub fn all_peers(&self) -> Vec<PeerInfo> {
        let mut all = Vec::new();
        for bucket in &self.buckets {
            for i in 0..bucket.count {
                if let Some(ref peer) = bucket.peers[i] {
                    all.push(*peer);
                }
            }
        }
        all
    }
}
