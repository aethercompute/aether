use bitvec::array::BitArray;
use bytemuck::Zeroable;
use fnv::FnvHasher;
use serde::{Deserialize, Deserializer, Serialize};
use std::{fmt, hash::Hasher};
use ts_rs::TS;

// Modified from https://github.com/solana-labs/solana/blob/27eff8408b7223bb3c4ab70523f8a8dca3ca6645/bloom/src/bloom.rs

/// Generate a stable hash of `self` for each `hash_index`
/// Best effort can be made for uniqueness of each hash.
pub trait BloomHashIndex {
    fn hash_at_index(&self, hash_index: u64) -> u64;
}

#[derive(Clone, PartialEq, Eq, Copy, Zeroable, TS)]
#[repr(C)]
pub struct Bloom<const U: usize, const K: usize> {
    pub keys: [u64; K],
    pub bits: BitArrayWrapper<U>,
}

#[derive(Clone, PartialEq, Eq, Copy, Default, Serialize, Deserialize, TS)]
#[repr(transparent)]
pub struct BitArrayWrapper<const U: usize>(#[ts(type = "number[]")] pub BitArray<[u64; U]>);

unsafe impl<const U: usize> Zeroable for BitArrayWrapper<U> {}

impl<const U: usize> BitArrayWrapper<U> {
    pub fn new(bits_data: [u64; U]) -> Self {
        Self(BitArray::new(bits_data))
    }
}

impl<const U: usize, const K: usize> Default for Bloom<U, K> {
    fn default() -> Self {
        Self {
            keys: [0u64; K],
            bits: Default::default(),
        }
    }
}

impl<const M: usize, const K: usize> Serialize for Bloom<M, K> {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("Bloom", 3)?;
        state.serialize_field("keys", &self.keys.to_vec())?;
        state.serialize_field("bits", &self.bits)?;
        state.end()
    }
}

impl<'de, const U: usize, const K: usize> Deserialize<'de> for Bloom<U, K> {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct BloomHelper<const U: usize> {
            keys: Vec<u64>,
            bits: BitArrayWrapper<U>,
        }

        let helper = BloomHelper::deserialize(deserializer)?;

        if helper.keys.len() != K {
            return Err(serde::de::Error::custom(format!(
                "Expected {} keys, got {}",
                K,
                helper.keys.len()
            )));
        }

        let mut keys = [0u64; K];
        keys.copy_from_slice(&helper.keys);

        Ok(Bloom {
            keys,
            bits: helper.bits,
        })
    }
}

impl<const U: usize, const K: usize> fmt::Debug for Bloom<U, K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Bloom {{ keys.len: {}, bits: ", self.keys.len())?;
        const MAX_PRINT_BITS: usize = 10;

        if Self::max_bits() <= MAX_PRINT_BITS {
            // Print individual bits for small filters
            for i in 0..Self::max_bits() {
                match self.bits.0.get(i) {
                    Some(x) => write!(f, "{}", *x as u8)?,
                    None => write!(f, "X")?,
                }
            }
        } else {
            // Print byte array for larger filters
            let words = self.bits.0.as_raw_slice();
            for byte in words.iter() {
                write!(f, "{byte:016x}")?; // full u64 output
            }
        }

        write!(f, " }}")
    }
}

impl<const U: usize, const K: usize> Bloom<U, K> {
    pub const fn max_bits() -> usize {
        U * std::mem::size_of::<u64>() * 8
    }

    pub fn new(num_bits: usize, keys_slice: &[u64]) -> Self {
        assert!(num_bits <= Self::max_bits());
        assert!(keys_slice.len() == K);
        let mut keys = [0u64; K];
        let keys_2 = [0u64; U];
        keys.copy_from_slice(keys_slice);
        let bits = BitArrayWrapper::new(keys_2);
        Bloom { keys, bits }
    }

    /// Create filter optimal for num size given the `FALSE_RATE`.
    ///
    /// The keys are randomized for picking data out of a collision resistant hash of size
    /// `keysize` bytes.
    ///
    /// See <https://hur.st/bloomfilter/>.
    #[cfg(feature = "rand")]
    pub fn random(num_items: usize, false_rate: f64) -> Self {
        use rand::Rng;
        let m = Self::num_bits(num_items as f64, false_rate);
        let num_bits = std::cmp::max(1, std::cmp::min(m as usize, Self::max_bits()));
        let keys: Vec<u64> = (0..K).map(|_| rand::rng().random()).collect();
        Self::new(num_bits, &keys)
    }

    #[cfg(feature = "rand")]
    fn num_bits(num_items: f64, false_rate: f64) -> f64 {
        let n = num_items;
        let p = false_rate;
        ((n * p.ln()) / (1f64 / 2f64.powf(2f64.ln())).ln()).ceil()
    }

    #[cfg(feature = "rand")]
    #[allow(dead_code)]
    fn num_keys(num_bits: f64, num_items: f64) -> f64 {
        let n = num_items;
        let m = num_bits;
        // infinity as usize is zero in rust 1.43 but 2^64-1 in rust 1.45; ensure it's zero here
        if n == 0.0 {
            0.0
        } else {
            1f64.max(((m / n) * 2f64.ln()).round())
        }
    }

    fn pos<T: BloomHashIndex>(&self, key: &T, k: u64) -> u64 {
        key.hash_at_index(k)
            .checked_rem(self.bits.0.len() as u64)
            .unwrap_or(0)
    }

    pub fn clear(&mut self) {
        let keys_2 = [0u64; U];
        let bits = BitArrayWrapper::new(keys_2);
        self.bits = bits;
    }

    pub fn add<T: BloomHashIndex>(&mut self, key: &T) {
        for k in &self.keys {
            let pos = self.pos(key, *k) as usize;
            if !*self.bits.0.get(pos).unwrap() {
                self.bits.0.set(pos, true);
            }
        }
    }

    pub fn contains<T: BloomHashIndex>(&self, key: &T) -> bool {
        for k in &self.keys {
            let pos = self.pos(key, *k) as usize;
            if !*self.bits.0.get(pos).unwrap() {
                return false;
            }
        }
        true
    }
}

fn slice_hash(slice: &[u8], hash_index: u64) -> u64 {
    let mut hasher = FnvHasher::with_key(hash_index);
    hasher.write(slice);
    hasher.finish()
}

impl<T: AsRef<[u8]>> BloomHashIndex for T {
    fn hash_at_index(&self, hash_index: u64) -> u64 {
        slice_hash(self.as_ref(), hash_index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bloom_new() {
        let keys = [1, 2, 3];
        let bloom = Bloom::<16, 3>::new(100, &keys);

        assert_eq!(bloom.keys, keys);
    }

    #[test]
    fn test_bloom_add_and_contains() {
        let mut bloom = Bloom::<8, 2>::new(100, &[1, 2]);

        let item1 = vec![1, 2, 3];
        let item2 = vec![4, 5, 6];

        bloom.add(&item1);
        assert!(bloom.contains(&item1));
        assert!(!bloom.contains(&item2));
    }

    #[test]
    fn test_bloom_clear() {
        let mut bloom = Bloom::<8, 2>::new(100, &[1, 2]);

        let item = vec![1, 2, 3];
        bloom.add(&item);
        assert!(bloom.contains(&item));

        bloom.clear();
        assert!(!bloom.contains(&item));
    }

    #[test]
    fn test_multiple_items() {
        let mut bloom = Bloom::<16, 3>::new(1000, &[1, 2, 3]);

        let items = vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]];

        for item in &items {
            bloom.add(item);
        }

        for item in &items {
            assert!(bloom.contains(item));
        }

        let non_existing = vec![1, 4, 7];
        assert!(!bloom.contains(&non_existing));
    }

    use crate::lcg::LCG;

    // A bloom filter MUST never produce a false negative: every inserted item is
    // always reported as present. This is the correctness contract the witness
    // voting layer relies on.
    #[test]
    fn no_false_negatives_across_many_items() {
        let mut bloom = Bloom::<32, 7>::new(Bloom::<32, 7>::max_bits(), &[1, 2, 3, 4, 5, 6, 7]);
        let mut rng = LCG::new(0xBEEF);

        let items: Vec<Vec<u8>> = (0..512)
            .map(|_| {
                let mut v = Vec::with_capacity(16);
                v.extend_from_slice(&rng.next_u64().to_le_bytes());
                v.extend_from_slice(&rng.next_u64().to_le_bytes());
                v
            })
            .collect();
        for item in &items {
            bloom.add(item);
        }
        for item in &items {
            assert!(bloom.contains(item), "false negative for inserted item");
        }
    }

    // The measured false-positive rate must be in the right ballpark for the
    // chosen parameters. We use a generous upper bound so the test is robust to
    // hash variance (FNV here) and never flaky.
    #[test]
    fn measured_false_positive_rate_is_bounded() {
        // m = U*64 = 2048 bits, k = 7 hashes, n = 100 items.
        // Theoretical FPR ≈ (1 - e^(-k*n/m))^k ≈ 0.3%. Assert < 5% for headroom.
        const U: usize = 32;
        const K: usize = 7;
        let mut bloom = Bloom::<U, K>::new(Bloom::<U, K>::max_bits(), &[1, 2, 3, 4, 5, 6, 7]);

        // Insert items derived from a small key space (distinct 8-byte vectors).
        let inserted: Vec<Vec<u8>> = (0..100u64).map(|i| i.to_le_bytes().to_vec()).collect();
        for item in &inserted {
            bloom.add(item);
        }

        // Probe disjoint items derived from a different key space: none were inserted.
        let mut rng = LCG::new(0xC0FFEE);
        let mut false_positives = 0usize;
        let probes = 20_000usize;
        for _ in 0..probes {
            // High 32 bits set -> never collides with the 0..100 inserted range.
            let probe = (rng.next_u64() | (1u64 << 63)).to_le_bytes().to_vec();
            if bloom.contains(&probe) {
                false_positives += 1;
            }
        }
        let rate = false_positives as f64 / probes as f64;
        assert!(
            rate < 0.05,
            "measured FPR {rate:.4} unexpectedly high (k={K}, m={})",
            U * 64
        );
    }

    // `clear()` must reset every bit; no previously-added item may remain present.
    #[test]
    fn clear_resets_all_bits_for_many_items() {
        let mut bloom = Bloom::<8, 3>::new(Bloom::<8, 3>::max_bits(), &[1, 2, 3]);
        let mut rng = LCG::new(0x1234);
        let items: Vec<Vec<u8>> = (0..64)
            .map(|_| {
                let a = rng.next_u64().to_le_bytes();
                let b = rng.next_u64().to_le_bytes();
                a.iter().chain(b.iter()).copied().collect()
            })
            .collect();
        for item in &items {
            bloom.add(item);
        }
        for item in &items {
            assert!(bloom.contains(item));
        }
        bloom.clear();
        for item in &items {
            assert!(!bloom.contains(item), "item still present after clear()");
        }
    }

    // Postcard round-trip must preserve keys and bits exactly.
    #[test]
    fn serde_roundtrip_preserves_membership() {
        let mut bloom = Bloom::<16, 4>::new(Bloom::<16, 4>::max_bits(), &[10, 20, 30, 40]);
        let items: Vec<Vec<u8>> = (0..32u64).map(|i| i.to_le_bytes().to_vec()).collect();
        for item in &items {
            bloom.add(item);
        }
        let back = aether_test_support::postcard_roundtrip(&bloom);
        for item in &items {
            assert!(back.contains(item));
        }
        assert!(!back.contains(&999u64.to_le_bytes()));
    }
}
