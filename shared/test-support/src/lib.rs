//! Shared test helpers for the psyche workspace.
//!
//! Pulled in as a `[dev-dependencies]` entry by crates that want:
//!   - reproducible randomness ([`seeded_rng`]),
//!   - one-shot serialization round-trip checks ([`assert_postcard_roundtrip`],
//!     [`assert_serde_json_roundtrip`]).
//!
//! Keeping these here avoids duplicating the same boilerplate in every crate's
//! `#[cfg(test)]` block.

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// A deterministic RNG seeded from a `u64`.
///
/// Tests that exercise randomness but must be reproducible (and thus parallel-safe
/// under `cargo test`) should derive all randomness from a fixed seed via this.
pub fn seeded_rng(seed: u64) -> ChaCha8Rng {
    ChaCha8Rng::seed_from_u64(seed)
}

/// Serialize `value` with postcard, deserialize it, and return the result.
/// Also asserts serialization is byte-deterministic.
///
/// Catches silent wire-format regressions on types that cross the network.
pub fn postcard_roundtrip<T>(value: &T) -> T
where
    T: serde::Serialize + serde::de::DeserializeOwned,
{
    let bytes = postcard::to_allocvec(value).expect("postcard serialize");
    let bytes2 = postcard::to_allocvec(value).expect("postcard re-serialize");
    assert_eq!(
        bytes, bytes2,
        "postcard serialization is not byte-deterministic"
    );
    postcard::from_bytes(&bytes).expect("postcard deserialize")
}

/// Serialize `value` with postcard, deserialize it, and assert the result equals
/// the original.
pub fn assert_postcard_roundtrip<T>(value: &T)
where
    T: serde::Serialize + serde::de::DeserializeOwned + PartialEq + std::fmt::Debug,
{
    let back = postcard_roundtrip(value);
    assert_eq!(*value, back, "postcard round-trip changed the value");
}

/// Serialize `value` with serde_json, deserialize it, and assert equality.
pub fn assert_serde_json_roundtrip<T>(value: &T)
where
    T: serde::Serialize + serde::de::DeserializeOwned + PartialEq + std::fmt::Debug,
{
    let s = serde_json::to_string(value).expect("serde_json serialize");
    let back: T = serde_json::from_str(&s).expect("serde_json deserialize");
    assert_eq!(*value, back, "serde_json round-trip changed the value");
}

#[cfg(test)]
mod tests {
    use rand::Rng;

    use super::*;

    #[derive(serde::Serialize, serde::Deserialize, PartialEq, Debug)]
    struct Pair(u32, String);

    #[test]
    fn roundtrip_helpers_work() {
        assert_postcard_roundtrip(&Pair(7, "hi".into()));
        assert_serde_json_roundtrip(&Pair(7, "hi".into()));
    }

    #[test]
    fn seeded_rng_is_deterministic() {
        let mut a = seeded_rng(42);
        let mut b = seeded_rng(42);
        for _ in 0..16 {
            assert_eq!(a.random::<u64>(), b.random::<u64>());
        }
    }
}
