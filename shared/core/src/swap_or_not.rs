use crate::sha256::sha256v;

const SHUFFLE_ROUND_COUNT: u8 = 90;

pub fn compute_shuffled_index(index: u64, index_count: u64, seed: &[u8; 32]) -> u64 {
    assert!(index < index_count);

    let mut current_index = index;

    for current_round in 0..SHUFFLE_ROUND_COUNT {
        let hash_result = sha256v(&[seed, &[current_round]]);

        let pivot = u64::from_le_bytes(hash_result[0..8].try_into().unwrap()) % index_count;
        let flip = (pivot + index_count - current_index) % index_count;
        let position = current_index.max(flip);

        let source = sha256v(&[
            seed,
            &[current_round],
            &(position / 256).to_le_bytes()[0..4],
        ]);

        let byte = source[(position % 256) as usize / 8];
        let bit = (byte >> (position % 8)) % 2;

        current_index = if bit == 1 { flip } else { current_index };
    }

    current_index
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_shuffled_index_basic() {
        let seed = [0u8; 32];
        let index_count = 10;

        for i in 0..index_count {
            let result = compute_shuffled_index(i, index_count, &seed);
            assert!(
                result < index_count,
                "Shuffled index should be within bounds"
            );
        }
    }

    #[test]
    fn test_compute_shuffled_index_deterministic() {
        let seed = [1u8; 32];
        let index_count = 100;

        for i in 0..index_count {
            let result1 = compute_shuffled_index(i, index_count, &seed);
            let result2 = compute_shuffled_index(i, index_count, &seed);
            assert_eq!(
                result1, result2,
                "Results should be deterministic for the same input"
            );
        }
    }

    #[test]
    fn test_compute_shuffled_index_different_seeds() {
        let seed1 = [1u8; 32];
        let seed2 = [2u8; 32];
        let index = 5;
        let index_count = 100;

        let result1 = compute_shuffled_index(index, index_count, &seed1);
        let result2 = compute_shuffled_index(index, index_count, &seed2);
        assert_ne!(
            result1, result2,
            "Different seeds should produce different results"
        );
    }

    #[test]
    #[should_panic(expected = "index < index_count")]
    fn test_compute_shuffled_index_out_of_bounds() {
        let seed = [0u8; 32];
        let index_count = 10;
        compute_shuffled_index(index_count, index_count, &seed);
    }

    // ── bijection: the swap-or-not shuffle MUST be a permutation of [0, n) ──
    // This is the single most important invariant of this function: every node in
    // the cluster must agree on committee/witness assignment, which only holds if
    // `compute_shuffled_index` is a bijection. The original tests only checked
    // bounds; a collision here would silently corrupt consensus.
    fn assert_is_permutation(index_count: u64, seed: &[u8; 32]) {
        let mut outputs: Vec<u64> = (0..index_count)
            .map(|i| compute_shuffled_index(i, index_count, seed))
            .collect();
        outputs.sort_unstable();
        let expected: Vec<u64> = (0..index_count).collect();
        assert_eq!(
            outputs, expected,
            "compute_shuffled_index must be a bijection on [0, {index_count})"
        );
    }

    #[test]
    fn bijection_fixed_sizes_and_seeds() {
        for seed in [[0u8; 32], [1u8; 32], [0xffu8; 32], {
            let mut s = [0u8; 32];
            s[31] = 7;
            s
        }] {
            for n in [
                1u64, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100, 255, 256, 257,
            ] {
                assert_is_permutation(n, &seed);
            }
        }
    }

    #[test]
    fn bijection_large_sizes() {
        // Powers of two and neighbors are the most likely to expose off-by-ones.
        assert_is_permutation(1_000, &[0xab; 32]);
        assert_is_permutation(1_023, &[0xcd; 32]);
        assert_is_permutation(1_024, &[0xcd; 32]);
        assert_is_permutation(1_025, &[0xcd; 32]);
    }

    #[test]
    fn image_is_independent_of_call_order() {
        // Collecting the image must not depend on which order indices are queried.
        let seed = [3u8; 32];
        let n = 200u64;
        let fwd: std::collections::HashSet<u64> = (0..n)
            .map(|i| compute_shuffled_index(i, n, &seed))
            .collect();
        let rev: std::collections::HashSet<u64> = (0..n)
            .rev()
            .map(|i| compute_shuffled_index(i, n, &seed))
            .collect();
        assert_eq!(fwd, rev);
        assert_eq!(fwd.len(), n as usize);
    }

    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(48))]
        #[test]
        fn prop_bijection(seed in any::<[u8; 32]>(), n in 2u64..129) {
            assert_is_permutation(n, &seed);
        }
    }
}
