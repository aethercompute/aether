use crate::lcg::LCG;

// Fisher-Yates shuffle, per Knuth
// https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle

pub fn deterministic_shuffle<T>(items: &mut [T], seed: u64) {
    let mut rng = LCG::new(seed);

    for i in (1..items.len()).rev() {
        let j = rng.next_range(i + 1);
        items.swap(i, j);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deterministic_shuffle_same_seed() {
        let mut vec1 = vec![1, 2, 3, 4, 5];
        let mut vec2 = vec![1, 2, 3, 4, 5];

        deterministic_shuffle(&mut vec1, 42);
        deterministic_shuffle(&mut vec2, 42);

        assert_eq!(vec1, vec2);
    }

    #[test]
    fn test_deterministic_shuffle_different_seeds() {
        let mut vec1 = vec![1, 2, 3, 4, 5];
        let mut vec2 = vec![1, 2, 3, 4, 5];

        deterministic_shuffle(&mut vec1, 42);
        deterministic_shuffle(&mut vec2, 43);

        assert_ne!(vec1, vec2);
    }

    #[test]
    fn test_deterministic_shuffle_all_elements_present() {
        let mut vec = vec![1, 2, 3, 4, 5];
        let original = vec.clone();

        deterministic_shuffle(&mut vec, 42);

        assert_eq!(vec.len(), original.len());
        for &item in &original {
            assert!(vec.contains(&item));
        }
    }

    #[test]
    fn test_deterministic_shuffle_empty_vec() {
        let mut vec: Vec<i32> = Vec::new();
        deterministic_shuffle(&mut vec, 42);
        assert!(vec.is_empty());
    }

    #[test]
    fn test_deterministic_shuffle_single_element() {
        let mut vec = vec![1];
        deterministic_shuffle(&mut vec, 42);
        assert_eq!(vec, vec![1]);
    }

    #[test]
    fn test_deterministic_shuffle_large_vec() {
        let mut vec = Vec::from_iter(1..1000);
        let original = vec.clone();

        deterministic_shuffle(&mut vec, 42);

        assert_ne!(vec, original);
        assert_eq!(vec.len(), original.len());
        for &item in &original {
            assert!(vec.contains(&item));
        }
    }

    // A shuffle is, by definition, a permutation: after shuffling, the multiset
    // of elements is unchanged (no duplicates introduced, none lost) regardless
    // of seed or size.
    fn assert_is_permutation(size: usize, seed: u64) {
        let mut v: Vec<u64> = (0..size as u64).collect();
        let original = v.clone();
        deterministic_shuffle(&mut v, seed);
        assert_eq!(v.len(), original.len(), "length changed");
        let mut sorted = v.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, original, "shuffle lost or duplicated elements");
    }

    #[test]
    fn shuffle_is_permutation_across_sizes_and_seeds() {
        for size in 0..=128usize {
            for seed in [0u64, 1, 42, 999, u64::MAX] {
                assert_is_permutation(size, seed);
            }
        }
    }

    // A non-trivial slice must actually move for at least one seed (guards against
    // a no-op regression), while empty/single-element slices are fixed points.
    #[test]
    fn shuffle_changes_order_for_some_seed() {
        let original: Vec<u64> = (0..64).collect();
        let mut moved = false;
        for seed in 0..32u64 {
            let mut v = original.clone();
            deterministic_shuffle(&mut v, seed);
            if v != original {
                moved = true;
            }
            // still a permutation
            let mut s = v.clone();
            s.sort_unstable();
            assert_eq!(s, original);
        }
        assert!(moved, "shuffle never changed the order across 32 seeds");
    }

    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]
        #[test]
        fn prop_shuffle_is_permutation(size in 0usize..200, seed in 0u64..10_000) {
            assert_is_permutation(size, seed);
        }
    }
}
