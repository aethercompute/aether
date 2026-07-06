#![allow(clippy::manual_is_multiple_of)]

use std::fmt::Debug;

use crate::sha256::sha256v;

use bytemuck::Zeroable;
use serde::{Deserialize, Serialize};
use ts_rs::TS;

// from https://github.com/solana-labs/solana/blob/27eff8408b7223bb3c4ab70523f8a8dca3ca6645/merkle-tree/src/merkle_tree.rs

// We need to discern between leaf and intermediate nodes to prevent trivial second
// pre-image attacks.
// https://flawed.net.nz/2018/02/21/attacking-merkle-trees-with-a-second-preimage-attack
const LEAF_PREFIX: &[u8] = &[0];
const INTERMEDIATE_PREFIX: &[u8] = &[1];

macro_rules! hash_leaf {
    {$d:ident} => {
        sha256v(&[LEAF_PREFIX, $d])
    }
}

macro_rules! hash_intermediate {
    {$l:ident, $r:ident} => {
        sha256v(&[INTERMEDIATE_PREFIX, $l.as_ref(), $r.as_ref()])
    }
}

/// This wrapper carries a 32-byte hash with convenient (de)serialization.
#[derive(Serialize, Deserialize, PartialEq, Eq, Clone, Default, Zeroable, Copy, TS)]
pub struct HashWrapper {
    pub inner: [u8; 32],
}

impl HashWrapper {
    pub fn new(inner: [u8; 32]) -> Self {
        Self { inner }
    }

    pub fn fmt_short(&self) -> String {
        data_encoding::HEXLOWER.encode(&self.inner[..5])
    }

    pub fn fmt_full(&self) -> String {
        data_encoding::HEXLOWER.encode(&self.inner)
    }
}

impl Debug for HashWrapper {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "HashWrapper({})", self.fmt_full())
    }
}

impl AsRef<[u8]> for HashWrapper {
    fn as_ref(&self) -> &[u8] {
        &self.inner
    }
}

#[derive(Debug)]
pub struct MerkleTree {
    leaf_count: usize,
    nodes: Vec<HashWrapper>,
}

#[derive(Debug, PartialEq, Eq)]
pub struct ProofEntry<'a>(
    &'a HashWrapper,
    Option<&'a HashWrapper>,
    Option<&'a HashWrapper>,
);

#[derive(Debug, PartialEq, Eq, Clone, Serialize, Deserialize)]
pub struct OwnedProofEntry {
    target: HashWrapper,
    left_sibling: Option<HashWrapper>,
    right_sibling: Option<HashWrapper>,
}

impl<'a> ProofEntry<'a> {
    pub fn new(
        target: &'a HashWrapper,
        left_sibling: Option<&'a HashWrapper>,
        right_sibling: Option<&'a HashWrapper>,
    ) -> Self {
        assert!(left_sibling.is_none() ^ right_sibling.is_none());
        Self(target, left_sibling, right_sibling)
    }
}

impl<'a> From<ProofEntry<'a>> for OwnedProofEntry {
    fn from(value: ProofEntry<'a>) -> Self {
        Self {
            target: *value.0,
            left_sibling: value.1.cloned(),
            right_sibling: value.2.cloned(),
        }
    }
}

#[derive(Debug, Default, PartialEq, Eq)]
pub struct Proof<'a>(Vec<ProofEntry<'a>>);

#[derive(Debug, Default, PartialEq, Eq, Clone, Deserialize, Serialize)]
pub struct OwnedProof {
    entries: Vec<OwnedProofEntry>,
}

impl<'a> From<Proof<'a>> for OwnedProof {
    fn from(value: Proof<'a>) -> Self {
        Self {
            entries: value.0.into_iter().map(|x| x.into()).collect(),
        }
    }
}

impl OwnedProof {
    pub fn verify(&self, candidate: HashWrapper) -> bool {
        let result = self.entries.iter().try_fold(candidate, |candidate, pe| {
            // The parent is H(prefix ‖ left ‖ right). `candidate` is whichever
            // child we're currently proving; the sibling occupies the other slot.
            let lsib = pe.left_sibling.unwrap_or(candidate);
            let rsib = pe.right_sibling.unwrap_or(candidate);
            let hash = HashWrapper::new(hash_intermediate!(lsib, rsib));

            if hash == pe.target {
                Some(hash)
            } else {
                None
            }
        });
        result.is_some()
    }

    pub fn verify_item<T: AsRef<[u8]>>(&self, item: &T) -> bool {
        let candidate_item = item.as_ref();
        self.verify(HashWrapper::new(hash_leaf!(candidate_item)))
    }

    pub fn get_root(&self) -> Option<&HashWrapper> {
        self.entries.last().map(|x| &x.target)
    }
}

impl<'a> Proof<'a> {
    pub fn push(&mut self, entry: ProofEntry<'a>) {
        self.0.push(entry)
    }

    pub fn verify(&self, candidate: HashWrapper) -> bool {
        let result = self.0.iter().try_fold(candidate, |candidate, pe| {
            let lsib = pe.1.unwrap_or(&candidate);
            let rsib = pe.2.unwrap_or(&candidate);
            let hash = HashWrapper::new(hash_intermediate!(lsib, rsib));

            if hash == *pe.0 {
                Some(hash)
            } else {
                None
            }
        });
        result.is_some()
    }

    pub fn verify_item<T: AsRef<[u8]>>(&self, item: &T) -> bool {
        let candidate_item = item.as_ref();
        self.verify(HashWrapper::new(hash_leaf!(candidate_item)))
    }

    pub fn get_root(&self) -> Option<&HashWrapper> {
        self.0.last().map(|x| x.0)
    }
}

impl MerkleTree {
    #[inline]
    fn next_level_len(level_len: usize) -> usize {
        if level_len == 1 {
            0
        } else {
            level_len.div_ceil(2)
        }
    }

    fn calculate_vec_capacity(leaf_count: usize) -> usize {
        // the most nodes consuming case is when n-1 is full balanced binary tree
        // then n will cause the previous tree add a left only path to the root
        // this cause the total nodes number increased by tree height, we use this
        // condition as the max nodes consuming case.
        // n is current leaf nodes number
        // assuming n-1 is a full balanced binary tree, n-1 tree nodes number will be
        // 2(n-1) - 1, n tree height is closed to log2(n) + 1
        // so the max nodes number is 2(n-1) - 1 + log2(n) + 1, finally we can use
        // 2n + log2(n+1) as a safe capacity value.
        // test results:
        // 8192 leaf nodes(full balanced):
        // computed cap is 16398, actually using is 16383
        // 8193 leaf nodes:(full balanced plus 1 leaf):
        // computed cap is 16400, actually using is 16398
        // about performance: current used fast_math log2 code is constant algo time
        if leaf_count > 0 {
            fast_math::log2_raw(leaf_count as f32) as usize + 2 * leaf_count + 1
        } else {
            0
        }
    }

    pub fn new<T: AsRef<[u8]>>(items: &[T]) -> Self {
        let cap = MerkleTree::calculate_vec_capacity(items.len());
        let mut mt = MerkleTree {
            leaf_count: items.len(),
            nodes: Vec::with_capacity(cap),
        };

        for item in items {
            let item = item.as_ref();
            let hash = HashWrapper::new(hash_leaf!(item));
            mt.nodes.push(hash);
        }

        let mut level_len = MerkleTree::next_level_len(items.len());
        let mut level_start = items.len();
        let mut prev_level_len = items.len();
        let mut prev_level_start = 0;
        while level_len > 0 {
            for i in 0..level_len {
                let prev_level_idx = 2 * i;
                let lsib = &mt.nodes[prev_level_start + prev_level_idx];
                let rsib = if prev_level_idx + 1 < prev_level_len {
                    &mt.nodes[prev_level_start + prev_level_idx + 1]
                } else {
                    // Duplicate last entry if the level length is odd
                    &mt.nodes[prev_level_start + prev_level_idx]
                };

                let hash = HashWrapper::new(hash_intermediate!(lsib, rsib));
                mt.nodes.push(hash);
            }
            prev_level_start = level_start;
            prev_level_len = level_len;
            level_start += level_len;
            level_len = MerkleTree::next_level_len(level_len);
        }

        mt
    }

    pub fn get_root(&self) -> Option<&HashWrapper> {
        self.nodes.iter().last()
    }

    pub fn find_path(&self, index: usize) -> Option<Proof<'_>> {
        if index >= self.leaf_count {
            return None;
        }

        let mut level_len = self.leaf_count;
        let mut level_start = 0;
        let mut path = Proof::default();
        let mut node_index = index;
        let mut lsib = None;
        let mut rsib = None;
        while level_len > 0 {
            let level = &self.nodes[level_start..(level_start + level_len)];

            let target = &level[node_index];
            if lsib.is_some() || rsib.is_some() {
                path.push(ProofEntry::new(target, lsib, rsib));
            }
            if node_index % 2 == 0 {
                lsib = None;
                rsib = if node_index + 1 < level.len() {
                    Some(&level[node_index + 1])
                } else {
                    Some(&level[node_index])
                };
            } else {
                lsib = Some(&level[node_index - 1]);
                rsib = None;
            }
            node_index /= 2;

            level_start += level_len;
            level_len = MerkleTree::next_level_len(level_len);
        }
        Some(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST: &[&[u8]] = &[
        b"my", b"very", b"eager", b"mother", b"just", b"served", b"us", b"nine", b"pizzas",
        b"make", b"prime",
    ];
    const BAD: &[&[u8]] = &[b"bad", b"missing", b"false"];

    #[test]
    fn test_tree_from_empty() {
        let mt = MerkleTree::new::<[u8; 0]>(&[]);
        assert_eq!(mt.get_root(), None);
    }

    #[test]
    fn test_tree_from_one() {
        let input = b"test";
        let mt = MerkleTree::new(&[input]);
        let expected = HashWrapper::new(hash_leaf!(input));
        assert_eq!(mt.get_root(), Some(&expected));
    }

    #[test]
    fn test_path_creation() {
        let mt = MerkleTree::new(TEST);
        for (i, _s) in TEST.iter().enumerate() {
            let _path = mt.find_path(i).unwrap();
        }
    }

    #[test]
    fn test_path_creation_bad_index() {
        let mt = MerkleTree::new(TEST);
        assert_eq!(mt.find_path(TEST.len()), None);
    }

    #[test]
    fn test_path_verify_good() {
        let mt = MerkleTree::new(TEST);
        for (i, s) in TEST.iter().enumerate() {
            let hash = HashWrapper::new(hash_leaf!(s));
            let path = mt.find_path(i).unwrap();
            assert!(path.verify(hash));
        }
    }

    #[test]
    fn test_path_verify_bad() {
        let mt = MerkleTree::new(TEST);
        for (i, s) in BAD.iter().enumerate() {
            let hash = HashWrapper::new(hash_leaf!(s));
            let path = mt.find_path(i).unwrap();
            assert!(!path.verify(hash));
        }
    }

    #[test]
    fn test_proof_entry_instantiation_lsib_set() {
        ProofEntry::new(&HashWrapper::default(), Some(&HashWrapper::default()), None);
    }

    #[test]
    fn test_proof_entry_instantiation_rsib_set() {
        ProofEntry::new(&HashWrapper::default(), None, Some(&HashWrapper::default()));
    }

    #[test]
    fn test_nodes_capacity_compute() {
        let iteration_count = |mut leaf_count: usize| -> usize {
            let mut capacity = 0;
            while leaf_count > 0 {
                capacity += leaf_count;
                leaf_count = MerkleTree::next_level_len(leaf_count);
            }
            capacity
        };

        // test max 64k leaf nodes compute
        for leaf_count in 0..65536 {
            let math_count = MerkleTree::calculate_vec_capacity(leaf_count);
            let iter_count = iteration_count(leaf_count);
            assert!(math_count >= iter_count);
        }
    }

    #[test]
    #[should_panic]
    fn test_proof_entry_instantiation_both_clear() {
        ProofEntry::new(&HashWrapper::default(), None, None);
    }

    #[test]
    #[should_panic]
    fn test_proof_entry_instantiation_both_set() {
        ProofEntry::new(
            &HashWrapper::default(),
            Some(&HashWrapper::default()),
            Some(&HashWrapper::default()),
        );
    }

    // ── exhaustive verification: for every tree size and every leaf, the proof
    //    verifies AND its root matches the tree root. Odd sizes (which trigger
    //    the "duplicate last node" branch) are the most bug-prone. ───────────────
    #[test]
    fn every_leaf_proof_verifies_and_matches_root() {
        for size in 1..=64usize {
            let items: Vec<Vec<u8>> = (0..size)
                .map(|i| (i as u32).to_le_bytes().to_vec())
                .collect();
            let mt = MerkleTree::new(&items);
            let root = mt.get_root().expect("non-empty tree has a root");
            for (i, item) in items.iter().enumerate() {
                let proof = mt.find_path(i).expect("valid leaf index");
                let item_bytes = item.as_slice();
                let leaf_hash = HashWrapper::new(hash_leaf!(item_bytes));
                assert!(
                    proof.verify(leaf_hash),
                    "proof failed for size={size} leaf={i}"
                );
                // A single-leaf tree has a trivial (empty) proof: its root *is* the
                // leaf, so the proof carries no entries. For every larger tree the
                // proof's reconstructed root must match the tree's root.
                if size == 1 {
                    assert!(proof.get_root().is_none(), "size-1 proof should be empty");
                    assert_eq!(
                        *root, leaf_hash,
                        "single-leaf root must equal the leaf hash"
                    );
                } else {
                    assert_eq!(
                        proof.get_root(),
                        Some(root),
                        "proof root != tree root for size={size} leaf={i}"
                    );
                }
            }
        }
    }

    // A proof must NOT verify for an item that was tampered with.
    #[test]
    fn proof_rejects_tampered_item() {
        let items: Vec<&[u8]> = vec![b"alpha", b"beta", b"gamma", b"delta"];
        let mt = MerkleTree::new(&items);
        let proof = mt.find_path(1).unwrap();
        assert!(!proof.verify_item(&b"beta!"));
        assert!(!proof.verify_item(&b"BETA"));
        assert!(proof.verify_item(&b"beta"));
    }

    // The leaf/intermediate prefixes are what defend against second-preimage
    // attacks. An attacker must not be able to pass off an interior node's
    // children as a leaf (or vice versa). The two hash functions must therefore
    // produce different digests for the same bytes.
    #[test]
    fn leaf_and_intermediate_prefixes_diverge() {
        let payload = b"same-bytes";
        let leaf = HashWrapper::new(hash_leaf! {payload});
        // An intermediate node hashing the payload against itself.
        let inter = HashWrapper::new(hash_intermediate!(leaf, leaf));
        assert_ne!(
            leaf.inner, inter.inner,
            "leaf and intermediate hashes must differ (prefix defense)"
        );
        assert_ne!(
            LEAF_PREFIX, INTERMEDIATE_PREFIX,
            "prefixes must be distinct bytes"
        );
    }

    // OwnedProof (the serializable form) must verify identically to the borrowed
    // Proof and survive a postcard round-trip.
    #[test]
    fn owned_proof_roundtrips_and_verifies() {
        let items: Vec<&[u8]> = vec![b"a", b"b", b"c", b"d", b"e"];
        let mt = MerkleTree::new(&items);
        for (i, item) in items.iter().enumerate() {
            let proof = mt.find_path(i).unwrap();
            let owned: OwnedProof = proof.into();
            let leaf_hash = HashWrapper::new(hash_leaf!(item));
            assert!(
                owned.verify(leaf_hash),
                "owned proof verify failed at leaf {i}"
            );

            let roundtripped = aether_test_support::postcard_roundtrip(&owned);
            assert!(
                roundtripped.verify(leaf_hash),
                "roundtripped owned proof failed at leaf {i}"
            );
            assert!(roundtripped.verify_item(item));
            assert!(!roundtripped.verify_item(&b"not-a-leaf"));
        }
    }

    // Determinism: identical inputs always produce the identical root.
    #[test]
    fn root_is_deterministic() {
        let items: Vec<&[u8]> = vec![b"x", b"y", b"z"];
        let r1 = MerkleTree::new(&items).get_root().copied();
        let r2 = MerkleTree::new(&items).get_root().copied();
        assert_eq!(r1, r2);
    }
}
