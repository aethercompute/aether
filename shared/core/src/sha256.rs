use sha2::{Digest, Sha256};

pub fn sha256(data: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hasher.finalize().into()
}

pub fn sha256v(data: &[&[u8]]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    for val in data {
        hasher.update(val)
    }
    hasher.finalize().into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sha256() {
        let data = b"Hello, world!";
        let hash = sha256(data);
        assert_eq!(
            hash,
            [
                0x31, 0x5f, 0x5b, 0xdb, 0x76, 0xd0, 0x78, 0xc4, 0x3b, 0x8a, 0xc0, 0x06, 0x4e, 0x4a,
                0x01, 0x64, 0x61, 0x2b, 0x1f, 0xce, 0x77, 0xc8, 0x69, 0x34, 0x5b, 0xfc, 0x94, 0xc7,
                0x58, 0x94, 0xed, 0xd3
            ]
        );
    }

    // Standard FIPS-180 known-answer vector for the empty string and "abc".
    // These pin the hash to a published reference so a dependency regression
    // (sha2 upgrade, optimizer bug) can't silently change digests.
    #[test]
    fn sha256_known_answer_vectors() {
        // sha256("") = e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
        assert_eq!(
            sha256(b""),
            [
                0xe3, 0xb0, 0xc4, 0x42, 0x98, 0xfc, 0x1c, 0x14, 0x9a, 0xfb, 0xf4, 0xc8, 0x99, 0x6f,
                0xb9, 0x24, 0x27, 0xae, 0x41, 0xe4, 0x64, 0x9b, 0x93, 0x4c, 0xa4, 0x95, 0x99, 0x1b,
                0x78, 0x52, 0xb8, 0x55,
            ]
        );
        // sha256("abc") = ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad
        assert_eq!(
            sha256(b"abc"),
            [
                0xba, 0x78, 0x16, 0xbf, 0x8f, 0x01, 0xcf, 0xea, 0x41, 0x41, 0x40, 0xde, 0x5d, 0xae,
                0x22, 0x23, 0xb0, 0x03, 0x61, 0xa3, 0x96, 0x17, 0x7a, 0x9c, 0xb4, 0x10, 0xff, 0x61,
                0xf2, 0x00, 0x15, 0xad,
            ]
        );
    }

    // `sha256v` must be equivalent to hashing the concatenation of its inputs.
    // A regression here breaks every Merkle leaf/intermediate hash and every
    // gossip-topic derivation at once.
    #[test]
    fn sha256v_matches_concatenation() {
        let concatenated = sha256(b"abc");
        // single segment
        assert_eq!(sha256v(&[b"abc"]), concatenated);
        // many segments, same bytes in order
        assert_eq!(sha256v(&[b"a", b"b", b"c"]), concatenated);
        // empty segments don't change the digest
        assert_eq!(sha256v(&[b"", b"abc", b""]), concatenated);
    }

    #[test]
    fn sha256v_empty_inputs() {
        // hashing no bytes == sha256("")
        assert_eq!(sha256v(&[]), sha256(b""));
        assert_eq!(sha256v(&[b"", b""]), sha256(b""));
    }
}
