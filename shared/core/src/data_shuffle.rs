use bytemuck::Zeroable;
use serde::{Deserialize, Serialize};
use ts_rs::TS;

#[derive(Serialize, Deserialize, Clone, Debug, Zeroable, Copy, PartialEq, TS)]
#[repr(C)]
#[derive(Default)]
pub enum Shuffle {
    #[default]
    DontShuffle,
    Seeded([u8; 32]),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_dont_shuffle() {
        assert_eq!(Shuffle::default(), Shuffle::DontShuffle);
    }

    #[test]
    fn seeded_shuffle_preserves_entire_seed() {
        let seed = [7u8; 32];
        let shuffle = Shuffle::Seeded(seed);
        match shuffle {
            Shuffle::Seeded(actual) => assert_eq!(actual, seed),
            Shuffle::DontShuffle => panic!("seeded shuffle lost its seed"),
        }
    }

    #[test]
    fn postcard_roundtrip_preserves_shuffle_mode() {
        for shuffle in [Shuffle::DontShuffle, Shuffle::Seeded([0xab; 32])] {
            let decoded = aether_test_support::postcard_roundtrip(&shuffle);
            assert_eq!(decoded, shuffle);
        }
    }
}
