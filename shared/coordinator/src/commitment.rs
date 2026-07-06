use bytemuck::Zeroable;
use serde::{Deserialize, Deserializer, Serialize};

#[derive(Clone, Debug, Zeroable, Copy)]
#[repr(C)]
pub struct Commitment {
    pub data_hash: [u8; 32],
    pub signature: [u8; 64],
}

impl Serialize for Commitment {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut bytes = Vec::with_capacity(32 + 64);
        bytes.extend_from_slice(&self.data_hash);
        bytes.extend_from_slice(&self.signature);

        serializer.serialize_bytes(&bytes)
    }
}

impl<'de> Deserialize<'de> for Commitment {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let bytes: Vec<u8> = <Vec<_> as serde::Deserialize>::deserialize(deserializer)?;

        if bytes.len() != 96 {
            return Err(serde::de::Error::custom("Invalid length for Commitment"));
        }

        let mut data_hash = [0u8; 32];
        let mut signature = [0u8; 64];

        data_hash.copy_from_slice(&bytes[0..32]);
        signature.copy_from_slice(&bytes[32..96]);

        Ok(Commitment {
            data_hash,
            signature,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn commitment_postcard_roundtrip() {
        let c = Commitment {
            data_hash: [1u8; 32],
            signature: [2u8; 64],
        };
        let back = aether_test_support::postcard_roundtrip(&c);
        assert_eq!(c.data_hash, back.data_hash);
        assert_eq!(c.signature, back.signature);
    }

    #[test]
    fn commitment_rejects_short_payload() {
        let encoded = postcard::to_allocvec(&vec![0u8; 95]).unwrap();
        let result: Result<Commitment, _> = postcard::from_bytes(&encoded);
        assert!(result.is_err());
    }

    #[test]
    fn commitment_rejects_long_payload() {
        let encoded = postcard::to_allocvec(&vec![0u8; 97]).unwrap();
        let result: Result<Commitment, _> = postcard::from_bytes(&encoded);
        assert!(result.is_err());
    }

    #[test]
    fn commitment_rejects_empty_payload() {
        let encoded = postcard::to_allocvec(&Vec::<u8>::new()).unwrap();
        let result: Result<Commitment, _> = postcard::from_bytes(&encoded);
        assert!(result.is_err());
    }
}
