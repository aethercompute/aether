use aether_core::{BatchId, TokenSize};
use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize, Debug, PartialEq)]
pub struct TokenizedData {
    pub input_ids: Vec<i32>,
    pub labels: Option<Vec<i32>>,
    pub position_ids: Option<Vec<i32>>,
    pub sequence_lengths: Option<Vec<i32>>,
}

impl TokenizedData {
    pub fn from_input_ids(input_ids: Vec<i32>) -> Self {
        Self {
            input_ids,
            labels: None,
            position_ids: None,
            sequence_lengths: None,
        }
    }

    pub fn new(
        input_ids: Vec<i32>,
        labels: Option<Vec<i32>>,
        position_ids: Option<Vec<i32>>,
        sequence_lengths: Option<Vec<i32>>,
    ) -> Self {
        Self {
            input_ids,
            labels,
            position_ids,
            sequence_lengths,
        }
    }

    pub fn empty() -> Self {
        Self {
            input_ids: vec![],
            labels: None,
            position_ids: None,
            sequence_lengths: None,
        }
    }
}

pub trait TokenizedDataProvider {
    fn get_samples(
        &mut self,
        data_ids: BatchId,
    ) -> impl std::future::Future<Output = Result<Vec<TokenizedData>>> + Send;
}

pub trait LengthKnownDataProvider {
    fn num_sequences(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.num_sequences() == 0
    }
}

pub(crate) fn bytes_to_tokens(data: &[u8], token_size: TokenSize) -> Result<Vec<i32>> {
    let token_len = usize::from(token_size);
    if !data.len().is_multiple_of(token_len) {
        bail!(
            "token data length {} is not divisible by token size {}",
            data.len(),
            token_len
        );
    }

    Ok(data
        .chunks_exact(token_len)
        .map(|chunk| match token_size {
            TokenSize::TwoBytes => u16::from_le_bytes([chunk[0], chunk[1]]) as i32,
            TokenSize::FourBytes => {
                u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as i32
            }
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenized_data_from_input_ids_sets_optional_fields_to_none() {
        let data = TokenizedData::from_input_ids(vec![1, 2, 3]);

        assert_eq!(data.input_ids, vec![1, 2, 3]);
        assert_eq!(data.labels, None);
        assert_eq!(data.position_ids, None);
        assert_eq!(data.sequence_lengths, None);
    }

    #[test]
    fn tokenized_data_empty_has_no_tokens_or_metadata() {
        assert_eq!(
            TokenizedData::empty(),
            TokenizedData::from_input_ids(vec![])
        );
    }

    #[test]
    fn bytes_to_two_byte_tokens_uses_little_endian_order() {
        let data = [0x34, 0x12, 0xff, 0x00, 0x00, 0x80];

        let tokens = bytes_to_tokens(&data, TokenSize::TwoBytes).unwrap();

        assert_eq!(tokens, vec![0x1234, 0x00ff, 0x8000]);
    }

    #[test]
    fn bytes_to_four_byte_tokens_uses_little_endian_order() {
        let data = [
            0x78, 0x56, 0x34, 0x12, // 0x12345678
            0xff, 0x00, 0x00, 0x00, // 255
            0x00, 0x00, 0x00, 0x80, // u32::MAX as i32 wrapping cast boundary
        ];

        let tokens = bytes_to_tokens(&data, TokenSize::FourBytes).unwrap();

        assert_eq!(tokens, vec![0x12345678, 255, i32::MIN]);
    }

    #[test]
    fn bytes_to_tokens_rejects_partial_token() {
        let err = bytes_to_tokens(&[1, 2, 3], TokenSize::TwoBytes)
            .unwrap_err()
            .to_string();

        assert_eq!(err, "token data length 3 is not divisible by token size 2");
    }

    #[test]
    fn bytes_to_tokens_accepts_empty_input() {
        assert_eq!(
            bytes_to_tokens(&[], TokenSize::FourBytes).unwrap(),
            Vec::<i32>::new()
        );
    }
}
