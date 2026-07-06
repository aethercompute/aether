use crate::{traits::TokenizedDataProvider, LengthKnownDataProvider, TokenizedData};
use aether_core::BatchId;
use anyhow::{bail, Result};

pub struct DummyDataProvider {
    seq_len: usize,
    num_sequences: u64,
}

impl DummyDataProvider {
    pub fn new(
        _token_size_in_bytes: aether_core::TokenSize,
        num_tokens_per_sequence: usize, // num tokens per sequence
        num_sequences: u64,
    ) -> Self {
        Self {
            seq_len: num_tokens_per_sequence,
            num_sequences,
        }
    }

    fn internal_get_samples(&self, num_samples: usize) -> Result<Vec<TokenizedData>> {
        let mut ret: Vec<_> = Vec::new();
        for _ in 0..num_samples {
            ret.push(TokenizedData::from_input_ids(vec![0; self.seq_len]));
        }
        Ok(ret)
    }
}

impl TokenizedDataProvider for DummyDataProvider {
    async fn get_samples(&mut self, data_ids: BatchId) -> Result<Vec<TokenizedData>> {
        for id in data_ids.iter() {
            if id >= self.num_sequences {
                bail!("id {id} >= self.num_sequences {}", self.num_sequences)
            }
        }
        self.internal_get_samples(data_ids.len())
    }
}

impl LengthKnownDataProvider for DummyDataProvider {
    fn num_sequences(&self) -> usize {
        self.num_sequences as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aether_core::{ClosedInterval, TokenSize};

    fn batch_id(start: u64, end: u64) -> BatchId {
        BatchId(ClosedInterval { start, end })
    }

    #[tokio::test]
    async fn returns_zero_filled_samples_with_requested_sequence_len() {
        let mut provider = DummyDataProvider::new(TokenSize::TwoBytes, 4, 10);

        let samples = provider.get_samples(batch_id(2, 4)).await.unwrap();

        assert_eq!(samples.len(), 3);
        assert!(samples.iter().all(|sample| sample.input_ids == vec![0; 4]));
        assert!(samples.iter().all(|sample| sample.labels.is_none()));
        assert!(samples.iter().all(|sample| sample.position_ids.is_none()));
        assert!(samples
            .iter()
            .all(|sample| sample.sequence_lengths.is_none()));
    }

    #[tokio::test]
    async fn rejects_first_out_of_range_sequence_id() {
        let mut provider = DummyDataProvider::new(TokenSize::TwoBytes, 4, 10);

        let err = provider
            .get_samples(batch_id(10, 10))
            .await
            .unwrap_err()
            .to_string();

        assert_eq!(err, "id 10 >= self.num_sequences 10");
    }

    #[tokio::test]
    async fn rejects_ranges_that_cross_the_end() {
        let mut provider = DummyDataProvider::new(TokenSize::FourBytes, 2, 3);

        let err = provider
            .get_samples(batch_id(1, 3))
            .await
            .unwrap_err()
            .to_string();

        assert_eq!(err, "id 3 >= self.num_sequences 3");
    }

    #[test]
    fn length_known_reports_configured_sequence_count() {
        let empty = DummyDataProvider::new(TokenSize::TwoBytes, 4, 0);
        assert_eq!(empty.num_sequences(), 0);
        assert!(empty.is_empty());

        let non_empty = DummyDataProvider::new(TokenSize::TwoBytes, 4, 7);
        assert_eq!(non_empty.num_sequences(), 7);
        assert!(!non_empty.is_empty());
    }
}
