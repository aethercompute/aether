use aether_coordinator::{Commitment, CommitteeProof};
use aether_core::{BatchId, MerkleRoot};
use aether_network::{BlobTicket, NetworkConnection, TransmittableDownload};
use serde::{Deserialize, Serialize};

pub type NC = NetworkConnection<Broadcast, TransmittableDownload>;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TrainingResult {
    pub batch_id: BatchId,
    pub ticket: BlobTicket,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Finished {
    pub broadcast_merkle: MerkleRoot,
    pub warmup: bool,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum BroadcastType {
    TrainingResult(TrainingResult),
    Finished(Finished),
}

impl BroadcastType {
    pub fn kind(&self) -> &'static str {
        match self {
            BroadcastType::TrainingResult(..) => "training_result",
            BroadcastType::Finished(..) => "finished",
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Broadcast {
    pub step: u32,
    pub proof: CommitteeProof,
    pub commitment: Commitment,
    pub nonce: u32,
    pub data: BroadcastType,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Serialize)]
    struct BroadcastWithRawCommitment {
        step: u32,
        proof: CommitteeProof,
        commitment: Vec<u8>,
        nonce: u32,
        data: BroadcastType,
    }

    fn encode_with_commitment_len(len: usize) -> Vec<u8> {
        postcard::to_allocvec(&BroadcastWithRawCommitment {
            step: 1,
            proof: CommitteeProof::default(),
            commitment: vec![0; len],
            nonce: 7,
            data: BroadcastType::Finished(Finished {
                broadcast_merkle: MerkleRoot::default(),
                warmup: false,
            }),
        })
        .unwrap()
    }

    #[test]
    fn broadcast_rejects_malformed_nested_commitments() {
        for len in [0, 31, 32, 95, 97, 160] {
            assert!(postcard::from_bytes::<Broadcast>(&encode_with_commitment_len(len)).is_err());
        }
        assert!(postcard::from_bytes::<Broadcast>(&encode_with_commitment_len(96)).is_ok());
    }
}
