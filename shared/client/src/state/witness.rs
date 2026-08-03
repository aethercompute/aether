use aether_coordinator::{Coordinator, Witness, WitnessMetadata};
use aether_core::{MerkleRoot, MerkleTree, NodeIdentity};
use aether_watcher::OpportunisticData;
use thiserror::Error;
use tokio::{
    sync::mpsc::{self},
    task::JoinHandle,
};
use tracing::{info, trace, warn};

use super::{
    evals::{EvalError, MaybeRunningEvals, ModelTaskRunner, RunningEvals},
    round_state::RoundState,
    types::PayloadState,
};

#[derive(Debug, Error)]
pub enum WitnessingError {
    #[error("Failed to stop evals")]
    StopEvals(#[from] EvalError),

    #[error("Couldn't start evals - no trainers passed to us")]
    NoTrainers,

    #[error("Failed to send witness, channel closed?")]
    Send,

    #[error("Witness send thread crashed")]
    SendThreadCrashed,
}

pub struct WitnessStepMetadata {
    pub identity: NodeIdentity,
    pub model_task_runner: ModelTaskRunner,
    pub tx_witness: mpsc::UnboundedSender<OpportunisticData>,
}

#[derive(Debug)]
pub struct WitnessStep {
    evals: RunningEvals,
    sending_witness: Option<JoinHandle<Result<(), WitnessingError>>>,
}

impl WitnessStepMetadata {
    pub fn start(
        &self,
        _client_index: u64,
        _state: &Coordinator,
        trainers: MaybeRunningEvals,
        previous_round: &mut RoundState,
        current_round: &mut RoundState,
        metadata: WitnessMetadata,
    ) -> Result<WitnessStep, WitnessingError> {
        if trainers.is_empty() {
            return Err(WitnessingError::NoTrainers);
        }

        let evals = self.model_task_runner.start_if_not_running(trainers);

        let sending_witness = if let Some(witness) =
            WitnessStep::get_witness_to_send(previous_round, current_round)
        {
            let tx_witness = self.tx_witness.clone();
            Some(tokio::task::spawn(async move {
                tx_witness
                    .send(OpportunisticData::WitnessStep(witness, metadata))
                    .map_err(|_| WitnessingError::Send)
            }))
        } else {
            None
        };
        Ok(WitnessStep {
            evals,
            sending_witness,
        })
    }
}

impl WitnessStep {
    pub async fn finish(self) -> Result<RunningEvals, WitnessingError> {
        if let Some(witness_thread) = self.sending_witness {
            witness_thread
                .await
                .map_err(|_| WitnessingError::SendThreadCrashed)??;
        }
        Ok(self.evals)
    }

    pub fn get_witness_to_send(
        previous_round: &mut RoundState,
        current_round: &mut RoundState,
    ) -> Option<Witness> {
        if previous_round.sent_witness {
            return None;
        }

        if previous_round
            .batch_ids_not_yet_trained_on
            .lock()
            .unwrap_or_else(|poisoned| {
                warn!("round batch tracking lock poisoned; recovering state");
                poisoned.into_inner()
            })
            .is_some()
        {
            info!("Withholding witness because expected result payloads are missing");
            return None;
        }

        if previous_round
            .downloads
            .lock()
            .unwrap_or_else(|poisoned| {
                warn!("round downloads lock poisoned; recovering state");
                poisoned.into_inner()
            })
            .values()
            .any(|payload| {
                !matches!(payload, PayloadState::Deserializing(task) if task.is_finished())
            })
        {
            info!("Withholding witness because result payload processing is incomplete");
            return None;
        }

        if !current_round.sent_finished
            || current_round
                .data_assignments
                .values()
                .any(|trainer| !current_round.clients_finished.contains_key(trainer))
        {
            info!("Withholding witness because assigned trainers have not finished");
            return None;
        }

        let (_, proof, _) = current_round.committee_info.as_ref()?;
        if proof.witness.is_false() {
            return None;
        }

        let merkle = MerkleTree::new(&previous_round.broadcasts);
        let broadcast_merkle = merkle.get_root().cloned().unwrap_or(MerkleRoot::default());

        let (participant_bloom, broadcast_bloom) = previous_round
            .blooms
            .lock()
            .unwrap_or_else(|poisoned| {
                warn!("round blooms lock poisoned; recovering state");
                poisoned.into_inner()
            })
            .unwrap_or_default();

        info!("Submitting witness blooms");
        previous_round.sent_witness = true;

        trace!("Participant bloom: {:?}", participant_bloom);
        trace!("Broadcast bloom: {:?}", broadcast_bloom);
        trace!("Merkle root: 0x{}", hex::encode(broadcast_merkle.inner));

        Some(Witness {
            proof: *proof,
            participant_bloom,
            broadcast_bloom,
            broadcast_merkle,
        })
    }
}

#[cfg(test)]
mod tests {
    use aether_coordinator::{CommitteeProof, CommitteeSelection, WitnessProof};
    use aether_core::{NodeIdentity, SmallBoolean};

    use super::*;

    #[test]
    fn get_witness_to_send_recovers_from_poisoned_blooms_lock() {
        let mut previous_round = RoundState::new();
        let mut current_round = RoundState::new();
        current_round.sent_finished = true;
        current_round.committee_info = Some((
            CommitteeProof::default(),
            WitnessProof {
                witness: SmallBoolean::TRUE,
                ..WitnessProof::default()
            },
            CommitteeSelection::new(0, 1, 0, 1, 7).unwrap(),
        ));

        let _ = std::panic::catch_unwind(|| {
            let _guard = previous_round
                .blooms
                .lock()
                .expect("test lock should start clean");
            panic!("poison blooms lock");
        });

        let witness = WitnessStep::get_witness_to_send(&mut previous_round, &mut current_round);

        assert!(witness.is_some());
        assert!(previous_round.sent_witness);
    }

    #[test]
    fn witness_is_withheld_when_an_expected_payload_is_missing() {
        let mut previous_round = RoundState::new();
        let mut current_round = elected_round();
        *previous_round.batch_ids_not_yet_trained_on.lock().unwrap() =
            Some(["B[7,7]".parse().unwrap()].into_iter().collect());

        assert!(
            WitnessStep::get_witness_to_send(&mut previous_round, &mut current_round).is_none()
        );
        assert!(!previous_round.sent_witness);
    }

    #[test]
    fn witness_is_withheld_until_every_assigned_trainer_finishes() {
        let mut previous_round = RoundState::new();
        let mut current_round = elected_round();
        let trainer = NodeIdentity::from_single_key([3; 32]);
        current_round
            .data_assignments
            .insert("B[7,7]".parse().unwrap(), trainer);

        assert!(
            WitnessStep::get_witness_to_send(&mut previous_round, &mut current_round).is_none()
        );
        assert!(!previous_round.sent_witness);
    }

    fn elected_round() -> RoundState {
        let mut round = RoundState::new();
        round.sent_finished = true;
        round.committee_info = Some((
            CommitteeProof::default(),
            WitnessProof {
                witness: SmallBoolean::TRUE,
                ..WitnessProof::default()
            },
            CommitteeSelection::new(0, 1, 0, 1, 7).unwrap(),
        ));
        round
    }
}
