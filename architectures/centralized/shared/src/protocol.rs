use psyche_coordinator::{model, Coordinator, HealthChecks};
use psyche_watcher::OpportunisticData;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum ClientToServerMessage {
    Join {
        run_id: String,
    },
    /// Sent by a client after it has finished downloading and loading the
    /// checkpoint for the current coordinator state. The server will only admit
    /// "ready" clients into an epoch, so slow joiners never disrupt active
    /// training.
    ReadyForEpoch,
    Witness(Box<OpportunisticData>),
    HealthCheck(HealthChecks),
    Checkpoint(model::Checkpoint),
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum ServerToClientMessage {
    Coordinator(Box<Coordinator>),
}

#[cfg(test)]
mod tests {
    use psyche_coordinator::{model, Witness};

    use super::*;

    #[test]
    fn client_to_server_join_roundtrip() {
        let msg = ClientToServerMessage::Join {
            run_id: "test-run-42".to_string(),
        };
        let back = psyche_test_support::postcard_roundtrip(&msg);
        assert!(matches!(back, ClientToServerMessage::Join { .. }));
    }

    #[test]
    fn client_to_server_ready_for_epoch_roundtrip() {
        let msg = ClientToServerMessage::ReadyForEpoch;
        let back = psyche_test_support::postcard_roundtrip(&msg);
        assert!(matches!(back, ClientToServerMessage::ReadyForEpoch));
    }

    #[test]
    fn client_to_server_witness_roundtrip() {
        let msg = ClientToServerMessage::Witness(Box::new(
            psyche_watcher::OpportunisticData::WarmupStep(Witness::default()),
        ));
        let back = psyche_test_support::postcard_roundtrip(&msg);
        assert!(matches!(back, ClientToServerMessage::Witness(_)));
    }

    #[test]
    fn client_to_server_health_check_roundtrip() {
        let msg = ClientToServerMessage::HealthCheck(vec![]);
        let back = psyche_test_support::postcard_roundtrip(&msg);
        assert!(matches!(back, ClientToServerMessage::HealthCheck(_)));
    }

    #[test]
    fn client_to_server_checkpoint_roundtrip() {
        let msg = ClientToServerMessage::Checkpoint(model::Checkpoint::Ephemeral);
        let back = psyche_test_support::postcard_roundtrip(&msg);
        assert!(matches!(back, ClientToServerMessage::Checkpoint(_)));
    }
}
