use aether_coordinator::{model, Coordinator, HealthChecks};
use aether_watcher::OpportunisticData;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum ClientToServerMessage {
    Join {
        run_id: String,
        checkpoint_upload: bool,
    },
    /// Sent by a client after it has finished downloading and loading the
    /// initial checkpoint. The server admits ready clients only before the
    /// first epoch; late admission is rejected until exact state synchronization
    /// is supported.
    ReadyForEpoch,
    Witness(Box<OpportunisticData>),
    HealthCheck(HealthChecks),
    Checkpoint(model::CheckpointUpdate),
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum ServerToClientMessage {
    Coordinator(Box<Coordinator>),
    Error {
        code: ServerErrorCode,
        message: String,
    },
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServerErrorCode {
    RunIdMismatch,
    NotAllowlisted,
    JoinRequired,
    LateJoinUnsupported,
    SingleClientOptimizer,
    CheckpointUploadRequired,
}

#[cfg(test)]
mod tests {
    use aether_coordinator::{model, Witness};
    use bytemuck::Zeroable;

    use super::*;

    #[test]
    fn client_to_server_join_roundtrip() {
        let msg = ClientToServerMessage::Join {
            run_id: "test-run-42".to_string(),
            checkpoint_upload: true,
        };
        let back = aether_test_support::postcard_roundtrip(&msg);
        assert!(matches!(back, ClientToServerMessage::Join { .. }));
    }

    #[test]
    fn client_to_server_ready_for_epoch_roundtrip() {
        let msg = ClientToServerMessage::ReadyForEpoch;
        let back = aether_test_support::postcard_roundtrip(&msg);
        assert!(matches!(back, ClientToServerMessage::ReadyForEpoch));
    }

    #[test]
    fn client_to_server_witness_roundtrip() {
        let msg = ClientToServerMessage::Witness(Box::new(
            aether_watcher::OpportunisticData::WarmupStep(Witness::default()),
        ));
        let back = aether_test_support::postcard_roundtrip(&msg);
        assert!(matches!(back, ClientToServerMessage::Witness(_)));
    }

    #[test]
    fn client_to_server_health_check_roundtrip() {
        let msg = ClientToServerMessage::HealthCheck(vec![]);
        let back = aether_test_support::postcard_roundtrip(&msg);
        assert!(matches!(back, ClientToServerMessage::HealthCheck(_)));
    }

    #[test]
    fn client_to_server_checkpoint_roundtrip() {
        let msg = ClientToServerMessage::Checkpoint(model::CheckpointUpdate {
            epoch: 1,
            step: 2,
            checkpoint: model::Checkpoint::Ephemeral,
        });
        let back = aether_test_support::postcard_roundtrip(&msg);
        assert!(matches!(back, ClientToServerMessage::Checkpoint(_)));
    }

    #[test]
    fn server_to_client_coordinator_roundtrip() {
        let msg = ServerToClientMessage::Coordinator(Box::new(Coordinator::zeroed()));
        let back = aether_test_support::postcard_roundtrip(&msg);

        assert!(matches!(back, ServerToClientMessage::Coordinator(_)));
    }

    #[test]
    fn server_to_client_error_roundtrip() {
        let msg = ServerToClientMessage::Error {
            code: ServerErrorCode::RunIdMismatch,
            message: "wrong run id".to_string(),
        };
        let back = aether_test_support::postcard_roundtrip(&msg);

        assert!(matches!(
            back,
            ServerToClientMessage::Error {
                code: ServerErrorCode::RunIdMismatch,
                ..
            }
        ));
    }

    #[test]
    fn safety_error_codes_roundtrip() {
        for code in [
            ServerErrorCode::LateJoinUnsupported,
            ServerErrorCode::SingleClientOptimizer,
            ServerErrorCode::CheckpointUploadRequired,
        ] {
            let msg = ServerToClientMessage::Error {
                code,
                message: "safety invariant".to_string(),
            };
            let back = aether_test_support::postcard_roundtrip(&msg);
            assert!(matches!(
                back,
                ServerToClientMessage::Error {
                    code: back_code,
                    ..
                } if back_code == code
            ));
        }
    }
}
