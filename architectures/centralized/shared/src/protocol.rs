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
