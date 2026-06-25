mod cli;
mod client;
mod fetch_data;
mod protocol;
mod state;
mod tui;

pub use cli::{prepare_environment, print_identity_keys, read_identity_secret_key, TrainArgs};
pub use client::Client;
pub use protocol::{Broadcast, BroadcastType, Finished, TrainingResult, NC};
pub use state::{
    CheckpointConfig, GcsUploadInfo, HubUploadInfo, InitRunError, RoundState, RunInitConfig,
    RunInitConfigAndIO, UploadInfo,
};
pub use tui::{ClientTUI, ClientTUIState};

#[derive(Clone, Debug)]
pub struct WandBInfo {
    pub project: String,
    pub run: String,
    pub group: Option<String>,
    pub entity: Option<String>,
    pub api_key: String,
}
