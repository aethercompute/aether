use aether_centralized_shared::{ClientToServerMessage, ServerErrorCode, ServerToClientMessage};
use aether_coordinator::model::{self, Checkpoint, LLMTrainingDataLocation, Model, LLM};
use aether_coordinator::{
    assign_data_for_state, Client, ClientState, CommitteeSelection, Coordinator, CoordinatorError,
    HealthChecks, Round, RunState, TickResult, SOLANA_MAX_NUM_CLIENTS,
};
use anyhow::{anyhow, bail, Result};
use async_trait::async_trait;

use aether_core::{FixedVec, NodeIdentity, OptimizerDefinition, Shuffle, SizedIterator, TokenSize};
use aether_data_provider::{
    download_model_file_async, download_model_from_gcs_async, download_model_repo_async,
    DataProvider, DataProviderTcpServer, DataServerTui, LocalDataProvider,
    PreprocessedDataProvider, Split,
};
use aether_network::{ClientNotification, PublicKey, TcpServer};
use aether_tui::{
    logging::LoggerWidget, maybe_start_render_loop, CustomWidget, MaybeTui, TabbedWidget,
};
use aether_watcher::{CoordinatorTui, OpportunisticData};
use rand::RngCore;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::hash::{DefaultHasher, Hasher};
use std::net::{Ipv4Addr, SocketAddr};
use std::ops::ControlFlow;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc::{channel, Receiver, Sender, UnboundedSender};
use tokio::sync::Notify;
use tokio::time::{interval, MissedTickBehavior};
use tokio::{select, time::Interval};
use tokio_util::sync::CancellationToken;
use tracing::{debug, info, info_span, warn, Instrument};
use wandb::LogData;

use crate::dashboard::{DashboardState, DashboardTui};
use crate::web::{self, LossPoint, WandbInfo, WebState};

/// Upper bound on the number of samples retained in `loss_history`. When this
/// is reached the older half of the history is decimated (every other point
/// dropped) instead of dropping the oldest samples, so the full step range is
/// always represented while memory stays bounded. Recent data keeps full
/// resolution; older data gets progressively coarser.
const MAX_LOSS_HISTORY: usize = 5000;

pub(super) type TabWidgetTypes = (
    DashboardTui,
    CoordinatorTui,
    MaybeTui<DataServerTui>,
    LoggerWidget,
);
pub(super) type Tabs = TabbedWidget<TabWidgetTypes>;
pub(super) const TAB_NAMES: [&str; 4] =
    ["Dashboard", "Coordinator", "Training Data Server", "Logger"];
type TabsData = <Tabs as CustomWidget>::Data;

struct Backend {
    net_server: TcpServer<ClientToServerMessage, ServerToClientMessage>,
    /// Clients that have connected and sent `Join` but have NOT yet finished
    /// downloading/loading the checkpoint. They are excluded from epoch
    /// admission so slow joiners never disrupt active training.
    pending_clients: HashSet<NodeIdentity>,
    /// Clients that have signalled `ReadyForEpoch` (checkpoint loaded). Only
    /// these are passed to the coordinator for epoch admission.
    ready_clients: HashSet<NodeIdentity>,
}

#[derive(Clone, Copy, Debug)]
struct CheckpointGate {
    epoch: u16,
    step: u32,
    publisher: NodeIdentity,
    published: bool,
}

#[derive(Clone, Copy, Debug)]
struct LossObservation {
    loss: f32,
    tokens_per_sec: f32,
    assigned_sequences: u64,
}

#[derive(Debug)]
struct PendingLossStep {
    tokens_processed: u64,
    expected_sequences: u64,
    unix_timestamp: u64,
    observations: HashMap<NodeIdentity, LossObservation>,
}

fn aggregate_loss_observations(observations: &[LossObservation]) -> Option<(f32, f32, f32, f32)> {
    if observations.is_empty() {
        return None;
    }
    let total_weight: u64 = observations
        .iter()
        .map(|observation| observation.assigned_sequences)
        .sum();
    let loss = if total_weight > 0 {
        observations
            .iter()
            .map(|observation| observation.loss as f64 * observation.assigned_sequences as f64)
            .sum::<f64>()
            / total_weight as f64
    } else {
        observations
            .iter()
            .map(|observation| observation.loss as f64)
            .sum::<f64>()
            / observations.len() as f64
    } as f32;
    let finite_throughputs: Vec<f32> = observations
        .iter()
        .map(|observation| observation.tokens_per_sec)
        .filter(|value| value.is_finite())
        .collect();
    let tokens_per_sec = if finite_throughputs.is_empty() {
        0.0
    } else {
        finite_throughputs.iter().sum::<f32>() / finite_throughputs.len() as f32
    };
    let loss_min = observations
        .iter()
        .map(|observation| observation.loss)
        .fold(f32::INFINITY, f32::min);
    let loss_max = observations
        .iter()
        .map(|observation| observation.loss)
        .fold(f32::NEG_INFINITY, f32::max);
    Some((loss, tokens_per_sec, loss_min, loss_max))
}

fn adamw_join_allowed(owner: Option<NodeIdentity>, identity: NodeIdentity) -> bool {
    owner.is_none_or(|owner| owner == identity)
}

fn initial_admission_open(coordinator: &Coordinator) -> bool {
    coordinator.run_state == RunState::WaitingForMembers
        && coordinator.progress.epoch == 0
        && coordinator.progress.step <= 1
}

fn requires_hosted_checkpoint(coordinator: &Coordinator) -> bool {
    let Model::LLM(llm) = coordinator.model;
    matches!(llm.checkpoint, Checkpoint::Hub(_) | Checkpoint::Gcs(_))
}

fn checkpoint_revision(coordinator: &Coordinator) -> model::CheckpointRevision {
    let Model::LLM(llm) = coordinator.model;
    model::CheckpointRevision {
        epoch: coordinator.progress.epoch,
        checkpoint: llm.checkpoint,
        training_method: llm.training_method,
    }
}

fn checkpoint_destination_matches(coordinator: &Coordinator, checkpoint: Checkpoint) -> bool {
    let Model::LLM(llm) = coordinator.model;
    match (llm.training_method, llm.checkpoint, checkpoint) {
        (model::LLMTrainingMethod::Full, Checkpoint::Hub(current), Checkpoint::Hub(published)) => {
            current.repo_id == published.repo_id && published.revision.is_some()
        }
        (model::LLMTrainingMethod::Full, Checkpoint::Gcs(current), Checkpoint::Gcs(published)) => {
            current == published
        }
        // LoRA publishes adapter-only checkpoints whose destination is not
        // represented by the base-model checkpoint in coordinator state.
        (model::LLMTrainingMethod::Lora(_), _, Checkpoint::Hub(repo)) => repo.revision.is_some(),
        (model::LLMTrainingMethod::Lora(_), _, Checkpoint::Gcs(_)) => true,
        _ => false,
    }
}

fn checkpoint_update_authorized(
    gate: Option<CheckpointGate>,
    run_state: RunState,
    coordinator_epoch: u16,
    from: NodeIdentity,
    update: model::CheckpointUpdate,
) -> bool {
    run_state == RunState::Cooldown
        && gate.is_some_and(|gate| {
            gate.epoch == coordinator_epoch
                && gate.epoch == update.epoch
                && gate.step == update.step
                && gate.publisher == from
                && !gate.published
        })
}

impl Backend {
    pub fn port(&self) -> u16 {
        self.net_server.local_addr().port()
    }
}

struct ChannelCoordinatorBackend {
    rx: Receiver<Coordinator>,
}

impl ChannelCoordinatorBackend {
    fn new() -> (Sender<Coordinator>, Self) {
        let (tx, rx) = channel(10);
        (tx, Self { rx })
    }
}

#[async_trait]
impl aether_watcher::Backend for ChannelCoordinatorBackend {
    async fn wait_for_new_state(&mut self) -> Result<Coordinator> {
        self.rx
            .recv()
            .await
            .ok_or_else(|| anyhow!("coordinator update channel closed"))
    }

    async fn send_witness(&mut self, _opportunistic_data: OpportunisticData) -> Result<()> {
        bail!("Server does not send witnesses");
    }

    async fn send_health_check(&mut self, _health_checks: HealthChecks) -> Result<()> {
        bail!("Server does not send health checks");
    }

    async fn send_checkpoint(&mut self, _checkpoint: model::CheckpointUpdate) -> Result<()> {
        bail!("Server does not send checkpoints");
    }
}

type DataServer = DataProviderTcpServer<DataProvider, ChannelCoordinatorBackend>;

pub struct App {
    cancel: CancellationToken,
    tx_tui_state: Option<Sender<TabsData>>,
    tick_interval: Interval,
    update_tui_interval: Interval,
    coordinator: Coordinator,
    backend: Backend,
    training_data_server: Option<(Sender<Coordinator>, DataServer)>,
    experiment_queue: VecDeque<Coordinator>,
    save_state_dir: Option<PathBuf>,
    coordinator_writer: Option<UnboundedSender<Coordinator>>,
    last_coordinator_hash: u64,
    original_warmup_time: u64,
    withdraw_on_disconnect: bool,
    pause: Option<Arc<Notify>>,
    loss_history: Vec<LossPoint>,
    web_state: Option<std::sync::Arc<std::sync::Mutex<WebState>>>,
    wandb_run: Option<Arc<wandb::Run>>,
    wandb_info: Option<WandbInfo>,
    last_admission_change_unix_timestamp: u64,
    admission_allowlist: Option<HashSet<NodeIdentity>>,
    adamw_owner: Option<NodeIdentity>,
    checkpoint_gate: Option<CheckpointGate>,
    checkpoint_uploader: Option<NodeIdentity>,
    pending_losses: BTreeMap<u32, PendingLossStep>,
}

/// Methods intended for testing purposes only.
///
/// These methods provide access to internal App parameters
/// to facilitate testing and debugging.
#[allow(dead_code)]
impl App {
    pub fn get_clients(&self) -> FixedVec<Client, SOLANA_MAX_NUM_CLIENTS> {
        self.coordinator.epoch_state.clients
    }

    pub fn get_pending_clients(&self) -> HashSet<NodeIdentity> {
        self.backend.pending_clients.clone()
    }

    pub fn get_ready_clients(&self) -> HashSet<NodeIdentity> {
        self.backend.ready_clients.clone()
    }

    /// All connected clients regardless of readiness (syncing + ready).
    pub fn get_all_connected_clients(&self) -> HashSet<NodeIdentity> {
        self.backend
            .pending_clients
            .union(&self.backend.ready_clients)
            .copied()
            .collect()
    }

    pub fn get_run_state(&self) -> RunState {
        self.coordinator.run_state
    }

    pub fn get_rounds(&self) -> [Round; 4] {
        self.coordinator.epoch_state.rounds
    }

    pub fn get_rounds_head(&self) -> u32 {
        self.coordinator.epoch_state.rounds_head
    }

    pub fn get_current_epoch(&self) -> u16 {
        self.coordinator.progress.epoch
    }

    pub fn get_checkpoint(&self) -> Checkpoint {
        match self.coordinator.model {
            Model::LLM(llm) => llm.checkpoint,
        }
    }

    pub fn get_port(&self) -> u16 {
        self.backend.port()
    }

    pub fn get_coordinator(&self) -> Coordinator {
        self.coordinator
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DataServerKind {
    LocalBin,
    Preprocessed,
}

fn default_data_server_kind() -> DataServerKind {
    DataServerKind::LocalBin
}

#[derive(Serialize, Deserialize, Debug)]
pub struct DataServerInfo {
    #[serde(default = "default_data_server_kind")]
    pub kind: DataServerKind,
    pub dir: PathBuf,
    #[serde(default)]
    pub token_size: Option<TokenSize>,
    pub seq_len: usize,
    pub shuffle_seed: [u8; 32],
    #[serde(default)]
    pub split: Option<String>,
    #[serde(default)]
    pub subset: Option<String>,
}

impl DataServerInfo {
    fn provider(self) -> Result<DataProvider> {
        match self.kind {
            DataServerKind::LocalBin => {
                let token_size = self
                    .token_size
                    .ok_or_else(|| anyhow!("local_bin data server config requires `token_size`"))?;
                Ok(DataProvider::Local(LocalDataProvider::new_from_directory(
                    self.dir,
                    token_size,
                    self.seq_len,
                    Shuffle::Seeded(self.shuffle_seed),
                )?))
            }
            DataServerKind::Preprocessed => Ok(DataProvider::Preprocessed(
                PreprocessedDataProvider::new_from_directory(
                    self.dir,
                    self.seq_len,
                    Shuffle::Seeded(self.shuffle_seed),
                    Some(parse_data_split(self.split.as_deref())?),
                    self.subset,
                )?,
            )),
        }
    }
}

fn parse_data_split(split: Option<&str>) -> Result<Split> {
    match split.unwrap_or("train").to_ascii_lowercase().as_str() {
        "train" => Ok(Split::Train),
        "validation" => Ok(Split::Validation),
        "test" => Ok(Split::Test),
        "dev" => Ok(Split::Dev),
        "val" => Ok(Split::Val),
        other => bail!("unsupported data split {other:?}"),
    }
}

impl App {
    #[allow(clippy::too_many_arguments)]
    pub async fn new(
        tui: bool,
        mut coordinator: Coordinator,
        data_server_config: Option<DataServerInfo>,
        experiment_queue: Vec<Coordinator>,
        coordinator_server_port: Option<u16>,
        save_state_dir: Option<PathBuf>,
        events_dir: Option<PathBuf>,
        init_warmup_time: Option<u64>,
        withdraw_on_disconnect: bool,
        web_port: Option<u16>,
        admission_allowlist: Option<HashSet<NodeIdentity>>,
    ) -> Result<Self> {
        async {
            Self::reset_ephemeral(&mut coordinator);

            Self::prepare_model_checkpoint(&coordinator).await?;

            debug!("potentially launching data server...");

            let training_data_server = match &coordinator.model {
                Model::LLM(LLM {
                    data_location,
                    ..
                }) => {
                    if let LLMTrainingDataLocation::Server(url) = data_location {
                        // The data server URL is "host:port". The server only needs the port
                        // (it binds 0.0.0.0); clients resolve the host themselves via DNS, so
                        // accept hostnames as well as IP literals — SocketAddr::from_str
                        // rejects the former, which broke domains like "host.example:39406".
                        let url_str = String::from(url);
                        let data_server_port = url_str
                            .rsplit_once(':')
                            .and_then(|(_, port_str)| port_str.trim().parse::<u16>().ok())
                            .ok_or_else(|| {
                                anyhow!(
                                    "Failed to parse training data server URL {url_str:?}: expected \"host:port\""
                                )
                            })?;
                        let data_provider = data_server_config.ok_or_else(|| anyhow!(
                            "Coordinator state requires we host training data, but no --data-config passed."
                        ))?.provider()?;

                        let (tx, backend) = ChannelCoordinatorBackend::new();
                        info!(
                            advertised = %url_str,
                            bind_port = data_server_port,
                            "starting training data TCP server"
                        );
                        let data_server =
                            DataProviderTcpServer::start(data_provider, backend, data_server_port)
                                .await?;
                        info!(
                            advertised = %url_str,
                            bind_port = data_server_port,
                            "training data TCP server started"
                        );
                        Some((tx, data_server))
                    } else {
                        None
                    }
                }
            };
            debug!("data server work done.");

            let (tabs, pause) = if tui {
                let widgets: TabWidgetTypes = Default::default();
                let pause = widgets.0.pause.clone();
                let tabs = Tabs::new(widgets, &TAB_NAMES);
                (Some(tabs), Some(pause))
            } else {
                (None, None)
            };
            let (cancel, tx_tui_state) =
                maybe_start_render_loop(tabs)?;

            let mut tick_interval = interval(Duration::from_millis(500));
            tick_interval.set_missed_tick_behavior(MissedTickBehavior::Skip); //important!

            let mut update_tui_interval = interval(Duration::from_millis(150));
            update_tui_interval.set_missed_tick_behavior(MissedTickBehavior::Skip);

            let net_server =
                TcpServer::<ClientToServerMessage, ServerToClientMessage>::start(
                    SocketAddr::new(
                        std::net::IpAddr::V4(Ipv4Addr::new(0, 0, 0, 0)),
                        coordinator_server_port.unwrap_or(0),
                    ),
                )
                .await?;

            let original_warmup_time = coordinator.config.warmup_time;

            let web_port = web_port.unwrap_or(8080);
            let web_state = web::start(
                WebState {
                    coordinator: Some(coordinator),
                    loss_history: Vec::new(),
                    syncing_clients: Vec::new(),
                    ready_clients: Vec::new(),
                    server_addr: String::new(),
                    wandb: None,
                },
                web_port,
                cancel.clone(),
            );

            if let Some(init_warmup_time) = init_warmup_time {
                coordinator.config.warmup_time = init_warmup_time;
            }

            let coordinator_writer = if let Some(ref dir) = events_dir {
                let coordinator_dir = dir.join("coordinator");
                std::fs::create_dir_all(&coordinator_dir)?;
                let file_path = coordinator_dir.join("state.bin");
                let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<Coordinator>();
                let record_size = std::mem::size_of::<i64>() + std::mem::size_of::<Coordinator>();
                tokio::spawn(async move {
                    use tokio::io::{AsyncSeekExt, AsyncWriteExt};
                    let mut file = match tokio::fs::OpenOptions::new()
                        .create(true)
                        .truncate(false)
                        .write(true)
                        .open(&file_path)
                        .await
                    {
                        Ok(file) => file,
                        Err(err) => {
                            warn!(
                                path = %file_path.display(),
                                "failed to open coordinator state file: {err}"
                            );
                            return;
                        }
                    };
                    // Truncate any partial record left by a previous crash so
                    // subsequent appends stay aligned to record boundaries.
                    let len = match file.metadata().await {
                        Ok(metadata) => metadata.len(),
                        Err(err) => {
                            warn!(
                                path = %file_path.display(),
                                "failed to read coordinator state file metadata: {err}"
                            );
                            0
                        }
                    };
                    let aligned = len - (len % record_size as u64);
                    if aligned != len {
                        tracing::warn!(
                            "coordinator state.bin has {len} bytes, truncating to {aligned} to discard partial record"
                        );
                        if let Err(err) = file.set_len(aligned).await {
                            warn!(
                                path = %file_path.display(),
                                "failed to truncate coordinator state file: {err}"
                            );
                            return;
                        }
                    }
                    if let Err(err) = file.seek(std::io::SeekFrom::End(0)).await {
                        warn!(
                            path = %file_path.display(),
                            "failed to seek coordinator state file: {err}"
                        );
                        return;
                    }
                    while let Some(coord) = rx.recv().await {
                        let timestamp = chrono::Utc::now().timestamp_millis();
                        let mut buf = Vec::with_capacity(
                            std::mem::size_of::<i64>()
                                + std::mem::size_of::<Coordinator>(),
                        );
                        buf.extend_from_slice(&timestamp.to_le_bytes());
                        buf.extend_from_slice(bytemuck::bytes_of(&coord));
                        if let Err(e) = file.write_all(&buf).await {
                            tracing::warn!("Failed to write coordinator record: {e}");
                            continue;
                        }
                        if let Err(e) = file.flush().await {
                            tracing::warn!("Failed to flush coordinator record: {e}");
                        }
                    }
                });
                Some(tx)
            } else {
                None
            };

            let run_id = String::from(&coordinator.run_id);
            let (wandb_run, wandb_info) = match init_wandb(&run_id).await {
                Some((run, info)) => (Some(run), Some(info)),
                None => (None, None),
            };

            Ok(Self {
                cancel,
                training_data_server,
                experiment_queue: experiment_queue.into(),
                tx_tui_state,
                tick_interval,
                update_tui_interval,
                coordinator,
                backend: Backend {
                    net_server,
                    pending_clients: HashSet::new(),
                    ready_clients: HashSet::new(),
                },
                save_state_dir,
                coordinator_writer,
                last_coordinator_hash: 0,
                original_warmup_time,
                withdraw_on_disconnect,
                pause,
                loss_history: Vec::new(),
                web_state: Some(web_state),
                wandb_run,
                wandb_info,
                last_admission_change_unix_timestamp: 0,
                admission_allowlist,
                adamw_owner: None,
                checkpoint_gate: None,
                checkpoint_uploader: None,
                pending_losses: BTreeMap::new(),
            })
        }.instrument(info_span!("App::new")).await
    }

    pub async fn run(&mut self) -> Result<()> {
        loop {
            if let ControlFlow::Break(()) = self.poll_next().await? {
                break;
            }
        }
        Ok(())
    }

    pub async fn poll_next(&mut self) -> Result<ControlFlow<(), ()>> {
        select! {
            _ = self.cancel.cancelled() => {
                info!("got cancel callback, exiting cleanly.");
                return Ok(ControlFlow::Break(()));
            }

            Some(event) = self.backend.net_server.next() => {
                match event {
                    ClientNotification::Message((from, message)) => {
                        self.on_client_message(from, message).await;
                    }
                    ClientNotification::Disconnected(from) => {
                        self.on_disconnect(from)?;
                        self.post_state_change(true).await;
                    }
                }
            }
            _ = self.tick_interval.tick() => {
                self.on_tick().await;
            }
            _ = self.update_tui_interval.tick() => {
                self.update_tui().await?;
            }
            _ = async {
                if let Some((_, server))  = &mut self.training_data_server {
                    server.poll().await
                } else {
                    tokio::task::yield_now().await;
                }
            } => {}
            _ = async { self.pause.as_ref().unwrap().notified().await }, if self.pause.is_some() => {
                self.pause();
            }
        }
        Ok(ControlFlow::Continue(()))
    }

    async fn update_tui(&mut self) -> Result<()> {
        if let Some(tx_tui_state) = &self.tx_tui_state {
            let states = (
                (&*self).into(),
                (&self.coordinator).into(),
                self.training_data_server.as_ref().map(|o| (&o.1).into()),
                Default::default(),
            );
            tx_tui_state.send(states).await?;
        }
        self.update_web_state();
        Ok(())
    }

    fn update_web_state(&mut self) {
        if let Some(ref shared) = self.web_state {
            if let Ok(mut state) = shared.lock() {
                state.coordinator = Some(self.coordinator);
                state.loss_history.clone_from(&self.loss_history);
                state.syncing_clients = self
                    .backend
                    .pending_clients
                    .iter()
                    .map(|c| c.to_string())
                    .collect();
                state.ready_clients = self
                    .backend
                    .ready_clients
                    .iter()
                    .map(|c| c.to_string())
                    .collect();
                state.server_addr = self.backend.net_server.local_addr().to_string();
                state.wandb.clone_from(&self.wandb_info);
            }
        }
    }

    fn log_to_wandb(&self, point: &LossPoint) {
        let Some(run) = self.wandb_run.clone() else {
            return;
        };
        let mut log = LogData::new();
        log.insert("_step", point.step);
        log.insert("train/loss", point.loss);
        log.insert("train/perplexity", point.loss.exp());
        log.insert(
            "train/lr",
            match &self.coordinator.model {
                Model::LLM(llm) => llm.lr_schedule.get_lr(point.step),
            },
        );
        log.insert("train/tokens_per_sec", point.tokens_per_sec);
        log.insert("train/total_tokens", point.tokens_processed as f64);
        log.insert("train/witness_count", point.witness_count as f64);
        log.insert("train/loss_min", point.loss_min);
        log.insert("train/loss_max", point.loss_max);
        log.insert(
            "train/global_token_batch_size",
            (self
                .coordinator
                .get_target_global_batch_size(self.coordinator.current_round()) as u32
                * self.coordinator.get_sequence_length()) as f64,
        );
        tokio::spawn(async move {
            run.log(log).await;
        });
    }

    fn push_loss_point(&mut self, point: LossPoint) {
        if self.loss_history.len() >= MAX_LOSS_HISTORY {
            let mid = self.loss_history.len() / 2;
            let mut downsampled: Vec<LossPoint> = self.loss_history[..mid]
                .iter()
                .step_by(2)
                .cloned()
                .collect();
            downsampled.extend_from_slice(&self.loss_history[mid..]);
            self.loss_history = downsampled;
        }
        self.loss_history.push(point);
    }

    fn admission_allowed(&self, identity: &NodeIdentity) -> bool {
        admission_allowed(self.admission_allowlist.as_ref(), identity)
    }

    fn uses_adamw(&self) -> bool {
        let Model::LLM(llm) = self.coordinator.model;
        matches!(llm.optimizer, OptimizerDefinition::AdamW { .. })
    }

    fn initial_admission_open(&self) -> bool {
        initial_admission_open(&self.coordinator)
    }

    fn elected_checkpoint_publisher(&self) -> Option<NodeIdentity> {
        self.checkpoint_uploader.filter(|identity| {
            self.coordinator
                .epoch_state
                .clients
                .iter()
                .any(|client| client.id == *identity && client.state == ClientState::Healthy)
        })
    }

    fn sync_checkpoint_gate(&mut self) {
        if self.coordinator.run_state != RunState::Cooldown
            || !requires_hosted_checkpoint(&self.coordinator)
        {
            self.checkpoint_gate = None;
            return;
        }
        let epoch = self.coordinator.progress.epoch;
        if self.checkpoint_gate.is_some_and(|gate| gate.epoch == epoch) {
            return;
        }
        self.checkpoint_gate =
            self.elected_checkpoint_publisher()
                .map(|publisher| CheckpointGate {
                    epoch,
                    step: self.coordinator.progress.step.saturating_sub(1),
                    publisher,
                    published: false,
                });
        if let Some(gate) = self.checkpoint_gate {
            info!(epoch, publisher = %gate.publisher, "waiting for authoritative checkpoint publication");
        } else {
            warn!(
                epoch,
                "cooldown has no healthy checkpoint publisher; progression is blocked"
            );
        }
    }

    fn record_loss_observation(
        &mut self,
        identity: NodeIdentity,
        step: u32,
        loss: f32,
        tokens_per_sec: f32,
    ) {
        let assigned_sequences = CommitteeSelection::from_coordinator(&self.coordinator, 0)
            .ok()
            .map(|selection| assign_data_for_state(&self.coordinator, &selection))
            .map(|assignments| {
                assignments
                    .iter()
                    .filter(|(_, owner)| **owner == identity)
                    .map(|(batch, _)| batch.0.end - batch.0.start + 1)
                    .sum()
            })
            .unwrap_or(0);
        let tokens_processed = self
            .coordinator
            .total_tokens_processed(self.coordinator.current_round());
        let expected_sequences =
            self.coordinator
                .get_target_global_batch_size(self.coordinator.current_round()) as u64;
        self.pending_losses
            .entry(step)
            .or_insert_with(|| PendingLossStep {
                tokens_processed,
                expected_sequences,
                unix_timestamp: Self::get_timestamp(),
                observations: HashMap::new(),
            })
            .observations
            .entry(identity)
            .or_insert(LossObservation {
                loss,
                tokens_per_sec,
                assigned_sequences,
            });
    }

    async fn verify_checkpoint_artifact(&self, update: model::CheckpointUpdate) -> bool {
        match update.checkpoint {
            Checkpoint::Hub(repo) => {
                let Some(revision) = repo.revision else {
                    return false;
                };
                let repo_id: String = (&repo.repo_id).into();
                let revision: String = (&revision).into();
                let path = match download_model_file_async(
                    &repo_id,
                    &revision,
                    "aether_checkpoint.json",
                    None,
                )
                .await
                {
                    Ok(path) => path,
                    Err(error) => {
                        warn!(%error, %repo_id, %revision, "failed to fetch checkpoint metadata");
                        return false;
                    }
                };
                let metadata = match tokio::fs::read(&path).await.ok().and_then(|bytes| {
                    serde_json::from_slice::<model::CheckpointMetadata>(&bytes).ok()
                }) {
                    Some(metadata) => metadata,
                    None => {
                        warn!(path = %path.display(), "invalid checkpoint metadata");
                        return false;
                    }
                };
                metadata.run_id == String::from(&self.coordinator.run_id)
                    && metadata.epoch == update.epoch
                    && metadata.step == update.step
            }
            // Do not release a gated epoch without backend-verified metadata.
            // The current production run uses Hub; GCS must add equivalent
            // manifest verification before gated training is enabled.
            Checkpoint::Gcs(_) => false,
            _ => false,
        }
    }

    fn finalize_completed_losses(&mut self) {
        let completed_steps: Vec<u32> = self
            .pending_losses
            .range(..self.coordinator.progress.step)
            .map(|(step, _)| *step)
            .collect();
        for step in completed_steps {
            let Some(pending) = self.pending_losses.remove(&step) else {
                continue;
            };
            if pending.observations.is_empty() {
                continue;
            }
            let observations: Vec<_> = pending.observations.values().collect();
            let owned_observations: Vec<_> = observations.into_iter().copied().collect();
            let observed_sequences: u64 = owned_observations
                .iter()
                .map(|observation| observation.assigned_sequences)
                .sum();
            if observed_sequences != pending.expected_sequences {
                warn!(
                    step,
                    observed_sequences,
                    expected_sequences = pending.expected_sequences,
                    "not publishing partial global loss"
                );
                continue;
            }
            let Some((loss, tokens_per_sec, loss_min, loss_max)) =
                aggregate_loss_observations(&owned_observations)
            else {
                continue;
            };
            let point = LossPoint {
                step,
                tokens_processed: pending.tokens_processed,
                loss,
                tokens_per_sec,
                unix_timestamp: pending.unix_timestamp,
                witness_count: owned_observations.len(),
                loss_min,
                loss_max,
            };
            self.log_to_wandb(&point);
            self.push_loss_point(point);
        }
    }

    fn has_joined(&self, identity: &NodeIdentity) -> bool {
        self.backend.pending_clients.contains(identity)
            || self.backend.ready_clients.contains(identity)
    }

    async fn reject_client(&mut self, to: PublicKey, code: ServerErrorCode, message: String) {
        if let Err(err) = self
            .backend
            .net_server
            .send_to(to, ServerToClientMessage::Error { code, message })
            .await
        {
            warn!(client = %to, "failed to send rejection: {err}");
        }
    }

    fn on_disconnect(&mut self, from: PublicKey) -> Result<()> {
        let from_identity = NodeIdentity::from_single_key(*from.as_bytes());
        let removed_pending = self.backend.pending_clients.remove(&from_identity);
        let removed_ready = self.backend.ready_clients.remove(&from_identity);
        if removed_pending || removed_ready {
            self.last_admission_change_unix_timestamp = Self::get_timestamp();
        }
        if self
            .checkpoint_gate
            .is_some_and(|gate| gate.publisher == from_identity && !gate.published)
        {
            tracing::error!(
                client = %from,
                "checkpoint publisher disconnected; run remains fail-closed in cooldown and must be restarted"
            );
        }

        if removed_pending
            && self.initial_admission_open()
            && self.adamw_owner == Some(from_identity)
        {
            self.adamw_owner = None;
        }
        if removed_pending
            && self.initial_admission_open()
            && self.checkpoint_uploader == Some(from_identity)
        {
            self.checkpoint_uploader = None;
        }

        if self.withdraw_on_disconnect || self.coordinator.active() {
            if let Some(index) = self.find_client_index(&from_identity) {
                match self.coordinator.withdraw(index as u64) {
                    Ok(_) => info!("Withdrew {from}"),
                    Err(err) => warn!("Coordinator withdraw error: {err}"),
                }
            }
        }

        Ok(())
    }

    async fn on_client_message(&mut self, from: PublicKey, event: ClientToServerMessage) {
        let from_identity = NodeIdentity::from_single_key(*from.as_bytes());
        if !matches!(&event, ClientToServerMessage::Join { .. }) && !self.has_joined(&from_identity)
        {
            warn!(client = %from, "ignoring message from client that has not joined");
            self.reject_client(
                from,
                ServerErrorCode::JoinRequired,
                "a successful Join is required before sending this message".to_string(),
            )
            .await;
            return;
        }

        let broadcast = match event {
            ClientToServerMessage::Join {
                run_id,
                checkpoint_upload,
            } => {
                let coord_run_id = String::from(&self.coordinator.run_id);
                if coord_run_id != run_id {
                    info!("{from:?} tried to join unknown run {run_id}");
                    self.reject_client(
                        from,
                        ServerErrorCode::RunIdMismatch,
                        format!("run id {run_id:?} does not match the active run"),
                    )
                    .await;
                } else if !self.admission_allowed(&from_identity) {
                    warn!("{from:?} is not in the admission allowlist for run {run_id}");
                    self.reject_client(
                        from,
                        ServerErrorCode::NotAllowlisted,
                        "client identity is not in the admission allowlist".to_string(),
                    )
                    .await;
                } else if self.uses_adamw()
                    && !self.has_joined(&from_identity)
                    && !self.initial_admission_open()
                {
                    warn!(client = %from, "rejecting late join for single-client AdamW run");
                    self.reject_client(
                        from,
                        ServerErrorCode::LateJoinUnsupported,
                        "AdamW optimizer state cannot be synchronized to a late client".to_string(),
                    )
                    .await;
                } else if checkpoint_upload
                    && self
                        .checkpoint_uploader
                        .is_some_and(|publisher| publisher != from_identity)
                {
                    self.reject_client(
                        from,
                        ServerErrorCode::CheckpointPublisherAlreadyAssigned,
                        "this run already has a checkpoint publisher; join as a normal volunteer without checkpoint upload flags".to_string(),
                    )
                    .await;
                } else if self.uses_adamw() && !adamw_join_allowed(self.adamw_owner, from_identity)
                {
                    self.reject_client(
                        from,
                        ServerErrorCode::SingleClientOptimizer,
                        "AdamW supports exactly one network client; use DisTrO or Muon for multi-client training".to_string(),
                    )
                    .await;
                } else {
                    if self.uses_adamw() {
                        self.adamw_owner.get_or_insert(from_identity);
                    }
                    if checkpoint_upload {
                        self.checkpoint_uploader.get_or_insert(from_identity);
                    }
                    info!("added pending client {from}");
                    if !self.backend.ready_clients.contains(&from_identity)
                        && self.backend.pending_clients.insert(from_identity)
                    {
                        self.last_admission_change_unix_timestamp = Self::get_timestamp();
                    }
                }
                false
            }
            ClientToServerMessage::ReadyForEpoch(revision) => {
                // The client has finished downloading/loading the checkpoint.
                // Admit it only while waiting and only for the exact model
                // revision that is currently authoritative.
                let mut changed = false;
                if self.backend.ready_clients.contains(&from_identity) {
                    // Idempotent duplicate from an already-ready client.
                } else if self.coordinator.run_state != RunState::WaitingForMembers
                    || revision != checkpoint_revision(&self.coordinator)
                {
                    self.backend.pending_clients.remove(&from_identity);
                    self.reject_client(
                        from,
                        ServerErrorCode::StaleCheckpoint,
                        "loaded checkpoint is no longer the authoritative revision for admission"
                            .to_string(),
                    )
                    .await;
                } else if self.backend.pending_clients.remove(&from_identity) {
                    info!("client {from} is ready for epoch admission");
                    self.backend.ready_clients.insert(from_identity);
                    self.last_admission_change_unix_timestamp = Self::get_timestamp();
                    changed = true;
                }
                changed
            }
            ClientToServerMessage::Witness(witness) => {
                let state_before = self.coordinator.run_state;
                if let Err(error) = match *witness {
                    OpportunisticData::WitnessStep(witness, witness_metadata) => {
                        let result = self.coordinator.witness(
                            &from_identity,
                            witness_metadata.step,
                            witness,
                            Self::get_timestamp(),
                        );
                        if result.is_ok() && witness_metadata.loss.is_finite() {
                            self.record_loss_observation(
                                from_identity,
                                witness_metadata.step,
                                witness_metadata.loss,
                                witness_metadata.tokens_per_sec,
                            );
                        }
                        result
                    }
                    OpportunisticData::WarmupStep(witness) => self.coordinator.warmup_witness(
                        &from_identity,
                        witness,
                        Self::get_timestamp(),
                        rand::rng().next_u64(),
                    ),
                } {
                    warn!("Error when processing witness: {error}");
                };
                self.coordinator.run_state != state_before
            }
            ClientToServerMessage::HealthCheck(health_checks) => {
                match self.coordinator.health_check(&from_identity, health_checks) {
                    Ok(dropped) => {
                        info!("Dropped {} clients from health check", dropped);
                        dropped > 0
                    }

                    Err(error) => {
                        warn!("Error when processing health check: {error}");
                        false
                    }
                }
            }
            ClientToServerMessage::Checkpoint(update) => {
                let authorized = checkpoint_update_authorized(
                    self.checkpoint_gate,
                    self.coordinator.run_state,
                    self.coordinator.progress.epoch,
                    from_identity,
                    update,
                );
                if !authorized {
                    warn!(client = %from, "ignoring checkpoint from non-publisher or outside cooldown");
                    false
                } else {
                    let checkpoint = update.checkpoint;
                    if !checkpoint_destination_matches(&self.coordinator, checkpoint) {
                        warn!(client = %from, "rejecting checkpoint with an unexpected destination or missing immutable revision");
                        false
                    } else if !self.verify_checkpoint_artifact(update).await {
                        warn!(client = %from, "rejecting checkpoint whose committed metadata does not match this epoch and step");
                        false
                    } else {
                        match self.find_client_index(&from_identity) {
                            Some(index) => {
                                match self.coordinator.checkpoint(
                                    &from_identity,
                                    index as u64,
                                    checkpoint,
                                ) {
                                    Ok(changed) => {
                                        if matches!(checkpoint, Checkpoint::Hub(_)) && !changed {
                                            warn!(client = %from, "rejecting stale Hub checkpoint revision");
                                            return self.post_state_change(false).await;
                                        }
                                        if let Some(gate) = &mut self.checkpoint_gate {
                                            gate.published = true;
                                        }
                                        info!(client = %from, epoch = self.coordinator.progress.epoch, "authoritative checkpoint published");
                                        changed
                                    }
                                    Err(error) => {
                                        warn!("Error when processing checkpoint: {error}");
                                        false
                                    }
                                }
                            }
                            None => {
                                warn!("Got checkpoint but could not find {from} in client list");
                                false
                            }
                        }
                    }
                }
            }
        };
        self.post_state_change(broadcast).await;
    }

    async fn on_tick(&mut self) {
        self.kick_unhealthy_clients();
        if self.coordinator.run_state == RunState::Finished {
            if let Err(err) = self.try_start_next_experiment_run().await {
                warn!("experiment transition failed: {err:#}");
            }
        }
        self.sync_checkpoint_gate();
        if self.coordinator.run_state == RunState::Cooldown
            && !self.checkpoint_gate.is_some_and(|gate| gate.published)
        {
            self.post_state_change(false).await;
            return;
        }
        // Initial clients are admitted only after loading the same base
        // checkpoint. New identities are rejected after training starts until
        // an exact model-revision synchronization protocol exists.
        let checkpoint_publisher_ready = !requires_hosted_checkpoint(&self.coordinator)
            || self
                .checkpoint_uploader
                .is_some_and(|publisher| self.backend.ready_clients.contains(&publisher));
        let admission_iter: Vec<&NodeIdentity> = if checkpoint_publisher_ready {
            self.backend.ready_clients.iter().collect()
        } else {
            debug!("waiting for the checkpoint publisher to become ready");
            Vec::new()
        };
        let admission_count = admission_iter.len();

        let now = Self::get_timestamp();
        let admission_ready_at = self
            .coordinator
            .run_state_start_unix_timestamp
            .max(self.last_admission_change_unix_timestamp)
            .saturating_add(self.coordinator.config.waiting_for_members_extra_time as u64);
        let should_wait_for_more_clients = self.coordinator.run_state
            == RunState::WaitingForMembers
            && admission_count as u16 >= self.coordinator.config.init_min_clients
            && now < admission_ready_at;

        let (admission_iter, admission_count) = if should_wait_for_more_clients {
            debug!(
                ready_at = admission_ready_at,
                admission_count, "waiting for client admission quiet period"
            );
            (Vec::new(), 0)
        } else {
            (admission_iter, admission_count)
        };

        match self.coordinator.tick(
            Some(SizedIterator::new(
                admission_iter.into_iter(),
                admission_count,
            )),
            now,
            rand::rng().next_u64(),
        ) {
            Ok(TickResult::EpochEnd(result)) => {
                if result {
                    if let Some(save_state_dir) = &self.save_state_dir {
                        let mut state = self.coordinator;
                        Self::reset_ephemeral(&mut state);
                        match toml::to_string_pretty(&state) {
                            Ok(toml) => {
                                let filename = format!(
                                    "{:?}-step{}.toml",
                                    self.coordinator.run_id,
                                    self.coordinator.progress.step - 1
                                );
                                info!("Saving state to {filename}");
                                if let Err(err) =
                                    std::fs::write(save_state_dir.join(filename), toml)
                                {
                                    tracing::error!("Error saving TOML: {err:#}");
                                }
                            }
                            Err(err) => tracing::error!("Error serialized to TOML: {err:#}"),
                        }
                    }
                } else {
                    warn!("Epoch abandoned")
                }
            }
            Ok(TickResult::Ticked) | Err(CoordinatorError::Halted) => {}
            Err(err) => warn!("Coordinator tick error: {err}"),
        }
        self.finalize_completed_losses();
        self.post_state_change(true).await;
    }

    fn find_client_index(&self, identity: &NodeIdentity) -> Option<usize> {
        self.coordinator
            .epoch_state
            .clients
            .iter()
            .position(|x| &x.id == identity)
    }

    fn get_timestamp() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0)
    }

    async fn post_state_change(&mut self, broadcast: bool) {
        self.sync_checkpoint_gate();
        if self.coordinator.active() {
            // reset to original values if we changed them to something special for init
            self.coordinator.config.warmup_time = self.original_warmup_time;
        }
        if broadcast {
            if let Err(err) = self
                .backend
                .net_server
                .broadcast(ServerToClientMessage::Coordinator(Box::new(
                    self.coordinator,
                )))
                .await
            {
                warn!("Error in on_tick: {err}");
            }
            if let Some((ref sender, _)) = &self.training_data_server {
                if let Err(err) = sender.send(self.coordinator).await {
                    warn!("Error sending coordinator state to training data server: {err}");
                }
            }
        }
        if let Some(ref writer) = self.coordinator_writer {
            let mut hasher = DefaultHasher::new();
            hasher.write(bytemuck::bytes_of(&self.coordinator));
            let hash = hasher.finish();
            if hash != self.last_coordinator_hash {
                self.last_coordinator_hash = hash;
                let _ = writer.send(self.coordinator);
            }
        }
    }

    fn reset_ephemeral(coordinator: &mut Coordinator) {
        coordinator.run_state = RunState::WaitingForMembers;
        for elem in coordinator.epoch_state.clients.iter_mut() {
            *elem = Client::default();
        }
        for elem in coordinator.epoch_state.exited_clients.iter_mut() {
            *elem = Client::default();
        }
    }

    async fn prepare_model_checkpoint(coordinator: &Coordinator) -> Result<()> {
        let Model::LLM(LLM { checkpoint, .. }) = coordinator.model;
        match checkpoint {
            Checkpoint::Hub(hub_repo) => {
                let repo_id = String::from(&hub_repo.repo_id);
                let revision = hub_repo.revision.map(|bytes| (&bytes).into());
                if revision.is_some()
                    || !tokio::fs::try_exists(PathBuf::from(repo_id.clone()))
                        .await
                        .unwrap_or_default()
                {
                    download_model_repo_async(&repo_id, revision, None, None, None, true).await?;
                }
            }
            Checkpoint::Ephemeral => bail!("Can't start up a run with an Ephemeral checkpoint."),
            Checkpoint::Dummy(_) => {}
            Checkpoint::P2P(_) | Checkpoint::P2PGcs(_) => {
                bail!("Can't start up a run with a P2P checkpoint.")
            }
            Checkpoint::Gcs(gcs_repo) => {
                let bucket: String = (&gcs_repo.bucket).into();
                let prefix: Option<String> = gcs_repo.prefix.map(|p| (&p).into());
                download_model_from_gcs_async(&bucket, prefix.as_deref()).await?;
            }
        }
        Ok(())
    }

    async fn try_start_next_experiment_run(&mut self) -> Result<()> {
        if self.experiment_queue.is_empty() {
            return Ok(());
        }

        let mut next = self
            .experiment_queue
            .pop_front()
            .expect("checked non-empty above");
        Self::reset_ephemeral(&mut next);
        Self::prepare_model_checkpoint(&next).await?;

        let run_id = String::from(&next.run_id);
        info!(run_id, "starting next experiment run");
        self.coordinator = next;
        self.original_warmup_time = self.coordinator.config.warmup_time;
        self.loss_history.clear();
        self.pending_losses.clear();
        self.checkpoint_gate = None;
        self.checkpoint_uploader = None;
        self.adamw_owner = None;
        self.last_admission_change_unix_timestamp = Self::get_timestamp();
        self.backend.pending_clients.clear();
        self.backend.ready_clients.clear();
        let (wandb_run, wandb_info) = match init_wandb(&run_id).await {
            Some((run, info)) => (Some(run), Some(info)),
            None => (None, None),
        };
        self.wandb_run = wandb_run;
        self.wandb_info = wandb_info;
        self.post_state_change(true).await;
        Ok(())
    }

    fn kick_unhealthy_clients(&mut self) {
        for client in self.coordinator.epoch_state.exited_clients {
            let removed_pending = self.backend.pending_clients.remove(&client.id);
            let removed_ready = self.backend.ready_clients.remove(&client.id);
            if removed_pending || removed_ready {
                info!(
                    client = %client.id,
                    state = %client.state,
                    "removed exited client from admission queue"
                );
                self.last_admission_change_unix_timestamp = Self::get_timestamp();
            }
        }
    }

    fn pause(&mut self) {
        if let Err(err) = match self.coordinator.run_state {
            RunState::Paused => self.coordinator.resume(Self::get_timestamp()),
            _ => self.coordinator.pause(Self::get_timestamp()),
        } {
            warn!("Error pausing: {}", err);
        }
    }
}

fn admission_allowed(
    admission_allowlist: Option<&HashSet<NodeIdentity>>,
    identity: &NodeIdentity,
) -> bool {
    admission_allowlist.is_none_or(|allowlist| allowlist.contains(identity))
}

impl From<&App> for DashboardState {
    fn from(app: &App) -> Self {
        Self {
            coordinator_state: (&app.coordinator).into(),
            server_addr: app.backend.net_server.local_addr().to_string(),
            nodes_next_epoch: app
                .backend
                .ready_clients
                .iter()
                .map(|c| c.to_string())
                .collect(),
        }
    }
}

/// Creates a wandb run for server-side metric logging, driven entirely by
/// environment variables. Returns `None` (and the server continues normally)
/// if `WANDB_API_KEY` is unset or the wandb backend is unreachable.
///
/// - `WANDB_API_KEY`  (required to enable)
/// - `WANDB_PROJECT`  (default: `aether`)
/// - `WANDB_RUN`      (default: `server-<run_id>-<UTC timestamp>`)
/// - `WANDB_ENTITY`   (optional)
/// - `WANDB_GROUP`    (optional)
async fn init_wandb(run_id: &str) -> Option<(Arc<wandb::Run>, WandbInfo)> {
    let api_key = std::env::var("WANDB_API_KEY").ok()?;
    let project = std::env::var("WANDB_PROJECT").unwrap_or_else(|_| "aethercompute".to_string());
    let run_name = std::env::var("WANDB_RUN").unwrap_or_else(|_| {
        format!(
            "server-{run_id}-{}",
            chrono::Utc::now().format("%Y-%m-%d_%H-%M-%S")
        )
    });
    let entity = std::env::var("WANDB_ENTITY").ok();
    let group = std::env::var("WANDB_GROUP").ok();
    let info = WandbInfo {
        project: project.clone(),
        run_name: run_name.clone(),
        entity: entity.clone(),
        group: group.clone(),
    };

    let wandb = wandb::WandB::new(wandb::BackendOptions::new(api_key));
    let mut run_info = wandb::RunInfo::new(project).name(run_name).config((
        ("run_id", run_id.to_string()),
        ("source", "server".to_string()),
    ));
    if let Some(entity) = entity {
        run_info = run_info.entity(entity);
    }
    if let Some(group) = group {
        run_info = run_info.group(group);
    }
    match run_info.build() {
        Ok(built) => match wandb.new_run(built).await {
            Ok(run) => {
                info!("Connected to wandb; logging server-side metrics.");
                Some((Arc::new(run), info))
            }
            Err(e) => {
                warn!("Could not connect to wandb ({e:?}); continuing without it.");
                None
            }
        },
        Err(e) => {
            warn!("wandb run build failed ({e:?}); continuing without it.");
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        adamw_join_allowed, admission_allowed, aggregate_loss_observations,
        checkpoint_update_authorized, initial_admission_open, ChannelCoordinatorBackend,
        CheckpointGate, LossObservation,
    };
    use aether_coordinator::{Coordinator, RunState};
    use aether_core::NodeIdentity;
    use aether_watcher::Backend;
    use bytemuck::Zeroable;
    use std::collections::HashSet;
    use tokio::sync::mpsc;

    #[test]
    fn admission_allowed_is_open_without_allowlist() {
        let identity = NodeIdentity::from_single_key([1; 32]);

        assert!(admission_allowed(None, &identity));
    }

    #[test]
    fn admission_allowed_requires_membership_when_allowlist_is_present() {
        let allowed = NodeIdentity::from_single_key([1; 32]);
        let denied = NodeIdentity::from_single_key([2; 32]);
        let allowlist = HashSet::from([allowed]);

        assert!(admission_allowed(Some(&allowlist), &allowed));
        assert!(!admission_allowed(Some(&allowlist), &denied));
    }

    #[test]
    fn adamw_allows_only_the_owner_identity() {
        let owner = NodeIdentity::from_single_key([1; 32]);
        let other = NodeIdentity::from_single_key([2; 32]);

        assert!(adamw_join_allowed(None, owner));
        assert!(adamw_join_allowed(Some(owner), owner));
        assert!(!adamw_join_allowed(Some(owner), other));
    }

    #[test]
    fn late_admission_closes_as_soon_as_training_starts() {
        let mut coordinator = Coordinator::zeroed();
        coordinator.run_state = RunState::WaitingForMembers;
        coordinator.progress.step = 1;
        assert!(initial_admission_open(&coordinator));

        coordinator.run_state = RunState::Warmup;
        assert!(!initial_admission_open(&coordinator));
        coordinator.run_state = RunState::WaitingForMembers;
        coordinator.progress.step = 2;
        assert!(!initial_admission_open(&coordinator));
    }

    #[test]
    fn checkpoint_publication_requires_exact_epoch_step_and_publisher() {
        let publisher = NodeIdentity::from_single_key([1; 32]);
        let other = NodeIdentity::from_single_key([2; 32]);
        let gate = CheckpointGate {
            epoch: 3,
            step: 99,
            publisher,
            published: false,
        };
        let update = aether_coordinator::model::CheckpointUpdate {
            epoch: 3,
            step: 99,
            checkpoint: aether_coordinator::model::Checkpoint::Ephemeral,
        };

        assert!(checkpoint_update_authorized(
            Some(gate),
            RunState::Cooldown,
            3,
            publisher,
            update,
        ));
        assert!(!checkpoint_update_authorized(
            Some(gate),
            RunState::Cooldown,
            3,
            other,
            update,
        ));
        assert!(!checkpoint_update_authorized(
            Some(gate),
            RunState::Cooldown,
            3,
            publisher,
            aether_coordinator::model::CheckpointUpdate { step: 98, ..update },
        ));
        assert!(!checkpoint_update_authorized(
            Some(CheckpointGate {
                published: true,
                ..gate
            }),
            RunState::Cooldown,
            3,
            publisher,
            update,
        ));
    }

    #[test]
    fn losses_are_weighted_by_assigned_sequences() {
        let observations = [
            LossObservation {
                loss: 1.0,
                tokens_per_sec: 100.0,
                assigned_sequences: 2,
            },
            LossObservation {
                loss: 4.0,
                tokens_per_sec: 200.0,
                assigned_sequences: 1,
            },
        ];
        let (loss, throughput, min, max) = aggregate_loss_observations(&observations).unwrap();

        assert!((loss - 2.0).abs() < f32::EPSILON);
        assert!((throughput - 150.0).abs() < f32::EPSILON);
        assert_eq!(min, 1.0);
        assert_eq!(max, 4.0);
    }

    #[test]
    fn losses_fall_back_to_arithmetic_mean_without_assignments() {
        let observations = [
            LossObservation {
                loss: 1.0,
                tokens_per_sec: 100.0,
                assigned_sequences: 0,
            },
            LossObservation {
                loss: 3.0,
                tokens_per_sec: f32::NAN,
                assigned_sequences: 0,
            },
        ];
        let (loss, throughput, _, _) = aggregate_loss_observations(&observations).unwrap();

        assert!((loss - 2.0).abs() < f32::EPSILON);
        assert!((throughput - 100.0).abs() < f32::EPSILON);
    }

    #[tokio::test]
    async fn wait_for_new_state_errors_when_channel_closed() {
        let (tx, rx) = mpsc::channel(1);
        drop(tx);

        let mut backend = ChannelCoordinatorBackend { rx };
        let err = backend
            .wait_for_new_state()
            .await
            .expect_err("closed channel should return an error");

        assert!(err
            .to_string()
            .contains("coordinator update channel closed"));
    }
}
