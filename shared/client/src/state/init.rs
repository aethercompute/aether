use crate::{fetch_data::DataFetcher, WandBInfo};
use aether_coordinator::{
    model::{self, HttpLLMTrainingDataLocation, LLMTrainingDataLocation},
    Coordinator, HealthChecks,
};
use aether_core::{
    Barrier, CancellableBarrier, IntegrationTestLogMarker, NodeIdentity, Shuffle, TokenSize,
};
use aether_data_provider::{
    download_dataset_repo_async, download_model_from_gcs_async, download_model_repo_async,
    http::{FileURLs, HttpDataProvider},
    DataProvider, DataProviderTcpClient, DownloadError, DummyDataProvider, LocalDataProvider,
    PreprocessedDataProvider, Split, WeightedDataProvider,
};
use aether_event_sourcing::event;
use aether_metrics::ClientMetrics;
use aether_modeling::{
    auto_tokenizer, AttentionImplementation, AutoConfig, AutoTokenizerError, CausalLM,
    CommunicatorId, DataParallel, DeepseekForCausalLM, Devices, DummyModel, LlamaConfig,
    LlamaForCausalLM, LocalTrainer, ModelLoadError, ParallelModels, PretrainedSource, Trainer,
};
use aether_network::{BlobTicket, SecretKey};
use aether_watcher::OpportunisticData;
use std::{collections::HashMap, path::PathBuf, sync::Arc};
use tch::{Kind, Tensor};
use thiserror::Error;
use tokenizers::{models::wordlevel::WordLevel, ModelWrapper, Tokenizer};
use tokio::{
    io,
    sync::{mpsc::UnboundedSender, oneshot},
    task::{JoinError, JoinHandle},
};
use tracing::{debug, error, info};

use super::{
    cooldown::CooldownStepMetadata, evals::ModelTaskRunner, stats::StatsLogger,
    steps::StepStateMachine, train::TrainingStepMetadata, types::DistroBroadcastAndPayload,
    warmup::WarmupStepMetadata, witness::WitnessStepMetadata, CheckpointConfig, FinishedBroadcast,
};
use iroh_blobs::api::Tag;

pub struct RunInitConfig {
    // identity for connecting to the data server
    pub identity: NodeIdentity,
    pub p2p_secret_key: SecretKey,

    // p2p model parameters sharing config
    pub max_concurrent_parameter_requests: usize,

    // model & dataload
    pub device: Devices,
    pub hub_read_token: Option<String>,
    pub hub_max_concurrent_downloads: usize,
    pub data_parallelism: usize,
    pub tensor_parallelism: usize,
    pub micro_batch_size: usize,
    pub optim_stats_every_n_steps: Option<u32>,
    pub grad_accum_in_fp32: bool,

    // evaluation
    pub eval_task_max_docs: Option<usize>,
    pub eval_tasks: Vec<aether_eval::Task>,
    pub prompt_task: bool,

    // logging
    pub wandb_info: Option<WandBInfo>,

    // debugging
    pub write_gradients_dir: Option<PathBuf>,

    // checkpointing
    pub checkpoint_config: Option<CheckpointConfig>,

    // configurable dummy training time (in seconds) for this client - relevant just for testing
    pub dummy_training_delay_secs: Option<u64>,

    pub sidecar_port: Option<u16>,
}

impl Clone for RunInitConfig {
    fn clone(&self) -> Self {
        Self {
            identity: self.identity,
            p2p_secret_key: self.p2p_secret_key.clone(),
            max_concurrent_parameter_requests: self.max_concurrent_parameter_requests,
            device: self.device.clone(),
            hub_read_token: self.hub_read_token.clone(),
            hub_max_concurrent_downloads: self.hub_max_concurrent_downloads,
            data_parallelism: self.data_parallelism,
            tensor_parallelism: self.tensor_parallelism,
            micro_batch_size: self.micro_batch_size,
            optim_stats_every_n_steps: self.optim_stats_every_n_steps,
            grad_accum_in_fp32: self.grad_accum_in_fp32,
            eval_task_max_docs: self.eval_task_max_docs,
            eval_tasks: Vec::new(),
            prompt_task: self.prompt_task,
            wandb_info: self.wandb_info.clone(),
            write_gradients_dir: self.write_gradients_dir.clone(),
            checkpoint_config: self.checkpoint_config.clone(),
            dummy_training_delay_secs: self.dummy_training_delay_secs,
            sidecar_port: self.sidecar_port,
        }
    }
}

impl RunInitConfig {
    pub fn apply_run_templates(&mut self, run_id: &str) {
        if let Some(wandb_info) = &mut self.wandb_info {
            wandb_info.run = wandb_info.run.replace("{run_id}", run_id);
            if let Some(group) = &mut wandb_info.group {
                *group = group.replace("{run_id}", run_id);
            }
        }

        if let Some(checkpoint_config) = &mut self.checkpoint_config {
            if let Some(crate::state::UploadInfo::Hub(hub_info)) =
                &mut checkpoint_config.upload_info
            {
                hub_info.hub_repo = hub_info.hub_repo.replace("{run_id}", run_id);
            }
            checkpoint_config.checkpoint_dir = PathBuf::from(
                checkpoint_config
                    .checkpoint_dir
                    .to_string_lossy()
                    .replace("{run_id}", run_id),
            );
        }
    }
}

fn validate_parallelism(
    data_parallelism: usize,
    tensor_parallelism: usize,
    requires_native_parallelism: bool,
) -> Result<(), InitRunError> {
    if data_parallelism == 0 || tensor_parallelism == 0 {
        return Err(InitRunError::InvalidParallelism {
            data_parallelism,
            tensor_parallelism,
        });
    }

    #[cfg(not(feature = "parallelism"))]
    if requires_native_parallelism && (data_parallelism > 1 || tensor_parallelism > 1) {
        return Err(InitRunError::ParallelismNotEnabled {
            data_parallelism,
            tensor_parallelism,
        });
    }

    #[cfg(feature = "parallelism")]
    let _ = requires_native_parallelism;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::UploadInfo;
    use crate::WandBInfo;
    use aether_data_provider::HubUploadInfo;

    fn base_config() -> RunInitConfig {
        RunInitConfig {
            identity: NodeIdentity::default(),
            p2p_secret_key: SecretKey::from_bytes(&[0u8; 32]),
            max_concurrent_parameter_requests: 1,
            device: Devices::Cpu,
            hub_read_token: None,
            hub_max_concurrent_downloads: 1,
            data_parallelism: 1,
            tensor_parallelism: 1,
            micro_batch_size: 1,
            optim_stats_every_n_steps: None,
            grad_accum_in_fp32: false,
            eval_task_max_docs: None,
            eval_tasks: Vec::new(),
            prompt_task: false,
            wandb_info: None,
            write_gradients_dir: None,
            checkpoint_config: None,
            dummy_training_delay_secs: None,
            sidecar_port: None,
        }
    }

    #[test]
    fn apply_run_templates_is_noop_without_placeholders() {
        let mut cfg = base_config();
        cfg.wandb_info = Some(WandBInfo {
            project: "p".into(),
            run: "fixed-run".into(),
            group: Some("fixed-group".into()),
            entity: None,
            api_key: "k".into(),
        });
        cfg.checkpoint_config = Some(CheckpointConfig {
            upload_info: Some(UploadInfo::Hub(HubUploadInfo {
                hub_repo: "fixed-repo".into(),
                hub_token: "t".into(),
            })),
            checkpoint_dir: PathBuf::from("/tmp/fixed"),
            delete_old_steps: false,
            keep_steps: 1,
            epoch_interval: 1,
        });

        let mut expected = cfg.clone();
        expected.apply_run_templates("does-not-matter");

        // No `{run_id}` placeholders present -> strings are unchanged.
        assert_eq!(
            cfg.wandb_info.as_ref().unwrap().run,
            expected.wandb_info.as_ref().unwrap().run
        );
    }

    #[test]
    fn apply_run_templates_substitutes_run_id_everywhere() {
        let mut cfg = base_config();
        cfg.wandb_info = Some(WandBInfo {
            project: "p".into(),
            run: "run-{run_id}".into(),
            group: Some("grp-{run_id}-x".to_string()),
            entity: None,
            api_key: "k".into(),
        });
        cfg.checkpoint_config = Some(CheckpointConfig {
            upload_info: Some(UploadInfo::Hub(HubUploadInfo {
                hub_repo: "repo-{run_id}".into(),
                hub_token: "t".into(),
            })),
            checkpoint_dir: PathBuf::from("/tmp/ck-{run_id}"),
            delete_old_steps: false,
            keep_steps: 1,
            epoch_interval: 1,
        });

        cfg.apply_run_templates("abc123");

        let wb = cfg.wandb_info.as_ref().unwrap();
        assert_eq!(wb.run, "run-abc123");
        assert_eq!(wb.group.as_deref(), Some("grp-abc123-x"));
        let cc = cfg.checkpoint_config.as_ref().unwrap();
        assert_eq!(cc.checkpoint_dir, PathBuf::from("/tmp/ck-abc123"));
        match &cc.upload_info {
            Some(UploadInfo::Hub(h)) => assert_eq!(h.hub_repo, "repo-abc123"),
            other => panic!("expected Hub upload info, got {other:?}"),
        }
    }

    #[test]
    fn apply_run_templates_noop_when_no_wandb_or_checkpoint() {
        let mut cfg = base_config();
        // Both wandb_info and checkpoint_config are None.
        cfg.apply_run_templates("anything");
        assert!(cfg.wandb_info.is_none());
        assert!(cfg.checkpoint_config.is_none());
    }

    #[test]
    fn apply_run_templates_substitutes_multiple_occurrences() {
        let mut cfg = base_config();
        cfg.wandb_info = Some(WandBInfo {
            project: "p".into(),
            run: "{run_id}-{run_id}".into(),
            group: None,
            entity: None,
            api_key: "k".into(),
        });
        cfg.apply_run_templates("z");
        assert_eq!(cfg.wandb_info.as_ref().unwrap().run, "z-z");
    }

    #[test]
    fn parallelism_counts_must_be_nonzero() {
        assert!(matches!(
            validate_parallelism(0, 1, true),
            Err(InitRunError::InvalidParallelism {
                data_parallelism: 0,
                tensor_parallelism: 1,
            })
        ));
        assert!(matches!(
            validate_parallelism(1, 0, true),
            Err(InitRunError::InvalidParallelism {
                data_parallelism: 1,
                tensor_parallelism: 0,
            })
        ));
    }

    #[cfg(not(feature = "parallelism"))]
    #[test]
    fn native_parallelism_requires_parallelism_feature() {
        for (data_parallelism, tensor_parallelism) in [(2, 1), (1, 2), (2, 2)] {
            assert!(matches!(
                validate_parallelism(data_parallelism, tensor_parallelism, true),
                Err(InitRunError::ParallelismNotEnabled {
                    data_parallelism: error_dp,
                    tensor_parallelism: error_tp,
                }) if error_dp == data_parallelism && error_tp == tensor_parallelism
            ));
        }
    }

    #[cfg(all(not(feature = "parallelism"), feature = "python"))]
    #[test]
    fn python_distributed_parallelism_does_not_require_native_feature() {
        assert!(validate_parallelism(2, 2, false).is_ok());
    }

    #[cfg(feature = "parallelism")]
    #[test]
    fn native_parallelism_is_accepted_when_enabled() {
        assert!(validate_parallelism(2, 2, true).is_ok());
    }
}

#[derive(Debug, Error)]
pub enum InitRunError {
    #[error("No model provided in Coordinator state, nothing to do.")]
    NoModel,

    #[error("Model is Ephemeral, it's impossible to join this run.")]
    ModelIsEphemeral,

    #[error("failed to read local model info: {0}")]
    LocalModelLoad(#[from] io::Error),

    #[error("failed to read HF model info: {0}")]
    HfModelLoad(#[from] hf_hub::api::tokio::ApiError),

    #[error("failed to download model from GCS: {0}")]
    GcsModelLoad(#[from] DownloadError),

    #[error("model loading thread crashed")]
    ModelLoadingThreadCrashed(JoinError),

    #[error("failed to load model: {0}")]
    ModelLoad(#[from] ModelLoadError),

    #[error("Couldn't load tokenizer: {0}")]
    TokenizerLoad(#[from] AutoTokenizerError),

    // TODO refactor data provider for real errors
    #[error("Couldn't initialize data provider: {0}")]
    DataProviderConnect(anyhow::Error),

    #[error("wandb setup thread crashed")]
    WandbThreadCrashed(JoinError),

    #[error("wandb failed to create run: {0}")]
    WandbLoad(#[from] wandb::ApiError),

    #[error("could not parse config: {0}")]
    FailedToParseConfig(#[from] serde_json::Error),

    #[error(
        "P2P model load failed: could not fetch model from peers after exhausting all retries"
    )]
    P2PModelLoad,

    #[error("Unsupported architecture: {0}")]
    UnsupportedArchitecture(String),

    #[error(
        "data_parallelism and tensor_parallelism must both be at least 1 (got data_parallelism={data_parallelism}, tensor_parallelism={tensor_parallelism})"
    )]
    InvalidParallelism {
        data_parallelism: usize,
        tensor_parallelism: usize,
    },

    #[error(
        "native data/tensor parallelism requires the `parallelism` feature (got data_parallelism={data_parallelism}, tensor_parallelism={tensor_parallelism})"
    )]
    ParallelismNotEnabled {
        data_parallelism: usize,
        tensor_parallelism: usize,
    },

    #[cfg(feature = "python")]
    #[error("Python distributed error: {0}")]
    PythonDistributedError(#[from] aether_modeling::PythonDistributedCausalLMError),

    #[cfg(feature = "python")]
    #[error("Python model error: {0}")]
    PythonModelError(#[from] aether_modeling::PythonCausalLMError),

    #[cfg(feature = "python")]
    #[error("Python distributed trainer error: {0}")]
    PythonDistributedTrainerError(#[from] aether_modeling::PythonDistributedTrainerError),
}

enum RawLoadedModelType {
    ParallelNativeModels(Vec<Box<dyn CausalLM>>),
    #[cfg(feature = "python")]
    Python(aether_modeling::PythonCausalLM),
    #[cfg(feature = "python")]
    PythonDistributed(aether_modeling::PythonDistributedCausalLM),
}

struct RawLoadedModel {
    models: RawLoadedModelType,
    tokenizer: Arc<Tokenizer>,
    model_task_runner: ModelTaskRunner,
    checkpoint_extra_files: Vec<PathBuf>,
}

type OneshotModelParameterSender = oneshot::Sender<HashMap<String, Tensor>>;
type OneShotModelConfigSender = oneshot::Sender<(String, Tokenizer, Vec<String>)>;

#[derive(Clone)]
pub struct RunInitConfigAndIO {
    pub init_config: RunInitConfig,

    pub tx_health_check: UnboundedSender<HealthChecks>,
    pub tx_witness: UnboundedSender<OpportunisticData>,
    pub tx_checkpoint: UnboundedSender<model::Checkpoint>,
    pub tx_model: UnboundedSender<HashMap<String, Tensor>>,
    pub tx_parameters_req: UnboundedSender<(Vec<String>, OneshotModelParameterSender)>,
    pub tx_config: UnboundedSender<(String, String)>,
    pub tx_distro_result: UnboundedSender<DistroBroadcastAndPayload>,
    pub tx_request_download: UnboundedSender<(BlobTicket, Tag)>,
    pub tx_request_model_config: UnboundedSender<OneShotModelConfigSender>,
    pub tx_broadcast_finished: UnboundedSender<FinishedBroadcast>,

    /// Fired once after the checkpoint has been downloaded and the model loaded,
    /// signalling to the server that this client is ready to be admitted into
    /// the next epoch.
    pub tx_ready_for_epoch: UnboundedSender<()>,

    pub metrics: Arc<ClientMetrics>,
}

impl RunInitConfigAndIO {
    /// Call this on first warmup - when we need to enter the run, we have to load the model, connect to the data server, etc
    pub async fn init_run(self, state: Coordinator) -> Result<StepStateMachine, InitRunError> {
        let Self {
            mut init_config,
            tx_witness,
            tx_health_check,
            tx_checkpoint,
            tx_model,
            tx_config,
            tx_parameters_req,
            tx_distro_result,
            tx_request_download,
            tx_request_model_config,
            tx_broadcast_finished,
            tx_ready_for_epoch,
            metrics,
        } = self;

        let run_id = String::from(&state.run_id);
        init_config.apply_run_templates(&run_id);

        let model::Model::LLM(llm) = state.model;

        let requires_native_parallelism = !cfg!(feature = "python")
            || matches!(
                &llm.architecture,
                model::LLMArchitecture::HfLlama | model::LLMArchitecture::HfDeepseek
            )
            || matches!(&llm.checkpoint, model::Checkpoint::Dummy(_));
        validate_parallelism(
            init_config.data_parallelism,
            init_config.tensor_parallelism,
            requires_native_parallelism,
        )?;

        tch::manual_seed(1337);

        // Check device availability early
        if !init_config.device.is_probably_available() {
            return Err(InitRunError::ModelLoad(
                aether_modeling::ModelLoadError::UnavailbleDevice(init_config.device),
            ));
        }

        let hub_read_token = init_config.hub_read_token.clone();
        let hub_max_concurrent_downloads = init_config.hub_max_concurrent_downloads;
        let data_future = async {
            debug!("Setting up data provider from {:?}", llm.data_location);
            let data_provider = match llm.data_location {
                LLMTrainingDataLocation::Server(data_server) => DataProvider::Server(
                    DataProviderTcpClient::connect(
                        (&data_server).into(),
                        init_config.p2p_secret_key.clone(),
                    )
                    .await?,
                ),
                LLMTrainingDataLocation::Local(local) => {
                    DataProvider::Local(LocalDataProvider::new_from_directory(
                        String::from(&local.path),
                        local.token_size_in_bytes,
                        llm.max_seq_len as usize,
                        local.shuffle,
                    )?)
                }
                LLMTrainingDataLocation::Dummy => {
                    DataProvider::Dummy(DummyDataProvider::new(TokenSize::TwoBytes, 2048, u64::MAX))
                }
                LLMTrainingDataLocation::Http(HttpLLMTrainingDataLocation {
                    location,
                    token_size_in_bytes,
                    shuffle,
                }) => {
                    let file_urls = FileURLs::from_location(&location).await?;
                    DataProvider::Http(HttpDataProvider::new(
                        file_urls,
                        token_size_in_bytes,
                        llm.max_seq_len,
                        shuffle,
                    )?)
                }
                LLMTrainingDataLocation::WeightedHttp(config_url) => DataProvider::WeightedHttp(
                    WeightedDataProvider::<HttpDataProvider>::from_config_url(
                        &String::from(&config_url),
                        llm.max_seq_len,
                    )
                    .await?,
                ),
                LLMTrainingDataLocation::Preprocessed(url) => {
                    let url: String = (&url).into();
                    let dir = if std::fs::exists(&url).unwrap_or_default() {
                        PathBuf::from(url)
                    } else {
                        download_dataset_repo_async(
                            url.clone(),
                            None,
                            None,
                            hub_read_token,
                            Some(hub_max_concurrent_downloads),
                            false,
                        )
                        .await?
                        .first()
                        .ok_or(anyhow::anyhow!("No files downloaded for {url}"))?
                        .parent()
                        .unwrap()
                        .into()
                    };
                    DataProvider::Preprocessed(PreprocessedDataProvider::new_from_directory(
                        dir,
                        llm.max_seq_len as usize,
                        Shuffle::DontShuffle,
                        Some(Split::Train),
                        None,
                    )?)
                }
            };
            Ok(data_provider)
        };

        let model_future: JoinHandle<Result<RawLoadedModel, InitRunError>> = match &llm.architecture
        {
            model::LLMArchitecture::HfLlama
            | model::LLMArchitecture::HfDeepseek
            | model::LLMArchitecture::HfAuto
            | model::LLMArchitecture::Torchtitan => match &llm.checkpoint {
                model::Checkpoint::Dummy(_) => tokio::spawn(async move {
                    let tokenizer = Arc::new(Tokenizer::new(ModelWrapper::WordLevel(
                        WordLevel::builder().build().unwrap(),
                    )));

                    let model = RawLoadedModel {
                        models: RawLoadedModelType::ParallelNativeModels(
                            (0..(init_config.data_parallelism * init_config.tensor_parallelism))
                                .map(|_| {
                                    if let Some(training_delay) =
                                        init_config.dummy_training_delay_secs
                                    {
                                        Box::new(DummyModel::new(training_delay))
                                            as Box<dyn CausalLM>
                                    } else {
                                        Box::new(DummyModel::default()) as Box<dyn CausalLM>
                                    }
                                })
                                .collect(),
                        ),
                        tokenizer: tokenizer.clone(),
                        checkpoint_extra_files: vec![],
                        model_task_runner: ModelTaskRunner::new(
                            vec![],
                            false,
                            tokenizer.clone(),
                            None,
                            0,
                        ),
                    };
                    // SAFETY: Tensor is !Sync but this Arc is consumed during
                    // model loading in a sequential context.
                    #[allow(clippy::arc_with_non_send_sync)]
                    let config = &PretrainedSource::ConfigAndTensors(
                        AutoConfig::Llama(LlamaConfig::dummy()),
                        Arc::new(aether_modeling::get_dummy_parameters()),
                    )
                    .serialize_config()?;
                    let tokenizer = tokenizer.to_string(false).unwrap();
                    info!("Config Uploaded: {}", config);
                    tx_config.send((config.to_string(), tokenizer)).unwrap();
                    Ok(model)
                }),
                model::Checkpoint::Hub(_)
                | model::Checkpoint::P2P(_)
                | model::Checkpoint::P2PGcs(_)
                | model::Checkpoint::Gcs(_) => {
                    let checkpoint = llm.checkpoint;
                    tokio::spawn(async move {
                        let (source, tokenizer, checkpoint_extra_files) = match checkpoint {
                            model::Checkpoint::Hub(hub_repo) => {
                                let repo_id: String = (&hub_repo.repo_id).into();
                                let potential_local_path = PathBuf::from(repo_id.clone());
                                let revision = hub_repo.revision.map(|bytes| (&bytes).into());

                                let model_is_local = if revision.is_none()
                                    && tokio::fs::try_exists(potential_local_path.clone())
                                        .await
                                        .unwrap_or_default()
                                {
                                    let mut ret = Vec::new();
                                    let mut read_dir =
                                        tokio::fs::read_dir(potential_local_path).await?;
                                    while let Some(dir_entry) = read_dir.next_entry().await? {
                                        ret.push(dir_entry.path())
                                    }
                                    ret
                                } else {
                                    info!(
                                        "Downloading {}, revision: {:?} (if needed)",
                                        hub_repo.repo_id, revision
                                    );
                                    event!(warmup::CheckpointDownloadStarted { size_bytes: 0 });
                                    match download_model_repo_async(
                                        &repo_id,
                                        revision,
                                        None,
                                        init_config.hub_read_token,
                                        Some(init_config.hub_max_concurrent_downloads),
                                        false,
                                    )
                                    .await
                                    {
                                        Ok(downloaded) => {
                                            event!(warmup::CheckpointDownloadComplete(Ok(())));
                                            downloaded
                                        }
                                        Err(e) => {
                                            event!(warmup::CheckpointDownloadComplete(Err(
                                                e.to_string()
                                            )));
                                            return Err(e.into());
                                        }
                                    }
                                };
                                let repo_files = model_is_local;
                                let checkpoint_extra_files = repo_files
                                    .iter()
                                    .filter(|file| {
                                        file.ends_with("config.json")
                                            || file.ends_with("tokenizer.json")
                                            || file.ends_with("tokenizer_config.json")
                                            || file.ends_with("special_tokens_map.json")
                                            || file.ends_with("generation_config.json")
                                            || file.ends_with(".py")
                                    })
                                    .cloned()
                                    .collect();
                                let tokenizer = Arc::new(auto_tokenizer(&repo_files)?);
                                (
                                    PretrainedSource::<AutoConfig>::RepoFiles(repo_files),
                                    tokenizer,
                                    checkpoint_extra_files,
                                )
                            }
                            model::Checkpoint::P2P(_) | model::Checkpoint::P2PGcs(_) => {
                                let (tx_model_config_response, rx_model_config_response) =
                                    oneshot::channel();
                                info!("Checkpoint is p2p, requesting model config over network");

                                tx_request_model_config
                                    .send(tx_model_config_response)
                                    .unwrap();

                                let (model_config, tokenizer, parameter_names) =
                                    rx_model_config_response
                                        .await
                                        .map_err(|_| InitRunError::P2PModelLoad)?;
                                debug!("Got p2p info, model_config: {}", model_config);

                                let model_config = match llm.architecture {
                                    model::LLMArchitecture::HfLlama => {
                                        AutoConfig::Llama(serde_json::from_str(&model_config)?)
                                    }
                                    model::LLMArchitecture::HfDeepseek => {
                                        AutoConfig::Deepseek(serde_json::from_str(&model_config)?)
                                    }
                                    model::LLMArchitecture::HfAuto
                                    | model::LLMArchitecture::Torchtitan => {
                                        #[cfg(feature = "python")]
                                        {
                                            AutoConfig::Auto(serde_json::from_str::<
                                                aether_modeling::PythonModelConfig,
                                            >(
                                                &model_config
                                            )?)
                                        }

                                        #[cfg(not(feature = "python"))]
                                        {
                                            return Err(InitRunError::UnsupportedArchitecture(
                                                llm.architecture.to_string(),
                                            ));
                                        }
                                    }
                                };
                                info!(
                                    "Requesting {} parameters over p2p network",
                                    parameter_names.len()
                                );

                                let (tx_params_response, rx_params_response) = oneshot::channel();
                                tx_parameters_req
                                    .send((parameter_names, tx_params_response))
                                    .unwrap();
                                // SAFETY: Tensor is !Sync but these parameters
                                // are loaded sequentially before model init.
                                #[allow(clippy::arc_with_non_send_sync)]
                                let parameters = Arc::new(
                                    rx_params_response
                                        .await
                                        .map_err(|_| InitRunError::P2PModelLoad)?,
                                );

                                (
                                    PretrainedSource::<AutoConfig>::ConfigAndTensors(
                                        model_config,
                                        parameters,
                                    ),
                                    Arc::new(tokenizer),
                                    vec![],
                                )
                            }
                            model::Checkpoint::Gcs(gcs_repo) => {
                                let bucket: String = (&gcs_repo.bucket).into();
                                let prefix: Option<String> = gcs_repo.prefix.map(|p| (&p).into());

                                info!(
                                    "Downloading model from gs://{}/{}",
                                    bucket,
                                    prefix.as_deref().unwrap_or("")
                                );

                                event!(warmup::CheckpointDownloadStarted { size_bytes: 0 });
                                let repo_files =
                                    match download_model_from_gcs_async(&bucket, prefix.as_deref())
                                        .await
                                    {
                                        Ok(files) => {
                                            event!(warmup::CheckpointDownloadComplete(Ok(())));
                                            files
                                        }
                                        Err(e) => {
                                            event!(warmup::CheckpointDownloadComplete(Err(
                                                e.to_string()
                                            )));
                                            return Err(e.into());
                                        }
                                    };

                                let checkpoint_extra_files = repo_files
                                    .iter()
                                    .filter(|file| {
                                        file.ends_with("config.json")
                                            || file.ends_with("tokenizer.json")
                                            || file.ends_with("tokenizer_config.json")
                                            || file.ends_with("special_tokens_map.json")
                                            || file.ends_with("generation_config.json")
                                            || file.ends_with(".py")
                                    })
                                    .cloned()
                                    .collect();
                                let tokenizer = Arc::new(auto_tokenizer(&repo_files)?);
                                (
                                    PretrainedSource::<AutoConfig>::RepoFiles(repo_files),
                                    tokenizer,
                                    checkpoint_extra_files,
                                )
                            }
                            _ => unreachable!(),
                        };

                        info!("Loading model...");
                        event!(warmup::ModelLoadStarted);

                        let model_task_runner = ModelTaskRunner::new(
                            init_config.eval_tasks,
                            init_config.prompt_task,
                            tokenizer.clone(),
                            init_config.eval_task_max_docs,
                            // if doing python fsdp we only have one effective dp rank for inference
                            if init_config.data_parallelism > 1
                                && llm.architecture == model::LLMArchitecture::HfAuto
                            {
                                1
                            } else {
                                init_config.data_parallelism
                            },
                        );

                        let serialized_config = source.serialize_config()?;
                        let attn_implementation: Option<AttentionImplementation> =
                            match llm.data_type {
                                model::LLMTrainingDataType::Finetuning => {
                                    #[cfg(feature = "parallelism")]
                                    {
                                        // use varlen backend if available
                                        Some(AttentionImplementation::FlashAttention2)
                                    }

                                    #[cfg(not(feature = "parallelism"))]
                                    None
                                }
                                model::LLMTrainingDataType::Pretraining => None,
                            };

                        let raw_loaded_model_type: RawLoadedModelType = match llm.architecture {
                            model::LLMArchitecture::HfAuto | model::LLMArchitecture::Torchtitan => {
                                #[cfg(feature = "python")]
                                {
                                    let dp = init_config.data_parallelism;
                                    let tp = init_config.tensor_parallelism;
                                    let num_local_ranks = init_config.device.size() as i64;

                                    tokio::task::spawn_blocking(move || {
                                        if tp != 1 || dp != 1 {
                                            aether_modeling::PythonDistributedCausalLM::new(
                                                llm.architecture.to_string(),
                                                source.try_into()?,
                                                tch::Device::cuda_if_available(),
                                                attn_implementation.unwrap_or_default(),
                                                aether_modeling::ParallelismConfig { dp, tp },
                                                Some(llm.max_seq_len as usize),
                                                init_config.sidecar_port,
                                                Some(num_local_ranks),
                                            )
                                            .map(RawLoadedModelType::PythonDistributed)
                                            .map_err(InitRunError::PythonDistributedError)
                                        } else {
                                            let device = init_config
                                                .device
                                                .device_for_rank(0)
                                                .ok_or_else(|| {
                                                    ModelLoadError::NoDeviceForRank(
                                                        0,
                                                        init_config.device,
                                                    )
                                                })?;
                                            aether_modeling::PythonCausalLM::new(
                                                &llm.architecture.to_string(),
                                                &source.try_into()?,
                                                device,
                                                attn_implementation.unwrap_or_default(),
                                                None,
                                                Some(llm.max_seq_len as usize),
                                            )
                                            .map(RawLoadedModelType::Python)
                                            .map_err(InitRunError::PythonModelError)
                                        }
                                    })
                                    .await
                                    .map_err(InitRunError::ModelLoadingThreadCrashed)??
                                }

                                #[cfg(not(feature = "python"))]
                                {
                                    return Err(InitRunError::UnsupportedArchitecture(
                                        llm.architecture.to_string(),
                                    ));
                                }
                            }
                            architecture => {
                                let mut futures: Vec<
                                    JoinHandle<Result<Box<dyn CausalLM>, ModelLoadError>>,
                                > = Vec::with_capacity(
                                    init_config.data_parallelism * init_config.tensor_parallelism,
                                );
                                let devices = init_config.device.clone();

                                for dp in 0..init_config.data_parallelism {
                                    #[cfg(feature = "parallelism")]
                                    let communicator_id: Option<CommunicatorId> =
                                        match init_config.tensor_parallelism {
                                            0 | 1 => None,
                                            _ => Some(tch::CStore::new().into()),
                                        };
                                    #[cfg(not(feature = "parallelism"))]
                                    let communicator_id: Option<CommunicatorId> = None;
                                    for tp in 0..init_config.tensor_parallelism {
                                        let tensor_parallelism_world =
                                            communicator_id.as_ref().map(|communicator_id| {
                                                (
                                                    communicator_id.clone(),
                                                    tp,
                                                    init_config.tensor_parallelism,
                                                )
                                            });
                                        let source = source.clone();
                                        let rank = dp * init_config.tensor_parallelism + tp;
                                        let devices = devices.clone();
                                        let device = devices.device_for_rank(rank);
                                        futures.push(tokio::task::spawn_blocking(move || {
                                            let device = device.ok_or_else(|| {
                                                ModelLoadError::NoDeviceForRank(rank, devices)
                                            })?;
                                            match architecture {
                                                model::LLMArchitecture::HfLlama => {
                                                    LlamaForCausalLM::from_pretrained(
                                                        &source.try_into()?,
                                                        Some(Kind::BFloat16),
                                                        attn_implementation,
                                                        Some(device),
                                                        tensor_parallelism_world,
                                                        Some(llm.max_seq_len as usize),
                                                    )
                                                    .map(|x| Box::new(x) as Box<dyn CausalLM>)
                                                }
                                                model::LLMArchitecture::HfDeepseek => {
                                                    DeepseekForCausalLM::from_pretrained(
                                                        &source.try_into()?,
                                                        Some(Kind::BFloat16),
                                                        attn_implementation,
                                                        Some(device),
                                                        tensor_parallelism_world,
                                                        Some(llm.max_seq_len as usize),
                                                    )
                                                    .map(|x| Box::new(x) as Box<dyn CausalLM>)
                                                }
                                                model::LLMArchitecture::HfAuto
                                                | model::LLMArchitecture::Torchtitan => {
                                                    unreachable!()
                                                }
                                            }
                                        }));
                                    }
                                }

                                let mut models: Vec<Box<dyn CausalLM>> = Vec::new();
                                for future in futures {
                                    let model = future
                                        .await
                                        .map_err(InitRunError::ModelLoadingThreadCrashed)??;
                                    models.push(model);
                                }

                                RawLoadedModelType::ParallelNativeModels(models)
                            }
                        };

                        debug!("Config uploaded: {}", serialized_config);
                        let serialized_tokenizer = tokenizer.to_string(false).unwrap();
                        tx_config
                            .send((serialized_config.clone(), serialized_tokenizer))
                            .unwrap();

                        event!(warmup::ModelLoadComplete);
                        info!(
                            integration_test_log_marker = %IntegrationTestLogMarker::LoadedModel,
                            checkpoint = %llm.checkpoint,
                            gpus = init_config.data_parallelism * init_config.tensor_parallelism,
                            dp = init_config.data_parallelism,
                            tp = init_config.tensor_parallelism,
                            "loaded_model",
                        );

                        Ok(RawLoadedModel {
                            models: raw_loaded_model_type,
                            tokenizer,
                            model_task_runner,
                            checkpoint_extra_files,
                        })
                    })
                }
                model::Checkpoint::Ephemeral => return Err(InitRunError::ModelIsEphemeral),
            },
        };

        let wandb_future: JoinHandle<Result<Option<wandb::Run>, wandb::ApiError>> = tokio::spawn({
            async move {
                match init_config.wandb_info {
                    Some(wandb_info) => {
                        let wandb =
                            wandb::WandB::new(wandb::BackendOptions::new(wandb_info.api_key));
                        let mut run_info = wandb::RunInfo::new(wandb_info.project)
                            .name(wandb_info.run)
                            .config((
                                (
                                    "global_batch_size_start",
                                    state.config.global_batch_size_start,
                                ),
                                ("global_batch_size_end", state.config.global_batch_size_end),
                                (
                                    "global_batch_size_warmup_tokens",
                                    state.config.global_batch_size_warmup_tokens,
                                ),
                                ("total_steps", state.config.total_steps),
                                ("run_id", run_id),
                            ));
                        if let Some(entity) = wandb_info.entity {
                            run_info = run_info.entity(entity);
                        }
                        if let Some(group) = wandb_info.group {
                            run_info = run_info.group(group);
                        }
                        match wandb.new_run(run_info.build()?).await {
                            Ok(run) => Ok(Some(run)),
                            Err(e) => {
                                error!(
                                    "[init_run] Could not connect to wandb. Will continue training without it."
                                );
                                debug!("[init_run] wandb error: {:?}", e);
                                Ok(None)
                            }
                        }
                    }
                    None => {
                        info!(
                            "[init_run] No wandb info provided. Will continue training without it."
                        );
                        Ok(None)
                    }
                }
            }
        });

        let (data, models, wandb_run) = tokio::join!(data_future, model_future, wandb_future);
        let RawLoadedModel {
            models,
            tokenizer,
            checkpoint_extra_files,
            model_task_runner,
        } = models.map_err(InitRunError::ModelLoadingThreadCrashed)??;

        // TODO add data fetching for verifying, too..
        let data_provider = data.map_err(InitRunError::DataProviderConnect)?;
        let data_fetcher = DataFetcher::new(data_provider, init_config.data_parallelism * 2);

        let trainers: Vec<Trainer> = match models {
            RawLoadedModelType::ParallelNativeModels(models) => {
                let mut tp_models: Vec<Vec<Box<dyn CausalLM>>> = Vec::new();
                for model in models {
                    if tp_models
                        .last()
                        .map(|x| x.len() == init_config.tensor_parallelism)
                        .unwrap_or(true)
                    {
                        tp_models.push(Vec::with_capacity(init_config.tensor_parallelism));
                    }
                    tp_models.last_mut().unwrap().push(model);
                }

                let data_parallel: Option<Vec<(CommunicatorId, Arc<dyn Barrier>)>> =
                    if init_config.data_parallelism > 1 {
                        #[cfg(feature = "parallelism")]
                        {
                            Some(
                                (0..init_config.tensor_parallelism)
                                    .map(|_| {
                                        (
                                            tch::CStore::new().into(),
                                            Arc::new(CancellableBarrier::new(
                                                init_config.tensor_parallelism,
                                            ))
                                                as Arc<dyn Barrier>,
                                        )
                                    })
                                    .collect(),
                            )
                        }

                        #[cfg(not(feature = "parallelism"))]
                        {
                            None
                        }
                    } else {
                        None
                    };

                tp_models
                    .into_iter()
                    .enumerate()
                    .map(|(dp, models)| {
                        let data_parallel = data_parallel.as_ref().map(|data_parallel| {
                            data_parallel
                                .iter()
                                .map(|(id, barrier)| DataParallel {
                                    id: id.clone(),
                                    barrier: barrier.clone(),
                                    rank: dp,
                                    world_size: init_config.data_parallelism,
                                })
                                .collect()
                        });
                        let barrier =
                            Arc::new(CancellableBarrier::new(init_config.tensor_parallelism))
                                as Arc<dyn Barrier>;
                        LocalTrainer::new(
                            ParallelModels {
                                models,
                                barrier,
                                data_parallel,
                            },
                            llm.lr_schedule,
                            llm.optimizer,
                            init_config.micro_batch_size,
                            init_config.optim_stats_every_n_steps,
                            init_config.grad_accum_in_fp32,
                        )
                        .into()
                    })
                    .collect()
            }
            #[cfg(feature = "python")]
            RawLoadedModelType::Python(model) => {
                vec![aether_modeling::LocalTrainer::new(
                    ParallelModels {
                        models: vec![Box::new(model) as Box<dyn CausalLM>],
                        barrier: Arc::new(aether_modeling::NopBarrier) as Arc<dyn Barrier>,
                        data_parallel: None,
                    },
                    llm.lr_schedule,
                    llm.optimizer,
                    init_config.micro_batch_size,
                    init_config.optim_stats_every_n_steps,
                    init_config.grad_accum_in_fp32,
                )
                .into()]
            }
            #[cfg(feature = "python")]
            RawLoadedModelType::PythonDistributed(model) => {
                vec![aether_modeling::PythonDistributedTrainer::new(
                    model,
                    llm.lr_schedule,
                    llm.optimizer,
                    init_config.micro_batch_size,
                    init_config.optim_stats_every_n_steps,
                    init_config.grad_accum_in_fp32,
                )?
                .into()]
            }
        };

        let wandb_run = wandb_run.map_err(InitRunError::WandbThreadCrashed)??;

        let stats_logger = StatsLogger::new(
            tokenizer,
            model_task_runner.clone(),
            llm.lr_schedule,
            wandb_run,
            metrics,
        );

        let warmup = WarmupStepMetadata {
            model_task_runner: model_task_runner.clone(),
        };

        let training = TrainingStepMetadata {
            data_fetcher,
            identity: init_config.identity,
            write_gradients_dir: init_config.write_gradients_dir,
            tx_health_check,
            tx_distro_result,

            model_task_runner: model_task_runner.clone(),
        };

        let witness = WitnessStepMetadata {
            model_task_runner: model_task_runner.clone(),
            identity: init_config.identity,
            tx_witness: tx_witness.clone(),
        };

        let cooldown = CooldownStepMetadata::new(
            tx_checkpoint,
            tx_model,
            init_config.checkpoint_config,
            checkpoint_extra_files,
            model_task_runner,
        );

        // Signal readiness so the server knows it can admit us into the next
        // epoch. This decouples checkpoint download from epoch warmup — a slow
        // joiner downloads in the background and is only admitted once ready,
        // so it never disrupts active training.
        info!("Checkpoint loaded — signalling readiness to server");
        let _ = tx_ready_for_epoch.send(());

        Ok(StepStateMachine::new(
            init_config.identity,
            warmup,
            training,
            witness,
            cooldown,
            trainers,
            state,
            tx_request_download,
            tx_witness,
            tx_broadcast_finished,
            tx_ready_for_epoch,
            stats_logger,
        ))
    }
}
