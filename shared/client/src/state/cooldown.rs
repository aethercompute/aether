use crate::UploadInfo;
use aether_coordinator::{
    model::{self, LLMTrainingMethod},
    Coordinator,
};
use aether_data_provider::{upload_to_gcs, upload_to_hub, GcsManifestMetadata, UploadError};
use aether_event_sourcing::event;
#[cfg(feature = "python")]
use aether_modeling::CausalLM;
use aether_modeling::{
    save_lora_adapter_into_safetensors, save_tensors_into_safetensors, AetherAdapterMetadata,
    LoraAdapterConfig, SaveSafetensorsError, Trainer, TrainerThreadCommunicationError,
};
use std::{
    cmp::Reverse,
    collections::{BinaryHeap, HashMap},
    path::PathBuf,
    sync::{Arc, Mutex as StdMutex},
};
use tch::Tensor;
use thiserror::Error;
use tokio::{
    sync::{mpsc, Mutex},
    task::JoinHandle,
};
use tracing::{info, info_span, warn, Instrument};

use super::{
    evals::{ModelTaskRunner, RunningEvals},
    CheckpointConfig,
};

#[derive(Error, Debug)]
pub enum CooldownError {
    #[error("no trainers available for checkpointing!")]
    NoTrainers,

    #[error("checkpointing thread crashed")]
    CheckpointThreadCrashed,

    #[error("error while checkpointing: {0}")]
    Checkpoint(#[from] CheckpointError),
}

pub struct CooldownStepMetadata {
    tx_checkpoint: mpsc::UnboundedSender<model::Checkpoint>,
    tx_model: mpsc::UnboundedSender<HashMap<String, Tensor>>,
    checkpoint_info: Option<CheckpointConfig>,
    checkpoint_extra_files: Vec<PathBuf>,

    model_task_runner: ModelTaskRunner,
    base_model_identity: Arc<StdMutex<Option<String>>>,
    // use a heap here as a best-effort attempt to ensure we get rid of the lowest step number dir even if we spawn multiple tasks
    // which may not finish writing their dirs in order. We note that even if we were to take the more complicated
    // route of actually enumerating the checkpoint_dir there would still be a race condition, unless we took a lockfile
    // or the like on the entire checkpoint_dir which probably isn't worth it just to support disk cleanup
    // we don't really expect there to be contention on this lock or real race conditions in practice though
    // as by the time one task spawns after a training round the previous write/upload task(s) should (hopefully) be long done
    delete_queue: Arc<Mutex<BinaryHeap<Reverse<u32>>>>,
}

impl CooldownStepMetadata {
    pub fn new(
        tx_checkpoint: mpsc::UnboundedSender<model::Checkpoint>,
        tx_model: mpsc::UnboundedSender<HashMap<String, Tensor>>,
        checkpoint_info: Option<CheckpointConfig>,
        checkpoint_extra_files: Vec<PathBuf>,
        model_task_runner: ModelTaskRunner,
    ) -> Self {
        Self {
            tx_checkpoint,
            tx_model,
            checkpoint_info,
            checkpoint_extra_files,
            model_task_runner,
            base_model_identity: Arc::new(StdMutex::new(None)),
            delete_queue: Arc::new(Mutex::new(BinaryHeap::new())),
        }
    }
}

#[derive(Error, Debug)]
pub enum CheckpointError {
    #[error("Extract thread crashed")]
    ExtractThreadCrashed,

    #[error("Trainer extract error: {0}")]
    Extract(#[from] TrainerThreadCommunicationError),

    #[error("Write thread crashed")]
    WriteThreadCrashed,

    #[error("Writing safetensors to disk failed: {0}")]
    WriteSafetensors(#[from] SaveSafetensorsError),

    #[error("Writing extra file to disk failed: {0}")]
    WriteExtraFile(#[from] tokio::io::Error),

    #[error("Couldn't upload model to huggingface or GCS: {0}")]
    UploadError(#[from] UploadError),

    #[error("Couldn't send checkpoint - channel closed")]
    SendCheckpoint,
}

async fn cleanup_dirs(
    delete_queue: Arc<Mutex<BinaryHeap<Reverse<u32>>>>,
    keep_steps: u32,
    run_id: String,
    delete_old_steps: bool,
    step: u32,
    checkpoint_dir: PathBuf,
) {
    if delete_old_steps {
        let mut delete_queue_guard = delete_queue.lock().await;
        delete_queue_guard.push(Reverse(step));
        // in the happy case this could be an if but if previous iterations failed somewhere
        // then we may have more than 1 dir to clean up
        while delete_queue_guard.len() > keep_steps as usize {
            let delete_step = delete_queue_guard.pop().unwrap().0;
            let delete_path = checkpoint_dir.join(format!("{run_id}-step{delete_step}"));
            if let Err(err) = tokio::fs::remove_dir_all(delete_path.clone()).await {
                warn!("Error removing {} : {}", delete_path.display(), err);
            } else {
                info!("Successfully removed {}", delete_path.display());
            }
        }
    }
}

impl CooldownStepMetadata {
    pub fn start(
        &self,
        mut trainers: Vec<Trainer>,
        state: &Coordinator,
    ) -> Result<CooldownStep, CooldownError> {
        let Some(trainer) = trainers.pop() else {
            return Err(CooldownError::NoTrainers);
        };

        let step = state.progress.step - 1;
        let run_id = String::from(&state.run_id);
        let epoch = state.progress.epoch as u32;
        let checkpoint_extra_files = self.checkpoint_extra_files.clone();
        let checkpoint_info = self.checkpoint_info.clone();
        let tx_checkpoint = self.tx_checkpoint.clone();
        let tx_model = self.tx_model.clone();
        let model_task_runner = self.model_task_runner.clone();
        let delete_queue = self.delete_queue.clone();
        let model::Model::LLM(llm) = state.model;
        let training_method = llm.training_method;
        let trainable_manifest_digest = trainer.manifest_digest();
        let base_model_identity = {
            let mut identity = self.base_model_identity.lock().unwrap();
            identity
                .get_or_insert_with(|| llm.checkpoint.to_string())
                .clone()
        };

        let checkpointing_and_evals: CheckpointAndEvalsHandle = tokio::task::spawn(
            async move {
                info!(
                    "Extracting {} model state...",
                    if matches!(training_method, LLMTrainingMethod::Lora(_)) {
                        "LoRA adapter"
                    } else {
                        "full"
                    }
                );
                event!(cooldown::ModelSerializationStarted);
                let (variables, trainer) = tokio::task::spawn_blocking::<
                    _,
                    Result<_, CheckpointError>,
                >(move || {
                    let mut trainer = trainer;
                    let variables: HashMap<String, Tensor> = match training_method {
                        LLMTrainingMethod::Full => {
                            trainer.truncate_bf16()?;
                            trainer
                                .extract()?
                                .into_iter()
                                .map(|(name, tensor)| (name, tensor.to_kind(tch::Kind::BFloat16)))
                                .collect()
                        }
                        LLMTrainingMethod::Lora(_) => trainer.extract_trainable()?,
                    };
                    info!("Model extracted; {} parameters", variables.len());
                    Ok((variables, trainer))
                })
                .await
                .map_err(|_| {
                    event!(cooldown::ModelSerializationFinished {
                        success: false,
                        error_string: Some("extract thread crashed".to_string())
                    });
                    CheckpointError::ExtractThreadCrashed
                })??;
                event!(cooldown::ModelSerializationFinished {
                    success: true,
                    error_string: None
                });

                let variables_clone: HashMap<String, Tensor> = variables
                    .iter()
                    .map(|(name, var)| (name.clone(), var.shallow_clone()))
                    .collect();

                // for p2p model sharing we use the native trainer shape
                tx_model
                    .send(variables_clone)
                    .map_err(|_| CheckpointError::SendCheckpoint)?;

                // convert from internal shape to serialized shape (e.g. torchtitan to hf)
                let (variables, trainer) = match (training_method, trainer) {
                    #[cfg(feature = "python")]
                    (LLMTrainingMethod::Full, trainer @ Trainer::PythonDistributed(_)) => {
                        info!("Converting distributed trainer variables for checkpointing...");
                        tokio::task::spawn_blocking(|| (trainer.convert(Some(variables)), trainer))
                            .await
                            .map_err(|_| CheckpointError::ExtractThreadCrashed)?
                    }
                    (_, trainer) => (variables, trainer),
                };

                trainers.push(trainer);
                let evals = model_task_runner.start(trainers);

                let Some(CheckpointConfig {
                    upload_info,
                    checkpoint_dir,
                    delete_old_steps,
                    keep_steps,
                    epoch_interval,
                }) = checkpoint_info
                else {
                    return Ok((evals, None));
                };

                // Throttle checkpointing to every N epochs. The P2P model share
                // above still happens every epoch; only the local save + HF/GCS
                // upload is skipped on off-interval epochs.
                if epoch_interval > 1 && !epoch.is_multiple_of(epoch_interval) {
                    return Ok((evals, None));
                }

                let upload_handle = tokio::task::spawn(async move {
                    let path = checkpoint_dir.join(format!("{run_id}-step{step}"));
                    let local = save_checkpoint_locally(
                        path,
                        variables,
                        checkpoint_extra_files,
                        CheckpointSaveContext {
                            training_method,
                            base_model_identity,
                            run_id: run_id.clone(),
                            epoch,
                            step,
                            trainable_manifest_digest,
                        },
                    )
                    .await?;

                    if let Some(upload_info) = upload_info {
                        let manifest_metadata = GcsManifestMetadata {
                            epoch,
                            run_id: run_id.clone(),
                        };
                        upload_checkpoint(
                            upload_info,
                            manifest_metadata,
                            local.clone(),
                            step as u64,
                            tx_checkpoint,
                        )
                        .await?;
                    }

                    cleanup_dirs(
                        delete_queue,
                        keep_steps,
                        run_id,
                        delete_old_steps,
                        step,
                        checkpoint_dir,
                    )
                    .await;

                    Ok(())
                });

                Ok((evals, Some(upload_handle)))
            }
            .instrument(info_span!("checkpointing")),
        );

        Ok(CooldownStep {
            checkpointing_and_evals,
        })
    }
}

struct CheckpointSaveContext {
    training_method: LLMTrainingMethod,
    base_model_identity: String,
    run_id: String,
    epoch: u32,
    step: u32,
    trainable_manifest_digest: [u8; 32],
}

async fn save_checkpoint_locally(
    path: PathBuf,
    variables: HashMap<String, Tensor>,
    checkpoint_extra_files: Vec<PathBuf>,
    context: CheckpointSaveContext,
) -> Result<Vec<PathBuf>, CheckpointError> {
    let CheckpointSaveContext {
        training_method,
        base_model_identity,
        run_id,
        epoch,
        step,
        trainable_manifest_digest,
    } = context;
    info!("Saving to {}", path.display());
    event!(cooldown::CheckpointWriteStarted);
    let mut local = tokio::task::spawn_blocking({
        let path = path.clone();
        move || match training_method {
            LLMTrainingMethod::Full => save_tensors_into_safetensors(variables, path),
            LLMTrainingMethod::Lora(config) => save_lora_adapter_into_safetensors(
                variables,
                path,
                &LoraAdapterConfig::new(
                    base_model_identity,
                    config.rank,
                    config.alpha,
                    config.dropout,
                ),
                &AetherAdapterMetadata {
                    artifact_type: "lora_adapter".to_string(),
                    format_version: 1,
                    run_id,
                    epoch,
                    step,
                    trainable_manifest_digest: trainable_manifest_digest
                        .iter()
                        .map(|byte| format!("{byte:02x}"))
                        .collect(),
                },
            ),
        }
    })
    .await
    .map_err(|_| {
        event!(cooldown::CheckpointWriteFinished {
            success: false,
            error_string: Some("write thread crashed".to_string())
        });
        CheckpointError::WriteThreadCrashed
    })??;

    for extra in checkpoint_extra_files {
        if matches!(training_method, LLMTrainingMethod::Lora(_))
            && extra.file_name().is_some_and(|name| {
                name == "adapter_config.json"
                    || name == "adapter_model.safetensors"
                    || name == "aether_adapter.json"
            })
        {
            continue;
        }
        let to = path.join(extra.file_name().unwrap());
        tokio::fs::copy(extra.clone(), to.clone())
            .await
            .map_err(|e| {
                event!(cooldown::CheckpointWriteFinished {
                    success: false,
                    error_string: Some(e.to_string())
                });
                CheckpointError::WriteExtraFile(e)
            })?;
        local.push(to);
    }

    event!(cooldown::CheckpointWriteFinished {
        success: true,
        error_string: None
    });
    Ok(local)
}

async fn upload_checkpoint(
    upload_info: UploadInfo,
    manifest_metadata: GcsManifestMetadata,
    local: Vec<PathBuf>,
    step: u64,
    tx_checkpoint: mpsc::UnboundedSender<model::Checkpoint>,
) -> Result<(), CheckpointError> {
    event!(cooldown::CheckpointUploadStarted);
    let result = match upload_info {
        UploadInfo::Gcs(gcs_info) => {
            upload_to_gcs(gcs_info, manifest_metadata, local, step, tx_checkpoint)
                .await
                .map_err(CheckpointError::UploadError)
        }
        UploadInfo::Hub(hub_info) => upload_to_hub(hub_info, local, step, tx_checkpoint)
            .await
            .map_err(CheckpointError::UploadError),
    };
    match &result {
        Ok(()) => event!(cooldown::CheckpointUploadFinished {
            success: true,
            error_string: None
        }),
        Err(e) => event!(cooldown::CheckpointUploadFinished {
            success: false,
            error_string: Some(e.to_string())
        }),
    }
    result
}

type CheckpointAndEvalsHandle = JoinHandle<
    Result<
        (
            RunningEvals,
            Option<JoinHandle<Result<(), CheckpointError>>>,
        ),
        CheckpointError,
    >,
>;

#[derive(Debug)]
pub struct CooldownStep {
    checkpointing_and_evals: CheckpointAndEvalsHandle,
}

impl CooldownStep {
    pub async fn finish(
        self,
    ) -> Result<
        (
            RunningEvals,
            Option<JoinHandle<Result<(), CheckpointError>>>,
        ),
        CooldownError,
    > {
        let (running_evals, upload_handle) = self
            .checkpointing_and_evals
            .await
            .map_err(|_| CooldownError::CheckpointThreadCrashed)??;

        Ok((running_evals, upload_handle))
    }
}
