use safetensors::{slice::TensorIndexer, SafeTensors};
use serde::Serialize;
use serde_json::json;
use std::{
    collections::{HashMap, HashSet},
    io,
    ops::Bound,
    path::PathBuf,
};
use tch::{
    nn::{Shard, VarStore},
    Device, Kind, Tensor,
};
use thiserror::Error;

const MAX_SAFETENSOR_PART_SIZE: usize = 1024 * 1024 * 1024 * 5;

#[derive(Error, Debug)]
pub enum LoadSafetensorsError {
    #[error("Failed to open safetensors file: {0}")]
    OpenFile(#[from] io::Error),

    #[error("Failed to deserialize safetensors: {0}")]
    Deserialize(#[from] safetensors::SafeTensorError),

    #[error("failed to perform tensor operation: {0}")]
    TchError(#[from] tch::TchError),

    #[error(
        "Cannot shard tensor {name} of shape {size:?} along dimension {dim} into {world_size} parts"
    )]
    CantShard {
        name: String,
        size: Vec<i64>,
        dim: usize,
        world_size: usize,
    },

    #[error("Failed to slice tensor {0}")]
    FailedToSlice(String),

    #[error("Checkpoint missing the following variables: {0:?}")]
    MissingVariables(HashSet<String>),
}

pub fn load_safetensors_into_variables(
    vs: &mut VarStore,
    repo_files: &[PathBuf],
) -> Result<(), LoadSafetensorsError> {
    let _no_grad = tch::no_grad_guard();
    let mut unmatched = vs.variables().keys().cloned().collect::<HashSet<_>>();
    for path in repo_files.iter().filter(|x| {
        x.extension()
            .is_some_and(|y| y.eq_ignore_ascii_case("safetensors"))
    }) {
        let file = std::fs::File::open(path)?;
        // SAFETY: the mapping is read-only, scoped to this function, and is
        // only used to parse immutable safetensors bytes before `file`/`content`
        // are dropped. Concurrent mutation of checkpoint files is outside the
        // supported loading contract.
        let content = unsafe { memmap2::MmapOptions::new().map(&file)? };
        let safetensors = SafeTensors::deserialize(&content)?;
        let mut variables = vs.variables_.lock().unwrap();
        let shards = variables.shards.clone();
        for (name, var) in variables.named_variables.iter_mut() {
            if let Ok(view) = safetensors.tensor(name) {
                let mut size: Vec<i64> = view.shape().iter().map(|&x| x as i64).collect();
                let kind: Kind = view.dtype().try_into()?;

                if let Some(Shard {
                    dim,
                    rank,
                    world_size,
                }) = shards.get(name)
                {
                    let (dim, rank, world_size) = (*dim, *rank, *world_size);
                    let total_size = size[dim];
                    if total_size % (world_size as i64) != 0 {
                        return Err(LoadSafetensorsError::CantShard {
                            name: name.clone(),
                            size,
                            dim,
                            world_size,
                        });
                    }
                    let block_size = total_size / (world_size as i64);
                    let start = (rank as i64) * block_size;
                    let stop = ((rank + 1) as i64) * block_size;

                    let slices: Vec<TensorIndexer> = (0..view.shape().len())
                        .map(|i| {
                            if i == dim {
                                TensorIndexer::Narrow(
                                    Bound::Included(start as usize),
                                    Bound::Excluded(stop as usize),
                                )
                            } else {
                                TensorIndexer::Narrow(Bound::Unbounded, Bound::Unbounded)
                            }
                        })
                        .collect();
                    let data_iterator = view
                        .sliced_data(&slices)
                        .map_err(|_| LoadSafetensorsError::FailedToSlice(name.clone()))?;
                    let data: Vec<u8> = data_iterator.flatten().cloned().collect();
                    size[dim] = block_size;
                    // SAFETY: `from_blob` borrows `data` only for the duration
                    // of `src_tensor`; `f_copy_` synchronously copies into the
                    // destination tensor before `data` is dropped.
                    let src_tensor =
                        unsafe { Tensor::from_blob(data.as_ptr(), &size, &[], kind, Device::Cpu) };
                    var.f_copy_(&src_tensor)?;
                } else {
                    // SAFETY: `from_blob` borrows the safetensors view backed by
                    // `content`; `f_copy_` synchronously copies into the
                    // destination tensor while the mmap is still alive.
                    let src_tensor = unsafe {
                        Tensor::from_blob(view.data().as_ptr(), &size, &[], kind, Device::Cpu)
                    };
                    var.f_copy_(&src_tensor)?;
                }
                unmatched.remove(name);
            }
        }
    }
    if !unmatched.is_empty() {
        return Err(LoadSafetensorsError::MissingVariables(unmatched));
    }
    Ok(())
}

#[derive(Default)]
struct FilePart {
    tensors: Vec<(String, Tensor)>,
    size: usize,
}

#[derive(Error, Debug)]
pub enum SaveSafetensorsError {
    #[error("No tensors to save")]
    NoTensors,

    #[error("Failed to create directory {0}: {1}")]
    CreateDir(PathBuf, io::Error),

    #[error(
        "Tensor {name} too big to save to file -- it's {size} bytes while we have a max of {MAX_SAFETENSOR_PART_SIZE} bytes"
    )]
    TensorTooBig { name: String, size: usize },

    #[error("Torch error: {0}")]
    TchError(#[from] tch::TchError),

    #[error("Failed to write: {0}")]
    Write(#[from] io::Error),

    #[error("LoRA tensor names {first} and {second} both convert to {converted}")]
    DuplicateConvertedTensorName {
        first: String,
        second: String,
        converted: String,
    },
}

#[derive(Clone, Debug, Serialize)]
pub struct LoraAdapterConfig {
    pub base_model_name_or_path: String,
    pub r: u32,
    pub lora_alpha: f32,
    pub lora_dropout: f32,
    pub target_modules: String,
    pub bias: String,
    pub task_type: String,
    pub peft_type: String,
    pub inference_mode: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct AetherAdapterMetadata {
    pub artifact_type: String,
    pub format_version: u32,
    pub run_id: String,
    pub epoch: u32,
    pub step: u32,
    pub trainable_manifest_digest: String,
}

impl LoraAdapterConfig {
    pub fn new(base_model_name_or_path: String, rank: u32, alpha: f32, dropout: f32) -> Self {
        Self {
            base_model_name_or_path,
            r: rank,
            lora_alpha: alpha,
            lora_dropout: dropout,
            target_modules: "all-linear".to_string(),
            bias: "none".to_string(),
            task_type: "CAUSAL_LM".to_string(),
            peft_type: "LORA".to_string(),
            inference_mode: true,
        }
    }
}

/// Converts PEFT's internal named-parameter form to its serialized adapter form.
pub fn peft_lora_tensor_name(name: &str) -> String {
    name.replace(".lora_A.default.", ".lora_A.")
        .replace(".lora_B.default.", ".lora_B.")
}

pub fn save_lora_adapter_into_safetensors(
    tensors: HashMap<String, Tensor>,
    dir: PathBuf,
    config: &LoraAdapterConfig,
    metadata: &AetherAdapterMetadata,
) -> Result<Vec<PathBuf>, SaveSafetensorsError> {
    if tensors.is_empty() {
        return Err(SaveSafetensorsError::NoTensors);
    }
    std::fs::create_dir_all(dir.clone())
        .map_err(|e| SaveSafetensorsError::CreateDir(dir.clone(), e))?;

    let mut converted = HashMap::with_capacity(tensors.len());
    let mut original_names = HashMap::with_capacity(tensors.len());
    for (name, tensor) in tensors {
        let converted_name = peft_lora_tensor_name(&name);
        if let Some(first) = original_names.insert(converted_name.clone(), name.clone()) {
            return Err(SaveSafetensorsError::DuplicateConvertedTensorName {
                first,
                second: name,
                converted: converted_name,
            });
        }
        converted.insert(converted_name, tensor);
    }

    let mut tensors = converted.into_iter().collect::<Vec<_>>();
    tensors.sort_by(|a, b| a.0.cmp(&b.0));
    for (name, tensor) in &tensors {
        let size = tensor.numel() * tensor.kind().elt_size_in_bytes();
        if size > MAX_SAFETENSOR_PART_SIZE {
            return Err(SaveSafetensorsError::TensorTooBig {
                name: name.clone(),
                size,
            });
        }
    }

    let tensor_path = dir.join("adapter_model.safetensors");
    let safetensor_metadata = HashMap::from([
        ("format".to_string(), "pt".to_string()),
        (
            "aether_artifact_type".to_string(),
            "lora_adapter".to_string(),
        ),
    ]);
    Tensor::write_safetensors(&tensors, tensor_path.clone(), &Some(safetensor_metadata))?;

    let config_path = dir.join("adapter_config.json");
    let config_json = serde_json::to_string_pretty(config)
        .map_err(|error| io::Error::other(error.to_string()))?;
    std::fs::write(&config_path, config_json)?;
    let metadata_path = dir.join("aether_adapter.json");
    let metadata_json = serde_json::to_string_pretty(metadata)
        .map_err(|error| io::Error::other(error.to_string()))?;
    std::fs::write(&metadata_path, metadata_json)?;
    Ok(vec![tensor_path, config_path, metadata_path])
}

pub fn save_tensors_into_safetensors(
    tensors: HashMap<String, Tensor>,
    dir: PathBuf,
) -> Result<Vec<PathBuf>, SaveSafetensorsError> {
    if tensors.is_empty() {
        return Err(SaveSafetensorsError::NoTensors);
    }
    std::fs::create_dir_all(dir.clone())
        .map_err(|e| SaveSafetensorsError::CreateDir(dir.clone(), e))?;
    let mut file_parts = vec![FilePart::default()];
    let mut tensors = tensors.into_iter().collect::<Vec<_>>();
    tensors.sort_by(|a, b| a.0.cmp(&b.0)); // sort so we have stable ordering for chunking
    for (name, tensor) in tensors {
        let size = tensor.numel() * tensor.kind().elt_size_in_bytes();
        if size > MAX_SAFETENSOR_PART_SIZE {
            return Err(SaveSafetensorsError::TensorTooBig { name, size });
        }
        if size + file_parts.last().unwrap().size > MAX_SAFETENSOR_PART_SIZE {
            file_parts.push(FilePart::default());
        }
        let last_part = file_parts.last_mut().unwrap();
        last_part.tensors.push((name, tensor));
        last_part.size += size;
    }
    if file_parts.len() == 1 {
        let path = dir.join("model.safetensors");
        let metadata = HashMap::from([("format".to_string(), "pt".to_string())]);
        Tensor::write_safetensors(&file_parts[0].tensors, path.clone(), &Some(metadata))?;
        Ok(vec![path])
    } else {
        let len = file_parts.len();
        let mut safetensors_index = json!({
            "metadata": {
                "total_size": file_parts.iter().fold(0, |acc, ele| acc + ele.size)
            },
            "weight_map": serde_json::Map::new(),
        });
        let paths: Result<Vec<PathBuf>, _> = file_parts
            .into_iter()
            .enumerate()
            .map(|(index, part)| {
                let filename = format!("model-{:05}-of-{:05}.safetensors", index + 1, len);
                let path = dir.join(filename.clone());
                safetensors_index
                    .get_mut("weight_map")
                    .unwrap()
                    .as_object_mut()
                    .unwrap()
                    .append(&mut serde_json::Map::from_iter(part.tensors.iter().map(
                        |(name, _)| (name.clone(), serde_json::Value::String(filename.clone())),
                    )));
                std::thread::spawn(move || {
                    let metadata = HashMap::from([("format".to_string(), "pt".to_string())]);
                    Tensor::write_safetensors(&part.tensors, path.clone(), &Some(metadata))
                        .and(Ok(path))
                })
            })
            .map(|future| future.join().unwrap())
            .collect();
        let mut paths = paths?;
        let safetensors_index_path = dir.join("model.safetensors.index.json");
        paths.push(safetensors_index_path.clone());
        std::fs::write(safetensors_index_path, safetensors_index.to_string())?;
        Ok(paths)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn converts_internal_lora_names_to_peft_names() {
        assert_eq!(
            peft_lora_tensor_name(
                "base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight"
            ),
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"
        );
        assert_eq!(
            peft_lora_tensor_name(
                "base_model.model.model.layers.0.self_attn.q_proj.lora_B.default.weight"
            ),
            "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight"
        );
    }

    #[test]
    fn adapter_config_contains_only_peft_fields() {
        let config = LoraAdapterConfig::new("org/base".to_string(), 8, 16.0, 0.1);
        let value = serde_json::to_value(config).unwrap();
        assert_eq!(value["base_model_name_or_path"], "org/base");
        assert_eq!(value["target_modules"], "all-linear");
        assert_eq!(value["peft_type"], "LORA");
        assert!(value.get("aether").is_none());
    }

    #[test]
    fn writes_peft_adapter_artifact() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("aether-lora-artifact-{unique}"));
        let tensors = HashMap::from([(
            "base_model.model.layer.lora_A.default.weight".to_string(),
            Tensor::ones([2, 2], (Kind::Float, Device::Cpu)),
        )]);
        let config = LoraAdapterConfig::new("org/base".to_string(), 2, 4.0, 0.0);
        let metadata = AetherAdapterMetadata {
            artifact_type: "lora_adapter".to_string(),
            format_version: 1,
            run_id: "run".to_string(),
            epoch: 1,
            step: 3,
            trainable_manifest_digest: "00".repeat(32),
        };

        let paths =
            save_lora_adapter_into_safetensors(tensors, dir.clone(), &config, &metadata).unwrap();
        assert_eq!(paths.len(), 3);
        let bytes = std::fs::read(dir.join("adapter_model.safetensors")).unwrap();
        let saved = SafeTensors::deserialize(&bytes).unwrap();
        assert!(saved.tensor("base_model.model.layer.lora_A.weight").is_ok());
        let config: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.join("adapter_config.json")).unwrap())
                .unwrap();
        assert_eq!(config["r"], 2);
        assert!(config.get("aether").is_none());
        let metadata: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.join("aether_adapter.json")).unwrap())
                .unwrap();
        assert_eq!(metadata["run_id"], "run");

        std::fs::remove_dir_all(dir).unwrap();
    }
}
