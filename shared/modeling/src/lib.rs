mod attention;
mod auto_config;
mod auto_model;
mod auto_tokenizer;
mod batcher;
mod causal_language_model;
mod device_utils;
mod distro;
mod dummy;
mod fp32_gradient_accumulator;
mod models;
mod muon;
mod optimizer;
mod parallelism;
#[cfg(feature = "python")]
mod python_causal_lm;
#[cfg(feature = "python")]
mod python_distributed_causal_lm;
#[cfg(feature = "python")]
mod python_distributed_trainer;
mod rms_norm;
mod rope;
mod safetensor_utils;
mod sampling;
mod token_output_stream;
mod trainer;
mod variable;

pub use attention::CausalSelfAttention;
pub use auto_config::{AttentionImplementation, AutoConfig, ModelLoadError, PretrainedSource};
pub use auto_model::auto_model_for_causal_lm_from_pretrained;
pub use auto_tokenizer::{auto_tokenizer, AutoTokenizerError};
pub use batcher::Batcher;
pub use causal_language_model::{
    CausalLM, CausalLanguageModel, EosToks, LanguageModelBuilder, LanguageModelConfig,
    LanguageModelForward, VariableRole,
};
pub use device_utils::{get_optimal_devices, Devices};
pub use distro::{
    distro_result_manifest, CompressDCT, Distro, DistroResult, DistroResultMetadata, TransformDCT,
};
pub use dummy::{get_dummy_parameters, DummyModel};
pub use fp32_gradient_accumulator::Fp32GradientAccumulator;
pub use models::*;
pub use muon::MuonOptimizer;
pub use optimizer::Optimizer;
pub use parallelism::{
    unsharded_cpu_trainable_variables, unsharded_cpu_variables, AllReduce, ColumnParallelLinear,
    Communicator, CommunicatorId, CudaSynchronize, ParallelExpandHeads, ParallelismConfig,
    ReduceType, RowParallelLinear,
};
#[cfg(feature = "python")]
pub use python_causal_lm::{
    PythonCausalLM, PythonCausalLMError, PythonLoraConfig, PythonModelConfig,
};
#[cfg(feature = "python")]
pub use python_distributed_causal_lm::{
    PythonDistributedCausalLM, PythonDistributedCausalLMError, TorchDistributedCommunicator,
};
#[cfg(feature = "python")]
pub use python_distributed_trainer::{
    NopBarrier, PythonDistributedTrainer, PythonDistributedTrainerError,
};
pub use rms_norm::RMSNorm;
pub use rope::{default_rope, rotate_half, yarn_get_mscale, RoPECache, RoPEConfig, RoPEType};
pub use safetensor_utils::{
    load_safetensors_into_variables, peft_lora_tensor_name, save_lora_adapter_into_safetensors,
    save_tensors_into_safetensors, AetherAdapterMetadata, LoadSafetensorsError, LoraAdapterConfig,
    SaveSafetensorsError,
};
pub use sampling::{LogitsProcessor, Sampling};
pub use token_output_stream::TokenOutputStream;
pub use trainer::{
    ApplyDistroResultError, Batch, BatchData, BatchDataCPU, BatchDataGPU, DataParallel,
    LocalTrainer, ParallelModels, TrainOutput, Trainer, TrainerThreadCommunicationError,
};
pub use variable::{
    variable_manifest, variable_manifest_digest, StableVarStoreIterator, StableVariableIterator,
    Variable, VariableManifestEntry,
};

#[allow(unused)]
pub fn set_torch_rng_seed() {
    use rand::Rng;

    let seed: i64 = rand::rng().random();
    tch::manual_seed(seed);
    println!("torch seed set to: {seed}");
}

pub fn set_suggested_env_vars() {
    std::env::set_var("NCCL_P2P_DIRECT_DISABLE", "1");
    std::env::set_var("NCCL_LAUNCH_MODE", "GROUP");
}

#[cfg(feature = "python")]
pub fn init_embedded_python() -> std::io::Result<()> {
    use pyo3::types::PyAnyMethods;

    const DEFAULT_TRITON_HOME: &str = "/tmp/aether-triton";
    let set_default_triton_home = std::env::var_os("TRITON_HOME").is_none();

    if set_default_triton_home {
        std::fs::create_dir_all(DEFAULT_TRITON_HOME)?;
    }

    pyo3::prepare_freethreaded_python();
    pyo3::Python::with_gil(|py| -> pyo3::PyResult<()> {
        if set_default_triton_home {
            let os = pyo3::Python::import(py, "os")?;
            os.getattr("environ")?
                .call_method1("setdefault", ("TRITON_HOME", DEFAULT_TRITON_HOME))?;
        }
        pyo3::Python::import(py, "aether")?;
        Ok(())
    })
    .map_err(|err| std::io::Error::other(err.to_string()))
}
