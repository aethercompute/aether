use std::{
    collections::HashMap,
    path::PathBuf,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use aether_modeling::{
    save_tensors_into_safetensors, AttentionImplementation, CausalLM, Deepseek, DeepseekConfig,
    DeepseekForCausalLM, ModelLoadError, PretrainedSource,
};
use tch::{nn, Device, Kind, Tensor};

fn dense_config_json(tied: bool) -> serde_json::Value {
    serde_json::json!({
        "model_type": "deepseek_v3",
        "hidden_size": 8,
        "intermediate_size": 16,
        "vocab_size": 11,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "rms_norm_eps": 0.00001,
        "rope_theta": 10000.0,
        "max_position_embeddings": 16,
        "tie_word_embeddings": tied,
        "bos_token_id": 0,
        "eos_token_id": 1,
        "rope_scaling": null,
        "q_lora_rank": null,
        "kv_lora_rank": 4,
        "qk_nope_head_dim": 2,
        "qk_rope_head_dim": 2,
        "v_head_dim": 2,
        "attention_bias": false
    })
}

fn moe_config_json() -> serde_json::Value {
    let mut config = dense_config_json(false);
    config.as_object_mut().expect("config object").extend(
        serde_json::json!({
            "n_routed_experts": 4,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 8,
            "routed_scaling_factor": 1.0,
            "n_group": 2,
            "topk_group": 1,
            "n_shared_experts": 1,
            "first_k_dense_replace": 0,
            "moe_layer_freq": 1,
            "scoring_func": "softmax",
            "topk_method": "greedy",
            "norm_topk_prob": true
        })
        .as_object()
        .expect("MoE fields object")
        .clone(),
    );
    config
}

fn parse_config(value: serde_json::Value) -> DeepseekConfig {
    serde_json::from_value(value).expect("valid tiny DeepSeek config")
}

fn temp_dir(name: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time after epoch")
        .as_nanos();
    PathBuf::from("/tmp/opencode").join(format!(
        "aether-deepseek-{name}-{}-{nanos}",
        std::process::id()
    ))
}

fn write_config(value: &serde_json::Value, name: &str) -> (PathBuf, PathBuf) {
    let dir = temp_dir(name);
    std::fs::create_dir_all(&dir).expect("create temporary checkpoint directory");
    let path = dir.join("config.json");
    std::fs::write(&path, serde_json::to_vec(value).expect("serialize config"))
        .expect("write config");
    (dir, path)
}

fn deterministic_tensor(name: &str, shape: &[i64]) -> Tensor {
    if name.contains("norm.weight") || name.contains("layernorm.weight") {
        return Tensor::ones(shape, (Kind::Float, Device::Cpu));
    }
    let elements = shape.iter().product::<i64>();
    (Tensor::arange(elements, (Kind::Float, Device::Cpu)).reshape(shape) / 97.0) - 0.2
}

fn state_for(config: &DeepseekConfig) -> HashMap<String, Tensor> {
    let variables = nn::VarStore::new(Device::Cpu);
    let _model = Deepseek::new(
        variables.root(),
        config,
        AttentionImplementation::Eager,
        None,
    );
    if !config.tie_word_embeddings {
        let _lm_head = nn::linear(
            &variables.root() / "lm_head",
            config.hidden_size as i64,
            config.vocab_size as i64,
            nn::LinearConfig {
                bias: false,
                ..Default::default()
            },
        );
    }
    variables
        .variables()
        .into_iter()
        .map(|(name, tensor)| {
            let value = deterministic_tensor(&name, &tensor.size());
            (name, value)
        })
        .collect()
}

fn clone_state(state: &HashMap<String, Tensor>) -> HashMap<String, Tensor> {
    state
        .iter()
        .map(|(name, tensor)| (name.clone(), tensor.shallow_clone()))
        .collect()
}

#[allow(clippy::arc_with_non_send_sync)]
fn load_model(config: DeepseekConfig, state: &HashMap<String, Tensor>) -> DeepseekForCausalLM {
    DeepseekForCausalLM::from_pretrained(
        &PretrainedSource::ConfigAndTensors(config, Arc::new(clone_state(state))),
        Some(Kind::Float),
        Some(AttentionImplementation::Eager),
        Some(Device::Cpu),
        None,
        None,
    )
    .expect("load tiny DeepSeek model")
}

fn snapshot(model: &dyn CausalLM) -> HashMap<String, Tensor> {
    model
        .state_variables()
        .map(|variable| {
            (
                variable.name().to_string(),
                variable.gather_full_tensor().to_device(Device::Cpu).copy(),
            )
        })
        .collect()
}

fn assert_state_equal(actual: &HashMap<String, Tensor>, expected: &HashMap<String, Tensor>) {
    assert_eq!(actual.len(), expected.len());
    for (name, expected) in expected {
        let actual = &actual[name];
        assert!(
            actual.allclose(expected, 0.0, 0.0, false),
            "parameter {name} changed"
        );
    }
}

fn forward_with_loss(model: &dyn CausalLM) -> (Tensor, Tensor) {
    let tokens = Tensor::from_slice2(&[[0_i64, 2, 4, 6], [1, 3, 5, 7]]);
    let (logits, loss) = model.forward(&tokens, Some(&tokens), None, None, None, None);
    (
        logits.expect("DeepSeek forward returns logits"),
        loss.expect("DeepSeek forward returns loss"),
    )
}

#[test]
fn dense_deepseek_configuration_loads_from_repo_files() {
    let (dir, config_path) = write_config(&dense_config_json(false), "dense-config");
    let source = PretrainedSource::<DeepseekConfig>::RepoFiles(vec![config_path]);

    let config = source.get_config().expect("load dense config");

    assert_eq!(config.hidden_size, 8);
    assert_eq!(config.num_hidden_layers, 1);
    assert_eq!(config.n_routed_experts, None);
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn moe_deepseek_configuration_loads_from_repo_files() {
    let (dir, config_path) = write_config(&moe_config_json(), "moe-config");
    let source = PretrainedSource::<DeepseekConfig>::RepoFiles(vec![config_path]);

    let config = source.get_config().expect("load MoE config");

    assert_eq!(config.n_routed_experts, Some(4));
    assert_eq!(config.num_experts_per_tok, Some(2));
    assert_eq!(config.first_k_dense_replace, Some(0));
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn malformed_deepseek_configuration_fields_are_rejected() {
    for (name, value) in [
        ("missing-hidden-size", {
            let mut value = dense_config_json(false);
            value.as_object_mut().unwrap().remove("hidden_size");
            value
        }),
        ("invalid-head-count", {
            let mut value = dense_config_json(false);
            value["num_attention_heads"] = serde_json::json!("two");
            value
        }),
        ("invalid-scoring-function", {
            let mut value = moe_config_json();
            value["scoring_func"] = serde_json::json!("invalid");
            value
        }),
    ] {
        let (dir, config_path) = write_config(&value, name);
        let source = PretrainedSource::<DeepseekConfig>::RepoFiles(vec![config_path]);
        assert!(matches!(
            source.get_config(),
            Err(ModelLoadError::FailedToParseConfig(_))
        ));
        let _ = std::fs::remove_dir_all(dir);
    }
}

#[test]
fn tied_and_untied_deepseek_embeddings_load_with_expected_state() {
    for tied in [false, true] {
        let config = parse_config(dense_config_json(tied));
        let state = state_for(&config);
        let model = load_model(config, &state);
        let loaded = snapshot(&model);

        assert_state_equal(&loaded, &state);
        assert_eq!(loaded.contains_key("lm_head.weight"), !tied);
        if tied {
            assert!(model
                .lm_head
                .allclose(&loaded["model.embed_tokens.weight"], 0.0, 0.0, false));
        }
    }
}

#[test]
fn deepseek_safetensors_checkpoint_reloads_identical_state() {
    let config = parse_config(dense_config_json(false));
    let state = state_for(&config);
    let original = load_model(config.clone(), &state);
    let (expected_logits, expected_loss) = forward_with_loss(&original);
    let dir = temp_dir("safetensors-reload");
    std::fs::create_dir_all(&dir).expect("create checkpoint directory");
    let config_path = dir.join("config.json");
    std::fs::write(
        &config_path,
        serde_json::to_vec(&config).expect("serialize config"),
    )
    .expect("write config");
    let mut files = save_tensors_into_safetensors(clone_state(&state), dir.clone())
        .expect("save DeepSeek state");
    files.push(config_path);

    let model = DeepseekForCausalLM::from_pretrained(
        &PretrainedSource::RepoFiles(files),
        Some(Kind::Float),
        Some(AttentionImplementation::Eager),
        Some(Device::Cpu),
        None,
        None,
    )
    .expect("reload DeepSeek safetensors checkpoint");

    assert_state_equal(&snapshot(&model), &state);
    let (actual_logits, actual_loss) = forward_with_loss(&model);
    assert!(actual_logits.allclose(&expected_logits, 1e-6, 1e-6, false));
    assert!(actual_loss.allclose(&expected_loss, 1e-6, 1e-6, false));
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
#[allow(clippy::arc_with_non_send_sync)]
fn deepseek_checkpoint_rejects_missing_and_extra_keys_without_panicking() {
    let config = parse_config(dense_config_json(false));
    let state = state_for(&config);

    let mut missing = clone_state(&state);
    let missing_name = missing.keys().next().expect("state is non-empty").clone();
    missing.remove(&missing_name);
    let missing_result = DeepseekForCausalLM::from_pretrained(
        &PretrainedSource::ConfigAndTensors(config.clone(), Arc::new(missing)),
        Some(Kind::Float),
        Some(AttentionImplementation::Eager),
        Some(Device::Cpu),
        None,
        None,
    );
    assert!(matches!(
        missing_result,
        Err(ModelLoadError::LoadTensorError(keys)) if keys.contains(&missing_name)
    ));

    let mut extra = clone_state(&state);
    extra.insert(
        "unexpected.weight".into(),
        Tensor::zeros([1], (Kind::Float, Device::Cpu)),
    );
    let extra_result = DeepseekForCausalLM::from_pretrained(
        &PretrainedSource::ConfigAndTensors(config, Arc::new(extra)),
        Some(Kind::Float),
        Some(AttentionImplementation::Eager),
        Some(Device::Cpu),
        None,
        None,
    );
    assert!(matches!(
        extra_result,
        Err(ModelLoadError::LoadTensorError(keys)) if keys.contains("unexpected.weight")
    ));
}
