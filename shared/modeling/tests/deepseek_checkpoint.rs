use std::{
    collections::HashMap,
    path::PathBuf,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use aether_core::{
    BatchId, CancellableBarrier, ClosedInterval, ConstantLR, LearningRateSchedule,
    OptimizerDefinition,
};
use aether_modeling::{
    save_tensors_into_safetensors, AttentionImplementation, Batch, BatchData, BatchDataCPU,
    CausalLM, Deepseek, DeepseekConfig, DeepseekForCausalLM, LocalTrainer, ModelLoadError,
    ParallelModels, PretrainedSource, Trainer,
};
use tch::{nn, COptimizer, Device, Kind, Tensor};
use tokio_util::sync::CancellationToken;

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

fn rms_norm(x: &Tensor, weight: &Tensor, eps: f64) -> Tensor {
    let variance = x.pow_tensor_scalar(2).mean_dim(-1, true, Kind::Float);
    x * (variance + eps).rsqrt() * weight
}

fn linear(x: &Tensor, state: &HashMap<String, Tensor>, name: &str) -> Tensor {
    x.matmul(&state[name].transpose(0, 1))
}

fn rotate_half(x: &Tensor) -> Tensor {
    let half = x.size()[x.dim() - 1] / 2;
    Tensor::cat(
        &[&x.narrow(-1, half, half).neg(), &x.narrow(-1, 0, half)],
        -1,
    )
}

fn apply_reference_rope(x: &Tensor) -> Tensor {
    let sequence_length = x.size()[2];
    let positions = Tensor::arange(sequence_length, (Kind::Float, Device::Cpu)).reshape([
        1,
        1,
        sequence_length,
        1,
    ]);
    x * positions.cos() + rotate_half(x) * positions.sin()
}

fn independent_deepseek_logits(state: &HashMap<String, Tensor>, tokens: &Tensor) -> Tensor {
    const BATCH: i64 = 2;
    const SEQUENCE: i64 = 4;
    const HIDDEN: i64 = 8;
    const HEADS: i64 = 2;
    const QK_DIM: i64 = 4;
    const ROPE_DIM: i64 = 2;
    const VALUE_DIM: i64 = 2;
    const KV_RANK: i64 = 4;

    let mut hidden = state["model.embed_tokens.weight"]
        .index_select(0, &tokens.view(-1))
        .reshape([BATCH, SEQUENCE, HIDDEN]);
    let attention_input = rms_norm(
        &hidden,
        &state["model.layers.0.input_layernorm.weight"],
        1e-5,
    );

    let query = linear(
        &attention_input,
        state,
        "model.layers.0.self_attn.q_proj.weight",
    )
    .reshape([BATCH, SEQUENCE, HEADS, QK_DIM])
    .transpose(1, 2);
    let query_nope = query.narrow(-1, 0, QK_DIM - ROPE_DIM);
    let query_rope = query.narrow(-1, QK_DIM - ROPE_DIM, ROPE_DIM);

    let compressed = linear(
        &attention_input,
        state,
        "model.layers.0.self_attn.kv_a_proj_with_mqa.weight",
    );
    let compressed_kv = compressed.narrow(-1, 0, KV_RANK);
    let key_rope = compressed
        .narrow(-1, KV_RANK, ROPE_DIM)
        .reshape([BATCH, SEQUENCE, 1, ROPE_DIM])
        .expand([BATCH, SEQUENCE, HEADS, ROPE_DIM], true)
        .transpose(1, 2);
    let compressed_kv = rms_norm(
        &compressed_kv,
        &state["model.layers.0.self_attn.kv_a_layernorm.weight"],
        1e-5,
    );
    let key_value = linear(
        &compressed_kv,
        state,
        "model.layers.0.self_attn.kv_b_proj.weight",
    )
    .reshape([BATCH, SEQUENCE, HEADS, QK_DIM - ROPE_DIM + VALUE_DIM])
    .transpose(1, 2);
    let key_nope = key_value.narrow(-1, 0, QK_DIM - ROPE_DIM);
    let values = key_value.narrow(-1, QK_DIM - ROPE_DIM, VALUE_DIM);

    let query = Tensor::cat(&[&query_nope, &apply_reference_rope(&query_rope)], -1);
    let key = Tensor::cat(&[&key_nope, &apply_reference_rope(&key_rope)], -1);
    let scores = query.matmul(&key.transpose(-2, -1)) / (QK_DIM as f64).sqrt();
    let causal_mask = Tensor::ones([SEQUENCE, SEQUENCE], (Kind::Float, Device::Cpu))
        .tril(0)
        .eq(0.0)
        .reshape([1, 1, SEQUENCE, SEQUENCE]);
    let attention = scores
        .masked_fill(&causal_mask, f64::NEG_INFINITY)
        .softmax(-1, Kind::Float)
        .matmul(&values)
        .transpose(1, 2)
        .contiguous()
        .reshape([BATCH, SEQUENCE, HEADS * VALUE_DIM]);
    hidden += linear(&attention, state, "model.layers.0.self_attn.o_proj.weight");

    let mlp_input = rms_norm(
        &hidden,
        &state["model.layers.0.post_attention_layernorm.weight"],
        1e-5,
    );
    let gate = linear(&mlp_input, state, "model.layers.0.mlp.gate_proj.weight").silu();
    let up = linear(&mlp_input, state, "model.layers.0.mlp.up_proj.weight");
    hidden += linear(&(gate * up), state, "model.layers.0.mlp.down_proj.weight");
    let hidden = rms_norm(&hidden, &state["model.norm.weight"], 1e-5);
    linear(&hidden, state, "lm_head.weight")
}

fn independent_causal_loss(logits: &Tensor, labels: &Tensor) -> Tensor {
    let sequence_length = logits.size()[1];
    let shift_logits = logits.narrow(1, 0, sequence_length - 1);
    let shift_labels = labels.narrow(1, 1, sequence_length - 1);
    shift_logits
        .log_softmax(-1, Kind::Float)
        .gather(-1, &shift_labels.unsqueeze(-1), false)
        .squeeze_dim(-1)
        .neg()
        .mean(Kind::Float)
}

fn oracle_batch() -> Batch {
    let rows = [[0, 2, 4, 6], [1, 3, 5, 7]];
    Batch {
        id: BatchId(ClosedInterval { start: 0, end: 1 }),
        data: BatchData::CPU(
            rows.into_iter()
                .map(|row| BatchDataCPU {
                    input_ids: row.to_vec(),
                    labels: Some(row.to_vec()),
                    position_ids: Some(vec![0, 1, 2, 3]),
                    sequence_lengths: Some(vec![4]),
                })
                .collect(),
        ),
    }
}

fn schedule() -> LearningRateSchedule {
    LearningRateSchedule::Constant(ConstantLR::new(0.01, 0, 0.0))
}

fn adamw(model: &dyn CausalLM) -> COptimizer {
    let mut optimizer = COptimizer::adamw(0.1, 0.9, 0.95, 0.01, 1e-8, false).expect("create AdamW");
    for variable in model.trainable_variables() {
        optimizer
            .add_parameters(&variable.logical_tensor(), 0)
            .expect("add model parameter");
    }
    optimizer
}

fn direct_train_step(model: &DeepseekForCausalLM, optimizer: &mut COptimizer) -> f32 {
    model.prepare_for_training();
    for variable in model.trainable_variables() {
        variable.zero_grad();
    }
    let batch = oracle_batch().gpu(Device::Cpu);
    let BatchData::GPU(data) = batch.data else {
        unreachable!("batch converted to tensor data")
    };
    let (_, loss) = model.forward(
        &data.input_ids,
        data.labels.as_ref(),
        data.position_ids.as_ref(),
        data.sequence_lengths.as_ref(),
        None,
        Some(1.0),
    );
    let loss = loss.expect("DeepSeek training loss");
    let value = loss.double_value(&[]) as f32;
    loss.backward();
    optimizer
        .set_learning_rate(schedule().get_lr(0))
        .expect("set learning rate");
    optimizer.step().expect("AdamW step");
    optimizer.zero_grad().expect("zero optimizer gradients");
    value
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

#[test]
fn tiny_deepseek_forward_matches_independent_reference() {
    let config = parse_config(dense_config_json(false));
    let state = state_for(&config);
    let model = load_model(config, &state);
    let tokens = Tensor::from_slice2(&[[0_i64, 2, 4, 6], [1, 3, 5, 7]]);

    let (actual, loss) = model.forward(&tokens, None, None, None, None, None);
    let expected = independent_deepseek_logits(&state, &tokens);

    assert!(loss.is_none());
    assert!(
        actual
            .expect("DeepSeek logits")
            .allclose(&expected, 1e-5, 1e-5, false),
        "DeepSeek logits differ from independent MLA reference"
    );
}

#[test]
fn tiny_deepseek_loss_matches_independent_negative_log_likelihood() {
    let config = parse_config(dense_config_json(false));
    let state = state_for(&config);
    let model = load_model(config, &state);
    let tokens = Tensor::from_slice2(&[[0_i64, 2, 4, 6], [1, 3, 5, 7]]);
    let expected_logits = independent_deepseek_logits(&state, &tokens);
    let expected_loss = independent_causal_loss(&expected_logits, &tokens);

    let (_, actual_loss) = model.forward(&tokens, Some(&tokens), None, None, None, None);

    assert!(
        actual_loss
            .expect("DeepSeek loss")
            .allclose(&expected_loss, 1e-6, 1e-6, false),
        "DeepSeek loss differs from independent negative log-likelihood"
    );
}

#[test]
fn tiny_deepseek_direct_optimizer_matches_local_trainer() {
    let config = parse_config(dense_config_json(false));
    let initial = state_for(&config);
    let direct = load_model(config.clone(), &initial);
    let mut direct_optimizer = adamw(&direct);
    let expected_loss = direct_train_step(&direct, &mut direct_optimizer);
    let expected_state = snapshot(&direct);

    let trainer: Trainer = LocalTrainer::new(
        ParallelModels {
            models: vec![Box::new(load_model(config, &initial)) as Box<dyn CausalLM>],
            barrier: Arc::new(CancellableBarrier::new(1)),
            data_parallel: None,
        },
        schedule(),
        OptimizerDefinition::AdamW {
            betas: [0.9, 0.95],
            weight_decay: 0.01,
            eps: 1e-8,
            clip_grad_norm: None,
        },
        2,
        None,
        false,
    )
    .into();
    let output = trainer
        .train(
            0,
            oracle_batch(),
            None,
            false,
            vec![],
            None,
            CancellationToken::new(),
        )
        .expect("trainer forward/backward");
    assert!((output.loss - expected_loss).abs() < 1e-5);
    let mut trainer = output
        .trainer
        .optimize(0, None, None)
        .expect("trainer AdamW step");
    let actual_state = trainer.extract().expect("extract trainer state");

    assert_eq!(actual_state.len(), expected_state.len());
    for (name, expected) in expected_state {
        assert!(
            actual_state[&name].allclose(&expected, 1e-5, 1e-5, false),
            "trainer parameter {name} differs from direct AdamW"
        );
    }
}
