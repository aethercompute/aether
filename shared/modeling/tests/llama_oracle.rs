use std::{collections::HashMap, path::PathBuf, sync::Arc, time::SystemTime};

use aether_core::{BatchId, CancellableBarrier, ClosedInterval, ConstantLR, LearningRateSchedule};
use aether_modeling::{
    save_tensors_into_safetensors, AttentionImplementation, Batch, BatchData, BatchDataCPU,
    CausalLM, EosToks, LlamaConfig, LlamaForCausalLM, LocalTrainer, ParallelModels,
    PretrainedSource, Trainer,
};
use tch::{COptimizer, Device, Kind, Tensor};
use tokio_util::sync::CancellationToken;

const VOCAB_SIZE: i64 = 11;
const HIDDEN_SIZE: i64 = 8;
const INTERMEDIATE_SIZE: i64 = 16;
const SEQ_LEN: usize = 5;

fn tiny_llama_config() -> LlamaConfig {
    LlamaConfig {
        hidden_size: HIDDEN_SIZE as usize,
        intermediate_size: INTERMEDIATE_SIZE as usize,
        vocab_size: VOCAB_SIZE as usize,
        num_hidden_layers: 1,
        num_attention_heads: 2,
        num_key_value_heads: Some(2),
        rms_norm_eps: 1e-5,
        rope_theta: 10_000.0,
        bos_token_id: Some(0),
        eos_token_id: Some(EosToks::Single(1)),
        rope_scaling: None,
        max_position_embeddings: 16,
        tie_word_embeddings: false,
        attention_bias: None,
    }
}

fn tiny_tied_llama_config() -> LlamaConfig {
    LlamaConfig {
        tie_word_embeddings: true,
        ..tiny_llama_config()
    }
}

fn tensor_for(name: &str, shape: &[i64]) -> Tensor {
    if name.contains("norm.weight") || name.contains("layernorm.weight") {
        return Tensor::ones(shape, (Kind::Float, Device::Cpu));
    }

    let scale = match name {
        "model.embed_tokens.weight" => 97.0,
        "lm_head.weight" => 89.0,
        name if name.contains("q_proj") => 83.0,
        name if name.contains("k_proj") => 79.0,
        name if name.contains("v_proj") => 73.0,
        name if name.contains("o_proj") => 71.0,
        name if name.contains("gate_proj") => 67.0,
        name if name.contains("up_proj") => 61.0,
        name if name.contains("down_proj") => 59.0,
        _ => 53.0,
    };
    let numel = shape.iter().product::<i64>();
    (Tensor::arange(numel, (Kind::Float, Device::Cpu)).reshape(shape) / scale) - 0.25
}

fn tiny_llama_state() -> HashMap<String, Tensor> {
    let specs: [(&str, &[i64]); 12] = [
        ("lm_head.weight", &[VOCAB_SIZE, HIDDEN_SIZE]),
        ("model.embed_tokens.weight", &[VOCAB_SIZE, HIDDEN_SIZE]),
        ("model.layers.0.input_layernorm.weight", &[HIDDEN_SIZE]),
        (
            "model.layers.0.mlp.down_proj.weight",
            &[HIDDEN_SIZE, INTERMEDIATE_SIZE],
        ),
        (
            "model.layers.0.mlp.gate_proj.weight",
            &[INTERMEDIATE_SIZE, HIDDEN_SIZE],
        ),
        (
            "model.layers.0.mlp.up_proj.weight",
            &[INTERMEDIATE_SIZE, HIDDEN_SIZE],
        ),
        (
            "model.layers.0.post_attention_layernorm.weight",
            &[HIDDEN_SIZE],
        ),
        (
            "model.layers.0.self_attn.k_proj.weight",
            &[HIDDEN_SIZE, HIDDEN_SIZE],
        ),
        (
            "model.layers.0.self_attn.o_proj.weight",
            &[HIDDEN_SIZE, HIDDEN_SIZE],
        ),
        (
            "model.layers.0.self_attn.q_proj.weight",
            &[HIDDEN_SIZE, HIDDEN_SIZE],
        ),
        (
            "model.layers.0.self_attn.v_proj.weight",
            &[HIDDEN_SIZE, HIDDEN_SIZE],
        ),
        ("model.norm.weight", &[HIDDEN_SIZE]),
    ];

    specs
        .into_iter()
        .map(|(name, shape)| (name.to_string(), tensor_for(name, shape)))
        .collect()
}

fn tiny_tied_llama_state() -> HashMap<String, Tensor> {
    let mut state = tiny_llama_state();
    state.remove("lm_head.weight");
    state
}

#[allow(clippy::arc_with_non_send_sync)]
fn new_llama() -> LlamaForCausalLM {
    let source =
        PretrainedSource::ConfigAndTensors(tiny_llama_config(), Arc::new(tiny_llama_state()));
    LlamaForCausalLM::from_pretrained(
        &source,
        Some(Kind::Float),
        Some(AttentionImplementation::Eager),
        Some(Device::Cpu),
        None,
        None,
    )
    .expect("tiny llama loads from deterministic state dict")
}

fn new_llama_from_repo_files(repo_files: Vec<PathBuf>) -> LlamaForCausalLM {
    LlamaForCausalLM::from_pretrained(
        &PretrainedSource::RepoFiles(repo_files),
        Some(Kind::Float),
        Some(AttentionImplementation::Eager),
        Some(Device::Cpu),
        None,
        None,
    )
    .expect("tiny llama reloads from safetensors repo files")
}

#[allow(clippy::arc_with_non_send_sync)]
fn new_llama_from_state(state: &HashMap<String, Tensor>) -> LlamaForCausalLM {
    let source =
        PretrainedSource::ConfigAndTensors(tiny_llama_config(), Arc::new(clone_state(state)));
    LlamaForCausalLM::from_pretrained(
        &source,
        Some(Kind::Float),
        Some(AttentionImplementation::Eager),
        Some(Device::Cpu),
        None,
        None,
    )
    .expect("tiny llama loads from supplied state dict")
}

fn schedule() -> LearningRateSchedule {
    LearningRateSchedule::Constant(ConstantLR::new(0.01, 0, 0.0))
}

fn batch() -> Batch {
    let rows = [[0, 2, 4, 6, 8], [1, 3, 5, 7, 9], [10, 8, 6, 4, 2]];
    Batch {
        id: BatchId(ClosedInterval { start: 0, end: 2 }),
        data: BatchData::CPU(
            rows.iter()
                .map(|row| BatchDataCPU {
                    input_ids: row.to_vec(),
                    labels: Some(row.to_vec()),
                    position_ids: Some((0..SEQ_LEN as i32).collect()),
                    sequence_lengths: Some(vec![SEQ_LEN as i32]),
                })
                .collect(),
        ),
    }
}

fn data_parallel_batch() -> Batch {
    let rows = [
        [0, 2, 4, 6, 8],
        [1, 3, 5, 7, 9],
        [10, 8, 6, 4, 2],
        [9, 7, 5, 3, 1],
    ];
    Batch {
        id: BatchId(ClosedInterval { start: 0, end: 3 }),
        data: BatchData::CPU(
            rows.iter()
                .map(|row| BatchDataCPU {
                    input_ids: row.to_vec(),
                    labels: Some(row.to_vec()),
                    position_ids: Some((0..SEQ_LEN as i32).collect()),
                    sequence_lengths: Some(vec![SEQ_LEN as i32]),
                })
                .collect(),
        ),
    }
}

fn split_batch_for_workers(batch: &Batch, worker_count: usize) -> Vec<Batch> {
    let BatchData::CPU(rows) = &batch.data else {
        panic!("tiny llama oracle batches must be CPU batches")
    };
    assert_eq!(rows.len() % worker_count, 0);
    let chunk_size = rows.len() / worker_count;
    rows.chunks(chunk_size)
        .enumerate()
        .map(|(worker, chunk)| Batch {
            id: BatchId(ClosedInterval {
                start: (worker * chunk_size) as u64,
                end: ((worker + 1) * chunk_size - 1) as u64,
            }),
            data: BatchData::CPU(chunk.to_vec()),
        })
        .collect()
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

fn clone_state(state: &HashMap<String, Tensor>) -> HashMap<String, Tensor> {
    state
        .iter()
        .map(|(key, tensor)| (key.clone(), tensor.shallow_clone()))
        .collect()
}

fn temp_checkpoint_dir(name: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .expect("system time after unix epoch")
        .as_nanos();
    PathBuf::from("/tmp/opencode").join(format!("aether-{name}-{}-{nanos}", std::process::id()))
}

fn assert_state_close(actual: &HashMap<String, Tensor>, expected: &HashMap<String, Tensor>) {
    let mut actual_keys = actual.keys().cloned().collect::<Vec<_>>();
    let mut expected_keys = expected.keys().cloned().collect::<Vec<_>>();
    actual_keys.sort();
    expected_keys.sort();
    assert_eq!(actual_keys, expected_keys, "state dict keys changed");

    for key in expected_keys {
        let actual_tensor = actual.get(&key).expect("actual tensor present");
        let expected_tensor = expected.get(&key).expect("expected tensor present");
        let diff = (actual_tensor - expected_tensor).abs();
        let max_abs = diff.max().double_value(&[]);
        assert!(
            actual_tensor.allclose(expected_tensor, 1e-5, 1e-5, false),
            "tensor {key} differs: max_abs={max_abs:.6e}, shape={:?}",
            actual_tensor.size()
        );
    }
}

fn assert_state_changed(initial: &HashMap<String, Tensor>, final_state: &HashMap<String, Tensor>) {
    assert!(
        initial.iter().any(|(key, initial_tensor)| {
            !initial_tensor.allclose(
                final_state.get(key).expect("final tensor present"),
                0.0,
                0.0,
                false,
            )
        }),
        "tiny llama training did not update any parameters"
    );
}

fn adamw(model: &dyn CausalLM) -> COptimizer {
    let mut optimizer =
        COptimizer::adamw(0.1, 0.9, 0.95, 0.01, 1e-8, false).expect("adamw optimizer initializes");
    for variable in model.trainable_variables() {
        optimizer
            .add_parameters(&variable.logical_tensor(), 0)
            .expect("parameter can be added to adamw");
    }
    optimizer
}

fn direct_train_step(model: &LlamaForCausalLM, optimizer: &mut COptimizer, batch: &Batch) -> f32 {
    for variable in model.trainable_variables() {
        variable.zero_grad();
    }
    let batch = batch.clone().gpu(Device::Cpu);
    let BatchData::GPU(data) = batch.data else {
        unreachable!("batch was moved to tensor form")
    };
    let (_, loss) = model.forward(
        &data.input_ids,
        data.labels.as_ref(),
        data.position_ids.as_ref(),
        data.sequence_lengths.as_ref(),
        None,
        Some(1.0),
    );
    let loss = loss.expect("tiny llama returns a loss");
    let loss_value = loss.double_value(&[]) as f32;
    loss.backward();
    optimizer
        .set_learning_rate(schedule().get_lr(0))
        .expect("valid learning rate");
    optimizer.step().expect("adamw step succeeds");
    optimizer.zero_grad().expect("zero gradients succeeds");
    loss_value
}

fn gradients_from_state(state: &HashMap<String, Tensor>, batch: &Batch) -> HashMap<String, Tensor> {
    let model = new_llama_from_state(state);
    for variable in model.trainable_variables() {
        variable.zero_grad();
    }
    let batch = batch.clone().gpu(Device::Cpu);
    let BatchData::GPU(data) = batch.data else {
        unreachable!("batch was moved to tensor form")
    };
    let (_, loss) = model.forward(
        &data.input_ids,
        data.labels.as_ref(),
        data.position_ids.as_ref(),
        data.sequence_lengths.as_ref(),
        None,
        Some(1.0),
    );
    loss.expect("tiny llama returns a loss").backward();

    model
        .trainable_variables()
        .map(|variable| {
            (
                variable.name().to_string(),
                variable
                    .logical_tensor()
                    .grad()
                    .to_device(Device::Cpu)
                    .copy(),
            )
        })
        .collect()
}

fn materialize_zero_grads(model: &dyn CausalLM, batch: &Batch) {
    for variable in model.trainable_variables() {
        variable.zero_grad();
    }
    let batch = batch.clone().gpu(Device::Cpu);
    let BatchData::GPU(data) = batch.data else {
        unreachable!("batch was moved to tensor form")
    };
    let (_, loss) = model.forward(
        &data.input_ids,
        data.labels.as_ref(),
        data.position_ids.as_ref(),
        data.sequence_lengths.as_ref(),
        None,
        Some(1.0),
    );
    (loss.expect("tiny llama returns a loss") * 0.0).backward();
    for variable in model.trainable_variables() {
        variable.zero_grad();
    }
}

fn simulated_data_parallel_train_step(
    model: &LlamaForCausalLM,
    optimizer: &mut COptimizer,
    batch: &Batch,
    worker_count: usize,
) {
    let state = snapshot(model);
    let worker_batches = split_batch_for_workers(batch, worker_count);
    let worker_grads = worker_batches
        .iter()
        .map(|batch| gradients_from_state(&state, batch))
        .collect::<Vec<_>>();

    materialize_zero_grads(model, worker_batches.first().expect("worker batch present"));
    for variable in model.trainable_variables() {
        let mut mean_grad = Tensor::zeros_like(
            worker_grads[0]
                .get(variable.name())
                .expect("worker gradient present"),
        );
        for grads in &worker_grads {
            mean_grad += grads.get(variable.name()).expect("worker gradient present");
        }
        mean_grad /= worker_grads.len() as f64;
        variable.set_grad(mean_grad);
    }

    optimizer
        .set_learning_rate(schedule().get_lr(0))
        .expect("valid learning rate");
    optimizer.step().expect("adamw step succeeds");
    optimizer.zero_grad().expect("zero gradients succeeds");
}

fn forward_loss(model: &dyn CausalLM, batch: &Batch) -> (Tensor, Tensor) {
    let batch = batch.clone().gpu(Device::Cpu);
    let BatchData::GPU(data) = batch.data else {
        unreachable!("batch was moved to tensor form")
    };
    let (logits, loss) = model.forward(
        &data.input_ids,
        data.labels.as_ref(),
        data.position_ids.as_ref(),
        data.sequence_lengths.as_ref(),
        None,
        None,
    );
    (
        logits.expect("tiny llama returns logits"),
        loss.expect("tiny llama returns a loss"),
    )
}

#[test]
fn tiny_llama_loads_state_and_forward_loss_is_finite() {
    let model = new_llama();
    assert_state_close(&snapshot(&model), &tiny_llama_state());

    let batch = batch().gpu(Device::Cpu);
    let BatchData::GPU(data) = batch.data else {
        unreachable!("batch was moved to tensor form")
    };
    let (logits, loss) = model.forward(
        &data.input_ids,
        data.labels.as_ref(),
        data.position_ids.as_ref(),
        data.sequence_lengths.as_ref(),
        None,
        None,
    );

    let logits = logits.expect("tiny llama returns logits");
    assert_eq!(logits.size(), [3, SEQ_LEN as i64, VOCAB_SIZE]);
    let loss = loss.expect("tiny llama returns a loss");
    assert!(loss.double_value(&[]).is_finite(), "loss must be finite");
}

#[test]
fn tiny_llama_local_trainer_matches_direct_adamw_reference() {
    let initial = tiny_llama_state();
    let reference = new_llama();
    let mut reference_optimizer = adamw(&reference);
    let expected_loss = direct_train_step(&reference, &mut reference_optimizer, &batch());

    let trainer: Trainer = LocalTrainer::new(
        ParallelModels {
            models: vec![Box::new(new_llama()) as Box<dyn CausalLM>],
            barrier: Arc::new(CancellableBarrier::new(1)),
            data_parallel: None,
        },
        schedule(),
        aether_core::OptimizerDefinition::AdamW {
            betas: [0.9, 0.95],
            weight_decay: 0.01,
            eps: 1e-8,
            clip_grad_norm: None,
        },
        3,
        None,
        false,
    )
    .into();

    let output = trainer
        .train(
            0,
            batch(),
            None,
            false,
            vec![],
            None,
            CancellationToken::new(),
        )
        .expect("tiny llama trainer train succeeds");
    assert!(
        (output.loss - expected_loss).abs() < 1e-5,
        "loss differs: actual={}, expected={expected_loss}",
        output.loss
    );
    let mut trainer = output
        .trainer
        .optimize(0, None, None)
        .expect("tiny llama optimizer step succeeds");

    let actual = trainer.extract().expect("extract tiny llama state");
    let expected = snapshot(&reference);
    assert_state_close(&actual, &expected);
    assert_state_changed(&initial, &actual);
}

#[test]
fn tiny_llama_simulated_data_parallel_matches_full_batch_reference() {
    let initial = tiny_llama_state();
    let reference = new_llama();
    let mut reference_optimizer = adamw(&reference);
    direct_train_step(&reference, &mut reference_optimizer, &data_parallel_batch());

    let distributed = new_llama();
    let mut distributed_optimizer = adamw(&distributed);
    simulated_data_parallel_train_step(
        &distributed,
        &mut distributed_optimizer,
        &data_parallel_batch(),
        2,
    );

    let actual = snapshot(&distributed);
    let expected = snapshot(&reference);
    assert_state_close(&actual, &expected);
    assert_state_changed(&initial, &actual);
}

#[test]
fn tiny_llama_safetensors_checkpoint_reloads_identical_state_and_forward() {
    let model = new_llama();
    let mut optimizer = adamw(&model);
    direct_train_step(&model, &mut optimizer, &batch());
    let trained_state = snapshot(&model);

    let checkpoint_dir = temp_checkpoint_dir("tiny-llama-checkpoint");
    std::fs::create_dir_all(&checkpoint_dir).expect("create checkpoint dir");
    let config_path = checkpoint_dir.join("config.json");
    std::fs::write(
        &config_path,
        serde_json::to_string(&tiny_llama_config()).expect("serialize tiny llama config"),
    )
    .expect("write tiny llama config");
    let mut repo_files =
        save_tensors_into_safetensors(clone_state(&trained_state), checkpoint_dir.clone())
            .expect("save tiny llama safetensors");
    repo_files.push(config_path);

    let reloaded = new_llama_from_repo_files(repo_files);
    let reloaded_state = snapshot(&reloaded);
    assert_state_close(&reloaded_state, &trained_state);

    let (expected_logits, expected_loss) = forward_loss(&model, &batch());
    let (actual_logits, actual_loss) = forward_loss(&reloaded, &batch());
    assert!(
        actual_logits.allclose(&expected_logits, 1e-6, 1e-6, false),
        "reloaded logits differ from original checkpoint"
    );
    assert!(
        actual_loss.allclose(&expected_loss, 1e-6, 1e-6, false),
        "reloaded loss differs from original checkpoint"
    );

    let _ = std::fs::remove_dir_all(checkpoint_dir);
}

#[test]
fn tiny_tied_llama_safetensors_checkpoint_loads_and_trains_shared_embeddings() {
    let checkpoint_dir = temp_checkpoint_dir("tiny-tied-llama-checkpoint");
    let initial = tiny_tied_llama_state();
    std::fs::create_dir_all(&checkpoint_dir).expect("create checkpoint dir");
    let config_path = checkpoint_dir.join("config.json");
    std::fs::write(
        &config_path,
        serde_json::to_string(&tiny_tied_llama_config()).expect("serialize tied tiny llama config"),
    )
    .expect("write tied tiny llama config");
    let mut repo_files =
        save_tensors_into_safetensors(clone_state(&initial), checkpoint_dir.clone())
            .expect("save tied tiny llama safetensors");
    repo_files.push(config_path);

    let model = new_llama_from_repo_files(repo_files);
    assert_state_close(&snapshot(&model), &initial);
    assert!(
        model.lm_head.allclose(
            &snapshot(&model)["model.embed_tokens.weight"],
            0.0,
            0.0,
            false
        ),
        "tied output head must share the input embedding parameter"
    );

    let mut optimizer = adamw(&model);
    direct_train_step(&model, &mut optimizer, &batch());
    let trained = snapshot(&model);
    assert_state_changed(&initial, &trained);
    assert!(
        model
            .lm_head
            .allclose(&trained["model.embed_tokens.weight"], 0.0, 0.0, false),
        "tied output head must track embedding optimizer updates"
    );

    let _ = std::fs::remove_dir_all(checkpoint_dir);
}
