use std::{collections::HashMap, sync::Arc};

use aether_core::{
    BatchId, CancellableBarrier, ClosedInterval, ConstantLR, LearningRateSchedule,
    OptimizerDefinition,
};
use aether_modeling::{
    Batch, BatchData, BatchDataCPU, CausalLM, DistroResult, EosToks, LocalTrainer, ParallelModels,
    StableVarStoreIterator, StableVariableIterator, Trainer,
};
use aether_network::{
    distro_results_from_reader, distro_results_to_bytes, SerializedDistroResult,
    TransmittableDistroResult,
};
use tch::{
    nn::{self, Module, VarStore},
    COptimizer, Device, Kind, Tensor,
};
use tokio_util::sync::CancellationToken;

const VOCAB_SIZE: i64 = 7;
const HIDDEN_SIZE: i64 = 5;
const SEQ_LEN: usize = 5;

type DistroResults = Vec<DistroResult>;

struct TinyCausalLm {
    vs: VarStore,
    embed: nn::Embedding,
    lm_head: nn::Linear,
}

// SAFETY: the test model is moved into one trainer-owned worker thread and all
// mutable tensor operations are sequenced by the trainer barrier.
unsafe impl Send for TinyCausalLm {}

impl TinyCausalLm {
    fn new() -> Self {
        let vs = VarStore::new(Device::Cpu);
        let root = vs.root();
        let embed = nn::embedding(&root / "embed", VOCAB_SIZE, HIDDEN_SIZE, Default::default());
        let lm_head = nn::linear(
            &root / "lm_head",
            HIDDEN_SIZE,
            VOCAB_SIZE,
            nn::LinearConfig {
                bias: false,
                ..Default::default()
            },
        );
        let _checkpoint_only = root.var("checkpoint_only", &[1], nn::Init::Const(3.0));

        let model = Self { vs, embed, lm_head };
        model.initialize_weights();
        model
    }

    fn from_state(state: &HashMap<String, Tensor>) -> Self {
        let model = Self::new();
        load_state(&model, state);
        model
    }

    fn initialize_weights(&self) {
        let _no_grad = tch::no_grad_guard();

        let embed_weight = (Tensor::arange(VOCAB_SIZE * HIDDEN_SIZE, (Kind::Float, Device::Cpu))
            .reshape([VOCAB_SIZE, HIDDEN_SIZE])
            / 23.0)
            - 0.7;
        let head_weight = (Tensor::arange(VOCAB_SIZE * HIDDEN_SIZE, (Kind::Float, Device::Cpu))
            .reshape([VOCAB_SIZE, HIDDEN_SIZE])
            .flip([0])
            / 29.0)
            - 0.4;

        let mut embed_ws = self.embed.ws.shallow_clone();
        embed_ws.copy_(&embed_weight);
        let mut head_ws = self.lm_head.ws.shallow_clone();
        head_ws.copy_(&head_weight);
    }
}

impl CausalLM for TinyCausalLm {
    fn forward(
        &self,
        x: &Tensor,
        labels: Option<&Tensor>,
        _position_ids: Option<&Tensor>,
        _sequence_lengths: Option<&Vec<Vec<i32>>>,
        num_logits_to_keep: Option<i64>,
        loss_scale: Option<f64>,
    ) -> (Option<Tensor>, Option<Tensor>) {
        let (_, t) = x.size2().expect("tiny oracle inputs are [batch, time]");
        let hidden = self.embed.forward(x);
        let hidden = match num_logits_to_keep {
            Some(num_logits_to_keep) => hidden.slice(1, t - num_logits_to_keep, t, 1),
            None => hidden,
        };
        let mut logits = self.lm_head.forward(&hidden);
        let loss = labels.map(|labels| {
            logits = logits.to_kind(Kind::Float);
            let shift_logits = logits.slice(1, 0, -1, 1).contiguous();
            let shift_labels = labels.slice(1, 1, None, 1).contiguous();
            let loss = shift_logits
                .view([-1, VOCAB_SIZE])
                .cross_entropy_loss::<Tensor>(
                    &shift_labels.view(-1).to_kind(Kind::Int64),
                    None,
                    tch::Reduction::Mean,
                    -100,
                    0.0,
                );
            match loss_scale {
                Some(loss_scale) => loss / loss_scale,
                None => loss,
            }
        });

        (Some(logits), loss)
    }

    fn bos_token_id(&self) -> Option<i64> {
        None
    }

    fn eos_token_ids(&self) -> Option<EosToks> {
        None
    }

    fn device(&self) -> Device {
        Device::Cpu
    }

    fn max_context_length(&self) -> usize {
        SEQ_LEN
    }

    fn trainable_variables(&self) -> StableVariableIterator {
        Box::new(
            StableVarStoreIterator::new(&self.vs, None)
                .filter(|variable| variable.name() != "checkpoint_only"),
        )
    }

    fn state_variables(&self) -> StableVariableIterator {
        Box::new(StableVarStoreIterator::new(&self.vs, None))
    }

    fn communicator(&self) -> Option<Arc<aether_modeling::Communicator>> {
        None
    }

    fn prepare_for_training(&self) {}

    fn clip_grad_norm(&self, _max_grad_norm: f64) {}

    fn convert(&self, state_dict: Option<HashMap<String, Tensor>>) -> HashMap<String, Tensor> {
        state_dict.unwrap_or_else(|| snapshot_state(self))
    }
}

struct DirectAdamW {
    model: TinyCausalLm,
    optimizer: COptimizer,
}

impl DirectAdamW {
    fn new() -> Self {
        let model = TinyCausalLm::new();
        let optimizer = adamw_optimizer(&model);
        Self { model, optimizer }
    }

    fn train_step(&mut self, step: u32, batch: &Batch) -> f32 {
        for variable in self.model.trainable_variables() {
            variable.zero_grad();
        }

        let batch = batch.clone().gpu(Device::Cpu);
        let BatchData::GPU(data) = batch.data else {
            unreachable!("batch was moved to GPU/CPU tensor form")
        };
        let (_, loss) = self.model.forward(
            &data.input_ids,
            data.labels.as_ref(),
            data.position_ids.as_ref(),
            data.sequence_lengths.as_ref(),
            None,
            Some(1.0),
        );
        let loss = loss.expect("tiny oracle model returns a loss");
        let loss_value = loss.double_value(&[]) as f32;
        loss.backward();

        self.optimizer
            .set_learning_rate(schedule().get_lr(step))
            .expect("valid learning rate");
        self.optimizer.step().expect("adamw step succeeds");
        self.optimizer.zero_grad().expect("zero gradients succeeds");

        loss_value
    }

    fn state(&self) -> HashMap<String, Tensor> {
        snapshot_state(&self.model)
    }
}

struct SimulatedDataParallelAdamW {
    model: TinyCausalLm,
    optimizer: COptimizer,
    worker_count: usize,
}

impl SimulatedDataParallelAdamW {
    fn new(worker_count: usize) -> Self {
        assert!(
            worker_count > 1,
            "use at least two workers for DP simulation"
        );
        let model = TinyCausalLm::new();
        let optimizer = adamw_optimizer(&model);
        Self {
            model,
            optimizer,
            worker_count,
        }
    }

    fn train_step(&mut self, step: u32, batch: &Batch) {
        let state = snapshot_state(&self.model);
        let worker_batches = split_batch_for_workers(batch, self.worker_count);
        let worker_grads = worker_batches
            .iter()
            .map(|batch| gradients_for_batch_from_state(&state, batch))
            .collect::<Vec<_>>();

        // Materialize parameter grad tensors once, then overwrite them with the
        // averaged worker gradients. This keeps the simulation independent of
        // production NCCL while still stepping the real tch AdamW optimizer.
        materialize_zero_grads(
            &self.model,
            worker_batches.first().expect("worker batch present"),
        );
        for variable in self.model.trainable_variables() {
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

        self.optimizer
            .set_learning_rate(schedule().get_lr(step))
            .expect("valid learning rate");
        self.optimizer.step().expect("adamw step succeeds");
        self.optimizer.zero_grad().expect("zero gradients succeeds");
    }

    fn state(&self) -> HashMap<String, Tensor> {
        snapshot_state(&self.model)
    }
}

fn schedule() -> LearningRateSchedule {
    LearningRateSchedule::Constant(ConstantLR::new(0.03, 0, 0.0))
}

fn adamw_definition() -> OptimizerDefinition {
    OptimizerDefinition::AdamW {
        betas: [0.9, 0.95],
        weight_decay: 0.01,
        eps: 1e-8,
        clip_grad_norm: None,
    }
}

fn adamw_optimizer(model: &dyn CausalLM) -> COptimizer {
    let OptimizerDefinition::AdamW {
        betas,
        weight_decay,
        eps,
        ..
    } = adamw_definition()
    else {
        unreachable!("adamw_definition returns AdamW")
    };

    let mut optimizer = COptimizer::adamw(
        0.1,
        betas[0] as f64,
        betas[1] as f64,
        weight_decay as f64,
        eps as f64,
        false,
    )
    .expect("adamw optimizer initializes");
    for variable in model.trainable_variables() {
        optimizer
            .add_parameters(&variable.logical_tensor(), 0)
            .expect("parameter can be added to adamw");
    }
    optimizer
}

fn distro_definition() -> OptimizerDefinition {
    OptimizerDefinition::Distro {
        clip_grad_norm: None,
        weight_decay: None,
        compression_decay: 1.0,
        compression_topk: u16::MAX,
        compression_chunk: 64,
        quantize_1bit: false,
    }
}

fn new_trainer(optimizer: OptimizerDefinition, micro_batch_size: usize) -> Trainer {
    LocalTrainer::new(
        ParallelModels {
            models: vec![Box::new(TinyCausalLm::new()) as Box<dyn CausalLM>],
            barrier: Arc::new(CancellableBarrier::new(1)),
            data_parallel: None,
        },
        schedule(),
        optimizer,
        micro_batch_size,
        None,
        false,
    )
    .into()
}

fn run_adamw_trainer_step(trainer: Trainer, step: u32, batch: Batch) -> (Trainer, f32) {
    let output = trainer
        .train(
            step,
            batch,
            None,
            false,
            vec![],
            None,
            CancellationToken::new(),
        )
        .expect("trainer train step succeeds");
    let loss = output.loss;
    let trainer = output
        .trainer
        .optimize(step, None, None)
        .expect("adamw optimize step succeeds");
    (trainer, loss)
}

fn run_distro_train(
    trainer: Trainer,
    step: u32,
    batch: Batch,
    prev_self_results: Option<DistroResults>,
) -> (Trainer, DistroResults) {
    let output = trainer
        .train(
            step,
            batch,
            None,
            false,
            vec![],
            Some(prev_self_results.map_or_else(Vec::new, |results| vec![results])),
            CancellationToken::new(),
        )
        .expect("distro worker train step succeeds");
    let results = output
        .distro_results
        .expect("distro worker emits transport results");
    (output.trainer, results)
}

fn run_distro_worker(step: u32, batch: Batch) -> (Trainer, DistroResults) {
    run_distro_train(
        new_trainer(distro_definition(), batch.data.size()),
        step,
        batch,
        None,
    )
}

fn optimize_distro_trainer(trainer: Trainer, step: u32, results: Vec<DistroResults>) -> Trainer {
    trainer
        .optimize(step, None, Some(results))
        .expect("distro aggregate optimize succeeds")
}

fn wire_roundtrip_distro_results(results: &[DistroResults]) -> Vec<DistroResults> {
    results
        .iter()
        .map(|worker_results| {
            let serialized = worker_results
                .iter()
                .map(|result| {
                    SerializedDistroResult::try_from(result)
                        .expect("distro result serializes for wire transport")
                })
                .collect::<Vec<_>>();
            let bytes = distro_results_to_bytes(&serialized)
                .expect("serialized distro results encode to postcard bytes");
            let decoded = distro_results_from_reader(bytes.as_slice())
                .collect::<Result<Vec<_>, _>>()
                .expect("serialized distro results decode from byte stream");
            assert_eq!(serialized, decoded, "serialized wire payload changed");
            decoded
                .iter()
                .map(|result| {
                    DistroResult::try_from(result)
                        .expect("serialized distro result converts back to native tensor result")
                })
                .collect()
        })
        .collect()
}

fn serialize_distro_results(results: &DistroResults) -> Vec<SerializedDistroResult> {
    results
        .iter()
        .map(|result| {
            SerializedDistroResult::try_from(result)
                .expect("distro result serializes for wire transport")
        })
        .collect()
}

fn batch_id(start: u64, end: u64) -> BatchId {
    BatchId(ClosedInterval { start, end })
}

fn batch_from_rows(rows: &[[i32; SEQ_LEN]]) -> Batch {
    Batch {
        id: batch_id(0, rows.len().saturating_sub(1) as u64),
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

fn batch_from_rows_and_labels(rows: &[([i32; SEQ_LEN], [i32; SEQ_LEN])]) -> Batch {
    Batch {
        id: batch_id(0, rows.len().saturating_sub(1) as u64),
        data: BatchData::CPU(
            rows.iter()
                .map(|(input_ids, labels)| BatchDataCPU {
                    input_ids: input_ids.to_vec(),
                    labels: Some(labels.to_vec()),
                    position_ids: Some((0..SEQ_LEN as i32).collect()),
                    sequence_lengths: Some(vec![SEQ_LEN as i32]),
                })
                .collect(),
        ),
    }
}

fn split_batch_for_workers(batch: &Batch, worker_count: usize) -> Vec<Batch> {
    let BatchData::CPU(rows) = &batch.data else {
        panic!("training oracle batches must be CPU batches")
    };
    assert_eq!(
        rows.len() % worker_count,
        0,
        "oracle DP simulation expects equal-size worker shards"
    );
    let chunk_size = rows.len() / worker_count;
    rows.chunks(chunk_size)
        .enumerate()
        .map(|(worker, chunk)| Batch {
            id: batch_id(
                worker as u64 * chunk_size as u64,
                (worker as u64 + 1) * chunk_size as u64 - 1,
            ),
            data: BatchData::CPU(chunk.to_vec()),
        })
        .collect()
}

fn training_batches() -> Vec<Batch> {
    vec![
        batch_from_rows(&[
            [0, 1, 2, 3, 4],
            [4, 5, 6, 0, 1],
            [1, 3, 5, 0, 2],
            [6, 4, 2, 1, 3],
        ]),
        batch_from_rows(&[
            [2, 4, 6, 1, 3],
            [3, 2, 1, 0, 6],
            [5, 0, 4, 2, 1],
            [1, 6, 3, 5, 0],
        ]),
        batch_from_rows(&[
            [6, 5, 4, 3, 2],
            [0, 2, 4, 6, 1],
            [3, 5, 0, 1, 2],
            [4, 1, 6, 2, 5],
        ]),
    ]
}

fn uneven_training_batches() -> Vec<Batch> {
    vec![
        batch_from_rows(&[
            [0, 1, 2, 3, 4],
            [4, 5, 6, 0, 1],
            [1, 3, 5, 0, 2],
            [6, 4, 2, 1, 3],
            [2, 0, 5, 1, 6],
        ]),
        batch_from_rows(&[
            [2, 4, 6, 1, 3],
            [3, 2, 1, 0, 6],
            [5, 0, 4, 2, 1],
            [1, 6, 3, 5, 0],
            [4, 3, 0, 6, 2],
        ]),
    ]
}

fn ignored_label_training_batches() -> Vec<Batch> {
    vec![
        batch_from_rows_and_labels(&[
            ([0, 1, 2, 3, 4], [0, 1, 2, -100, 4]),
            ([4, 5, 6, 0, 1], [4, -100, 6, 0, 1]),
            ([1, 3, 5, 0, 2], [1, 3, -100, -100, 2]),
            ([6, 4, 2, 1, 3], [6, 4, 2, 1, -100]),
            ([2, 0, 5, 1, 6], [2, -100, 5, 1, 6]),
        ]),
        batch_from_rows_and_labels(&[
            ([2, 4, 6, 1, 3], [2, 4, -100, 1, 3]),
            ([3, 2, 1, 0, 6], [3, 2, 1, -100, 6]),
            ([5, 0, 4, 2, 1], [5, -100, 4, 2, -100]),
            ([1, 6, 3, 5, 0], [1, 6, 3, 5, 0]),
            ([4, 3, 0, 6, 2], [4, 3, -100, 6, 2]),
        ]),
    ]
}

fn distro_worker_batches() -> Vec<Batch> {
    vec![
        batch_from_rows(&[[0, 1, 2, 3, 4], [4, 5, 6, 0, 1]]),
        batch_from_rows(&[[1, 3, 5, 0, 2], [6, 4, 2, 1, 3]]),
    ]
}

fn distro_worker_batches_by_round() -> Vec<Vec<Batch>> {
    vec![
        distro_worker_batches(),
        vec![
            batch_from_rows(&[[2, 4, 6, 1, 3], [3, 2, 1, 0, 6]]),
            batch_from_rows(&[[5, 0, 4, 2, 1], [1, 6, 3, 5, 0]]),
        ],
        vec![
            batch_from_rows(&[[6, 5, 4, 3, 2], [0, 2, 4, 6, 1]]),
            batch_from_rows(&[[3, 5, 0, 1, 2], [4, 1, 6, 2, 5]]),
        ],
    ]
}

fn snapshot_state(model: &dyn CausalLM) -> HashMap<String, Tensor> {
    model
        .state_variables()
        .map(|variable| {
            (
                variable.name().to_owned(),
                variable.gather_full_tensor().to_device(Device::Cpu).copy(),
            )
        })
        .collect()
}

fn load_state(model: &dyn CausalLM, state: &HashMap<String, Tensor>) {
    let _no_grad = tch::no_grad_guard();
    for variable in model.state_variables() {
        let mut tensor = variable.logical_tensor();
        tensor.copy_(state.get(variable.name()).expect("state tensor present"));
    }
}

fn extract_state(trainer: &mut Trainer) -> HashMap<String, Tensor> {
    trainer
        .extract()
        .expect("trainer state extraction succeeds")
}

#[test]
fn extraction_defaults_to_full_state_and_can_select_trainable_state() {
    let mut trainer = new_trainer(OptimizerDefinition::Dummy, 1);

    let full = trainer.extract().expect("full state extraction succeeds");
    let trainable = trainer
        .extract_trainable()
        .expect("trainable state extraction succeeds");

    assert!(full.contains_key("checkpoint_only"));
    assert!(!trainable.contains_key("checkpoint_only"));
    assert_eq!(full.len(), trainable.len() + 1);
}

fn assert_state_close(
    actual: &HashMap<String, Tensor>,
    expected: &HashMap<String, Tensor>,
    atol: f64,
    rtol: f64,
) {
    let mut actual_keys = actual.keys().cloned().collect::<Vec<_>>();
    let mut expected_keys = expected.keys().cloned().collect::<Vec<_>>();
    actual_keys.sort();
    expected_keys.sort();
    assert_eq!(actual_keys, expected_keys, "state dict keys changed");

    for key in expected_keys {
        let actual_tensor = actual.get(&key).expect("actual tensor present");
        let expected_tensor = expected.get(&key).expect("expected tensor present");
        assert_eq!(
            actual_tensor.size(),
            expected_tensor.size(),
            "shape changed for {key}"
        );

        let diff = (actual_tensor - expected_tensor).abs();
        let max_abs = diff.max().double_value(&[]);
        let rel = &diff / expected_tensor.abs().clamp_min(1e-12);
        let max_rel = rel.max().double_value(&[]);
        assert!(
            actual_tensor.allclose(expected_tensor, rtol, atol, false),
            "tensor {key} differs: max_abs={max_abs:.6e}, max_rel={max_rel:.6e}, shape={:?}",
            actual_tensor.size()
        );
    }
}

fn assert_state_changed(initial: &HashMap<String, Tensor>, final_state: &HashMap<String, Tensor>) {
    let changed = initial.iter().any(|(key, initial_tensor)| {
        let final_tensor = final_state.get(key).expect("final tensor present");
        !initial_tensor.allclose(final_tensor, 0.0, 0.0, false)
    });
    assert!(changed, "training oracle did not update any parameters");
}

fn gradients_for_batch_from_state(
    state: &HashMap<String, Tensor>,
    batch: &Batch,
) -> HashMap<String, Tensor> {
    let model = TinyCausalLm::from_state(state);
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
    loss.expect("tiny oracle model returns a loss").backward();

    model
        .trainable_variables()
        .map(|variable| {
            let grad = variable.logical_tensor().grad();
            assert!(grad.defined(), "missing gradient for {}", variable.name());
            (
                variable.name().to_owned(),
                grad.to_device(Device::Cpu).copy(),
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
    let zero_loss = loss.expect("tiny oracle model returns a loss") * 0.0;
    zero_loss.backward();
    for variable in model.trainable_variables() {
        variable.zero_grad();
    }
}

fn gradients_for_batch(batch: &Batch) -> HashMap<String, Tensor> {
    gradients_for_batch_from_state(&snapshot_state(&TinyCausalLm::new()), batch)
}

fn sign_mean_reference_update(worker_batches: &[Batch], lr: f64) -> HashMap<String, Tensor> {
    let initial = snapshot_state(&TinyCausalLm::new());
    let worker_grads = worker_batches
        .iter()
        .map(gradients_for_batch)
        .collect::<Vec<_>>();

    initial
        .iter()
        .map(|(key, initial_tensor)| {
            let Some(first_grad) = worker_grads[0].get(key) else {
                return (key.clone(), initial_tensor.shallow_clone());
            };
            let mut mean_grad = Tensor::zeros_like(first_grad);
            for grads in &worker_grads {
                mean_grad += grads.get(key).expect("worker gradient present");
            }
            mean_grad /= worker_grads.len() as f64;
            (key.clone(), initial_tensor - mean_grad.sign() * lr)
        })
        .collect()
}

fn sign_mean_reference_rounds(rounds: &[Vec<Batch>]) -> HashMap<String, Tensor> {
    let mut state = snapshot_state(&TinyCausalLm::new());
    let mut prev_worker_grads: Vec<Option<HashMap<String, Tensor>>> = Vec::new();

    for (step, worker_batches) in rounds.iter().enumerate() {
        if prev_worker_grads.is_empty() {
            prev_worker_grads = (0..worker_batches.len()).map(|_| None).collect();
        }
        let prev_lr = match step {
            0 => schedule().get_lr(0),
            step => schedule().get_lr((step - 1) as u32),
        };
        let worker_grads = worker_batches
            .iter()
            .zip(prev_worker_grads.iter())
            .map(|(batch, prev_grad)| {
                let corrected_state = match prev_grad {
                    Some(prev_grad) => state
                        .iter()
                        .map(|(key, tensor)| {
                            let corrected = match prev_grad.get(key) {
                                Some(prev) => tensor - prev.sign() * prev_lr,
                                None => tensor.shallow_clone(),
                            };
                            (key.clone(), corrected)
                        })
                        .collect(),
                    None => clone_state(&state),
                };
                gradients_for_batch_from_state(&corrected_state, batch)
            })
            .collect::<Vec<_>>();
        let lr = schedule().get_lr(step as u32);

        state = state
            .iter()
            .map(|(key, current_tensor)| {
                let Some(first_grad) = worker_grads[0].get(key) else {
                    return (key.clone(), current_tensor.shallow_clone());
                };
                let mut mean_grad = Tensor::zeros_like(first_grad);
                for grads in &worker_grads {
                    mean_grad += grads.get(key).expect("worker gradient present");
                }
                mean_grad /= worker_grads.len() as f64;
                (key.clone(), current_tensor - mean_grad.sign() * lr)
            })
            .collect();
        prev_worker_grads = worker_grads.into_iter().map(Some).collect();
    }

    state
}

fn clone_state(state: &HashMap<String, Tensor>) -> HashMap<String, Tensor> {
    state
        .iter()
        .map(|(key, tensor)| (key.clone(), tensor.shallow_clone()))
        .collect()
}

#[test]
fn adamw_local_trainer_matches_direct_full_batch_reference() {
    let initial = snapshot_state(&TinyCausalLm::new());
    let mut reference = DirectAdamW::new();
    let mut trainer = new_trainer(adamw_definition(), 4);

    for (step, batch) in training_batches().into_iter().enumerate() {
        let expected_loss = reference.train_step(step as u32, &batch);
        let (next_trainer, actual_loss) = run_adamw_trainer_step(trainer, step as u32, batch);
        trainer = next_trainer;
        assert!(
            (actual_loss - expected_loss).abs() < 1e-6,
            "loss differs at step {step}: actual={actual_loss}, expected={expected_loss}"
        );
    }

    let actual = extract_state(&mut trainer);
    let expected = reference.state();
    assert_state_close(&actual, &expected, 1e-6, 1e-6);
    assert_state_changed(&initial, &actual);
}

#[test]
fn adamw_microbatch_accumulation_matches_full_batch_reference() {
    let initial = snapshot_state(&TinyCausalLm::new());
    let mut reference = DirectAdamW::new();
    let mut trainer = new_trainer(adamw_definition(), 2);

    for (step, batch) in training_batches().into_iter().enumerate() {
        let expected_loss = reference.train_step(step as u32, &batch);
        let (next_trainer, actual_loss) = run_adamw_trainer_step(trainer, step as u32, batch);
        trainer = next_trainer;
        assert!(
            (actual_loss - expected_loss).abs() < 1e-6,
            "loss differs at step {step}: actual={actual_loss}, expected={expected_loss}"
        );
    }

    let actual = extract_state(&mut trainer);
    let expected = reference.state();
    assert_state_close(&actual, &expected, 1e-6, 1e-6);
    assert_state_changed(&initial, &actual);
}

#[test]
fn adamw_uneven_microbatch_accumulation_matches_full_batch_reference() {
    let initial = snapshot_state(&TinyCausalLm::new());
    let mut reference = DirectAdamW::new();
    let mut trainer = new_trainer(adamw_definition(), 2);

    for (step, batch) in uneven_training_batches().into_iter().enumerate() {
        let expected_loss = reference.train_step(step as u32, &batch);
        let (next_trainer, actual_loss) = run_adamw_trainer_step(trainer, step as u32, batch);
        trainer = next_trainer;
        assert!(
            (actual_loss - expected_loss).abs() < 1e-6,
            "loss differs at step {step}: actual={actual_loss}, expected={expected_loss}"
        );
    }

    let actual = extract_state(&mut trainer);
    let expected = reference.state();
    assert_state_close(&actual, &expected, 1e-6, 1e-6);
    assert_state_changed(&initial, &actual);
}

#[test]
fn adamw_microbatch_with_ignored_labels_matches_full_batch_reference() {
    let initial = snapshot_state(&TinyCausalLm::new());
    let mut reference = DirectAdamW::new();
    let mut trainer = new_trainer(adamw_definition(), 2);

    for (step, batch) in ignored_label_training_batches().into_iter().enumerate() {
        let expected_loss = reference.train_step(step as u32, &batch);
        let (next_trainer, actual_loss) = run_adamw_trainer_step(trainer, step as u32, batch);
        trainer = next_trainer;
        assert!(
            (actual_loss - expected_loss).abs() < 1e-6,
            "loss differs at step {step}: actual={actual_loss}, expected={expected_loss}"
        );
    }

    let actual = extract_state(&mut trainer);
    let expected = reference.state();
    assert_state_close(&actual, &expected, 1e-6, 1e-6);
    assert_state_changed(&initial, &actual);
}

#[test]
fn adamw_simulated_data_parallel_matches_full_batch_reference() {
    let initial = snapshot_state(&TinyCausalLm::new());
    let mut reference = DirectAdamW::new();
    let mut distributed = SimulatedDataParallelAdamW::new(2);

    for (step, batch) in training_batches().into_iter().enumerate() {
        reference.train_step(step as u32, &batch);
        distributed.train_step(step as u32, &batch);
    }

    let actual = distributed.state();
    let expected = reference.state();
    assert_state_close(&actual, &expected, 1e-6, 1e-6);
    assert_state_changed(&initial, &actual);
}

#[test]
fn distro_full_density_step_matches_sign_mean_reference() {
    let worker_batches = distro_worker_batches();
    let lr = schedule().get_lr(0);
    let expected = sign_mean_reference_update(&worker_batches, lr);
    let mut worker_outputs = worker_batches
        .into_iter()
        .map(|batch| run_distro_worker(0, batch))
        .collect::<Vec<_>>();
    let (aggregator, first_results) = worker_outputs.remove(0);
    let worker_results = worker_outputs
        .into_iter()
        .map(|(_, results)| results)
        .chain(std::iter::once(first_results))
        .collect::<Vec<_>>();

    let mut aggregator = aggregator
        .optimize(0, None, Some(worker_results))
        .expect("distro aggregate optimize succeeds");
    let actual = extract_state(&mut aggregator);

    assert_state_close(&actual, &expected, 2e-5, 2e-5);
}

#[test]
fn serialized_distro_results_apply_like_in_memory_results() {
    let worker_batches = distro_worker_batches();
    let worker_results = worker_batches
        .iter()
        .map(|batch| run_distro_worker(0, batch.clone()).1)
        .collect::<Vec<_>>();
    let wire_results = wire_roundtrip_distro_results(&worker_results);

    let original_aggregator = run_distro_worker(0, worker_batches[0].clone()).0;
    let wire_aggregator = run_distro_worker(0, worker_batches[0].clone()).0;

    let mut original_aggregator = optimize_distro_trainer(original_aggregator, 0, worker_results);
    let mut wire_aggregator = optimize_distro_trainer(wire_aggregator, 0, wire_results);

    let original_state = extract_state(&mut original_aggregator);
    let wire_state = extract_state(&mut wire_aggregator);
    assert_state_close(&wire_state, &original_state, 0.0, 0.0);
}

#[test]
fn transmittable_distro_result_hash_survives_postcard_roundtrip() {
    let results = run_distro_worker(7, distro_worker_batches()[0].clone()).1;
    let payload = TransmittableDistroResult {
        format_version: aether_network::DISTRO_RESULT_FORMAT_VERSION,
        manifest_digest: [9; 32],
        step: 7,
        trainer_nonce: 42,
        batch_id: batch_id(11, 12),
        distro_results: serialize_distro_results(&results),
    };
    let hash = payload.comptue_hash();

    let bytes = postcard::to_allocvec(&payload).expect("transmittable result serializes");
    let decoded: TransmittableDistroResult =
        postcard::from_bytes(&bytes).expect("transmittable result deserializes");

    assert_eq!(decoded.step, payload.step);
    assert_eq!(decoded.trainer_nonce, payload.trainer_nonce);
    assert_eq!(decoded.batch_id.0.start, payload.batch_id.0.start);
    assert_eq!(decoded.batch_id.0.end, payload.batch_id.0.end);
    assert_eq!(decoded.distro_results, payload.distro_results);
    assert_eq!(decoded.comptue_hash(), hash, "hash changed after roundtrip");

    let mut changed_step = decoded.clone();
    changed_step.step += 1;
    assert_ne!(
        changed_step.comptue_hash(),
        hash,
        "hash must commit to the training step"
    );

    let mut changed_nonce = decoded.clone();
    changed_nonce.trainer_nonce += 1;
    assert_ne!(
        changed_nonce.comptue_hash(),
        hash,
        "hash must commit to the trainer nonce"
    );

    let mut changed_metadata = decoded.clone();
    changed_metadata.distro_results[0].totalk += 1;
    assert_ne!(
        changed_metadata.comptue_hash(),
        hash,
        "hash must commit to result metadata"
    );

    let mut changed_batch = decoded;
    changed_batch.batch_id = batch_id(12, 13);
    assert_ne!(
        changed_batch.comptue_hash(),
        hash,
        "hash must commit to the batch id"
    );
}

#[test]
fn distro_full_density_multi_round_matches_stateful_sign_mean_reference() {
    let rounds = distro_worker_batches_by_round();
    let worker_count = rounds.first().expect("at least one round").len();
    assert!(
        worker_count > 1,
        "multi-round oracle needs multiple workers"
    );
    assert!(
        rounds.iter().all(|round| round.len() == worker_count),
        "each round must use the same worker count"
    );

    let expected = sign_mean_reference_rounds(&rounds);
    let mut workers = (0..worker_count)
        .map(|_| new_trainer(distro_definition(), 2))
        .collect::<Vec<_>>();
    let mut prev_self_results = vec![None; worker_count];

    for (step, worker_batches) in rounds.into_iter().enumerate() {
        let mut trained = Vec::with_capacity(worker_count);
        let mut next_prev_self_results = Vec::with_capacity(worker_count);

        for ((trainer, batch), prev_self) in workers
            .into_iter()
            .zip(worker_batches)
            .zip(prev_self_results)
        {
            let (trainer, results) = run_distro_train(trainer, step as u32, batch, prev_self);
            next_prev_self_results.push(Some(results.clone()));
            trained.push((trainer, results));
        }

        let aggregate_results = trained
            .iter()
            .map(|(_, results)| results.clone())
            .collect::<Vec<_>>();
        workers = trained
            .into_iter()
            .map(|(trainer, _)| {
                optimize_distro_trainer(trainer, step as u32, aggregate_results.clone())
            })
            .collect();
        prev_self_results = next_prev_self_results;
    }

    let mut final_states = workers.iter_mut().map(extract_state).collect::<Vec<_>>();
    let first_state = final_states.remove(0);
    assert_state_close(&first_state, &expected, 2e-5, 2e-5);
    for state in final_states {
        assert_state_close(&state, &first_state, 2e-5, 2e-5);
        assert_state_close(&state, &expected, 2e-5, 2e-5);
    }
}
