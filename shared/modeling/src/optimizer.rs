use crate::{CausalLM, Distro, MuonOptimizer};
use aether_core::OptimizerDefinition;
use tch::COptimizer;

pub enum Optimizer {
    Torch {
        optimizer: COptimizer,
        clip_grad_norm: Option<f32>,
    },
    Distro {
        optimizer: Box<Distro>,
        clip_grad_norm: Option<f32>,
        quantize_1bit: bool,
    },
    Muon {
        optimizer: Box<MuonOptimizer>,
        clip_grad_norm: Option<f32>,
        quantize_1bit: bool,
    },
    Null,
}

impl Optimizer {
    pub fn new(definition: OptimizerDefinition, model: &dyn CausalLM) -> Self {
        if model.trainable_variables().next().is_none() {
            return Self::Null;
        }
        match definition {
            OptimizerDefinition::AdamW {
                betas,
                weight_decay,
                eps,
                clip_grad_norm,
            } => Self::Torch {
                optimizer: {
                    let mut adamw = COptimizer::adamw(
                        1.0e-1,
                        betas[0] as f64,
                        betas[1] as f64,
                        weight_decay as f64,
                        eps as f64,
                        false,
                    )
                    .unwrap();
                    for var in model.trainable_variables() {
                        let tensor = var.logical_tensor();
                        adamw.add_parameters(&tensor, 0).unwrap();
                    }
                    adamw
                },
                clip_grad_norm,
            },
            OptimizerDefinition::Distro {
                clip_grad_norm,
                weight_decay,
                compression_decay,
                compression_topk,
                compression_chunk,
                quantize_1bit,
            } => Self::Distro {
                optimizer: Distro::new(
                    model,
                    compression_decay as f64,
                    compression_chunk as i64,
                    compression_topk as i64,
                    weight_decay.unwrap_or(0.0) as f64,
                )
                .into(),
                clip_grad_norm,
                quantize_1bit,
            },
            OptimizerDefinition::Muon {
                momentum,
                weight_decay,
                clip_grad_norm,
                nesterov,
                ns_steps,
                lookahead,
                compression_decay,
                compression_topk,
                compression_chunk,
                quantize_1bit,
            } => Self::Muon {
                optimizer: MuonOptimizer::new(
                    model,
                    momentum as f64,
                    weight_decay as f64,
                    nesterov,
                    ns_steps as i64,
                    lookahead,
                    compression_decay as f64,
                    compression_chunk as i64,
                    compression_topk as i64,
                )
                .into(),
                clip_grad_norm,
                quantize_1bit,
            },
            OptimizerDefinition::Dummy => Self::Null,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Communicator, EosToks, StableVariableIterator};
    use std::{collections::HashMap, sync::Arc};
    use tch::{Device, Tensor};

    struct EmptyModel;

    impl CausalLM for EmptyModel {
        fn forward(
            &self,
            _x: &Tensor,
            _labels: Option<&Tensor>,
            _position_ids: Option<&Tensor>,
            _sequence_lengths: Option<&Vec<Vec<i32>>>,
            _num_logits_to_keep: Option<i64>,
            _loss_scale: Option<f64>,
        ) -> (Option<Tensor>, Option<Tensor>) {
            (None, None)
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
            1
        }

        fn trainable_variables(&self) -> StableVariableIterator {
            Box::new(std::iter::empty())
        }

        fn state_variables(&self) -> StableVariableIterator {
            Box::new(std::iter::empty())
        }

        fn communicator(&self) -> Option<Arc<Communicator>> {
            None
        }

        fn prepare_for_training(&self) {}

        fn clip_grad_norm(&self, _max_grad_norm: f64) {}

        fn convert(&self, state_dict: Option<HashMap<String, Tensor>>) -> HashMap<String, Tensor> {
            state_dict.unwrap_or_default()
        }
    }

    #[test]
    fn no_trainable_parameters_use_null_optimizer() {
        let optimizer = Optimizer::new(
            OptimizerDefinition::AdamW {
                betas: [0.9, 0.95],
                weight_decay: 0.01,
                eps: 1e-8,
                clip_grad_norm: Some(1.0),
            },
            &EmptyModel,
        );

        assert!(matches!(optimizer, Optimizer::Null));
    }
}
