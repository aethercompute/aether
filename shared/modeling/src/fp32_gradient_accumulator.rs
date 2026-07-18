use crate::{AllReduce, CausalLM, Communicator, ReduceType};

use std::sync::Arc;
use tch::{Device, Kind, Tensor};

pub struct Fp32GradientAccumulator {
    parameters: Vec<(Tensor, (i64, i64))>,
    fp32_grads: Tensor,
}

impl Fp32GradientAccumulator {
    pub fn new(model: &dyn CausalLM) -> Self {
        let _no_grad = tch::no_grad_guard();
        let parameters = model
            .trainable_variables()
            .map(|parameter| parameter.local_tensor())
            .collect::<Vec<_>>();
        Self::from_parameters(parameters, model.device())
    }

    fn from_parameters(parameters: Vec<Tensor>, device: Device) -> Self {
        let mut total_numel: i64 = 0;

        let parameters = parameters
            .into_iter()
            .filter_map(|parameter| match parameter.requires_grad() {
                true => {
                    let numel = parameter.numel() as i64;
                    let ret = (
                        parameter.shallow_clone(),
                        (total_numel, total_numel + numel),
                    );
                    total_numel += numel;
                    Some(ret)
                }
                false => None,
            })
            .collect::<Vec<_>>();

        let fp32_grads = Tensor::zeros([total_numel], (Kind::Float, device));

        Self {
            parameters,
            fp32_grads,
        }
    }

    pub fn accumulate_gradients(&mut self) {
        let _no_grad = tch::no_grad_guard();
        for (param, (start, end)) in &mut self.parameters {
            let grad = param.grad();
            if !grad.defined() {
                continue;
            }
            let mut grad_slice = self.fp32_grads.slice(0, *start, *end, 1);
            let _t = grad_slice.g_add_(&grad.to_kind(Kind::Float).view([-1]));
            param.zero_grad();
        }
    }

    pub fn apply_accumulation(&mut self) {
        let _no_grad = tch::no_grad_guard();
        for (param, (start, end)) in &self.parameters {
            let mut grad = param.grad();
            if !grad.defined() {
                continue;
            }
            let grad_slice = self.fp32_grads.slice(0, *start, *end, 1);
            grad.copy_(&grad_slice.to_kind(param.kind()).view_as(param));
        }
    }

    pub fn zero_grad(&mut self) {
        let _ = self.fp32_grads.zero_();
    }

    pub fn get_full_grad_buffer(&self) -> &Tensor {
        &self.fp32_grads
    }

    pub fn reduce_gradients(&mut self, comm: Arc<Communicator>) {
        self.fp32_grads.all_reduce(&Some(comm), ReduceType::Mean);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parameter(values: &[f32], kind: Kind) -> Tensor {
        Tensor::from_slice(values)
            .to_kind(kind)
            .set_requires_grad(true)
    }

    fn backward(parameter: &Tensor, coefficients: &[f32]) {
        (parameter * Tensor::from_slice(coefficients).to_kind(parameter.kind()))
            .sum(parameter.kind())
            .backward();
    }

    #[test]
    fn accumulates_multiple_gradients_and_resets_buffer() {
        let parameter = parameter(&[0.5, -0.5], Kind::Float);
        let mut accumulator =
            Fp32GradientAccumulator::from_parameters(vec![parameter.shallow_clone()], Device::Cpu);

        backward(&parameter, &[1.0, 2.0]);
        accumulator.accumulate_gradients();
        backward(&parameter, &[3.0, 4.0]);
        accumulator.accumulate_gradients();

        assert!(accumulator
            .get_full_grad_buffer()
            .equal(&Tensor::from_slice(&[4.0_f32, 6.0])));
        assert!(parameter
            .grad()
            .equal(&Tensor::zeros([2], (Kind::Float, Device::Cpu))));

        accumulator.apply_accumulation();
        assert!(parameter.grad().equal(&Tensor::from_slice(&[4.0_f32, 6.0])));
        accumulator.zero_grad();
        assert!(accumulator
            .get_full_grad_buffer()
            .equal(&Tensor::zeros([2], (Kind::Float, Device::Cpu))));
        accumulator.apply_accumulation();
        assert!(parameter
            .grad()
            .equal(&Tensor::zeros([2], (Kind::Float, Device::Cpu))));
    }

    #[test]
    fn accumulates_in_fp32_and_converts_back_to_parameter_dtype() {
        let parameter = parameter(&[0.5, -0.5], Kind::BFloat16);
        let mut accumulator =
            Fp32GradientAccumulator::from_parameters(vec![parameter.shallow_clone()], Device::Cpu);

        backward(&parameter, &[1.5, -2.5]);
        accumulator.accumulate_gradients();

        assert_eq!(accumulator.get_full_grad_buffer().kind(), Kind::Float);
        assert!(accumulator.get_full_grad_buffer().allclose(
            &Tensor::from_slice(&[1.5_f32, -2.5]),
            0.0,
            0.0,
            false
        ));
        accumulator.apply_accumulation();
        assert_eq!(parameter.grad().kind(), Kind::BFloat16);
        assert!(parameter.grad().to_kind(Kind::Float).allclose(
            &Tensor::from_slice(&[1.5_f32, -2.5]),
            0.0,
            0.0,
            false
        ));
    }

    #[test]
    fn missing_gradients_remain_zero_and_undefined() {
        let used = parameter(&[1.0, 2.0], Kind::Float);
        let unused = parameter(&[3.0, 4.0], Kind::Float);
        let mut accumulator = Fp32GradientAccumulator::from_parameters(
            vec![used.shallow_clone(), unused.shallow_clone()],
            Device::Cpu,
        );

        backward(&used, &[2.0, 3.0]);
        assert!(!unused.grad().defined());
        accumulator.accumulate_gradients();
        accumulator.apply_accumulation();

        assert!(accumulator
            .get_full_grad_buffer()
            .equal(&Tensor::from_slice(&[2.0_f32, 3.0, 0.0, 0.0])));
        assert!(!unused.grad().defined());
    }
}
