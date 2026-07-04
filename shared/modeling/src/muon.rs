//! Muon optimizer — MomentUm Orthogonalized by Newton-Schulz.
//!
//! Mirrors the `Distro` optimizer's compressed-distributed protocol (DCT + top-k
//! sparsification + error feedback + sign quantization), but replaces the per-node
//! update direction: instead of `sign(Σ lr·grad)` (SignSGD), each node contributes
//! the Newton-Schulz-orthogonalized Nesterov momentum of its gradient.
//!
//! Muon's benefit is structural (orthogonalization), not magnitude-based, so it
//! degrades gracefully under the sign channel — `sign(O) ≠ sign(M)` because the
//! orthogonalization is a nonlinear mixing of rows/columns that bakes itself into
//! the sign pattern. This is why Muon (unlike Adam) is a natural fit for the
//! 1-bit/compressed pipeline: the per-tensor magnitude collapses to a shape-derived
//! scalar applied at `apply`, so the wire format is identical to Distro's.
//!
//! Routing:
//!   - ndim >= 2  → Newton-Schulz orthogonalization (`scale = max(1, out/in)^0.5`).
//!   - ndim <  2  → SGD-with-Nesterov-momentum fallback (scale = 1), Adam-free.
//!
//! Reference: https://github.com/KellerJordan/Muon
use crate::{CausalLM, CompressDCT, Distro, DistroResult, TransformDCT, Variable};

use std::collections::HashMap;
use tch::{Kind, Tensor};

struct MuonState {
    /// EMA momentum buffer `M` (param-kind, matches canonical Muon).
    momentum: Box<dyn Variable>,
    /// Error-feedback / transport residual `e` (plays Distro's `delta` role).
    transport: Box<dyn Variable>,
}

pub struct MuonOptimizer {
    momentum_decay: f64,
    weight_decay: f64,
    nesterov: bool,
    ns_steps: i64,
    lookahead: bool,
    compression_decay: f64,
    compression_topk: i64,
    state: Vec<MuonState>,
    /// Per-parameter shape-derived Muon scale (`max(1, out/in)^0.5`, 1.0 for 1D).
    scales: Vec<f64>,
    transform: TransformDCT,
}

/// Per-tensor Muon update scale, computed from the *full* (unsharded) shape.
/// Mirrors canonical `update *= max(1, size(-2)/size(-1))**0.5`, where for a 2D
/// weight `size(-2)` is the leading dim and `size(-1)` the product of the rest
/// (so conv-style `[a, b, c, d]` collapses to `[a, b*c*d]`).
fn muon_scale(shape: &[i64]) -> f64 {
    if shape.len() >= 2 {
        let d0 = shape[0] as f64;
        let rest: i64 = shape[1..].iter().product();
        (1.0_f64).max(d0 / rest as f64).sqrt()
    } else {
        1.0
    }
}

impl MuonOptimizer {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        vs: &dyn CausalLM,
        momentum_decay: f64,
        weight_decay: f64,
        nesterov: bool,
        ns_steps: i64,
        lookahead: bool,
        compression_decay: f64,
        compression_chunk: i64,
        compression_topk: i64,
    ) -> Self {
        let _no_grad = tch::no_grad_guard();

        let mut state = Vec::new();
        let mut scales = Vec::new();
        for variable in vs.variables() {
            let name = variable.name().to_string();
            state.push(MuonState {
                momentum: variable.zeros_like(format!("{name}.momentum")),
                transport: variable.zeros_like(format!("{name}.transport")),
            });
            scales.push(muon_scale(&variable.full_tensor_shape()));
            variable.zero_grad();
        }

        let transform = TransformDCT::new(vs.variables(), compression_chunk);

        Self {
            momentum_decay,
            weight_decay,
            nesterov,
            ns_steps,
            lookahead,
            compression_decay,
            compression_topk,
            state,
            scales,
            transform,
        }
    }

    /// Newton-Schulz quintic iteration approximating the zeroth power (orthogonal
    /// polar factor) of a 2D matrix. Coefficients `(a, b, c)` selected to maximize
    /// slope at zero; run in bfloat16 for speed (stable there by construction).
    /// Input `g` must be 2D.
    fn newton_schulz(g: &Tensor, steps: i64) -> Tensor {
        let (a, b, c) = (3.4445_f64, -4.7750_f64, 2.0315_f64);
        let shape = g.size();
        let rows = shape[0];
        let cols = shape[1];

        let mut x = g.to_kind(Kind::BFloat16);
        let transposed = rows > cols;
        if transposed {
            x = x.transpose(0, 1);
        }

        // Guarantee spectral norm ≤ 1 via the Frobenius-norm bound.
        let frob: f64 = x
            .to_kind(Kind::Float)
            .norm_scalaropt_dtype(2, Kind::Float)
            .try_into()
            .unwrap();
        let _ = x.g_div_scalar_(frob + 1e-7);

        for _ in 0..steps {
            let at = x.transpose(0, 1);
            let a_mat = x.matmul(&at); // X Xᵀ
            let a2 = a_mat.matmul(&a_mat); // (X Xᵀ)²
            let b_mat = &a_mat.multiply_scalar(b) + &a2.multiply_scalar(c);
            let bx = b_mat.matmul(&x);
            x = &x.multiply_scalar(a) + &bx;
        }

        if transposed {
            x = x.transpose(0, 1);
        }
        x.to_kind(g.kind())
    }

    pub fn generate(
        &mut self,
        variables: &dyn CausalLM,
        prev_self_results: &[Vec<DistroResult>],
        prev_lr: f64,
        lr: f64,
        stats: bool,
    ) -> Vec<DistroResult> {
        let _no_grad = tch::no_grad_guard();

        let mut ret = Vec::new();
        for (index, var) in variables.variables().enumerate() {
            let scale = self.scales[index];
            let mut variable = var.logical_tensor();

            let grad_energy: Option<f64> = match stats {
                true => Some(
                    variable
                        .grad()
                        .norm_scalaropt_dtype(1, Kind::Float)
                        .try_into()
                        .unwrap(),
                ),
                _ => None,
            };

            // 1. Lookahead: anticipate own last contribution to the aggregated step.
            //    (Distro uses `delta.sign()·prev_lr`; Muon multiplies by the per-tensor scale.)
            if self.lookahead {
                let transport = self.state[index].transport.logical_tensor();
                let _t = variable.g_add_(&transport.sign().multiply_scalar(prev_lr * scale));
            }

            // 2. Error feedback: remove the component already transmitted last round.
            if !prev_self_results.is_empty() {
                let device = variable.device();
                let indicies = prev_self_results
                    .iter()
                    .map(|x| x[index].sparse_idx.to_device(device))
                    .collect::<Vec<_>>();
                let val_kind = variable.kind();
                let values = prev_self_results
                    .iter()
                    .map(|x| {
                        let sparse_val = x[index].sparse_val.to_device(device);
                        if sparse_val.kind() == Kind::Bool {
                            Distro::unpack_tensor_sign_from_boolean(sparse_val, val_kind)
                        } else {
                            sparse_val
                        }
                    })
                    .collect::<Vec<_>>();

                let decompressed = CompressDCT::batch_decompress(
                    &indicies,
                    &values,
                    &prev_self_results[0][index].xshape,
                    prev_self_results[0][index].totalk,
                    val_kind,
                    device,
                );
                let transmit_grad = self.transform.decode(&decompressed);
                let mut transport = self.state[index].transport.logical_tensor();
                let _t = transport.g_sub_(&var.shard_other_tensor_like_me(transmit_grad));
            }

            // 3. Decoupled weight decay.
            if self.weight_decay != 0.0 {
                let _t = variable.g_mul_scalar_(1.0 - lr * self.weight_decay);
            }

            // 4. Momentum EMA update: M += (1-β)(g - M).
            let grad = variable.grad();
            let direction = {
                let mut m = self.state[index].momentum.logical_tensor();
                let one_minus_beta = 1.0 - self.momentum_decay;
                let grad_minus_m = &grad - &m;
                let _t = m.g_add_(&grad_minus_m.multiply_scalar(one_minus_beta));

                // Nesterov: u = (1-β)g + βM  (i.e. g.lerp(M, β)); else u = M.
                let u = if self.nesterov {
                    let m_minus_grad = &m - &grad;
                    &grad + &m_minus_grad.multiply_scalar(self.momentum_decay)
                } else {
                    m.shallow_clone()
                };

                // 5. Orthogonalize 2D matrices over the *full* tensor (collective under TP).
                let shape_full = var.full_tensor_shape();
                if shape_full.len() >= 2 {
                    let u_full = self.state[index].momentum.gather_other_tensor_like_me(u);
                    let rest: i64 = shape_full[1..].iter().product();
                    let u2d = u_full.reshape([shape_full[0], rest]);
                    let mut o = Self::newton_schulz(&u2d, self.ns_steps);
                    o = o.reshape(shape_full.as_slice());
                    var.shard_other_tensor_like_me(o)
                } else {
                    // 1D / scalar fallback: no orthogonalization, scale stays 1.0.
                    var.shard_other_tensor_like_me(u)
                }
            };

            // 6. Accumulate the (unscaled) direction into the transport buffer.
            {
                let mut transport = self.state[index].transport.logical_tensor();
                let _t = transport.g_add_(&direction);
            }

            // Optional transport-buffer decay (default 1.0; momentum lives in M, so the
            // error-feedback residual is not normally decayed).
            if self.compression_decay != 1.0 {
                let mut transport = self.state[index].transport.logical_tensor();
                let _t = transport.g_mul_scalar_(self.compression_decay);
            }

            // 7. Compress & transmit (identical channel to Distro).
            let full_transport = self.state[index].transport.gather_full_tensor();
            let transport_energy: Option<f64> = match stats {
                true => Some(
                    full_transport
                        .norm_scalaropt_dtype(1, Kind::Float)
                        .try_into()
                        .unwrap(),
                ),
                _ => None,
            };
            let (sparse_idx, sparse_val, xshape, totalk) = CompressDCT::compress(
                &self.transform.encode(&full_transport),
                self.compression_topk,
            );

            ret.push(DistroResult {
                sparse_idx,
                sparse_val,
                xshape,
                totalk,
                stats: match stats {
                    true => {
                        let name = var.name().to_string();
                        Some(HashMap::from([
                            (
                                format!("{name}.transport_energy"),
                                transport_energy.unwrap(),
                            ),
                            (format!("{name}.grad_energy"), grad_energy.unwrap()),
                        ]))
                    }
                    false => None,
                },
            });
        }
        ret
    }

    pub fn apply(&mut self, vars: &dyn CausalLM, results: &[Vec<DistroResult>], lr: f64) {
        let _no_grad = tch::no_grad_guard();
        if results.is_empty() {
            return;
        }

        for (index, var) in vars.variables().enumerate() {
            let scale = self.scales[index];
            let mut variable = var.logical_tensor();
            let device = variable.device();
            let indicies = results
                .iter()
                .map(|x| x[index].sparse_idx.to_device(device))
                .collect::<Vec<_>>();

            let val_kind = variable.kind();
            let values = results
                .iter()
                .map(|x| {
                    let sparse_val = x[index].sparse_val.to_device(device);
                    if sparse_val.kind() == Kind::Bool {
                        Distro::unpack_tensor_sign_from_boolean(sparse_val, val_kind)
                    } else {
                        sparse_val
                    }
                })
                .collect::<Vec<_>>();

            let decompressed = CompressDCT::batch_decompress(
                &indicies,
                &values,
                &results[0][index].xshape,
                results[0][index].totalk,
                val_kind,
                device,
            );
            let aggregated = self.transform.decode(&decompressed);
            // Shard the aggregated update to the local parameter view, then sign
            // (v1 runs in sign-mode — same wire format & bandwidth as Distro).
            let aggregated_local = var.shard_other_tensor_like_me(aggregated);
            let update = aggregated_local.sign().multiply_scalar(lr * scale);
            let _t = variable.g_sub_(&update);
        }

        for var in vars.variables() {
            var.zero_grad();
        }
    }

    /// Undo the lookahead applied at the end of the previous `generate`, so the
    /// next forward/backward runs on the true (post-`apply`) weights. No-op when
    /// `lookahead` is disabled.
    pub fn error_correction(&mut self, vars: &dyn CausalLM, prev_lr: f64) {
        if !self.lookahead {
            return;
        }
        let _no_grad = tch::no_grad_guard();
        for (index, var) in vars.variables().enumerate() {
            let scale = self.scales[index];
            let mut variable = var.logical_tensor();
            let transport = self.state[index].transport.logical_tensor();
            let _t = variable.g_sub_(&transport.sign().multiply_scalar(prev_lr * scale));
        }
    }

    pub fn zero_optim(&mut self) {
        for state in &mut self.state {
            let _ = state.momentum.logical_tensor().zero_();
            let _ = state.transport.logical_tensor().zero_();
        }
    }
}

unsafe impl Send for MuonOptimizer {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::set_torch_rng_seed;

    #[test]
    fn test_scale_formula() {
        // square matrix → scale 1
        assert!((muon_scale(&[64, 64]) - 1.0).abs() < 1e-9);
        // wide [out=128, in=64] → sqrt(128/64) = sqrt(2)
        assert!((muon_scale(&[128, 64]) - 2.0_f64.sqrt()).abs() < 1e-9);
        // tall [out=64, in=128] → max(1, 0.5) = 1
        assert!((muon_scale(&[64, 128]) - 1.0).abs() < 1e-9);
        // 1D → 1
        assert!((muon_scale(&[7]) - 1.0).abs() < 1e-9);
        // 4D collapses to [a, b*c*d]: [8,2,3,4] → max(1, 8/24)=1
        assert!((muon_scale(&[8, 2, 3, 4]) - 1.0).abs() < 1e-9);
        // [24,2,1,1] → max(1, 24/2)=sqrt(12)
        assert!((muon_scale(&[24, 2, 1, 1]) - 12.0_f64.sqrt()).abs() < 1e-9);
    }

    #[test]
    fn test_newton_schulz_orthogonalizes() {
        // NS of an already-orthogonal matrix should recover (approximately) itself.
        let q = Tensor::eye(32, (Kind::Float, tch::Device::Cpu));
        let o = MuonOptimizer::newton_schulz(&q, 6);
        assert!(
            o.allclose(&q, 0.2, 0.2, false),
            "Newton-Schulz did not recover an orthogonal input"
        );
    }

    #[test]
    fn test_newton_schulz_sign_differs_from_input_sign() {
        // The core Muon-under-sign claim: sign(O) != sign(M) in general, because
        // the orthogonalization mixes rows/columns non-linearly.
        set_torch_rng_seed();
        let g = Tensor::randn([32, 32], (Kind::Float, tch::Device::Cpu));
        let o = MuonOptimizer::newton_schulz(&g, 5);
        let same_sign = g.sign().equal(&o.sign());
        assert!(
            !same_sign,
            "sign(O) equals sign(input) everywhere — orthogonalization had no structural effect"
        );
    }
}
