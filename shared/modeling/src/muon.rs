//! Muon optimizer — MomentUm Orthogonalized by Newton-Schulz.
//!
//! Rides the *exact same* compressed-distributed transport as `Distro` (DCT +
//! top-k + error feedback): the transport carries a residual of the gradient
//! **momentum** (or Nesterov momentum), a compressible signal since gradients are
//! smooth/low-rank and the DCT concentrates their energy. The Muon difference is
//! entirely at `apply`:
//!
//!   Distro:        `w -= lr · sign(mean(delta))`            (SignSGD)
//!   Muon (2D):     `w -= lr · scale · NewtonSchulz(mean(e))` (orthogonalized momentum)
//!   fallback:      `w -= lr · FALLBACK_SCALE · sign(mean(e))` (embed/head/1D)
//!
//! Why orthogonalization happens at `apply`, not in `generate`: Newton-Schulz
//! produces a *whitened* update whose energy is spread uniformly (singular values
//! → 1). DCT + top-k assumes the signal is concentrated in a few coefficients —
//! true for gradients, false for orthogonalized matrices. Compressing the
//! orthogonalized update therefore keeps a near-random subset and discards the
//! rest, and the aggregated buffer degenerates to noise. Compressing the raw
//! momentum (this design) preserves the signal; orthogonalizing post-aggregation
//! is also exactly how canonical single-device Muon applies its update.
//!
//! Because the 2D update is the continuous orthogonal factor (entries ~ 1/√d),
//! the per-coordinate step is `lr · scale / √d` — the canonical Muon regime, which
//! wants `lr ≈ 0.02`, ~an order of magnitude above SignSGD/Distro.
//!
//! Reference: https://github.com/KellerJordan/Muon
use crate::{CausalLM, CompressDCT, Distro, DistroResult, TransformDCT, Variable};

use std::collections::HashMap;
use tch::{Kind, Tensor};

/// Keep normal transformer projections (including wide MLP up/gate/down matrices)
/// on Muon, but route vocabulary-shaped tensors to the sign fallback even if their
/// name is unfamiliar.
const MAX_MUON_ASPECT: f64 = 32.0;
/// The sign-fallback effective LR is `lr · FALLBACK_SCALE`, keeping embedding /
/// head / 1D steps on the SignSGD LR scale (~1e-3 at the default Muon lr=0.02)
/// instead of the full orthogonalized-step LR.
const FALLBACK_SCALE: f64 = 0.05;

struct MuonState {
    /// Dense momentum EMA. This is optimizer state, not directly transmitted.
    momentum: Box<dyn Variable>,
    /// Error-feedback residual for the compressed transport. Each round subtracts
    /// the part already sent, optionally decays stale leftovers, then adds the
    /// current momentum/Nesterov signal.
    residual: Box<dyn Variable>,
}

pub struct MuonOptimizer {
    momentum: f64,
    nesterov: bool,
    weight_decay: f64,
    ns_steps: i64,
    lookahead: bool,
    compression_decay: f64,
    compression_topk: i64,
    state: Vec<MuonState>,
    /// Per-parameter Muon scale (`max(1, out/in)^0.5`); meaningful only for
    /// Muon-eligible dense matrix params.
    scales: Vec<f64>,
    /// Per-parameter routing: true -> Newton-Schulz at apply; false -> sign fallback.
    eligible: Vec<bool>,
    transform: TransformDCT,
}

fn collapse_rest(shape: &[i64]) -> i64 {
    shape[1..].iter().product()
}

/// `max(1, out/in)^0.5` with the matrix viewed as `[out, in*…]` (canonical Muon
/// scaling; conv-style leading dim vs. the product of the rest).
fn muon_scale(shape: &[i64]) -> f64 {
    let d0 = shape[0] as f64;
    let rest = collapse_rest(shape) as f64;
    (1.0_f64).max(d0 / rest).sqrt()
}

fn is_named_embedding_or_head(name: &str) -> bool {
    name.contains("embed") || name.contains("lm_head")
}

/// True for dense 2D transformer weights (attention and MLP projections). False
/// for 1D params, embeddings/lm_head, and extreme vocab-shaped matrices.
fn is_muon_eligible(name: &str, shape: &[i64]) -> bool {
    if shape.len() < 2 {
        return false;
    }
    if is_named_embedding_or_head(name) {
        return false;
    }
    let d0 = shape[0] as f64;
    let rest = collapse_rest(shape) as f64;
    if rest == 0.0 {
        return false;
    }
    let aspect = (d0 / rest).max(rest / d0);
    aspect <= MAX_MUON_ASPECT
}

fn momentum_signal(grad: &Tensor, momentum: &Tensor, beta: f64, nesterov: bool) -> Tensor {
    if nesterov {
        grad + &momentum.multiply_scalar(beta)
    } else {
        momentum.shallow_clone()
    }
}

fn decay_residual(residual: &mut Tensor, compression_decay: f64) {
    if compression_decay != 1.0 {
        let _ = residual.g_mul_scalar_(compression_decay);
    }
}

impl MuonOptimizer {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        vs: &dyn CausalLM,
        momentum: f64,
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
        let mut eligible = Vec::new();
        for variable in vs.variables() {
            let name = variable.name().to_string();
            let shape = variable.full_tensor_shape();
            let elig = is_muon_eligible(&name, &shape);
            state.push(MuonState {
                momentum: variable.zeros_like(format!("{name}.muon_momentum")),
                residual: variable.zeros_like(format!("{name}.muon_residual")),
            });
            scales.push(if elig { muon_scale(&shape) } else { 1.0 });
            eligible.push(elig);
            variable.zero_grad();
        }

        let transform = TransformDCT::new(vs.variables(), compression_chunk);

        Self {
            momentum,
            nesterov,
            weight_decay,
            ns_steps,
            lookahead,
            compression_decay,
            compression_topk,
            state,
            scales,
            eligible,
            transform,
        }
    }

    /// Newton-Schulz quintic iteration approximating the zeroth power (orthogonal
    /// polar factor) of a 2D matrix. Coefficients selected to maximize slope at
    /// zero; run in bfloat16 (stable there by construction). Input must be 2D.
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

    /// Per-parameter update `U` such that `w -= lr · U`, computed from the
    /// aggregated transport buffer `ē`. Extracted so the apply math is testable
    /// without a full `CausalLM`.
    ///
    ///   eligible matrix -> `NewtonSchulz(ē) · scale`  (orthogonalized momentum)
    ///   fallback        -> `sign(ē) · FALLBACK_SCALE` (SignSGD-style fallback)
    ///
    /// The result is always a descent direction: `<ē, U>_F > 0`.
    fn param_update(
        ē: &Tensor,
        shape: &[i64],
        scale: f64,
        eligible: bool,
        ns_steps: i64,
    ) -> Tensor {
        if eligible {
            let rest = collapse_rest(shape);
            let o = Self::newton_schulz(&ē.reshape([shape[0], rest]), ns_steps);
            o.reshape(shape).multiply_scalar(scale)
        } else {
            ē.sign().multiply_scalar(FALLBACK_SCALE)
        }
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
            let elig = self.eligible[index];
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

            // 1. Lookahead (flag-gated, off by default): anticipate own residual
            //    direction. Rough — the true applied step is NS-based.
            if self.lookahead {
                let residual = self.state[index].residual.logical_tensor();
                let s = if elig { scale } else { FALLBACK_SCALE };
                let _t = variable.g_add_(&residual.sign().multiply_scalar(prev_lr * s));
            }

            // 2. Error feedback: remove the component already transmitted.
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
                let mut residual = self.state[index].residual.logical_tensor();
                let _t = residual.g_sub_(&var.shard_other_tensor_like_me(transmit_grad));
            }

            // 3. Decay stale error-feedback residual if configured. This is off
            //    by default, but useful when old top-k leftovers become noisy.
            {
                let mut residual = self.state[index].residual.logical_tensor();
                decay_residual(&mut residual, self.compression_decay);
            }

            // 4. Decoupled weight decay.
            if self.weight_decay != 0.0 {
                let _t = variable.g_mul_scalar_(1.0 - lr * self.weight_decay);
            }

            // 5. Momentum EMA, then add the chosen signal to the transport residual.
            let grad = variable.grad();
            {
                let mut momentum = self.state[index].momentum.logical_tensor();
                let _t = momentum.g_mul_scalar_(self.momentum);
                let _t = momentum.g_add_(&grad);

                let signal = momentum_signal(&grad, &momentum, self.momentum, self.nesterov);
                let mut residual = self.state[index].residual.logical_tensor();
                let _t = residual.g_add_(&signal);
            }

            // 6. Compress & transmit the error-feedback residual.
            let full_residual = self.state[index].residual.gather_full_tensor();
            let transport_energy: Option<f64> = match stats {
                true => Some(
                    full_residual
                        .norm_scalaropt_dtype(1, Kind::Float)
                        .try_into()
                        .unwrap(),
                ),
                _ => None,
            };
            let (sparse_idx, sparse_val, xshape, totalk) = CompressDCT::compress(
                &self.transform.encode(&full_residual),
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
            let elig = self.eligible[index];
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
            // Aggregated gradient momentum (full tensor).
            let ē = self.transform.decode(&decompressed);
            let shape = var.full_tensor_shape();
            let update_full = Self::param_update(&ē, &shape, scale, elig, self.ns_steps);
            let update_local = var.shard_other_tensor_like_me(update_full);
            let _t = variable.g_sub_(&update_local.multiply_scalar(lr));
        }

        for var in vars.variables() {
            var.zero_grad();
        }
    }

    /// Undo the previous lookahead (no-op when `lookahead` is disabled).
    pub fn error_correction(&mut self, vars: &dyn CausalLM, prev_lr: f64) {
        if !self.lookahead {
            return;
        }
        let _no_grad = tch::no_grad_guard();
        for (index, var) in vars.variables().enumerate() {
            let scale = self.scales[index];
            let elig = self.eligible[index];
            let mut variable = var.logical_tensor();
            let residual = self.state[index].residual.logical_tensor();
            let s = if elig { scale } else { FALLBACK_SCALE };
            let _t = variable.g_sub_(&residual.sign().multiply_scalar(prev_lr * s));
        }
    }

    pub fn zero_optim(&mut self) {
        for state in &mut self.state {
            let _ = state.momentum.logical_tensor().zero_();
            let _ = state.residual.logical_tensor().zero_();
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
        assert!((muon_scale(&[64, 64]) - 1.0).abs() < 1e-9);
        assert!((muon_scale(&[128, 64]) - 2.0_f64.sqrt()).abs() < 1e-9);
        assert!((muon_scale(&[64, 128]) - 1.0).abs() < 1e-9);
        // conv-style collapse: [a, b, c, d] → [a, b*c*d]
        assert!((muon_scale(&[24, 2, 1, 1]) - 12.0_f64.sqrt()).abs() < 1e-9);
    }

    #[test]
    fn test_eligibility_routing() {
        // Attention projections -> Muon.
        assert!(is_muon_eligible(
            "model.layers.0.self_attn.q_proj.weight",
            &[768, 768]
        ));
        assert!(is_muon_eligible(
            "model.layers.0.self_attn.o_proj.weight",
            &[768, 768]
        ));

        // MLP projections are intentionally Muon-eligible even when wider than 8:1.
        assert!(is_muon_eligible(
            "model.layers.0.mlp.gate_proj.weight",
            &[8192, 768]
        ));
        assert!(is_muon_eligible(
            "model.layers.0.mlp.up_proj.weight",
            &[8192, 768]
        ));
        assert!(is_muon_eligible(
            "model.layers.0.mlp.down_proj.weight",
            &[768, 8192]
        ));

        // 1D/norm params -> fallback.
        assert!(!is_muon_eligible(
            "model.layers.0.input_layernorm.weight",
            &[768]
        ));

        // Embeddings/lm_head -> fallback by name.
        assert!(!is_muon_eligible(
            "model.embed_tokens.weight",
            &[129280, 768]
        ));
        assert!(!is_muon_eligible("lm_head.weight", &[129280, 768]));

        // Unknown extreme vocab-shaped matrices are also protected.
        assert!(!is_muon_eligible("unknown.weight", &[129280, 768]));
    }

    #[test]
    fn test_momentum_signal_respects_nesterov_flag() {
        let grad = Tensor::from_slice(&[1.0_f32, -2.0, 3.0]);
        let momentum = Tensor::from_slice(&[4.0_f32, 5.0, -6.0]);

        let plain = momentum_signal(&grad, &momentum, 0.9, false);
        assert!(plain.equal(&momentum));

        let nesterov = momentum_signal(&grad, &momentum, 0.9, true);
        let expected = Tensor::from_slice(&[4.6_f32, 2.5, -2.4]);
        assert!(nesterov.allclose(&expected, 1e-6, 1e-6, false));
    }

    #[test]
    fn test_decay_residual_is_optional() {
        let mut residual = Tensor::from_slice(&[2.0_f32, -4.0]);
        decay_residual(&mut residual, 1.0);
        assert!(residual.equal(&Tensor::from_slice(&[2.0_f32, -4.0])));

        decay_residual(&mut residual, 0.5);
        assert!(residual.equal(&Tensor::from_slice(&[1.0_f32, -2.0])));
    }

    #[test]
    fn test_newton_schulz_orthogonalizes() {
        let q = Tensor::eye(32, (Kind::Float, tch::Device::Cpu));
        let o = MuonOptimizer::newton_schulz(&q, 6);
        assert!(
            o.allclose(&q, 0.2, 0.2, false),
            "Newton-Schulz did not recover an orthogonal input"
        );
    }

    #[test]
    fn test_newton_schulz_sign_differs_from_input_sign() {
        set_torch_rng_seed();
        let g = Tensor::randn([32, 32], (Kind::Float, tch::Device::Cpu));
        let o = MuonOptimizer::newton_schulz(&g, 5);
        assert!(
            !g.sign().equal(&o.sign()),
            "sign(O) equals sign(input) everywhere — orthogonalization had no structural effect"
        );
    }

    /// The update must be a descent direction: `<gradient_momentum, update>_F > 0`,
    /// so `w -= lr·update` reduces a loss whose gradient is the momentum. This is
    /// the core correctness property that the v1 (compress-the-orthogonalization)
    /// design violated in practice.
    #[test]
    fn test_param_update_is_descent_direction() {
        set_torch_rng_seed();
        let dev = tch::Device::Cpu;

        // Eligible 2D: NS(ē) stays aligned with ē (ēᵀ·NS(ē) = (ēᵀē)^{1/2} ≻ 0).
        let shape = [384, 768];
        let e = Tensor::randn(shape, (Kind::Float, dev));
        let scale = muon_scale(&shape);
        let u = MuonOptimizer::param_update(&e, &shape, scale, true, 5);
        let inner = (&e * &u).sum(Kind::Float).double_value(&[]);
        assert!(
            inner > 0.0,
            "eligible update not a descent direction: {inner}"
        );

        // Fallback (1D): sign(ē) is trivially aligned with ē.
        let shape1d = [384];
        let e1 = Tensor::randn(shape1d, (Kind::Float, dev));
        let u1 = MuonOptimizer::param_update(&e1, &shape1d, 1.0, false, 5);
        let inner1 = (&e1 * &u1).sum(Kind::Float).double_value(&[]);
        assert!(
            inner1 > 0.0,
            "fallback update not a descent direction: {inner1}"
        );

        // Fallback for a wide matrix (embedding-shaped).
        let shapew = [1024, 384];
        let ew = Tensor::randn(shapew, (Kind::Float, dev));
        let uw = MuonOptimizer::param_update(&ew, &shapew, 1.0, false, 5);
        let innerw = (&ew * &uw).sum(Kind::Float).double_value(&[]);
        assert!(
            innerw > 0.0,
            "wide-matrix fallback not a descent direction: {innerw}"
        );
    }

    /// The whole point of v2: the orthogonalized update is built at `apply` from
    /// the *momentum* (compressible), not pre-orthogonalized before compression.
    /// Sanity-check that `param_update` on a typical 2D weight produces a
    /// finite, non-degenerate matrix (no NaNs / all-zero rows).
    #[test]
    fn test_param_update_is_well_formed() {
        set_torch_rng_seed();
        let shape = [64, 64];
        let e = Tensor::randn(shape, (Kind::Float, tch::Device::Cpu)) * 1e-3;
        let u = MuonOptimizer::param_update(&e, &shape, muon_scale(&shape), true, 5);
        assert!(
            u.isfinite().all().double_value(&[]) > 0.0,
            "update has non-finite entries"
        );
        // orthogonalized → roughly unit spectral norm (times scale); entries ~1/√d
        let abs_max = u.abs().max().double_value(&[]);
        assert!(
            abs_max > 0.0 && abs_max < 1.0,
            "unexpected update magnitude: {abs_max}"
        );
    }
}
