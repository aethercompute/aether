use std::f32::consts::PI;

use aether_modeling::{RoPECache, RoPEConfig, RoPEType};
use tch::{Device, Tensor};

fn tensor_values(tensor: &Tensor) -> Vec<f32> {
    Vec::<f32>::try_from(tensor.to_device(Device::Cpu).contiguous().view([-1]))
        .expect("convert tensor to values")
}

fn assert_values_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance,
            "value {index} differs: actual={actual}, expected={expected}"
        );
    }
}

fn default_frequencies(head_dim: usize, theta: f32) -> Vec<f32> {
    (0..head_dim)
        .step_by(2)
        .map(|index| theta.powf(-(index as f32 / head_dim as f32)))
        .collect()
}

#[test]
fn rope_matches_independent_formula_at_zero_maximum_and_beyond_limit() {
    let cache = RoPECache::new(&None, 4, 10_000.0, &Device::Cpu);
    let input = Tensor::from_slice(&[
        1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ])
    .reshape([1, 1, 3, 4]);
    let positions = Tensor::from_slice(&[0_i64, 16, 17]).reshape([1, 3]);

    let actual = cache.apply_rotary_emb(&input, Some(&positions));
    let input_values = tensor_values(&input);
    let frequencies = [1.0_f32, 0.01];
    let mut expected = Vec::with_capacity(input_values.len());
    for (sequence, position) in [0.0_f32, 16.0, 17.0].into_iter().enumerate() {
        let values = &input_values[sequence * 4..sequence * 4 + 4];
        let rotated = [-values[2], -values[3], values[0], values[1]];
        for dimension in 0..4 {
            let frequency = frequencies[dimension % 2];
            expected.push(
                values[dimension] * (position * frequency).cos()
                    + rotated[dimension] * (position * frequency).sin(),
            );
        }
    }

    assert_values_close(&tensor_values(&actual), &expected, 1e-5);
    assert_values_close(
        &tensor_values(&actual.narrow(2, 0, 1)),
        &input_values[..4],
        0.0,
    );
}

#[test]
fn llama3_rope_scaling_matches_independent_frequency_formula() {
    let head_dim = 8;
    let theta = 10_000.0;
    let factor = 4.0;
    let original_context = 64.0;
    let low_factor = 1.0;
    let high_factor = 4.0;
    let config = RoPEConfig {
        factor: Some(factor),
        low_freq_factor: Some(low_factor),
        high_freq_factor: Some(high_factor),
        original_max_position_embeddings: Some(original_context as usize),
        rope_type: RoPEType::Llama3,
        ..Default::default()
    };

    let cache = RoPECache::new(&Some(config), head_dim, theta, &Device::Cpu);
    let expected = default_frequencies(head_dim, theta)
        .into_iter()
        .map(|frequency| {
            let wavelength = 2.0 * PI / frequency;
            if wavelength < original_context / high_factor {
                frequency
            } else if wavelength > original_context / low_factor {
                frequency / factor
            } else {
                let smooth =
                    (original_context / wavelength - low_factor) / (high_factor - low_factor);
                (1.0 - smooth) * frequency / factor + smooth * frequency
            }
        })
        .collect::<Vec<_>>();

    assert_values_close(&tensor_values(&cache.inv_freq), &expected, 1e-7);
    assert_eq!(cache.mscale, 1.0);
}

fn correction_dimension(
    rotations: f32,
    dimension: usize,
    theta: f32,
    original_context: usize,
) -> f32 {
    dimension as f32 * (original_context as f32 / (rotations * 2.0 * PI)).ln() / (2.0 * theta.ln())
}

fn yarn_mscale(scale: f32, multiplier: f32) -> f32 {
    if scale <= 1.0 {
        1.0
    } else {
        0.1 * multiplier * scale.ln() + 1.0
    }
}

#[test]
fn yarn_rope_scaling_matches_independent_frequency_and_mscale_formulas() {
    let head_dim = 8;
    let theta = 10_000.0;
    let factor = 4.0;
    let original_context = 64;
    let beta_fast = 32.0;
    let beta_slow = 1.0;
    let mscale = 0.5;
    let mscale_all_dim = 1.0;
    let config = RoPEConfig {
        factor: Some(factor),
        original_max_position_embeddings: Some(original_context),
        rope_type: RoPEType::YaRN,
        beta_fast: Some(beta_fast),
        beta_slow: Some(beta_slow),
        mscale: Some(mscale),
        mscale_all_dim: Some(mscale_all_dim),
        ..Default::default()
    };

    let cache = RoPECache::new(&Some(config), head_dim, theta, &Device::Cpu);
    let extra = default_frequencies(head_dim, theta);
    let interpolated = default_frequencies(head_dim, theta * factor);
    let low = correction_dimension(beta_fast, head_dim, theta, original_context).floor() as usize;
    let high = (correction_dimension(beta_slow, head_dim, theta, original_context).ceil() as usize)
        .min(head_dim - 1);
    let ramp_max = if low == high { high + 1 } else { high };
    let expected = extra
        .iter()
        .zip(interpolated)
        .enumerate()
        .map(|(index, (extra, interpolated))| {
            let ramp =
                ((index as f32 - low as f32) / (ramp_max as f32 - low as f32)).clamp(0.0, 1.0);
            interpolated * ramp + extra * (1.0 - ramp)
        })
        .collect::<Vec<_>>();
    let expected_mscale = yarn_mscale(factor, mscale) / yarn_mscale(factor, mscale_all_dim);

    assert_values_close(&tensor_values(&cache.inv_freq), &expected, 1e-7);
    assert!((cache.mscale - expected_mscale as f64).abs() < 1e-7);
}
