//! Shared test helpers for the aether workspace.
//!
//! Pulled in as a `[dev-dependencies]` entry by crates that want:
//!   - reproducible randomness ([`seeded_rng`]),
//!   - race-free loopback listeners ([`bind_unused_loopback`]),
//!   - deterministic tensors and numerical assertions ([`deterministic_tensor`],
//!     [`assert_tensors_close`], [`assert_tensor_finite`]),
//!   - one-shot serialization round-trip checks ([`assert_postcard_roundtrip`],
//!     [`assert_serde_json_roundtrip`]).
//!
//! Keeping these here avoids duplicating the same boilerplate in every crate's
//! `#[cfg(test)]` block.

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::net::{Ipv4Addr, SocketAddr, TcpListener};
use std::time::Duration;
use tch::{Device, Kind, Tensor};

/// A deterministic RNG seeded from a `u64`.
///
/// Tests that exercise randomness but must be reproducible (and thus parallel-safe
/// under `cargo test`) should derive all randomness from a fixed seed via this.
pub fn seeded_rng(seed: u64) -> ChaCha8Rng {
    ChaCha8Rng::seed_from_u64(seed)
}

/// Binds an ephemeral loopback port and returns the listener that reserves it.
///
/// Keep the listener alive until the test server takes ownership; returning a
/// bare port number would allow another process to claim it in between.
pub fn bind_unused_loopback() -> std::io::Result<(TcpListener, SocketAddr)> {
    let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0))?;
    let address = listener.local_addr()?;
    Ok((listener, address))
}

/// Polls until `operation` returns `expected`, bounded by one total deadline.
pub async fn assert_eventually_eq<T, F, Fut>(
    timeout: Duration,
    interval: Duration,
    mut operation: F,
    expected: T,
) where
    T: PartialEq + std::fmt::Debug,
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = T>,
{
    let deadline = tokio::time::Instant::now() + timeout;

    loop {
        let result = tokio::time::timeout_at(deadline, operation())
            .await
            .unwrap_or_else(|_| panic!("poll operation exceeded its total deadline"));
        if result == expected {
            return;
        }

        let now = tokio::time::Instant::now();
        assert!(
            now < deadline,
            "poll deadline elapsed; got {result:?}, expected {expected:?}"
        );
        tokio::time::sleep_until(std::cmp::min(deadline, now + interval)).await;
    }
}

/// Builds a CPU tensor containing `0..numel`, reshaped and converted to `kind`.
pub fn deterministic_tensor(shape: &[i64], kind: Kind) -> Tensor {
    assert!(shape.iter().all(|dimension| *dimension >= 0));
    let numel = shape
        .iter()
        .try_fold(1_i64, |total, dimension| total.checked_mul(*dimension));
    let numel = numel.expect("tensor shape element count overflowed i64");
    Tensor::arange(numel, (Kind::Int64, Device::Cpu))
        .reshape(shape)
        .to_kind(kind)
}

/// Returns `(relative, absolute)` tolerances suitable for a floating-point dtype.
pub fn tensor_tolerances(kind: Kind) -> (f64, f64) {
    match kind {
        Kind::Double => (1e-9, 1e-12),
        Kind::Float => (1e-5, 1e-6),
        Kind::Half | Kind::BFloat16 => (1e-2, 1e-3),
        _ => panic!("no numerical tolerance configured for {kind:?}"),
    }
}

/// Asserts equal shape and dtype, then compares values with dtype-specific tolerances.
pub fn assert_tensors_close(actual: &Tensor, expected: &Tensor) {
    assert_eq!(actual.size(), expected.size(), "tensor shapes differ");
    assert_eq!(actual.kind(), expected.kind(), "tensor dtypes differ");
    let (relative, absolute) = tensor_tolerances(actual.kind());
    assert!(
        actual.allclose(expected, relative, absolute, false),
        "tensors differ beyond rtol={relative} and atol={absolute}"
    );
}

/// Asserts that every tensor value is finite.
pub fn assert_tensor_finite(tensor: &Tensor) {
    assert_eq!(
        tensor.isfinite().all().int64_value(&[]),
        1,
        "tensor contains NaN or infinity"
    );
}

/// Serialize `value` with postcard, deserialize it, and return the result.
/// Also asserts serialization is byte-deterministic.
///
/// Catches silent wire-format regressions on types that cross the network.
pub fn postcard_roundtrip<T>(value: &T) -> T
where
    T: serde::Serialize + serde::de::DeserializeOwned,
{
    let bytes = postcard::to_allocvec(value).expect("postcard serialize");
    let bytes2 = postcard::to_allocvec(value).expect("postcard re-serialize");
    assert_eq!(
        bytes, bytes2,
        "postcard serialization is not byte-deterministic"
    );
    postcard::from_bytes(&bytes).expect("postcard deserialize")
}

/// Serialize `value` with postcard, deserialize it, and assert the result equals
/// the original.
pub fn assert_postcard_roundtrip<T>(value: &T)
where
    T: serde::Serialize + serde::de::DeserializeOwned + PartialEq + std::fmt::Debug,
{
    let back = postcard_roundtrip(value);
    assert_eq!(*value, back, "postcard round-trip changed the value");
}

/// Serialize `value` with serde_json, deserialize it, and assert equality.
pub fn assert_serde_json_roundtrip<T>(value: &T)
where
    T: serde::Serialize + serde::de::DeserializeOwned + PartialEq + std::fmt::Debug,
{
    let s = serde_json::to_string(value).expect("serde_json serialize");
    let back: T = serde_json::from_str(&s).expect("serde_json deserialize");
    assert_eq!(*value, back, "serde_json round-trip changed the value");
}

#[cfg(test)]
mod tests {
    use rand::Rng;

    use super::*;

    #[derive(serde::Serialize, serde::Deserialize, PartialEq, Debug)]
    struct Pair(u32, String);

    #[test]
    fn roundtrip_helpers_work() {
        assert_postcard_roundtrip(&Pair(7, "hi".into()));
        assert_serde_json_roundtrip(&Pair(7, "hi".into()));
    }

    #[test]
    fn seeded_rng_is_deterministic() {
        let mut a = seeded_rng(42);
        let mut b = seeded_rng(42);
        for _ in 0..16 {
            assert_eq!(a.random::<u64>(), b.random::<u64>());
        }
    }

    #[test]
    fn loopback_helper_reserves_port_until_listener_is_dropped() {
        let (listener, address) = bind_unused_loopback().unwrap();
        assert!(address.ip().is_loopback());
        assert_ne!(address.port(), 0);
        assert!(TcpListener::bind(address).is_err());

        drop(listener);
        TcpListener::bind(address).expect("port should be reusable after listener drop");
    }

    #[tokio::test(start_paused = true)]
    async fn eventually_eq_uses_fixed_interval_until_success() {
        let start = tokio::time::Instant::now();
        let mut attempts = 0;
        assert_eventually_eq(
            Duration::from_secs(1),
            Duration::from_millis(10),
            || {
                attempts += 1;
                std::future::ready(attempts)
            },
            3,
        )
        .await;
        assert_eq!(attempts, 3);
        assert_eq!(start.elapsed(), Duration::from_millis(20));
    }

    #[tokio::test(start_paused = true)]
    #[should_panic(expected = "poll deadline elapsed")]
    async fn eventually_eq_fails_at_total_deadline() {
        assert_eventually_eq(
            Duration::from_millis(25),
            Duration::from_millis(10),
            || std::future::ready(false),
            true,
        )
        .await;
    }

    #[tokio::test(start_paused = true)]
    #[should_panic(expected = "poll operation exceeded its total deadline")]
    async fn eventually_eq_bounds_blocked_operations() {
        assert_eventually_eq(
            Duration::from_millis(25),
            Duration::from_millis(10),
            || std::future::pending::<bool>(),
            true,
        )
        .await;
    }

    #[test]
    fn deterministic_tensor_has_expected_shape_dtype_and_values() {
        let tensor = deterministic_tensor(&[2, 3], Kind::Float);
        assert_eq!(tensor.size(), [2, 3]);
        assert_eq!(tensor.kind(), Kind::Float);
        assert_eq!(
            Vec::<f32>::try_from(tensor.view([-1])).unwrap(),
            [0., 1., 2., 3., 4., 5.]
        );
    }

    #[test]
    fn tensor_tolerances_reflect_dtype_precision() {
        let (double_relative, double_absolute) = tensor_tolerances(Kind::Double);
        let (float_relative, float_absolute) = tensor_tolerances(Kind::Float);
        let (half_relative, half_absolute) = tensor_tolerances(Kind::Half);
        assert!(double_relative < float_relative && float_relative < half_relative);
        assert!(double_absolute < float_absolute && float_absolute < half_absolute);
    }

    #[test]
    fn tensor_assertions_accept_close_finite_values() {
        let actual = Tensor::from_slice(&[1.0_f32, 2.0]);
        let expected = Tensor::from_slice(&[1.0_f32 + 1e-7, 2.0]);
        assert_tensors_close(&actual, &expected);
        assert_tensor_finite(&actual);
    }

    #[test]
    fn tensor_assertions_reject_large_differences_and_non_finite_values() {
        let mismatch = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            assert_tensors_close(
                &Tensor::from_slice(&[1.0_f32]),
                &Tensor::from_slice(&[2.0_f32]),
            );
        }));
        assert!(mismatch.is_err());

        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                assert_tensor_finite(&Tensor::from_slice(&[value]));
            }));
            assert!(result.is_err(), "expected {value:?} to be rejected");
        }
    }
}
