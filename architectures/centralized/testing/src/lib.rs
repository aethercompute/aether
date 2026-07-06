pub mod client;
pub mod server;
pub mod test_utils;

// Model Parameters
pub const WARMUP_TIME: u64 = 20;
pub const MAX_ROUND_TRAIN_TIME: u64 = 5;
pub const ROUND_WITNESS_TIME: u64 = 2;
pub const COOLDOWN_TIME: u64 = 3;

/// Runs an async test future on a multi-threaded tokio runtime whose worker
/// threads carry a large (64 MB) stack.
///
/// The default `#[tokio::test]` runtime uses 2 MB worker stacks, which is
/// insufficient for the deeply-nested `select!` state machines in the server
/// (`poll_next`) and client main loops when compiled in debug mode — the
/// futures overflow the stack and abort. Use this in place of
/// `#[tokio::test(flavor = "multi_thread")]`:
///
/// ```ignore
/// #[test_log::test]
/// fn my_test() {
///     aether_centralized_testing::run_test(async {
///         // … async test body …
///     });
/// }
/// ```
pub fn run_test<F>(future: F) -> F::Output
where
    F: std::future::Future,
{
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .thread_stack_size(64 * 1024 * 1024)
        .build()
        .expect("failed to build test runtime");
    runtime.block_on(future)
}
