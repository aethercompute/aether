use std::time::Duration;

use crate::client::ClientHandle;
use crate::server::CoordinatorServerHandle;
use aether_client::TrainArgs;
use clap::Parser;
use rand::distr::{Alphanumeric, SampleString};
use std::env;
use tokio_util::sync::CancellationToken;

pub fn repo_path() -> String {
    let cargo_manifest_dir = env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set");
    std::path::Path::new(&cargo_manifest_dir)
        .ancestors()
        .nth(3)
        .expect("Failed to determine repository root")
        .to_str()
        .unwrap()
        .to_string()
}

pub async fn spawn_clients(
    num_clients: usize,
    server_port: u16,
    run_id: &str,
) -> Vec<ClientHandle> {
    let mut client_handles = Vec::new();
    for _ in 0..num_clients {
        client_handles.push(ClientHandle::default(server_port, run_id).await)
    }
    client_handles
}

pub async fn spawn_clients_with_training_delay(
    num_clients: usize,
    server_port: u16,
    run_id: &str,
    training_delay_secs: u64,
) -> Vec<ClientHandle> {
    let mut client_handles = Vec::new();
    for _ in 0..num_clients {
        client_handles.push(
            ClientHandle::new_with_training_delay(server_port, run_id, training_delay_secs).await,
        )
    }
    client_handles
}

pub async fn assert_with_retries<T, F, Fut>(function: F, y: T)
where
    T: PartialEq + std::fmt::Debug,
    Fut: std::future::Future<Output = T>,
    F: FnMut() -> Fut,
{
    aether_test_support::assert_eventually_eq(
        Duration::from_secs(50),
        Duration::from_millis(10),
        function,
        y,
    )
    .await;
}

pub fn sample_rand_run_id() -> String {
    Alphanumeric.sample_string(&mut rand::rng(), 16)
}

/// Sums the healthy score of all nodes and assert it vs expected_score
pub async fn assert_witnesses_healthy_score(
    server_handle: &CoordinatorServerHandle,
    round_number: usize,
    expected_score: u16,
) {
    let clients = server_handle.get_clients().await;

    // get witnesses
    let rounds = server_handle.get_rounds().await;
    let witnesses = &rounds[round_number].witnesses;

    // calculate score
    let mut score = 0;
    clients.iter().for_each(|client| {
        score += aether_coordinator::Coordinator::trainer_healthy_score_by_witnesses(
            &client.id, witnesses,
        );
    });

    assert_eq!(
        score, expected_score,
        "Score {score} != expected score {expected_score}"
    );
}

pub struct AppParams {
    pub(crate) cancel: CancellationToken,
    pub(crate) server_addr: String,
    pub(crate) train_args: TrainArgs,
}

#[derive(Parser)]
struct TestCli {
    #[command(flatten)]
    train_args: TrainArgs,
}

#[rustfmt::skip]
pub fn dummy_client_app_params_with_training_delay(
    server_port: u16,
    run_id: &str,
    training_delay_secs: u64,
) -> AppParams {
    AppParams {
        cancel: CancellationToken::default(),
        server_addr: format!("localhost:{server_port}").to_string(),
        train_args: TestCli::parse_from([
            "dummy",
            "--run-id", run_id,
            "--iroh-relay", "disabled",
            "--iroh-discovery", "local",
            "--data-parallelism", "1",
            "--tensor-parallelism", "1",
            "--micro-batch-size", "1",
            "--max-concurrent-parameter-requests", "10",
            "--hub-max-concurrent-downloads", "1",
            "--dummy-training-delay-secs", training_delay_secs.to_string().as_str(),
        ])
        .train_args,
    }
}
