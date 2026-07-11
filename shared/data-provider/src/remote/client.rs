use aether_core::BatchId;
use aether_network::{SecretKey, TcpClient};
use anyhow::{bail, Context, Result};
use std::time::{Duration, Instant};
use tokio::time::sleep;
use tracing::{trace, warn};

use crate::{TokenizedData, TokenizedDataProvider};

use super::shared::{ClientToServerMessage, ServerToClientMessage};

pub struct DataProviderTcpClient {
    address: String,
    tcp_client: TcpClient<ClientToServerMessage, ServerToClientMessage>,
}

impl DataProviderTcpClient {
    pub async fn connect(addr: String, secret_key: SecretKey) -> Result<Self> {
        const RETRY_INTERVAL: Duration = Duration::from_secs(2);
        const RETRY_TIMEOUT: Duration = Duration::from_secs(30 * 60);

        let started = Instant::now();
        let mut attempt = 1u32;
        let tcp_client = loop {
            match TcpClient::<ClientToServerMessage, ServerToClientMessage>::connect(
                &addr,
                secret_key.clone(),
            )
            .await
            {
                Ok(tcp_client) => break tcp_client,
                Err(err) if started.elapsed() >= RETRY_TIMEOUT => {
                    return Err(err).with_context(|| {
                        format!("timed out connecting to data provider at {addr}")
                    });
                }
                Err(err) => {
                    if attempt == 1 || attempt % 15 == 0 {
                        warn!(
                            address = %addr,
                            attempt,
                            error = %err,
                            "data provider unavailable; retrying"
                        );
                    }
                    attempt += 1;
                    sleep(RETRY_INTERVAL).await;
                }
            }
        };
        Ok(Self {
            tcp_client,
            address: addr.to_owned(),
        })
    }

    async fn receive_training_data(&mut self, data_ids: BatchId) -> Result<Vec<TokenizedData>> {
        self.tcp_client
            .send(ClientToServerMessage::RequestTrainingData { data_ids })
            .await?;

        let message = self.tcp_client.receive().await?;
        match message {
            ServerToClientMessage::TrainingData {
                data_ids: received_id,
                raw_data,
            } => {
                if received_id == data_ids {
                    Ok(raw_data)
                } else {
                    bail!("Received data_id does not match requested data_id")
                }
            }
            e => bail!("Unexpected message from server {:?}", e),
        }
    }

    pub fn address(&self) -> &str {
        &self.address
    }
}

impl TokenizedDataProvider for DataProviderTcpClient {
    async fn get_samples(&mut self, data_ids: BatchId) -> Result<Vec<TokenizedData>> {
        trace!("[{:?}] get samples..", self.tcp_client.get_identity());
        self.receive_training_data(data_ids).await
    }
}
