use aether_coordinator::{model, Coordinator, HealthChecks, Witness, WitnessMetadata};
use anyhow::Result;
use serde::{Deserialize, Serialize};

#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OpportunisticData {
    WitnessStep(Witness, WitnessMetadata),
    WarmupStep(Witness),
}

impl OpportunisticData {
    pub fn kind(&self) -> &'static str {
        match self {
            OpportunisticData::WitnessStep(..) => "witness",
            OpportunisticData::WarmupStep(..) => "warmup",
        }
    }
}

#[async_trait::async_trait]
pub trait Backend: Send + Sync {
    /// # Cancel safety
    ///
    /// This method must be cancel safe.
    async fn wait_for_new_state(&mut self) -> Result<Coordinator>;
    async fn send_witness(&mut self, opportunistic_data: OpportunisticData) -> Result<()>;
    async fn send_health_check(&mut self, health_check: HealthChecks) -> Result<()>;
    async fn send_checkpoint(&mut self, checkpoint: model::Checkpoint) -> Result<()>;

    /// Called once the client has finished downloading and loading the
    /// checkpoint and is ready to be admitted into the next epoch.
    ///
    /// Default no-op: architectures that handle admission differently (e.g.
    /// on-chain coordinators) can ignore this.
    async fn send_ready_for_epoch(&mut self) -> Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn witness_step_kind_is_witness() {
        let dummy_witness = Witness::default();
        let data = OpportunisticData::WitnessStep(dummy_witness, WitnessMetadata::default());
        assert_eq!(data.kind(), "witness");
    }

    #[test]
    fn warmup_step_kind_is_warmup() {
        let dummy_witness = Witness::default();
        let data = OpportunisticData::WarmupStep(dummy_witness);
        assert_eq!(data.kind(), "warmup");
    }
}
