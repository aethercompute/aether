use std::str::FromStr;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IntegrationTestLogMarker {
    StateChange,
    Loss,
    LoadedModel,
    HealthCheck,
    UntrainedBatches,
    SolanaSubscription,
    WitnessElected,
    Error,
    RpcFallback,
}

impl std::fmt::Display for IntegrationTestLogMarker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Self::StateChange => "state_change",
                Self::Loss => "loss",
                Self::LoadedModel => "loaded_model",
                Self::HealthCheck => "health_check",
                Self::UntrainedBatches => "untrained_batches",
                Self::SolanaSubscription => "solana_subscription",
                Self::WitnessElected => "witness_elected",
                Self::Error => "error",
                Self::RpcFallback => "rpc_fallback",
            }
        )
    }
}

impl FromStr for IntegrationTestLogMarker {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
            "state_change" => Self::StateChange,
            "loss" => Self::Loss,
            "loaded_model" => Self::LoadedModel,
            "health_check" => Self::HealthCheck,
            "untrained_batches" => Self::UntrainedBatches,
            "solana_subscription" => Self::SolanaSubscription,
            "witness_elected" => Self::WitnessElected,
            "error" => Self::Error,
            "rpc_fallback" => Self::RpcFallback,
            _ => return Err(()),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn markers_roundtrip_through_display_and_from_str() {
        let markers = [
            IntegrationTestLogMarker::StateChange,
            IntegrationTestLogMarker::Loss,
            IntegrationTestLogMarker::LoadedModel,
            IntegrationTestLogMarker::HealthCheck,
            IntegrationTestLogMarker::UntrainedBatches,
            IntegrationTestLogMarker::SolanaSubscription,
            IntegrationTestLogMarker::WitnessElected,
            IntegrationTestLogMarker::Error,
            IntegrationTestLogMarker::RpcFallback,
        ];

        for marker in markers {
            let encoded = marker.to_string();
            assert_eq!(encoded.parse::<IntegrationTestLogMarker>(), Ok(marker));
        }
    }

    #[test]
    fn marker_strings_are_stable() {
        assert_eq!(
            IntegrationTestLogMarker::StateChange.to_string(),
            "state_change"
        );
        assert_eq!(IntegrationTestLogMarker::Loss.to_string(), "loss");
        assert_eq!(
            IntegrationTestLogMarker::LoadedModel.to_string(),
            "loaded_model"
        );
        assert_eq!(
            IntegrationTestLogMarker::HealthCheck.to_string(),
            "health_check"
        );
        assert_eq!(
            IntegrationTestLogMarker::UntrainedBatches.to_string(),
            "untrained_batches"
        );
        assert_eq!(
            IntegrationTestLogMarker::SolanaSubscription.to_string(),
            "solana_subscription"
        );
        assert_eq!(
            IntegrationTestLogMarker::WitnessElected.to_string(),
            "witness_elected"
        );
        assert_eq!(IntegrationTestLogMarker::Error.to_string(), "error");
        assert_eq!(
            IntegrationTestLogMarker::RpcFallback.to_string(),
            "rpc_fallback"
        );
    }

    #[test]
    fn unknown_marker_is_rejected() {
        assert_eq!("".parse::<IntegrationTestLogMarker>(), Err(()));
        assert_eq!("STATE_CHANGE".parse::<IntegrationTestLogMarker>(), Err(()));
        assert_eq!("unknown".parse::<IntegrationTestLogMarker>(), Err(()));
    }
}
