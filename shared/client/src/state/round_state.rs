use crate::{fetch_data::BatchIdSet, Finished, TrainingResult};

use aether_coordinator::{
    Commitment, CommitteeProof, CommitteeSelection, WitnessBloom, WitnessProof,
};
use aether_core::{BatchId, NodeIdentity};
use aether_modeling::DistroResult;
use std::{
    collections::{BTreeMap, HashMap},
    sync::{Arc, Mutex},
};
use tracing::warn;

use super::types::PayloadState;

pub struct RoundState {
    pub height: u32,
    pub step: u32,
    pub sent_witness: bool,
    pub sent_finished: bool,
    pub downloads: Arc<Mutex<HashMap<aether_network::Hash, PayloadState>>>,
    #[allow(clippy::type_complexity)]
    pub results: HashMap<BatchId, Vec<(NodeIdentity, (Commitment, TrainingResult))>>,
    pub clients_finished: HashMap<NodeIdentity, Finished>,
    pub data_assignments: BTreeMap<BatchId, NodeIdentity>,
    pub blooms: Arc<Mutex<Option<(WitnessBloom, WitnessBloom)>>>,
    pub broadcasts: Vec<[u8; 32]>,
    pub committee_info: Option<(CommitteeProof, WitnessProof, CommitteeSelection)>,
    pub batch_ids_not_yet_trained_on: Arc<Mutex<Option<BatchIdSet>>>,
    pub self_distro_results: Vec<Vec<DistroResult>>,
}

impl RoundState {
    pub fn new() -> Self {
        Self {
            height: 0,
            step: 0,
            sent_witness: false,
            sent_finished: false,
            downloads: Arc::new(Mutex::new(HashMap::new())),
            results: HashMap::new(),
            broadcasts: Vec::new(),
            clients_finished: HashMap::new(),
            data_assignments: BTreeMap::new(),
            blooms: Arc::new(Mutex::new(None)),
            committee_info: None,
            batch_ids_not_yet_trained_on: Arc::new(Mutex::new(None)),
            self_distro_results: vec![],
        }
    }

    pub fn distro_result_blob_downloaded(&self, hash: &aether_network::Hash) -> bool {
        self.downloads
            .lock()
            .unwrap_or_else(|poisoned| {
                warn!("round downloads lock poisoned; recovering state");
                poisoned.into_inner()
            })
            .contains_key(hash)
    }
}

impl Default for RoundState {
    fn default() -> Self {
        RoundState::new()
    }
}

#[cfg(test)]
mod tests {
    use super::RoundState;

    #[test]
    fn new_round_state_starts_empty() {
        let rs = RoundState::new();
        assert_eq!(rs.height, 0);
        assert_eq!(rs.step, 0);
        assert!(!rs.sent_witness);
        assert!(!rs.sent_finished);
        assert!(rs.broadcasts.is_empty());
        assert!(rs.results.is_empty());
        assert!(rs.clients_finished.is_empty());
        assert!(rs.data_assignments.is_empty());
        assert!(rs.self_distro_results.is_empty());
        assert!(rs.committee_info.is_none());
    }

    #[test]
    fn default_equals_new() {
        // Default delegates to new(); keep that contract explicit so that
        // RoundState::default() never silently diverges.
        let new = RoundState::new();
        let default = RoundState::default();
        assert_eq!(new.height, default.height);
        assert_eq!(new.step, default.step);
        assert_eq!(new.sent_witness, default.sent_witness);
        assert_eq!(new.sent_finished, default.sent_finished);
    }

    #[test]
    fn distro_result_blob_downloaded_empty_for_new_state() {
        let rs = RoundState::new();
        let hash = aether_network::Hash::from_bytes([1u8; 32]);
        // Nothing inserted yet -> nothing reported as downloaded.
        assert!(!rs.distro_result_blob_downloaded(&hash));
        // The downloads map starts empty.
        assert!(rs.downloads.lock().expect("test lock poisoned").is_empty());
    }

    #[test]
    fn distro_result_blob_downloaded_recovers_from_poisoned_downloads_lock() {
        let rs = RoundState::new();
        let hash = aether_network::Hash::from_bytes([1u8; 32]);

        let _ = std::panic::catch_unwind(|| {
            let _guard = rs.downloads.lock().expect("test lock should start clean");
            panic!("poison downloads lock");
        });

        assert!(!rs.distro_result_blob_downloaded(&hash));
    }
}
