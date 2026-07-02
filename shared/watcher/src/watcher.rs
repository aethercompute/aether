use crate::traits::Backend;
use anyhow::Result;
use psyche_coordinator::{Client, Coordinator, RunState};
use std::collections::HashMap;
use std::hash::{DefaultHasher, Hasher};

pub struct BackendWatcher<B>
where
    B: Backend + Send + 'static,
{
    backend: B,
    client_lookup: HashMap<[u8; 32], Client>,
    state: Option<(Coordinator, u64)>,
}

impl<B> BackendWatcher<B>
where
    B: Backend + Send + 'static,
{
    pub fn new(backend: B) -> Self {
        Self {
            backend,
            client_lookup: HashMap::new(),
            state: None,
        }
    }

    /// # Cancel safety
    ///
    /// This method is cancel safe. If `poll_next` is used as the event in a
    /// [`tokio::select!`](crate::select) statement and some other branch
    /// completes first, it is guaranteed that no state changes are missed.
    pub async fn poll_next(&mut self) -> Result<(Option<(Coordinator, u64)>, &(Coordinator, u64))> {
        let new_state = self.backend.wait_for_new_state().await?;
        let mut hasher = DefaultHasher::new();
        hasher.write(bytemuck::bytes_of(&new_state));
        let new_state_hash = hasher.finish();
        if new_state.run_state == RunState::Warmup {
            self.client_lookup = HashMap::from_iter(
                new_state
                    .epoch_state
                    .clients
                    .iter()
                    .map(|client| (*client.id.p2p_identity(), *client)),
            );
        }
        let old_state = self.state.replace((new_state, new_state_hash));
        let new_state = self.state.as_ref().unwrap();

        Ok((old_state, new_state))
    }

    pub fn coordinator_state(&self) -> Option<Coordinator> {
        self.state.as_ref().map(|c| c.0)
    }

    pub fn backend(&self) -> &B {
        &self.backend
    }

    pub fn backend_mut(&mut self) -> &mut B {
        &mut self.backend
    }

    pub fn get_client_for_p2p_public_key(&self, p2p_public_key: &[u8; 32]) -> Option<&Client> {
        self.client_lookup.get(p2p_public_key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::OpportunisticData;
    use async_trait::async_trait;
    use psyche_coordinator::{model, Client, ClientState, HealthChecks, RunState};
    use psyche_core::{FixedVec, NodeIdentity};

    struct MockBackend {
        states: Vec<Coordinator>,
    }

    #[async_trait]
    impl Backend for MockBackend {
        async fn wait_for_new_state(&mut self) -> Result<Coordinator> {
            Ok(self.states.remove(0))
        }

        async fn send_witness(&mut self, _: OpportunisticData) -> Result<()> {
            Ok(())
        }

        async fn send_health_check(&mut self, _: HealthChecks) -> Result<()> {
            Ok(())
        }

        async fn send_checkpoint(&mut self, _: model::Checkpoint) -> Result<()> {
            Ok(())
        }
    }

    fn coord_with_state(run_state: RunState, clients: &[NodeIdentity]) -> Coordinator {
        let mut coord: Coordinator = bytemuck::Zeroable::zeroed();
        coord.run_state = run_state;
        coord.epoch_state.clients = FixedVec::from_iter(clients.iter().map(|id| Client {
            id: *id,
            state: ClientState::Healthy,
            exited_height: 0,
        }));
        coord
    }

    #[tokio::test]
    async fn poll_next_returns_initial_state() {
        let backend = MockBackend {
            states: vec![coord_with_state(RunState::WaitingForMembers, &[])],
        };
        let mut watcher = BackendWatcher::new(backend);

        let (old, new) = watcher.poll_next().await.unwrap();
        assert!(old.is_none());
        assert_eq!(new.0.run_state, RunState::WaitingForMembers);
    }

    #[tokio::test]
    async fn poll_next_returns_previous_and_current_state() {
        let backend = MockBackend {
            states: vec![
                coord_with_state(RunState::WaitingForMembers, &[]),
                coord_with_state(RunState::Warmup, &[]),
            ],
        };
        let mut watcher = BackendWatcher::new(backend);

        let (first_old, _) = watcher.poll_next().await.unwrap();
        assert!(first_old.is_none());

        let (second_old, second_new) = watcher.poll_next().await.unwrap();
        assert!(second_old.is_some());
        assert_eq!(second_old.unwrap().0.run_state, RunState::WaitingForMembers);
        assert_eq!(second_new.0.run_state, RunState::Warmup);
    }

    #[tokio::test]
    async fn coordinator_state_returns_latest_after_poll() {
        let backend = MockBackend {
            states: vec![coord_with_state(RunState::RoundTrain, &[])],
        };
        let mut watcher = BackendWatcher::new(backend);

        assert!(watcher.coordinator_state().is_none());

        watcher.poll_next().await.unwrap();
        assert_eq!(
            watcher.coordinator_state().unwrap().run_state,
            RunState::RoundTrain
        );
    }

    #[tokio::test]
    async fn client_lookup_is_built_on_warmup() {
        let key = [0xabu8; 32];
        let client_id = NodeIdentity::from_single_key(key);
        let backend = MockBackend {
            states: vec![coord_with_state(RunState::Warmup, &[client_id])],
        };
        let mut watcher = BackendWatcher::new(backend);
        watcher.poll_next().await.unwrap();

        let found = watcher.get_client_for_p2p_public_key(&key);
        assert!(found.is_some());
        assert_eq!(found.unwrap().id.p2p_identity(), &key);

        let missing = watcher.get_client_for_p2p_public_key(&[0u8; 32]);
        assert!(missing.is_none());
    }

    #[tokio::test]
    async fn client_lookup_is_not_built_on_non_warmup() {
        let key = [0xabu8; 32];
        let client_id = NodeIdentity::from_single_key(key);
        let backend = MockBackend {
            states: vec![coord_with_state(RunState::WaitingForMembers, &[client_id])],
        };
        let mut watcher = BackendWatcher::new(backend);
        watcher.poll_next().await.unwrap();

        let found = watcher.get_client_for_p2p_public_key(&key);
        assert!(found.is_none());
    }

    #[test]
    fn backend_accessors_return_inner() {
        let backend = MockBackend { states: vec![] };
        let mut watcher = BackendWatcher::new(backend);

        let _backend_ref = watcher.backend();
        let _backend_mut = watcher.backend_mut();
    }
}
