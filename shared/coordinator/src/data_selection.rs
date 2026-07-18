use crate::{Committee, CommitteeSelection, Coordinator, Round};

use aether_core::{deterministic_shuffle, BatchId, ClosedInterval, NodeIdentity};
use std::{collections::BTreeMap, fmt};

/// Assigns data batches to nodes based on committee roles.
pub fn assign_data_for_state(
    coordinator: &Coordinator,
    committee_selection: &CommitteeSelection,
) -> BTreeMap<BatchId, NodeIdentity> {
    let round = coordinator.current_round().unwrap();

    let trainer_nodes: Vec<_> = (0..coordinator.epoch_state.clients.len())
        .filter_map(|i| {
            let client = &coordinator.epoch_state.clients[i];
            let committee = committee_selection.get_committee(i as u64).committee;

            matches!(committee, Committee::Trainer).then_some(client)
        })
        .collect();

    if trainer_nodes.is_empty() {
        return BTreeMap::new();
    }

    let mut trainer_nodes = trainer_nodes;
    deterministic_shuffle(&mut trainer_nodes, round.random_seed);

    let total_size = coordinator.get_target_global_batch_size(coordinator.current_round()) as u64;
    let num_trainers = trainer_nodes.len() as u64;
    let base_size = total_size / num_trainers;
    let remainder = total_size % num_trainers;

    let mut assignments = BTreeMap::new();
    let mut current_index = round.data_index;

    for (i, node) in trainer_nodes.iter().enumerate() {
        let node_batch_size = base_size + if (i as u64) < remainder { 1 } else { 0 };

        if node_batch_size > 0 {
            let end_index = current_index + node_batch_size - 1;
            assignments.insert(
                BatchId(ClosedInterval::new(current_index, end_index)),
                node.id,
            );
            current_index = end_index + 1;
        }
    }

    assignments
}

pub fn get_batch_ids_for_round(
    round: &Round,
    coordinator: &Coordinator,
    num_trainer_nodes: u64,
) -> Vec<BatchId> {
    let start = round.data_index;
    let total_size = coordinator.get_target_global_batch_size(Some(round)) as u64;
    let end = start + total_size;

    let base_size = total_size / num_trainer_nodes;
    let remainder = total_size % num_trainer_nodes;

    let mut batch_ids = Vec::with_capacity(num_trainer_nodes as usize);
    let mut current = start;

    for i in 0..num_trainer_nodes {
        let node_size = base_size + if i < remainder { 1 } else { 0 };

        if node_size > 0 {
            let batch_end = current + node_size - 1;
            batch_ids.push(BatchId(ClosedInterval::new(current, batch_end)));
            current = batch_end + 1;

            if current >= end {
                break;
            }
        }
    }

    batch_ids
}

/// Retrieves all batch IDs assigned to a specific node from an interval tree, converting data indices to batches.
pub fn get_batch_ids_for_node<V: fmt::Display + Eq + std::hash::Hash>(
    tree: &BTreeMap<BatchId, V>,
    node_identity: &V,
) -> Vec<BatchId> {
    tree.iter()
        .filter_map(|(interval, assigned_node)| {
            if assigned_node == node_identity {
                Some(*interval)
            } else {
                None
            }
        })
        .collect()
}

pub fn get_data_index_for_step(coordinator: &Coordinator, target_step: u32) -> u64 {
    if target_step <= 1 || target_step > coordinator.config.total_steps {
        return 0;
    }

    let mut current_data_index: u64 = 0;
    let max_seq_len = coordinator.get_sequence_length() as u64;

    for _ in 1..target_step {
        let tokens_processed_before_step = current_data_index * max_seq_len;

        let batch_size_for_step = coordinator
            .config
            .get_batch_size(tokens_processed_before_step) as u64;

        current_data_index += batch_size_for_step;
    }

    current_data_index
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Client, ClientState, CommitteeSelection, Coordinator};
    use aether_core::{FixedVec, NodeIdentity};
    use bytemuck::Zeroable;
    use proptest::prelude::*;

    fn create_test_coordinator(
        num_nodes: usize,
        global_batch_size: u16,
        total_steps: u32,
    ) -> Coordinator {
        let clients: Vec<_> = (0..num_nodes)
            .map(|i| {
                let mut key = [0u8; 32];
                key[0] = i as u8;
                Client {
                    id: NodeIdentity::from_single_key(key),
                    state: ClientState::Healthy,
                    exited_height: 0,
                }
            })
            .collect();

        let mut coordinator = Coordinator::zeroed();
        coordinator.config.total_steps = total_steps;
        coordinator.config.global_batch_size_start = global_batch_size;
        coordinator.config.global_batch_size_end = global_batch_size;
        coordinator.epoch_state.clients = FixedVec::from_iter(clients);

        coordinator.current_round_mut().unwrap().clients_len =
            coordinator.epoch_state.clients.len() as u16;

        coordinator
    }

    #[test]
    fn test_even_distribution() {
        // 4 trainers, global batch size 100 -> each gets 25
        let coordinator = create_test_coordinator(4, 100, 10);

        let assignments = assign_data_for_state(
            &coordinator,
            &CommitteeSelection::from_coordinator(&coordinator, 0).unwrap(),
        );
        assert_eq!(assignments.len(), 4);

        for batch_id in assignments.keys() {
            let size = batch_id.0.end - batch_id.0.start + 1;
            assert_eq!(size, 25);
        }

        let total_assigned: u64 = assignments.keys().map(|b| b.0.end - b.0.start + 1).sum();
        assert_eq!(total_assigned, 100);
    }

    #[test]
    fn test_uneven_distribution_with_remainder() {
        // 24 trainers, global batch size 384
        let coordinator = create_test_coordinator(23, 384, 10);

        let assignments = assign_data_for_state(
            &coordinator,
            &CommitteeSelection::from_coordinator(&coordinator, 0).unwrap(),
        );
        assert_eq!(assignments.len(), 23);

        let mut sizes: Vec<u64> = assignments
            .keys()
            .map(|b| b.0.end - b.0.start + 1)
            .collect();
        sizes.sort();

        let mut expected = [16; 7].to_vec();
        expected.extend([17; 16]);
        assert_eq!(sizes, expected);

        let total: u64 = sizes.iter().sum();
        assert_eq!(total, 384);
    }

    #[test]
    fn test_larger_remainder() {
        // 5 trainers, global batch size 13 -> remainder of 3
        // Expected: base_size=2, so 3 nodes get 3, 2 nodes get 2
        let coordinator = create_test_coordinator(5, 13, 10);

        let assignments = assign_data_for_state(
            &coordinator,
            &CommitteeSelection::from_coordinator(&coordinator, 0).unwrap(),
        );
        assert_eq!(assignments.len(), 5);

        let mut sizes: Vec<u64> = assignments
            .keys()
            .map(|b| b.0.end - b.0.start + 1)
            .collect();
        sizes.sort();

        // Base: 13/5 = 2, remainder: 13%5 = 3
        // First 3 nodes get 3, last 2 get 2
        assert_eq!(sizes, vec![2, 2, 3, 3, 3]);

        let total: u64 = sizes.iter().sum();
        assert_eq!(total, 13);
    }

    // ── partition invariant: assigned batches must be contiguous, non-overlapping
    //    and cover exactly [data_index, data_index + total_size). A gap or
    //    overlap here would silently starve or duplicate training data. ─────────
    fn assert_is_clean_partition(assignments: &BTreeMap<BatchId, NodeIdentity>) {
        let mut intervals: Vec<(u64, u64)> =
            assignments.keys().map(|b| (b.0.start, b.0.end)).collect();
        intervals.sort();
        for w in intervals.windows(2) {
            // no overlap and no gap: prev.end + 1 == next.start
            assert_eq!(
                w[0].1 + 1,
                w[1].0,
                "batches are not contiguous/non-overlapping: {:?}",
                intervals
            );
        }
    }

    #[test]
    fn assign_data_for_state_is_clean_partition_across_shapes() {
        for (num_nodes, batch_size) in [
            (1, 1),
            (1, 100),
            (2, 3),
            (3, 10),
            (5, 13),
            (7, 1000),
            (23, 384),
            (16, 256),
        ] {
            let coordinator = create_test_coordinator(num_nodes, batch_size, 10);
            let assignments = assign_data_for_state(
                &coordinator,
                &CommitteeSelection::from_coordinator(&coordinator, 0).unwrap(),
            );
            assert_eq!(
                assignments.len(),
                num_nodes,
                "expected {num_nodes} assignments for batch_size={batch_size}"
            );
            assert_is_clean_partition(&assignments);

            // total coverage equals the global batch size
            let total: u64 = assignments.keys().map(|b| b.0.end - b.0.start + 1).sum();
            assert_eq!(
                total, batch_size as u64,
                "total coverage for ({num_nodes}, {batch_size})"
            );
        }
    }

    #[test]
    fn assign_data_for_state_skips_non_trainer_committees() {
        let mut coordinator = create_test_coordinator(10, 100, 10);
        coordinator.current_round_mut().unwrap().tie_breaker_tasks = 2;
        coordinator.config.verification_percent = 50;

        let selection = CommitteeSelection::from_coordinator(&coordinator, 0).unwrap();
        let assignments = assign_data_for_state(&coordinator, &selection);

        assert_eq!(selection.get_num_trainer_nodes(), 4);
        assert_eq!(assignments.len(), 4);
        assert_is_clean_partition(&assignments);

        let total: u64 = assignments.keys().map(|b| b.0.end - b.0.start + 1).sum();
        assert_eq!(total, 100);

        for assigned_node in assignments.values() {
            let index = coordinator
                .epoch_state
                .clients
                .iter()
                .position(|client| client.id == *assigned_node)
                .expect("assigned node should exist");
            assert_eq!(
                selection.get_committee(index as u64).committee,
                Committee::Trainer
            );
        }
    }

    #[test]
    fn assign_data_for_state_starts_at_round_data_index() {
        let mut coordinator = create_test_coordinator(4, 100, 10);
        coordinator.current_round_mut().unwrap().data_index = 1234;
        let assignments = assign_data_for_state(
            &coordinator,
            &CommitteeSelection::from_coordinator(&coordinator, 0).unwrap(),
        );
        let first_start = assignments.keys().next().unwrap().0.start;
        assert_eq!(first_start, 1234);
    }

    #[test]
    fn ejected_owner_keeps_inflight_assignment_but_gets_no_future_batches() {
        let mut coordinator = create_test_coordinator(4, 13, 10);
        coordinator.current_round_mut_unchecked().random_seed = 42;
        let selection = CommitteeSelection::from_coordinator(&coordinator, 0).unwrap();
        let inflight = assign_data_for_state(&coordinator, &selection);
        let ejected = *inflight.values().next().unwrap();
        coordinator
            .epoch_state
            .clients
            .iter_mut()
            .find(|client| client.id == ejected)
            .unwrap()
            .state = ClientState::Ejected;

        assert_eq!(assign_data_for_state(&coordinator, &selection), inflight);

        coordinator
            .epoch_state
            .clients
            .retain(|client| client.state == ClientState::Healthy);
        let clients_len = coordinator.epoch_state.clients.len() as u16;
        coordinator.current_round_mut_unchecked().clients_len = clients_len;
        let next_selection = CommitteeSelection::from_coordinator(&coordinator, 0).unwrap();
        let next = assign_data_for_state(&coordinator, &next_selection);

        assert!(!next.values().any(|identity| *identity == ejected));
        assert_is_clean_partition(&next);
        assert_eq!(
            next.keys()
                .map(|batch| batch.0.end - batch.0.start + 1)
                .sum::<u64>(),
            13
        );
    }

    #[test]
    fn replacement_client_assignments_cover_only_current_members() {
        let mut coordinator = create_test_coordinator(4, 13, 10);
        coordinator.current_round_mut_unchecked().random_seed = 42;
        let removed = coordinator.epoch_state.clients.remove(1).unwrap();
        let mut replacement_key = [0_u8; 32];
        replacement_key[0] = 250;
        let replacement = NodeIdentity::from_single_key(replacement_key);
        coordinator
            .epoch_state
            .clients
            .push(Client {
                id: replacement,
                state: ClientState::Healthy,
                exited_height: 0,
            })
            .unwrap();
        let clients_len = coordinator.epoch_state.clients.len() as u16;
        coordinator.current_round_mut_unchecked().clients_len = clients_len;

        let selection = CommitteeSelection::from_coordinator(&coordinator, 0).unwrap();
        let assignments = assign_data_for_state(&coordinator, &selection);
        assert_is_clean_partition(&assignments);
        assert_eq!(
            assignments
                .keys()
                .map(|batch| batch.0.end - batch.0.start + 1)
                .sum::<u64>(),
            13
        );
        assert!(!assignments.values().any(|identity| *identity == removed.id));
        assert!(assignments.values().all(|identity| coordinator
            .epoch_state
            .clients
            .iter()
            .any(|client| client.id == *identity)));
    }

    proptest! {
        #[test]
        fn prop_assignments_cover_exact_range_without_overlap(
            num_nodes in 1_usize..64,
            batch_size in 1_u16..=u16::MAX,
            data_index in 0_u64..=(u64::MAX - u16::MAX as u64),
            seed in any::<u64>(),
        ) {
            let mut coordinator = create_test_coordinator(num_nodes, batch_size, 10);
            let round = coordinator.current_round_mut().unwrap();
            round.data_index = data_index;
            round.random_seed = seed;
            let selection = CommitteeSelection::from_coordinator(&coordinator, 0).unwrap();
            let assignments = assign_data_for_state(&coordinator, &selection);
            let intervals = assignments.keys().collect::<Vec<_>>();

            prop_assert_eq!(intervals.len(), num_nodes.min(batch_size as usize));
            prop_assert_eq!(intervals.first().unwrap().0.start, data_index);
            prop_assert_eq!(
                intervals.last().unwrap().0.end,
                data_index + batch_size as u64 - 1
            );
            for pair in intervals.windows(2) {
                prop_assert_eq!(pair[0].0.end + 1, pair[1].0.start);
            }
            let covered = intervals
                .iter()
                .map(|batch| batch.0.end - batch.0.start + 1)
                .sum::<u64>();
            prop_assert_eq!(covered, batch_size as u64);

            let assigned = assignments.values().copied().collect::<std::collections::HashSet<_>>();
            prop_assert_eq!(assigned.len(), assignments.len());
            let all_clients_exist = assigned.iter().all(|id| {
                coordinator.epoch_state.clients.iter().any(|client| client.id == *id)
            });
            prop_assert!(all_clients_exist);
        }
    }

    // get_batch_ids_for_round produces the same clean partition as the assignment.
    #[test]
    fn get_batch_ids_for_round_covers_full_range() {
        let coordinator = create_test_coordinator(6, 100, 10);
        let round = *coordinator.current_round().unwrap();
        let batch_ids = get_batch_ids_for_round(&round, &coordinator, 6);

        let total: u64 = batch_ids.iter().map(|b| b.0.end - b.0.start + 1).sum();
        assert_eq!(total, 100);
        // sorted + contiguous
        for w in batch_ids.windows(2) {
            assert_eq!(w[0].0.end + 1, w[1].0.start);
        }
    }

    #[test]
    fn get_batch_ids_for_node_returns_matching_assignments() {
        let coordinator = create_test_coordinator(4, 100, 10);
        let assignments = assign_data_for_state(
            &coordinator,
            &CommitteeSelection::from_coordinator(&coordinator, 0).unwrap(),
        );
        // Pick the first assigned node.
        let first_node = *assignments.values().next().unwrap();
        let ids = get_batch_ids_for_node(&assignments, &first_node);
        assert_eq!(ids.len(), 1);
        assert_eq!(
            ids[0],
            *assignments
                .iter()
                .find(|(_, v)| **v == first_node)
                .unwrap()
                .0
        );
    }

    #[test]
    fn get_batch_ids_for_node_returns_empty_for_unknown_node() {
        let coordinator = create_test_coordinator(3, 30, 10);
        let assignments = assign_data_for_state(
            &coordinator,
            &CommitteeSelection::from_coordinator(&coordinator, 0).unwrap(),
        );
        let unknown = NodeIdentity::from_single_key([0xffu8; 32]);
        let ids = get_batch_ids_for_node(&assignments, &unknown);
        assert!(ids.is_empty());
    }

    // get_data_index_for_step: 0 before the schedule starts and after it ends,
    // and strictly non-decreasing across steps.
    #[test]
    fn get_data_index_for_step_bounds_and_monotonicity() {
        let coordinator = create_test_coordinator(4, 32, 10);
        // step 0 / 1 -> 0
        assert_eq!(get_data_index_for_step(&coordinator, 0), 0);
        assert_eq!(get_data_index_for_step(&coordinator, 1), 0);
        // past total_steps -> 0
        assert_eq!(get_data_index_for_step(&coordinator, 11), 0);

        // monotonic non-decreasing across the schedule
        let mut prev = 0u64;
        for step in 1..=10 {
            let idx = get_data_index_for_step(&coordinator, step);
            assert!(
                idx >= prev,
                "data index decreased at step {step}: {prev} -> {idx}"
            );
            prev = idx;
        }
        // the last step's index is the cumulative sum of all prior batch sizes
        assert!(prev > 0);
    }
}
