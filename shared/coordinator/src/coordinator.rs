use crate::{
    model::{AdapterCheckpoint, Checkpoint, LLMTrainingMethod, Model},
    Commitment, Committee, CommitteeProof, CommitteeSelection, WitnessProof,
};

use aether_core::{sha256, Bloom, FixedString, FixedVec, MerkleRoot, NodeIdentity, SmallBoolean};
use bytemuck::{Pod, Zeroable};
use serde::{Deserialize, Serialize};
use std::{collections::HashSet, hash::Hash};
use tracing::warn;
use ts_rs::TS;

pub const SOLANA_MAX_STRING_LEN: usize = 64;
pub const SOLANA_MAX_URL_STRING_LEN: usize = 192;
pub const SOLANA_MAX_NUM_CLIENTS: usize = 256;
pub const SOLANA_MAX_NUM_WITNESSES: usize = 32;
// run_id must be at most 32 bytes because of PDA constraints
pub const SOLANA_RUN_ID_MAX_LEN: usize = 32;

pub const BLOOM_FALSE_RATE: f64 = 0.01f64;
pub const WITNESS_QUORUM_RAIO: f64 = 2.0f64 / 3.0f64;
pub const WAITING_FOR_MEMBERS_EXTRA_SECONDS: u64 = 10;
// max amount of tokens to send in a witness message
pub const MAX_TOKENS_TO_SEND: usize = 16;

// bloom filter with 1024 bits (16 u64)
pub type WitnessBloom = Bloom<16, 8>;

#[derive(Clone, Copy, Debug, Default, PartialEq, Zeroable, Serialize, Deserialize, TS)]
#[repr(u8)]
pub enum RunState {
    #[default]
    Uninitialized = 0,
    WaitingForMembers = 1,
    Warmup = 2,
    RoundTrain = 3,
    RoundWitness = 4,
    Cooldown = 5,
    Finished = 6,
    Paused = 7,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Zeroable, Serialize, Deserialize, TS)]
#[repr(u8)]
pub enum ClientState {
    #[default]
    Healthy = 0,
    Dropped = 1,
    Withdrawn = 2,
    Ejected = 3,
}

#[derive(Clone, Debug, Zeroable, Default, Copy, Serialize, Deserialize, TS)]
#[repr(C)]
pub struct Client {
    pub id: NodeIdentity,
    pub state: ClientState,
    pub exited_height: u32,
}

impl std::fmt::Display for ClientState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ClientState::Healthy => write!(f, "Healthy"),
            ClientState::Dropped => write!(f, "Dropped"),
            ClientState::Withdrawn => write!(f, "Withdrawn"),
            ClientState::Ejected => write!(f, "Ejected"),
        }
    }
}

impl Hash for Client {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

#[derive(Clone, Default, Debug, Zeroable, Copy, Serialize, Deserialize, PartialEq, TS)]
#[repr(C)]
pub struct Round {
    pub witnesses: FixedVec<Witness, { SOLANA_MAX_NUM_WITNESSES }>,

    pub data_index: u64,
    pub random_seed: u64,
    pub height: u32,
    pub clients_len: u16,
    pub tie_breaker_tasks: u16,
}

#[derive(Clone, Debug, Zeroable, Default, Copy, Serialize, Deserialize, PartialEq, TS)]
#[repr(C)]
pub struct Witness {
    pub proof: WitnessProof,
    pub participant_bloom: WitnessBloom,
    pub broadcast_bloom: WitnessBloom,
    pub broadcast_merkle: MerkleRoot,
}

#[derive(Clone, Copy, Zeroable, Serialize, Deserialize, TS, Default, Debug)]
#[repr(C)]
pub struct WitnessMetadata {
    pub step: u32,
    pub tokens_per_sec: f32,
    pub bandwidth_per_sec: f32,
    pub loss: f32,
    pub evals: FixedVec<WitnessEvalResult, 8>,
    pub prompt_results: FixedVec<i32, { MAX_TOKENS_TO_SEND }>,
    pub prompt_index: u8,
    pub efficency: f32,
}

#[derive(Clone, Copy, Zeroable, Serialize, Deserialize, TS, Default, Debug)]
#[repr(C)]
pub struct WitnessEvalResult {
    pub name: FixedString<32>,
    pub value: f32,
}

impl WitnessEvalResult {
    pub fn new_trunc_name(name: &str, value: f32) -> Self {
        Self {
            name: FixedString::from_str_truncated(name),
            value,
        }
    }
}

#[derive(Clone, Copy, Debug)]
#[repr(u8)]
pub enum CoordinatorError {
    NoActiveRound,
    InvalidWitness,
    InvalidRunState,
    DuplicateWitness,
    InvalidHealthCheck,
    Halted,
    WitnessesFull,
    CannotResume,
    InvalidWithdraw,
    InvalidCommitteeSelection,
    InvalidCommitteeProof,
    StaleWitness,
    FutureWitness,
    InvalidCheckpoint,
}

pub enum TickResult {
    Ticked,
    EpochEnd(bool), // if successfully finished
}

pub type HealthChecks = Vec<(NodeIdentity, CommitteeProof)>;

pub const NUM_STORED_ROUNDS: usize = 4;

#[derive(Clone, Debug, Zeroable, Copy, Serialize, Deserialize, TS)]
#[repr(C)]
pub struct CoordinatorConfig {
    pub warmup_time: u64,
    pub cooldown_time: u64,

    pub max_round_train_time: u64,
    pub round_witness_time: u64,
    pub global_batch_size_warmup_tokens: u64,

    pub epoch_time: u64,
    pub total_steps: u32,

    pub init_min_clients: u16,
    pub min_clients: u16,
    pub witness_nodes: u16,

    pub global_batch_size_start: u16,
    pub global_batch_size_end: u16,

    pub verification_percent: u8,
    pub waiting_for_members_extra_time: u8,
}

#[derive(Clone, Debug, Zeroable, Copy, Serialize, Deserialize, TS)]
#[repr(C)]
pub struct CoordinatorEpochState {
    pub rounds: [Round; NUM_STORED_ROUNDS],
    /// **WARNING**: Using this can be a footgun:
    /// If you need to access the clients list for a particular round,
    /// e.g. when applying a message that could be from the previous round,
    /// This list might not be the list of clients at *that* round.
    /// Consider carefully if `get_client_at_historical_index` or
    /// `get_historical_clients` is what you actually want.
    pub clients: FixedVec<Client, { SOLANA_MAX_NUM_CLIENTS }>,
    pub exited_clients: FixedVec<Client, { SOLANA_MAX_NUM_CLIENTS }>,
    pub rounds_head: u32,
    pub start_step: u32,
    pub last_step: u32,
    pub start_timestamp: u64,
    pub first_round: SmallBoolean,
    pub cold_start_epoch: SmallBoolean,
}

#[derive(Clone, Debug, Zeroable, Copy, Serialize, Deserialize, TS)]
#[repr(C)]
pub struct CoordinatorProgress {
    pub epoch: u16,
    pub step: u32,
    pub epoch_start_data_index: u64,
}

#[derive(Clone, Debug, Zeroable, Copy, Serialize, Deserialize, TS)]
#[repr(C)]
pub struct Coordinator {
    pub run_id: FixedString<{ SOLANA_RUN_ID_MAX_LEN }>,

    pub run_state: RunState,

    pub model: Model,

    pub config: CoordinatorConfig,

    #[serde(default)]
    pub progress: CoordinatorProgress,

    #[serde(default)]
    pub epoch_state: CoordinatorEpochState, // note, gets zeroed at the start of every epoch (not persistent through epochs)

    #[serde(default)]
    pub run_state_start_unix_timestamp: u64,

    #[serde(default)]
    pub pending_pause: SmallBoolean,
}

// SAFETY: `Coordinator` is persisted and replayed as raw bytes by the event
// timeline. Every field is `repr(C)`/fixed-size and is itself `Pod` or a
// transparent fixed-capacity wrapper over `Pod` data; there are no pointers,
// references, heap allocations, or invalid bit-pattern booleans in this layout.
unsafe impl Pod for Coordinator {}

impl TryFrom<usize> for RunState {
    type Error = CoordinatorError;

    fn try_from(value: usize) -> std::result::Result<Self, Self::Error> {
        match value {
            0 => Ok(RunState::Uninitialized),
            1 => Ok(RunState::WaitingForMembers),
            2 => Ok(RunState::Warmup),
            3 => Ok(RunState::RoundTrain),
            4 => Ok(RunState::RoundWitness),
            5 => Ok(RunState::Cooldown),
            6 => Ok(RunState::Finished),
            7 => Ok(RunState::Paused),
            _ => Err(CoordinatorError::InvalidRunState),
        }
    }
}

impl From<RunState> for usize {
    fn from(val: RunState) -> Self {
        match val {
            RunState::Uninitialized => 0,
            RunState::WaitingForMembers => 1,
            RunState::Warmup => 2,
            RunState::RoundTrain => 3,
            RunState::RoundWitness => 4,
            RunState::Cooldown => 5,
            RunState::Finished => 6,
            RunState::Paused => 7,
        }
    }
}
impl PartialEq for Client {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for Client {}

impl std::fmt::Display for CoordinatorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CoordinatorError::NoActiveRound => write!(f, "No active round"),
            CoordinatorError::InvalidWitness => write!(f, "Invalid witness"),
            CoordinatorError::InvalidRunState => write!(f, "Invalid run state"),
            CoordinatorError::DuplicateWitness => write!(f, "Duplicate witness"),
            CoordinatorError::InvalidHealthCheck => write!(f, "Invalid health check"),
            CoordinatorError::Halted => write!(f, "Halted"),
            CoordinatorError::WitnessesFull => write!(f, "Witnesses full"),
            CoordinatorError::CannotResume => write!(f, "Cannot resume"),
            CoordinatorError::InvalidWithdraw => write!(f, "Invalid withdraw"),
            CoordinatorError::InvalidCommitteeSelection => write!(f, "Invalid committee selection"),
            CoordinatorError::InvalidCommitteeProof => write!(f, "Invalid committee proof"),
            CoordinatorError::StaleWitness => write!(f, "Stale witness"),
            CoordinatorError::FutureWitness => write!(f, "Future witness"),
            CoordinatorError::InvalidCheckpoint => write!(f, "Invalid checkpoint"),
        }
    }
}

impl std::error::Error for CoordinatorError {}

impl std::fmt::Display for RunState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RunState::Uninitialized => write!(f, "Uninitialized"),
            RunState::WaitingForMembers => write!(f, "Waiting for members"),
            RunState::Warmup => write!(f, "Warmup"),
            RunState::RoundTrain => write!(f, "Training"),
            RunState::RoundWitness => write!(f, "Witness"),
            RunState::Cooldown => write!(f, "Cooldown"),
            RunState::Finished => write!(f, "Finished"),
            RunState::Paused => write!(f, "Paused"),
        }
    }
}

impl Default for CoordinatorEpochState {
    fn default() -> Self {
        Self {
            rounds: Default::default(),
            rounds_head: Default::default(),
            first_round: true.into(),
            clients: Default::default(),
            exited_clients: Default::default(),
            cold_start_epoch: false.into(),
            start_step: Default::default(),
            last_step: Default::default(),
            start_timestamp: Default::default(),
        }
    }
}

impl Default for CoordinatorProgress {
    fn default() -> Self {
        Self {
            epoch: Default::default(),
            step: 1,
            epoch_start_data_index: Default::default(),
        }
    }
}

impl Client {
    pub fn new(id: NodeIdentity) -> Self {
        Self {
            id,
            state: ClientState::Healthy,
            exited_height: 0,
        }
    }
}

impl Coordinator {
    pub fn tick<'a, 'b>(
        &'a mut self,
        new_clients: Option<impl ExactSizeIterator<Item = &'b NodeIdentity>>,
        unix_timestamp: u64,
        random_seed: u64,
    ) -> std::result::Result<TickResult, CoordinatorError> {
        match self.run_state {
            RunState::Uninitialized | RunState::Finished | RunState::Paused => {
                Err(CoordinatorError::Halted)
            }
            RunState::WaitingForMembers => {
                self.tick_waiting_for_members(new_clients, unix_timestamp)
            }
            RunState::Warmup => self.tick_warmup(unix_timestamp, random_seed),
            RunState::RoundTrain => self.tick_round_train(unix_timestamp),
            RunState::RoundWitness => self.tick_round_witness(unix_timestamp, random_seed),
            RunState::Cooldown => self.tick_cooldown(unix_timestamp),
        }
    }

    pub fn warmup_witness(
        &mut self,
        from: &NodeIdentity,
        witness: Witness,
        unix_timestamp: u64,
        random_seed: u64,
    ) -> std::result::Result<(), CoordinatorError> {
        if self.halted() {
            return Err(CoordinatorError::Halted);
        }

        // If we received a warmup witness but we already transitioned to the next state, we just ignore it.
        if matches!(self.run_state, RunState::RoundTrain) {
            return Ok(());
        }

        if !matches!(self.run_state, RunState::Warmup) {
            return Err(CoordinatorError::InvalidRunState);
        }

        let witness_index = witness.proof.index as usize;
        if self
            .epoch_state
            .clients
            .get(witness_index)
            .map(|client| client.id)
            != Some(*from)
        {
            return Err(CoordinatorError::InvalidWitness);
        }

        // Everyone can send a witness in the warmup phase, so no committee proof is needed.
        let round = self.current_round().unwrap();
        if round
            .witnesses
            .iter()
            .any(|existing| existing.proof.index == witness.proof.index)
        {
            return Err(CoordinatorError::DuplicateWitness);
        }

        let round = self.current_round_mut_unchecked();
        round
            .witnesses
            .push(witness)
            .map_err(|_| CoordinatorError::WitnessesFull)?;

        if round.witnesses.len() == self.epoch_state.clients.len() {
            self.move_clients_without_warmup_witness_to_exited(0);
            if (self.epoch_state.clients.len() as u16) < self.config.min_clients {
                self.start_waiting_for_members(unix_timestamp);
            } else {
                self.start_round_train(unix_timestamp, random_seed, 0);
            }
        }

        Ok(())
    }

    pub fn witness(
        &mut self,
        from: &NodeIdentity,
        submitted_step: u32,
        witness: Witness,
        unix_timestamp: u64,
    ) -> std::result::Result<(), CoordinatorError> {
        if self.halted() {
            return Err(CoordinatorError::Halted);
        }

        let witness_nodes = if self.config.witness_nodes == 0 {
            self.epoch_state.clients.len().min(SOLANA_MAX_NUM_WITNESSES)
        } else {
            self.config.witness_nodes as usize
        };

        if !matches!(
            self.run_state,
            RunState::RoundWitness | RunState::RoundTrain,
        ) {
            return Err(CoordinatorError::InvalidRunState);
        }

        if submitted_step < self.progress.step {
            return Err(CoordinatorError::StaleWitness);
        }
        if submitted_step > self.progress.step {
            return Err(CoordinatorError::FutureWitness);
        }

        if !CommitteeSelection::from_coordinator(self, 0)?.verify_witness_for_client(
            from,
            &witness.proof,
            &self.epoch_state.clients,
        ) || witness.proof.witness.is_false()
        {
            return Err(CoordinatorError::InvalidWitness);
        }

        let round = self.current_round().unwrap();
        for witness in round.witnesses.iter() {
            if self.epoch_state.clients[witness.proof.index as usize].id == *from {
                return Err(CoordinatorError::DuplicateWitness);
            }
        }
        let round = self.current_round_mut_unchecked();
        round
            .witnesses
            .push(witness)
            .map_err(|_| CoordinatorError::WitnessesFull)?;

        if round.witnesses.len() == witness_nodes && !(self.run_state == RunState::RoundWitness) {
            self.change_state(unix_timestamp, RunState::RoundWitness);
        }
        Ok(())
    }

    pub fn health_check(
        &mut self,
        _from: &NodeIdentity,
        checks: HealthChecks,
    ) -> std::result::Result<u32, CoordinatorError> {
        if self.halted() {
            return Err(CoordinatorError::Halted);
        }
        // only health check after pipeline has been filled
        if self
            .current_round()
            .ok_or(CoordinatorError::NoActiveRound)?
            .height
            < 2
        {
            return Err(CoordinatorError::InvalidHealthCheck);
        }
        for (id, proof) in &checks {
            if self.healthy(id, proof)? {
                return Err(CoordinatorError::InvalidHealthCheck);
            }
        }
        let mut dropped = 0;
        for (_id, proof) in &checks {
            let index = proof.index as usize;
            let client = &mut self.epoch_state.clients[index];
            if client.state == ClientState::Healthy {
                client.state = ClientState::Dropped;
                dropped += 1;
            }
        }
        // todo: reward `from` for `dropped` health checks
        Ok(dropped)
    }

    pub fn checkpoint(
        &mut self,
        from: &NodeIdentity,
        index: u64,
        checkpoint_repo: Checkpoint,
    ) -> std::result::Result<bool, CoordinatorError> {
        let index = index as usize;
        if index >= self.epoch_state.clients.len() || self.epoch_state.clients[index].id != *from {
            return Err(CoordinatorError::InvalidCommitteeProof);
        }

        // TODO: In the case of more than one checkpointer, this will overwrite the checkpoint
        // with the last checkpointed one. We could instead have a vector of checkpoints to have
        // more download options.
        let valid_upload = match checkpoint_repo {
            Checkpoint::Hub(repo) => !repo.repo_id.is_empty(),
            Checkpoint::Gcs(repo) => !repo.bucket.is_empty(),
            _ => false,
        };
        if !valid_upload {
            return Err(CoordinatorError::InvalidCheckpoint);
        }

        let Model::LLM(llm) = &mut self.model;
        let old_checkpoint = llm.checkpoint;
        let old_training_method = llm.training_method;
        let mut accepted = true;
        match &mut llm.training_method {
            LLMTrainingMethod::Full => match (&llm.checkpoint, checkpoint_repo) {
                // If current is P2P, wrap the new checkpoint in P2P
                (Checkpoint::P2P(_), Checkpoint::Hub(hub_repo)) => {
                    llm.checkpoint = Checkpoint::P2P(hub_repo);
                }
                (Checkpoint::P2PGcs(_), Checkpoint::Gcs(gcs_repo)) => {
                    llm.checkpoint = Checkpoint::P2PGcs(gcs_repo);
                }
                // If current is Hub, only accept Hub updates
                (Checkpoint::Hub(_), Checkpoint::Hub(hub_repo)) => {
                    llm.checkpoint = Checkpoint::Hub(hub_repo);
                }
                // If current is Gcs, only accept Gcs updates
                (Checkpoint::Gcs(_), Checkpoint::Gcs(gcs_repo)) => {
                    llm.checkpoint = Checkpoint::Gcs(gcs_repo);
                }
                (Checkpoint::P2PGcs(_), Checkpoint::Hub(hub_repo)) => {
                    llm.checkpoint = Checkpoint::P2P(hub_repo);
                }
                (Checkpoint::P2P(_), Checkpoint::Gcs(gcs_repo)) => {
                    llm.checkpoint = Checkpoint::P2PGcs(gcs_repo);
                }
                _ => accepted = false,
            },
            LLMTrainingMethod::Lora(config) => {
                match (&config.adapter_checkpoint, checkpoint_repo) {
                    (AdapterCheckpoint::Fresh, Checkpoint::Hub(repo))
                    | (AdapterCheckpoint::Hub(_), Checkpoint::Hub(repo)) => {
                        config.adapter_checkpoint = AdapterCheckpoint::Hub(repo);
                    }
                    (AdapterCheckpoint::Fresh, Checkpoint::Gcs(repo))
                    | (AdapterCheckpoint::Gcs(_), Checkpoint::Gcs(repo)) => {
                        config.adapter_checkpoint = AdapterCheckpoint::Gcs(repo);
                    }
                    (AdapterCheckpoint::P2P(_), Checkpoint::Hub(repo))
                    | (AdapterCheckpoint::P2PGcs(_), Checkpoint::Hub(repo)) => {
                        config.adapter_checkpoint = AdapterCheckpoint::P2P(repo);
                    }
                    (AdapterCheckpoint::P2P(_), Checkpoint::Gcs(repo))
                    | (AdapterCheckpoint::P2PGcs(_), Checkpoint::Gcs(repo)) => {
                        config.adapter_checkpoint = AdapterCheckpoint::P2PGcs(repo);
                    }
                    _ => accepted = false,
                }
            }
        }

        if !accepted {
            return Err(CoordinatorError::InvalidCheckpoint);
        }

        Ok(old_checkpoint != llm.checkpoint || old_training_method != llm.training_method)
    }

    pub fn withdraw(&mut self, index: u64) -> std::result::Result<(), CoordinatorError> {
        let index = index as usize;
        if index < self.epoch_state.clients.len() {
            let client = &mut self.epoch_state.clients[index];
            if client.state == ClientState::Healthy {
                client.state = ClientState::Withdrawn;
                return Ok(());
            }
        }
        Err(CoordinatorError::InvalidWithdraw)
    }

    pub fn withdraw_all(&mut self) {
        if !self.epoch_state.clients.is_empty() {
            let clients_max_index = self.epoch_state.clients.len() - 1;
            for client_index in 0..=clients_max_index {
                let _ = self.withdraw(client_index as u64); // we need to withdraw everyone, ignore error of already withdrawn
            }
        }
    }

    pub fn pause(&mut self, unix_timestamp: u64) -> std::result::Result<(), CoordinatorError> {
        if !self.halted() {
            if self.active() {
                self.pending_pause = true.into();
            } else {
                self.withdraw_all();
                self.change_state(unix_timestamp, RunState::Paused);
                self.epoch_state.cold_start_epoch = true.into();
            }
            Ok(())
        } else {
            Err(CoordinatorError::Halted)
        }
    }

    pub fn resume(&mut self, unix_timestamp: u64) -> Result<(), CoordinatorError> {
        if self.run_state != RunState::Paused {
            return Err(CoordinatorError::CannotResume);
        }
        self.start_waiting_for_members(unix_timestamp);
        Ok(())
    }

    pub fn healthy(
        &self,
        id: &NodeIdentity,
        proof: &CommitteeProof,
    ) -> Result<bool, CoordinatorError> {
        let round = self
            .previous_round()
            .ok_or(CoordinatorError::NoActiveRound)?;
        let index = proof.index;
        if index < round.clients_len as u64 {
            let client = self
                .get_client_at_historical_index(index as usize, round.clients_len)
                .ok_or(CoordinatorError::InvalidCommitteeProof)?;
            let selection = CommitteeSelection::from_coordinator(self, -1)?;
            if client.id != *id
                || !selection.verify_committee_for_client(
                    &client.id,
                    proof,
                    &self.epoch_state.clients,
                )
            {
                return Err(CoordinatorError::InvalidCommitteeProof);
            }
            match proof.committee {
                // Non-trainer committees (tie-breakers, verifiers) do not
                // participate in training, so their participation is not
                // recorded in the witnesses' `participant_bloom` filters.
                // Training-based health checking therefore cannot determine
                // their health; treat them as healthy rather than risk
                // dropping every non-trainer each round.
                Committee::TieBreaker | Committee::Verifier => Ok(true),
                Committee::Trainer => self.trainer_healthy(&client.id),
            }
        } else {
            Err(CoordinatorError::InvalidCommitteeProof)
        }
    }

    pub fn witness_quorum(&self, num_witnesses: u16) -> u16 {
        let witness_nodes = match self.config.witness_nodes {
            0 => num_witnesses,
            witness_nodes => witness_nodes,
        };
        match witness_nodes {
            0 => unreachable!(),
            1 => 1,
            2 => 2,
            3 => 2,
            witness_nodes => ((witness_nodes as f64 * WITNESS_QUORUM_RAIO) as u16).max(1),
        }
    }

    pub fn trainer_healthy(&self, id: &NodeIdentity) -> Result<bool, CoordinatorError> {
        let prev_round_witnesses = &self
            .previous_round()
            .ok_or(CoordinatorError::NoActiveRound)?
            .witnesses;

        let score = Self::trainer_healthy_score_by_witnesses(id, prev_round_witnesses);
        Ok(score >= self.witness_quorum(prev_round_witnesses.len() as u16))
    }

    /// Computes the health score of a client based on witness confirmations.
    /// The score increases for each witness whose participant bloom filter contains the client's hashed ID.
    pub fn trainer_healthy_score_by_witnesses(id: &NodeIdentity, witnesses: &[Witness]) -> u16 {
        let hash = sha256(id.signer());

        let mut score = 0u16;
        for witness in witnesses {
            if witness.participant_bloom.contains(&hash) {
                score += 1;
            }
        }

        score
    }

    pub fn select_consensus_commitment_by_witnesses(
        commitments: &[Commitment],
        witnesses: &[Witness],
        witness_quorum: u16,
    ) -> Option<usize> {
        let mut scores = vec![0; commitments.len()];
        for witness in witnesses {
            for (index, commitment) in commitments.iter().enumerate() {
                if witness.broadcast_bloom.contains(&commitment.data_hash) {
                    scores[index] += 1;
                    break;
                }
            }
        }
        scores
            .into_iter()
            .enumerate()
            .filter(|(_, score)| *score >= witness_quorum)
            .max_by(|(left_index, left_score), (right_index, right_score)| {
                left_score.cmp(right_score).then_with(|| {
                    commitments[*left_index]
                        .data_hash
                        .cmp(&commitments[*right_index].data_hash)
                })
            })
            .map(|(index, _)| index)
    }

    pub fn current_round(&self) -> Option<&Round> {
        self.epoch_state
            .rounds
            .get(self.epoch_state.rounds_head as usize)
    }

    pub fn current_round_mut(&mut self) -> Option<&mut Round> {
        self.epoch_state
            .rounds
            .get_mut(self.epoch_state.rounds_head as usize)
    }

    pub fn current_round_unchecked(&self) -> &Round {
        &self.epoch_state.rounds[self.epoch_state.rounds_head as usize]
    }

    pub fn current_round_mut_unchecked(&mut self) -> &mut Round {
        &mut self.epoch_state.rounds[self.epoch_state.rounds_head as usize]
    }

    pub fn previous_round(&self) -> Option<&Round> {
        match self.current_round() {
            Some(round) => match self.epoch_state.rounds_head == 0 && round.height == 0 {
                true => None,
                false => match self.epoch_state.rounds_head == 0 {
                    true => Some(&self.epoch_state.rounds[NUM_STORED_ROUNDS - 1]),
                    false => {
                        Some(&self.epoch_state.rounds[self.epoch_state.rounds_head as usize - 1])
                    }
                },
            },
            None => None,
        }
    }

    pub fn previous_previous_round(&self) -> Option<&Round> {
        match self.current_round() {
            Some(round) => match self.epoch_state.rounds_head == 0 && round.height <= 1 {
                true => None,
                false => match self.epoch_state.rounds_head {
                    0 => Some(&self.epoch_state.rounds[NUM_STORED_ROUNDS - 2]),
                    1 => Some(&self.epoch_state.rounds[NUM_STORED_ROUNDS - 1]),
                    n => Some(&self.epoch_state.rounds[n as usize - 2]),
                },
            },
            None => None,
        }
    }

    pub fn active(&self) -> bool {
        !matches!(
            self.run_state,
            RunState::WaitingForMembers
                | RunState::Warmup
                | RunState::Uninitialized
                | RunState::Finished
                | RunState::Paused
        )
    }

    pub fn halted(&self) -> bool {
        matches!(
            self.run_state,
            RunState::Uninitialized | RunState::Finished | RunState::Paused
        )
    }

    pub fn get_client_at_historical_index(
        &self,
        n: usize,
        prev_clients_len: u16,
    ) -> Option<&Client> {
        if n < self.epoch_state.clients.len() {
            Some(&self.epoch_state.clients[n])
        } else if n < prev_clients_len as usize {
            let offset: usize = prev_clients_len as usize - n - 1;
            self.epoch_state.exited_clients.iter().rev().nth(offset)
        } else {
            None
        }
    }

    pub fn get_historical_clients(&self, clients_len: u16) -> Vec<&Client> {
        (0..clients_len)
            .filter_map(|i| self.get_client_at_historical_index(i as usize, clients_len))
            .collect()
    }

    pub fn get_sequence_length(&self) -> u32 {
        match &self.model {
            Model::LLM(llm) => llm.max_seq_len,
        }
    }

    pub fn get_target_global_batch_size(&self, round: Option<&Round>) -> u16 {
        let tokens_processed = self.total_tokens_processed(round);
        self.config.get_batch_size(tokens_processed)
    }

    pub fn total_tokens_processed(&self, round: Option<&Round>) -> u64 {
        // if no round active yet (e.g., warmup), use epoch_start_data_index
        let current_data_start_index = round
            .map(|r| r.data_index)
            .unwrap_or(self.progress.epoch_start_data_index);

        current_data_start_index * self.get_sequence_length() as u64
    }

    pub fn get_cold_start_warmup_bounds(&self) -> Option<(u32, u32)> {
        let Model::LLM(llm) = &self.model;
        let cold_start_warmup_steps = llm.cold_start_warmup_steps;
        if self.epoch_state.cold_start_epoch.is_false() || cold_start_warmup_steps == 0 {
            return None;
        }
        Some((
            self.epoch_state.start_step,
            self.epoch_state.start_step + cold_start_warmup_steps,
        ))
    }

    /// Check that cold_start_warmup_steps can be completed within a single epoch.
    pub fn check_cold_start_warmup_steps(&self) -> bool {
        let Model::LLM(llm) = &self.model;
        if llm.cold_start_warmup_steps == 0 {
            return true;
        }
        let training_time = self.config.epoch_time - self.config.warmup_time;
        let estimated_training_rounds = training_time / self.config.max_round_train_time;
        if llm.cold_start_warmup_steps as u64 > estimated_training_rounds {
            warn!(
                "cold_start_warmup_steps ({}) exceeds estimated training rounds per epoch ((epoch_time={} - warmup_time={}) / max_round_train_time={} = {})",
                llm.cold_start_warmup_steps,
                self.config.epoch_time,
                self.config.warmup_time,
                self.config.max_round_train_time,
                estimated_training_rounds
            );
            return false;
        }
        true
    }

    fn get_global_batch_size_for_tokens(&self, tokens_processed: u64) -> u16 {
        self.config.get_batch_size(tokens_processed)
    }

    fn tick_waiting_for_members<'a, 'b>(
        &'a mut self,
        pending_clients: Option<impl ExactSizeIterator<Item = &'b NodeIdentity>>,
        unix_timestamp: u64,
    ) -> std::result::Result<TickResult, CoordinatorError> {
        let Some(pending_clients) = pending_clients else {
            return Ok(TickResult::Ticked);
        };

        if pending_clients.len() as u16 >= self.config.init_min_clients
            && self.check_timeout(
                unix_timestamp,
                self.config.waiting_for_members_extra_time as u64,
            )
        // This extra time allows for more clients to join even if the minimum number of clients is reached
        {
            // Make sure that all unhealthy clients are kicked at this point
            let height = self.current_round_unchecked().height;
            self.move_clients_to_exited(height);

            // Read the pending clients
            let pending_clients_ordered: Vec<&NodeIdentity> = pending_clients.collect();

            let cold_start_epoch = self.epoch_state.cold_start_epoch;
            bytemuck::write_zeroes(&mut self.epoch_state);
            self.epoch_state.first_round = true.into();
            self.epoch_state.cold_start_epoch = cold_start_epoch;
            self.epoch_state.start_step = self.progress.step;
            self.epoch_state.start_timestamp = unix_timestamp;
            self.epoch_state
                .clients
                .extend(
                    pending_clients_ordered
                        .into_iter()
                        .take(SOLANA_MAX_NUM_CLIENTS)
                        .map(|x| Client::new(*x)),
                )
                .unwrap();

            self.start_warmup(unix_timestamp);
        }

        Ok(TickResult::Ticked)
    }

    fn tick_warmup(
        &mut self,
        unix_timestamp: u64,
        random_seed: u64,
    ) -> std::result::Result<TickResult, CoordinatorError> {
        if self.check_timeout(unix_timestamp, self.config.warmup_time) {
            self.move_clients_without_warmup_witness_to_exited(0);
            if (self.epoch_state.clients.len() as u16) >= self.config.min_clients {
                self.start_round_train(unix_timestamp, random_seed, 0);
            }
        } else {
            self.move_clients_to_exited(0);
        }
        if (self.epoch_state.clients.len() as u16) < self.config.min_clients {
            self.start_waiting_for_members(unix_timestamp);
            Ok(TickResult::EpochEnd(false))
        } else {
            Ok(TickResult::Ticked)
        }
    }

    fn tick_round_train(
        &mut self,
        unix_timestamp: u64,
    ) -> std::result::Result<TickResult, CoordinatorError> {
        if self.check_timeout(unix_timestamp, self.config.max_round_train_time) {
            self.change_state(unix_timestamp, RunState::RoundWitness);
        }
        Ok(TickResult::Ticked)
    }

    fn tick_round_witness(
        &mut self,
        unix_timestamp: u64,
        random_seed: u64,
    ) -> std::result::Result<TickResult, CoordinatorError> {
        let expected_witnesses = if self.config.witness_nodes == 0 {
            self.epoch_state.clients.len() as u16
        } else {
            self.config.witness_nodes
        };
        let has_quorum = self.current_round().is_some_and(|round| {
            round.witnesses.len() as u16 >= self.witness_quorum(expected_witnesses)
        });
        let witness_timed_out = self.check_timeout(unix_timestamp, self.config.round_witness_time);
        if has_quorum || witness_timed_out {
            self.epoch_state.first_round = false.into();
            self.progress.step += 1;
            let current_round = self.current_round_unchecked();
            let height = current_round.height;
            let num_witnesses = current_round.witnesses.len() as u16;

            if witness_timed_out {
                if let Err(error) = self.eject_missing_witnesses() {
                    warn!(?error, "could not identify missing elected witnesses");
                }
            }
            self.move_clients_to_exited(height);

            // A round without witnesses cannot be verified. End the epoch, but
            // preserve non-witness clients so the next epoch can admit them.
            if num_witnesses == 0 {
                self.start_cooldown(unix_timestamp);
                return Ok(TickResult::Ticked);
            }

            // Once the timeout for the whole epoch is reached, we set the last step as the current
            // step plus two.
            if self.check_epoch_timeout(unix_timestamp) && !self.epoch_state.last_step_set() {
                let last_step: u32 = self.progress.step + 2;
                // Just a sanity check to be sure the epoch doesn't end too early since we need
                // at least 4 rounds per epoch for overlapped pipeling
                if last_step >= 4 {
                    self.epoch_state.last_step = last_step;
                }
            }

            // DisTrO updates are applied two rounds after they are trained. Stop
            // training at the configured limit, then drain those two updates.
            if self.progress.step >= self.config.total_steps && !self.epoch_state.last_step_set() {
                self.epoch_state.last_step = self.progress.step + 2;
            }

            // We reached the last step of the epoch, we transition to Cooldown
            if self.epoch_state.last_step_set() && self.progress.step == self.epoch_state.last_step
            {
                self.start_cooldown(unix_timestamp);
                return Ok(TickResult::Ticked);
            }

            // If we don't reach the min number of clients or registered witnesses for the current round,
            // we change to Cooldown
            if self.epoch_state.clients.len() < self.config.min_clients as usize
                || num_witnesses < self.witness_quorum(num_witnesses)
                || self.pending_pause.is_true()
            {
                self.start_cooldown(unix_timestamp);
                return Ok(TickResult::Ticked);
            }

            self.start_round_train(unix_timestamp, random_seed, 0);
        }
        Ok(TickResult::Ticked)
    }

    fn tick_cooldown(
        &mut self,
        unix_timestamp: u64,
    ) -> std::result::Result<TickResult, CoordinatorError> {
        if self.check_timeout(unix_timestamp, self.config.cooldown_time) {
            let last_round_batch_size = self.get_target_global_batch_size(self.current_round());
            self.progress.epoch_start_data_index =
                self.current_round_unchecked().data_index + last_round_batch_size as u64;
            self.progress.epoch += 1;

            let current_round = self.current_round_unchecked();
            let height = current_round.height;
            self.move_clients_to_exited(height);

            // Checkpoints stay Hub/GCS permanently. The seed node pushes fresh
            // weights every epoch, so late joiners can always pre-download from
            // the hosting service during WaitingForMembers — no P2P download
            // required, no warmup deadline pressure.

            if self.pending_pause.is_true() {
                self.withdraw_all();
                self.change_state(unix_timestamp, RunState::Paused);
                self.pending_pause = false.into();
                self.epoch_state.cold_start_epoch = true.into();
            } else {
                self.start_waiting_for_members(unix_timestamp);
                self.epoch_state.cold_start_epoch = false.into();
            }

            Ok(TickResult::EpochEnd(true))
        } else {
            Ok(TickResult::Ticked)
        }
    }

    fn check_timeout(&self, unix_timestamp: u64, duration: u64) -> bool {
        self.run_state_start_unix_timestamp != unix_timestamp
            && unix_timestamp >= duration + self.run_state_start_unix_timestamp
    }

    fn check_epoch_timeout(&self, unix_timestamp: u64) -> bool {
        self.epoch_state.start_timestamp != unix_timestamp
            && unix_timestamp >= self.epoch_state.start_timestamp + self.config.epoch_time
    }

    fn start_cooldown(&mut self, unix_timestamp: u64) {
        self.current_round_mut_unchecked().witnesses.clear(); // clear witnesses for re-use in warmup
        self.change_state(unix_timestamp, RunState::Cooldown);
    }

    fn start_round_train(&mut self, unix_timestamp: u64, random_seed: u64, tie_breaker_tasks: u16) {
        let (next_rounds_head, next_height, next_data_index) =
            if self.epoch_state.first_round.into() {
                // very first round, don't increment -- just start here
                (0usize, 0u32, self.progress.epoch_start_data_index)
            } else {
                let prev_round = &self.epoch_state.rounds[self.epoch_state.rounds_head as usize];
                let prev_round_start_tokens =
                    prev_round.data_index * self.get_sequence_length() as u64;
                let prev_round_batch_size =
                    self.get_global_batch_size_for_tokens(prev_round_start_tokens);
                (
                    (self.epoch_state.rounds_head + 1) as usize % self.epoch_state.rounds.len(),
                    prev_round.height + 1,
                    prev_round.data_index + prev_round_batch_size as u64,
                )
            };
        let round = &mut self.epoch_state.rounds[next_rounds_head];
        self.epoch_state.rounds_head = next_rounds_head as u32;
        round.clients_len = self.epoch_state.clients.len() as u16;
        round.height = next_height;
        round.data_index = next_data_index;
        round.tie_breaker_tasks = tie_breaker_tasks;
        round.random_seed = random_seed;
        round.witnesses.clear();
        self.change_state(unix_timestamp, RunState::RoundTrain);
    }

    fn start_warmup(&mut self, unix_timestamp: u64) {
        self.change_state(unix_timestamp, RunState::Warmup);
    }

    fn start_waiting_for_members(&mut self, unix_timestamp: u64) {
        self.change_state(
            unix_timestamp,
            if self.progress.step < self.config.total_steps {
                RunState::WaitingForMembers
            } else {
                RunState::Finished
            },
        );
    }

    fn change_state(&mut self, unix_timestamp: u64, new_state: RunState) {
        assert!(self.run_state != new_state);
        self.run_state_start_unix_timestamp = unix_timestamp;
        self.run_state = new_state;
    }

    fn move_clients_to_exited(&mut self, height: u32) {
        // WARNING: O(n) on number of clients, need to refactor
        self.epoch_state.clients.retain(|x| {
            if x.state != ClientState::Healthy {
                self.epoch_state.exited_clients.push(*x).unwrap();
                self.epoch_state
                    .exited_clients
                    .last_mut()
                    .unwrap()
                    .exited_height = height;
                false
            } else {
                true
            }
        });
    }

    fn eject_missing_witnesses(&mut self) -> Result<usize, CoordinatorError> {
        let selection = CommitteeSelection::from_coordinator(self, 0)?;
        let received: HashSet<u64> = self
            .current_round_unchecked()
            .witnesses
            .iter()
            .map(|witness| witness.proof.index)
            .collect();
        let clients_len = self.current_round_unchecked().clients_len as usize;
        let mut ejected = 0;

        for index in 0..clients_len.min(self.epoch_state.clients.len()) {
            let client = &mut self.epoch_state.clients[index];
            if client.state == ClientState::Healthy
                && selection.get_witness(index as u64).witness.is_true()
                && !received.contains(&(index as u64))
            {
                client.state = ClientState::Ejected;
                ejected += 1;
            }
        }

        Ok(ejected)
    }

    fn move_clients_without_warmup_witness_to_exited(&mut self, height: u32) {
        let warmup_witnesses: HashSet<usize> = self
            .current_round_unchecked()
            .witnesses
            .iter()
            .map(|witness| witness.proof.index as usize)
            .collect();

        // WARNING: O(n) on number of clients, need to refactor
        let mut client_index = 0usize;
        self.epoch_state.clients.retain(|client| {
            let keep = warmup_witnesses.contains(&client_index);
            client_index += 1;

            if !keep {
                let mut exited = *client;
                exited.state = ClientState::Dropped;
                self.epoch_state
                    .exited_clients
                    .push(Client {
                        exited_height: height,
                        ..exited
                    })
                    .unwrap();
            }

            keep
        });
    }

    pub fn is_warmup_just_starting(&self) -> bool {
        self.epoch_state.first_round.is_true() && self.run_state == RunState::Warmup
    }

    pub fn is_training_just_starting(&self) -> bool {
        self.epoch_state.first_round.is_true() && self.run_state == RunState::RoundTrain
    }
}

impl CoordinatorEpochState {
    // When an epoch reaches its timeout, the last step is set as the
    // current step + 2. When last_step is set to 0, we assume it has not
    // been set.
    pub fn last_step_set(&self) -> bool {
        self.last_step != 0
    }
}

#[derive(Debug)]
pub enum ConfigError {
    EpochTime,
    WarmupTime,
    MaxRoundTrainTime,
    RoundWitnessTime,
    MinClients,
    InitMinClients,
    GlobalBatchSize,
    TotalSteps,
    WitnessNodes,
    CooldownTime,
    WaitingForMembersExtraTime,
}

impl CoordinatorConfig {
    pub fn check(&self) -> bool {
        self.check_error().is_ok()
    }

    #[inline(always)]
    pub fn check_error(&self) -> Result<(), ConfigError> {
        if self.epoch_time == 0 {
            return Err(ConfigError::EpochTime);
        }
        if self.warmup_time >= self.epoch_time {
            return Err(ConfigError::WarmupTime);
        }
        if self.max_round_train_time == 0 || self.max_round_train_time >= self.epoch_time {
            return Err(ConfigError::MaxRoundTrainTime);
        }
        if self.round_witness_time == 0 {
            return Err(ConfigError::RoundWitnessTime);
        }
        if self.min_clients == 0 {
            return Err(ConfigError::MinClients);
        }
        if self.init_min_clients < self.min_clients
            || self.init_min_clients as usize > SOLANA_MAX_NUM_CLIENTS
        {
            return Err(ConfigError::InitMinClients);
        }
        if self.global_batch_size_start == 0
            || self.global_batch_size_end == 0
            || self.global_batch_size_end < self.global_batch_size_start
        {
            return Err(ConfigError::GlobalBatchSize);
        }
        if self.total_steps == 0 {
            return Err(ConfigError::TotalSteps);
        }
        if self.witness_nodes > self.min_clients
            || self.witness_nodes as usize > SOLANA_MAX_NUM_WITNESSES
        {
            return Err(ConfigError::WitnessNodes);
        }
        if self.cooldown_time == 0 {
            return Err(ConfigError::CooldownTime);
        }
        if self.waiting_for_members_extra_time == 0 {
            return Err(ConfigError::WaitingForMembersExtraTime);
        }
        Ok(())
    }

    pub fn get_batch_size(&self, total_tokens_processed: u64) -> u16 {
        if total_tokens_processed >= self.global_batch_size_warmup_tokens {
            self.global_batch_size_end
        } else {
            let progress =
                total_tokens_processed as f64 / self.global_batch_size_warmup_tokens as f64;
            (self.global_batch_size_start as f64
                + (self.global_batch_size_end as f64 - self.global_batch_size_start as f64)
                    * progress)
                .round() as u16
        }
    }
}

impl CoordinatorProgress {
    pub fn check(&self) -> bool {
        self.step > 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{GcsRepo, HubRepo, LoraConfig, LLM};
    use aether_core::{sha256, FixedVec, NodeIdentity, OptimizerDefinition};
    use bytemuck::Zeroable;
    use proptest::prelude::*;

    fn identity(n: u8) -> NodeIdentity {
        let mut key = [0u8; 32];
        key[0] = n;
        NodeIdentity::from_single_key(key)
    }

    fn hub_repo(repo_id: &str) -> HubRepo {
        HubRepo {
            repo_id: FixedString::try_from(repo_id).unwrap(),
            revision: None,
        }
    }

    fn checkpoint_coordinator(training_method: LLMTrainingMethod) -> Coordinator {
        let mut coordinator = Coordinator::zeroed();
        coordinator.epoch_state.clients = FixedVec::from_iter([Client::new(identity(1))]);
        let mut llm = LLM::dummy();
        llm.checkpoint = Checkpoint::Hub(hub_repo("base/model"));
        llm.training_method = training_method;
        coordinator.model = Model::LLM(llm);
        coordinator
    }

    fn witness_coordinator(run_state: RunState, client_count: u8) -> Coordinator {
        let mut coordinator = Coordinator::zeroed();
        coordinator.run_state = run_state;
        coordinator.progress.step = 10;
        coordinator.config.witness_nodes = client_count as u16;
        coordinator.config.min_clients = client_count as u16;
        coordinator.epoch_state.clients =
            FixedVec::from_iter((0..client_count).map(|index| Client::new(identity(index + 1))));
        coordinator.current_round_mut_unchecked().clients_len = client_count as u16;
        coordinator
    }

    fn witness_for(coordinator: &Coordinator, index: u64) -> Witness {
        Witness {
            proof: CommitteeSelection::from_coordinator(coordinator, 0)
                .unwrap()
                .get_witness(index),
            ..Witness::default()
        }
    }

    #[test]
    fn warmup_miss_marks_client_dropped_before_exit() {
        let mut coordinator = Coordinator::zeroed();
        coordinator.epoch_state.clients =
            FixedVec::from_iter([Client::new(identity(1)), Client::new(identity(2))]);
        coordinator
            .current_round_mut_unchecked()
            .witnesses
            .push(Witness {
                proof: WitnessProof {
                    position: 0,
                    index: 0,
                    witness: Default::default(),
                },
                participant_bloom: Default::default(),
                broadcast_bloom: Default::default(),
                broadcast_merkle: Default::default(),
            })
            .unwrap();

        coordinator.move_clients_without_warmup_witness_to_exited(7);

        assert_eq!(coordinator.epoch_state.clients.len(), 1);
        assert_eq!(coordinator.epoch_state.clients[0].id, identity(1));
        assert_eq!(coordinator.epoch_state.exited_clients.len(), 1);
        assert_eq!(coordinator.epoch_state.exited_clients[0].id, identity(2));
        assert_eq!(
            coordinator.epoch_state.exited_clients[0].state,
            ClientState::Dropped
        );
        assert_eq!(coordinator.epoch_state.exited_clients[0].exited_height, 7);
    }

    fn witness_timeout_coordinator(client_count: u8, witness_nodes: u16) -> Coordinator {
        let mut coordinator = Coordinator::zeroed();
        coordinator.run_state = RunState::RoundWitness;
        coordinator.run_state_start_unix_timestamp = 10;
        coordinator.config.round_witness_time = 14;
        coordinator.config.witness_nodes = witness_nodes;
        coordinator.config.min_clients = 1;
        coordinator.config.total_steps = 100;
        coordinator.epoch_state.clients =
            FixedVec::from_iter((0..client_count).map(|n| Client::new(identity(n + 1))));

        let round = coordinator.current_round_mut_unchecked();
        round.clients_len = client_count as u16;
        round.height = 7;
        round.random_seed = 42;
        coordinator
    }

    #[test]
    fn total_steps_drains_two_pipeline_rounds_before_cooldown() {
        let mut coordinator = witness_timeout_coordinator(1, 1);
        coordinator.config.total_steps = 1;
        coordinator.config.epoch_time = u64::MAX / 2;

        for expected_step in 1..=3 {
            let selection = CommitteeSelection::from_coordinator(&coordinator, 0).unwrap();
            coordinator
                .current_round_mut_unchecked()
                .witnesses
                .push(Witness {
                    proof: selection.get_witness(0),
                    ..Default::default()
                })
                .unwrap();
            coordinator.run_state = RunState::RoundWitness;
            coordinator.tick_round_witness(11, 99).unwrap();

            assert_eq!(coordinator.progress.step, expected_step);
            if expected_step < 3 {
                assert_eq!(coordinator.run_state, RunState::RoundTrain);
                assert_eq!(coordinator.epoch_state.last_step, 3);
            }
        }

        assert_eq!(coordinator.run_state, RunState::Cooldown);
    }

    #[test]
    fn witness_timeout_ejects_only_missing_sole_witness() {
        let mut coordinator = witness_timeout_coordinator(3, 1);
        let selection = CommitteeSelection::from_coordinator(&coordinator, 0).unwrap();
        let elected_index = (0..3)
            .find(|index| selection.get_witness(*index).witness.is_true())
            .unwrap() as usize;
        let elected_id = coordinator.epoch_state.clients[elected_index].id;

        coordinator.tick_round_witness(24, 99).unwrap();

        assert_eq!(coordinator.run_state, RunState::Cooldown);
        assert_eq!(coordinator.progress.step, 1);
        assert_eq!(coordinator.epoch_state.clients.len(), 2);
        assert!(coordinator
            .epoch_state
            .clients
            .iter()
            .all(|client| client.state == ClientState::Healthy));
        assert_eq!(coordinator.epoch_state.exited_clients.len(), 1);
        assert_eq!(coordinator.epoch_state.exited_clients[0].id, elected_id);
        assert_eq!(
            coordinator.epoch_state.exited_clients[0].state,
            ClientState::Ejected
        );
    }

    #[test]
    fn witness_timeout_preserves_clients_that_submitted_or_were_not_elected() {
        let mut coordinator = witness_timeout_coordinator(4, 2);
        let selection = CommitteeSelection::from_coordinator(&coordinator, 0).unwrap();
        let elected: Vec<_> = (0..4)
            .filter(|index| selection.get_witness(*index).witness.is_true())
            .collect();
        assert_eq!(elected.len(), 2);

        let submitted_index = elected[0];
        coordinator
            .current_round_mut_unchecked()
            .witnesses
            .push(Witness {
                proof: selection.get_witness(submitted_index),
                ..Default::default()
            })
            .unwrap();
        let missing_id = coordinator.epoch_state.clients[elected[1] as usize].id;

        coordinator.tick_round_witness(24, 99).unwrap();

        assert_eq!(coordinator.run_state, RunState::Cooldown);
        assert_eq!(coordinator.epoch_state.exited_clients.len(), 1);
        assert_eq!(coordinator.epoch_state.exited_clients[0].id, missing_id);
        assert_eq!(
            coordinator.epoch_state.exited_clients[0].state,
            ClientState::Ejected
        );
        assert_eq!(coordinator.epoch_state.clients.len(), 3);
        assert!(coordinator
            .epoch_state
            .clients
            .iter()
            .all(|client| client.state == ClientState::Healthy));
    }

    #[test]
    fn witness_round_advances_before_timeout_at_quorum() {
        let mut coordinator = witness_timeout_coordinator(4, 3);
        let selection = CommitteeSelection::from_coordinator(&coordinator, 0).unwrap();
        for index in (0..4)
            .filter(|index| selection.get_witness(*index).witness.is_true())
            .take(2)
        {
            coordinator
                .current_round_mut_unchecked()
                .witnesses
                .push(Witness {
                    proof: selection.get_witness(index),
                    ..Default::default()
                })
                .unwrap();
        }

        coordinator.tick_round_witness(11, 99).unwrap();

        assert_eq!(coordinator.run_state, RunState::RoundTrain);
        assert_eq!(coordinator.progress.step, 1);
    }

    #[test]
    fn witness_quorum_is_stable_during_simultaneous_disconnects() {
        let mut coordinator = witness_timeout_coordinator(5, 3);
        coordinator.config.min_clients = 4;
        let selection = CommitteeSelection::from_coordinator(&coordinator, 0).unwrap();
        let submitted = (0..5)
            .filter(|index| selection.get_witness(*index).witness.is_true())
            .take(2)
            .collect::<Vec<_>>();
        assert_eq!(submitted.len(), 2);
        for &index in &submitted {
            coordinator
                .current_round_mut_unchecked()
                .witnesses
                .push(Witness {
                    proof: selection.get_witness(index),
                    ..Default::default()
                })
                .unwrap();
        }

        let disconnected = (0..5)
            .filter(|index| !submitted.contains(index))
            .take(2)
            .collect::<Vec<_>>();
        for &index in &disconnected {
            coordinator.withdraw(index).unwrap();
        }

        coordinator.tick_round_witness(11, 99).unwrap();

        assert_eq!(coordinator.progress.step, 1);
        assert_eq!(coordinator.run_state, RunState::Cooldown);
        assert_eq!(coordinator.epoch_state.clients.len(), 3);
        assert_eq!(coordinator.epoch_state.exited_clients.len(), 2);
        assert!(coordinator
            .epoch_state
            .exited_clients
            .iter()
            .all(|client| client.state == ClientState::Withdrawn));
    }

    // ── RunState <-> usize roundtrip ───────────────────────────────────────────
    #[test]
    fn runstate_usize_roundtrip() {
        for v in 0..=7 {
            let state = RunState::try_from(v).unwrap();
            let back: usize = state.into();
            assert_eq!(back, v);
        }
        assert!(RunState::try_from(8).is_err());
        assert!(RunState::try_from(255).is_err());
    }

    #[test]
    fn runstate_default_is_uninitialized() {
        assert_eq!(RunState::default(), RunState::Uninitialized);
    }

    // ── witness_quorum table ───────────────────────────────────────────────────
    // Quorum is the 2/3 ratio with small-N special cases. A bug here either
    // stalls consensus (quorum too high) or accepts forged results (too low).
    #[test]
    fn witness_quorum_special_cases_and_ratio() {
        let mut c = Coordinator::zeroed();
        // config.witness_nodes == 0 -> use the passed num_witnesses.
        c.config.witness_nodes = 0;
        assert_eq!(c.witness_quorum(1), 1);
        assert_eq!(c.witness_quorum(2), 2);
        assert_eq!(c.witness_quorum(3), 2);
        // floor(n * 2/3) for n >= 4
        assert_eq!(c.witness_quorum(4), 2); // 2.66 -> 2
        assert_eq!(c.witness_quorum(6), 4); // 4.0
        assert_eq!(c.witness_quorum(9), 6); // 6.0
        assert_eq!(c.witness_quorum(12), 8); // 8.0
    }

    #[test]
    fn witness_quorum_uses_config_when_set() {
        let mut c = Coordinator::zeroed();
        c.config.witness_nodes = 5; // (5 * 2/3) as u16 = 3
                                    // num_witnesses arg is ignored when config.witness_nodes != 0.
        assert_eq!(c.witness_quorum(1), 3);
        assert_eq!(c.witness_quorum(100), 3);
        c.config.witness_nodes = 1;
        assert_eq!(c.witness_quorum(99), 1);
    }

    #[test]
    #[should_panic]
    fn witness_quorum_zero_panics() {
        let mut c = Coordinator::zeroed();
        c.config.witness_nodes = 0;
        c.witness_quorum(0);
    }

    // ── batch-size ramp (CoordinatorConfig::get_batch_size) ────────────────────
    fn ramp_config(start: u16, end: u16, warmup_tokens: u64) -> CoordinatorConfig {
        let mut cfg = CoordinatorConfig::zeroed();
        cfg.global_batch_size_start = start;
        cfg.global_batch_size_end = end;
        cfg.global_batch_size_warmup_tokens = warmup_tokens;
        cfg
    }

    #[test]
    fn batch_size_ramp_endpoints_and_midpoint() {
        let cfg = ramp_config(10, 100, 1000);
        assert_eq!(cfg.get_batch_size(0), 10);
        assert_eq!(cfg.get_batch_size(1000), 100);
        assert_eq!(cfg.get_batch_size(500), 55); // 10 + 90*0.5
                                                 // past warmup -> clamps to end
        assert_eq!(cfg.get_batch_size(10_000), 100);
    }

    #[test]
    fn batch_size_ramp_is_monotonic_non_decreasing() {
        let cfg = ramp_config(10, 100, 1000);
        let mut prev = 0u16;
        for tokens in (0..1000).step_by(25) {
            let b = cfg.get_batch_size(tokens);
            assert!(b >= prev, "batch size decreased at {tokens}: {prev} -> {b}");
            prev = b;
        }
    }

    // ── ring buffer: previous_round / previous_previous_round ──────────────────
    // The most bug-prone logic in the crate. The rounds[] array is a ring; at
    // rounds_head == 0 the "previous" slot wraps to the end. A wrong wrap reads
    // a stale round and corrupts committee verification across epochs.
    fn ring_buf_coordinator(head: u32, heights: [u32; NUM_STORED_ROUNDS]) -> Coordinator {
        let mut c = Coordinator::zeroed();
        for (i, &h) in heights.iter().enumerate() {
            c.epoch_state.rounds[i].height = h;
        }
        c.epoch_state.rounds_head = head;
        c
    }

    #[test]
    fn previous_round_returns_slot_before_head() {
        let c = ring_buf_coordinator(2, [10, 20, 30, 40]);
        // head=2 -> current is rounds[2], previous is rounds[1].
        assert_eq!(c.current_round().unwrap().height, 30);
        assert_eq!(c.previous_round().unwrap().height, 20);
    }

    #[test]
    fn previous_round_wraps_at_head_zero() {
        // head=0 with a non-zero current height wraps to the last slot.
        let c = ring_buf_coordinator(0, [100, 20, 30, 400]);
        assert_eq!(c.current_round().unwrap().height, 100);
        assert_eq!(c.previous_round().unwrap().height, 400); // rounds[N-1]
    }

    #[test]
    fn previous_round_none_at_epoch_start() {
        // head=0 and current height 0 -> there is no previous round.
        let c = ring_buf_coordinator(0, [0, 20, 30, 40]);
        assert!(c.previous_round().is_none());
    }

    #[test]
    fn previous_previous_round_wrap_cases() {
        // head=0, current height > 1 -> rounds[N-2] = rounds[2]
        let c = ring_buf_coordinator(0, [5, 20, 300, 40]);
        assert_eq!(c.previous_previous_round().unwrap().height, 300);
        // head=1 -> rounds[N-1] = rounds[3]
        let c = ring_buf_coordinator(1, [10, 20, 30, 400]);
        assert_eq!(c.previous_previous_round().unwrap().height, 400);
        // head=2 -> rounds[0]
        let c = ring_buf_coordinator(2, [100, 20, 30, 40]);
        assert_eq!(c.previous_previous_round().unwrap().height, 100);
        // head=3 -> rounds[1]
        let c = ring_buf_coordinator(3, [10, 200, 30, 40]);
        assert_eq!(c.previous_previous_round().unwrap().height, 200);
    }

    #[test]
    fn previous_previous_round_none_when_height_le_one_at_head_zero() {
        let c = ring_buf_coordinator(0, [1, 20, 30, 40]); // height 1 <= 1
        assert!(c.previous_previous_round().is_none());
        let c = ring_buf_coordinator(0, [0, 20, 30, 40]); // height 0 <= 1
        assert!(c.previous_previous_round().is_none());
    }

    proptest::proptest! {
        #[test]
        fn prop_ring_buffer_indexes_previous_rounds_for_any_head(
            head in 0_usize..NUM_STORED_ROUNDS,
            current_height in any::<u32>(),
            mut heights in any::<[u32; NUM_STORED_ROUNDS]>(),
        ) {
            heights[head] = current_height;
            let coordinator = ring_buf_coordinator(head as u32, heights);

            let expected_previous = if head == 0 && current_height == 0 {
                None
            } else {
                Some(heights[(head + NUM_STORED_ROUNDS - 1) % NUM_STORED_ROUNDS])
            };
            let expected_previous_previous = if head == 0 && current_height <= 1 {
                None
            } else {
                Some(heights[(head + NUM_STORED_ROUNDS - 2) % NUM_STORED_ROUNDS])
            };

            prop_assert_eq!(coordinator.previous_round().map(|round| round.height), expected_previous);
            prop_assert_eq!(
                coordinator.previous_previous_round().map(|round| round.height),
                expected_previous_previous
            );
        }
    }

    // ── trainer_healthy_score_by_witnesses ─────────────────────────────────────
    #[test]
    fn health_score_counts_witnesses_containing_id() {
        let id = identity(7);
        let hash = sha256(id.signer());

        let mut bloom_with = WitnessBloom::default();
        bloom_with.add(&hash);
        let witness_with = Witness {
            proof: WitnessProof {
                position: 0,
                index: 0,
                witness: Default::default(),
            },
            participant_bloom: bloom_with,
            broadcast_bloom: Default::default(),
            broadcast_merkle: Default::default(),
        };
        let witness_without = Witness {
            participant_bloom: Default::default(),
            ..witness_with
        };

        let witnesses = [witness_with, witness_without, witness_with];
        // two of three contain the id
        assert_eq!(
            Coordinator::trainer_healthy_score_by_witnesses(&id, &witnesses),
            2
        );
        // a different id scores 0
        assert_eq!(
            Coordinator::trainer_healthy_score_by_witnesses(&identity(8), &witnesses),
            0
        );
    }

    // ── healthy() for non-trainer committees ───────────────────────────────────
    // Tie-breaker and verifier nodes do not train, so the witness
    // participant_bloom filters carry no evidence about them. They must be
    // reported as healthy (not droppable) rather than panicking.
    #[test]
    fn healthy_non_trainer_committees_are_healthy() {
        let mut coordinator = Coordinator::zeroed();
        let clients: Vec<_> = (0..20u8).map(|i| Client::new(identity(i))).collect();
        coordinator.epoch_state.clients = FixedVec::from_iter(clients);
        coordinator.config.verification_percent = 50;

        // two rounds so previous_round() is valid; previous round drives the
        // committee selection used by healthy().
        coordinator.epoch_state.rounds_head = 1;
        let prev = &mut coordinator.epoch_state.rounds[0];
        prev.height = 5;
        prev.clients_len = 20;
        prev.tie_breaker_tasks = 2;
        prev.random_seed = 42;
        coordinator.epoch_state.rounds[1].height = 6;

        let selection = CommitteeSelection::from_coordinator(&coordinator, -1).unwrap();

        let mut saw_tie_breaker = false;
        let mut saw_verifier = false;
        for (index, client) in coordinator.epoch_state.clients.iter().enumerate() {
            let proof = selection.get_committee(index as u64);
            match proof.committee {
                Committee::TieBreaker => {
                    saw_tie_breaker = true;
                    assert!(
                        coordinator
                            .healthy(&client.id, &proof)
                            .expect("tie-breaker health check should not error"),
                        "tie-breaker node should be healthy"
                    );
                }
                Committee::Verifier => {
                    saw_verifier = true;
                    assert!(
                        coordinator
                            .healthy(&client.id, &proof)
                            .expect("verifier health check should not error"),
                        "verifier node should be healthy"
                    );
                }
                Committee::Trainer => {}
            }
        }
        assert!(saw_tie_breaker, "test must exercise a TieBreaker node");
        assert!(saw_verifier, "test must exercise a Verifier node");
    }

    // ── select_consensus_commitment_by_witnesses ───────────────────────────────
    fn commitment_with_hash(byte: u8) -> Commitment {
        Commitment {
            data_hash: [byte; 32],
            signature: [0u8; 64],
        }
    }

    fn witness_containing(byte: u8) -> Witness {
        let mut bloom = WitnessBloom::default();
        bloom.add(&[byte; 32]);
        Witness {
            proof: WitnessProof {
                position: 0,
                index: 0,
                witness: Default::default(),
            },
            participant_bloom: Default::default(),
            broadcast_bloom: bloom,
            broadcast_merkle: Default::default(),
        }
    }

    #[test]
    fn consensus_returns_none_below_quorum() {
        let commitments = [commitment_with_hash(1), commitment_with_hash(2)];
        // only 1 witness votes for commitment 0; quorum is 2 -> None
        let witnesses = [witness_containing(1)];
        assert_eq!(
            Coordinator::select_consensus_commitment_by_witnesses(&commitments, &witnesses, 2),
            None
        );
    }

    #[test]
    fn consensus_picks_commitment_reaching_quorum() {
        let commitments = [commitment_with_hash(1), commitment_with_hash(2)];
        // 3 witnesses all contain commitment[1]'s hash
        let witnesses = [
            witness_containing(2),
            witness_containing(2),
            witness_containing(2),
        ];
        assert_eq!(
            Coordinator::select_consensus_commitment_by_witnesses(&commitments, &witnesses, 2),
            Some(1)
        );
    }

    #[test]
    fn consensus_tiebreak_picks_highest_score() {
        let commitments = [commitment_with_hash(1), commitment_with_hash(2)];
        let witnesses = [
            witness_containing(1),
            witness_containing(1),
            witness_containing(2),
            witness_containing(2),
            witness_containing(2),
        ];
        // both reach quorum 2; commitment[1] has score 3 > 2 -> index 1
        assert_eq!(
            Coordinator::select_consensus_commitment_by_witnesses(&commitments, &witnesses, 2),
            Some(1)
        );
    }

    #[test]
    fn consensus_ties_are_deterministic_across_order_and_duplicate_commitments() {
        let low = commitment_with_hash(1);
        let high = commitment_with_hash(2);
        let witnesses = [
            witness_containing(1),
            witness_containing(1),
            witness_containing(2),
            witness_containing(2),
        ];

        for commitments in [[low, high], [high, low]] {
            let selected =
                Coordinator::select_consensus_commitment_by_witnesses(&commitments, &witnesses, 2)
                    .unwrap();
            assert_eq!(commitments[selected].data_hash, high.data_hash);
        }

        let duplicates = [low, low];
        assert_eq!(
            Coordinator::select_consensus_commitment_by_witnesses(
                &duplicates,
                &[witness_containing(1), witness_containing(1)],
                2,
            ),
            Some(0)
        );
    }

    // ── withdraw ───────────────────────────────────────────────────────────────
    #[test]
    fn withdraw_marks_healthy_client_and_rejects_others() {
        let mut c = Coordinator::zeroed();
        c.epoch_state.clients =
            FixedVec::from_iter([Client::new(identity(1)), Client::new(identity(2))]);
        // index 0 is healthy -> ok
        assert!(c.withdraw(0).is_ok());
        assert_eq!(c.epoch_state.clients[0].state, ClientState::Withdrawn);
        // withdrawing the same client again -> already withdrawn -> Err
        assert!(c.withdraw(0).is_err());
        // out-of-range index -> Err
        assert!(c.withdraw(99).is_err());
        // index 1 still healthy, untouched
        assert_eq!(c.epoch_state.clients[1].state, ClientState::Healthy);
    }

    #[test]
    fn checkpoint_updates_full_model_checkpoint() {
        let mut coordinator = checkpoint_coordinator(LLMTrainingMethod::Full);
        let Model::LLM(llm) = &mut coordinator.model;
        llm.checkpoint = Checkpoint::P2P(hub_repo("base/model"));

        coordinator
            .checkpoint(&identity(1), 0, Checkpoint::Hub(hub_repo("full/upload")))
            .unwrap();

        let Model::LLM(llm) = coordinator.model;
        assert!(matches!(
            llm.checkpoint,
            Checkpoint::P2P(repo) if repo == hub_repo("full/upload")
        ));
    }

    #[test]
    fn checkpoint_initializes_fresh_lora_adapter_without_changing_base() {
        let mut coordinator = checkpoint_coordinator(LLMTrainingMethod::Lora(LoraConfig {
            rank: 16,
            alpha: 32.0,
            dropout: 0.05,
            init_seed: 42,
            adapter_checkpoint: AdapterCheckpoint::Fresh,
        }));

        coordinator
            .checkpoint(&identity(1), 0, Checkpoint::Hub(hub_repo("adapter/upload")))
            .unwrap();

        let Model::LLM(llm) = coordinator.model;
        assert!(matches!(
            llm.checkpoint,
            Checkpoint::Hub(repo) if repo == hub_repo("base/model")
        ));
        assert!(matches!(
            llm.training_method,
            LLMTrainingMethod::Lora(LoraConfig {
                adapter_checkpoint: AdapterCheckpoint::Hub(repo),
                ..
            }) if repo == hub_repo("adapter/upload")
        ));
    }

    #[test]
    fn checkpoint_preserves_lora_adapter_p2p_mode() {
        let mut coordinator = checkpoint_coordinator(LLMTrainingMethod::Lora(LoraConfig {
            rank: 16,
            alpha: 32.0,
            dropout: 0.05,
            init_seed: 42,
            adapter_checkpoint: AdapterCheckpoint::P2P(hub_repo("adapter/old")),
        }));
        let uploaded = GcsRepo {
            bucket: FixedString::try_from("adapter-bucket").unwrap(),
            prefix: Some(FixedString::try_from("checkpoint").unwrap()),
        };

        coordinator
            .checkpoint(&identity(1), 0, Checkpoint::Gcs(uploaded))
            .unwrap();

        let Model::LLM(llm) = coordinator.model;
        assert!(matches!(
            llm.checkpoint,
            Checkpoint::Hub(repo) if repo == hub_repo("base/model")
        ));
        assert!(matches!(
            llm.training_method,
            LLMTrainingMethod::Lora(LoraConfig {
                adapter_checkpoint: AdapterCheckpoint::P2PGcs(repo),
                ..
            }) if repo == uploaded
        ));
    }

    #[test]
    fn duplicate_and_invalid_checkpoint_updates_do_not_mutate_state() {
        let mut coordinator = checkpoint_coordinator(LLMTrainingMethod::Full);
        let upload = Checkpoint::Hub(hub_repo("full/upload"));

        assert!(coordinator.checkpoint(&identity(1), 0, upload).unwrap());
        assert!(!coordinator.checkpoint(&identity(1), 0, upload).unwrap());

        for invalid in [
            Checkpoint::Ephemeral,
            Checkpoint::Dummy(HubRepo::dummy()),
            Checkpoint::Hub(HubRepo::dummy()),
            Checkpoint::Gcs(GcsRepo::dummy()),
        ] {
            assert!(matches!(
                coordinator.checkpoint(&identity(1), 0, invalid),
                Err(CoordinatorError::InvalidCheckpoint)
            ));
            let Model::LLM(llm) = coordinator.model;
            assert_eq!(llm.checkpoint, upload);
        }
    }

    #[test]
    fn duplicate_regular_witness_is_rejected_without_mutation() {
        let mut coordinator = witness_coordinator(RunState::RoundTrain, 2);
        let witness = witness_for(&coordinator, 0);

        coordinator
            .witness(&identity(1), coordinator.progress.step, witness, 100)
            .unwrap();
        assert!(matches!(
            coordinator.witness(&identity(1), coordinator.progress.step, witness, 100),
            Err(CoordinatorError::DuplicateWitness)
        ));
        assert_eq!(coordinator.current_round().unwrap().witnesses.len(), 1);
        assert_eq!(coordinator.run_state, RunState::RoundTrain);
    }

    #[test]
    fn duplicate_and_forged_warmup_witnesses_are_rejected_safely() {
        let mut coordinator = witness_coordinator(RunState::Warmup, 2);
        let witness = Witness {
            proof: WitnessProof {
                index: 0,
                position: 0,
                witness: Default::default(),
            },
            ..Witness::default()
        };

        coordinator
            .warmup_witness(&identity(1), witness, 100, 42)
            .unwrap();
        assert!(matches!(
            coordinator.warmup_witness(&identity(1), witness, 100, 42),
            Err(CoordinatorError::DuplicateWitness)
        ));

        for forged_index in [1, 99] {
            let forged = Witness {
                proof: WitnessProof {
                    index: forged_index,
                    ..witness.proof
                },
                ..witness
            };
            assert!(matches!(
                coordinator.warmup_witness(&identity(1), forged, 100, 42),
                Err(CoordinatorError::InvalidWitness)
            ));
        }
        assert_eq!(coordinator.current_round().unwrap().witnesses.len(), 1);
    }

    #[test]
    fn regular_witness_must_match_current_step() {
        for (submitted_step, expected_error) in [
            (9, Some(CoordinatorError::StaleWitness)),
            (10, None),
            (11, Some(CoordinatorError::FutureWitness)),
        ] {
            let mut coordinator = witness_coordinator(RunState::RoundTrain, 1);
            let witness = witness_for(&coordinator, 0);
            let result = coordinator.witness(&identity(1), submitted_step, witness, 100);

            match expected_error {
                Some(CoordinatorError::StaleWitness) => {
                    assert!(matches!(result, Err(CoordinatorError::StaleWitness)))
                }
                Some(CoordinatorError::FutureWitness) => {
                    assert!(matches!(result, Err(CoordinatorError::FutureWitness)))
                }
                None => assert!(result.is_ok()),
                _ => unreachable!(),
            }

            if expected_error.is_some() {
                assert!(coordinator.current_round().unwrap().witnesses.is_empty());
                assert_eq!(coordinator.run_state, RunState::RoundTrain);
                assert_eq!(coordinator.progress.step, 10);
            } else {
                assert_eq!(coordinator.current_round().unwrap().witnesses.len(), 1);
                assert_eq!(coordinator.run_state, RunState::RoundWitness);
            }
        }
    }

    // ── Model::check validator ─────────────────────────────────────────────────
    fn good_llm() -> LLM {
        let mut l = LLM::dummy();
        // LLM::dummy() uses the Dummy optimizer, which Model::check rejects.
        l.optimizer = OptimizerDefinition::AdamW {
            betas: [0.9, 0.999],
            weight_decay: 0.0,
            eps: 1e-8,
            clip_grad_norm: None,
        };
        l
    }

    #[test]
    fn model_check_accepts_valid_model() {
        let model = Model::LLM(good_llm());
        assert!(model.check());
    }

    #[test]
    fn model_check_rejects_zero_seq_len() {
        let mut llm = good_llm();
        llm.max_seq_len = 0;
        assert!(!Model::LLM(llm).check());
    }

    #[test]
    fn model_check_rejects_dummy_optimizer() {
        // LLM::dummy() has the Dummy optimizer.
        assert!(!Model::LLM(LLM::dummy()).check());
    }

    #[test]
    fn model_check_rejects_ephemeral_checkpoint() {
        let mut llm = good_llm();
        llm.checkpoint = Checkpoint::Ephemeral;
        assert!(!Model::LLM(llm).check());
    }

    #[test]
    fn model_check_rejects_empty_hub_checkpoint() {
        let mut llm = good_llm();
        llm.checkpoint = Checkpoint::Hub(HubRepo::dummy()); // repo_id empty
        assert!(!Model::LLM(llm).check());
    }

    // ── Coordinator Pod validity ───────────────────────────────────────────────
    // Coordinator is `unsafe impl Pod` and gets reinterpreted from raw bytes by
    // the on-disk timeline. A zeroed Coordinator must round-trip through bytes
    // losslessly (the cross-crate COORD_RECORD_SIZE guard lives in event-sourcing).
    #[test]
    fn coordinator_round_trips_through_bytes() {
        let c = Coordinator::zeroed();
        let bytes = bytemuck::bytes_of(&c);
        let back: &Coordinator = bytemuck::from_bytes(bytes);
        // bytes are identical
        assert_eq!(bytes, bytemuck::bytes_of(back));
        // size is non-zero and a multiple of the u64 word (Pod repr(C) invariant)
        let sz = std::mem::size_of::<Coordinator>();
        assert!(sz > 0);
        assert_eq!(sz % std::mem::size_of::<u64>(), 0);
    }
}
