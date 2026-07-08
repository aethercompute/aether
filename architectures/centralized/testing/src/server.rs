use crate::{test_utils::sample_rand_run_id, COOLDOWN_TIME};
use crate::{MAX_ROUND_TRAIN_TIME, ROUND_WITNESS_TIME, WARMUP_TIME};
use aether_centralized_server::app::App as ServerApp;
use aether_coordinator::{
    model::{Checkpoint, Model, LLM},
    Coordinator, CoordinatorConfig, CoordinatorEpochState, RunState, SOLANA_MAX_NUM_CLIENTS,
};
use aether_coordinator::{Client, Round};
use aether_core::{FixedVec, NodeIdentity};
use bytemuck::Zeroable;
use std::{collections::HashSet, mem::Discriminant, ops::ControlFlow};
use tokio::{
    select,
    sync::{
        mpsc::{self, Receiver},
        oneshot,
    },
};
use tracing::debug;

enum TestingQueryMsg {
    Clients {
        respond_to: oneshot::Sender<FixedVec<Client, SOLANA_MAX_NUM_CLIENTS>>,
    },
    ClientsLen {
        respond_to: oneshot::Sender<usize>,
    },
    PendingClients {
        respond_to: oneshot::Sender<HashSet<NodeIdentity>>,
    },
    PendingClientsLen {
        respond_to: oneshot::Sender<usize>,
    },
    ReadyClients {
        respond_to: oneshot::Sender<HashSet<NodeIdentity>>,
    },
    ReadyClientsLen {
        respond_to: oneshot::Sender<usize>,
    },
    /// All connected clients (syncing + ready).
    ConnectedClientsLen {
        respond_to: oneshot::Sender<usize>,
    },
    RunState {
        respond_to: oneshot::Sender<RunState>,
    },
    Rounds {
        respond_to: oneshot::Sender<[Round; 4]>,
    },
    RoundsHead {
        respond_to: oneshot::Sender<u32>,
    },
    Epoch {
        respond_to: oneshot::Sender<u16>,
    },
    Checkpoint {
        respond_to: oneshot::Sender<Checkpoint>,
    },
    Coordinator {
        respond_to: oneshot::Sender<Coordinator>,
    },
}

struct CoordinatorServer {
    inner: ServerApp,
    query_chan_receiver: Receiver<TestingQueryMsg>,
    port: u16,
    run_id: String,
}

impl CoordinatorServer {
    fn send_response<T>(respond_to: oneshot::Sender<T>, response: T) {
        if respond_to.send(response).is_err() {
            debug!("testing query response receiver dropped");
        }
    }

    pub async fn new(
        query_chan_receiver: Receiver<TestingQueryMsg>,
        min_clients: u16,
        global_batch_size: u16,
        witness_nodes: u16,
    ) -> Self {
        let coordinator_config = CoordinatorConfig {
            warmup_time: WARMUP_TIME,
            cooldown_time: COOLDOWN_TIME,
            max_round_train_time: MAX_ROUND_TRAIN_TIME,
            round_witness_time: ROUND_WITNESS_TIME,
            min_clients,
            init_min_clients: min_clients,
            global_batch_size_start: global_batch_size,
            global_batch_size_end: global_batch_size,
            global_batch_size_warmup_tokens: 0,
            verification_percent: 0,
            witness_nodes,
            total_steps: 100,
            waiting_for_members_extra_time: 2,
            epoch_time: 30,
        };

        let epoch_state = CoordinatorEpochState {
            first_round: true.into(),
            ..CoordinatorEpochState::zeroed()
        };

        let run_id = sample_rand_run_id();
        let coordinator: Coordinator = Coordinator {
            run_id: run_id.as_str().try_into().unwrap(),
            model: Model::LLM(LLM::dummy()),
            config: coordinator_config,
            epoch_state,
            ..Coordinator::zeroed()
        };

        debug!("ServerApp::new() waiting...");

        let server = ServerApp::new(
            false,
            coordinator,
            None,
            Vec::new(),
            None,
            None,
            None,
            Some(WARMUP_TIME),
            true,
            None,
            None,
        )
        .await
        .unwrap();
        debug!("ServerApp::new() done!");

        let port = server.get_port();

        Self {
            inner: server,
            query_chan_receiver,
            port,
            run_id,
        }
    }

    pub async fn handle_message(&mut self, msg: TestingQueryMsg) {
        match msg {
            TestingQueryMsg::Clients { respond_to } => {
                let clients = self.inner.get_clients();
                Self::send_response(respond_to, clients);
            }
            TestingQueryMsg::ClientsLen { respond_to } => {
                let clients = self.inner.get_clients();
                Self::send_response(respond_to, clients.len());
            }
            TestingQueryMsg::PendingClients { respond_to } => {
                let clients = self.inner.get_pending_clients();
                Self::send_response(respond_to, clients);
            }
            TestingQueryMsg::PendingClientsLen { respond_to } => {
                let clients = self.inner.get_pending_clients();
                Self::send_response(respond_to, clients.len());
            }
            TestingQueryMsg::ReadyClients { respond_to } => {
                let clients = self.inner.get_ready_clients();
                Self::send_response(respond_to, clients);
            }
            TestingQueryMsg::ReadyClientsLen { respond_to } => {
                let clients = self.inner.get_ready_clients();
                Self::send_response(respond_to, clients.len());
            }
            TestingQueryMsg::ConnectedClientsLen { respond_to } => {
                let clients = self.inner.get_all_connected_clients();
                Self::send_response(respond_to, clients.len());
            }
            TestingQueryMsg::RunState { respond_to } => {
                let run_state = self.inner.get_run_state();
                Self::send_response(respond_to, run_state);
            }
            TestingQueryMsg::Rounds { respond_to } => {
                let rounds = self.inner.get_rounds();
                Self::send_response(respond_to, rounds);
            }
            TestingQueryMsg::RoundsHead { respond_to } => {
                let rounds = self.inner.get_rounds_head();
                Self::send_response(respond_to, rounds);
            }
            TestingQueryMsg::Epoch { respond_to } => {
                let current_epoch = self.inner.get_current_epoch();
                Self::send_response(respond_to, current_epoch);
            }
            TestingQueryMsg::Checkpoint { respond_to } => {
                let checkpoint = self.inner.get_checkpoint();
                Self::send_response(respond_to, checkpoint);
            }
            TestingQueryMsg::Coordinator { respond_to } => {
                let coordinator = self.inner.get_coordinator();
                Self::send_response(respond_to, coordinator);
            }
        }
    }

    pub async fn run(&mut self) {
        loop {
            select! {
                res = self.inner.poll_next() => {
                    if let ControlFlow::Break(()) = res.unwrap() {
                        break
                    }
                },
                Some(client_msg) = self.query_chan_receiver.recv() => self.handle_message(client_msg).await
            }
        }
    }
}

pub struct CoordinatorServerHandle {
    query_chan_sender: mpsc::Sender<TestingQueryMsg>,
    pub server_port: u16,
    pub run_id: String,
}

impl CoordinatorServerHandle {
    pub async fn new(init_min_clients: u16, global_batch_size: u16, witness_nodes: u16) -> Self {
        debug!("creating coordinator server...");
        let (query_chan_sender, query_chan_receiver) = mpsc::channel(64);

        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_time()
            .enable_io()
            .thread_stack_size(64 * 1024 * 1024)
            .max_blocking_threads(8192)
            .build()
            .unwrap();

        let mut server = rt
            .spawn(CoordinatorServer::new(
                query_chan_receiver,
                init_min_clients,
                global_batch_size,
                witness_nodes,
            ))
            .await
            .unwrap();

        let server_port = server.port;
        let run_id = server.run_id.clone();

        // server.run() drives poll_next(), whose nested tokio::select! state
        // machines (each holding copies of the ~76 KB Coordinator struct) are
        // very large in debug builds. Rust's default thread stack (2 MB) is
        // insufficient and overflows, so we run on a thread with an explicit
        // large stack. This replaces the previous "trust us" workaround which
        // used the default stack size and still overflowed in CI.
        std::thread::Builder::new()
            .stack_size(64 * 1024 * 1024)
            .spawn(move || {
                rt.block_on(server.run());
            })
            .expect("failed to spawn coordinator server thread");
        debug!("coordinator server created on port {server_port}");

        Self {
            query_chan_sender,
            server_port,
            run_id,
        }
    }

    pub async fn get_clients(&self) -> FixedVec<Client, SOLANA_MAX_NUM_CLIENTS> {
        let (send, recv) = oneshot::channel();
        let msg = TestingQueryMsg::Clients { respond_to: send };
        let _ = self.query_chan_sender.send(msg).await;
        recv.await.expect("Coordinator actor task has been killed")
    }

    pub async fn get_clients_len(&self) -> usize {
        let (send, recv) = oneshot::channel();
        let msg = TestingQueryMsg::ClientsLen { respond_to: send };
        let _ = self.query_chan_sender.send(msg).await;
        recv.await.expect("Coordinator actor task has been killed")
    }

    pub async fn get_pending_clients(&self) -> HashSet<NodeIdentity> {
        let (send, recv) = oneshot::channel();
        let msg = TestingQueryMsg::PendingClients { respond_to: send };
        let _ = self.query_chan_sender.send(msg).await;
        recv.await.expect("Coordinator actor task has been killed")
    }

    pub async fn get_pending_clients_len(&self) -> usize {
        let (send, recv) = oneshot::channel();
        let msg = TestingQueryMsg::PendingClientsLen { respond_to: send };
        let _ = self.query_chan_sender.send(msg).await;
        recv.await.expect("Coordinator actor task has been killed")
    }

    pub async fn get_ready_clients(&self) -> HashSet<NodeIdentity> {
        let (send, recv) = oneshot::channel();
        let msg = TestingQueryMsg::ReadyClients { respond_to: send };
        let _ = self.query_chan_sender.send(msg).await;
        recv.await.expect("Coordinator actor task has been killed")
    }

    pub async fn get_ready_clients_len(&self) -> usize {
        let (send, recv) = oneshot::channel();
        let msg = TestingQueryMsg::ReadyClientsLen { respond_to: send };
        let _ = self.query_chan_sender.send(msg).await;
        recv.await.expect("Coordinator actor task has been killed")
    }

    pub async fn get_connected_clients_len(&self) -> usize {
        let (send, recv) = oneshot::channel();
        let msg = TestingQueryMsg::ConnectedClientsLen { respond_to: send };
        let _ = self.query_chan_sender.send(msg).await;
        recv.await.expect("Coordinator actor task has been killed")
    }

    pub async fn get_run_state(&self) -> RunState {
        let (send, recv) = oneshot::channel::<RunState>();
        let msg = TestingQueryMsg::RunState { respond_to: send };
        let _ = self.query_chan_sender.send(msg).await;
        recv.await.expect("Coordinator actor task has been killed")
    }

    pub async fn get_rounds(&self) -> [Round; 4] {
        let (send, recv) = oneshot::channel::<[Round; 4]>();
        let msg = TestingQueryMsg::Rounds { respond_to: send };
        let _ = self.query_chan_sender.send(msg).await;
        recv.await.expect("Coordinator actor task has been killed")
    }

    pub async fn get_rounds_head(&self) -> u32 {
        let (send, recv) = oneshot::channel::<u32>();
        let msg = TestingQueryMsg::RoundsHead { respond_to: send };
        let _ = self.query_chan_sender.send(msg).await;
        recv.await.expect("Coordinator actor task has been killed")
    }

    pub async fn get_current_epoch(&self) -> u16 {
        let (send, recv) = oneshot::channel::<u16>();
        let msg = TestingQueryMsg::Epoch { respond_to: send };
        let _ = self.query_chan_sender.send(msg).await;
        recv.await.expect("Coordinator actor task has been killed")
    }

    // We only care about checking the checkpoint variant but not the hub repo value so we get the discriminant.
    pub async fn get_checkpoint(&self) -> Discriminant<Checkpoint> {
        let (send, recv) = oneshot::channel::<Checkpoint>();
        let msg = TestingQueryMsg::Checkpoint { respond_to: send };
        let _ = self.query_chan_sender.send(msg).await;
        let checkpoint = recv.await.expect("Coordinator actor task has been killed");
        std::mem::discriminant(&checkpoint)
    }

    pub async fn get_coordinator(&self) -> Coordinator {
        let (send, recv) = oneshot::channel::<Coordinator>();
        let msg = TestingQueryMsg::Coordinator { respond_to: send };
        let _ = self.query_chan_sender.send(msg).await;
        recv.await.expect("Coordinator actor task has been killed")
    }
}
