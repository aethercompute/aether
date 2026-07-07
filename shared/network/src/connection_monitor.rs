use aether_event_sourcing::event;
use aether_metrics::SelectedPath;
use iroh::endpoint::{AfterHandshakeOutcome, ConnectionInfo, EndpointHooks};
use iroh::{EndpointId, Watcher};
use n0_future::task::AbortOnDropHandle;
use std::collections::HashMap;
use std::sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard};
use std::time::Duration;
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};
use tokio::task::JoinSet;
use tracing::{debug, info, warn, Instrument};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PeerBandwidth {
    /// No download data yet - peer should be tried at least once
    NotMeasured,
    /// Measured bandwidth in bytes/sec from actual download activity
    Measured(f64),
}

#[derive(Debug, Clone)]
pub struct ConnectionData {
    pub endpoint_id: EndpointId,
    /// Throughput measured from actual downloads
    pub bandwidth: PeerBandwidth,
    pub selected_path: Option<SelectedPath>,
}

impl ConnectionData {
    /// Get the latency from the selected path if available
    pub fn latency(&self) -> Option<Duration> {
        self.selected_path.as_ref().map(|p| p.rtt)
    }
}

/// track active connections and their metadata
#[derive(Clone, Debug)]
pub struct ConnectionMonitor {
    tx: UnboundedSender<ConnectionInfo>,
    connections: Arc<RwLock<HashMap<EndpointId, ConnectionData>>>,
    _task: Arc<AbortOnDropHandle<()>>,
}

impl EndpointHooks for ConnectionMonitor {
    async fn after_handshake(&self, conn: &ConnectionInfo) -> AfterHandshakeOutcome {
        self.tx.send(conn.clone()).ok();
        AfterHandshakeOutcome::Accept
    }
}

impl Default for ConnectionMonitor {
    fn default() -> Self {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let connections = Arc::new(RwLock::new(HashMap::new()));
        let connections_clone = connections.clone();

        let task = tokio::spawn(
            Self::run(rx, connections_clone).instrument(tracing::debug_span!("connection_monitor")),
        );

        Self {
            tx,
            connections,
            _task: Arc::new(AbortOnDropHandle::new(task)),
        }
    }
}

impl ConnectionMonitor {
    fn read_connections(
        connections: &RwLock<HashMap<EndpointId, ConnectionData>>,
    ) -> RwLockReadGuard<'_, HashMap<EndpointId, ConnectionData>> {
        connections.read().unwrap_or_else(|err| {
            warn!("connection monitor RwLock poisoned; recovering read access");
            err.into_inner()
        })
    }

    fn write_connections(
        connections: &RwLock<HashMap<EndpointId, ConnectionData>>,
    ) -> RwLockWriteGuard<'_, HashMap<EndpointId, ConnectionData>> {
        connections.write().unwrap_or_else(|err| {
            warn!("connection monitor RwLock poisoned; recovering write access");
            err.into_inner()
        })
    }

    async fn run(
        mut rx: UnboundedReceiver<ConnectionInfo>,
        connections: Arc<RwLock<HashMap<EndpointId, ConnectionData>>>,
    ) {
        let mut tasks = JoinSet::new();

        loop {
            tokio::select! {
                Some(conn) = rx.recv() => {
                    let remote_id = conn.remote_id();
                    let alpn = String::from_utf8_lossy(conn.alpn()).to_string();

                    let selected_path = Self::extract_selected_path(&conn.paths());

                    if let Some(ref path) = selected_path {
                        info!(
                            remote = %remote_id.fmt_short(),
                            %alpn,
                            path = %path,
                            "new connection"
                        );
                    } else {
                        info!(
                            remote = %remote_id.fmt_short(),
                            %alpn,
                            "new connection (no selected path)"
                        );
                    }

                    {
                        let mut conns = Self::write_connections(&connections);
                        let prev_bandwidth = conns
                            .get(&remote_id)
                            .map(|d| d.bandwidth)
                            .unwrap_or(PeerBandwidth::NotMeasured);
                        conns.insert(remote_id, ConnectionData {
                            endpoint_id: remote_id,
                            bandwidth: prev_bandwidth,
                            selected_path: selected_path.clone(),
                        });
                    }

                    event!(p2p::ConnectionChanged { endpoint_id: remote_id, connection_path: selected_path });

                    // spawn a task to monitor this connection continuously
                    let connections_clone = connections.clone();
                    let paths_watcher = conn.paths();
                    tasks.spawn(async move {
                        let mut update_interval = tokio::time::interval(Duration::from_secs(5));
                        update_interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

                        loop {
                            tokio::select! {
                                _ = update_interval.tick() => {
                                    let selected_path = Self::extract_selected_path(&paths_watcher);

                                    let mut conns = Self::write_connections(&connections_clone);
                                    if let Some(data) = conns.get_mut(&remote_id) {
                                        let path_changed = data.selected_path != selected_path;

                                        // calculate latency delta if both old and new have latency info
                                        let latency_delta = match (&data.selected_path, &selected_path) {
                                            (Some(old_path), Some(new_path)) => {
                                                old_path.rtt.as_millis().abs_diff(new_path.rtt.as_millis())
                                            }
                                            _ => 0,
                                        };

                                        data.selected_path = selected_path.clone();

                                        if path_changed {
                                            if let Some(ref path) = selected_path {
                                                info!(
                                                    remote = %remote_id.fmt_short(),
                                                    path = %path,
                                                    "selected path changed"
                                                );
                                            } else {
                                                info!(
                                                    remote = %remote_id.fmt_short(),
                                                    "selected path removed"
                                                );
                                            }
                                            event!(p2p::ConnectionChanged { endpoint_id: remote_id, connection_path: selected_path });

                                        } else if latency_delta > 50 {
                                            if let Some(ref path) = selected_path {
                                                debug!(
                                                    remote = %remote_id.fmt_short(),
                                                    latency_ms = path.rtt.as_millis(),
                                                    delta_ms = latency_delta,
                                                    "latency changed"
                                                );
                                                event!(p2p::ConnectionLatencyChanged { endpoint_id: remote_id, latency_ms: path.rtt.as_millis() as u64 });
                                            }
                                        }
                                    }
                                }
                                result = conn.closed() => {
                                    match result {
                                        Some((close_reason, stats)) => {
                                            info!(
                                                remote = %remote_id.fmt_short(),
                                                %alpn,
                                                ?close_reason,
                                                udp_rx = stats.udp_rx.bytes,
                                                udp_tx = stats.udp_tx.bytes,
                                                "connection closed"
                                            );
                                        }
                                        None => {
                                            debug!(
                                                remote = %remote_id.fmt_short(),
                                                %alpn,
                                                "connection closed before tracking started"
                                            );
                                        }
                                    }
                                    // Keep the entry (preserving bandwidth) but clear path info
                                    if let Some(data) = Self::write_connections(&connections_clone).get_mut(&remote_id) {
                                        data.selected_path = None;
                                    }
                                    break;
                                }
                            }
                        }
                    }.instrument(tracing::Span::current()));
                }
                Some(res) = tasks.join_next(), if !tasks.is_empty() => {
                    if let Err(err) = res {
                        warn!("connection monitor task failed: {err}");
                    }
                }
                else => break,
            }
        }

        while let Some(res) = tasks.join_next().await {
            if let Err(err) = res {
                warn!("connection monitor task failed: {err}");
            }
        }
    }

    /// extract selected path info from a paths watcher
    fn extract_selected_path<T: Watcher<Value = iroh::endpoint::PathInfoList>>(
        paths_watcher: &T,
    ) -> Option<SelectedPath> {
        let paths = paths_watcher.peek();
        paths
            .iter()
            .find(|p| p.is_selected())
            .map(|path| SelectedPath {
                addr: format!("{:?}", path.remote_addr()),
                rtt: path.rtt().unwrap_or_default(),
            })
    }

    /// update bandwidth for a specific peer from application-level download data
    pub fn update_peer_bandwidth(&self, endpoint_id: &EndpointId, bandwidth: PeerBandwidth) {
        let mut conns = Self::write_connections(&self.connections);
        if let Some(data) = conns.get_mut(endpoint_id) {
            data.bandwidth = bandwidth;
        }
    }

    /// get connection data for a specific endpoint
    pub fn get_connection(&self, endpoint_id: &EndpointId) -> Option<ConnectionData> {
        let conns = Self::read_connections(&self.connections);
        conns.get(endpoint_id).cloned()
    }

    /// get all active connections
    pub fn get_all_connections(&self) -> Vec<ConnectionData> {
        let conns = Self::read_connections(&self.connections);
        conns.values().cloned().collect()
    }

    /// get latency for a specific endpoint
    pub fn get_latency(&self, endpoint_id: &EndpointId) -> Option<Duration> {
        let conns = Self::read_connections(&self.connections);
        conns.get(endpoint_id).and_then(|data| data.latency())
    }

    /// get measured throughput for a specific endpoint
    pub fn get_bandwidth(&self, endpoint_id: &EndpointId) -> Option<PeerBandwidth> {
        let conns = Self::read_connections(&self.connections);
        conns.get(endpoint_id).map(|data| data.bandwidth)
    }

    /// reset all bandwidth measurements to NotMeasured
    pub fn clear_all_bandwidth(&self) {
        let mut conns = Self::write_connections(&self.connections);
        for data in conns.values_mut() {
            data.bandwidth = PeerBandwidth::NotMeasured;
        }
    }
}
