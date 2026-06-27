use anyhow::Result;
use psyche_coordinator::{
    model::{Checkpoint, LLMArchitecture, LLMTrainingDataType, Model},
    ClientState, RunState, NUM_STORED_ROUNDS,
};
use psyche_core::{LearningRateSchedule, OptimizerDefinition};
use russh::keys::ssh_key::{Algorithm, PublicKey};
use russh::server::*;
use russh::{Channel, ChannelId, Pty};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;
use tokio_util::sync::CancellationToken;
use tracing::{info, warn};

use crate::web::{LossPoint, SharedState, WebState};

const BRAND_A: &str = "\x1b[38;2;218;78;138m";
const BRAND_B: &str = "\x1b[38;2;82;184;205m";
const ACCENT_AMBER: &str = "\x1b[38;2;226;136;68m";
const BLOOM_BONE: &str = "\x1b[38;2;226;204;184m";
const DIM: &str = "\x1b[38;2;116;98;104m";
const PANEL_HI: &str = "\x1b[38;2;70;56;64m";
const BOLD: &str = "\x1b[1m";
const RESET: &str = "\x1b[0m";

struct ClientView {
    handle: Handle,
    channel: ChannelId,
    width: u16,
    height: u16,
    frame: u64,
}

#[derive(Clone)]
struct MonitorServer {
    state: SharedState,
    clients: Arc<Mutex<HashMap<usize, ClientView>>>,
    id: usize,
}

impl MonitorServer {
    fn new(state: SharedState) -> Self {
        Self {
            state,
            clients: Arc::new(Mutex::new(HashMap::new())),
            id: 0,
        }
    }

    async fn run(
        &mut self,
        port: u16,
        host_key_path: Option<PathBuf>,
        cancel: CancellationToken,
    ) -> Result<()> {
        self.start_render_loop(cancel.clone());

        let config = Config {
            inactivity_timeout: Some(Duration::from_secs(3600)),
            auth_rejection_time: Duration::from_millis(250),
            auth_rejection_time_initial: Some(Duration::from_millis(0)),
            keys: vec![load_or_generate_host_key(host_key_path)?],
            nodelay: true,
            ..Default::default()
        };

        tokio::select! {
            result = self.run_on_address(Arc::new(config), ("0.0.0.0", port)) => result.map_err(Into::into),
            _ = cancel.cancelled() => Ok(()),
        }
    }

    fn start_render_loop(&self, cancel: CancellationToken) {
        let clients = self.clients.clone();
        let state = self.state.clone();
        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(Duration::from_millis(1000));
            ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

            loop {
                tokio::select! {
                    _ = cancel.cancelled() => break,
                    _ = ticker.tick() => {
                        let snapshot = snapshot_state(&state);
                        let mut clients = clients.lock().await;
                        for client in clients.values_mut() {
                            client.frame = client.frame.wrapping_add(1);
                            let data = render_ansi(&snapshot, client.width, client.height, client.frame);
                            if let Err(err) = client.handle.data(client.channel, data.into_bytes().into()).await {
                                warn!(?err, "failed to send ssh monitor frame");
                            }
                        }
                    }
                }
            }
        });
    }

    async fn draw_client(&mut self) {
        let snapshot = snapshot_state(&self.state);
        let mut clients = self.clients.lock().await;
        if let Some(client) = clients.get_mut(&self.id) {
            let data = render_ansi(&snapshot, client.width, client.height, client.frame);
            if let Err(err) = client
                .handle
                .data(client.channel, data.into_bytes().into())
                .await
            {
                warn!(?err, "failed to send ssh monitor frame");
            }
        }
    }

    async fn resize(&mut self, col_width: u32, row_height: u32) {
        let mut clients = self.clients.lock().await;
        if let Some(client) = clients.get_mut(&self.id) {
            client.width = col_width.max(1).min(u16::MAX as u32) as u16;
            client.height = row_height.max(1).min(u16::MAX as u32) as u16;
        }
    }
}

impl Server for MonitorServer {
    type Handler = Self;

    fn new_client(&mut self, _: Option<std::net::SocketAddr>) -> Self {
        let mut handler = self.clone();
        handler.id = self.id;
        self.id += 1;
        handler
    }
}

impl Handler for MonitorServer {
    type Error = anyhow::Error;

    async fn channel_open_session(
        &mut self,
        channel: Channel<Msg>,
        session: &mut Session,
    ) -> Result<bool, Self::Error> {
        self.clients.lock().await.insert(
            self.id,
            ClientView {
                handle: session.handle(),
                channel: channel.id(),
                width: 80,
                height: 24,
                frame: 0,
            },
        );
        Ok(true)
    }

    async fn auth_none(&mut self, _: &str) -> Result<Auth, Self::Error> {
        Ok(Auth::Accept)
    }

    async fn auth_password(&mut self, _: &str, _: &str) -> Result<Auth, Self::Error> {
        Ok(Auth::Accept)
    }

    async fn auth_publickey_offered(
        &mut self,
        _: &str,
        _: &PublicKey,
    ) -> Result<Auth, Self::Error> {
        Ok(Auth::Accept)
    }

    async fn auth_publickey(&mut self, _: &str, _: &PublicKey) -> Result<Auth, Self::Error> {
        Ok(Auth::Accept)
    }

    async fn shell_request(
        &mut self,
        channel: ChannelId,
        session: &mut Session,
    ) -> Result<(), Self::Error> {
        session.channel_success(channel)?;
        self.draw_client().await;
        Ok(())
    }

    async fn exec_request(
        &mut self,
        channel: ChannelId,
        _: &[u8],
        session: &mut Session,
    ) -> Result<(), Self::Error> {
        session.channel_success(channel)?;
        self.draw_client().await;
        Ok(())
    }

    async fn env_request(
        &mut self,
        channel: ChannelId,
        _: &str,
        _: &str,
        session: &mut Session,
    ) -> Result<(), Self::Error> {
        session.channel_success(channel)?;
        Ok(())
    }

    async fn data(
        &mut self,
        channel: ChannelId,
        data: &[u8],
        session: &mut Session,
    ) -> Result<(), Self::Error> {
        if data == b"q" || data == [3] {
            self.clients.lock().await.remove(&self.id);
            session.close(channel)?;
        }
        Ok(())
    }

    async fn channel_close(&mut self, _: ChannelId, _: &mut Session) -> Result<(), Self::Error> {
        self.clients.lock().await.remove(&self.id);
        Ok(())
    }

    async fn window_change_request(
        &mut self,
        _: ChannelId,
        col_width: u32,
        row_height: u32,
        _: u32,
        _: u32,
        _: &mut Session,
    ) -> Result<(), Self::Error> {
        self.resize(col_width, row_height).await;
        self.draw_client().await;
        Ok(())
    }

    async fn pty_request(
        &mut self,
        channel: ChannelId,
        _: &str,
        col_width: u32,
        row_height: u32,
        _: u32,
        _: u32,
        _: &[(Pty, u32)],
        session: &mut Session,
    ) -> Result<(), Self::Error> {
        self.resize(col_width, row_height).await;
        session.channel_success(channel)?;
        self.draw_client().await;
        Ok(())
    }
}

impl Drop for MonitorServer {
    fn drop(&mut self) {
        let id = self.id;
        let clients = self.clients.clone();
        tokio::spawn(async move {
            clients.lock().await.remove(&id);
        });
    }
}

pub fn start(
    state: SharedState,
    port: u16,
    host_key_path: Option<PathBuf>,
    cancel: CancellationToken,
) {
    tokio::spawn(async move {
        let mut server = MonitorServer::new(state);
        info!(port, "starting ssh training monitor");
        if let Err(err) = server.run(port, host_key_path, cancel).await {
            warn!(?err, "ssh training monitor stopped");
        }
    });
}

fn load_or_generate_host_key(path: Option<PathBuf>) -> Result<russh::keys::PrivateKey> {
    let Some(path) = path else {
        warn!("ssh monitor host key path not set; using ephemeral host key");
        return Ok(russh::keys::PrivateKey::random(
            &mut rand08::thread_rng(),
            Algorithm::Ed25519,
        )?);
    };

    if path.exists() {
        info!(path = %path.display(), "loading ssh monitor host key");
        return Ok(russh::keys::PrivateKey::read_openssh_file(&path)?);
    }

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let key = russh::keys::PrivateKey::random(&mut rand08::thread_rng(), Algorithm::Ed25519)?;
    key.write_openssh_file(&path, russh::keys::ssh_key::LineEnding::LF)?;

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600))?;
    }

    info!(path = %path.display(), "generated ssh monitor host key");
    Ok(key)
}

struct Snapshot {
    waiting: bool,
    run_state: String,
    overview: Vec<String>,
    config: Vec<String>,
    model: Vec<String>,
    timing: Vec<String>,
    clients: Vec<String>,
    rounds: Vec<String>,
    wandb: Vec<String>,
}

fn snapshot_state(state: &SharedState) -> Snapshot {
    let s = state.lock().unwrap();
    snapshot_from_web_state(&s)
}

fn snapshot_from_web_state(s: &WebState) -> Snapshot {
    let Some(coord) = &s.coordinator else {
        return Snapshot {
            waiting: true,
            run_state: "Waiting for coordinator data".into(),
            overview: vec![kv_line("Status", "Waiting for coordinator data")],
            config: Vec::new(),
            model: Vec::new(),
            timing: Vec::new(),
            clients: vec![
                kv_line("Ready", &s.ready_clients.len().to_string()),
                kv_line("Syncing", &s.syncing_clients.len().to_string()),
            ],
            rounds: Vec::new(),
            wandb: wandb_rows(s),
        };
    };

    let mut healthy = 0;
    let mut dropped = 0;
    let mut withdrawn = 0;
    let mut ejected = 0;
    for client in coord.epoch_state.clients.iter() {
        match client.state {
            ClientState::Healthy => healthy += 1,
            ClientState::Dropped => dropped += 1,
            ClientState::Withdrawn => withdrawn += 1,
            ClientState::Ejected => ejected += 1,
        }
    }

    let now = current_unix_timestamp();
    let state_duration = now.saturating_sub(coord.run_state_start_unix_timestamp);
    let epoch_duration = if coord.epoch_state.start_timestamp > 0 {
        format_duration(now.saturating_sub(coord.epoch_state.start_timestamp))
    } else {
        "-".into()
    };
    let remaining_steps = coord.config.total_steps.saturating_sub(coord.progress.step);
    let tokens_per_step = coord.get_target_global_batch_size(coord.current_round()) as f64
        * coord.get_sequence_length() as f64;
    let remaining_tokens = remaining_steps as f64 * tokens_per_step;
    let current_tps = s.loss_history.last().and_then(|p| {
        p.tokens_per_sec
            .is_finite()
            .then_some(p.tokens_per_sec as f64)
    });
    let avg_tps = average_tokens_per_sec(&s.loss_history);
    let weighted_tps = weighted_tokens_per_sec(&s.loss_history);
    let eta_weighted = estimate_remaining_time(remaining_tokens, weighted_tps.unwrap_or(0.0));
    let eta_current = estimate_remaining_time(remaining_tokens, current_tps.unwrap_or(0.0));

    let pending = if coord.pending_pause.is_true() {
        " pending pause"
    } else {
        ""
    };
    let overview = vec![
        kv_line(
            "Run State",
            &format!("{}{pending}", format_run_state(coord.run_state)),
        ),
        kv_line("Run ID", &coord.run_id.to_string()),
        kv_line("Epoch", &coord.progress.epoch.to_string()),
        kv_line(
            "Step",
            &format!("{} / {}", coord.progress.step, coord.config.total_steps),
        ),
        kv_line(
            "Height",
            &coord.epoch_state.rounds[coord.epoch_state.rounds_head as usize]
                .height
                .to_string(),
        ),
        kv_line(
            "Epoch Steps",
            &format!(
                "{} - {}",
                coord.epoch_state.start_step, coord.epoch_state.last_step
            ),
        ),
        kv_line(
            "Clients",
            &format!(
                "{} ({} exited)",
                coord.epoch_state.clients.len(),
                coord.epoch_state.exited_clients.len()
            ),
        ),
        kv_line(
            "Connected",
            &format!(
                "{} ready, {} syncing",
                s.ready_clients.len(),
                s.syncing_clients.len()
            ),
        ),
        kv_line(
            "Server",
            if s.server_addr.is_empty() {
                "-"
            } else {
                &s.server_addr
            },
        ),
        kv_line("State Duration", &format_duration(state_duration)),
        kv_line("Epoch Duration", &epoch_duration),
        kv_line(
            "Data Index",
            &coord.progress.epoch_start_data_index.to_string(),
        ),
        kv_line(
            "First Round",
            bool_str(coord.epoch_state.first_round.is_true()),
        ),
        kv_line(
            "Cold Start",
            bool_str(coord.epoch_state.cold_start_epoch.is_true()),
        ),
    ];

    let config = vec![
        kv_line("Total Steps", &coord.config.total_steps.to_string()),
        kv_line("Epoch Time", &format_duration(coord.config.epoch_time)),
        kv_line("Warmup Time", &format_duration(coord.config.warmup_time)),
        kv_line(
            "Cooldown Time",
            &format_duration(coord.config.cooldown_time),
        ),
        kv_line(
            "Max Train Time",
            &format_duration(coord.config.max_round_train_time),
        ),
        kv_line(
            "Witness Time",
            &format_duration(coord.config.round_witness_time),
        ),
        kv_line(
            "Warmup Tokens",
            &coord.config.global_batch_size_warmup_tokens.to_string(),
        ),
        kv_line("Init Min", &coord.config.init_min_clients.to_string()),
        kv_line("Min Clients", &coord.config.min_clients.to_string()),
        kv_line("Witness Nodes", &coord.config.witness_nodes.to_string()),
        kv_line(
            "Batch Start",
            &coord.config.global_batch_size_start.to_string(),
        ),
        kv_line("Batch End", &coord.config.global_batch_size_end.to_string()),
        kv_line(
            "Verify %",
            &format!("{}%", coord.config.verification_percent),
        ),
        kv_line(
            "Waiting Extra",
            &format_duration(coord.config.waiting_for_members_extra_time.into()),
        ),
    ];

    let model = match &coord.model {
        Model::LLM(llm) => vec![
            kv_line("Architecture", format_llm_architecture(&llm.architecture)),
            kv_line("Max Seq Len", &llm.max_seq_len.to_string()),
            kv_line("Cold Warmup", &llm.cold_start_warmup_steps.to_string()),
            kv_line("Data Type", format_data_type(&llm.data_type)),
            kv_line("Checkpoint", format_checkpoint_label(&llm.checkpoint)),
            kv_line(
                "LR Schedule",
                &format_lr_schedule(&llm.lr_schedule, coord.progress.step),
            ),
            kv_line("Optimizer", &format_optimizer(&llm.optimizer)),
        ],
    };

    let timing = vec![
        kv_line(
            "Elapsed",
            &format_duration_opt(
                s.loss_history
                    .first()
                    .map(|p| now.saturating_sub(p.unix_timestamp))
                    .or_else(|| {
                        (coord.run_state_start_unix_timestamp > 0).then_some(state_duration)
                    }),
            ),
        ),
        kv_line("Remaining", &remaining_steps.to_string()),
        kv_line("Tokens / Step", &format!("{tokens_per_step:.0}")),
        kv_line(
            "ETA Weighted",
            &format!(
                "{} ({})",
                format_duration_opt(eta_weighted),
                format_tps(weighted_tps)
            ),
        ),
        kv_line(
            "ETA Overall",
            &format!(
                "{} ({})",
                format_duration_opt(estimate_remaining_time(
                    remaining_tokens,
                    avg_tps.unwrap_or(0.0)
                )),
                format_tps(avg_tps)
            ),
        ),
        kv_line(
            "ETA Current",
            &format!(
                "{} ({})",
                format_duration_opt(eta_current),
                format_tps(current_tps)
            ),
        ),
    ];

    let mut clients = vec![
        kv_line(
            "Summary",
            &format!("{} total", healthy + dropped + withdrawn + ejected),
        ),
        kv_line("Healthy", &healthy.to_string()),
        kv_line("Dropped", &dropped.to_string()),
        kv_line("Withdrawn", &withdrawn.to_string()),
        kv_line("Ejected", &ejected.to_string()),
        kv_line("Ready", &s.ready_clients.len().to_string()),
        kv_line("Syncing", &s.syncing_clients.len().to_string()),
    ];
    clients.push(table_header(&["Client ID", "Status", "Exited"]));
    if coord.epoch_state.clients.is_empty() {
        clients.push(dim_line("No clients connected"));
    } else {
        for client in coord.epoch_state.clients.iter() {
            clients.push(table_row(&[
                &shorten(&client.id.to_string(), 18),
                &client.state.to_string(),
                &client.exited_height.to_string(),
            ]));
        }
    }
    if !coord.epoch_state.exited_clients.is_empty() {
        clients.push(dim_line("Exited Clients"));
        for client in coord.epoch_state.exited_clients.iter() {
            clients.push(table_row(&[
                &shorten(&client.id.to_string(), 18),
                &client.state.to_string(),
                &client.exited_height.to_string(),
            ]));
        }
    }
    if !s.syncing_clients.is_empty() || !s.ready_clients.is_empty() {
        clients.push(dim_line("Connected - Not Yet Admitted"));
        for client in &s.syncing_clients {
            clients.push(table_row(&[&shorten(client, 18), "Syncing", "-"]));
        }
        for client in &s.ready_clients {
            clients.push(table_row(&[&shorten(client, 18), "Ready", "-"]));
        }
    }

    let head = coord.epoch_state.rounds_head as usize;
    let mut rounds = vec![table_header(&[
        "Height", "Data", "Seed", "Clients", "TB", "Wit",
    ])];
    for offset in 0..NUM_STORED_ROUNDS {
        let idx = (head + NUM_STORED_ROUNDS - offset) % NUM_STORED_ROUNDS;
        let round = &coord.epoch_state.rounds[idx];
        let marker = if offset == 0 { "*" } else { " " };
        rounds.push(table_row(&[
            &format!("{marker}{}", round.height),
            &round.data_index.to_string(),
            &round.random_seed.to_string(),
            &round.clients_len.to_string(),
            &round.tie_breaker_tasks.to_string(),
            &round.witnesses.len().to_string(),
        ]));
    }
    rounds.push(dim_line("* current round"));

    Snapshot {
        waiting: false,
        run_state: format_run_state(coord.run_state).into(),
        overview,
        config,
        model,
        timing,
        clients,
        rounds,
        wandb: wandb_rows(s),
    }
}

fn render_ansi(s: &Snapshot, width: u16, height: u16, frame: u64) -> String {
    if width < 50 || height < 18 {
        return format!(
            "\x1b[2J\x1b[H{BRAND_A}{BOLD}◆ AETHERCOMPUTE{RESET}\r\n\r\n{BLOOM_BONE}Resize terminal to at least 50x18.{RESET}\r\n"
        );
    }

    let mut out = String::new();
    out.push_str("\x1b[?25l\x1b[2J\x1b[H");
    let brand = if frame % 8 == 0 { BRAND_B } else { BRAND_A };
    let state_color = if s.waiting { ACCENT_AMBER } else { BRAND_B };
    let status = if s.waiting {
        "initializing"
    } else {
        &s.run_state
    };
    line(&mut out, &format!("{brand}{BOLD}◆ AETHERCOMPUTE{RESET}  {DIM}Training Monitor{RESET}  {state_color}{status}{RESET}"));
    line(
        &mut out,
        &format!("{PANEL_HI}{}{RESET}", "─".repeat(width as usize)),
    );

    render_panel_pair(
        &mut out,
        width,
        "Overview",
        s.overview.clone(),
        "Timing",
        s.timing.clone(),
    );
    blank(&mut out);

    render_panel_pair(
        &mut out,
        width,
        "Configuration",
        s.config.clone(),
        "Model",
        s.model.clone(),
    );
    blank(&mut out);

    if width >= 100 {
        render_panel_pair(
            &mut out,
            width,
            "Clients",
            fit_rows(s.clients.clone(), height, 14),
            "Rounds",
            fit_rows(s.rounds.clone(), height, 14),
        );
        blank(&mut out);
        render_panel(&mut out, "Weights & Biases", s.wandb.clone());
    } else {
        render_panel(&mut out, "Clients", fit_rows(s.clients.clone(), height, 16));
        blank(&mut out);
        render_panel(&mut out, "Rounds", fit_rows(s.rounds.clone(), height, 16));
        blank(&mut out);
        render_panel(&mut out, "Weights & Biases", s.wandb.clone());
    }

    line(
        &mut out,
        &format!("{DIM}live SSH view · q/Ctrl-C: close · monitor.aethercompute.org{RESET}"),
    );
    out
}

fn render_panel(out: &mut String, title: &str, rows: Vec<String>) {
    let width = rows
        .iter()
        .map(|r| visible_len(r))
        .max()
        .unwrap_or(0)
        .max(title.len() + 2)
        + 4;
    let border_w = width.saturating_sub(2);
    line(
        out,
        &format!(
            "{PANEL_HI}┌─ {BLOOM_BONE}{BOLD}{title}{RESET}{PANEL_HI} {}┐{RESET}",
            "─".repeat(border_w.saturating_sub(title.len() + 3))
        ),
    );
    for row in rows {
        line(out, &boxed_line(&row, width));
    }
    line(out, &format!("{PANEL_HI}└{}┘{RESET}", "─".repeat(border_w)));
}

fn render_panel_pair(
    out: &mut String,
    width: u16,
    left_title: &str,
    mut left: Vec<String>,
    right_title: &str,
    mut right: Vec<String>,
) {
    if width < 100 {
        render_panel(out, left_title, left);
        blank(out);
        render_panel(out, right_title, right);
        return;
    }

    let col_w = (width as usize - 3) / 2;
    let left_top = panel_top(left_title, col_w);
    let right_top = panel_top(right_title, col_w);
    line(out, &format!("{left_top} {right_top}"));

    let rows = left.len().max(right.len());
    left.resize(rows, String::new());
    right.resize(rows, String::new());
    for i in 0..rows {
        line(
            out,
            &format!(
                "{} {}",
                boxed_line(&left[i], col_w),
                boxed_line(&right[i], col_w),
            ),
        );
    }
    line(
        out,
        &format!("{} {}", panel_bottom(col_w), panel_bottom(col_w)),
    );
}

fn panel_top(title: &str, width: usize) -> String {
    let inner = width.saturating_sub(2);
    let title = format!("─ {title} ");
    let fill = inner.saturating_sub(visible_len(&title));
    format!("{PANEL_HI}┌{title}{fill}┐{RESET}", fill = "─".repeat(fill))
}

fn panel_bottom(width: usize) -> String {
    format!("{PANEL_HI}└{}┘{RESET}", "─".repeat(width.saturating_sub(2)))
}

fn boxed_line(content: &str, width: usize) -> String {
    let inner = width.saturating_sub(4);
    let content = truncate_visible(content, inner);
    let pad = inner.saturating_sub(visible_len(&content));
    format!(
        "{PANEL_HI}│{RESET} {}{} {PANEL_HI}│{RESET}",
        content,
        " ".repeat(pad)
    )
}

fn kv_line(key: &str, value: &str) -> String {
    format!("{DIM}{key:<15}{RESET}{BLOOM_BONE}{value}{RESET}")
}

fn dim_line(value: &str) -> String {
    format!("{DIM}{value}{RESET}")
}

fn table_header(cols: &[&str]) -> String {
    format!("{DIM}{}{RESET}", table_plain(cols))
}

fn table_row(cols: &[&str]) -> String {
    format!("{BLOOM_BONE}{}{RESET}", table_plain(cols))
}

fn table_plain(cols: &[&str]) -> String {
    const WIDTHS: [usize; 6] = [18, 12, 8, 8, 6, 6];
    cols.iter()
        .enumerate()
        .map(|(i, col)| {
            format!(
                "{:<width$}",
                shorten(col, WIDTHS.get(i).copied().unwrap_or(10)),
                width = WIDTHS.get(i).copied().unwrap_or(10)
            )
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn fit_rows(mut rows: Vec<String>, height: u16, reserved: usize) -> Vec<String> {
    let max_rows = (height as usize).saturating_sub(reserved).max(6);
    if rows.len() <= max_rows {
        return rows;
    }
    let hidden = rows.len() - max_rows + 1;
    rows.truncate(max_rows.saturating_sub(1));
    rows.push(dim_line(&format!("... {hidden} more")));
    rows
}

fn wandb_rows(s: &WebState) -> Vec<String> {
    let Some(wandb) = &s.wandb else {
        return vec![kv_line("Status", "Disabled or unavailable")];
    };
    let mut rows = vec![
        kv_line("Project", &wandb.project),
        kv_line("Run", &wandb.run_name),
        kv_line("Entity", wandb.entity.as_deref().unwrap_or("-")),
        kv_line("Group", wandb.group.as_deref().unwrap_or("-")),
    ];
    if let Some(entity) = &wandb.entity {
        rows.push(kv_line(
            "URL",
            &format!(
                "https://wandb.ai/{entity}/{}/runs/{}",
                wandb.project, wandb.run_name
            ),
        ));
    }
    rows.push(dim_line(
        "Graphs are available in W&B; SSH shows run metadata.",
    ));
    rows
}

fn visible_len(s: &str) -> usize {
    let mut len = 0;
    let mut chars = s.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '\x1b' && chars.peek() == Some(&'[') {
            chars.next();
            for c in chars.by_ref() {
                if c.is_ascii_alphabetic() {
                    break;
                }
            }
        } else {
            len += 1;
        }
    }
    len
}

fn truncate_visible(s: &str, width: usize) -> String {
    if visible_len(s) <= width {
        return s.to_string();
    }

    let target = width.saturating_sub(1);
    let mut out = String::new();
    let mut len = 0;
    let mut chars = s.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '\x1b' && chars.peek() == Some(&'[') {
            out.push(ch);
            out.push(chars.next().unwrap());
            for c in chars.by_ref() {
                out.push(c);
                if c.is_ascii_alphabetic() {
                    break;
                }
            }
        } else if len < target {
            out.push(ch);
            len += 1;
        } else {
            break;
        }
    }
    out.push('…');
    out.push_str(RESET);
    out
}

fn line(out: &mut String, value: &str) {
    out.push_str(value);
    out.push_str("\r\n");
}

fn blank(out: &mut String) {
    out.push_str("\r\n");
}

fn format_tps(tps: Option<f64>) -> String {
    tps.map(|v| format!("{v:.1} tok/s"))
        .unwrap_or_else(|| "-".into())
}

fn bool_str(value: bool) -> &'static str {
    if value {
        "Yes"
    } else {
        "No"
    }
}

fn average_tokens_per_sec(points: &[LossPoint]) -> Option<f64> {
    let points: Vec<f64> = points
        .iter()
        .filter_map(|p| {
            (p.tokens_per_sec.is_finite() && p.tokens_per_sec > 0.0)
                .then_some(p.tokens_per_sec as f64)
        })
        .collect();
    (!points.is_empty()).then(|| points.iter().sum::<f64>() / points.len() as f64)
}

fn current_unix_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

fn estimate_remaining_time(remaining_tokens: f64, tokens_per_sec: f64) -> Option<u64> {
    if remaining_tokens <= 0.0 {
        Some(0)
    } else if tokens_per_sec.is_finite() && tokens_per_sec > 0.0 {
        Some((remaining_tokens / tokens_per_sec).ceil() as u64)
    } else {
        None
    }
}

fn weighted_tokens_per_sec(points: &[LossPoint]) -> Option<f64> {
    let points: Vec<&LossPoint> = points
        .iter()
        .filter(|p| p.tokens_per_sec.is_finite() && p.tokens_per_sec > 0.0)
        .collect();
    if points.is_empty() {
        return None;
    }
    let mut weighted_total = 0.0;
    let mut total_weight = 0.0;
    for i in 0..points.len() {
        let weight = if i + 1 < points.len() {
            points[i + 1]
                .unix_timestamp
                .saturating_sub(points[i].unix_timestamp)
                .max(1) as f64
        } else if i > 0 {
            points[i]
                .unix_timestamp
                .saturating_sub(points[i - 1].unix_timestamp)
                .max(1) as f64
        } else {
            1.0
        };
        weighted_total += points[i].tokens_per_sec as f64 * weight;
        total_weight += weight;
    }
    (total_weight > 0.0).then(|| weighted_total / total_weight)
}

fn format_duration_opt(secs: Option<u64>) -> String {
    secs.map(format_duration).unwrap_or_else(|| "-".into())
}

fn format_duration(secs: u64) -> String {
    let h = secs / 3600;
    let m = (secs % 3600) / 60;
    let s = secs % 60;
    if h > 0 {
        format!("{}h {:02}m {:02}s", h, m, s)
    } else if m > 0 {
        format!("{}m {:02}s", m, s)
    } else {
        format!("{}s", s)
    }
}

fn shorten(value: &str, max: usize) -> String {
    if value.chars().count() <= max {
        value.into()
    } else {
        format!(
            "{}…",
            value
                .chars()
                .take(max.saturating_sub(1))
                .collect::<String>()
        )
    }
}

fn format_run_state(state: RunState) -> &'static str {
    match state {
        RunState::Uninitialized => "Uninitialized",
        RunState::WaitingForMembers => "Waiting for members",
        RunState::Warmup => "Warmup",
        RunState::RoundTrain => "Training",
        RunState::RoundWitness => "Witness",
        RunState::Cooldown => "Cooldown",
        RunState::Finished => "Finished",
        RunState::Paused => "Paused",
    }
}

fn format_llm_architecture(arch: &LLMArchitecture) -> &'static str {
    match arch {
        LLMArchitecture::HfLlama => "HuggingFace LLaMA",
        LLMArchitecture::HfDeepseek => "HuggingFace DeepSeek",
        LLMArchitecture::HfAuto => "HuggingFace Auto",
        LLMArchitecture::Torchtitan => "Torchtitan",
    }
}

fn format_data_type(dt: &LLMTrainingDataType) -> &'static str {
    match dt {
        LLMTrainingDataType::Pretraining => "Pretraining",
        LLMTrainingDataType::Finetuning => "Finetuning",
    }
}

fn format_lr_schedule(schedule: &LearningRateSchedule, current_step: u32) -> String {
    let schedule_type = match schedule {
        LearningRateSchedule::Constant(_) => "Constant",
        LearningRateSchedule::Linear(_) => "Linear",
        LearningRateSchedule::Cosine(_) => "Cosine",
        LearningRateSchedule::WarmupStableDecay(_) => "WarmupStableDecay",
    };
    format!(
        "{} warmup={} init={:.8} current={:.8}",
        schedule_type,
        schedule.get_warmup_steps(),
        schedule.get_warmup_init_lr(),
        schedule.get_lr(current_step)
    )
}

fn format_optimizer(opt: &OptimizerDefinition) -> String {
    match opt {
        OptimizerDefinition::Dummy => "Dummy".into(),
        OptimizerDefinition::AdamW {
            betas,
            weight_decay,
            eps,
            clip_grad_norm,
        } => format!(
            "AdamW beta=[{},{}] wd={} eps={} clip={}",
            betas[0],
            betas[1],
            weight_decay,
            eps,
            clip_grad_norm
                .map(|v| v.to_string())
                .unwrap_or_else(|| "None".into())
        ),
        OptimizerDefinition::Distro {
            clip_grad_norm,
            weight_decay,
            compression_decay,
            compression_topk,
            compression_chunk,
            quantize_1bit,
        } => format!(
            "Distro clip={} wd={} decay={} topk={} chunk={} 1bit={}",
            clip_grad_norm
                .map(|v| v.to_string())
                .unwrap_or_else(|| "None".into()),
            weight_decay
                .map(|v| v.to_string())
                .unwrap_or_else(|| "None".into()),
            compression_decay,
            compression_topk,
            compression_chunk,
            quantize_1bit
        ),
    }
}

fn format_checkpoint_label(cp: &Checkpoint) -> &'static str {
    match cp {
        Checkpoint::Ephemeral => "Ephemeral",
        Checkpoint::Dummy(_) => "Dummy",
        Checkpoint::Hub(_) => "Hub",
        Checkpoint::P2P(_) => "P2P",
        Checkpoint::Gcs(_) => "GCS",
        Checkpoint::P2PGcs(_) => "P2P+GCS",
    }
}
