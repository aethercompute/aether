use axum::{
    extract::State,
    response::Html,
    routing::get,
    Router,
};
use psyche_coordinator::{Coordinator, RunState};
use serde::Serialize;
use std::sync::{Arc, Mutex};

#[derive(Clone, Debug, Serialize)]
pub struct LossPoint {
    pub step: u32,
    pub loss: f32,
    pub tokens_per_sec: f32,
}

#[derive(Clone)]
pub struct WebState {
    pub coordinator: Option<Coordinator>,
    pub loss_history: Vec<LossPoint>,
    pub pending_clients: Vec<String>,
    pub server_addr: String,
}

type SharedState = Arc<Mutex<WebState>>;

pub fn start(
    state: WebState,
    port: u16,
    cancel: tokio_util::sync::CancellationToken,
) -> SharedState {
    let shared = Arc::new(Mutex::new(state));
    let app = Router::new()
        .route("/", get(index))
        .route("/partials/overview", get(overview_partial))
        .route("/partials/clients", get(clients_partial))
        .route("/partials/loss", get(loss_partial))
        .route("/api/state", get(api_state))
        .with_state(shared.clone());

    tokio::spawn(async move {
        let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{port}"))
            .await
            .expect("Failed to bind web server");
        axum::serve(listener, app)
            .with_graceful_shutdown(async move {
                cancel.cancelled().await;
            })
            .await
            .ok();
    });

    shared
}

async fn index() -> Html<&'static str> {
    Html(INDEX_HTML)
}

async fn overview_partial(State(state): State<SharedState>) -> Html<String> {
    let s = state.lock().unwrap();
    match &s.coordinator {
        Some(coord) => {
            let run_state = format_run_state(coord.run_state);
            let clients_count = coord.epoch_state.clients.len();
            let exited = coord.epoch_state.exited_clients.len();
            let step = coord.progress.step;
            let total_steps = coord.config.total_steps;
            let epoch = coord.progress.epoch;
            let height = coord.epoch_state.rounds[coord.epoch_state.rounds_head as usize].height;
            Html(format!(
                r#"<div class="stat-grid">
<div class="stat"><span class="stat-label">Run State</span><span class="stat-value run-state">{run_state}</span></div>
<div class="stat"><span class="stat-label">Step</span><span class="stat-value">{step} / {total_steps}</span></div>
<div class="stat"><span class="stat-label">Epoch</span><span class="stat-value">{epoch}</span></div>
<div class="stat"><span class="stat-label">Height (Round)</span><span class="stat-value">{height}</span></div>
<div class="stat"><span class="stat-label">Clients</span><span class="stat-value">{clients_count} ({exited} exited)</span></div>
<div class="stat"><span class="stat-label">Server</span><span class="stat-value" style="font-size:14px">{server_addr}</span></div>
</div>"#,
                run_state = run_state,
                step = step,
                total_steps = total_steps,
                epoch = epoch,
                height = height,
                clients_count = clients_count,
                exited = exited,
                server_addr = s.server_addr,
            ))
        }
        None => Html(
            r#"<div class="stat-grid"><div class="stat"><span class="stat-value">Waiting for coordinator data...</span></div></div>"#.into(),
        ),
    }
}

async fn clients_partial(State(state): State<SharedState>) -> Html<String> {
    let s = state.lock().unwrap();
    match &s.coordinator {
        Some(coord) => {
            let mut rows = String::new();
            for i in 0..coord.epoch_state.clients.len() {
                let client = &coord.epoch_state.clients[i];
                let id = client.id.to_string();
                let state_str = format!("{}", client.state);
                let exited = client.exited_height;
                rows.push_str(&format!(
                    r#"<tr><td>{}</td><td><span class="badge badge-{}">{}</span></td><td>{}</td></tr>"#,
                    id, state_str, state_str, exited,
                ));
            }
            if rows.is_empty() {
                rows = r#"<tr><td colspan="3" style="color:#8b949e;text-align:center;padding:20px;">No clients connected</td></tr>"#.into();
            }
            Html(format!(
                r#"<table class="clients-table">
<thead><tr><th>Client ID</th><th>Status</th><th>Exited Height</th></tr></thead>
<tbody>{rows}</tbody>
</table>"#,
                rows = rows,
            ))
        }
        None => Html(
            r#"<div style="color:#8b949e;text-align:center;padding:20px;">Waiting for coordinator data...</div>"#.into(),
        ),
    }
}

fn render_loss_svg(losses: &[LossPoint]) -> String {
    let width: f64 = 800.0;
    let height: f64 = 250.0;
    let pad_top = 20.0;
    let pad_bot = 30.0;
    let pad_left = 60.0;
    let pad_right = 20.0;
    let plot_x0 = pad_left;
    let plot_x1 = width - pad_right;
    let plot_y0 = pad_top;
    let plot_y1 = height - pad_bot;
    let plot_w = plot_x1 - plot_x0;
    let plot_h = plot_y1 - plot_y0;

    let filtered: Vec<&LossPoint> = losses.iter().filter(|l| l.loss.is_finite()).collect();
    let n = filtered.len();
    if n < 2 {
        return r#"<div style="color:#8b949e;text-align:center;padding:40px;">Not enough loss data yet</div>"#.into();
    }

    let steps: Vec<u32> = filtered.iter().map(|l| l.step).collect();
    let vals: Vec<f32> = filtered.iter().map(|l| l.loss).collect();

    let min_step = *steps.first().unwrap_or(&0) as f64;
    let max_step = *steps.last().unwrap_or(&1) as f64;
    let step_range = (max_step - min_step).max(1.0);

    let min_loss = vals.iter().copied().fold(f32::INFINITY, |a, b| a.min(b));
    let max_loss = vals.iter().copied().fold(f32::NEG_INFINITY, |a, b| a.max(b));
    let loss_range = (max_loss - min_loss).max(0.01) as f64;

    let mut points: Vec<String> = Vec::with_capacity(n);
    for i in 0..n {
        let x = plot_x0 + (steps[i] as f64 - min_step) / step_range * plot_w;
        let y = plot_y1 - (vals[i] as f64 - min_loss as f64) / loss_range * plot_h;
        points.push(format!("{:.1},{:.1}", x, y));
    }
    let points_str = points.join(" ");

    let y_ticks = 5;
    let mut y_labels = String::new();
    for i in 0..=y_ticks {
        let val = min_loss as f64 + (max_loss - min_loss) as f64 * (i as f64 / y_ticks as f64);
        let y = plot_y1 - (i as f64 / y_ticks as f64) * plot_h;
        y_labels.push_str(&format!(
            r##"<text x="{}" y="{}" text-anchor="end" fill="#8b949e" font-size="11">{:.2}</text>"##,
            pad_left - 5.0,
            y + 4.0,
            val,
        ));
    }

    let x_ticks = 6;
    let mut x_labels = String::new();
    for i in 0..=x_ticks {
        let val = min_step + (max_step - min_step) * (i as f64 / x_ticks as f64);
        let x = plot_x0 + (i as f64 / x_ticks as f64) * plot_w;
        x_labels.push_str(&format!(
            r##"<text x="{x}" y="{}" text-anchor="middle" fill="#8b949e" font-size="11">{:.0}</text>"##,
            height - 8.0,
            val,
        ));
    }

    let last_loss = vals.last().copied().unwrap_or(0.0);

    format!(
        r##"<div>
<div style="margin-bottom:8px;">
<span style="color: #8b949e; font-size:12px;">Latest Loss:</span>
<span style="color: #c9d1d9; font-size:20px; font-weight:bold;">{last_loss:.4}</span>
<span style="color: #8b949e; font-size:12px; margin-left:12px;">Steps: {min_step} - {max_step}</span>
</div>
<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">
<rect x="{plot_x0}" y="{plot_y0}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#30363d" stroke-width="1"/>
{y_labels}
{x_labels}
<polyline points="{points_str}" fill="none" stroke="#58a6ff" stroke-width="2"/>
</svg>
</div>"##,
        last_loss = last_loss,
        min_step = min_step,
        max_step = max_step,
        width = width,
        height = height,
        plot_x0 = plot_x0,
        plot_y0 = plot_y0,
        plot_w = plot_w,
        plot_h = plot_h,
        y_labels = y_labels,
        x_labels = x_labels,
        points_str = points_str,
    )
}

async fn loss_partial(State(state): State<SharedState>) -> Html<String> {
    let s = state.lock().unwrap();
    let svg = render_loss_svg(&s.loss_history);
    Html(svg)
}

async fn api_state(State(state): State<SharedState>) -> impl axum::response::IntoResponse {
    let s = state.lock().unwrap();
    let json = serde_json::json!({
        "coordinator": &s.coordinator,
        "loss_history": &s.loss_history,
        "pending_clients": &s.pending_clients,
        "server_addr": &s.server_addr,
    });
    axum::Json(json)
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

const INDEX_HTML: &str = r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Psyche Monitor</title>
<script src="https://unpkg.com/htmx.org@2.0.4"></script>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Courier New',monospace;background:#0d1117;color:#c9d1d9;padding:24px}
h1{color:#58a6ff;font-size:24px;margin-bottom:20px}
.dashboard{display:flex;flex-direction:column;gap:16px;max-width:1000px}
.panel{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:16px}
.panel h2{color:#8b949e;font-size:13px;text-transform:uppercase;letter-spacing:1px;margin-bottom:12px}
.stat-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:12px}
.stat{display:flex;flex-direction:column}
.stat-label{color:#8b949e;font-size:11px;margin-bottom:2px}
.stat-value{color:#c9d1d9;font-size:22px;font-weight:bold}
.run-state{color:#58a6ff}
.clients-table{width:100%;border-collapse:collapse;font-size:13px}
.clients-table th{text-align:left;color:#8b949e;font-size:11px;padding:6px 8px;border-bottom:1px solid #30363d}
.clients-table td{padding:6px 8px;border-bottom:1px solid #21262d}
.badge{display:inline-block;padding:2px 8px;border-radius:12px;font-size:11px;font-weight:bold}
.badge-Healthy{background:#1b4721;color:#3fb950}
.badge-Dropped{background:#47211b;color:#f85149}
.badge-Withdrawn{background:#47361b;color:#d29922}
.badge-Ejected{background:#47211b;color:#f85149}
</style>
</head>
<body>
<h1>Psyche Monitor</h1>
<div class="dashboard">
<div class="panel" id="overview-panel"><h2>Overview</h2><div hx-get="/partials/overview" hx-trigger="every 2s" hx-swap="innerHTML">Loading...</div></div>
<div class="panel" id="clients-panel"><h2>Clients</h2><div hx-get="/partials/clients" hx-trigger="every 2s" hx-swap="innerHTML">Loading...</div></div>
<div class="panel" id="loss-panel"><h2>Loss</h2><div hx-get="/partials/loss" hx-trigger="every 5s" hx-swap="innerHTML">Loading...</div></div>
</div>
</body>
</html>"#;
