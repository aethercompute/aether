//! Runtime environment resolution + a streamed, animatable `cargo build` of the
//! training client, and the final `exec` into that client.

use crate::config;
use anyhow::Result;
use std::{
    env,
    io::{BufRead, BufReader},
    path::PathBuf,
    process::{Child, Command, Stdio},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

// --- Environment -----------------------------------------------------------

/// Resolved sandbox/system environment used both to build the client and to run
/// it. Prefers the installer's sandbox under `.aethercompute/` and falls back to
/// the user's system toolchain when the sandbox isn't present.
pub struct Env {
    pub cargo: PathBuf,
    pub rustup_home: Option<PathBuf>,
    pub cargo_home: Option<PathBuf>,
    /// All directories that must be on the loader path for libtorch to resolve:
    /// `<torch>/lib` plus every `nvidia/*/lib` from the CUDA pip wheels.
    pub torch_lib_dirs: Vec<PathBuf>,
}

impl Env {
    pub fn detect() -> Self {
        let cargo_home = config::sandbox_cargo_home();
        let rustup_home = config::sandbox_rustup_home();

        let sandboxed_cargo = cargo_home.join("bin").join("cargo");
        let cargo = if sandboxed_cargo.exists() {
            sandboxed_cargo
        } else {
            PathBuf::from("cargo")
        };

        let rustup_home = rustup_home.exists().then_some(rustup_home);
        let cargo_home = cargo_home.exists().then_some(cargo_home);

        let torch_lib_dirs = detect_torch_lib_dirs();

        Self {
            cargo,
            rustup_home,
            cargo_home,
            torch_lib_dirs,
        }
    }

    /// Apply the full sandbox + libtorch environment to a command. Used for
    /// cargo builds (torch-sys needs libtorch at build time) and for running
    /// the client (it needs libtorch at run time).
    pub fn apply(&self, cmd: &mut Command) {
        if let Some(h) = &self.rustup_home {
            cmd.env("RUSTUP_HOME", h);
        }
        if let Some(h) = &self.cargo_home {
            cmd.env("CARGO_HOME", h);
        }
        cmd.env("LIBTORCH_USE_PYTORCH", "1")
            .env("LIBTORCH_BYPASS_VERSION_CHECK", "1")
            .env("RUST_MIN_STACK", "268435456");
        prepend_library_paths(cmd, "LD_LIBRARY_PATH", &self.torch_lib_dirs);
        prepend_library_paths(cmd, "DYLD_LIBRARY_PATH", &self.torch_lib_dirs);
    }
}

fn prepend_library_paths(cmd: &mut Command, var: &str, entries: &[PathBuf]) {
    if entries.is_empty() {
        return;
    }

    let mut paths = entries.to_vec();
    if let Some(existing) = env::var_os(var) {
        paths.extend(env::split_paths(&existing));
    }
    if let Ok(joined) = env::join_paths(paths) {
        cmd.env(var, joined);
    }
}

/// Locate every lib dir libtorch needs. Collects `<torch>/lib` plus each
/// `nvidia/*/lib` shipped by the CUDA pip wheels (libcudart, libcublas, …).
///
/// Prefer the sandbox venv over system Python. torch-sys links against libtorch
/// C++ symbols, and arbitrary system torch versions can differ at link time.
fn detect_torch_lib_dirs() -> Vec<PathBuf> {
    let mut candidates: Vec<PathBuf> = Vec::new();
    let venv_py = config::sandbox_venv().join("bin").join("python");
    if venv_py.exists() && !candidates.contains(&venv_py) {
        candidates.push(venv_py);
    }
    if let Some(p) = which("python3").or_else(|| which("python")) {
        candidates.push(p);
    }
    for python in &candidates {
        if let Some(dirs) = probe_torch_lib_dirs(python) {
            return dirs;
        }
    }
    Vec::new()
}

/// Run the dir-collecting snippet against one python; returns `Some` only when
/// that python can actually `import torch`.
fn probe_torch_lib_dirs(python: &PathBuf) -> Option<Vec<PathBuf>> {
    let script = "import pathlib\n\
try:\n    import torch\nexcept Exception:\n    raise SystemExit(1)\n\
torch_file = pathlib.Path(torch.__file__).resolve()\n\
dirs = [str(torch_file.parent / 'lib')]\n\
site = torch_file.parent.parent\n\
nv = site / 'nvidia'\n\
if nv.is_dir():\n    dirs += [str(d) for d in sorted(nv.glob('*/lib'))]\n\
print(':'.join(dirs))";
    let out = Command::new(python).arg("-c").arg(script).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout);
    let dirs: Vec<PathBuf> = s
        .trim()
        .split(':')
        .filter(|d| !d.is_empty())
        .map(PathBuf::from)
        .filter(|p| p.is_dir())
        .collect();
    if dirs.is_empty() {
        None
    } else {
        Some(dirs)
    }
}

fn which(bin: &str) -> Option<PathBuf> {
    let out = Command::new("sh")
        .arg("-c")
        .arg(format!("command -v {bin}"))
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout);
    let p = PathBuf::from(s.trim());
    p.exists().then_some(p)
}

// --- Build orchestration ---------------------------------------------------

#[derive(Clone, Debug)]
pub enum BuildState {
    Running,
    Success,
    Failed(String),
}

#[derive(Clone, Debug)]
pub struct BuildSnapshot {
    pub state: BuildState,
    pub elapsed: Duration,
    /// Tail of captured output, oldest -> newest.
    pub lines: Vec<String>,
    /// Rough proxy for progress: how many "Compiling" lines we've seen.
    pub compiles: usize,
    #[allow(dead_code)]
    pub crate_name: String,
}

struct Shared {
    state: BuildState,
    started: Instant,
    lines: std::collections::VecDeque<String>,
    compiles: usize,
    crate_name: String,
}

/// A backgrounded `cargo build` whose output can be polled from the UI thread.
pub struct BuildJob {
    shared: Arc<Mutex<Shared>>,
    cancel: Arc<AtomicBool>,
    join: Option<JoinHandle<()>>,
}

impl BuildJob {
    /// Start a backgrounded build. When `force` is true, `torch-sys`'s build
    /// artifacts are cleaned first so it re-detects the active libtorch — this
    /// is needed when a stale client was linked against a different/older
    /// libtorch than the one currently installed in the sandbox.
    pub fn start(crate_name: &str, force: bool) -> Self {
        let crate_name = crate_name.to_string();
        let shared = Arc::new(Mutex::new(Shared {
            state: BuildState::Running,
            started: Instant::now(),
            lines: std::collections::VecDeque::with_capacity(512),
            compiles: 0,
            crate_name: crate_name.clone(),
        }));

        let env = Env::detect();
        let shared_for_thread = shared.clone();
        let cancel = Arc::new(AtomicBool::new(false));
        let cancel_for_thread = cancel.clone();
        let join = thread::spawn(move || {
            if force {
                push_line(
                    &shared_for_thread,
                    "forcing torch-sys rebuild (libtorch changed)".into(),
                );
                if cancel_for_thread.load(Ordering::Relaxed) {
                    set_state(
                        &shared_for_thread,
                        BuildState::Failed("build cancelled".into()),
                    );
                    return;
                }
                let mut clean = Command::new(&env.cargo);
                clean.arg("clean").arg("-p").arg("torch-sys");
                env.apply(&mut clean);
                clean.current_dir(config::repo_root());
                let _ = clean.output(); // best-effort; fast
            }

            let mut cmd = Command::new(&env.cargo);
            cmd.arg("build")
                .arg("--release")
                .arg("-p")
                .arg(&crate_name)
                .arg("--features")
                .arg("python")
                .current_dir(config::repo_root())
                .stdout(Stdio::null())
                .stderr(Stdio::piped());
            env.apply(&mut cmd);

            let child = cmd.spawn();
            let mut child = match child {
                Ok(c) => c,
                Err(e) => {
                    push_line(&shared_for_thread, format!("failed to start cargo: {e}"));
                    set_state(&shared_for_thread, BuildState::Failed(e.to_string()));
                    return;
                }
            };
            if let Some(stderr) = child.stderr.take() {
                let shared_for_stderr = shared_for_thread.clone();
                let stderr_reader = thread::spawn(move || {
                    for line in BufReader::new(stderr).lines().map_while(Result::ok) {
                        push_line(&shared_for_stderr, line);
                    }
                });

                let state = wait_for_build_child(child, &cancel_for_thread, &shared_for_thread);
                let _ = stderr_reader.join();
                set_state(&shared_for_thread, state);
            } else {
                let state = wait_for_build_child(child, &cancel_for_thread, &shared_for_thread);
                set_state(&shared_for_thread, state);
            }
        });

        Self {
            shared,
            cancel,
            join: Some(join),
        }
    }

    pub fn snapshot(&self) -> BuildSnapshot {
        let guard = self.shared.lock().unwrap();
        let tail: Vec<String> = guard
            .lines
            .iter()
            .rev()
            .take(16)
            .filter(|l| !l.trim().is_empty())
            .cloned()
            .collect();
        let tail: Vec<String> = tail.into_iter().rev().collect();
        BuildSnapshot {
            state: guard.state.clone(),
            elapsed: guard.started.elapsed(),
            compiles: guard.compiles,
            lines: tail,
            crate_name: guard.crate_name.clone(),
        }
    }
}

fn wait_for_build_child(
    mut child: Child,
    cancel: &AtomicBool,
    shared: &Arc<Mutex<Shared>>,
) -> BuildState {
    loop {
        if cancel.load(Ordering::Relaxed) {
            if let Err(err) = child.kill() {
                push_line(shared, format!("failed to stop cargo: {err}"));
            }
            let _ = child.wait();
            return BuildState::Failed("build cancelled".into());
        }

        match child.try_wait() {
            Ok(Some(status)) if status.success() => return BuildState::Success,
            Ok(Some(status)) => return BuildState::Failed(format!("cargo exited with {status}")),
            Ok(None) => thread::sleep(Duration::from_millis(100)),
            Err(err) => return BuildState::Failed(err.to_string()),
        }
    }
}

impl Drop for BuildJob {
    fn drop(&mut self) {
        self.cancel.store(true, Ordering::Relaxed);
        if let Some(join) = self.join.take() {
            if join.join().is_err() {
                push_line(&self.shared, "build worker thread panicked".into());
                set_state(
                    &self.shared,
                    BuildState::Failed("build worker thread panicked".into()),
                );
            }
        }
    }
}

fn push_line(shared: &Arc<Mutex<Shared>>, line: String) {
    let mut g = shared.lock().unwrap();
    if line.starts_with("    Compiling ") {
        g.compiles += 1;
    }
    g.lines.push_back(line);
    if g.lines.len() > 512 {
        g.lines.pop_front();
    }
}

fn set_state(shared: &Arc<Mutex<Shared>>, state: BuildState) {
    let mut g = shared.lock().unwrap();
    // Don't overwrite a terminal state (e.g. a stray late line).
    if matches!(g.state, BuildState::Success | BuildState::Failed(_))
        && !matches!(state, BuildState::Success | BuildState::Failed(_))
    {
        return;
    }
    g.state = state;
}

/// Smoke-test the existing client binary against the active libtorch. The
/// dynamic linker resolves libtorch symbols at process start, so even
/// `--help` fails (e.g. with an "undefined symbol" error) when the binary was
/// linked against a different libtorch than the one currently installed.
pub fn client_runs() -> bool {
    client_check_error().is_none()
}

pub fn client_check_error() -> Option<String> {
    let bin = config::client_bin();
    if !bin.exists() {
        return Some(format!("client binary not found at {}", bin.display()));
    }
    let env = Env::detect();
    let mut cmd = Command::new(&bin);
    cmd.arg("--help");
    env.apply(&mut cmd);
    cmd.stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped());

    match cmd.output() {
        Ok(o) if o.status.success() => None,
        Ok(o) => {
            let stderr = String::from_utf8_lossy(&o.stderr);
            let dirs = env
                .torch_lib_dirs
                .iter()
                .map(|p| p.display().to_string())
                .collect::<Vec<_>>()
                .join(":");
            Some(format!(
                "binary failed smoke test with status {}\n\nstderr:\n{}\n\ndetected torch lib dirs:\n{}",
                o.status,
                stderr.trim(),
                if dirs.is_empty() { "<none>" } else { &dirs }
            ))
        }
        Err(e) => Some(format!("failed to run {} --help: {e}", bin.display())),
    }
}

/// True if libtorch was (re)installed more recently than the client binary was
/// built — i.e. the binary is ABI-stale relative to the active torch. A rebuild
/// is needed even if the smoke check passes, since a 2.x-built binary running on
/// 2.y can load but then crash on ABI differences.
pub fn torch_changed_since_build() -> bool {
    let Ok(bin_meta) = std::fs::metadata(config::client_bin()) else {
        return false;
    };
    let Ok(bin_mod) = bin_meta.modified() else {
        return false;
    };
    let Some(torch_dir) = Env::detect().torch_lib_dirs.first().cloned() else {
        return false;
    };
    let torch_so = torch_dir.join("libtorch.so");
    let Ok(t_meta) = std::fs::metadata(&torch_so) else {
        return false;
    };
    matches!(t_meta.modified(), Ok(t) if t > bin_mod)
}

/// True if any client source file (or build manifest) under `shared/` /
/// `architecture/` / the workspace root is newer than the client binary — i.e.
/// the checked-out source has changed since the binary was built and the client
/// must be recompiled even though it still loads. Without this, a `git pull` that
/// changes the on-wire types (e.g. adding an `OptimizerDefinition` variant) leaves
/// the volunteer exec'ing a stale client that deserializes the new payload as
/// "Serde Deserialization Error".
pub fn source_changed_since_build() -> bool {
    let Some(bin_mod) = bin_modified() else {
        return false; // no binary yet — handled by the `bin_exists` check
    };
    let root = config::repo_root();
    for sub in ["shared", "architectures"] {
        if newer_file_in(&root.join(sub), &bin_mod) {
            return true;
        }
    }
    for manifest in ["Cargo.toml", "Cargo.lock", "rust-toolchain.toml"] {
        if let Ok(m) = std::fs::metadata(root.join(manifest)) {
            if matches!(m.modified(), Ok(t) if t > bin_mod) {
                return true;
            }
        }
    }
    false
}

fn bin_modified() -> Option<std::time::SystemTime> {
    std::fs::metadata(config::client_bin())
        .ok()
        .and_then(|m| m.modified().ok())
}

/// Recursively look for a build-input file (`.rs` / `.toml` / `.lock`) newer than
/// `threshold`. Skips generated / vendored / build-artifact trees.
fn newer_file_in(dir: &std::path::Path, threshold: &std::time::SystemTime) -> bool {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return false;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let Ok(ft) = entry.file_type() else {
            continue;
        };
        if ft.is_dir() {
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if matches!(
                name,
                "target" | "vendor" | ".git" | ".aethercompute" | "bindings" | "node_modules"
            ) {
                continue;
            }
            if newer_file_in(&path, threshold) {
                return true;
            }
        } else if ft.is_file() {
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            let interesting =
                name.ends_with(".rs") || name.ends_with(".toml") || name.ends_with(".lock");
            if interesting {
                if let Ok(m) = std::fs::metadata(&path) {
                    if matches!(m.modified(), Ok(t) if t > *threshold) {
                        return true;
                    }
                }
            }
        }
    }
    false
}

/// Replace this process with the training client.
pub fn exec_client(launch: &config::LaunchConfig) -> Result<()> {
    let bin = config::client_bin();
    if !bin.exists() {
        anyhow::bail!(
            "client binary not found at {}. The build screen should have produced it.",
            bin.display()
        );
    }

    let env = Env::detect();
    let mut cmd = Command::new(&bin);
    cmd.args(launch.client_args())
        .current_dir(config::repo_root());
    env.apply(&mut cmd);

    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        let err = cmd.exec();
        // exec only returns on failure.
        Err(anyhow::Error::from(err).context("exec training client"))
    }
    #[cfg(not(unix))]
    {
        let status = cmd.status().context("run training client")?;
        if !status.success() {
            anyhow::bail!("training client exited with {status}");
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn set_mtime(path: &std::path::Path, when: std::time::SystemTime) {
        let f = std::fs::OpenOptions::new()
            .write(true)
            .open(path)
            .unwrap_or_else(|_| std::fs::File::create(path).unwrap());
        f.set_modified(when).unwrap();
    }

    #[test]
    fn newer_file_in_detects_source_changes_and_skips_artifacts() {
        let root = std::env::temp_dir().join(format!(
            "aether-prepare-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("src")).unwrap();
        std::fs::create_dir_all(root.join("target")).unwrap();
        std::fs::create_dir_all(root.join("vendor")).unwrap();

        let now = std::time::SystemTime::now();
        let old = now - Duration::from_secs(3600);
        let newer = now + Duration::from_secs(60);

        // Old source file — below the threshold.
        let old_rs = root.join("src").join("lib.rs");
        std::fs::write(&old_rs, b"// old").unwrap();
        set_mtime(&old_rs, old);
        // A newer non-source file (.md) — must be ignored.
        std::fs::write(root.join("README.md"), b"# hi").unwrap();
        set_mtime(&root.join("README.md"), newer);
        // A newer .rs inside target/ — must be skipped (build artifact).
        let tgt = root.join("target").join("out.rs");
        std::fs::write(&tgt, b"// built").unwrap();
        set_mtime(&tgt, newer);

        assert!(
            !newer_file_in(&root, &now),
            "old source + skipped artifacts should not trigger a rebuild"
        );

        // Add a genuinely-newer source file → must trigger.
        let new_rs = root.join("src").join("new.rs");
        std::fs::write(&new_rs, b"// new").unwrap();
        set_mtime(&new_rs, newer);
        assert!(
            newer_file_in(&root, &now),
            "a newer .rs under src should trigger a rebuild"
        );

        let _ = std::fs::remove_dir_all(&root);
    }
}
