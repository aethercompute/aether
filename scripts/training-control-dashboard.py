#!/usr/bin/env python3
import html
import ipaddress
import json
import os
import base64
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
import tomllib
import secrets
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, quote, urlparse
from urllib.request import Request, urlopen


CONFIG_PATH = Path(os.environ.get("TRAINING_RUN_CONFIG", "config/training-run.toml"))
CONTROL_HOST = os.environ.get("CONTROL_HOST", "127.0.0.1")
CONTROL_PORT = int(os.environ.get("CONTROL_PORT", "8080"))
CONTROL_USERNAME = os.environ.get("CONTROL_USERNAME", "admin")
_CONFIGURED_CONTROL_PASSWORD = os.environ.get("CONTROL_PASSWORD", "")
CONTROL_PASSWORD = _CONFIGURED_CONTROL_PASSWORD or secrets.token_urlsafe(24)
GENERATED_CONTROL_PASSWORD = not bool(_CONFIGURED_CONTROL_PASSWORD)
CSRF_TOKEN = secrets.token_urlsafe(32)
MAX_LOG_LINES = 500

DATASET_SCRIPTS = (
    Path("scripts/prepare-ultra-fineweb-local.py"),
    Path("scripts/prepare-sft-local.py"),
)
MODEL_SCRIPTS = (Path("scripts/push-new-model-hf.py"),)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def is_loopback_host(host: str) -> bool:
    normalized = host.strip().lower()
    if normalized == "localhost":
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def validate_control_settings(host: str, password_provided: bool) -> None:
    if not is_loopback_host(host) and not password_provided:
        raise RuntimeError(
            "CONTROL_PASSWORD must be explicitly set when CONTROL_HOST is not loopback"
        )


def known_repo_script(configured: str, default: Path, allowed: tuple[Path, ...]) -> str:
    raw_path = Path(str(configured).strip() or default)
    candidate = raw_path if raw_path.is_absolute() else repo_root() / raw_path
    resolved = candidate.resolve()
    for script in allowed:
        if resolved == (repo_root() / script).resolve():
            return str(script)
    choices = ", ".join(str(script) for script in allowed)
    raise RuntimeError(f"script is not allowed: {configured!r}; choose one of: {choices}")


def load_config() -> dict:
    with (repo_root() / CONFIG_PATH).open("rb") as f:
        return tomllib.load(f)


def bool_value(value: str) -> bool:
    return value.lower() in {"1", "true", "yes", "on"}


def int_value(value: str) -> int:
    return int(value.strip())


def format_sources_text(sources: list) -> str:
    """Render a list of source dicts as an editable textarea.

    One source per line: ``dataset|split=...|subset=...|text_field=...|weight=...``.
    Only non-default keys are emitted to keep the textarea compact. Empty for
    no sources (falls back to singular dataset fields).
    """
    lines = []
    for src in sources or []:
        if not isinstance(src, dict):
            continue
        dataset = src.get("dataset") or src.get("name") or ""
        parts = [dataset] if dataset else []
        if src.get("split"):
            parts.append(f"split={src['split']}")
        if src.get("subset"):
            parts.append(f"subset={src['subset']}")
        tf = src.get("text_field") or src.get("text-field")
        if tf and tf != "text":
            parts.append(f"text_field={tf}")
        w = src.get("weight")
        if w is not None and float(w) != 1.0:
            parts.append(f"weight={w}")
        lines.append("|".join(parts))
    return "\n".join(lines)


def parse_sources_text(text: str) -> list[dict]:
    """Inverse of `format_sources_text`."""
    sources = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        bits = [b.strip() for b in line.split("|")]
        if not bits or not bits[0]:
            continue
        src = {"dataset": bits[0]}
        for bit in bits[1:]:
            if "=" not in bit:
                continue
            key, value = bit.split("=", 1)
            key = key.strip().lower().replace("-", "_")
            value = value.strip()
            if key == "weight":
                try:
                    src["weight"] = float(value)
                except ValueError:
                    continue
            elif key in {"split", "subset", "text_field"}:
                src[key] = value
        sources.append(src)
    return sources


def _toml_inline_value(value) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    escaped = str(value).replace('\\', '\\\\').replace('"', '\\"')
    return f'"{escaped}"'


def write_config(config: dict) -> None:
    lines: list[str] = []
    for section in ("server", "dataset", "model"):
        lines.append(f"[{section}]")
        items = list(config.get(section, {}).items())
        # Emit scalars/arrays FIRST. Arrays-of-tables ([[section.key]]) must
        # come last within a section, otherwise any bare key=value written
        # after them is reattached to the last table by TOML's grammar.
        scalars = []
        table_arrays = []
        for key, value in items:
            if isinstance(value, list) and value and all(isinstance(v, dict) for v in value):
                table_arrays.append((key, value))
            else:
                scalars.append((key, value))
        for key, value in scalars:
            if isinstance(value, list):
                inline = ", ".join(_toml_inline_value(v) for v in value)
                lines.append(f"{key} = [{inline}]")
            else:
                lines.append(f"{key} = {_toml_inline_value(value)}")
        for key, value in table_arrays:
            for entry in value:
                lines.append("")
                lines.append(f"[[{section}.{key}]]")
                for sub_k, sub_v in entry.items():
                    if sub_v is None:
                        continue
                    lines.append(f"{sub_k} = {_toml_inline_value(sub_v)}")
        lines.append("")
    (repo_root() / CONFIG_PATH).write_text("\n".join(lines), encoding="utf-8")




def update_config_from_form(form: dict[str, list[str]]) -> None:
    config = load_config()
    schema = {
        "server": {
            "state_path": str,
            "data_config": str,
            "experiment_path": str,
            "experiment_enabled": bool_value,
            "server_port": int_value,
            "live_web_port": int_value,
            "tui": bool_value,
        },
        "dataset": {
            "enabled": bool_value,
            "objective": str,
            "script": str,
            "output_dir": str,
            "sequence_length": int_value,
            "num_sequences": int_value,
            "shard_size": int_value,
            "token_bytes": int_value,
            "dataset": str,
            "split": str,
            "subset": str,
            "text_field": str,
            "prompt_field": str,
            "response_field": str,
            "sft_mode": str,
            "messages_field": str,
            "system_prompt": str,
            "tokenizer": str,
            "seed": int_value,
            "buffer_docs": int_value,
            "trust_remote_code": bool_value,
        },
        "model": {
            "enabled": bool_value,
            "script": str,
            "config": str,
            "repo": str,
            "tokenizer": str,
            "private": bool_value,
            "dtype": str,
            "device": str,
        },
    }
    for section, keys in schema.items():
        config.setdefault(section, {})
        for key, converter in keys.items():
            field_name = f"{section}.{key}"
            if converter is bool_value:
                config[section][key] = field_name in form
            elif field_name in form:
                config[section][key] = converter(form[field_name][0])

    # Multi-dataset sources textarea (one source per line, see
    # `format_sources_text`). Empty textarea clears the sources list so the
    # singular dataset/split/subset/text_field fields take effect again.
    sources_text = form.get("dataset.sources", [""])[0] if "dataset.sources" in form else ""
    if sources_text.strip():
        config["dataset"]["sources"] = parse_sources_text(sources_text)
    elif "sources" in config.get("dataset", {}):
        del config["dataset"]["sources"]

    write_config(config)


def shell_join(args: list[str]) -> str:
    return " ".join(shlex.quote(arg) for arg in args)


def is_sft_dataset(dataset: dict) -> bool:
    objective = str(dataset.get("objective", "")).strip().lower()
    if objective == "sft":
        return True
    script = Path(str(dataset.get("script", ""))).name
    return script == "prepare-sft-local.py"


def marker_dir() -> Path:
    path = repo_root() / ".aether-control"
    path.mkdir(exist_ok=True)
    return path


def model_marker_path() -> Path:
    return marker_dir() / "model-push.json"


def read_model_markers() -> dict[str, float]:
    marker = model_marker_path()
    if not marker.exists():
        return {}
    try:
        data = json.loads(marker.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    if isinstance(data.get("repos"), dict):
        return {str(repo): float(timestamp) for repo, timestamp in data["repos"].items()}
    if data.get("repo"):
        return {str(data["repo"]): float(data.get("timestamp", 0))}
    return {}


def mark_model_repos(repos: list[str]) -> None:
    markers = read_model_markers()
    now = time.time()
    for repo in repos:
        markers[repo] = now
    model_marker_path().write_text(json.dumps({"repos": markers}, indent=2) + "\n", encoding="utf-8")


def dataset_status(config: dict) -> tuple[bool, str]:
    dataset = config.get("dataset", {})
    output_dir = repo_root() / dataset.get("output_dir", "")
    metadata_path = output_dir / "subset_metadata.json"
    if not output_dir.exists():
        return False, f"missing {output_dir}"
    if not metadata_path.exists():
        return False, f"missing {metadata_path}"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as err:
        return False, f"invalid metadata: {err}"
    expected_sequences = int(dataset.get("num_sequences", 0))
    actual_sequences = int(metadata.get("num_sequences", 0))
    if actual_sequences < expected_sequences:
        return False, f"has {actual_sequences:,}/{expected_sequences:,} sequences"
    expected_seq_len = int(dataset.get("sequence_length", 0))
    if int(metadata.get("sequence_length", 0)) != expected_seq_len:
        return False, "sequence length does not match config"
    if is_sft_dataset(dataset):
        shard_count = len(list(output_dir.rglob("*.parquet")))
        if shard_count == 0:
            return False, "metadata exists but no .parquet shards were found"
        return True, f"ready: {actual_sequences:,} SFT examples across {shard_count:,} shards"
    expected_token_bytes = int(dataset.get("token_bytes", 0))
    if int(metadata.get("token_bytes", 0)) != expected_token_bytes:
        return False, "token byte width does not match config"
    shard_count = len(list(output_dir.glob("*.bin")))
    if shard_count == 0:
        return False, "metadata exists but no .bin shards were found"
    sources = metadata.get("sources") or []
    mix_note = ""
    if isinstance(sources, list) and len(sources) > 1:
        mix_note = f" (mix of {len(sources)} datasets)"
    return True, f"ready: {actual_sequences:,} sequences across {shard_count:,} shards{mix_note}"


def model_status(config: dict) -> tuple[bool, str]:
    if not model_push_enabled(config):
        reason = model_push_disabled_reason(config)
        return True, reason or "disabled"
    model = config.get("model", {})
    markers = read_model_markers()
    if not markers:
        return False, "no successful push recorded by this dashboard"
    repo = model.get("repo", "")
    if repo not in markers:
        return False, f"no successful push recorded for {repo}"
    return True, f"last pushed {time.ctime(markers[repo])}"


def experiment_state_paths(config: dict) -> list[Path]:
    experiment_path = repo_root() / config.get("server", {}).get(
        "experiment_path", "config/experiment-run.toml"
    )
    with experiment_path.open("rb") as f:
        experiment = tomllib.load(f)
    base_dir = experiment_path.parent
    paths = []
    for run in experiment.get("runs", []):
        state = Path(run["state"])
        paths.append(state if state.is_absolute() else base_dir / state)
    return paths


def configured_state_path(config: dict) -> Path:
    state_path = Path(config.get("server", {}).get("state_path", ""))
    return state_path if state_path.is_absolute() else repo_root() / state_path


def state_uses_lora(state_path: Path) -> bool:
    try:
        with state_path.open("rb") as f:
            state = tomllib.load(f)
        training_method = state.get("model", {}).get("LLM", {}).get("training_method", {})
        return isinstance(training_method, dict) and "Lora" in training_method
    except (OSError, tomllib.TOMLDecodeError, AttributeError):
        return False


def state_hub_checkpoint_repo(state_path: Path) -> str | None:
    try:
        with state_path.open("rb") as f:
            state = tomllib.load(f)
        hub = (
            state.get("model", {})
            .get("LLM", {})
            .get("checkpoint", {})
            .get("Hub", {})
        )
        repo = str(hub.get("repo_id", "")).strip()
        if repo:
            return repo
    except (OSError, tomllib.TOMLDecodeError, AttributeError):
        pass
    return None


def state_checkpoint_repo(state_path: Path) -> str:
    repo = state_hub_checkpoint_repo(state_path)
    if repo:
        return repo
    text = state_path.read_text(encoding="utf-8")
    for line in text.splitlines():
        if line.strip().startswith("repo_id"):
            return line.split("=", 1)[1].strip().strip('"')
    raise RuntimeError(f"checkpoint repo not found in {state_path}")


def experiment_model_repos(config: dict) -> list[str]:
    repos: list[str] = []
    for state_path in experiment_state_paths(config):
        repo = state_checkpoint_repo(state_path)
        if repo not in repos:
            repos.append(repo)
    return repos


def experiment_model_configs(config: dict) -> list[str]:
    configs: list[dict] = []
    
    for state_path in experiment_state_paths(config):
        # NOTE: assumes state.toml and model-config.json are in same dir 
        model_config_path = state_path.with_name("model-config.json")
        repo = state_checkpoint_repo(state_path)
    
        model = dict(config["model"])
        model["repo"] = repo
        model["config"] = str(model_config_path)

        next_config = dict(config)
        next_config["model"] = model
        configs.append(next_config)

    return configs


def experiment_model_status(config: dict) -> tuple[bool, str]:
    if not model_push_enabled(config):
        return True, "disabled"
    repos = experiment_model_repos(config)
    markers = read_model_markers()
    missing = [repo for repo in repos if repo not in markers]
    if missing:
        return False, f"missing init model pushes for: {', '.join(missing)}"
    return True, f"ready: {len(repos)} experiment repos pushed"


def state_checkpoint(config: dict) -> str:
    state_path = repo_root() / config.get("server", {}).get("state_path", "")
    if not state_path.exists():
        return "state file missing"
    try:
        return state_checkpoint_repo(state_path)
    except RuntimeError:
        return "checkpoint repo not found in state file"


def hf_token() -> str:
    return os.environ.get("HF_TOKEN", "") or os.environ.get("HUGGING_FACE_HUB_TOKEN", "")


def ensure_checkpoint_available(config: dict) -> None:
    state_path = repo_root() / config.get("server", {}).get("state_path", "")
    if not state_path.exists():
        raise RuntimeError(f"state file is missing: {state_path}")
    repo = state_hub_checkpoint_repo(state_path)
    if not repo:
        return
    if (repo_root() / repo).exists():
        return

    url = f"https://huggingface.co/api/models/{quote(repo, safe='/')}"
    token = hf_token()
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    request = Request(url, headers=headers)
    try:
        with urlopen(request, timeout=10) as response:
            if response.status < 400:
                return
    except HTTPError as err:
        body = err.read().decode("utf-8", errors="replace").strip()
        detail = f": {body[:300]}" if body else ""
        if err.code in (401, 403):
            hint = (
                "set HF_TOKEN or HUGGING_FACE_HUB_TOKEN for the dashboard process"
                if not token
                else "check that the HF token can access this repo"
            )
            raise RuntimeError(
                f"checkpoint repo {repo} is not accessible ({err.code} {err.reason}); {hint}{detail}"
            ) from err
        raise RuntimeError(
            f"checkpoint repo {repo} is not accessible ({err.code} {err.reason}){detail}"
        ) from err
    except URLError as err:
        raise RuntimeError(f"could not verify checkpoint repo {repo}: {err.reason}") from err


def state_data_server(config: dict) -> str | None:
    state_path = repo_root() / config.get("server", {}).get("state_path", "")
    if not state_path.exists():
        return None
    text = state_path.read_text(encoding="utf-8")
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("Server") and "=" in stripped:
            return stripped.split("=", 1)[1].strip().strip('"')
    return None


def is_local_host(host: str) -> bool:
    host = host.strip().lower()
    if host in {"localhost", "localhost."}:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def data_server_endpoint_status(config: dict) -> tuple[bool, str]:
    endpoint = state_data_server(config)
    if not endpoint:
        return True, "not configured"
    if ":" not in endpoint:
        return False, f"invalid endpoint {endpoint}"
    host, port_raw = endpoint.rsplit(":", 1)
    try:
        port = int(port_raw)
    except ValueError:
        return False, f"invalid endpoint {endpoint}"
    try:
        with socket.create_connection((host, port), timeout=1.0):
            return True, f"reachable: {endpoint}"
    except OSError as err:
        if is_local_host(host) and data_config_path(config) is not None:
            return True, f"will be hosted by training server on {endpoint}"
        return False, f"not reachable: {endpoint} ({err})"


def model_push_enabled(config: dict) -> bool:
    if state_uses_lora(configured_state_path(config)):
        return False
    model = config.get("model", {})
    if not model.get("enabled", False):
        return False
    required = ("config", "repo", "tokenizer")
    return all(str(model.get(key, "")).strip() for key in required)


def model_push_disabled_reason(config: dict) -> str | None:
    if state_uses_lora(configured_state_path(config)):
        return "not required for LoRA runs; using existing base checkpoint"
    model = config.get("model", {})
    if not model.get("enabled", False):
        return "model.enabled is false"
    missing = [
        key
        for key in ("config", "repo", "tokenizer")
        if not str(model.get(key, "")).strip()
    ]
    if missing:
        return f"missing model fields: {', '.join(missing)}"
    return None


def config_for_form(config: dict) -> dict:
    display = {
        section: dict(values) if isinstance(values, dict) else values
        for section, values in config.items()
    }
    server = display.get("server", {})
    dataset = display.get("dataset", {})
    model = display.setdefault("model", {})
    if isinstance(model, dict):
        model.setdefault("enabled", False)
        model.setdefault("script", "scripts/push-new-model-hf.py")
        state_raw = str(server.get("state_path", "")) if isinstance(server, dict) else ""
        if state_raw:
            model.setdefault("config", str(Path(state_raw).with_name("model-config.json")))
            state_path = Path(state_raw)
            if not state_path.is_absolute():
                state_path = repo_root() / state_path
            try:
                model.setdefault("repo", state_checkpoint_repo(state_path))
            except RuntimeError:
                model.setdefault("repo", "")
        else:
            model.setdefault("config", "")
            model.setdefault("repo", "")
        if isinstance(dataset, dict):
            model.setdefault("tokenizer", str(dataset.get("tokenizer", "")))
        else:
            model.setdefault("tokenizer", "")
        model.setdefault("private", False)
        model.setdefault("dtype", "bfloat16")
        model.setdefault("device", "")
    return display


@dataclass
class Job:
    name: str
    process: subprocess.Popen | None = None
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    returncode: int | None = None
    command: list[str] = field(default_factory=list)
    log: list[str] = field(default_factory=list)

    @property
    def running(self) -> bool:
        return self.process is not None and self.process.poll() is None


class ControlState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.job: Job | None = None
        self.server: Job | None = None

    def append_log(self, job: Job, line: str) -> None:
        with self.lock:
            job.log.append(line.rstrip())
            del job.log[:-MAX_LOG_LINES]


STATE = ControlState()


def run_background(name: str, command: list[str], on_success=None, long_running: bool = False) -> Job:
    with STATE.lock:
        active = STATE.server if long_running else STATE.job
        if active and active.running:
            raise RuntimeError(f"{active.name} is already running")
        job = Job(name=name, command=command)
        target_attr = "server" if long_running else "job"
        setattr(STATE, target_attr, job)

    def worker() -> None:
        try:
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            process = subprocess.Popen(
                command,
                cwd=repo_root(),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            with STATE.lock:
                job.process = process
            assert process.stdout is not None
            for line in process.stdout:
                STATE.append_log(job, line)
            returncode = process.wait()
            with STATE.lock:
                job.returncode = returncode
                job.finished_at = time.time()
            if returncode == 0 and on_success is not None:
                on_success()
        except Exception as exc:
            with STATE.lock:
                job.log.append(f"ERROR: {exc}")
                job.returncode = -1
                job.finished_at = time.time()

    threading.Thread(target=worker, daemon=True).start()
    return job


def _format_source_arg(src: dict) -> str:
    """Render a source dict as a single `--source` argument value."""
    parts = [f"dataset={src['dataset']}"]
    if src.get("split"):
        parts.append(f"split={src['split']}")
    if src.get("subset"):
        parts.append(f"subset={src['subset']}")
    tf = src.get("text_field") or src.get("text-field")
    if tf:
        parts.append(f"text_field={tf}")
    if src.get("weight") is not None:
        parts.append(f"weight={src['weight']}")
    return ",".join(parts)


def prepare_dataset_command(config: dict) -> list[str]:
    dataset = config["dataset"]
    sources = [s for s in dataset.get("sources", []) if isinstance(s, dict) and s.get("dataset")]
    script = known_repo_script(
        dataset.get("script", ""), DATASET_SCRIPTS[0], DATASET_SCRIPTS
    )

    command = [
        sys.executable,
        script,
    ]
    if is_sft_dataset(dataset):
        command.extend(
            [
                "--dataset",
                dataset["dataset"],
                "--split",
                dataset.get("split", "train"),
                "--prompt-field",
                dataset.get("prompt_field", "english"),
                "--response-field",
                dataset.get("response_field", "pirate"),
                "--tokenizer",
                dataset["tokenizer"],
                "--output-dir",
                dataset["output_dir"],
                "--sequence-length",
                str(dataset["sequence_length"]),
                "--shard-size",
                str(dataset["shard_size"]),
                "--seed",
                str(dataset["seed"]),
                "--buffer-docs",
                str(dataset["buffer_docs"]),
                "--mode",
                dataset.get("sft_mode", "chat"),
            ]
        )
        if dataset.get("num_sequences"):
            command.extend(["--num-sequences", str(dataset["num_sequences"])])
        if dataset.get("subset"):
            command.extend(["--subset", dataset["subset"]])
        if dataset.get("system_prompt"):
            command.extend(["--system-prompt", dataset["system_prompt"]])
        if dataset.get("messages_field"):
            command.extend(["--messages-field", dataset["messages_field"]])
        if dataset.get("trust_remote_code", False):
            command.append("--trust-remote-code")
        return command

    if sources:
        # Multi-source mixing: one --source per entry. Each carries its own
        # split / subset / text_field / weight; the singular dataset fields
        # below are ignored by the script when --source is present.
        for src in sources:
            command.extend(["--source", _format_source_arg(src)])
    else:
        # Legacy single-source fallback.
        command.extend(
            [
                "--dataset",
                dataset["dataset"],
                "--split",
                dataset["split"],
                "--text-field",
                dataset["text_field"],
            ]
        )
        if dataset.get("subset"):
            command.extend(["--subset", dataset["subset"]])

    command.extend(
        [
            "--tokenizer",
            dataset["tokenizer"],
            "--output-dir",
            dataset["output_dir"],
            "--sequence-length",
            str(dataset["sequence_length"]),
            "--num-sequences",
            str(dataset["num_sequences"]),
            "--shard-size",
            str(dataset["shard_size"]),
            "--token-bytes",
            str(dataset["token_bytes"]),
            "--seed",
            str(dataset["seed"]),
            "--buffer-docs",
            str(dataset["buffer_docs"]),
        ]
    )
    if dataset.get("trust_remote_code", False):
        command.append("--trust-remote-code")
    return command



def push_model_command(config: dict) -> list[str]:
    if not model_push_enabled(config):
        raise RuntimeError("init model push is disabled for this run")
    model = config["model"]
    script = known_repo_script(model.get("script", ""), MODEL_SCRIPTS[0], MODEL_SCRIPTS)
    command = [
        sys.executable,
        script,
        "--config",
        model["config"],
        "--repo",
        model["repo"],
        "--tokenizer",
        model["tokenizer"],
    ]
    if model.get("private", False):
        command.append("--private")
    if model.get("device"):
        command.extend(["--device", model["device"]])
    dtype = model.get("dtype", "")
    if dtype and dtype != "bfloat16":
        command.extend(["--dtype", dtype])
    return command


def run_background_sequence(name: str, commands: list[list[str]], on_success=None) -> Job:
    if not commands:
        raise RuntimeError("no commands to run")
    with STATE.lock:
        active = STATE.job
        if active and active.running:
            raise RuntimeError(f"{active.name} is already running")
        job = Job(name=name, command=[" && ".join(shell_join(command) for command in commands)])
        STATE.job = job

    def worker() -> None:
        try:
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            for command in commands:
                STATE.append_log(job, f"$ {shell_join(command)}")
                process = subprocess.Popen(
                    command,
                    cwd=repo_root(),
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                with STATE.lock:
                    job.process = process
                assert process.stdout is not None
                for line in process.stdout:
                    STATE.append_log(job, line)
                returncode = process.wait()
                if returncode != 0:
                    with STATE.lock:
                        job.returncode = returncode
                        job.finished_at = time.time()
                        job.process = None
                    return
            with STATE.lock:
                job.returncode = 0
                job.finished_at = time.time()
                job.process = None
            if on_success is not None:
                on_success()
        except Exception as exc:
            with STATE.lock:
                job.log.append(f"ERROR: {exc}")
                job.returncode = -1
                job.finished_at = time.time()
                job.process = None

    threading.Thread(target=worker, daemon=True).start()
    return job


def validate_command(config: dict) -> list[str]:
    command = [
        "aether-centralized-server",
        "validate-config",
        "--state",
        config["server"]["state_path"],
    ]
    data_config = data_config_path(config)
    if data_config is not None:
        command.extend(["--data-config", str(data_config)])
    return command


def data_config_path(config: dict) -> Path | None:
    server = config.get("server", {})
    configured = str(server.get("data_config", "")).strip()
    if configured:
        return Path(configured)

    # Compatibility for older configs: data.toml next to state.toml means the
    # coordinator hosts a training-data TCP server.
    sibling = Path(server.get("state_path", "")).with_name("data.toml")
    return sibling if (repo_root() / sibling).exists() else None


def repo_relative(path: Path) -> Path:
    return path if path.is_absolute() else repo_root() / path


def ensure_data_server_config(config: dict) -> None:
    endpoint = state_data_server(config)
    if not endpoint:
        return
    data_config = data_config_path(config)
    if data_config is None:
        raise RuntimeError(
            f"state advertises training data server {endpoint}, but server.data_config is not set"
        )
    if not repo_relative(data_config).exists():
        raise RuntimeError(
            f"state advertises training data server {endpoint}, but data config is missing: {data_config}"
        )


def server_command(config: dict) -> list[str]:
    server = config["server"]
    command = [
        "aether-centralized-server",
        "run",
        "--state",
        server["state_path"],
        "--server-port",
        str(server["server_port"]),
        "--web-port",
        str(server["live_web_port"]),
    ]
    data_config = data_config_path(config)
    if data_config is not None:
        command.extend(["--data-config", str(data_config)])
    data_server_addr = os.environ.get("DATA_SERVER_ADDR", "").strip()
    if data_server_addr:
        command.extend(["--data-server-addr", data_server_addr])
    if not server.get("tui", False):
        command.extend(["--tui=false"])
    return command


def experiment_server_command(config: dict) -> list[str]:
    server = config["server"]
    experiment_path = server.get("experiment_path", "config/experiment-run.toml")
    command = [
        "aether-centralized-server",
        "run",
        "--experiment",
        experiment_path,
        "--server-port",
        str(server["server_port"]),
        "--web-port",
        str(server["live_web_port"]),
    ]
    data_config = data_config_path(config)
    if data_config is not None:
        command.extend(["--data-config", str(data_config)])
    data_server_addr = os.environ.get("DATA_SERVER_ADDR", "").strip()
    if data_server_addr:
        command.extend(["--data-server-addr", data_server_addr])
    if not server.get("tui", False):
        command.extend(["--tui=false"])
    return command


def ensure_training_prereqs(config: dict) -> None:
    data_ready, data_message = dataset_status(config)
    if config.get("dataset", {}).get("enabled", True) and not data_ready:
        raise RuntimeError(f"dataset is not ready: {data_message}")
    ensure_data_server_config(config)
    ensure_checkpoint_available(config)
    model_ready, model_message = model_status(config)
    if model_push_enabled(config) and not model_ready:
        raise RuntimeError(f"init model is not ready: {model_message}")


def ensure_experiment_training_prereqs(config: dict) -> None:
    data_ready, data_message = dataset_status(config)
    if config.get("dataset", {}).get("enabled", True) and not data_ready:
        raise RuntimeError(f"dataset is not ready: {data_message}")
    ensure_data_server_config(config)
    ensure_checkpoint_available(config)
    model_ready, model_message = experiment_model_status(config)
    if model_push_enabled(config) and not model_ready:
        raise RuntimeError(f"experiment init models are not ready: {model_message}")


def stop_server_job() -> str:
    with STATE.lock:
        server = STATE.server
    if server and server.running and server.process:
        server.process.send_signal(signal.SIGTERM)
        return "Stop signal sent."
    return "Training server is not running."


def render_actions(config: dict) -> str:
    push_enabled = model_push_enabled(config)
    push_disabled_reason = model_push_disabled_reason(config)
    experiment_enabled = config.get("server", {}).get("experiment_enabled", False)
    actions = [
        (
            "/prepare-dataset",
            "Prepare dataset",
            config.get("dataset", {}).get("enabled", True),
            "dataset.enabled is false",
        ),
        ("/push-model", "Push init model", push_enabled, push_disabled_reason),
        (
            "/push-experiment-models",
            "Push experiment init models",
            push_enabled and experiment_enabled,
            push_disabled_reason if not push_enabled else "server.experiment_enabled is false",
        ),
        ("/validate", "Validate state config", True, None),
        ("/start-server", "Start training server", True, None),
        ("/stop-server", "Stop training server", True, None),
        (
            "/start-experiment-server",
            "Start experiment server",
            experiment_enabled,
            "server.experiment_enabled is false",
        ),
        (
            "/stop-experiment-server",
            "Stop experiment server",
            experiment_enabled,
            "server.experiment_enabled is false",
        ),
    ]
    rendered = []
    for path, label, enabled, reason in actions:
        disabled = "" if enabled else " disabled"
        hint = f"<small>{html.escape(reason or 'disabled')}</small>" if not enabled else ""
        rendered.append(
            f'<form method="post" action="{path}">'
            f'<input type="hidden" name="_csrf" value="{CSRF_TOKEN}">'
            f'<button type="submit"{disabled}>{label}</button>{hint}</form>'
        )
    return "".join(rendered)


TAB_SCRIPT = """
<script>
const t = document.querySelectorAll('.tabs button.tab');
const p = document.querySelectorAll('[data-panel]');
t.forEach(b => b.addEventListener('click', () => {
  t.forEach(x => x.classList.remove('active'));
  b.classList.add('active');
  p.forEach(s => { s.hidden = s.dataset.panel !== b.dataset.tab; });
}));
</script>
"""


def html_page(message: str | None = None) -> str:
    config = load_config()
    data_ready, data_message = dataset_status(config)
    data_server_ready, data_server_message = data_server_endpoint_status(config)
    model_ready, model_message = model_status(config)
    if config.get("server", {}).get("experiment_enabled", False) and model_push_enabled(config):
        try:
            experiment_model_ready, experiment_model_message = experiment_model_status(config)
        except Exception as err:
            experiment_model_ready, experiment_model_message = False, str(err)
    else:
        experiment_model_ready, experiment_model_message = True, "disabled"
    checkpoint = state_checkpoint(config)
    with STATE.lock:
        job = STATE.job
        server = STATE.server
    if server and server.running:
        server_short, server_cls = "running", "ok"
    elif server is None:
        server_short, server_cls = "idle", "warn"
    elif server.returncode == 0:
        server_short, server_cls = "stopped", "ok"
    else:
        server_short, server_cls = "stopped", "bad"
    data_short = "ready" if data_ready else "pending"
    data_cls = "ok" if data_ready else "bad"
    data_server_cls = "ok" if data_server_ready else "bad"
    model_short = "ready" if model_ready else "pending"
    model_cls = "ok" if model_ready else "warn"
    actions = render_actions(config)
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Aether Training Control</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 0; background: #0d1117; color: #c9d1d9; font-size: 13px; line-height: 1.45; }}
    .wrap {{ max-width: 1100px; margin: 0 auto; padding: 0 1rem 2rem; }}
    .topbar {{ position: sticky; top: 0; z-index: 10; background: #0d1117; border-bottom: 1px solid #30363d; }}
    .topbar .wrap {{ display: flex; align-items: baseline; justify-content: space-between; gap: 1rem; flex-wrap: wrap; padding-top: .6rem; padding-bottom: .6rem; }}
    .brand {{ font-size: 15px; font-weight: 700; color: #f0f6fc; }}
    .statusline {{ font-size: 12px; color: #8b949e; }}
    .msg {{ padding: .5rem .75rem; margin: 1rem 0 0; border: 1px solid #30363d; }}
    .tabs {{ display: flex; border-bottom: 1px solid #30363d; margin-top: 1rem; }}
    .tabs button.tab {{ all: unset; cursor: pointer; padding: .45rem .8rem; color: #8b949e; border-bottom: 2px solid transparent; font: inherit; }}
    .tabs button.tab:hover {{ color: #e6edf3; background: transparent; }}
    .tabs button.tab.active {{ color: #f0f6fc; border-bottom-color: #58a6ff; background: transparent; }}
    [data-panel] {{ margin-top: 1rem; }}
    input[type="text"], input:not([type]) {{ width: 100%; box-sizing: border-box; padding: .3rem; background: #161b22; color: #e6edf3; border: 1px solid #30363d; font: inherit; }}
    input[type="checkbox"] {{ accent-color: #58a6ff; }}
    label {{ display: block; font-weight: 600; margin-top: .55rem; font-size: 12px; }}
    fieldset {{ border: 1px solid #30363d; margin: 1rem 0; padding: .75rem; }}
    legend {{ color: #8b949e; padding: 0 .35rem; }}
    button {{ padding: .4rem .7rem; background: #21262d; color: #e6edf3; border: 1px solid #30363d; cursor: pointer; font: inherit; }}
    button:hover {{ background: #2d333b; }}
    button.primary {{ background: #1f6feb; border-color: #1f6feb; color: #fff; }}
    .ok {{ color: #3fb950; font-weight: 700; }}
    .warn {{ color: #d29922; font-weight: 700; }}
    .bad {{ color: #f85149; font-weight: 700; }}
    pre {{ background: #161b22; color: #e6edf3; padding: .75rem; overflow: auto; max-height: 22rem; border: 1px solid #30363d; font-size: 12px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: .5rem .75rem; }}
    .actions {{ display: flex; flex-wrap: wrap; gap: .5rem; }}
    .actions form {{ margin: 0; }}
    h3 {{ margin: 1rem 0 .4rem; font-size: 13px; color: #c9d1d9; }}
    code {{ color: #f0f6fc; font-size: 12px; }}
    a {{ color: #58a6ff; }}
    p {{ margin: .3rem 0; }}
  </style>
</head>
<body>
  <header class="topbar"><div class="wrap">
    <span class="brand">Aether Training Control</span>
    <span class="statusline">Dataset: <span class="{data_cls}">{data_short}</span> &middot; Init model: <span class="{model_cls}">{model_short}</span> &middot; Server: <span class="{server_cls}">{server_short}</span></span>
  </div></header>
  <main class="wrap">
    {f'<div class="msg warn">{html.escape(message)}</div>' if message else ''}
    <nav class="tabs">
      <button type="button" class="tab active" data-tab="status">Status</button>
      <button type="button" class="tab" data-tab="config">Config</button>
      <button type="button" class="tab" data-tab="actions">Actions</button>
      <button type="button" class="tab" data-tab="logs">Logs</button>
    </nav>
    <section data-panel="status">
      <p>Dataset: <span class="{'ok' if data_ready else 'bad'}">{html.escape(data_message)}</span></p>
      <p>Data server endpoint: <span class="{data_server_cls}">{html.escape(data_server_message)}</span></p>
      <p>Init model: <span class="{'ok' if model_ready else 'warn'}">{html.escape(model_message)}</span></p>
      <p>Experiment init models: <span class="{'ok' if experiment_model_ready else 'warn'}">{html.escape(experiment_model_message)}</span></p>
      <p>State checkpoint: <code>{html.escape(checkpoint)}</code></p>
      <p>Training server: {render_job_status(server, live=True)}</p>
      <p>Live dashboard: <a href="http://{html.escape(os.environ.get('PUBLIC_HOST', 'localhost'))}:{config['server']['live_web_port']}/">port {config['server']['live_web_port']}</a></p>
    </section>
    <section data-panel="config" hidden>
      <form method="post" action="/save">
        <input type="hidden" name="_csrf" value="{CSRF_TOKEN}">
        {render_config_form(config)}
        <button class="primary" type="submit">Save configuration</button>
      </form>
    </section>
    <section data-panel="actions" hidden>
      <div class="actions">
        {actions}
      </div>
    </section>
    <section data-panel="logs" hidden>
      <h3>Last Job</h3>
      {render_job(job)}
      <h3>Server Log</h3>
      {render_job(server)}
    </section>
  </main>
  {TAB_SCRIPT}
</body>
</html>"""


def render_job_status(job: Job | None, live: bool = False) -> str:
    if job is None:
        return "not started"
    if job.running:
        return f'<span class="ok">running</span> <code>{html.escape(shell_join(job.command))}</code>'
    css = "ok" if job.returncode == 0 else "bad"
    noun = "stopped" if live else "finished"
    return f'<span class="{css}">{noun} ({job.returncode})</span>'


def render_job(job: Job | None) -> str:
    if job is None:
        return "<p>No job has run yet.</p>"
    lines = "\n".join(html.escape(line) for line in job.log)
    return f"<p>{render_job_status(job)}</p><p><code>{html.escape(shell_join(job.command))}</code></p><pre>{lines}</pre>"


def render_config_form(config: dict) -> str:
    config = config_for_form(config)
    sections = []
    for section, values in config.items():
        fields = []
        for key, value in values.items():
            name = f"{section}.{key}"
            label = html.escape(name)
            # Array of source tables -> multi-line textarea. Only the
            # `dataset.sources` shape is supported; other list/dict values
            # fall through to the generic string rendering below.
            if (
                key == "sources"
                and isinstance(value, list)
                and all(isinstance(v, dict) for v in value)
            ):
                text = html.escape(format_sources_text(value))
                fields.append(
                    f'<label>{label}<textarea name="{label}" rows="6" '
                    f'placeholder="dataset|split=...|subset=...|text_field=...|weight=...">'
                    f"{text}</textarea></label>"
                    "<small>One dataset per line. Format: "
                    "<code>dataset|split=train|subset=&lt;config&gt;|text_field=content|weight=0.6</code>. "
                    "Leave empty to use the singular fields above.</small>"
                )
                continue
            if isinstance(value, bool):
                checked = " checked" if value else ""
                fields.append(f'<label><input style="width:auto" type="checkbox" name="{label}"{checked}> {label}</label>')
            elif isinstance(value, list):
                inline = html.escape(json.dumps(value))
                fields.append(f'<label>{label}<input name="{label}" value="{inline}"></label>')
            else:
                fields.append(f'<label>{label}<input name="{label}" value="{html.escape(str(value))}"></label>')
        sections.append(f"<fieldset><legend>{html.escape(section)}</legend><div class=\"grid\">{''.join(fields)}</div></fieldset>")
    return "".join(sections)



class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path == "/health":
            self.send_response(HTTPStatus.OK)
            self.send_header("content-type", "text/plain")
            self.end_headers()
            self.wfile.write(b"ok\n")
            return
        if not self.authorized():
            self.request_auth()
            return
        self.respond(html_page())

    def do_POST(self) -> None:
        if not self.authorized():
            self.request_auth()
            return
        length = int(self.headers.get("content-length", "0"))
        body = self.rfile.read(length).decode("utf-8")
        form = parse_qs(body)
        csrf_token = form.get("_csrf", [""])[0]
        if not secrets.compare_digest(csrf_token, CSRF_TOKEN):
            self.send_error(HTTPStatus.FORBIDDEN, "invalid CSRF token")
            return
        path = urlparse(self.path).path
        message = None
        try:
            config = load_config()
            if path == "/save":
                update_config_from_form(form)
                message = "Configuration saved."
            elif path == "/prepare-dataset":
                command = prepare_dataset_command(config)
                run_background("prepare dataset", command)
                message = "Dataset preparation started."
            elif path == "/push-model":
                if not model_push_enabled(config):
                    raise RuntimeError("init model push is disabled for this run")
                command = push_model_command(config)

                def mark_model() -> None:
                    mark_model_repos([config["model"]["repo"]])

                run_background("push init model", command, on_success=mark_model)
                message = "Init model push started."
            elif path == "/push-experiment-models":
                if not config.get("server", {}).get("experiment_enabled", False):
                    raise RuntimeError("experiment mode is disabled for this run")
                if not model_push_enabled(config):
                    raise RuntimeError("experiment init model push is disabled for this run")
                configs = experiment_model_configs(config)
                commands = [push_model_command(exp_config) for exp_config in configs]

                repos = [exp_config["model"]["repo"] for exp_config in configs]
                def mark_experiment_models() -> None:
                    mark_model_repos(repos)

                run_background_sequence(
                    "push experiment init models",
                    commands,
                    on_success=mark_experiment_models,
                )
                message = f"Experiment init model pushes started for {len(repos)} repos."
            elif path == "/validate":
                run_background("validate config", validate_command(config))
                message = "Config validation started."
            elif path == "/start-server":
                ensure_training_prereqs(config)
                run_background("training server", server_command(config), long_running=True)
                message = "Training server started."
            elif path == "/stop-server":
                message = stop_server_job()
            elif path == "/start-experiment-server":
                if not config.get("server", {}).get("experiment_enabled", False):
                    raise RuntimeError("experiment mode is disabled for this run")
                ensure_experiment_training_prereqs(config)
                experiment_path = repo_root() / config.get("server", {}).get(
                    "experiment_path", "config/experiment-run.toml"
                )
                if not experiment_path.exists():
                    raise RuntimeError(f"experiment config is missing: {experiment_path}")
                run_background(
                    "experiment server",
                    experiment_server_command(config),
                    long_running=True,
                )
                message = "Experiment server started."
            elif path == "/stop-experiment-server":
                message = stop_server_job()
            else:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
        except Exception as err:
            message = str(err)
        self.respond(html_page(message))

    def respond(self, body: str) -> None:
        data = body.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("content-type", "text/html; charset=utf-8")
        self.send_header("content-length", str(len(data)))
        self.send_header("cache-control", "no-store")
        self.send_header("referrer-policy", "same-origin")
        self.end_headers()
        self.wfile.write(data)

    def authorized(self) -> bool:
        header = self.headers.get("authorization", "")
        if not header.startswith("Basic "):
            return False
        try:
            decoded = base64.b64decode(header.removeprefix("Basic ")).decode("utf-8")
        except Exception:
            return False
        username, _, password = decoded.partition(":")
        return secrets.compare_digest(username, CONTROL_USERNAME) and secrets.compare_digest(
            password, CONTROL_PASSWORD
        )

    def request_auth(self) -> None:
        self.send_response(HTTPStatus.UNAUTHORIZED)
        self.send_header("www-authenticate", 'Basic realm="Aether Training Control"')
        self.send_header("content-type", "text/plain; charset=utf-8")
        self.end_headers()
        self.wfile.write(b"Authentication required\n")

    def log_message(self, format: str, *args) -> None:
        sys.stderr.write(f"{self.address_string()} - {format % args}\n")


def main() -> None:
    os.chdir(repo_root())
    try:
        validate_control_settings(CONTROL_HOST, not GENERATED_CONTROL_PASSWORD)
    except RuntimeError as err:
        raise SystemExit(str(err)) from err
    server = ThreadingHTTPServer((CONTROL_HOST, CONTROL_PORT), Handler)
    print(
        f"training control dashboard listening on {CONTROL_HOST}:{CONTROL_PORT}",
        flush=True,
    )
    if GENERATED_CONTROL_PASSWORD:
        print(
            f"generated control dashboard credentials: {CONTROL_USERNAME}:{CONTROL_PASSWORD}",
            flush=True,
        )
    server.serve_forever()


if __name__ == "__main__":
    main()
