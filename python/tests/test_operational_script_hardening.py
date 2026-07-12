import importlib.util
import sys
from pathlib import Path

import pytest


def load_dashboard():
    script_path = (
        Path(__file__).parents[2] / "scripts" / "training-control-dashboard.py"
    )
    spec = importlib.util.spec_from_file_location(
        "training_control_dashboard_under_test", script_path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_remote_dashboard_bind_requires_explicit_password():
    dashboard = load_dashboard()

    dashboard.validate_control_settings("127.0.0.1", False)
    dashboard.validate_control_settings("::1", False)
    with pytest.raises(RuntimeError, match="CONTROL_PASSWORD"):
        dashboard.validate_control_settings("0.0.0.0", False)
    dashboard.validate_control_settings("0.0.0.0", True)


def test_dashboard_commands_only_allow_known_repo_scripts():
    dashboard = load_dashboard()

    assert dashboard.known_repo_script(
        "scripts/prepare-sft-local.py",
        dashboard.DATASET_SCRIPTS[0],
        dashboard.DATASET_SCRIPTS,
    ) == "scripts/prepare-sft-local.py"

    with pytest.raises(RuntimeError, match="not allowed"):
        dashboard.known_repo_script(
            "/tmp/untrusted.py",
            dashboard.DATASET_SCRIPTS[0],
            dashboard.DATASET_SCRIPTS,
        )


def test_dashboard_action_forms_include_csrf_token():
    dashboard = load_dashboard()
    config = {
        "server": {"experiment_enabled": False},
        "dataset": {},
        "model": {"enabled": False},
    }

    actions = dashboard.render_actions(config)

    assert actions.count('name="_csrf"') == actions.count('<form method="post"')
    assert f'value="{dashboard.CSRF_TOKEN}"' in actions


def test_dashboard_rejects_inconsistent_sft_manifest(tmp_path, monkeypatch):
    dashboard = load_dashboard()
    output = tmp_path / "data"
    output.mkdir()
    (output / "train-00000.parquet").touch()
    (output / "subset_metadata.json").write_text(
        '{"num_sequences": 2, "sequence_length": 8, '
        '"files": ["train-00000.parquet"], "file_rows": {"train-00000.parquet": 1}}'
    )
    monkeypatch.setattr(dashboard, "repo_root", lambda: tmp_path)
    config = {
        "dataset": {
            "output_dir": "data",
            "num_sequences": 2,
            "sequence_length": 8,
            "script": "scripts/prepare-sft-local.py",
        }
    }

    ready, message = dashboard.dataset_status(config)

    assert not ready
    assert "row counts" in message
