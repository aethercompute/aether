import importlib.util
import sys
from pathlib import Path


def load_script(name):
    script_path = Path(__file__).parents[2] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"{name}_under_test", script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_cobertura(path, filename="scripts/tool.py", hits=(1, 0, 1), line_rate="0.6667"):
    lines = "".join(
        f'<line number="{index}" hits="{hit}" />'
        for index, hit in enumerate(hits, 1)
    )
    path.write_text(
        f'''<?xml version="1.0" ?>
<coverage line-rate="{line_rate}" branch-rate="0.5">
  <packages><package><classes>
    <class filename="{filename}"><lines>{lines}</lines></class>
  </classes></package></packages>
</coverage>
'''
    )


def test_junit_skip_writer_and_summary_report_allowed_skip(tmp_path, capsys, monkeypatch):
    writer = load_script("write_junit_skip")
    summary = load_script("junit_summary")
    report = tmp_path / "skip.xml"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "write_junit_skip.py",
            "--path",
            str(report),
            "--classname",
            "optional",
            "--name",
            "missing_gpu",
            "--message",
            "GPU runner unavailable",
        ],
    )
    assert writer.main() == 0

    monkeypatch.setattr(sys, "argv", ["junit_summary.py", str(report), "--label", "optional"])
    assert summary.main() == 0

    output = capsys.readouterr().out
    assert "optional: 1 tests, 0 failures, 0 errors, 1 skipped" in output
    assert "1x GPU runner unavailable" in output


def test_junit_summary_rejects_required_skips(tmp_path, monkeypatch):
    writer = load_script("write_junit_skip")
    summary = load_script("junit_summary")
    report = tmp_path / "skip.xml"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "write_junit_skip.py",
            "--path",
            str(report),
            "--classname",
            "required",
            "--name",
            "skipped_test",
            "--message",
            "not allowed",
        ],
    )
    assert writer.main() == 0

    monkeypatch.setattr(
        sys,
        "argv",
        ["junit_summary.py", str(report), "--label", "required", "--forbid-skips"],
    )
    assert summary.main() == 1


def test_coverage_baseline_accepts_tolerance(tmp_path):
    coverage_ci = load_script("coverage_ci")
    report = tmp_path / "coverage.xml"
    write_cobertura(report, line_rate="0.59")

    args = coverage_ci.build_parser().parse_args(
        [
            "check-baseline",
            "--coverage",
            str(report),
            "--workspace",
            str(tmp_path),
            "--label",
            "Rust",
            "--min-line-rate",
            "60",
            "--tolerance",
            "1",
        ]
    )

    assert args.func(args) == 0


def test_coverage_baseline_rejects_regression_beyond_tolerance(tmp_path):
    coverage_ci = load_script("coverage_ci")
    report = tmp_path / "coverage.xml"
    write_cobertura(report, line_rate="0.58")

    args = coverage_ci.build_parser().parse_args(
        [
            "check-baseline",
            "--coverage",
            str(report),
            "--workspace",
            str(tmp_path),
            "--label",
            "Python",
            "--min-line-rate",
            "60",
            "--tolerance",
            "1",
        ]
    )

    assert args.func(args) == 1


def test_changed_line_coverage_counts_only_changed_source_lines():
    coverage_ci = load_script("coverage_ci")
    diff = """diff --git a/scripts/tool.py b/scripts/tool.py
--- a/scripts/tool.py
+++ b/scripts/tool.py
@@ -1,0 +1,3 @@
+covered()
+
+uncovered()
diff --git a/python/tests/test_tool.py b/python/tests/test_tool.py
--- a/python/tests/test_tool.py
+++ b/python/tests/test_tool.py
@@ -1,0 +1 @@
+def test_new(): pass
"""

    changed = coverage_ci.git_changed_lines_from_diff(diff, {".py"})

    assert changed == {"scripts/tool.py": {1, 3}}


def test_changed_line_coverage_enforces_threshold(tmp_path, monkeypatch):
    coverage_ci = load_script("coverage_ci")
    report = tmp_path / "coverage.xml"
    write_cobertura(report)

    monkeypatch.setattr(
        coverage_ci,
        "git_changed_lines",
        lambda base_ref, workspace, extensions: {"scripts/tool.py": {1, 2, 3}},
    )
    args = coverage_ci.build_parser().parse_args(
        [
            "changed-lines",
            "--coverage",
            str(report),
            "--workspace",
            str(tmp_path),
            "--base-ref",
            "origin/main",
            "--label",
            "Python",
            "--threshold",
            "80",
            "--extension",
            ".py",
        ]
    )

    assert args.func(args) == 1


def test_crate_summary_groups_rust_coverage_by_workspace_package(tmp_path):
    coverage_ci = load_script("coverage_ci")
    (tmp_path / "shared" / "core" / "src").mkdir(parents=True)
    (tmp_path / "Cargo.toml").write_text('[workspace]\nmembers = ["shared/*"]\n')
    (tmp_path / "shared" / "core" / "Cargo.toml").write_text(
        '[package]\nname = "aether-core"\nversion = "0.1.0"\nedition = "2021"\n'
    )
    report = tmp_path / "coverage.xml"
    write_cobertura(report, filename="shared/core/src/lib.rs", hits=(1, 0))
    output = tmp_path / "summary.md"

    args = coverage_ci.build_parser().parse_args(
        [
            "crate-summary",
            "--coverage",
            str(report),
            "--workspace",
            str(tmp_path),
            "--output",
            str(output),
        ]
    )

    assert args.func(args) == 0
    assert "| aether-core | 1 | 2 | 50.00% |" in output.read_text()
