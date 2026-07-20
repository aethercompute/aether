#!/usr/bin/env python3
"""Coverage checks used by CI."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import re
import subprocess
import sys
import tomllib
import xml.etree.ElementTree as ET


@dataclass(frozen=True)
class FileCoverage:
    lines: dict[int, int]

    @property
    def total(self) -> int:
        return len(self.lines)

    @property
    def covered(self) -> int:
        return sum(1 for hits in self.lines.values() if hits > 0)


@dataclass(frozen=True)
class CoverageReport:
    files: dict[str, FileCoverage]
    line_rate: float
    branch_rate: float | None


@dataclass(frozen=True)
class Package:
    name: str
    directory: str


def normalized_path(path: str, workspace: Path) -> str:
    raw = Path(path)
    if raw.is_absolute():
        try:
            return raw.resolve().relative_to(workspace.resolve()).as_posix()
        except ValueError:
            return raw.as_posix().lstrip("/")

    text = path.replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    while text.startswith("../"):
        text = text[3:]
    return text


def parse_percent(value: str | None) -> float | None:
    if value is None:
        return None
    parsed = float(value)
    return parsed * 100.0 if parsed <= 1.0 else parsed


def read_cobertura(path: Path, workspace: Path) -> CoverageReport:
    root = ET.parse(path).getroot()
    files: dict[str, FileCoverage] = {}
    for class_node in root.iter("class"):
        filename = class_node.get("filename")
        if not filename:
            continue
        lines = {
            int(line.get("number", "0")): int(line.get("hits", "0"))
            for line in class_node.iter("line")
            if line.get("number")
        }
        files[normalized_path(filename, workspace)] = FileCoverage(lines)

    covered = sum(file.covered for file in files.values())
    total = sum(file.total for file in files.values())
    line_rate = parse_percent(root.get("line-rate"))
    if line_rate is None:
        line_rate = (covered / total * 100.0) if total else 100.0
    return CoverageReport(files, line_rate, parse_percent(root.get("branch-rate")))


def coverage_for_path(report: CoverageReport, path: str) -> FileCoverage | None:
    normalized = path.replace("\\", "/")
    if normalized in report.files:
        return report.files[normalized]

    suffix = f"/{normalized}"
    matches = [coverage for name, coverage in report.files.items() if name.endswith(suffix)]
    if len(matches) == 1:
        return matches[0]
    return None


def check_metric(label: str, name: str, actual: float | None, minimum: float, tolerance: float) -> bool:
    if actual is None:
        print(f"{label} {name} coverage is unavailable", file=sys.stderr)
        return False
    print(f"{label} {name} coverage: {actual:.2f}% (minimum {minimum:.2f}%, tolerance {tolerance:.2f}%)")
    if actual + tolerance < minimum:
        print(f"{label} {name} coverage is below the approved floor", file=sys.stderr)
        return False
    return True


def command_check_baseline(args: argparse.Namespace) -> int:
    report = read_cobertura(args.coverage, args.workspace)
    ok = check_metric(args.label, "line", report.line_rate, args.min_line_rate, args.tolerance)
    if args.min_branch_rate is not None:
        ok = check_metric(args.label, "branch", report.branch_rate, args.min_branch_rate, args.tolerance) and ok
    return 0 if ok else 1


def workspace_packages(workspace: Path) -> list[Package]:
    root_manifest = workspace / "Cargo.toml"
    workspace_data = tomllib.loads(root_manifest.read_text())
    members = workspace_data.get("workspace", {}).get("members", [])
    excludes = set(workspace_data.get("workspace", {}).get("exclude", []))

    packages = []
    for member in members:
        for manifest in workspace.glob(f"{member}/Cargo.toml"):
            directory = manifest.parent.relative_to(workspace).as_posix()
            if directory in excludes:
                continue
            package_data = tomllib.loads(manifest.read_text())
            name = package_data.get("package", {}).get("name")
            if name:
                packages.append(Package(name, directory))
    return sorted(packages, key=lambda package: package.directory)


def package_for_file(packages: list[Package], filename: str) -> str:
    matches = [package for package in packages if filename == package.directory or filename.startswith(f"{package.directory}/")]
    if not matches:
        return "unmatched"
    return max(matches, key=lambda package: len(package.directory)).name


def command_crate_summary(args: argparse.Namespace) -> int:
    report = read_cobertura(args.coverage, args.workspace)
    packages = workspace_packages(args.workspace)
    totals: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for filename, coverage in report.files.items():
        package = package_for_file(packages, filename)
        totals[package][0] += coverage.covered
        totals[package][1] += coverage.total

    lines = ["# Rust Coverage By Crate", "", "| Crate | Covered Lines | Total Lines | Line Coverage |", "| --- | ---: | ---: | ---: |"]
    for package, (covered, total) in sorted(totals.items()):
        percent = (covered / total * 100.0) if total else 100.0
        lines.append(f"| {package} | {covered} | {total} | {percent:.2f}% |")
    covered = sum(value[0] for value in totals.values())
    total = sum(value[1] for value in totals.values())
    percent = (covered / total * 100.0) if total else 100.0
    lines.append(f"| TOTAL | {covered} | {total} | {percent:.2f}% |")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n")
    print(args.output.read_text(), end="")
    return 0


def parse_changed_lines(diff: str, extensions: set[str]) -> dict[str, set[int]]:
    changed: dict[str, set[int]] = defaultdict(set)
    current_file: str | None = None
    current_line: int | None = None

    for line in diff.splitlines():
        if line.startswith("+++ b/"):
            current_file = line[6:]
            current_line = None
            if not any(current_file.endswith(extension) for extension in extensions):
                current_file = None
            continue
        if line.startswith("@@"):
            match = re.search(r"\+(\d+)(?:,(\d+))?", line)
            current_line = int(match.group(1)) if match else None
            continue
        if current_file is None or current_line is None:
            continue
        if line.startswith("+"):
            if line[1:].strip():
                changed[current_file].add(current_line)
            current_line += 1
        elif not line.startswith("-"):
            current_line += 1
    return dict(changed)


def is_test_path(path: str) -> bool:
    parts = path.split("/")
    name = parts[-1]
    return "tests" in parts or name.startswith("test_") or name.endswith("_test.rs")


def git_changed_lines(base_ref: str, workspace: Path, extensions: set[str]) -> dict[str, set[int]]:
    merge_base = subprocess.check_output(
        ["git", "merge-base", base_ref, "HEAD"], cwd=workspace, text=True
    ).strip()
    patterns = [f":(glob)**/*{extension}" for extension in sorted(extensions)]
    diff = subprocess.check_output(
        ["git", "diff", "--unified=0", "--no-ext-diff", f"{merge_base}...HEAD", "--", *patterns],
        cwd=workspace,
        text=True,
    )
    return git_changed_lines_from_diff(diff, extensions)


def git_changed_lines_from_diff(diff: str, extensions: set[str]) -> dict[str, set[int]]:
    return {path: lines for path, lines in parse_changed_lines(diff, extensions).items() if not is_test_path(path)}


def command_changed_lines(args: argparse.Namespace) -> int:
    report = read_cobertura(args.coverage, args.workspace)
    extensions = set(args.extension)
    changed = git_changed_lines(args.base_ref, args.workspace, extensions)

    covered = 0
    total = 0
    missing_files = []
    for path, lines in sorted(changed.items()):
        coverage = coverage_for_path(report, path)
        if coverage is None:
            missing_files.append(path)
            total += len(lines)
            continue
        for line in lines:
            total += 1
            if coverage.lines.get(line, 0) > 0:
                covered += 1

    percent = (covered / total * 100.0) if total else 100.0
    print(f"{args.label} changed-line coverage: {covered}/{total} lines ({percent:.2f}%)")
    for path in missing_files:
        print(f"{args.label} changed source file missing from coverage report: {path}")
    if percent < args.threshold:
        print(
            f"{args.label} changed-line coverage is below {args.threshold:.2f}%",
            file=sys.stderr,
        )
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(required=True)

    baseline = subparsers.add_parser("check-baseline")
    baseline.add_argument("--coverage", required=True, type=Path)
    baseline.add_argument("--workspace", default=Path.cwd(), type=Path)
    baseline.add_argument("--label", required=True)
    baseline.add_argument("--min-line-rate", required=True, type=float)
    baseline.add_argument("--min-branch-rate", type=float)
    baseline.add_argument("--tolerance", default=0.0, type=float)
    baseline.set_defaults(func=command_check_baseline)

    crate_summary = subparsers.add_parser("crate-summary")
    crate_summary.add_argument("--coverage", required=True, type=Path)
    crate_summary.add_argument("--workspace", default=Path.cwd(), type=Path)
    crate_summary.add_argument("--output", required=True, type=Path)
    crate_summary.set_defaults(func=command_crate_summary)

    changed_lines = subparsers.add_parser("changed-lines")
    changed_lines.add_argument("--coverage", required=True, type=Path)
    changed_lines.add_argument("--workspace", default=Path.cwd(), type=Path)
    changed_lines.add_argument("--base-ref", required=True)
    changed_lines.add_argument("--label", required=True)
    changed_lines.add_argument("--threshold", required=True, type=float)
    changed_lines.add_argument("--extension", action="append", required=True)
    changed_lines.set_defaults(func=command_changed_lines)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
