#!/usr/bin/env python3
"""Print and validate compact JUnit test summaries."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys
import xml.etree.ElementTree as ET


def iter_cases(path: Path):
    root = ET.parse(path).getroot()
    if root.tag == "testcase":
        yield root
        return
    yield from root.iter("testcase")


def skip_message(case: ET.Element) -> str:
    skipped = case.find("skipped")
    if skipped is None:
        return ""
    message = skipped.get("message") or (skipped.text or "")
    return " ".join(message.split()) or "no reason recorded"


def summarize(paths: list[Path]) -> tuple[int, int, int, int, Counter[str]]:
    tests = skipped = failures = errors = 0
    skip_reasons: Counter[str] = Counter()
    for path in paths:
        for case in iter_cases(path):
            tests += 1
            if case.find("skipped") is not None:
                skipped += 1
                skip_reasons[skip_message(case)] += 1
            if case.find("failure") is not None:
                failures += 1
            if case.find("error") is not None:
                errors += 1
    return tests, skipped, failures, errors, skip_reasons


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("junit", nargs="+", type=Path)
    parser.add_argument("--label", default="test suite")
    parser.add_argument("--expected-tests", type=int)
    parser.add_argument("--forbid-skips", action="store_true")
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()

    paths = [path for path in args.junit if path.exists()]
    missing = [str(path) for path in args.junit if not path.exists()]
    if missing and not args.allow_missing:
        print(f"Missing JUnit report(s): {', '.join(missing)}", file=sys.stderr)
        return 1

    tests, skipped, failures, errors, skip_reasons = summarize(paths)
    print(
        f"{args.label}: {tests} tests, {failures} failures, "
        f"{errors} errors, {skipped} skipped"
    )

    if skipped:
        skip_kind = "unexpected" if args.forbid_skips else "allowed"
        print(f"{args.label} {skip_kind} skip summary:")
        for reason, count in sorted(skip_reasons.items()):
            print(f"  {count}x {reason}")
    elif not args.forbid_skips:
        print(f"{args.label} allowed skip summary: 0 skips")

    if args.expected_tests is not None and tests != args.expected_tests:
        print(
            f"Expected {args.expected_tests} tests for {args.label}, found {tests}",
            file=sys.stderr,
        )
        return 1
    if args.forbid_skips and skipped:
        print(f"Unexpected skips in required suite {args.label}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
