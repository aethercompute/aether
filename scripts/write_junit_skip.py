#!/usr/bin/env python3
"""Write a one-test JUnit report for an intentionally skipped optional suite."""

from __future__ import annotations

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", required=True, type=Path)
    parser.add_argument("--classname", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--message", required=True)
    args = parser.parse_args()

    testsuite = ET.Element(
        "testsuite",
        name=args.classname,
        tests="1",
        failures="0",
        errors="0",
        skipped="1",
    )
    testcase = ET.SubElement(
        testsuite,
        "testcase",
        classname=args.classname,
        name=args.name,
        time="0",
    )
    ET.SubElement(testcase, "skipped", message=args.message)

    args.path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(testsuite).write(args.path, encoding="utf-8", xml_declaration=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
