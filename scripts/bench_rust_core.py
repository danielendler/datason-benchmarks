#!/usr/bin/env python3
"""Simple benchmark harness for exercising the DataSON core.

This script performs repeated serialisation/deserialisation cycles on a
moderately large payload.  It is intentionally minimal; the heavy lifting is
handled by external profilers such as ``py-spy`` or ``scalene``.
"""

from __future__ import annotations

import argparse
import os

import datason


def generate_payload(size: int = 1 * 1024 * 1024) -> list:
    """Return a mixed payload roughly ``size`` bytes in size."""

    unit = "x" * 10
    count = max(1, size // 50)
    return [
        {
            "id": i,
            "value": unit,
            "flags": {"enabled": True, "ratio": i % 7},
            "items": [unit, unit[::-1], unit.upper()],
        }
        for i in range(count)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark DataSON core")
    parser.add_argument("--with-rust", choices=["on", "off"], default="off")
    parser.add_argument("--loops", type=int, default=200, help="Iteration count")
    args = parser.parse_args()

    os.environ["DATASON_RUST"] = "1" if args.with_rust == "on" else "0"

    payload = generate_payload()

    for _ in range(args.loops):
        s = datason.save_string(payload)
        datason.load_basic(s)


if __name__ == "__main__":
    main()

