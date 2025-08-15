#!/usr/bin/env python3
"""Profile DataSON stage timings across multiple scenarios.

This script enables DataSON's internal stage timers via the ``DATASON_PROFILE``
environment variable and collects timing information from ``datason.profile_sink``.
Payloads of varying sizes and shapes are generated to exercise different code
paths.  Results are aggregated (median and 95th percentile) and written to JSON
and CSV files under the ``results`` directory.

The script is intentionally lightweight to avoid adding dependencies to the core
``datason`` package; all profiling tools live in this benchmarks repository.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, List

import numpy as np

import datason


def generate_payload(kind: str, size: int):
    """Generate simple payloads roughly ``size`` bytes in size.

    The goal isn't to be exact, just to create data large enough to exercise the
    serializer.
    """

    unit = "x" * 10

    if kind == "flat":
        count = max(1, size // len(unit))
        return {f"k{i}": unit for i in range(count)}

    if kind == "nested":
        root: Dict[str, Dict[str, str]] = {}
        current = root
        depth = max(1, size // 100)
        for i in range(depth):
            current[f"level{i}"] = {"value": unit}
            current = current[f"level{i}"]
        return root

    # mixed
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


def aggregate_events(events: Iterable[dict], bucket: Dict[str, List[float]]) -> None:
    """Add timing events into ``bucket``.

    ``datason.profile_sink`` is expected to be an iterable of mappings containing
    a stage name and a duration (in nanoseconds).  The exact keys may evolve, so
    we probe several possibilities to remain forward compatible.
    """

    for ev in events:
        stage = (
            ev.get("stage")
            or ev.get("name")
            or ev.get("phase")
            or "unknown"
        )
        duration = (
            ev.get("duration")
            or ev.get("time")
            or ev.get("elapsed")
            or 0
        )
        # Convert nanoseconds to milliseconds if the value looks large.
        if duration > 1e5:
            duration_ms = duration / 1_000_000.0
        else:
            duration_ms = float(duration)
        bucket.setdefault(stage, []).append(duration_ms)


def summarise(bucket: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    for stage, values in bucket.items():
        summary[stage] = {
            "median_ms": float(median(values)),
            "p95_ms": float(np.percentile(values, 95)),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile DataSON stage timings")
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Number of repetitions per scenario",
    )
    parser.add_argument(
        "--with-rust",
        choices=["on", "off"],
        default="off",
        help="Whether to use the Rust backend if available",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory to write profiling artefacts",
    )
    args = parser.parse_args()

    if args.with_rust == "on" and not getattr(datason, "RUST_AVAILABLE", True):
        # When the Rust extension is missing, emit a message and exit gracefully.
        print("skipped (rust unavailable)")
        return

    os.environ["DATASON_PROFILE"] = "1"
    os.environ["DATASON_RUST"] = "1" if args.with_rust == "on" else "0"

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    sizes = [10 * 1024, 100 * 1024, 1 * 1024 * 1024]
    shapes = ["flat", "nested", "mixed"]

    stage_bucket: Dict[str, List[float]] = {}

    for size in sizes:
        for shape in shapes:
            payload = generate_payload(shape, size)
            for _ in range(args.runs):
                sink: List[dict] = []
                setattr(datason, "profile_sink", sink)
                encoded = datason.save_string(payload)
                datason.load_basic(encoded)
                aggregate_events(sink, stage_bucket)

    summary = summarise(stage_bucket)

    suffix = "on" if args.with_rust == "on" else "off"
    json_path = output / f"stage_times_{suffix}.json"
    csv_path = output / f"stage_times_{suffix}.csv"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["stage", "median_ms", "p95_ms"])
        for stage, stats in sorted(summary.items()):
            writer.writerow([stage, f"{stats['median_ms']:.6f}", f"{stats['p95_ms']:.6f}"])

    print(f"Wrote {json_path} and {csv_path}")


if __name__ == "__main__":
    main()

