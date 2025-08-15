#!/usr/bin/env python3
"""Collect per-stage timing information for Datason operations.

The script generates payloads of various sizes and structures, toggles
`DATASON_PROFILE`, and records serialization/deserialization times.  The
results are aggregated into median and 95th percentile statistics and
written to JSON and CSV files under ``results/``.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import time
from pathlib import Path
from typing import Dict, List

import datason


def generate_payload(size: int, kind: str) -> dict:
    """Generate synthetic payloads of approximately ``size`` bytes."""
    blob = "x" * size
    if kind == "flat":
        return {f"k{i}": blob for i in range(1)}
    if kind == "nested":
        return {"level1": {"level2": {"data": blob}}}
    # mixed
    return {"list": [blob, blob], "dict": {"a": blob, "b": blob}}


def measure(data: dict, iterations: int) -> Dict[str, List[float]]:
    """Measure serialization and deserialization times."""
    serialize: List[float] = []
    deserialize: List[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        dumped = datason.dumps(data)
        serialize.append(time.perf_counter() - start)
        start = time.perf_counter()
        datason.load_basic(dumped)
        deserialize.append(time.perf_counter() - start)
    return {"serialize": serialize, "deserialize": deserialize}


def aggregate(stage_times: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
    """Aggregate stage timings into median and p95 statistics."""
    results: Dict[str, Dict[str, float]] = {}
    for stage, values in stage_times.items():
        if not values:
            continue
        p95 = statistics.quantiles(values, n=100)[94] if len(values) > 1 else values[0]
        results[stage] = {
            "median": statistics.median(values),
            "p95": p95,
        }
    return results


def profile(with_rust: bool, iterations: int, output_dir: Path) -> None:
    os.environ["DATASON_PROFILE"] = "1"
    os.environ["DATASON_RUST"] = "1" if with_rust else "0"
    output_dir.mkdir(parents=True, exist_ok=True)

    sizes = [10_000, 100_000, 1_000_000]
    structures = ["flat", "nested", "mixed"]

    aggregated: Dict[str, Dict[str, Dict[str, float]]] = {}
    for size in sizes:
        for struct in structures:
            payload = generate_payload(size, struct)
            timings = measure(payload, iterations)
            aggregated[f"{struct}_{size}"] = aggregate(timings)

    suffix = "on" if with_rust else "off"
    json_path = output_dir / f"stage_times_{suffix}.json"
    csv_path = output_dir / f"stage_times_{suffix}.csv"

    with json_path.open("w") as jf:
        json.dump(aggregated, jf, indent=2)
    with csv_path.open("w", newline="") as cf:
        writer = csv.writer(cf)
        writer.writerow(["scenario", "stage", "median", "p95"])
        for scenario, stages in aggregated.items():
            for stage, stats in stages.items():
                writer.writerow([
                    scenario,
                    stage,
                    f"{stats['median']:.6f}",
                    f"{stats['p95']:.6f}",
                ])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile Datason stage timings")
    parser.add_argument("--with-rust", choices=["on", "off"], default="off")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()
    profile(args.with_rust == "on", args.iterations, Path(args.output_dir))

