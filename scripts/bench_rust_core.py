#!/usr/bin/env python3
"""Minimal workload used for py-spy flamegraph generation.

This script repeatedly serializes and deserializes a sample payload to
exercise the core engine.  The ``--with-rust`` flag toggles optional
Rust acceleration to showcase performance differences in the flamegraphs.
"""
import argparse
import os
import datason


def build_payload() -> dict:
    """Create a moderately sized payload for profiling."""
    return {"numbers": list(range(1000)), "text": "x" * 1024}


def run(with_rust: bool, iterations: int) -> None:
    """Execute the workload with optional Rust acceleration."""
    os.environ["DATASON_RUST"] = "1" if with_rust else "0"
    payload = build_payload()
    for _ in range(iterations):
        data = datason.dumps(payload)
        datason.load_basic(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Datason core workload for profiling")
    parser.add_argument("--with-rust", choices=["on", "off"], default="on")
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    run(args.with_rust == "on", args.iterations)

