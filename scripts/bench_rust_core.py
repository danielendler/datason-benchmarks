"""Benchmark the DataSON Rust core accelerator.

This script provides a thin command-line interface to measure the
performance of ``datason.save_string`` and ``datason.load_basic`` with the
Rust extension toggled on or off.  It is intentionally lightweight and is
meant as a starting point for the full benchmark suite described in the
project's PRD.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, Tuple

import sys

# Ensure project root is on sys.path so that ``benchmarks`` can be imported
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import datason

try:  # pragma: no cover - optional dependency
    from benchmarks.payloads_json_basic import (
        make_flat,
        make_mixed,
        make_nested,
    )
except Exception:  # pragma: no cover - keep runtime failures obvious
    raise SystemExit("payload generators missing; ensure repository layout is correct")

SHAPES: Dict[str, Callable[[int], Tuple[object, str]]] = {
    "flat": make_flat,
    "nested": make_nested,
    "mixed": make_mixed,
}


def benchmark(op: str, shape: str, size: int, repeats: int) -> Dict[str, float]:
    """Run a simple timing benchmark.

    This function intentionally keeps logic simple; it does not attempt to
    implement all PRD features but establishes a structure that can be
    extended incrementally.
    """
    obj, json_payload = SHAPES[shape](size)

    def run_once() -> None:
        if op == "save_string":
            # Older versions of DataSON may not expose ``save_string``;
            # fall back to ``serialize`` so the skeleton benchmark still runs.
            save = getattr(datason, "save_string", getattr(datason, "serialize"))
            save(obj)
        else:
            load = getattr(datason, "load_basic", getattr(datason, "parse"))
            load(json_payload)

    # Warm up
    run_once()
    start = time.perf_counter()
    for _ in range(repeats):
        run_once()
    end = time.perf_counter()
    duration = (end - start) / repeats
    return {"median_s": duration, "throughput_ops_per_s": 1.0 / duration}


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "op",
        nargs="?",
        default="save_string",
        choices=["save_string", "load_basic"],
    )
    parser.add_argument("--with-rust", choices=["on", "off", "auto"], default="auto")
    parser.add_argument("--sizes", default="10k,100k,1m")
    parser.add_argument("--shapes", default="flat,nested,mixed")
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("results.json"))
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    if args.with_rust != "auto":
        os.environ["DATASON_RUST"] = "1" if args.with_rust == "on" else "0"

    sizes = []
    for label in args.sizes.split(","):
        if label.endswith("k"):
            sizes.append(int(label[:-1]) * 1000)
        elif label.endswith("m"):
            sizes.append(int(label[:-1]) * 1000 * 1000)
        else:
            sizes.append(int(label))

    shapes = args.shapes.split(",")

    results = {"metadata": {"datason_version": datason.__version__}, "cases": []}
    for shape in shapes:
        for size in sizes:
            stats = benchmark(args.op, shape, size, args.repeat)
            results["cases"].append({"op": args.op, "shape": shape, "size": size, **stats})

    args.output.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
