#!/usr/bin/env bash
# Generate flamegraphs for DataSON with and without the Rust backend.
set -euo pipefail

mkdir -p results

py-spy record -o results/flame_off.svg -- \
    python scripts/bench_rust_core.py save_string --repeat 200 --with-rust off
py-spy record -o results/flame_on.svg --native -- \
    python scripts/bench_rust_core.py save_string --repeat 200 --with-rust on

echo "Flamegraphs written to results/flame_off.svg and results/flame_on.svg"

