#!/usr/bin/env bash
# Generate flamegraphs using py-spy for Datason with and without Rust.
set -euo pipefail
MODE=${1:-both}
mkdir -p results
if [[ "$MODE" == "off" || "$MODE" == "both" ]]; then
  py-spy record -o results/flame_off.svg -- python scripts/bench_rust_core.py --with-rust off
fi
if [[ "$MODE" == "on" || "$MODE" == "both" ]]; then
  py-spy record -o results/flame_on.svg --native -- python scripts/bench_rust_core.py --with-rust on
fi

