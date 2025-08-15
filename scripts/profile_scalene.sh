#!/usr/bin/env bash
# Optional scalene run for per-line profiling.
set -euo pipefail
mkdir -p results
python -m scalene --outfile results/scalene_report.txt scripts/bench_rust_core.py --with-rust off

