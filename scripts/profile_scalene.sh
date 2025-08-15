#!/usr/bin/env bash
# Optional profiling lane using Scalene for per-line attribution.
set -euo pipefail

mkdir -p results

# Scalene writes its own report; we capture stdout for convenience.
python -m scalene scripts/bench_rust_core.py save_string --repeat 200 --with-rust off \
    2>&1 | tee results/scalene_report.txt

echo "Scalene report saved to results/scalene_report.txt"

