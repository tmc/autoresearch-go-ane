#!/bin/bash
# bench_infer.sh — Run BenchmarkEvalLogits + correctness check, emit JSON.
#
# Usage:
#   bash scripts/bench_infer.sh              # default: -benchtime 5x -count 6
#   bash scripts/bench_infer.sh 10x 10       # custom benchtime and count
#
# Exits non-zero if correctness fails.

set -euo pipefail
cd "$(dirname "$0")/.."

BENCHTIME="${1:-5x}"
COUNT="${2:-6}"

echo "=== Correctness check ===" >&2
if ! go test -run TestInferenceCorrectness -timeout 120s -v 2>&1 | tee /dev/stderr | grep -q "^ok"; then
    echo '{"error": "correctness check failed"}'
    exit 1
fi
echo "=== Correctness OK ===" >&2

echo "=== Running BenchmarkEvalLogits (benchtime=${BENCHTIME}, count=${COUNT}) ===" >&2
go test -bench BenchmarkEvalLogits -benchtime "${BENCHTIME}" -count "${COUNT}" -timeout 300s 2>/dev/null \
    | python3 scripts/parse_bench.py
