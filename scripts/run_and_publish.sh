#!/usr/bin/env bash
# run_and_publish.sh — Run inference benchmark, parse, publish to ensue, and store in git notes.
#
# Usage:
#   bash scripts/run_and_publish.sh '<description>' [keep|discard|crash]
#
# Steps:
#   1. Run BenchmarkEvalLogits with benchstat-friendly flags
#   2. Attach raw output as git note via bench-note
#   3. Parse output through parse_bench.py to get JSON
#   4. Publish result to ensue coordinator
#
# Requires: bench-note binary, python3, requests pip package

set -euo pipefail

DESC="${1:?Usage: run_and_publish.sh '<description>' [keep|discard|crash]}"
STATUS="${2:-keep}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

BENCH_RAW=$(mktemp)
BENCH_JSON=$(mktemp)
trap 'rm -f "$BENCH_RAW" "$BENCH_JSON"' EXIT

echo "==> Running BenchmarkEvalLogits (-benchtime 5x -count 6)..."
go test -bench BenchmarkEvalLogits -benchtime 5x -count 6 -run '^$' . 2>&1 | tee "$BENCH_RAW"

echo ""
echo "==> Attaching results as git note..."
if [ -x ./bench-note ]; then
    ./bench-note run --from-file="$BENCH_RAW" --benchtime=5x --count=6 2>/dev/null || true
fi

echo ""
echo "==> Parsing benchmark output..."
python3 "$SCRIPT_DIR/parse_bench.py" < "$BENCH_RAW" > "$BENCH_JSON"
cat "$BENCH_JSON"

echo ""
echo "==> Publishing to ensue..."
GIT_DIFF=""
if git log --oneline -1 >/dev/null 2>&1; then
    GIT_DIFF=$(git diff HEAD~1 2>/dev/null || echo "")
fi

python3 "$SCRIPT_DIR/coordinator.py" publish_result "$DESC" "$(cat "$BENCH_JSON")" "$GIT_DIFF" "$STATUS"

echo ""
echo "==> Done."
