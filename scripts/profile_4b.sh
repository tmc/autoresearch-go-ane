#!/bin/bash
# Profile Qwen3-4B inference to find performance bottlenecks.
#
# Usage: bash scripts/profile_4b.sh
#
# Produces:
#   - profile_sample.txt: macOS sample-based CPU profile
#   - profile_cpu.pprof: Go CPU profile (viewable with go tool pprof)
#   - profile_mem.pprof: Go heap profile

set -e

MODEL="${MODEL:-qwen3-4b.bin}"
TOKENIZER="${TOKENIZER:-$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/*/}"
TOKENS="${TOKENS:-32}"

echo "=== Building generate command ==="
go build -o /tmp/ane-generate ./cmd/generate

echo ""
echo "=== Running with Go CPU profile ==="
/tmp/ane-generate \
    --model "$MODEL" \
    --hf-tokenizer $TOKENIZER \
    --prompt "Hello world" \
    --max-tokens "$TOKENS" \
    --temperature 0 \
    --seq 128 \
    --cpuprofile profile_cpu.pprof \
    --memprofile profile_mem.pprof \
    2>&1

echo ""
echo "=== Top CPU consumers (pprof) ==="
go tool pprof -top -cum profile_cpu.pprof 2>/dev/null | head -30

echo ""
echo "=== Running with macOS sample profiler ==="
/tmp/ane-generate \
    --model "$MODEL" \
    --hf-tokenizer $TOKENIZER \
    --prompt "Hello world" \
    --max-tokens "$TOKENS" \
    --temperature 0 \
    --seq 128 &
PID=$!
sleep 2  # Let it start generating
sample "$PID" 5 -file profile_sample.txt 2>/dev/null || true
wait "$PID" 2>/dev/null || true

echo ""
echo "=== Profile files ==="
ls -lh profile_cpu.pprof profile_mem.pprof profile_sample.txt 2>/dev/null

echo ""
echo "=== View detailed profile ==="
echo "  go tool pprof -http=:8080 profile_cpu.pprof"
echo "  open profile_sample.txt"
