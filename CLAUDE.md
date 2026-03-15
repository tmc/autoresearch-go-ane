# CLAUDE.md

## Project Overview

Go implementation of LLM inference on Apple Neural Engine (ANE). Supports TinyStories 110M, Qwen3-0.6B, Qwen3-4B dense models, and MoE architectures. macOS/darwin only for ANE paths; `_other.go` stubs exist for non-darwin builds.

Module: `github.com/tmc/autoresearch-go-ane`

## Build & Test

```bash
# Setup (downloads ~460MB of data + model)
bash scripts/setup.sh

# Build
go build ./...

# Tests (need DATA and MODEL env vars, or defaults from setup.sh)
go test ./...

# Correctness oracle — must pass before benchmarking
go test -run TestInferenceCorrectness

# Benchmarks
go test -bench BenchmarkEvalLogits -benchtime 5x -count 6

# Save golden logits (run once after model changes)
go test -run TestSaveGoldenLogits
```

Requires **Go 1.25.2+**.

## Running Qwen3 Models

```bash
# Convert HuggingFace model to .bin (requires pip: transformers safetensors huggingface_hub)
python3 scripts/convert_hf.py Qwen/Qwen3-0.6B --output qwen3-0.6b.bin
python3 scripts/convert_hf.py Qwen/Qwen3-4B --output qwen3-4b.bin

# Convert MoE models
python3 scripts/convert_hf_moe.py Qwen/Qwen3-Coder-Next --output qwen3-coder-next.moe.bin

# Generate text (Go-native tokenization, no Python needed at runtime)
go run ./cmd/generate \
  --model qwen3-0.6b.bin \
  --hf-tokenizer ~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/*/ \
  --prompt "Once upon a time" \
  --max-tokens 64 --temperature 0.8

# With raw token IDs (skip tokenizer)
go run ./cmd/generate --model qwen3-4b.bin --prompt "9707" --raw --max-tokens 16
```

## Profiling & Performance

```bash
# Full profiling suite (pprof + macOS sample)
bash scripts/profile_4b.sh

# CPU profile only
go run ./cmd/generate --model qwen3-4b.bin --cpuprofile cpu.pprof --max-tokens 16
go tool pprof -http=:8080 cpu.pprof

# Memory profile
go run ./cmd/generate --model qwen3-4b.bin --memprofile mem.pprof --max-tokens 16

# Per-token timing breakdown (printed for first 3 tokens)
# Shows: total, qkv, attention, wo, ffn, classifier, fp16conv
go run ./cmd/generate --model qwen3-4b.bin --max-tokens 8 --temperature 0

# macOS sample profiler
go build -o /tmp/ane-gen ./cmd/generate && /tmp/ane-gen --model qwen3-4b.bin &
sample $! 5 -file profile.txt
```

### Performance Tooling

| Tool | File | Purpose |
|------|------|---------|
| `--cpuprofile` | cmd/generate | Go pprof CPU profile |
| `--memprofile` | cmd/generate | Go pprof heap profile |
| `TokenTimings` | storiesane/eval_cached.go | Per-token component breakdown |
| `scripts/profile_4b.sh` | scripts/ | Automated profiling suite |
| `BNNSLinearFP16` | storiesane/bnns_darwin.go | BNNS fp16-weight GEMV |
| `MetalLinearSingle` | storiesane/metal_darwin.go | MPS GPU matmul |
| `linearSingleGEMV` | storiesane/accel_darwin.go | cblas_sgemv for seq=1 |

### Current Performance

| Model | tok/s | Path | Notes |
|-------|-------|------|-------|
| TinyStories 110M | 28.6 | ANE kernels | Fused QKV+W1W3, BLAS classifier |
| Qwen3-0.6B | 46.5 | KV cache + BLAS | Go-native BPE tokenization |
| Qwen3-4B | 3.7 | KV cache + BLAS | Memory-bandwidth bound (15GB fp32) |

## Project Structure

- `ane/stories/` — Model config, weights, checkpoints, tokenizer
  - `config.go` — ModelConfig, Qwen3 presets, GQA support
  - `weights.go` — ModelWeights, LayerWeightsFP16, CompressToFP16
  - `hf_tokenizer.go` — Go-native BPE tokenizer (encode+decode from tokenizer.json)
  - `moe_config.go` — MoEConfig, ExpertWeights, MoELayerWeights
  - `moe_weights.go` — LoadMoEPretrained (.moe.bin format)
  - `generate.go` — SampleToken, GenerateOptions
- `ane/storiesane/` — Training/inference engine (core logic)
  - `engine.go` — Engine, Open, EvalLogits, Step
  - `eval_cached.go` — EvalNextToken (KV-cached), TokenTimings
  - `generate.go` — Generate, GenerateStream (autoregressive)
  - `layer_tiled_darwin.go` — Tiled ANE layer forward for large models
  - `moe_layer_darwin.go` — MoE expert dispatch
  - `accel_darwin.go` — Accelerate BLAS (sgemm, sgemv, siluMul)
  - `bnns_darwin.go` — BNNS fp16-weight GEMV
  - `metal_darwin.go` — Metal MPS GPU matmul
  - `fp16_pack_darwin.go` — NEON fp16↔fp32 conversion
  - `kvcache.go` — KV cache for incremental generation
- `ane/mil/` — MIL program generators (ANE kernel definitions)
  - `gqa_dynamic.go` — GQA attention MIL generator
- `ane/dynamicmatmul/` — Runtime-provided weights matmul
- `ane/runtime/` — Framework loading & Obj-C discovery
- `ane/iosurface/` — IOSurface GPU memory management
- `ane/model/` — MIL kernel compilation
- `cmd/generate/` — Text generation CLI with profiling
- `scripts/` — Model conversion, tokenization, profiling
  - `convert_hf.py` — HuggingFace → .bin converter (dense models)
  - `convert_hf_moe.py` — HuggingFace → .moe.bin converter (MoE)
  - `tokenize.py` — Python tokenizer wrapper
  - `profile_4b.sh` — Automated Qwen3-4B profiling
- `experiment.go` — Hyperparameter config (agent-editable)
- `main.go`, `bench_test.go`, `correctness_test.go` — Read-only harness files
- `testdata/golden_logits.bin` — Reference inference logits

## Conventions

- **Imports**: stdlib first, blank line, then local packages. No third-party import groups mixed in.
- **Python**: Use `uv` for package management, not `pip`.
- **Platform files**: `_darwin.go` for macOS/ANE, `_other.go` for stubs.
- **Error handling**: Always wrap with context — `fmt.Errorf("doing X: %w", err)`. No panics in library code.
- **Constructors**: `New*()` or `Open()`/`Close()` pairs for resources.
- **Configuration**: Options struct pattern, not variadic options.
- **Hot paths**: Preallocate buffers. No allocations in inner loops. Use scratch buffers.
- **Lazy init**: Defer compilation/allocation until first use, not in constructors.
- **Naming**: CamelCase exported, lowercase unexported. Keep abbreviations uppercase: `RMS`, `FFN`, `QKV`, `KV`, `GQA`, `MoE`.
- **Testing**: Golden file testing via `testdata/`. Correctness oracle uses fp16 tolerance (1e-3 rel, 1e-4 abs).
- **Token type**: `int32` throughout (supports Qwen3 vocab 151K, exceeds uint16 max).

## Agent Experiment Protocol

See `program.md` for full details. Key rules:

1. ONE change per experiment (isolation for causality)
2. Correctness oracle must pass before benchmarking
3. Use benchstat for statistical rigor
4. Public Engine API is stable: `EvalLogits`, `Step`, `Open`, `Close`
