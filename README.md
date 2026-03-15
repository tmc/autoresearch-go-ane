# autoresearch-go-ane

Autonomous AI research on Apple Silicon using ANE-accelerated training, inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch).

An LLM agent edits a single file (`experiment.go`), runs 5-minute training experiments on a TinyStories language model using the Apple Neural Engine, and keeps changes that improve validation loss. Go benchmarks + [benchstat](https://pkg.go.dev/golang.org/x/perf/cmd/benchstat) provide statistically rigorous comparison between experiments.

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Go 1.24+

## Setup

```bash
git clone https://github.com/tmc/autoresearch-go-ane.git
cd autoresearch-go-ane
bash scripts/setup.sh   # downloads ~40MB token data + ~420MB model
```

## Quick start

```bash
# Run baseline (random init, 5 min training)
go run . -data tinystories_data00.bin > run.log 2>&1
grep "^val_loss:" run.log

# Run benchmarks
go test -bench . -benchtime 5x -v
```

## Benchmarks

`go test -bench .` reports:

| Benchmark | Key metrics |
|---|---|
| `BenchmarkStep` | training loss, step time (ms), ANE eval time, Adam time |
| `BenchmarkEvalLogits` | single-window inference throughput |
| `BenchmarkEvalLoss` | **validation loss** (the optimization target) |
| `BenchmarkLRSchedule` | LR schedule computation overhead |

Compare experiments with [benchstat](https://pkg.go.dev/golang.org/x/perf/cmd/benchstat):

```bash
go install golang.org/x/perf/cmd/benchstat@latest

# Capture baseline
go test -bench . -benchtime 5x -count 6 | tee bench_before.txt

# Edit experiment.go, then capture new results
go test -bench . -benchtime 5x -count 6 | tee bench_after.txt

# Compare
benchstat bench_before.txt bench_after.txt
```

## How it works

1. Agent edits `experiment.go` (hyperparameters, LR schedule, etc.)
2. Runs Go benchmarks to measure training throughput and validation loss
3. Uses `benchstat` to compare before/after with statistical rigor
4. Keeps the change if `val_loss` improved significantly, discards otherwise
5. Repeats

See [program.md](program.md) for full agent instructions.

## Model

110M-parameter Llama2-style transformer trained on [TinyStories](https://huggingface.co/datasets/enio/TinyStories):

| Parameter | Value |
|---|---|
| Vocab | 32,000 (Llama2 BPE) |
| Dim | 768 |
| Hidden | 2,048 |
| Heads | 12 |
| Layers | 12 |
| Sequence length | 256 (default) |

## Inference Optimization Results (mar14-infer)

**977x speedup**: 21.5s -> 22ms per inference on Apple M5 Max.

### Tokens/s History

| # | Change | ns/op | tokens/s | vs baseline |
|---|--------|-------|----------|-------------|
| 0 | Baseline (broken ANE, naive CPU loops) | 21,500,000,000 | **12** | 1x |
| 1 | BLAS-accelerated `linearCF` + pre-alloc buffers | 700,000,000 | **366** | 30x |
| 2 | CPU BLAS classifier head (replace ANE tiles) | 616,000,000 | **416** | 35x |
| 3 | Fix ANE RMSNorm weight tensor expansion | 606,000,000 | **423** | 35x |
| 4 | Fix ANE RMSNorm input layout (both paths) | 24,000,000 | **10,667** | 889x |
| 5 | vDSP-accelerated residual blending | 22,900,000 | **11,178** | 932x |
| 6 | Pre-scale Wo/W2 weights (eliminate CPU residual) | 22,670,000 | **11,289** | 941x |

### What Was Wrong

Two critical bugs caused ANE inference to silently fall back to a naive CPU path:

1. **No BLAS for forward matmuls** -- `linearCF` used a naive Go triple-loop instead of `cblas_sgemm`. Fixed by adding Accelerate BLAS to the forward path.

2. **ANE RMSNorm input layout mismatch** -- The ANE compiler transforms tensor shapes unpredictably. The RMSNorm kernel's compiled inputs expected different sizes than what the Go code provided, causing `WriteInputFP16` to fail. The ANE path silently fell back to CPU for ALL inference. Fixed by detecting the compiled layout and adapting input data to match.

### Time Breakdown (22ms total on M5 Max)

```
Embed lookup:     0.2ms  ( 1%)
12 ANE layers:   13.5ms  (61%)  <- ANE compute bound
RMSNorm (ANE):    0.2ms  ( 1%)
Classifier BLAS:  8.2ms  (37%)  <- 32K x 768 x 256 matmul
```

### Architecture

```
EvalLogits -> evalLogitsANEInto:
  EmbedLookup (CPU)
    -> 12x LayerForward (ANE: RMSNorm + QKV + RoPE + SDPA + Wo + residual + FFN)
      -> RMSNorm (ANE)
        -> Classifier (CPU BLAS, cblas_sgemm 32000x768x256)
```

## Credits

- [karpathy/autoresearch](https://github.com/karpathy/autoresearch) — autonomous research pattern
- [maderix/ANE](https://github.com/maderix/ANE) — Go Apple Neural Engine training (vendored in `ane/`)
- [tmc/apple](https://github.com/tmc/apple) — Go Apple platform bindings

## License

MIT
