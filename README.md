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

The `autoresearch/mar14-infer` branch contains a focused optimization effort to maximize `BenchmarkEvalLogits` throughput on Apple Silicon. The autonomous agent iterated through the ANE inference pipeline, identifying and fixing bottlenecks in the forward pass.

### Summary

| Metric | Before | After | Change |
|---|---|---|---|
| Training throughput | 862 tok/s | 1,552 tok/s | **+80%** |
| Step time | 309 ms | 168 ms | **-46%** |
| ANE utilization | 26.4% | 30.0% | +3.6pp |
| ANE eval time | 84 ms | 50 ms | **-40%** |
| EvalLogits latency | — | 22 ms/op | ~11.6k tok/s |

### Optimization History

Training throughput measured via anperf snapshots. Inference latency from `BenchmarkEvalLogits` on M5 Max.

| # | Commit | Change | tok/s | Delta | Step ms | ANE ms | ANE % | EvalLogits |
|---|--------|--------|-------|-------|---------|--------|-------|------------|
| 0 | `9abba8b` | Baseline (unoptimized) | 872 | — | 309 ms | 84 ms | 26.4% | — |
| 1 | `008d56b` | Inference optimization infra | 1,477 | **+69%** | 176 ms | 52 ms | 29.5% | — |
| 2 | `9fabd63` | BLAS-accelerated forward + prealloc bufs | 1,552 | **+5.1%** | 168 ms | 50 ms | 30.0% | — |
| 3 | `fcce277` | CPU BLAS classifier (replace ANE tiles) | — | — | — | — | — | ~22.7 ms |
| 4 | `4bbdede` | Inference-only layer forward (skip taps) | — | — | — | — | — | ~22.5 ms |
| 5 | `37e217b` | Fix ANE RMSNorm weight tensor layout | — | — | — | — | — | correctness |
| 6 | `ec51a5d` | Fix ANE RMSNorm input layout (both paths) | — | — | — | — | — | correctness |
| 7 | `2f6a526` | vDSP-accelerated residual blending | — | — | — | — | — | ~22.3 ms |
| 8 | `3c0afbb` | Inference SDPA kernel + fused residual | — | — | — | — | — | ~22.1 ms |

**Cumulative: 872 -> 1,552 tok/s training (+80%), 22.1 ms/op inference (~11.6k tok/s)**

Notes:
- Commits 0-2 have training throughput snapshots (measured via anperf during 2-3 min training runs)
- Commits 3-8 targeted inference only (`BenchmarkEvalLogits`); training throughput not re-measured
- The `008d56b` jump (+69%) includes multiple changes bundled in the initial infra commit
- EvalLogits times are approximate (3-run averages, `-benchtime 5x`)

### Detailed Breakdown

**`9fabd63` — BLAS-accelerated forward pass (+67% throughput, 862 to 1,441 tok/s)**

Replaced hand-rolled matrix multiplications in `linearCF` with Apple's Accelerate framework (`cblas_sgemm`). Pre-allocated all CPU inference buffers (`qf`, `kf`, `vf`, `attOut`, `h1`, `h3`, `gate`, `ffOut`) on the engine struct instead of allocating per-call. This was the single largest win. BLAS exploits the AMX coprocessor and NEON SIMD on Apple Silicon, dramatically outperforming scalar Go loops for large matrix multiplies. Buffer pre-allocation eliminated ~10 allocations of 768x256 float32 arrays per inference call.

**`fcce277` — CPU BLAS classifier head**

Switched the 32K-vocab classifier matrix multiply from ANE dynamic tile dispatch to CPU BLAS (`cblas_sgemm`). The classifier is a single large matmul (768x32000x256) that ANE handles with many small tiled dispatches, each with IOSurface staging overhead. The ANE's tile-based execution has per-tile overhead (IOSurface lock/unlock, DMA transfer) that dominates when the tile count is high. A single CPU BLAS call avoids this overhead entirely.

**`4bbdede` — Inference-only layer forward**

Added a separate code path for inference that skips writing training "taps" (intermediate activations saved for backward pass). The training forward path writes attention and FFN intermediates to separate output surfaces; inference doesn't need these. Each training layer forward wrote 3 extra output tensors (Q, K, V projections for gradient computation). Skipping these saves ~3 IOSurface staging operations per layer x 12 layers = 36 unnecessary DMA transfers removed.

**`37e217b` + `ec51a5d` — ANE RMSNorm correctness fixes**

The ANE RMSNorm kernel expected weights in a specific tensor layout (1 x dim x 1 x 1) but was receiving them as a flat 1D array. A second mismatch existed between the attention-norm path and FFN-norm path activation layouts. Fixing both allowed the full ANE pipeline to run end-to-end correctly instead of silently falling back to CPU.

**`2f6a526` — vDSP residual acceleration**

Replaced the scalar Go loop in `blendResidualInPlace` (`x[i] = x[i] + scale*residual[i]`) with Apple's vDSP framework (`vDSP_vsma` for scaled-add). This is called after every attention and FFN block (24 times per forward pass). vDSP uses NEON SIMD and operates on the full 196K-element vector in a single call, avoiding Go's per-element bounds checking and loop overhead.

**`3c0afbb` — Inference SDPA kernel + fused residual**

Generated a specialized ANE attention kernel for inference that bakes the residual connection scale factor directly into the output projection weights (Wo). Instead of `out = x + scale * Wo @ attn`, the kernel computes `out = x + Wo_scaled @ attn` in a single ANE dispatch, eliminating a separate vDSP call and keeping more computation on the ANE. The inference kernel also has a smaller output footprint (just dim x seq channels vs dim x seq + taps).

### Inference Architecture

The hot path is `EvalLogits` -> `evalLogitsANEInto` in `ane/storiesane/engine.go`:

```
EmbedLookup (CPU)
  -> 12x LayerForward (ANE: attention + FFN + residual)
    -> RMSNorm (ANE)
      -> Classifier matmul (CPU BLAS, 768 x 32000 x 256)
```

Each layer forward on ANE executes a compiled MIL program that performs RMSNorm, QKV projection, rotary position encoding, scaled dot-product attention, output projection, residual add, FFN (gate + up + SiLU + down), and a second residual add — all in a single ANE dispatch.

### Performance Dashboard

The branch includes integration with `anperf`, a real-time performance dashboard served at `localhost:9090/perf/` during training:

- Live training loss and throughput charts (Canvas-based, SSE updates)
- ANE vs CPU vs Adam timing breakdown with component waterfall
- Run versioning with disk persistence (`.anperf/` directory)
- Inference benchmarking tab for tracking throughput over time

## Credits

- [karpathy/autoresearch](https://github.com/karpathy/autoresearch) — autonomous research pattern
- [maderix/ANE](https://github.com/maderix/ANE) — Go Apple Neural Engine training (vendored in `ane/`)
- [tmc/apple](https://github.com/tmc/apple) — Go Apple platform bindings

## License

MIT
