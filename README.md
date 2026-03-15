# autoresearch-go-ane

Autonomous AI research on Apple Silicon using ANE-accelerated training and inference, inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch).

A Claude Code agent autonomously optimizes ML inference on Apple Neural Engine. It edits code, runs benchmarks, verifies correctness, and iterates -- all without human intervention. The agent uses [Ensue](https://ensue-network.ai) for persistent memory across sessions and [benchstat](https://pkg.go.dev/golang.org/x/perf/cmd/benchstat) for statistically rigorous comparison.

## Autoresearch Results

### Inference Speed: 12 -> 11,289 tokens/s (977x faster)

The agent ran autonomously on the `autoresearch/mar14-infer` branch, identifying and fixing critical performance bugs in the ANE inference pipeline. All optimizations were discovered, implemented, verified, and committed by Claude Code without human guidance on what to change.

| # | Change | tokens/s | Speedup | How it was found |
|---|--------|----------|---------|------------------|
| 0 | Baseline | **12** | 1x | Agent established baseline measurement |
| 1 | BLAS-accelerated `linearCF` | **366** | 30x | Agent noticed `linearCF` was a naive Go triple-loop, added `cblas_sgemm` |
| 2 | CPU BLAS classifier head | **416** | 35x | Agent profiled and found ANE tile dispatch slower than single BLAS call |
| 3 | Fix ANE RMSNorm weight layout | **423** | 35x | Agent added timing instrumentation, discovered ANE path was failing silently |
| 4 | Fix ANE RMSNorm input layout | **10,667** | 889x | Agent traced the silent fallback to CPU, fixed both ANE input mismatches |
| 5 | vDSP residual blending | **11,178** | 932x | Agent replaced scalar Go loop with vDSP_vsmul + vDSP_vsma |
| 6 | Pre-scale Wo/W2 weights | **11,289** | 941x | Agent baked residual scale into weights to skip CPU blending entirely |

The biggest win (step 4, 889x) came from discovering that ANE inference was **completely broken** -- the RMSNorm kernel's compiled tensor layouts didn't match what the Go code provided, causing a silent fallback to unaccelerated CPU loops for every inference call.

### Time Breakdown (22ms per inference, M5 Max)

```
Embed lookup:     0.2ms  ( 1%)
12 ANE layers:   13.5ms  (61%)  <- ANE compute-bound
RMSNorm (ANE):    0.2ms  ( 1%)
Classifier BLAS:  8.2ms  (37%)  <- 32K x 768 x 256 cblas_sgemm
```

### Autonomous Agent Loop

The agent follows a protocol defined in [program.md](program.md):

```
LOOP FOREVER:
  1. RECALL:    python3 scripts/coordinator.py analyze     (Ensue persistent memory)
  2. CHECK:     python3 scripts/coordinator.py check_tried (deduplicate experiments)
  3. IMPLEMENT: edit ane/ files
  4. VERIFY:    go test -run TestInferenceCorrectness       (golden logits oracle)
  5. BENCHMARK: go test -bench BenchmarkEvalLogits          (measure speed)
  6. COMPARE:   benchstat bench_before.txt bench_after.txt  (statistical rigor)
  7. PUBLISH:   python3 scripts/coordinator.py publish_result (persist to Ensue)
  8. DECIDE:    keep or git reset --hard HEAD~1
```

Key design: every experiment is published to Ensue (even failures), so the agent never repeats a failed approach across sessions.

## Performance Dashboard (anperf)

The project includes `anperf`, a real-time performance dashboard served at `localhost:9090/perf/` during training:

- Live training loss and throughput charts (Canvas-based, SSE updates)
- ANE vs CPU vs Adam timing breakdown with component waterfall
- Run versioning with disk persistence (`.anperf/` directory)
- Inference benchmarking tab

## Architecture

```
EvalLogits -> evalLogitsANEInto:
  EmbedLookup (CPU, 0.2ms)
    -> 12x LayerForward (ANE: RMSNorm + QKV + RoPE + SDPA + Wo + residual + FFN)
      -> RMSNorm (ANE, 0.2ms)
        -> Classifier (CPU BLAS, 8.2ms)
```

110M-parameter Llama2-style transformer on [TinyStories](https://huggingface.co/datasets/enio/TinyStories):

| Parameter | Value |
|---|---|
| Vocab | 32,000 (Llama2 BPE) |
| Dim | 768 |
| Hidden | 2,048 |
| Heads | 12 |
| Layers | 12 |
| Sequence length | 256 |

## Setup

```bash
git clone https://github.com/tmc/autoresearch-go-ane.git
cd autoresearch-go-ane
bash scripts/setup.sh   # downloads ~40MB token data + ~420MB model
go test -bench BenchmarkEvalLogits -benchtime 5x -v
```

Requirements: macOS with Apple Silicon (M1+), Go 1.24+

## Credits

- [karpathy/autoresearch](https://github.com/karpathy/autoresearch) -- autonomous research pattern
- [tmc/apple](https://github.com/tmc/apple) -- Go Apple platform bindings
- [Ensue](https://ensue-network.ai) -- persistent agent memory across sessions

## License

MIT
