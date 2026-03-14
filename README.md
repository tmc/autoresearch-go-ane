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

## Credits

- [karpathy/autoresearch](https://github.com/karpathy/autoresearch) — autonomous research pattern
- [maderix/ANE](https://github.com/maderix/ANE) — Go Apple Neural Engine training (vendored in `ane/`)
- [tmc/apple](https://github.com/tmc/apple) — Go Apple platform bindings

## License

MIT
