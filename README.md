# autoresearch-go-ane

A Go port of [karpathy/autoresearch](https://github.com/karpathy/autoresearch) for Apple Silicon. Give an AI agent a real LLM training setup on the Neural Engine and let it experiment autonomously overnight. It modifies the code, trains for 5 minutes, checks if the result improved, keeps or discards, and repeats. You wake up in the morning to a log of experiments and (hopefully) a better model.

The training code is a 110M-parameter Llama2-style transformer on [TinyStories](https://huggingface.co/datasets/enio/TinyStories), running entirely in Go with ANE acceleration via [purego](https://github.com/ebitengine/purego) (no CGo). Experiments are measured with Go benchmarks and compared with [benchstat](https://pkg.go.dev/golang.org/x/perf/cmd/benchstat) for statistical rigor.

## How it works

The repo is deliberately kept small and only really has three files that matter:

- **`harness.go`** — fixed evaluation harness, data loading, random init. Not modified.
- **`experiment.go`** — the primary file the agent edits. Contains hyperparameters, learning rate schedule, and training configuration. **This file is edited and iterated on by the agent.**
- **`program.md`** — instructions for the agent. Point your agent here and let it go. **This file is edited and iterated on by the human.**

The agent can also edit files in `ane/` for deeper changes (optimizer, forward/backward pass, loss function, activations). See [program.md](program.md) for the full scope.

By design, training runs for a **fixed 5-minute time budget** (wall clock, excluding warmup/compilation), regardless of the details of your Apple Silicon chip. The metric is **val_loss** (validation cross-entropy in nats) — lower is better.

## Quick start

**Requirements:** macOS with Apple Silicon (M1/M2/M3/M4), Go 1.24+.

```bash
# 1. Clone and set up
git clone https://github.com/tmc/autoresearch-go-ane.git
cd autoresearch-go-ane
bash scripts/setup.sh   # downloads ~40MB token data + ~420MB model

# 2. Run baseline benchmarks (~2 min)
go test -bench . -benchtime 5x -count 3 -v

# 3. Install benchstat for comparing experiments
go install golang.org/x/perf/cmd/benchstat@latest
```

## Running the agent

Spin up Claude Code (or whatever agent you prefer) in this repo, then prompt something like:

```
Read program.md and let's kick off a new experiment! Do the setup first.
```

The `program.md` file is the agent's complete instruction set — it knows how to create a branch, run benchmarks, compare results, and loop forever.

## Project structure

```
experiment.go   — hyperparameters, LR schedule, config (agent modifies this)
harness.go      — evaluation harness, data loading (do not modify)
bench_test.go   — Go benchmarks (do not modify)
program.md      — agent instructions
ane/            — ANE training engine (agent can modify for deeper experiments)
```

## Benchmarks

`go test -bench .` reports:

| Benchmark | Key metrics |
|---|---|
| `BenchmarkStep` | training loss, step time, ANE eval time, optimizer time, ANE watts/compute |
| `BenchmarkEvalLogits` | single-window inference throughput, ANE utilization |
| `BenchmarkEvalLoss` | **val_loss** (the optimization target), ANE power |
| `BenchmarkLRSchedule` | LR schedule computation overhead |

Compare experiments:

```bash
go test -bench . -benchtime 5x -count 6 | tee bench_before.txt
# edit experiment.go ...
go test -bench . -benchtime 5x -count 6 | tee bench_after.txt
benchstat bench_before.txt bench_after.txt
```

## Design choices

- **Go + purego, no CGo.** The entire stack is pure Go using [purego](https://github.com/ebitengine/purego) for Apple framework calls (Accelerate, ANE). No C compiler needed.
- **Fixed time budget.** Training always runs for exactly 5 minutes. This makes experiments directly comparable regardless of what the agent changes. Each experiment takes ~2-4 minutes to benchmark, so you get 15-30 experiments per hour, ~100-200 overnight.
- **Statistical rigor.** Go benchmarks + `benchstat` give p-values for every comparison. A change is only kept if it improves val_loss with p < 0.05.
- **ANE utilization tracking.** Benchmarks report real-time ANE power (watts), compute utilization (%), and energy consumption via [aneperf](https://github.com/tmc/aneperf).

## Model

110M-parameter Llama2-style transformer on TinyStories:

| | |
|---|---|
| Vocab | 32,000 (Llama2 BPE) |
| Dim | 768 |
| Hidden | 2,048 |
| Heads | 12 |
| Layers | 12 |
| Sequence length | 256 (default) |

## Credits

- [karpathy/autoresearch](https://github.com/karpathy/autoresearch) — the original autonomous research pattern
- [maderix/ANE](https://github.com/maderix/ANE) — Go Apple Neural Engine training (vendored in `ane/`)
- [tmc/apple](https://github.com/tmc/apple) — Go Apple platform bindings (Accelerate, CoreML, ANE)
- [tmc/aneperf](https://github.com/tmc/aneperf) — ANE performance monitoring

## License

MIT
