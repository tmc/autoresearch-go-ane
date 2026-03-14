# autoresearch-go-ane

This is an experiment to have the LLM do its own ML research, running training experiments on Apple Neural Engine.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar14`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current main.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `main.go` — fixed training loop, evaluation harness, data loading, results logging. Do not modify.
   - `experiment.go` — the file you modify. Hyperparameters, learning rate schedule, training configuration.
   - `bench_test.go` — Go benchmarks for measuring step throughput, eval loss, etc. Do not modify.
4. **Verify data exists**: Check that `tinystories_data00.bin` exists. If not, tell the human to run `bash scripts/setup.sh`.
5. **Install benchstat**: `go install golang.org/x/perf/cmd/benchstat@latest`
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment trains on Apple Neural Engine for a **fixed time budget of 5 minutes** (wall clock training time, excluding the first 3 warmup/compilation steps).

**What you CAN do:**
- Modify `experiment.go` — this is the only file you edit. Everything is fair game: hyperparameters, learning rate schedule, optimizer config, gradient accumulation, sequence length, ANE settings.

**What you CANNOT do:**
- Modify `main.go`. It is read-only. It contains the fixed training loop, evaluation harness, data loading, and time budget.
- Modify `bench_test.go`. It is read-only.
- Add new Go dependencies. You can only use what's already imported.
- Modify the evaluation harness. The `evalLoss` function in `main.go` is the ground truth metric.

**The goal is simple: get the lowest val_loss.** Since the time budget is fixed, you don't need to worry about training time — it's always 5 minutes. Everything in `experiment.go` is fair game: change the learning rate, the schedule, the accumulation steps, the sequence length, the optimizer params, the ANE settings. The only constraint is that the code compiles and finishes within the time budget.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training as is.

## Output format

The training run prints a summary like this:

```
val_loss:         3.456789
training_seconds: 300.1
steps:            1500
sequence_length:  256
accum_steps:      4
learning_rate:    3.0e-04
use_ane:          true
hybrid_backward:  true
seed:             42
```

Extract the key metric:

```bash
grep "^val_loss:" run.log
```

## Benchmarking with benchstat

Use Go benchmarks + `benchstat` to measure the impact of each change with statistical rigor. This is the primary way to evaluate experiments.

**Capture a baseline** (before changing anything):
```bash
go test -bench . -benchtime 5x -count 6 | tee bench_before.txt
```

**After editing `experiment.go`**, capture the new results:
```bash
go test -bench . -benchtime 5x -count 6 | tee bench_after.txt
```

**Compare:**
```bash
benchstat bench_before.txt bench_after.txt
```

This shows per-benchmark differences with p-values. The key metrics:
- `BenchmarkEvalLoss` `val_loss` — **this is the metric you are optimizing**
- `BenchmarkStep` `loss` and `step_ms` — training loss trajectory and throughput
- `BenchmarkEvalLogits` — inference speed

A change is worth keeping if `val_loss` decreased with `p < 0.05`. If benchstat shows `~` (no significant difference), the change has no effect — discard it.

The `-count 6` flag runs each benchmark 6 times for meaningful statistics. Use `-benchtime 3x` for faster exploration, `-benchtime 10x` for precise final measurements.

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 6 columns:

```
commit	val_loss	steps	train_secs	status	description
```

1. git commit hash (short, 7 chars)
2. val_loss achieved (e.g. 3.456789) — use 0.000000 for crashes
3. number of training steps completed
4. training wall clock seconds
5. status: `keep`, `discard`, or `crash`
6. short text description of what this experiment tried

Example:

```
commit	val_loss	steps	train_secs	status	description
a1b2c3d	3.456789	1500	300.1	keep	baseline
b2c3d4e	3.412345	1520	300.3	keep	increase LR to 1e-3
c3d4e5f	3.501234	1480	300.0	discard	switch to linear decay
d4e5f6g	0.000000	0	0.0	crash	sequence length 1024 (OOM)
```

NOTE: do not commit `results.tsv` — leave it untracked by git.

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar14`).

LOOP FOREVER:

1. Capture baseline benchmarks: `go test -bench . -benchtime 5x -count 6 | tee bench_before.txt`
2. Edit `experiment.go` with an experimental idea.
3. Verify it compiles: `go build .`
4. git commit: `git add experiment.go && git commit -m "<description>"`
5. Run benchmarks: `go test -bench . -benchtime 5x -count 6 | tee bench_after.txt`
6. Compare: `benchstat bench_before.txt bench_after.txt`
7. If `val_loss` improved (decreased) with statistical significance:
   - You "advance" the branch, keeping the git commit.
   - `mv bench_after.txt bench_before.txt` (new baseline).
   - Log results to `results.tsv`.
8. If `val_loss` is equal or worse:
   - `git reset --hard HEAD~1` to revert.
   - Log results to `results.tsv` with status `discard`.
9. If the build or run crashed:
   - If it's something easy to fix (typo, bad constant), fix and re-run.
   - If the idea is fundamentally broken, `git reset --hard HEAD~1`, log `crash`, move on.
10. Go to step 2.

**Timeout**: Each benchmark run should take ~2 minutes for `BenchmarkStep` + `BenchmarkEvalLoss`. If a run exceeds 10 minutes, kill it and treat it as a failure.

**Crashes**: Use your judgment. If it's a dumb mistake (e.g. a constant out of range), fix it. If the idea itself is broken (e.g. sequence length too large for ANE), skip it and move on.

**NEVER STOP**: Once the experiment loop has begun, do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep or away from the computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — re-read `experiment.go` and `main.go` for new angles, try combining previous near-misses, try more radical parameter changes. The loop runs until the human interrupts you, period.

As a use case: the user might leave you running while they sleep. If each experiment takes ~2-4 minutes then you can run 15-30 per hour, for a total of about 100-200 over a night. The user wakes up to experimental results, all completed by you while they slept.

## What you can change in experiment.go

### Constants
- `SequenceLength` — input sequence length (default 256, max ~512)
- `AccumSteps` — gradient accumulation steps (effective batch = AccumSteps * SeqLen tokens)
- `LearningRate` — peak learning rate
- `AdamBeta1`, `AdamBeta2`, `AdamEps` — Adam optimizer parameters
- `WeightDecay` — L2 regularization
- `GradClip` — gradient clipping threshold
- `LossScale` — loss scaling for mixed precision
- `UseANE` — enable Apple Neural Engine acceleration
- `HybridBackward` — enable ANE backward pass
- `Seed` — random seed for initialization

### Learning rate schedule
The `lrSchedule(progress float64) float64` function controls how learning rate varies during training. `progress` goes from 0.0 to 1.0 over the training budget.

Default: linear warmup (5%) then cosine decay to 10% of peak.

You can try:
- Different warmup fractions
- Different minimum LR ratios
- Linear decay, polynomial decay, step schedules
- Warm restarts

### Adding new code
You can add new functions, constants, or logic to `experiment.go`. The `experimentConfig()` function builds the `storiesane.Options` struct — you can modify how it's constructed.

## ANE-specific notes

- `UseANE=true` offloads forward passes to the Neural Engine. Usually faster but adds compilation overhead on the first few steps.
- `HybridBackward=true` uses ANE for backward dx propagation. Can be faster or slower depending on sequence length.
- The first 3 training steps are warmup (excluded from the time budget) to absorb ANE compilation cost.
- If ANE compilation fails or produces NaN, try `UseANE=false` as a fallback.

## Parameter space guidance

Good starting experiments:
1. **Learning rate sweep**: Try 1e-4, 3e-4, 1e-3, 3e-3
2. **Accumulation steps**: Try 1, 2, 4, 8 (more steps = larger effective batch)
3. **Sequence length**: Try 128, 256, 512 (longer = more context but slower)
4. **Weight decay**: Try 0.0, 0.01, 0.1
5. **Warmup fraction**: Try 0.01, 0.05, 0.1, 0.2
6. **LR schedule**: Cosine vs linear decay, different min LR ratios
7. **Gradient clipping**: Try 0.5, 1.0, 2.0, or disable (999.0)

## Model architecture

The model is a 110M-parameter Llama2-style transformer trained on TinyStories:
- Vocab: 32,000 (Llama2 BPE tokenizer)
- Dim: 768
- Hidden: 2,048
- Heads: 12
- Layers: 12
- Default sequence length: 256
