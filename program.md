# autoresearch-go-ane

This is an experiment to have the LLM do its own ML research, running training experiments on Apple Neural Engine.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar14`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current main.
3. **Read the in-scope files**: Read these files for full context:
   - `experiment.go` — primary experiment file. Hyperparameters, LR schedule, training config.
   - `harness.go` — evaluation harness, data loading, random init. Do not modify.
   - `bench_test.go` — Go benchmarks for measuring step throughput, eval loss, etc. Do not modify.
   - `ane/train_full.go` — forward pass, backward pass, Adam optimizer, gradient clipping. Editable.
   - `ane/train_util.go` — cross-entropy loss, RMS norm gradients, residual scaling, RoPE. Editable.
   - `ane/common.go` — CPU primitives (linear, attention, softmax, silu), grad task concurrency. Editable.
   - `ane/accel_darwin.go` — Accelerate BLAS wrappers for GEMM. Editable.
   - `ane/accel_stub.go` — pure Go fallbacks for non-darwin. Keep in sync with accel_darwin.go.
   - `ane/engine.go` — Engine struct, Open, Step, EvalLogits, Close. Editable with care.
   - `ane/stories/cpu.go` — low-level CPU kernels (embed, matmul, softmax, RMS norm). Editable.
4. **Verify data exists**: Check that `tinystories_data00.bin` exists. If not, tell the human to run `bash scripts/setup.sh`.
5. **Install benchstat**: `go install golang.org/x/perf/cmd/benchstat@latest`
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment trains on Apple Neural Engine for a **fixed time budget of 5 minutes** (wall clock training time, excluding the first 3 warmup/compilation steps).

### Editable files

You have two tiers of editable files:

**Tier 1 — Primary experiment surface** (`experiment.go`):
- Hyperparameters, LR schedule, optimizer config, `experimentConfig()`.
- This is the fastest iteration loop — change constants, rebuild, benchmark.

**Tier 2 — Training internals** (`ane/` package):
- `ane/train_full.go` — forward/backward pass, Adam optimizer, gradient accumulation
- `ane/train_util.go` — loss computation, RMS norm gradients, residual scaling, RoPE
- `ane/common.go` — CPU primitives (linearCF, causalAttentionCF, softmax, silu), grad concurrency
- `ane/accel_darwin.go` / `ane/accel_stub.go` — Accelerate BLAS wrappers
- `ane/engine.go` — Engine struct, Step logic, data sampling
- `ane/stories/cpu.go` — low-level CPU kernels

These are more impactful but riskier. Changes here can affect correctness, so verify carefully.

### Read-only files

- `harness.go` — evaluation harness (`evalLoss`), data loading. The ground truth metric.
- `bench_test.go` — benchmark harness.
- `ane/runtime.go` — ANE kernel compilation and weight refresh.
- `ane/layer_darwin.go`, `ane/backward_darwin.go` — ANE MIL kernel dispatch.
- `ane/offload_darwin.go` — ANE offload (RMS norm, classifier, softmax kernels).

**The goal is simple: get the lowest val_loss.** The time budget is fixed at 5 minutes. Everything in the editable files is fair game.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training as is.

## Benchmarking with benchstat

Use Go benchmarks + `benchstat` to measure the impact of each change with statistical rigor. This is the primary way to evaluate experiments.

**Capture a baseline** (before changing anything):
```bash
go test -bench . -benchtime 5x -count 6 | tee bench_before.txt
```

**After editing**, capture the new results:
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
2. Edit files with an experimental idea (see editable files above).
3. Verify it compiles: `go test -c -o /dev/null .`
4. git commit: `git add -A && git commit -m "<description>"`
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

**NEVER STOP**: Once the experiment loop has begun, do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep or away from the computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — re-read the editable files for new angles, try combining previous near-misses, try more radical changes. The loop runs until the human interrupts you, period.

As a use case: the user might leave you running while they sleep. If each experiment takes ~2-4 minutes then you can run 15-30 per hour, for a total of about 100-200 over a night. The user wakes up to experimental results, all completed by you while they slept.

## What you can change

### Tier 1: experiment.go (fast iteration)

#### Constants
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

#### Learning rate schedule
The `lrSchedule(progress float64) float64` function controls how learning rate varies during training. `progress` goes from 0.0 to 1.0 over the training budget.

Default: linear warmup (5%) then cosine decay to 10% of peak.

You can try:
- Different warmup fractions
- Different minimum LR ratios
- Linear decay, polynomial decay, step schedules
- Warm restarts

#### Adding new code
You can add new functions, constants, or logic to `experiment.go`. The `experimentConfig()` function builds the `ane.Options` struct — you can modify how it's constructed.

### Tier 2: ane/ package (deeper changes)

#### Optimizer (ane/train_full.go)
- `adamUpdateCFWithInv` — the core Adam update loop. You can try:
  - AdamW variants, different weight decay formulations
  - Gradient centralization
  - Momentum reset or warm restart
  - Alternative optimizers (Lion, LAMB, etc.)
- `clipLayerGradients` — gradient clipping. You can try per-layer clipping, different norms.
- `flushPending` — gradient accumulation flush. You can modify the scaling or add gradient noise.

#### Forward pass (ane/train_full.go)
- `forwardTrainingCPU` — the full forward pass. The architecture is fixed (Llama2-style), but you can:
  - Modify residual scaling (`layerResidualScale` in train_util.go)
  - Change how residual connections combine
  - Add or remove layer operations
- `causalAttentionCF` in common.go — the attention implementation. You can try:
  - Different attention scaling
  - Attention temperature
  - Modified softmax (log-softmax, etc.)

#### Backward pass (ane/train_full.go)
- `backwardAndAccumulate` / `backwardAndApply` — the backward pass. You can:
  - Add gradient noise for regularization
  - Implement selective gradient freezing (freeze some layers)
  - Change gradient scaling per layer
- `backwardFFNCPU` / `backwardAttentionCPU` — individual backward operations

#### Loss function (ane/train_util.go)
- `crossEntropyLossFromProbs` — the loss computation. You can try:
  - Label smoothing
  - Focal loss
  - z-loss regularization (penalize large logits)

#### Activation functions (ane/common.go)
- `silu32` — the SiLU/Swish activation. You can try GeLU, ReLU, or other activations.
  Note: changing this affects both forward and backward — update `siluBackward` too.

#### CPU primitives (ane/stories/cpu.go)
- `EmbedLookup`, `RMSNorm`, `CrossEntropyLoss`, `MatMulVocabSeq`, etc.
- These are the low-level kernels. Changes here can affect numerical accuracy.

#### Important constraints for Tier 2 changes
- **Keep accel_stub.go in sync** with accel_darwin.go — both must have the same function signatures.
- **Don't break the Engine API** — `Open`, `Step`, `EvalLogits`, `Close` must keep their signatures.
- **Don't modify ANE kernel compilation** (runtime.go, layer_darwin.go, backward_darwin.go) — these are complex and fragile.
- **Test carefully** — a wrong gradient formula will silently produce garbage loss values. If val_loss explodes after a Tier 2 change, revert immediately.

## ANE-specific notes

- `UseANE=true` offloads forward passes to the Neural Engine. Usually faster but adds compilation overhead on the first few steps.
- `HybridBackward=true` uses ANE for backward dx propagation. Can be faster or slower depending on sequence length.
- The first 3 training steps are warmup (excluded from the time budget) to absorb ANE compilation cost.
- If ANE compilation fails or produces NaN, try `UseANE=false` as a fallback.

## Parameter space guidance

Good starting experiments (Tier 1 — fast):
1. **Learning rate sweep**: Try 1e-4, 3e-4, 1e-3, 3e-3
2. **Accumulation steps**: Try 1, 2, 4, 8 (more steps = larger effective batch)
3. **Sequence length**: Try 128, 256, 512 (longer = more context but slower)
4. **Weight decay**: Try 0.0, 0.01, 0.1
5. **Warmup fraction**: Try 0.01, 0.05, 0.1, 0.2
6. **LR schedule**: Cosine vs linear decay, different min LR ratios
7. **Gradient clipping**: Try 0.5, 1.0, 2.0, or disable (999.0)

Deeper experiments (Tier 2 — more impactful, more risk):
1. **Label smoothing**: Add to `crossEntropyLossFromProbsRange` in train_util.go
2. **Residual scaling**: Change `layerResidualScale` or the blending formula
3. **Optimizer tweaks**: Gradient centralization, different weight decay formulations
4. **Activation function**: Replace SiLU with GeLU or other activations
5. **Selective layer freezing**: Skip gradient updates for early layers
6. **Gradient noise**: Add noise before optimizer update for regularization
7. **Per-layer learning rates**: Different LR for attention vs FFN weights

## Model architecture

The model is a 110M-parameter Llama2-style transformer trained on TinyStories:
- Vocab: 32,000 (Llama2 BPE tokenizer)
- Dim: 768
- Hidden: 2,048
- Heads: 12
- Layers: 12
- Default sequence length: 256
