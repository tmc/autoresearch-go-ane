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
   - `ane/train_util.go` — cross-entropy loss, RMS norm gradients, RoPE. Editable.
   - `ane/common.go` — CPU primitives (linear, attention, softmax, reluSquared, QK-norm, logit softcap, smear, backout), grad task concurrency. Editable.
   - `ane/accel_darwin.go` — Accelerate BLAS wrappers for GEMM. Editable.
   - `ane/accel_stub.go` — pure Go fallbacks for non-darwin. Keep in sync with accel_darwin.go.
   - `ane/engine.go` — Engine struct, Open, Step, EvalLogits, Close. Editable with care.
   - `ane/stories/cpu.go` — low-level CPU kernels (embed, matmul, softmax, RMS norm). Editable.
4. **Verify data exists**: Check that `tinystories_data00.bin` exists. If not, tell the human to run `bash scripts/setup.sh`.
5. **Install benchstat**: `go install golang.org/x/perf/cmd/benchstat@latest`
6. **Build bench-note**: `go build -o bench-note ./cmd/bench-note/`
7. **Confirm and go**: Confirm setup looks good.

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
- `ane/train_util.go` — loss computation, RMS norm gradients, RoPE
- `ane/common.go` — CPU primitives (linearCF, causalAttentionCF, softmax, reluSquared, QK-norm, logit softcap, smear, backout), grad concurrency
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

**The goal is simple: get the lowest val_bpb (bits per byte).** The time budget is fixed at 5 minutes. Everything in the editable files is fair game.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training as is.

## Benchmarking with bench-note

Use `bench-note` (`cmd/bench-note`) to run benchmarks, attach results to git commits as notes, and compare across commits. Results are stored in `refs/notes/benchmarks` using txtar format.

**Build bench-note** (once per session):
```bash
go build -o bench-note ./cmd/bench-note/
```

**Run benchmarks and attach to HEAD**:
```bash
./bench-note run --benchtime=5x --count=6
```

This runs `go test -bench`, attaches the output as a git note to HEAD, and automatically runs benchstat against the nearest ancestor that has a bench note.

**Attach existing output** (e.g. from a `tee` file):
```bash
./bench-note run --from-file=bench_after.txt --benchtime=5x --count=6
```

**View results**:
```bash
./bench-note show           # full txtar note for HEAD
./bench-note show abc1234   # for a specific commit
./bench-note raw            # just the raw go test output (for piping)
```

**Compare two commits**:
```bash
./bench-note compare abc1234 def5678
```

**List all commits with bench notes**:
```bash
./bench-note history            # detailed
./bench-note history --oneline  # compact
```

### Key metrics

- `BenchmarkEvalLoss` `val_bpb` — **this is the metric you are optimizing** (bits per byte, lower is better, vocab-size-independent)
- `BenchmarkEvalLoss` `val_loss` — cross-entropy in nats (equivalent to val_bpb but vocab-dependent)
- `BenchmarkStep` `loss` — training loss trajectory (should decrease over steps)
- `BenchmarkStep` `tok/s` — training throughput (tokens per second)
- `BenchmarkStep` `ane-compute-%` — ANE utilization during training
- `BenchmarkEvalLogits` `ns/op` — inference latency per 256-token window

A change is worth keeping if `val_bpb` decreased with `p < 0.05`. If benchstat shows `~` (no significant difference), the change has no effect — discard it.

The `-count 6` flag runs each benchmark 6 times for meaningful statistics. Use `-benchtime 3x` for faster exploration, `-benchtime 10x` for precise final measurements.

## Logging results

The primary benchmark record lives in git notes (`refs/notes/benchmarks`), attached by `bench-note run`. Use `bench-note history` to review the full history with raw output and benchstat deltas.

Additionally, log a summary to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 7 columns:

```
commit	val_bpb	val_loss	steps	train_secs	status	description
```

1. git commit hash (short, 7 chars)
2. val_bpb achieved (e.g. 1.234567) — use 0.000000 for crashes
3. val_loss in nats (e.g. 3.456789) — use 0.000000 for crashes
4. number of training steps completed
5. training wall clock seconds
6. status: `keep`, `discard`, or `crash`
7. short text description of what this experiment tried

Example:

```
commit	val_bpb	val_loss	steps	train_secs	status	description
a1b2c3d	1.345678	3.456789	1500	300.1	keep	baseline
b2c3d4e	1.328765	3.412345	1520	300.3	keep	increase LR to 1e-3
c3d4e5f	1.363210	3.501234	1480	300.0	discard	switch to linear decay
d4e5f6g	0.000000	0.000000	0	0.0	crash	sequence length 1024 (OOM)
```

NOTE: do not commit `results.tsv` — leave it untracked by git.

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar14`).

LOOP FOREVER:

1. Edit files with an experimental idea (see editable files above).
2. Verify it compiles: `go test -c -o /dev/null .`
3. git commit: `git add -A && git commit -m "<description>"`
4. Run benchmarks and attach as git note: `./bench-note run --benchtime=5x --count=6`
   - This runs benchmarks, attaches results to HEAD, and auto-compares against the nearest ancestor with a bench note.
5. Review the benchstat delta: `./bench-note show`
6. If `val_bpb` improved (decreased) with statistical significance:
   - You "advance" the branch, keeping the git commit.
   - Log results to `results.tsv`.
7. If `val_bpb` is equal or worse:
   - `git reset --hard HEAD~1` to revert.
   - Log results to `results.tsv` with status `discard`.
8. If the build or run crashed:
   - If it's something easy to fix (typo, bad constant), fix and re-run.
   - If the idea is fundamentally broken, `git reset --hard HEAD~1`, log `crash`, move on.
9. Go to step 1.

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
- `EmbedLRMult` — LR multiplier for embedding params (Embed, VEEmbed)
- `ScalarLRMult` — LR multiplier for scalar params (VEGate, SmearLambda, BackoutLambda)
- `LambdaLRMult` — LR multiplier for lambda params, relative to scalar LR (default 0.01)
- `LambdaBeta1`, `LambdaBeta2` — custom Adam betas for lambda params (default 0.96, 0.95)

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
- `forwardTrainingCPU` — the full forward pass (nanochat-style). You can:
  - Tune residual lambda init values (in stories/weights.go RandomInit)
  - Change smear/backout lambda init values
  - Modify QK-norm scale factor (currently 1.2)
  - Adjust logit softcap temperature (currently 15.0)
  - Change VE gating formula
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
- `reluSquared32` — the ReLU² activation (`max(0,x)²`). You can try GeLU, SiLU, or other activations.
  Note: changing this affects both forward and backward — update `reluSquaredBackwardAccel` too.

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
2. **Residual lambda tuning**: Adjust init values for `ResidLambdas`/`X0Lambdas` in RandomInit
3. **Optimizer tweaks**: Gradient centralization, different weight decay formulations
4. **Activation function**: Replace ReLU² with GeLU or other activations
5. **Selective layer freezing**: Skip gradient updates for early layers
6. **Gradient noise**: Add noise before optimizer update for regularization
7. **Per-layer learning rates**: Different LR for attention vs FFN weights
8. **QK-norm scale**: Adjust the 1.2 scale factor in `qkNormCF`
9. **Logit softcap temperature**: Adjust the 15.0 cap in `logitSoftcap`
10. **Smear/Backout lambda init**: Tune initial values for smear and backout lambdas

## Model architecture

The model is a 110M-parameter nanochat-style transformer (based on karpathy/nanochat) trained on TinyStories:
- Vocab: 32,000 (Llama2 BPE tokenizer)
- Dim: 768
- Hidden: 2,048
- Heads: 12
- Layers: 12
- Default sequence length: 256

### Architecture details (nanochat features)

- **ReLU² activation**: FFN uses `relu(x)²` instead of SiLU
- **Parameterless RMSNorm**: All norms have no learnable per-channel weights
- **QK-norm**: Per-head RMSNorm on Q,K after RoPE, scaled by 1.2; attention scale = 1.0
- **Embedding norm**: Parameterless RMSNorm after token embedding lookup
- **Per-layer residual lambdas**: `cur = resid_lambdas[i]*cur + x0_lambdas[i]*x0` before each layer
- **Value embeddings (VE)**: Alternating-layer (even) VE tables with gated V mixing
- **Smear**: Bigram mixing (gated shift) after embedding norm
- **Backout**: Subtract `backout_lambda * mid_layer_residual` before final norm
- **Logit softcap**: `15 * tanh(logits/15)` before loss and eval
- **Checkpoint format**: V4 (backward compatible with V3 for loading old weights)

## Collaborative mode

For multi-agent swarm optimization via Ensue, see `.skills/autoresearch/SKILL.md` and `collab.md`.
