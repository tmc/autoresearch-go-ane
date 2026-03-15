# autoresearch-go-ane — Inference Speed Optimization

This is an experiment to have the LLM do its own ML research, optimizing **inference speed** (tokens/s) on Apple Neural Engine.

## Goal

**Maximize `BenchmarkEvalLogits` throughput (tokens/s).** The hot path is `EvalLogits` → `evalLogitsANEInto` in `ane/storiesane/engine.go`. Every optimization must preserve correctness (verified by `TestInferenceCorrectness`).

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar14-infer`). The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current main.
3. **Read the in-scope files**: The key files for inference optimization:
   - `ane/storiesane/engine.go` — `EvalLogits`, `evalLogitsANEInto` (the inference hot path)
   - `ane/storiesane/layer_dynamic_darwin.go` — layer forward with IOSurface staging
   - `ane/storiesane/fp16_pack_darwin.go` — fp16 conversion bottleneck
   - `ane/mil/stories_dynamic.go` — MIL program generators (ANE kernel structure)
   - `ane/storiesane/offload_darwin.go` — RMSNorm + classifier forward offload
   - `ane/dynamicmatmul/` — matrix multiply executors
   - `ane/storiesane/residual.go` — residual connections
   - `experiment.go` — hyperparameters and config
   - `bench_test.go` — benchmarks (read-only)
   - `correctness_test.go` — golden logits correctness oracle (read-only)
4. **Generate golden logits** (once): `go test -run TestSaveGoldenLogits -v`
5. **Install benchstat**: `go install golang.org/x/perf/cmd/benchstat@latest`
6. **Verify coordinator**: `python3 scripts/coordinator.py analyze`
7. **Confirm and go**: Confirm setup looks good.

## What You CAN Edit

- `experiment.go` — hyperparameters, config knobs
- Anything under `ane/` — the entire ANE package is in scope:
  - `ane/storiesane/` — engine, layers, fp16 packing, offload, residual, etc.
  - `ane/mil/` — MIL program generators (ANE kernel structure)
  - `ane/dynamicmatmul/` — matrix multiply executors
  - `ane/linear/` — linear layer execution
  - `ane/forward/` — forward pass helpers
  - `ane/iosurface/` — IOSurface handling
  - `ane/stories/` — model weights, CPU ops, config

## What You MUST NOT Modify

- `main.go` — fixed training loop, evaluation harness
- `bench_test.go` — benchmark definitions
- `correctness_test.go` — correctness oracle
- Public `Engine` API — `EvalLogits`, `Step`, `Open`, `Close` signatures must not change

## The Experiment Loop

**LOOP FOREVER:**

1. **RECALL**: `python3 scripts/coordinator.py analyze`
   - Review past experiments, insights, open hypotheses.

2. **CHECK PRIOR ART**: `python3 scripts/coordinator.py check_tried "<idea>"`
   - Don't repeat failed experiments.

3. **THINK**: Pick a hypothesis from open list, or propose a new approach.
   - ONE change per experiment. Isolate effects for clear causality.

4. **IMPLEMENT**: Edit `ane/` files and/or `experiment.go`.
   - Keep changes minimal and focused.

5. **VERIFY**: Compile and check correctness.
   ```bash
   go build . && go test -run TestInferenceCorrectness -v
   ```
   - If correctness fails → revert immediately, log as error.

6. **BENCHMARK**: Measure inference speed.
   ```bash
   go test -bench BenchmarkEvalLogits -benchtime 5x -count 6 | tee bench_after.txt
   ```

7. **COMPARE**: Statistical comparison.
   ```bash
   benchstat bench_before.txt bench_after.txt
   ```
   - Key metric: `BenchmarkEvalLogits` ns/op (lower = better).
   - Report as tokens/s: `tokens_per_sec = (256 * 1e9) / ns_per_op`

8. **PUBLISH**: Always publish, even negative results.
   ```bash
   python3 scripts/coordinator.py publish_result "<description>" '<bench_json>' "$(git diff HEAD~1)" "keep|discard|error"
   ```

9. **DECIDE**:
   - If faster with `p < 0.05`: keep commit, `mv bench_after.txt bench_before.txt`.
   - If equal or slower: `git reset --hard HEAD~1`, log as `discard`.

10. Go to step 1.

## Optimization Avenues

The inference hot path is `EvalLogits` → `evalLogitsANEInto`:

1. **IOSurface transfer** (`fp16_pack_darwin.go`): fp16 pack/unpack is per-channel — batch channels, reduce lock/unlock overhead.
2. **Kernel fusion** (`mil/stories_dynamic.go`): fuse attention+FFN into fewer ANE dispatches.
3. **Layer pipelining**: overlap layer N output read with layer N+1 input write.
4. **Classifier tiling** (`dynamicmatmul/`): optimize tile sizes for 32K vocab matmul.
5. **Residual ops** (`residual.go`): fuse CPU residual adds into ANE kernels.
6. **Embed lookup**: cache or precompute for inference.
7. **Static vs dynamic layers**: static compilation bakes weights, potentially faster inference.
8. **RMSNorm fusion**: fuse final RMSNorm into classifier kernel.
9. **Memory layout**: optimize tensor layouts for ANE's preferred access patterns.
10. **Compilation caching**: reduce layer compilation overhead.

## Correctness

Every change MUST pass `TestInferenceCorrectness`. This test compares `EvalLogits` output against golden reference logits with fp16 tolerance (~1e-3 relative error, 0.1% mismatch threshold).

- Golden logits are stored in `testdata/golden_logits.bin`
- Regenerate only if the CPU reference path changes: `go test -run TestSaveGoldenLogits -v`
- If an optimization changes numerical output beyond tolerance, it's a bug — revert.

## Benchmarking

**Capture baseline** (before changing anything):
```bash
go test -bench BenchmarkEvalLogits -benchtime 5x -count 6 | tee bench_before.txt
```

**After editing**, capture new results:
```bash
go test -bench BenchmarkEvalLogits -benchtime 5x -count 6 | tee bench_after.txt
```

**Compare:**
```bash
benchstat bench_before.txt bench_after.txt
```

**Quick wrapper** (includes correctness check):
```bash
bash scripts/bench_infer.sh
```

## Coordinator (Ensue)

The coordinator provides persistent memory across sessions.

```bash
# Full research summary
python3 scripts/coordinator.py analyze

# Check if something similar was tried
python3 scripts/coordinator.py check_tried "batch IOSurface lock/unlock"

# Publish experiment result
python3 scripts/coordinator.py publish_result "batched fp16 packing" '{"tokens_per_sec": 21000, "ns_per_op": 12190}' "$(git diff HEAD~1)" "keep"

# Record an insight
python3 scripts/coordinator.py post_insight "IOSurface lock/unlock accounts for 15% of layer time"

# Add hypothesis to try
python3 scripts/coordinator.py publish_hypothesis "Fuse RMSNorm into classifier" "Combine final RMSNorm and classifier matmul into single ANE dispatch" 1

# Get current best
python3 scripts/coordinator.py pull_best

# Semantic search
python3 scripts/coordinator.py ask "fp16 packing bottleneck"
```

## Logging Results

Log results to `results.tsv` (tab-separated):

```
commit	tokens_per_sec	ns_per_op	status	description
a1b2c3d	20736.5	12345678	keep	baseline
b2c3d4e	22500.0	11377778	keep	batched IOSurface transfers
c3d4e5f	19800.0	12929293	discard	fused attention kernel (slower)
```

NOTE: do not commit `results.tsv` — leave it untracked by git.

## Protocol Rules

- **ONE change per experiment**: Isolate effects. Don't combine multiple optimizations.
- **ALWAYS verify correctness**: Run `TestInferenceCorrectness` after every change.
- **ALWAYS publish**: Even negative results prevent re-trying failed approaches.
- **Simplicity criterion**: Simpler is better. A small speedup that adds ugly complexity is not worth it.
- **NEVER STOP**: Once the loop begins, do NOT pause to ask. The human may be asleep. Loop until manually interrupted.
- **If correctness fails → revert immediately**: `git reset --hard HEAD~1`, log as error.
- **Crashes**: Fix if trivial, skip if fundamentally broken. Always log.

## Model Architecture (Fixed)

- 110M parameters, Llama2-style transformer
- Vocab: 32,000 (Llama2 BPE)
- Dim: 768, Hidden: 2,048, Heads: 12, Layers: 12
- Default sequence length: 256
