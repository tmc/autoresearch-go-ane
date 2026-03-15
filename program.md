# autoresearch-go-ane

This is an experiment to have the LLM do its own ML research, optimizing inference speed on Apple Neural Engine.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar15`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current main.
3. **Read the in-scope files**: Read these files for full context:
   - `experiment.go` — primary experiment file. Inference config, dispatch strategy, tuning knobs.
   - `bench_test.go` — Go benchmarks for measuring inference throughput. Do not modify.
   - `correctness_test.go` — Correctness oracle. Do not modify.
   - `ane/engine.go` — Engine struct, Open, Step, EvalLogits, Close. Editable with care.
   - `ane/eval_cached.go` — EvalNextToken (KV-cached), TokenTimings. Editable.
   - `ane/accel_darwin.go` — Accelerate BLAS wrappers (sgemm, sgemv, siluMul). Editable.
   - `ane/bnns_darwin.go` — BNNS fp16-weight GEMV. Editable.
   - `ane/metal_darwin.go` — Metal MPS GPU matmul. Editable.
   - `ane/fp16_pack_darwin.go` — NEON fp16↔fp32 conversion. Editable.
   - `ane/kvcache.go` — KV cache for incremental generation. Editable.
   - `ane/layer_tiled_darwin.go` — Tiled ANE layer forward for large models. Editable.
   - `ane/storiesane/metal_gemv_darwin.go` — Custom Metal compute shader for fp16 matvec. Editable.
4. **Verify model exists**: Check that a model `.bin` file is available (e.g. `stories110M.bin` or `qwen3-0.6b.bin`). If not, run `bash scripts/setup.sh` or `python3 scripts/convert_hf.py Qwen/Qwen3-0.6B --output qwen3-0.6b.bin`.
5. **Install benchstat**: `go install golang.org/x/perf/cmd/benchstat@latest`
6. **Build bench-note**: `go build -o bench-note ./cmd/bench-note/`
7. **Verify ensue connectivity**: `python3 scripts/coordinator.py analyze`
8. **Check for API key**: Ensure `ENSUE_API_KEY` env or `.autoresearch-key` file exists.
9. **Install requests if needed**: `uv pip install requests`
10. **Review prior work**: `python3 scripts/coordinator.py pull_best` to see current best result.
11. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs a **fixed inference benchmark**: `BenchmarkEvalLogits` with `-benchtime 5x -count 6`. A single benchmark run takes ~1-2 minutes.

### Editable files

You have two tiers of editable files:

**Tier 1 — Primary experiment surface** (`experiment.go`):
- Inference configuration: dispatch strategy, acceleration flags, buffer sizes.
- This is the fastest iteration loop — change constants, rebuild, benchmark.

**Tier 2 — Inference internals** (`ane/` package):
- `ane/eval_cached.go` — KV-cached inference forward pass, token timing breakdown
- `ane/accel_darwin.go` — Accelerate BLAS wrappers (sgemm, sgemv, siluMul)
- `ane/bnns_darwin.go` — BNNS fp16-weight GEMV
- `ane/metal_darwin.go` — Metal MPS GPU matmul
- `ane/fp16_pack_darwin.go` — NEON fp16↔fp32 conversion
- `ane/kvcache.go` — KV cache layout and memory access
- `ane/layer_tiled_darwin.go` — Tiled ANE layer forward for large models
- `ane/storiesane/metal_gemv_darwin.go` — Custom Metal compute shader
- `ane/engine.go` — Engine struct, Step logic

These are more impactful but riskier. Changes here can affect correctness, so verify carefully.

### Read-only files

- `bench_test.go` — benchmark harness.
- `correctness_test.go` — correctness oracle.
- `main.go` — top-level harness.
- `ane/runtime.go` — ANE kernel compilation and weight refresh.
- `ane/layer_darwin.go`, `ane/backward_darwin.go` — ANE MIL kernel dispatch.
- `ane/offload_darwin.go` — ANE offload (RMS norm, classifier, softmax kernels).

**The goal is simple: get the highest `tokens_per_sec` (tokens per second, higher is better).** Everything in the editable files is fair game.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win.

**Correctness gate**: The correctness oracle (`go test -run TestInferenceCorrectness`) MUST pass before benchmarking. If it fails, the change broke inference — fix or revert.

**The first run**: Your very first run should always be to establish the baseline, so you will run the benchmark as-is.

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

- `BenchmarkEvalLogits` `ns/op` — inference latency per 256-token window (lower is better)
- `BenchmarkEvalLogits` `tokens_per_sec` — **this is the metric you are optimizing** (higher is better, derived: `256 * 1e9 / ns_per_op`)

A change is worth keeping if `tokens_per_sec` increased (i.e. `ns/op` decreased) with `p < 0.05`. If benchstat shows `~` (no significant difference), the change has no effect — discard it.

The `-count 6` flag runs each benchmark 6 times for meaningful statistics. Use `-benchtime 3x` for faster exploration, `-benchtime 10x` for precise final measurements.

## Ensue integration

The ensue.dev coordinator (`scripts/coordinator.py`) provides persistent memory across sessions for tracking experiments, insights, and the current best configuration.

**Key commands**:
```bash
# Review all prior work
python3 scripts/coordinator.py analyze

# Check if an idea was already tried
python3 scripts/coordinator.py check_tried '<idea>'

# Publish a result (after benchmarking)
python3 scripts/coordinator.py publish_result '<desc>' '<bench_json>' '<git_diff>' <status>

# Post a research insight
python3 scripts/coordinator.py post_insight '<learning>'

# Publish a hypothesis for future experiments
python3 scripts/coordinator.py publish_hypothesis '<title>' '<hypothesis>' [priority]

# See current best
python3 scripts/coordinator.py pull_best

# Semantic search over all data
python3 scripts/coordinator.py ask '<query>'
```

The `scripts/parse_bench.py` script converts raw Go benchmark output to JSON:
```bash
go test -bench BenchmarkEvalLogits -benchtime 5x -count 6 | python3 scripts/parse_bench.py
```

Output: `{"benchmark": "BenchmarkEvalLogits", "ns_per_op": ..., "tokens_per_sec": ..., "runs": ...}`

**Convenience wrapper** — run, parse, and publish in one step:
```bash
bash scripts/run_and_publish.sh '<description>' [keep|discard|crash]
```

## Logging results

The primary benchmark record lives in git notes (`refs/notes/benchmarks`), attached by `bench-note run`. Use `bench-note history` to review the full history with raw output and benchstat deltas.

Additionally, log a summary to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 6 columns:

```
commit	tokens_per_sec	ns_per_op	runs	status	description
```

1. git commit hash (short, 7 chars)
2. tokens_per_sec achieved (e.g. 46.5) — use 0.0 for crashes
3. ns_per_op median (e.g. 5505376344) — use 0 for crashes
4. number of benchmark runs
5. status: `keep`, `discard`, or `crash`
6. short text description of what this experiment tried

Example:

```
commit	tokens_per_sec	ns_per_op	runs	status	description
a1b2c3d	46.5	5505376344	6	keep	baseline
b2c3d4e	48.2	5311203320	6	keep	switch QKV to BNNS fp16
c3d4e5f	45.9	5577342047	6	discard	Metal GPU for attention
d4e5f6g	0.0	0	0	crash	tiled layer size too large (OOM)
```

NOTE: do not commit `results.tsv` — leave it untracked by git.

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar15`).

LOOP FOREVER:

0. Check if idea was already tried: `python3 scripts/coordinator.py check_tried '<idea>'`
1. Edit files with an experimental idea (see editable files above).
2. Verify correctness: `go test -run TestInferenceCorrectness`
3. Verify it compiles: `go test -c -o /dev/null .`
4. git commit: `git add -A && git commit -m "<description>"`
5. Run benchmarks and attach as git note: `./bench-note run --benchtime=5x --count=6`
   - This runs benchmarks, attaches results to HEAD, and auto-compares against the nearest ancestor with a bench note.
   - Also pipe raw output through parse_bench: `./bench-note raw | python3 scripts/parse_bench.py > /tmp/bench_result.json`
6. Review the benchstat delta: `./bench-note show`
7. If `tokens_per_sec` improved (increased) with statistical significance:
   - You "advance" the branch, keeping the git commit.
   - Log results to `results.tsv`.
   - Publish to ensue: `python3 scripts/coordinator.py publish_result '<desc>' "$(cat /tmp/bench_result.json)" "$(git diff HEAD~1)" keep`
8. If `tokens_per_sec` is equal or worse:
   - `git reset --hard HEAD~1` to revert.
   - Log results to `results.tsv` with status `discard`.
   - Publish to ensue: `python3 scripts/coordinator.py publish_result '<desc>' "$(cat /tmp/bench_result.json)" '' discard`
9. If the build or run crashed:
   - If it's something easy to fix (typo, bad constant), fix and re-run.
   - If the idea is fundamentally broken, `git reset --hard HEAD~1`, log `crash`, move on.
   - Publish to ensue: `python3 scripts/coordinator.py publish_result '<desc>' '{"error":"<reason>"}' '' crash`
10. Post an insight after every experiment: `python3 scripts/coordinator.py post_insight '<learning>'`
11. Publish a hypothesis for the next experiment: `python3 scripts/coordinator.py publish_hypothesis '<title>' '<hypothesis>'`
12. Go to step 0.

**Timeout**: Each benchmark run should take ~1-2 minutes. If a run exceeds 5 minutes, kill it and treat it as a failure.

**Crashes**: Use your judgment. If it's a dumb mistake (e.g. a constant out of range), fix it. If the idea itself is broken (e.g. tiled layer size too large for ANE), skip it and move on.

**NEVER STOP**: Once the experiment loop has begun, do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep or away from the computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — re-read the editable files for new angles, try combining previous near-misses, try more radical changes. Check ensue for insights and hypotheses: `python3 scripts/coordinator.py analyze`. The loop runs until the human interrupts you, period.

As a use case: the user might leave you running while they sleep. If each experiment takes ~2-3 minutes then you can run 20-30 per hour, for a total of about 150-200 over a night. The user wakes up to experimental results, all completed by you while they slept.

## What you can change

### Tier 1: experiment.go (fast iteration)

#### Constants
- `UseMetal` — enable Metal GPU matmul for large layers
- `UseBNNS` — enable BNNS fp16-weight GEMV
- `UseANE` — enable Apple Neural Engine acceleration
- `TileSize` — tile dimensions for tiled ANE layers
- `PreallocBuffers` — pre-allocate scratch buffers at init

#### Adding new code
You can add new functions, constants, or logic to `experiment.go`. The `experimentConfig()` function builds the inference options — you can modify how it's constructed.

### Tier 2: ane/ package (deeper changes)

#### BLAS dispatch (ane/accel_darwin.go)
- `linearSingleGEMV` — cblas_sgemv for sequence-length-1 inference. You can try:
  - Different BLAS routines or dispatch thresholds
  - Batched BLAS calls
  - Mixed-precision dispatch (fp16 accumulate)

#### BNNS fp16 (ane/bnns_darwin.go)
- `BNNSLinearFP16` — BNNS fp16-weight GEMV. You can try:
  - Different filter configurations
  - Layer-specific fp16 vs fp32 decisions

#### Metal GPU (ane/metal_darwin.go, ane/storiesane/metal_gemv_darwin.go)
- `MetalLinearSingle` — MPS GPU matmul
- Custom Metal compute shaders for fp16 matvec
- You can try:
  - Different threadgroup sizes
  - Fused kernels (e.g., matmul + activation)
  - GPU for attention computation

#### FP16 conversion (ane/fp16_pack_darwin.go)
- NEON fp16↔fp32 conversion routines
- You can try:
  - Wider SIMD (if available)
  - Lazy conversion (only convert on demand)

#### KV cache (ane/kvcache.go)
- Cache layout and memory access patterns
- You can try:
  - Contiguous vs strided layouts
  - Cache quantization (fp16 KV cache)
  - Pre-allocation strategies

#### Tiled layers (ane/layer_tiled_darwin.go)
- Tiled ANE layer forward for models too large for single-pass ANE
- You can try:
  - Different tile sizes and overlap strategies
  - Pipelining tile computation

#### Forward pass (ane/eval_cached.go)
- KV-cached inference forward pass with per-component timing
- You can try:
  - Fusing operations (e.g., RMS norm + linear)
  - Reordering computation for better cache locality
  - Parallel dispatch of independent operations

#### Important constraints for Tier 2 changes
- **Keep stub files in sync** with darwin files — both must have the same function signatures.
- **Don't break the Engine API** — `Open`, `Step`, `EvalLogits`, `Close` must keep their signatures.
- **Don't modify ANE kernel compilation** (runtime.go, layer_darwin.go, backward_darwin.go) — these are complex and fragile.
- **Test carefully** — run correctness oracle after every change. If logits diverge after a Tier 2 change, revert immediately.

## Parameter space guidance

Good starting experiments (Tier 1 — fast):
1. **BLAS dispatch**: cblas_sgemv vs BNNS fp16 vs Metal GPU for different layer sizes
2. **KV cache layout**: contiguous vs strided, fp32 vs fp16 cache
3. **Buffer pre-allocation**: pre-allocate all scratch buffers at engine init
4. **Metal vs CPU**: which layers benefit from GPU offload
5. **Tile size**: optimize tile dimensions for the model's layer sizes

Deeper experiments (Tier 2 — more impactful, more risk):
1. **Custom Metal kernels**: hand-tuned fp16 matvec shaders
2. **Fused operations**: combine RMS norm + linear, or attention score + softmax
3. **Memory layout optimization**: ensure cache-friendly access patterns in attention
4. **FP16 weight storage**: keep weights in fp16, convert only at compute time
5. **Parallel layer dispatch**: run independent operations concurrently
6. **KV cache quantization**: store KV cache in fp16 or int8
7. **Speculative prefetch**: prefetch next layer's weights during current layer compute
8. **Attention kernel optimization**: BLAS-based vs custom attention for single-token decode
9. **Classifier head optimization**: vocabulary projection dispatch strategy
10. **SiLU/activation fusion**: fuse activation with preceding linear

## Model architecture

The models are Qwen3 transformers optimized for ANE inference:

### Qwen3-0.6B (primary benchmark target)
- Vocab: 151,936
- Dim: 1,024
- Hidden: 3,072
- Heads: 16 (KV heads: 8, GQA)
- Layers: 28
- Default sequence length: 256

### Qwen3-4B
- Vocab: 151,936
- Dim: 2,560
- Hidden: 6,912
- Heads: 32 (KV heads: 8, GQA)
- Layers: 36

### Architecture details
- **SiLU activation**: FFN uses SiLU gating (`silu(gate) * up`)
- **RMSNorm**: Pre-norm with learnable weights
- **GQA**: Grouped-query attention (fewer KV heads than Q heads)
- **RoPE**: Rotary positional embeddings
- **BPE tokenizer**: Go-native tokenization from HuggingFace tokenizer.json
