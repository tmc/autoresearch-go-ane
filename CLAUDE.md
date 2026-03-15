# autoresearch-go-ane

LLM-driven ML research on Apple Neural Engine. See `program.md` for full experiment protocol.

## Quick reference

```bash
# Build bench-note (once per session)
go build -o bench-note ./cmd/bench-note/

# Run benchmarks and attach to HEAD
./bench-note run --benchtime=5x --count=6

# View results
./bench-note show           # full note for HEAD
./bench-note raw            # raw go test output only
./bench-note compare c1 c2  # benchstat between two commits
./bench-note history        # all commits with bench notes
```

## bench-note

Benchmark results are stored as git notes in `refs/notes/benchmarks` using txtar format. Each note contains metadata, raw `go test -bench` output, and a benchstat delta against the nearest ancestor with a bench note.

- **Always use `bench-note run`** after committing an experiment change. This is the canonical way to record benchmark results.
- Use `--from-file` to attach already-captured output without re-running.
- The benchstat delta is computed automatically — no need to manage `bench_before.txt`/`bench_after.txt` manually.
- Use `bench-note history --oneline` for a quick overview of all benchmarked commits.

## Key files

| File | Role | Editable |
|------|------|----------|
| `experiment.go` | Hyperparameters, LR schedule | Yes (Tier 1) |
| `ane/train_full.go` | Forward/backward pass, optimizer | Yes (Tier 2) |
| `ane/train_util.go` | Loss, RMS norm grads, RoPE | Yes (Tier 2) |
| `ane/common.go` | CPU primitives | Yes (Tier 2) |
| `cmd/bench-note/main.go` | Benchmark result tool | Yes |
| `harness.go` | Eval harness | No |
| `bench_test.go` | Benchmark harness | No |

## Git notes

- Benchmark notes: `refs/notes/benchmarks` (managed by bench-note)
- Commit metadata: `refs/notes/commits` (managed by git-auto-commit-message)
- Do not mix these refs.
