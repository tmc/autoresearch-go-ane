# Collaborative autoresearch

Multiple agents, different Macs, same goal: lowest val_bpb on Apple Neural Engine. Results flow through the Ensue memory network (`@travis_cline`). Git stays local. Ensue is the shared brain.

**Full protocol**: See `.skills/autoresearch/SKILL.md` for the complete experiment loop, Ensue tool usage, key schemas, and safety rules.

## Quick reference

- **Org**: `travis_cline`
- **Workloads**: `train` (default), `infer`
- **Key prefix**: `@travis_cline/<workload>/`
- **Namespaces**: `results/`, `claims/`, `hypotheses/`, `insights/`, `best/`
- **Key format**: `<agent>--<slug>--<6char_hash>`
- **Primary metric**: `val_bpb` (lower is better)
- **Chip tiers**: base (M1, <=12 TOPS), mid (M2, <=17), high (M3, <=20), ultra (M4/M5, >20)

## Ensue access (priority order)

1. **Ensue MCP tools** — `create_memory`, `get_memory`, `search_memories`, `list_keys`, `update_memory`
2. **`ensue-api.sh`** — shell wrapper (if ensue-skill plugin installed)
3. **`curl`** to `https://api.ensue-network.ai/` — direct JSON-RPC

## Go helpers

The `coordinator` package provides chip detection and key generation (no network calls):

```go
chip := coordinator.DetectChip()                          // .Name, .Tier, .TOPS
key := coordinator.ExperimentKey("phoenix", "description") // agent--slug--hash
pfx := coordinator.Pfx("train", "results", key)           // @travis_cline/train/results/...
```

## Hub setup

One-time: `ENSUE_API_KEY=<key> ./scripts/setup-hub.sh`
