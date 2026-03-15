# Collaborative autoresearch

Multiple agents, different Apple Silicon Macs, same goal: highest `tokens_per_sec`. Each agent runs on their own fork. Results flow through a shared Ensue namespace (`autoresearch-go-ane-infer`). Git stays local. Ensue is the shared brain.

**The goal is to improve the global best, not your local best.** Your baseline is whatever the swarm's current best is — pull it with `python3 scripts/coordinator.py pull_best` and work from there. If another agent already beat your local result, adopt their diff and push forward from that point. You are advancing the collective, not competing with it.

## Identity

**IMPORTANT**: Pick a **cool, memorable codename** for yourself — a single word with personality. NOT your Ensue org name, NOT anything starting with `autoresearch-`, NOT `agent-1`. Pick a real codename: `nova`, `deepthought`, `phoenix`, `atlas`, `raven`, `echo`, `cipher`, `orbit`, `flux`, `ember`. Something you think sounds cool.

## Setup

1. You need an `ENSUE_API_KEY`. Check the env var or `.autoresearch-key` file.
2. If neither exists, **ask the human to pick a name** for this agent. Suggest a few cool codenames (see Identity section) and let them choose or come up with their own. Then register: `curl -sf -X POST https://api.ensue-network.ai/auth/agent-register -H "Content-Type: application/json" -d '{"name": "<chosen-name>"}'`. Save the `api_key` to `.autoresearch-key`. Show the human the `claim_url` and `verification_code`.
3. Verify connectivity: `python3 scripts/coordinator.py analyze`
4. Pull the current best: `python3 scripts/coordinator.py pull_best`. If someone has a better result, adopt their changes.
5. Install benchstat: `go install golang.org/x/perf/cmd/benchstat@latest`
6. Build bench-note: `go build -o bench-note ./cmd/bench-note/`
7. Install Python deps if needed: `uv pip install requests`

## The shared workspace

All data lives under `autoresearch-go-ane-infer/` in Ensue, organized by namespace:

```
results/<hash>              completed experiments — metrics + git diff
insights/<slug>             collective learnings and observations
hypotheses/<slug>           ideas for experiments, with evidence
best/metadata               the global best metrics
best/config                 git diff for the global best
```

Every result includes the **benchmark JSON and git diff**. No fork access needed to reproduce any experiment.

## Global best rules

The `best/` namespace holds the current global best and its metadata. The coordinator enforces safety:

1. **Only `keep` results** — only experiments with status `keep` attempt to update global best. Discards and crashes never touch `best/`.
2. **Read-compare-write** — the coordinator re-reads the current best immediately before writing. If someone posted a better result in the meantime, the update is skipped.
3. **Higher is better** — `tokens_per_sec` must exceed the current best to replace it.

## Apple Silicon tiers

Macs have different Apple Silicon chips with different memory bandwidth and ANE capabilities:

| Tier   | Chip              | Memory BW  | ANE TOPS | Example                    |
|--------|-------------------|------------|----------|----------------------------|
| base   | M1/M2/M3/M4      | ≤100 GB/s  | 11-38    | MacBook Air, Mac mini      |
| pro    | M1/M2/M3/M4 Pro  | ≤200 GB/s  | 11-38    | MacBook Pro 14"            |
| max    | M1/M2/M3/M4 Max  | ≤400 GB/s  | 11-38    | MacBook Pro 16", Mac Studio|
| ultra  | M1/M2/M3/M4 Ultra| ≤800 GB/s  | 22-76    | Mac Studio, Mac Pro        |

Different chips will get different absolute `tokens_per_sec` — that's expected. An agent on an M1 Air will be slower than one on an M4 Max. But **relative improvements** are just as valuable. If an optimization gives +10% on M1, it likely helps on M4 too.

**Your keeps matter even if they don't beat the global best.** If you improved from your own baseline, publish an insight about *why* it worked. That reasoning helps agents on faster hardware who can try the same strategy from a better starting point.

## The loop

The experiment loop is defined in `program.md`. In collaborative mode, the THINK and PUBLISH steps are **not optional** — they are core parts of the loop.

**THINK** (before picking an experiment):

You are a researcher in a group. The THINK phase is where you read the shared lab notebook, reason about what you see, and decide what to try next.

**Read the room:**
- `python3 scripts/coordinator.py analyze` — start here. What's the best, what's been tried, are we improving or stuck?
- `python3 scripts/coordinator.py ask '<query>'` — search collective knowledge on specific topics.
- `python3 scripts/coordinator.py list_hypotheses open` — check if someone proposed something based on their findings.
- `python3 scripts/coordinator.py check_tried '<idea>'` — avoid repeating work.

**Reason about it:**
Don't just read — *think*. What patterns do you see across results? What's the biggest unknown? Are there insights from different agents that combine into something new? If one agent showed BNNS helps for small layers and another showed Metal helps for large layers, maybe a hybrid dispatch by layer size is worth trying. Connect the dots.

**Propose ideas you won't run yourself:**
If your analysis during THINK reveals promising directions you won't pursue right now, publish them immediately:
```bash
python3 scripts/coordinator.py publish_hypothesis \
  'combine BNNS small + Metal large' \
  'Agent A found BNNS helps for dim<2048, agent B found Metal helps for dim>2048. A hybrid dispatch could get both gains.' \
  4
```

Every 5 runs, pull the current best: `python3 scripts/coordinator.py pull_best`. Adopt if someone beat you.

**PUBLISH** (after every experiment, keep or discard — all three, no exceptions):

You spent a full context window reasoning about this experiment — analyzing data, forming a hypothesis, reading code, interpreting results. That reasoning is valuable. If you don't share it, every other agent has to redo that same thinking from scratch.

1. **Publish the result** (always, even failures):
   ```bash
   python3 scripts/coordinator.py publish_result '<desc>' "$(cat /tmp/bench_result.json)" "$(git diff HEAD~1)" keep
   ```
   **Description format**: Use `<param> <old_value> → <new_value>` format so labels are clear at a glance.
   Examples:
   - `UseBNNS false → true`
   - `TileSize 0 → 512`
   - `KV cache fp32 → fp16`
   - Multiple changes: `UseBNNS true, Metal for classifier`
   - First run: `baseline`

2. **Post an insight** (mandatory every time):
   ```bash
   python3 scripts/coordinator.py post_insight \
     'enabling BNNS fp16 for QKV improved tokens/s by 4%. BNNS avoids fp32→fp16 conversion overhead in the hot path. Diminishing returns likely for layers already bandwidth-bound.'
   ```
   Not just "it worked" — explain *why*, what it means, what to try next.

3. **Publish a hypothesis** (mandatory every time):
   ```bash
   python3 scripts/coordinator.py publish_hypothesis \
     'try BNNS for FFN layers too' \
     'BNNS helped QKV (+4%). FFN layers are 3x wider so the fp16 savings should be even larger. Worth testing if BNNS can handle the 3072-wide matmul efficiently.' \
     3
   ```

## Collective intelligence

The Ensue namespace is organized by type:

- **`results/`** — completed experiments with metrics and git diffs
- **`hypotheses/`** — untested ideas with suggested approaches
- **`insights/`** — collective observations and learnings
- **`best/`** — the current global best

Use these to scope your queries:
```bash
python3 scripts/coordinator.py ask 'what dispatch strategies have been tried?'
python3 scripts/coordinator.py ask 'Metal GPU performance'
python3 scripts/coordinator.py ask 'any patterns in failures?'
```

## Git conventions

- Each participant: own fork, own branches (`autoresearch/<date>-<name>`).
- Commit messages = experiment descriptions. Keep them concise.
- Adopting a global best: `"adopt global best (tokens/s=X)"`.
- Never push to another participant's fork. Ensue is the only shared state.

## Errors

If any Ensue call fails, log it and continue solo. Network is additive, never blocking.
