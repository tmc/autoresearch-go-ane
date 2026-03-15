"""
Ensue coordinator for ANE inference speed research.

Ported from cvxpy-agi's coordinator.py. Provides persistent memory across
Claude Code sessions for tracking experiments, insights, hypotheses, and
the current best configuration.

Usage (CLI):
    python3 scripts/coordinator.py analyze
    python3 scripts/coordinator.py publish_result '<desc>' '<bench_json>' '<git_diff>' keep
    python3 scripts/coordinator.py check_tried '<idea>'
    python3 scripts/coordinator.py publish_hypothesis '<title>' '<hypothesis>' [priority]
    python3 scripts/coordinator.py list_hypotheses [status]
    python3 scripts/coordinator.py post_insight '<text>' [evidence_key1,key2,...]
    python3 scripts/coordinator.py pull_best
    python3 scripts/coordinator.py ask '<query>'
"""

import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import requests

ENSUE_BASE_URL = "https://api.ensue-network.ai"
NAMESPACE = "autoresearch-go-ane-infer"


def _load_api_key() -> str:
    """Load API key from env or file."""
    key = os.environ.get("ENSUE_API_KEY", "")
    if not key:
        key_file = Path(__file__).parent.parent / ".autoresearch-key"
        if key_file.exists():
            key = key_file.read_text().strip()
    return key


def _slugify(text: str) -> str:
    """Convert text to URL-safe slug."""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[\s_]+', '-', text)
    return text[:80]


def _experiment_hash(description: str) -> str:
    """Short hash for experiment deduplication."""
    return hashlib.sha256(description.encode()).hexdigest()[:12]


class Coordinator:
    """Solo research coordinator backed by Ensue shared memory."""

    def __init__(self):
        self.api_key = _load_api_key()
        self._connected = None
        self._rpc_id = 0

    @property
    def connected(self) -> bool:
        """Test connectivity to Ensue."""
        if self._connected is None:
            try:
                self._call_tool("list_keys", {"prefix": f"{NAMESPACE}/best/%"})
                self._connected = True
            except Exception:
                self._connected = False
        return self._connected

    def _call_tool(self, tool_name: str, arguments: dict) -> Any:
        """Make a JSON-RPC 2.0 call to Ensue."""
        if not self.api_key:
            raise RuntimeError(
                "No Ensue API key found. Set ENSUE_API_KEY or create .autoresearch-key"
            )

        self._rpc_id += 1
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments,
            },
            "id": self._rpc_id,
        }

        resp = requests.post(
            ENSUE_BASE_URL,
            headers=headers,
            json=payload,
            timeout=15,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Ensue error {resp.status_code}: {resp.text[:200]}")

        # Response may be SSE format (data: {...}) or plain JSON
        text = resp.text.strip()
        if text.startswith("data: "):
            text = text[len("data: "):]
        data = json.loads(text)

        if "error" in data:
            raise RuntimeError(f"Ensue error: {data['error']}")
        return data.get("result")

    def _structured(self, result: Any) -> dict:
        """Extract structuredContent from an Ensue RPC result."""
        if isinstance(result, dict):
            return result.get("structuredContent") or result
        return {}

    def _set_memory(self, key: str, value: str, embed: bool = False) -> Any:
        """Create or update a memory item."""
        if embed:
            try:
                self._call_tool("delete_memory", {"key_names": [key]})
            except Exception:
                pass
        try:
            return self._call_tool("create_memory", {
                "items": [{
                    "key_name": key,
                    "description": key,
                    "value": value,
                    "embed": embed,
                }],
            })
        except Exception:
            return self._call_tool("update_memory", {
                "key_name": key,
                "value": value,
            })

    def _get_memory(self, key: str) -> str | None:
        """Get a memory item value by key."""
        result = self._call_tool("get_memory", {"key_names": [key]})
        sc = self._structured(result)
        results = sc.get("results", [])
        if results and results[0].get("status") == "success":
            return results[0].get("value")
        return None

    def _list_keys(self, prefix: str) -> list[str]:
        """List keys with a given prefix (use SQL LIKE % wildcard)."""
        result = self._call_tool("list_keys", {"prefix": f"{prefix}%"})
        sc = self._structured(result)
        return [k["key_name"] for k in sc.get("keys", []) if "key_name" in k]

    def _search(self, query: str, limit: int = 10, prefix: str = "") -> list[dict]:
        """Semantic search over memories."""
        args = {"query": query, "limit": limit}
        if prefix:
            args["prefix"] = prefix
        result = self._call_tool("search_memories", args)
        sc = self._structured(result)
        return sc.get("results", [])

    # -- Results ---------------------------------------------------------------

    def publish_result(
        self,
        description: str,
        benchmark_json: dict | str,
        git_diff: str = "",
        status: str = "keep",
    ) -> str:
        """Publish an experiment result."""
        if isinstance(benchmark_json, str):
            benchmark_json = json.loads(benchmark_json)

        exp_hash = _experiment_hash(description)
        key = f"{NAMESPACE}/results/{exp_hash}"

        value = {
            "description": description,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "status": status,
            "benchmark": benchmark_json,
            "git_diff": git_diff,
        }

        self._set_memory(key, json.dumps(value), embed=True)

        # Update best if this is better
        if status == "keep" and "tokens_per_sec" in benchmark_json:
            self._maybe_update_best(description, benchmark_json, git_diff)

        print(f"Published result: {description} [{status}] -> {key}")
        return key

    def _maybe_update_best(self, description: str, benchmark: dict, git_diff: str):
        """Update best/ if this result beats the current best."""
        new_score = benchmark.get("tokens_per_sec", 0)

        try:
            current = self._get_memory(f"{NAMESPACE}/best/metadata")
            if current:
                current_data = json.loads(current)
                if current_data.get("tokens_per_sec", 0) >= new_score:
                    return  # Current best is still better
        except Exception:
            pass  # No current best, this becomes the best

        metadata = {
            "description": description,
            "tokens_per_sec": new_score,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "benchmark": benchmark,
        }
        self._set_memory(f"{NAMESPACE}/best/metadata", json.dumps(metadata))
        if git_diff:
            self._set_memory(f"{NAMESPACE}/best/config", git_diff)
        print(f"  ** New best! tokens/s={new_score:.1f}")

    def list_results(self) -> list[dict]:
        """List all published results."""
        keys = self._list_keys(f"{NAMESPACE}/results/")
        results = []
        for key in keys:
            try:
                val = self._get_memory(key)
                if val:
                    results.append(json.loads(val))
            except Exception:
                continue
        return results

    # -- Insights --------------------------------------------------------------

    def post_insight(self, text: str, evidence_keys: list[str] | None = None) -> str:
        """Post a research insight."""
        slug = _slugify(text)
        key = f"{NAMESPACE}/insights/{slug}"

        value = {
            "text": text,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "evidence": evidence_keys or [],
        }
        self._set_memory(key, json.dumps(value), embed=True)
        print(f"Posted insight: {text[:60]}...")
        return key

    def list_insights(self) -> list[dict]:
        """List all insights."""
        keys = self._list_keys(f"{NAMESPACE}/insights/")
        insights = []
        for key in keys:
            try:
                val = self._get_memory(key)
                if val:
                    insights.append(json.loads(val))
            except Exception:
                continue
        return insights

    # -- Hypotheses ------------------------------------------------------------

    def publish_hypothesis(
        self,
        title: str,
        hypothesis: str,
        priority: int = 2,
        status: str = "open",
    ) -> str:
        """Publish a research hypothesis to try."""
        slug = _slugify(title)
        key = f"{NAMESPACE}/hypotheses/{slug}"

        value = {
            "title": title,
            "hypothesis": hypothesis,
            "priority": priority,
            "status": status,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        self._set_memory(key, json.dumps(value), embed=True)
        print(f"Published hypothesis [P{priority}]: {title}")
        return key

    def list_hypotheses(self, status: str | None = None) -> list[dict]:
        """List hypotheses, optionally filtered by status."""
        keys = self._list_keys(f"{NAMESPACE}/hypotheses/")
        hypotheses = []
        for key in keys:
            try:
                val = self._get_memory(key)
                if val:
                    h = json.loads(val)
                    if status is None or h.get("status") == status:
                        hypotheses.append(h)
            except Exception:
                continue
        hypotheses.sort(key=lambda h: h.get("priority", 99))
        return hypotheses

    # -- Best Config -----------------------------------------------------------

    def pull_best(self) -> dict:
        """Get the current best configuration."""
        try:
            val = self._get_memory(f"{NAMESPACE}/best/metadata")
            if val:
                meta = json.loads(val)
                diff_val = self._get_memory(f"{NAMESPACE}/best/config")
                meta["git_diff"] = diff_val or ""
                return meta
        except Exception:
            pass
        return {"description": "No best yet", "tokens_per_sec": None}

    # -- Analysis --------------------------------------------------------------

    def analyze(self) -> str:
        """Print a summary analysis of all research so far."""
        results = self.list_results()
        insights = self.list_insights()
        hypotheses = self.list_hypotheses()
        best = self.pull_best()

        lines = []
        lines.append("=" * 60)
        lines.append("ANE Inference Speed Research Summary")
        lines.append("=" * 60)

        # Best
        lines.append(f"\nBest config: {best.get('description', 'none')}")
        if best.get("tokens_per_sec"):
            lines.append(f"  tokens/s: {best['tokens_per_sec']:.1f}")

        # Results
        lines.append(f"\nResults: {len(results)} experiments")
        keep = [r for r in results if r.get("status") == "keep"]
        discard = [r for r in results if r.get("status") == "discard"]
        errors = [r for r in results if r.get("status") == "error"]
        lines.append(f"  keep={len(keep)}, discard={len(discard)}, error={len(errors)}")

        for r in sorted(results, key=lambda x: x.get("timestamp", ""))[-5:]:
            tps = r.get("benchmark", {}).get("tokens_per_sec", "?")
            lines.append(
                f"  [{r.get('status', '?')}] {r.get('description', '?')[:50]} "
                f"(tokens/s={tps})"
            )

        # Insights
        lines.append(f"\nInsights: {len(insights)}")
        for i in insights[-5:]:
            lines.append(f"  - {i.get('text', '?')[:70]}")

        # Hypotheses
        open_h = [h for h in hypotheses if h.get("status") == "open"]
        lines.append(f"\nHypotheses: {len(hypotheses)} total, {len(open_h)} open")
        for h in open_h[:5]:
            lines.append(f"  [P{h.get('priority', '?')}] {h.get('title', '?')}")

        output = "\n".join(lines)
        print(output)
        return output

    # -- Dedup / Prior Art -----------------------------------------------------

    def check_tried(self, description: str, limit: int = 3) -> list[dict]:
        """Search for similar past experiments."""
        matches = self._search(description, limit=limit, prefix=f"{NAMESPACE}/results/")
        if matches:
            print(f"Found {len(matches)} similar past experiments:")
            for m in matches:
                if isinstance(m, dict):
                    key = m.get("key_name", "")
                    val = m.get("value", "")
                    try:
                        data = json.loads(val) if isinstance(val, str) else val
                        desc = data.get("description", key)
                        status = data.get("status", "?")
                        tps = data.get("benchmark", {}).get("tokens_per_sec", "?")
                        print(f"  [{status}] {desc[:60]} (tokens/s={tps})")
                    except (json.JSONDecodeError, AttributeError):
                        print(f"  {key}")
        else:
            print("No similar experiments found.")
        return matches

    # -- Search ----------------------------------------------------------------

    def ask(self, query: str) -> list[dict]:
        """Semantic search over results and insights."""
        matches = self._search(query, prefix=f"{NAMESPACE}/")
        if matches:
            print(f"Found {len(matches)} matches for '{query}':")
            for m in matches[:10]:
                if isinstance(m, dict):
                    key = m.get("key_name", "")
                    desc = m.get("description", key)
                    print(f"  {desc[:80]}")
                else:
                    print(f"  {str(m)[:80]}")
        else:
            print(f"No matches for '{query}'")
        return matches


def _cli():
    """CLI interface for calling coordinator methods from shell."""
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/coordinator.py <command> [args...]")
        print("Commands: analyze, publish_result, check_tried, publish_hypothesis,")
        print("          list_hypotheses, post_insight, pull_best, ask")
        sys.exit(1)

    coord = Coordinator()
    cmd = sys.argv[1]

    if cmd == "analyze":
        if not coord.connected:
            print("ERROR: Cannot connect to Ensue. Check API key.")
            sys.exit(1)
        coord.analyze()

    elif cmd == "publish_result":
        # publish_result '<desc>' '<bench_json>' '<git_diff>' <status>
        if len(sys.argv) < 4:
            print("Usage: publish_result '<desc>' '<bench_json>' ['<git_diff>'] [status]")
            sys.exit(1)
        desc = sys.argv[2]
        bench = sys.argv[3]
        git_diff = sys.argv[4] if len(sys.argv) > 4 else ""
        status = sys.argv[5] if len(sys.argv) > 5 else "keep"
        coord.publish_result(desc, bench, git_diff, status)

    elif cmd == "check_tried":
        if len(sys.argv) < 3:
            print("Usage: check_tried '<idea>'")
            sys.exit(1)
        coord.check_tried(sys.argv[2])

    elif cmd == "publish_hypothesis":
        # publish_hypothesis '<title>' '<hypothesis>' [priority] [status]
        if len(sys.argv) < 4:
            print("Usage: publish_hypothesis '<title>' '<hypothesis>' [priority] [status]")
            sys.exit(1)
        title = sys.argv[2]
        hypothesis = sys.argv[3]
        priority = int(sys.argv[4]) if len(sys.argv) > 4 else 2
        status = sys.argv[5] if len(sys.argv) > 5 else "open"
        coord.publish_hypothesis(title, hypothesis, priority, status)

    elif cmd == "list_hypotheses":
        status = sys.argv[2] if len(sys.argv) > 2 else None
        hypotheses = coord.list_hypotheses(status)
        for h in hypotheses:
            print(f"  [P{h.get('priority', '?')}] [{h.get('status', '?')}] {h.get('title', '?')}")

    elif cmd == "post_insight":
        if len(sys.argv) < 3:
            print("Usage: post_insight '<text>' [evidence_key1,key2,...]")
            sys.exit(1)
        text = sys.argv[2]
        evidence = sys.argv[3].split(",") if len(sys.argv) > 3 else []
        coord.post_insight(text, evidence)

    elif cmd == "pull_best":
        best = coord.pull_best()
        print(json.dumps(best, indent=2))

    elif cmd == "ask":
        if len(sys.argv) < 3:
            print("Usage: ask '<query>'")
            sys.exit(1)
        coord.ask(sys.argv[2])

    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)


if __name__ == "__main__":
    _cli()
