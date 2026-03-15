"""
Collaborative autoresearch coordinator for ANE inference speed research.

Bridges the autoresearch experiment loop with the Ensue memory network,
enabling distributed research across multiple Apple Silicon participants.

Uses `requests` for JSON-RPC calls. Zero new deps beyond requests.

Usage (CLI):
    python3 scripts/coordinator.py analyze
    python3 scripts/coordinator.py analyze_swarm
    python3 scripts/coordinator.py publish_result '<desc>' '<bench_json>' '<git_diff>' keep
    python3 scripts/coordinator.py check_tried '<idea>'
    python3 scripts/coordinator.py claim '<description>'
    python3 scripts/coordinator.py publish_hypothesis '<title>' '<hypothesis>' [priority]
    python3 scripts/coordinator.py list_hypotheses [status]
    python3 scripts/coordinator.py post_insight '<text>' [evidence_key1,key2,...]
    python3 scripts/coordinator.py pull_best
    python3 scripts/coordinator.py ask '<query>' [namespace]
    python3 scripts/coordinator.py list_namespace '<namespace>'
    python3 scripts/coordinator.py join_hub
    python3 scripts/coordinator.py announce
    python3 scripts/coordinator.py get_swarm_insights '<topic>'
    python3 scripts/coordinator.py get_unclaimed_hypotheses
"""

import base64
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Any, Optional

import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

HUB_ORG = "austin_mac"
API_URL = "https://api.ensue-network.ai/"
KEY_FILE = ".autoresearch-key"
NAMESPACE = "autoresearch-go-ane-infer"
INVITE_TOKEN = "596e13c30e7848e7b50e3bb2bf5f319d5eaa99eeb5df4337bab90310bfd76385"

CLAIM_TTL = 900              # 15 min soft expiry (3x expected 5-min experiment)
VERIFY_DELAY = 2             # seconds between claim and verify
SEMANTIC_THRESHOLD = 0.92    # block if active claim is this similar
MAX_CLAIM_ATTEMPTS = 5       # alternatives before giving up
SYNC_EVERY_N = 5             # pull global best every N experiments

# Apple Silicon chip tiers (by ANE TOPS)
CHIP_TIERS: dict[str, int] = {
    "base": 11,    # M1 family (11 TOPS)
    "mid": 16,     # M2 family (15.8 TOPS)
    "high": 18,    # M3 family (18 TOPS)
    "ultra": 38,   # M4 family (38 TOPS)
}
_TIER_THRESHOLDS: list[tuple[str, int]] = [
    ("base", 12),
    ("mid", 17),
    ("high", 20),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_api_key() -> Optional[str]:
    """Read API key from env var or key file."""
    key = os.environ.get("ENSUE_API_KEY")
    if key:
        return key.strip()
    if os.path.exists(KEY_FILE):
        with open(KEY_FILE) as f:
            return f.read().strip()
    return None


def _experiment_hash(description: str) -> str:
    """Hash an experiment description for dedup keying."""
    return hashlib.sha256(description.lower().strip().encode()).hexdigest()[:12]


def _slugify(text: str, max_len: int = 40) -> str:
    """Turn text into a URL-safe slug."""
    slug = text.lower().strip()
    slug = re.sub(r'[^a-z0-9]+', '-', slug)
    slug = slug.strip('-')
    return slug[:max_len].rstrip('-')


def _experiment_key(agent_id: str, description: str) -> str:
    """
    Human-readable experiment key: <agent>--<slug>--<short_hash>

    Example: nova--bnns-fp16-qkv--a7f3b2
    """
    slug = _slugify(description)
    short_hash = _experiment_hash(description)[:6]
    agent = _slugify(agent_id, max_len=20) or "unknown"
    return f"{agent}--{slug}--{short_hash}"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_remote_url() -> Optional[str]:
    """Get the GitHub HTTPS URL for the current repo."""
    try:
        url = subprocess.check_output(
            ["git", "remote", "get-url", "origin"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        if url.startswith("git@github.com:"):
            url = "https://github.com/" + url[len("git@github.com:"):]
        if url.endswith(".git"):
            url = url[:-4]
        return url
    except Exception:
        return None


def _git_branch() -> Optional[str]:
    """Get the current git branch name."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def _git_commit_short() -> Optional[str]:
    """Get the short commit hash of HEAD."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def detect_chip_info() -> tuple[Optional[str], Optional[int]]:
    """Detect Apple Silicon chip name and ANE TOPS. Returns (chip_name, ane_tops)."""
    try:
        chip = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        # Parse ANE TOPS from chip name
        # M1: 11, M2: 15.8, M3: 18, M4: 38
        if "M4" in chip:
            return chip, 38
        elif "M3" in chip:
            return chip, 18
        elif "M2" in chip:
            return chip, 16
        elif "M1" in chip:
            return chip, 11
        return chip, None
    except Exception:
        return None, None


def get_chip_tier(ane_tops: int) -> str:
    """Classify ANE TOPS into a tier name."""
    for tier_name, threshold in _TIER_THRESHOLDS:
        if ane_tops <= threshold:
            return tier_name
    return "ultra"


# ---------------------------------------------------------------------------
# JSON-RPC
# ---------------------------------------------------------------------------

def ensue_rpc(api_key: str, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Make a JSON-RPC call to the Ensue MCP API."""
    payload = {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {"name": tool_name, "arguments": arguments},
        "id": 1,
    }
    resp = requests.post(
        API_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=30,
    )
    resp.raise_for_status()

    text = resp.text.strip()
    if text.startswith("data: "):
        text = text[len("data: "):]

    data = json.loads(text)

    if "error" in data:
        raise RuntimeError(f"RPC error: {data['error']}")

    result = data.get("result", {})
    content = result.get("content", [])
    if content and isinstance(content, list):
        first = content[0]
        if isinstance(first, dict) and "text" in first:
            return json.loads(first["text"])
    return result


# ---------------------------------------------------------------------------
# Coordinator
# ---------------------------------------------------------------------------

class Coordinator:
    """
    Synchronous coordinator for collaborative ANE inference speed research.

    All methods catch exceptions and return gracefully so the experiment loop
    never crashes due to network issues.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or _get_api_key()
        self.agent_id: Optional[str] = None
        self.experiment_count = 0
        self.chip_name, self.ane_tops = detect_chip_info()
        self.chip_tier: Optional[str] = get_chip_tier(self.ane_tops) if self.ane_tops is not None else None

    def _log(self, msg: str) -> None:
        """Print with agent identity prefix."""
        tag = self.agent_id or "coordinator"
        print(f"[{tag}] {msg}")

    @property
    def connected(self) -> bool:
        return self.api_key is not None

    def _rpc(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """RPC call with the stored API key."""
        if not self.api_key:
            raise RuntimeError("No API key configured")
        return ensue_rpc(self.api_key, tool_name, arguments)

    def _make_key(self, description: str) -> str:
        """Create a human-readable experiment key using agent_id."""
        return _experiment_key(self.agent_id or "unknown", description)

    # --- Onboarding ---

    def join_hub(self) -> dict[str, Any]:
        """Claim the hub invite to join autoresearch-at-home."""
        try:
            result = self._rpc("claim_invite", {"token": INVITE_TOKEN})
            self._log(f"Joined hub: {result}")
            return result
        except Exception as e:
            self._log(f"join_hub failed: {e}")
            return {"error": str(e)}

    def test_connectivity(self) -> bool:
        """Test if the API key works."""
        try:
            self._rpc("list_keys", {"limit": 1})
            return True
        except Exception:
            return False

    def announce(self) -> None:
        """Print a startup banner with swarm state."""
        try:
            tag = self.agent_id or "unknown"

            # Get global best
            meta_key = f"@{HUB_ORG}/best/metadata"
            meta = self._rpc("get_memory", {"key_names": [meta_key]})
            meta_results = meta.get("results", [])
            best_line = "no results yet"
            if meta_results and meta_results[0].get("status") == "success":
                current = json.loads(meta_results[0].get("value", "{}"))
                best_tps = current.get("tokens_per_sec", "?")
                best_by = current.get("agent_id", current.get("achieved_by", "?"))
                best_line = f"tokens/s={best_tps} (by {best_by})"

            # Count results
            result_list = self._rpc("list_keys", {
                "prefix": f"@{HUB_ORG}/results/",
                "limit": 200,
            })
            result_keys = result_list.get("keys", [])
            total = len(result_keys)

            # Count active claims
            claim_list = self._rpc("list_keys", {
                "prefix": f"@{HUB_ORG}/claims/",
                "limit": 50,
            })
            active_claims = len(claim_list.get("keys", []))

            # Chip info
            chip_line = "unknown"
            if self.chip_name:
                chip_line = f"{self.chip_name}"
                if self.ane_tops:
                    chip_line += f" ({self.ane_tops} ANE TOPS, tier: {self.chip_tier})"

            banner = f"""
{'=' * 54}
  AUTORESEARCH AGENT: {tag}
  Swarm: {HUB_ORG}
  Chip: {chip_line}
  Global best: {best_line}
  Experiments completed: {total}
  Active claims: {active_claims}
{'=' * 54}"""
            print(banner)
        except Exception as e:
            self._log(f"announce error (non-fatal): {e}")
            print(f"\n  AUTORESEARCH AGENT: {self.agent_id or 'unknown'}\n")

    # --- Work Claiming ---

    def check_claimed(self, experiment_key: str) -> bool:
        """Check if an experiment is already claimed (active) or completed."""
        try:
            # Check if result already exists
            result_key = f"@{HUB_ORG}/results/{experiment_key}"
            result = self._rpc("get_memory", {"key_names": [result_key]})
            results = result.get("results", [])
            if results and results[0].get("status") == "success":
                return True

            # Fallback: check old hash-only format
            if "--" in experiment_key:
                old_hash = experiment_key.rsplit("--", 1)[-1]
                if len(old_hash) <= 12:
                    old_result_key = f"@{HUB_ORG}/results/{old_hash}"
                    old_result = self._rpc("get_memory", {"key_names": [old_result_key]})
                    old_results = old_result.get("results", [])
                    if old_results and old_results[0].get("status") == "success":
                        return True

            # Check for active claim
            claim_key = f"@{HUB_ORG}/claims/{experiment_key}"
            claim = self._rpc("get_memory", {"key_names": [claim_key]})
            claims = claim.get("results", [])
            if claims and claims[0].get("status") == "success":
                value = json.loads(claims[0].get("value", "{}"))
                claimed_at = value.get("claimed_at", "")
                if claimed_at:
                    try:
                        claimed_time = datetime.fromisoformat(claimed_at)
                        age = (datetime.now(timezone.utc) - claimed_time).total_seconds()
                        if age < CLAIM_TTL:
                            return True
                    except (ValueError, TypeError):
                        pass
            return False
        except Exception as e:
            self._log(f"check_claimed error: {e}")
            return False

    def check_similar_claimed(self, description: str) -> list[dict]:
        """Semantic search for similar in-progress work."""
        try:
            result = self._rpc("search_memories", {
                "query": description,
                "limit": 5,
                "prefix": f"@{HUB_ORG}/claims/",
            })
            matches = result.get("results", [])
            similar = []
            for match in matches:
                score = match.get("score", 0)
                if score < SEMANTIC_THRESHOLD:
                    continue
                value = json.loads(match.get("value", "{}"))
                claimed_at = value.get("claimed_at", "")
                if claimed_at:
                    try:
                        claimed_time = datetime.fromisoformat(claimed_at)
                        age = (datetime.now(timezone.utc) - claimed_time).total_seconds()
                        if age < CLAIM_TTL:
                            similar.append({
                                "description": value.get("description", ""),
                                "score": score,
                                "agent": value.get("agent_id", ""),
                            })
                    except (ValueError, TypeError):
                        pass
            return similar
        except Exception as e:
            self._log(f"check_similar_claimed error: {e}")
            return []

    def claim_experiment(self, description: str) -> Optional[str]:
        """
        Attempt to claim an experiment. Returns the experiment key if claimed,
        or None if already taken / similar work in progress.

        The key is human-readable: <agent>--<slug>--<short_hash>
        """
        exp_key = self._make_key(description)

        try:
            if self.check_claimed(exp_key):
                self._log(f"Experiment already claimed/completed: {exp_key}")
                return None

            similar = self.check_similar_claimed(description)
            if similar:
                self._log(f"Similar work in progress: {similar[0]['description']} (score={similar[0]['score']:.3f} by {similar[0]['agent']})")
                return None

            claim_key = f"@{HUB_ORG}/claims/{exp_key}"
            claim_data = {
                "agent_id": self.agent_id or "unknown",
                "description": description,
                "experiment_key": exp_key,
                "claimed_at": _now_iso(),
                "expected_duration_seconds": 300,
                "status": "claimed",
            }
            value_b64 = base64.b64encode(json.dumps(claim_data).encode()).decode()
            self._rpc("create_memory", {"items": [{
                "key_name": claim_key,
                "description": f"[autoresearch] Claim: {description}",
                "value": value_b64,
                "base64": True,
                "embed": True,
                "embed_source": "description",
            }]})

            time.sleep(VERIFY_DELAY)
            verify = self._rpc("get_memory", {"key_names": [claim_key]})
            verify_results = verify.get("results", [])
            if verify_results and verify_results[0].get("status") == "success":
                value = json.loads(verify_results[0].get("value", "{}"))
                if value.get("agent_id") == (self.agent_id or "unknown"):
                    self._log(f"Claimed experiment: {exp_key}")
                    return exp_key

            self._log(f"Lost claim race for: {exp_key}")
            return None

        except Exception as e:
            self._log(f"claim_experiment error: {e}")
            return exp_key

    # --- Results ---

    def publish_result(
        self,
        description: str,
        benchmark_json: dict | str,
        git_diff: str = "",
        status: str = "keep",
    ) -> str:
        """Publish an experiment result to the hub."""
        if isinstance(benchmark_json, str):
            benchmark_json = json.loads(benchmark_json)

        exp_key = self._make_key(description)
        commit = _git_commit_short()
        branch = _git_branch()
        repo_url = _git_remote_url()

        tokens_per_sec = benchmark_json.get("tokens_per_sec", 0)

        # Get current bests for delta calculations
        global_best_tps = self._get_global_best_tps()
        delta_vs_best = (tokens_per_sec - global_best_tps) if global_best_tps is not None else None
        agent_best_tps = self._get_agent_best_tps()
        delta_vs_own_best = (tokens_per_sec - agent_best_tps) if agent_best_tps is not None else None

        result_data = {
            "agent_id": self.agent_id or "unknown",
            "tokens_per_sec": tokens_per_sec,
            "chip_name": self.chip_name,
            "chip_tier": self.chip_tier,
            "ane_tops": self.ane_tops,
            "status": status,
            "commit": commit,
            "description": description,
            "git_diff": git_diff,
            "benchmark": benchmark_json,
            "repo_url": repo_url,
            "branch": branch,
            "commit_url": f"{repo_url}/commit/{commit}" if repo_url and commit else None,
            "completed_at": _now_iso(),
            "delta_vs_best": delta_vs_best,
            "global_best_at_publish": global_best_tps,
            "delta_vs_own_best": delta_vs_own_best,
            "agent_best_at_publish": agent_best_tps,
        }

        result_key = f"@{HUB_ORG}/results/{exp_key}"
        value_b64 = base64.b64encode(json.dumps(result_data).encode()).decode()

        agent = self.agent_id or "unknown"
        desc_prefix = f"[{agent} {status.upper()}] tokens/s={tokens_per_sec:.1f}"
        if delta_vs_best is not None:
            desc_prefix += f" (delta={delta_vs_best:+.1f})"
        desc_prefix += f" | {description}"

        try:
            self._rpc("create_memory", {"items": [{
                "key_name": result_key,
                "description": f"[autoresearch] Result: {desc_prefix}",
                "value": value_b64,
                "base64": True,
                "embed": True,
                "embed_source": "description",
            }]})
        except Exception:
            # Key might already exist, try update
            self._rpc("update_memory", {
                "key_name": result_key,
                "value": value_b64,
                "base64": True,
            })

        delta_str = ""
        if delta_vs_best is not None:
            delta_str = f" (delta={delta_vs_best:+.1f} vs global best {global_best_tps:.1f})"
        self._log(f"RESULT: tokens/s={tokens_per_sec:.1f}{delta_str} ({status})")

        # Update bests if this is a keep
        if status == "keep":
            self._update_agent_best(tokens_per_sec, result_data)
            self._maybe_update_best(tokens_per_sec, result_data, git_diff)
            if self.chip_tier:
                self._update_tier_best(tokens_per_sec, result_data, git_diff)

        print(f"Published result: {description} [{status}] -> {result_key}")
        return result_key

    def _get_global_best_tps(self) -> Optional[float]:
        """Read the current global best tokens_per_sec."""
        try:
            meta_key = f"@{HUB_ORG}/best/metadata"
            meta = self._rpc("get_memory", {"key_names": [meta_key]})
            meta_results = meta.get("results", [])
            if meta_results and meta_results[0].get("status") == "success":
                current = json.loads(meta_results[0].get("value", "{}"))
                return current.get("tokens_per_sec")
        except Exception:
            pass
        return None

    def _get_agent_best_tps(self, agent_id: str = None) -> Optional[float]:
        """Read an agent's personal best tokens_per_sec."""
        agent = agent_id or self.agent_id or "unknown"
        try:
            key = f"@{HUB_ORG}/best/agent/{agent}"
            result = self._rpc("get_memory", {"key_names": [key]})
            results = result.get("results", [])
            if results and results[0].get("status") == "success":
                data = json.loads(results[0].get("value", "{}"))
                return data.get("tokens_per_sec")
        except Exception:
            pass
        return None

    def _update_agent_best(self, tokens_per_sec: float, result_data: dict) -> None:
        """Update this agent's personal best if this result beats it."""
        agent = self.agent_id or "unknown"
        try:
            key = f"@{HUB_ORG}/best/agent/{agent}"
            current = self._get_agent_best_tps()

            # Higher is better for tokens_per_sec
            if current is not None and tokens_per_sec <= current:
                return

            best_data = {
                "agent_id": agent,
                "tokens_per_sec": tokens_per_sec,
                "description": result_data.get("description"),
                "chip_name": self.chip_name,
                "chip_tier": self.chip_tier,
                "ane_tops": self.ane_tops,
                "achieved_at": _now_iso(),
                "previous_best_tokens_per_sec": current,
            }
            value_b64 = base64.b64encode(json.dumps(best_data).encode()).decode()
            try:
                self._rpc("update_memory", {
                    "key_name": key,
                    "value": value_b64,
                    "base64": True,
                })
            except Exception:
                self._rpc("create_memory", {"items": [{
                    "key_name": key,
                    "description": f"[autoresearch] Personal best for {agent}: tokens/s={tokens_per_sec:.1f}",
                    "value": value_b64,
                    "base64": True,
                }]})

            improvement = (tokens_per_sec - current) if current else 0
            self._log(f"New personal best! tokens/s={tokens_per_sec:.1f} (improved +{improvement:.1f})")

        except Exception as e:
            self._log(f"_update_agent_best error: {e}")

    def _maybe_update_best(
        self,
        tokens_per_sec: float,
        result_data: dict,
        git_diff: str,
    ) -> bool:
        """
        Update the global best if this result beats it. Returns True if updated.

        Safety rules:
        - Reject tokens_per_sec <= 0
        - Reject improvement > 100% in a single step
        - Read-compare-write to minimize race window
        - Previous best always preserved for recovery
        """
        try:
            if tokens_per_sec <= 0:
                self._log(f"REJECTED best update: tokens/s={tokens_per_sec} <= 0")
                return False

            meta_key = f"@{HUB_ORG}/best/metadata"
            meta = self._rpc("get_memory", {"key_names": [meta_key]})
            meta_results = meta.get("results", [])

            previous_best_tps = None
            previous_best_by = None
            previous_best_description = None
            if meta_results and meta_results[0].get("status") == "success":
                current = json.loads(meta_results[0].get("value", "{}"))
                previous_best_tps = current.get("tokens_per_sec")
                previous_best_by = current.get("agent_id", current.get("achieved_by"))
                previous_best_description = current.get("description")
                # Higher is better
                if previous_best_tps is not None and tokens_per_sec <= previous_best_tps:
                    return False

            # Sanity: improvement shouldn't be impossibly large (>100% in one step)
            if previous_best_tps is not None and previous_best_tps > 0:
                improvement_pct = (tokens_per_sec - previous_best_tps) / previous_best_tps
                if improvement_pct > 1.0:
                    self._log(f"REJECTED best update: {improvement_pct:.0%} improvement is suspiciously large")
                    return False

            # Re-read right before write to minimize race window
            meta2 = self._rpc("get_memory", {"key_names": [meta_key]})
            meta2_results = meta2.get("results", [])
            if meta2_results and meta2_results[0].get("status") == "success":
                current2 = json.loads(meta2_results[0].get("value", "{}"))
                current2_tps = current2.get("tokens_per_sec")
                if current2_tps is not None and tokens_per_sec <= current2_tps:
                    self._log(f"Lost best update race: someone posted {current2_tps:.1f} while we were checking")
                    return False

            best_data = {
                **{k: v for k, v in result_data.items() if k != "git_diff"},
                "tokens_per_sec": tokens_per_sec,
                "achieved_by": self.agent_id or "unknown",
                "achieved_at": _now_iso(),
                "previous_best_tokens_per_sec": previous_best_tps,
                "previous_best_by": previous_best_by,
                "previous_best_description": previous_best_description,
                "improvement_over_previous": (tokens_per_sec - previous_best_tps) if previous_best_tps is not None else None,
            }

            # Upsert best/config (git diff)
            code_key = f"@{HUB_ORG}/best/config"
            if git_diff:
                code_b64 = base64.b64encode(git_diff.encode()).decode()
                try:
                    self._rpc("update_memory", {
                        "key_name": code_key,
                        "value": code_b64,
                        "base64": True,
                    })
                except Exception:
                    self._rpc("create_memory", {"items": [{
                        "key_name": code_key,
                        "description": "[autoresearch] Current best config (git diff)",
                        "value": code_b64,
                        "base64": True,
                    }]})

            # Upsert best/metadata
            meta_b64 = base64.b64encode(json.dumps(best_data).encode()).decode()
            try:
                self._rpc("update_memory", {
                    "key_name": meta_key,
                    "value": meta_b64,
                    "base64": True,
                })
            except Exception:
                self._rpc("create_memory", {"items": [{
                    "key_name": meta_key,
                    "description": "[autoresearch] Metadata for current best ANE inference config",
                    "value": meta_b64,
                    "base64": True,
                }]})

            improvement = (tokens_per_sec - previous_best_tps) if previous_best_tps else 0
            prev_info = f" (improved +{improvement:.1f} over {previous_best_by}'s {previous_best_tps:.1f})" if previous_best_tps else ""
            self._log(f"NEW GLOBAL BEST! tokens/s={tokens_per_sec:.1f}{prev_info}")
            return True

        except Exception as e:
            self._log(f"_maybe_update_best error: {e}")
            return False

    # --- Chip Tier Bests ---

    def _get_tier_best_tps(self, tier: str) -> Optional[float]:
        """Read the current best tokens_per_sec for a chip tier."""
        try:
            key = f"@{HUB_ORG}/best/tier/{tier}/metadata"
            result = self._rpc("get_memory", {"key_names": [key]})
            results = result.get("results", [])
            if results and results[0].get("status") == "success":
                data = json.loads(results[0].get("value", "{}"))
                return data.get("tokens_per_sec")
        except Exception:
            pass
        return None

    def _update_tier_best(self, tokens_per_sec: float, result_data: dict, git_diff: str) -> bool:
        """Update the best config for this agent's chip tier if the result beats it."""
        tier = self.chip_tier
        if not tier:
            return False
        try:
            if tokens_per_sec <= 0:
                return False

            meta_key = f"@{HUB_ORG}/best/tier/{tier}/metadata"
            code_key = f"@{HUB_ORG}/best/tier/{tier}/config"

            current_tps = self._get_tier_best_tps(tier)
            if current_tps is not None and tokens_per_sec <= current_tps:
                return False

            previous_best_tps = current_tps
            previous_best_by = None
            if current_tps is not None:
                meta = self._rpc("get_memory", {"key_names": [meta_key]})
                meta_results = meta.get("results", [])
                if meta_results and meta_results[0].get("status") == "success":
                    prev = json.loads(meta_results[0].get("value", "{}"))
                    previous_best_by = prev.get("agent_id", prev.get("achieved_by"))

            tier_data = {
                **{k: v for k, v in result_data.items() if k != "git_diff"},
                "chip_tier": tier,
                "tokens_per_sec": tokens_per_sec,
                "achieved_by": self.agent_id or "unknown",
                "achieved_at": _now_iso(),
                "previous_best_tokens_per_sec": previous_best_tps,
                "previous_best_by": previous_best_by,
            }

            # Upsert tier config
            if git_diff:
                code_b64 = base64.b64encode(git_diff.encode()).decode()
                try:
                    self._rpc("update_memory", {
                        "key_name": code_key,
                        "value": code_b64,
                        "base64": True,
                    })
                except Exception:
                    self._rpc("create_memory", {"items": [{
                        "key_name": code_key,
                        "description": f"[autoresearch] Best config for chip tier '{tier}'",
                        "value": code_b64,
                        "base64": True,
                    }]})

            # Upsert tier metadata
            meta_b64 = base64.b64encode(json.dumps(tier_data).encode()).decode()
            try:
                self._rpc("update_memory", {
                    "key_name": meta_key,
                    "value": meta_b64,
                    "base64": True,
                })
            except Exception:
                self._rpc("create_memory", {"items": [{
                    "key_name": meta_key,
                    "description": f"[autoresearch] Metadata for best config in chip tier '{tier}'",
                    "value": meta_b64,
                    "base64": True,
                }]})

            improvement = (tokens_per_sec - previous_best_tps) if previous_best_tps is not None else 0
            self._log(f"NEW TIER BEST ({tier})! tokens/s={tokens_per_sec:.1f} (improved +{improvement:.1f})")
            return True

        except Exception as e:
            self._log(f"_update_tier_best error: {e}")
            return False

    def get_tier_best(self, tier: str) -> Optional[dict]:
        """Get the best result metadata for a specific chip tier."""
        try:
            key = f"@{HUB_ORG}/best/tier/{tier}/metadata"
            result = self._rpc("get_memory", {"key_names": [key]})
            results = result.get("results", [])
            if results and results[0].get("status") == "success":
                return json.loads(results[0].get("value", "{}"))
        except Exception as e:
            self._log(f"get_tier_best error: {e}")
        return None

    def get_all_tier_bests(self) -> dict[str, Optional[dict]]:
        """Get the best result for every chip tier."""
        tier_bests: dict[str, Optional[dict]] = {}
        for tier_name in CHIP_TIERS:
            tier_bests[tier_name] = self.get_tier_best(tier_name)
        return tier_bests

    # --- Config Sharing ---

    def pull_best(self) -> dict:
        """Get the current best configuration and metadata."""
        try:
            meta_key = f"@{HUB_ORG}/best/metadata"
            meta = self._rpc("get_memory", {"key_names": [meta_key]})
            meta_results = meta.get("results", [])
            if meta_results and meta_results[0].get("status") == "success":
                metadata = json.loads(meta_results[0]["value"])
                # Also pull the git diff
                code_key = f"@{HUB_ORG}/best/config"
                try:
                    code = self._rpc("get_memory", {"key_names": [code_key]})
                    code_results = code.get("results", [])
                    if code_results and code_results[0].get("status") == "success":
                        metadata["git_diff"] = code_results[0]["value"]
                except Exception:
                    pass
                return metadata
        except Exception as e:
            self._log(f"pull_best error: {e}")
        return {"description": "No best yet", "tokens_per_sec": None}

    def pull_best_for_tier(self, tier: Optional[str] = None) -> dict:
        """Pull the best config for the given chip tier. Falls back to global best."""
        tier = tier or self.chip_tier
        if tier:
            try:
                meta_key = f"@{HUB_ORG}/best/tier/{tier}/metadata"
                meta = self._rpc("get_memory", {"key_names": [meta_key]})
                meta_results = meta.get("results", [])
                if meta_results and meta_results[0].get("status") == "success":
                    metadata = json.loads(meta_results[0]["value"])
                    code_key = f"@{HUB_ORG}/best/tier/{tier}/config"
                    try:
                        code = self._rpc("get_memory", {"key_names": [code_key]})
                        code_results = code.get("results", [])
                        if code_results and code_results[0].get("status") == "success":
                            metadata["git_diff"] = code_results[0]["value"]
                    except Exception:
                        pass
                    self._log(f"Pulled tier '{tier}' best: tokens/s={metadata.get('tokens_per_sec', '?')}")
                    return metadata
                self._log(f"No tier-specific best for '{tier}', falling back to global best")
            except Exception as e:
                self._log(f"pull_best_for_tier error (falling back to global): {e}")
        return self.pull_best()

    def should_sync(self) -> bool:
        """Check if it's time to sync with the global best (every N experiments)."""
        self.experiment_count += 1
        return self.experiment_count % SYNC_EVERY_N == 0

    def get_all_agent_bests(self) -> list[dict]:
        """Get every agent's personal best. Useful for seeing which strategies work across hardware."""
        try:
            result = self._rpc("list_keys", {
                "prefix": f"@{HUB_ORG}/best/agent/",
                "limit": 50,
            })
            keys = result.get("keys", [])
            key_names = []
            for k in keys:
                name = k.get("key_name", k) if isinstance(k, dict) else k
                if not name.startswith(f"@{HUB_ORG}/"):
                    name = f"@{HUB_ORG}/{name}"
                key_names.append(name)

            if not key_names:
                return []

            bests = []
            for kn in key_names:
                try:
                    r = self._rpc("get_memory", {"key_names": [kn]})
                    results = r.get("results", [])
                    if results and results[0].get("status") == "success":
                        data = json.loads(results[0].get("value", "{}"))
                        bests.append(data)
                except Exception:
                    pass

            # Higher is better for tokens_per_sec
            return sorted(bests, key=lambda x: x.get("tokens_per_sec", 0), reverse=True)

        except Exception as e:
            self._log(f"get_all_agent_bests error: {e}")
            return []

    # --- Collective Intelligence ---

    def ask_swarm(self, question: str, namespace: str = None) -> dict:
        """Ask the swarm a question via semantic search. Scope with namespace."""
        try:
            prefix = f"@{HUB_ORG}/"
            if namespace:
                prefix = f"@{HUB_ORG}/{namespace}/"

            result = self._rpc("search_memories", {
                "query": question,
                "limit": 20,
                "prefix": prefix,
            })

            matches = result.get("results", [])
            relevant = []
            for match in matches:
                try:
                    data = json.loads(match.get("value", "{}"))
                    data["_score"] = match.get("score", 0)
                    data["_key"] = match.get("key_name", "")
                    relevant.append(data)
                except (json.JSONDecodeError, KeyError):
                    pass

            relevant.sort(key=lambda x: x.get("_score", 0), reverse=True)
            best_match = relevant[0] if relevant else None

            lines = [f"Swarm answer for: {question}"]
            lines.append(f"Namespace: {namespace or 'all'} | {len(relevant)} results")
            lines.append("")
            for r in relevant[:5]:
                agent = r.get("agent_id", "?")
                tps = r.get("tokens_per_sec")
                status = r.get("status", "")
                desc = r.get("description", r.get("title", r.get("insight", "?")))
                score = r.get("_score", 0)
                if tps is not None:
                    lines.append(f"  [{agent}] tokens/s={tps:.1f} ({status}) — {desc} (relevance={score:.2f})")
                else:
                    lines.append(f"  [{agent}] {desc} (relevance={score:.2f})")

            summary = "\n".join(lines)
            print(summary)
            return {
                "relevant_results": relevant,
                "best_match": best_match,
                "namespace_searched": namespace or "all",
                "summary": summary,
            }

        except Exception as e:
            self._log(f"ask_swarm error: {e}")
            return {"relevant_results": [], "best_match": None, "namespace_searched": namespace or "all", "summary": f"Error: {e}"}

    def list_namespace(self, namespace: str, limit: int = 50) -> list[dict]:
        """List all keys under a namespace prefix (results, claims, insights, hypotheses)."""
        try:
            result = self._rpc("list_keys", {
                "prefix": f"@{HUB_ORG}/{namespace}/",
                "limit": limit,
            })
            keys = result.get("keys", [])
            entries = []
            for k in keys:
                if isinstance(k, dict):
                    entries.append(k)
                elif isinstance(k, str):
                    entries.append({"key_name": k})
            for e in entries:
                print(f"  {e.get('key_name', e)}")
            return entries

        except Exception as e:
            self._log(f"list_namespace error: {e}")
            return []

    def get_swarm_insights(self, topic: str) -> list[dict]:
        """Search insights by topic to see what the group has learned."""
        try:
            result = self._rpc("search_memories", {
                "query": topic,
                "limit": 10,
                "prefix": f"@{HUB_ORG}/insights/",
            })
            insights = []
            for match in result.get("results", []):
                try:
                    data = json.loads(match.get("value", "{}"))
                    data["_score"] = match.get("score", 0)
                    data["_key"] = match.get("key_name", "")
                    insights.append(data)
                except (json.JSONDecodeError, KeyError):
                    pass

            for i in insights:
                agent = i.get("agent_id", "?")
                text = i.get("insight", i.get("text", "?"))
                print(f"  [{agent}] {text[:80]}")
            return insights

        except Exception as e:
            self._log(f"get_swarm_insights error: {e}")
            return []

    def get_unclaimed_hypotheses(self, limit: int = 10) -> list[dict]:
        """Get hypotheses that haven't been claimed/tested yet."""
        try:
            result = self._rpc("search_memories", {
                "query": "autoresearch hypothesis experiment suggestion ANE inference",
                "limit": limit,
                "prefix": f"@{HUB_ORG}/hypotheses/",
            })
            hypotheses = []
            for match in result.get("results", []):
                try:
                    hyp = json.loads(match.get("value", "{}"))
                    hypotheses.append(hyp)
                except (json.JSONDecodeError, KeyError):
                    pass

            for h in hypotheses:
                print(f"  [P{h.get('priority', '?')}] {h.get('title', '?')}")
            return hypotheses

        except Exception as e:
            self._log(f"get_unclaimed_hypotheses error: {e}")
            return []

    def get_recent_results(self, limit: int = 20) -> list[dict]:
        """Get recent experiment results from the swarm."""
        try:
            result = self._rpc("search_memories", {
                "query": "autoresearch experiment result tokens_per_sec ANE inference",
                "limit": limit,
                "prefix": f"@{HUB_ORG}/results/",
            })
            results = []
            for match in result.get("results", []):
                try:
                    data = json.loads(match.get("value", "{}"))
                    data["_score"] = match.get("score", 0)
                    data["_key"] = match.get("key_name", "")
                    results.append(data)
                except (json.JSONDecodeError, KeyError):
                    pass
            return results

        except Exception as e:
            self._log(f"get_recent_results error: {e}")
            return []

    # --- Insights & Hypotheses ---

    def post_insight(self, insight: str, evidence_keys: list[str] = None) -> None:
        """Post an observation/learning to the collective."""
        try:
            slug = _slugify(insight)
            agent = _slugify(self.agent_id or "unknown", max_len=20)
            short_hash = hashlib.sha256(insight.encode()).hexdigest()[:6]
            insight_key = f"@{HUB_ORG}/insights/{agent}--{slug}--{short_hash}"

            insight_data = {
                "agent_id": self.agent_id or "unknown",
                "insight": insight,
                "evidence_keys": evidence_keys or [],
                "posted_at": _now_iso(),
            }

            value_b64 = base64.b64encode(json.dumps(insight_data).encode()).decode()
            self._rpc("create_memory", {"items": [{
                "key_name": insight_key,
                "description": f"[autoresearch] Insight by {self.agent_id or 'unknown'}: {insight}",
                "value": value_b64,
                "base64": True,
                "embed": True,
                "embed_source": "description",
            }]})

            self._log(f"Published insight: {insight}")

        except Exception as e:
            self._log(f"post_insight error: {e}")

    def publish_hypothesis(
        self,
        title: str,
        hypothesis: str,
        suggested_config: Optional[dict] = None,
        evidence_keys: Optional[list[str]] = None,
        priority: int = 3,
    ) -> None:
        """Publish a research hypothesis for other agents to consider."""
        try:
            slug = _slugify(title)
            agent = _slugify(self.agent_id or "unknown", max_len=20)
            short_hash = hashlib.sha256(title.encode()).hexdigest()[:6]
            hyp_key = f"@{HUB_ORG}/hypotheses/{agent}--{slug}--{short_hash}"

            hyp_data = {
                "agent_id": self.agent_id or "unknown",
                "title": title,
                "hypothesis": hypothesis,
                "suggested_config": suggested_config,
                "evidence_keys": evidence_keys or [],
                "priority": priority,
                "created_at": _now_iso(),
            }

            value_b64 = base64.b64encode(json.dumps(hyp_data).encode()).decode()
            self._rpc("create_memory", {"items": [{
                "key_name": hyp_key,
                "description": f"[autoresearch] Hypothesis: {title}",
                "value": value_b64,
                "base64": True,
                "embed": True,
                "embed_source": "description",
            }]})

            self._log(f"Published hypothesis: {title}")

        except Exception as e:
            self._log(f"publish_hypothesis error: {e}")

    def search_experiments(self, query: str, limit: int = 10) -> list[dict]:
        """Semantic search over past experiment results."""
        try:
            result = self._rpc("search_memories", {
                "query": query,
                "limit": limit,
                "prefix": f"@{HUB_ORG}/results/",
            })
            results = []
            for match in result.get("results", []):
                try:
                    data = json.loads(match.get("value", "{}"))
                    data["_score"] = match.get("score", 0)
                    data["_key"] = match.get("key_name", "")
                    results.append(data)
                except (json.JSONDecodeError, KeyError):
                    pass
            return results

        except Exception as e:
            self._log(f"search_experiments error: {e}")
            return []

    # --- Analysis ---

    def analyze(self) -> str:
        """Print a summary analysis (solo mode, local namespace)."""
        try:
            best = self.pull_best()
            lines = []
            lines.append("=" * 60)
            lines.append("ANE Inference Speed Research Summary")
            lines.append("=" * 60)

            lines.append(f"\nBest config: {best.get('description', 'none')}")
            if best.get("tokens_per_sec"):
                lines.append(f"  tokens/s: {best['tokens_per_sec']:.1f}")

            output = "\n".join(lines)
            print(output)
            return output
        except Exception as e:
            print(f"analyze error: {e}")
            return f"Error: {e}"

    def analyze_swarm(self) -> dict:
        """Comprehensive swarm state analysis."""
        try:
            # Global best
            global_best = None
            meta_key = f"@{HUB_ORG}/best/metadata"
            meta = self._rpc("get_memory", {"key_names": [meta_key]})
            meta_results = meta.get("results", [])
            if meta_results and meta_results[0].get("status") == "success":
                global_best = json.loads(meta_results[0].get("value", "{}"))

            # Recent results
            result_search = self._rpc("search_memories", {
                "query": "experiment result tokens_per_sec ANE inference",
                "limit": 30,
                "prefix": f"@{HUB_ORG}/results/",
            })
            all_results = []
            for match in result_search.get("results", []):
                try:
                    data = json.loads(match.get("value", "{}"))
                    data["_key"] = match.get("key_name", "")
                    all_results.append(data)
                except (json.JSONDecodeError, KeyError):
                    pass

            # Higher is better
            recent_keeps = sorted(
                [r for r in all_results if r.get("status") == "keep"],
                key=lambda x: x.get("tokens_per_sec", 0),
                reverse=True,
            )
            recent_failures = [r for r in all_results if r.get("status") in ("discard", "crash")]

            # Active claims
            claim_search = self._rpc("search_memories", {
                "query": "autoresearch claim experiment",
                "limit": 20,
                "prefix": f"@{HUB_ORG}/claims/",
            })
            active_claims = []
            for match in claim_search.get("results", []):
                try:
                    data = json.loads(match.get("value", "{}"))
                    claimed_at = data.get("claimed_at", "")
                    if claimed_at:
                        claimed_time = datetime.fromisoformat(claimed_at)
                        age = (datetime.now(timezone.utc) - claimed_time).total_seconds()
                        if age < CLAIM_TTL:
                            active_claims.append(data)
                except Exception:
                    pass

            # Unclaimed hypotheses
            unclaimed = self.get_unclaimed_hypotheses(limit=5)

            # Per-agent bests
            agent_bests = self.get_all_agent_bests()

            # Per-tier bests
            tier_bests = self.get_all_tier_bests()
            has_tier_data = any(v is not None for v in tier_bests.values())

            # Build summary
            lines = ["=" * 54, "  SWARM ANALYSIS", "=" * 54]
            if global_best:
                lines.append(f"Global best: tokens/s={global_best.get('tokens_per_sec', '?'):.1f} by {global_best.get('agent_id', global_best.get('achieved_by', '?'))}")
                if global_best.get("achieved_at"):
                    lines.append(f"  Achieved at: {global_best['achieved_at']}")
                if global_best.get("description"):
                    lines.append(f"  Description: {global_best['description']}")
            else:
                lines.append("Global best: none yet")

            lines.append(f"\nKeeps ({len(recent_keeps)}):")
            for r in recent_keeps[:5]:
                lines.append(f"  [{r.get('agent_id', '?')}] tokens/s={r.get('tokens_per_sec', 0):.1f} — {r.get('description', '?')}")

            lines.append(f"\nFailures ({len(recent_failures)}):")
            for r in recent_failures[:5]:
                lines.append(f"  [{r.get('agent_id', '?')}] {r.get('status', '?')} — {r.get('description', '?')}")

            lines.append(f"\nActive claims ({len(active_claims)}):")
            for c in active_claims:
                lines.append(f"  [{c.get('agent_id', '?')}] {c.get('description', '?')}")

            lines.append(f"\nUnclaimed hypotheses ({len(unclaimed)}):")
            for h in unclaimed[:3]:
                lines.append(f"  {h.get('title', '?')} (priority={h.get('priority', '?')})")

            lines.append(f"\nAgent personal bests ({len(agent_bests)}):")
            for ab in agent_bests:
                prev = ab.get("previous_best_tokens_per_sec")
                improvement = f" (improved +{ab['tokens_per_sec'] - prev:.1f} from {prev:.1f})" if prev else ""
                tier_tag = f" [{ab.get('chip_tier', '?')}]" if ab.get("chip_tier") else ""
                lines.append(f"  [{ab.get('agent_id', '?')}]{tier_tag} tokens/s={ab.get('tokens_per_sec', 0):.1f}{improvement} — {ab.get('description', '?')}")

            if has_tier_data:
                lines.append(f"\nChip tier bests:")
                for tier_name, tb in tier_bests.items():
                    tops = CHIP_TIERS[tier_name]
                    label = f"≤{tops} TOPS"
                    if tb:
                        lines.append(f"  {tier_name} ({label}): tokens/s={tb.get('tokens_per_sec', 0):.1f} by {tb.get('agent_id', tb.get('achieved_by', '?'))} — {tb.get('description', '?')}")
                    else:
                        lines.append(f"  {tier_name} ({label}): no results yet")

            lines.append("=" * 54)

            summary = "\n".join(lines)
            print(summary)

            return {
                "global_best": global_best,
                "recent_keeps": recent_keeps,
                "recent_failures": recent_failures,
                "active_claims": active_claims,
                "unclaimed_hypotheses": unclaimed,
                "agent_bests": agent_bests,
                "tier_bests": tier_bests,
                "summary": summary,
            }

        except Exception as e:
            self._log(f"analyze_swarm error: {e}")
            return {
                "global_best": None, "recent_keeps": [], "recent_failures": [],
                "active_claims": [], "unclaimed_hypotheses": [], "agent_bests": [],
                "tier_bests": {}, "summary": f"Error: {e}",
            }

    # --- Dedup / Prior Art (backward compat) ---

    def check_tried(self, description: str, limit: int = 3) -> list[dict]:
        """Search for similar past experiments."""
        matches = []
        try:
            result = self._rpc("search_memories", {
                "query": description,
                "limit": limit,
                "prefix": f"@{HUB_ORG}/results/",
            })
            matches = result.get("results", [])
        except Exception:
            # Also try local namespace
            pass

        if not matches:
            try:
                result = self._rpc("search_memories", {
                    "query": description,
                    "limit": limit,
                    "prefix": f"{NAMESPACE}/results/",
                })
                matches = result.get("results", [])
            except Exception:
                pass

        if matches:
            print(f"Found {len(matches)} similar past experiments:")
            for m in matches:
                if isinstance(m, dict):
                    val = m.get("value", "")
                    try:
                        data = json.loads(val) if isinstance(val, str) else val
                        desc = data.get("description", m.get("key_name", ""))
                        status = data.get("status", "?")
                        tps = data.get("tokens_per_sec", data.get("benchmark", {}).get("tokens_per_sec", "?"))
                        print(f"  [{status}] {str(desc)[:60]} (tokens/s={tps})")
                    except (json.JSONDecodeError, AttributeError):
                        print(f"  {m.get('key_name', m)}")
        else:
            print("No similar experiments found.")
        return matches


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli():
    """CLI interface for calling coordinator methods from shell."""
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/coordinator.py <command> [args...]")
        print()
        print("Commands:")
        print("  analyze              Local summary of best result")
        print("  analyze_swarm        Full swarm state (claims, results, hypotheses, bests)")
        print("  announce             Print startup banner with swarm state")
        print("  join_hub             Join the autoresearch-at-home hub")
        print("  publish_result       '<desc>' '<bench_json>' ['<git_diff>'] [status]")
        print("  check_tried          '<idea>'")
        print("  claim                '<description>'")
        print("  publish_hypothesis   '<title>' '<hypothesis>' [priority]")
        print("  list_hypotheses      [status]")
        print("  get_unclaimed_hypotheses")
        print("  post_insight         '<text>' [evidence_key1,key2,...]")
        print("  get_swarm_insights   '<topic>'")
        print("  pull_best            Get current global best")
        print("  pull_best_for_tier   [tier]  Get best for chip tier")
        print("  ask                  '<query>' [namespace]")
        print("  list_namespace       '<namespace>'")
        sys.exit(1)

    coord = Coordinator()
    cmd = sys.argv[1]

    if cmd == "analyze":
        if not coord.connected:
            print("ERROR: Cannot connect to Ensue. Check API key.")
            sys.exit(1)
        coord.analyze()

    elif cmd == "analyze_swarm":
        if not coord.connected:
            print("ERROR: Cannot connect to Ensue. Check API key.")
            sys.exit(1)
        coord.analyze_swarm()

    elif cmd == "announce":
        coord.announce()

    elif cmd == "join_hub":
        coord.join_hub()

    elif cmd == "publish_result":
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

    elif cmd == "claim":
        if len(sys.argv) < 3:
            print("Usage: claim '<description>'")
            sys.exit(1)
        result = coord.claim_experiment(sys.argv[2])
        if result:
            print(f"Claimed: {result}")
        else:
            print("Could not claim experiment.")

    elif cmd == "publish_hypothesis":
        if len(sys.argv) < 4:
            print("Usage: publish_hypothesis '<title>' '<hypothesis>' [priority]")
            sys.exit(1)
        title = sys.argv[2]
        hypothesis = sys.argv[3]
        priority = int(sys.argv[4]) if len(sys.argv) > 4 else 3
        coord.publish_hypothesis(title, hypothesis, priority=priority)

    elif cmd == "list_hypotheses":
        hypotheses = coord.get_unclaimed_hypotheses()
        if not hypotheses:
            print("No hypotheses found.")

    elif cmd == "get_unclaimed_hypotheses":
        hypotheses = coord.get_unclaimed_hypotheses()
        if not hypotheses:
            print("No unclaimed hypotheses.")

    elif cmd == "post_insight":
        if len(sys.argv) < 3:
            print("Usage: post_insight '<text>' [evidence_key1,key2,...]")
            sys.exit(1)
        text = sys.argv[2]
        evidence = sys.argv[3].split(",") if len(sys.argv) > 3 else []
        coord.post_insight(text, evidence)

    elif cmd == "get_swarm_insights":
        if len(sys.argv) < 3:
            print("Usage: get_swarm_insights '<topic>'")
            sys.exit(1)
        coord.get_swarm_insights(sys.argv[2])

    elif cmd == "pull_best":
        best = coord.pull_best()
        print(json.dumps(best, indent=2))

    elif cmd == "pull_best_for_tier":
        tier = sys.argv[2] if len(sys.argv) > 2 else None
        best = coord.pull_best_for_tier(tier)
        print(json.dumps(best, indent=2))

    elif cmd == "ask":
        if len(sys.argv) < 3:
            print("Usage: ask '<query>' [namespace]")
            sys.exit(1)
        query = sys.argv[2]
        namespace = sys.argv[3] if len(sys.argv) > 3 else None
        coord.ask_swarm(query, namespace)

    elif cmd == "list_namespace":
        if len(sys.argv) < 3:
            print("Usage: list_namespace '<namespace>'")
            sys.exit(1)
        coord.list_namespace(sys.argv[2])

    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)


if __name__ == "__main__":
    _cli()
