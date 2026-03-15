#!/usr/bin/env python3
"""Parse Go benchmark output for BenchmarkEvalLogits and emit JSON.

Usage:
    go test -bench BenchmarkEvalLogits -benchtime 5x -count 6 | python3 scripts/parse_bench.py

Output (JSON):
    {
        "benchmark": "BenchmarkEvalLogits",
        "ns_per_op": 12345678,
        "tokens_per_sec": 20736.5,
        "runs": 6,
        "raw_lines": ["BenchmarkEvalLogits-10  5  12345678 ns/op", ...]
    }

tokens_per_sec = (256 * 1e9) / ns_per_op
"""

import json
import re
import sys

SEQ_LEN = 256


def parse_bench_output(text: str) -> dict:
    """Parse Go benchmark lines and compute tokens/s."""
    pattern = re.compile(
        r'^(BenchmarkEvalLogits\S*)\s+(\d+)\s+(\d+(?:\.\d+)?)\s+ns/op'
    )

    ns_values = []
    raw_lines = []

    for line in text.splitlines():
        m = pattern.match(line.strip())
        if m:
            ns = float(m.group(3))
            ns_values.append(ns)
            raw_lines.append(line.strip())

    if not ns_values:
        return {
            "benchmark": "BenchmarkEvalLogits",
            "error": "no matching benchmark lines found",
            "raw_input": text[:500],
        }

    # Use median ns/op for the summary
    ns_values.sort()
    median_ns = ns_values[len(ns_values) // 2]
    tokens_per_sec = (SEQ_LEN * 1e9) / median_ns if median_ns > 0 else 0

    return {
        "benchmark": "BenchmarkEvalLogits",
        "ns_per_op": int(median_ns),
        "tokens_per_sec": round(tokens_per_sec, 1),
        "runs": len(ns_values),
        "raw_lines": raw_lines,
    }


if __name__ == "__main__":
    text = sys.stdin.read()
    result = parse_bench_output(text)
    print(json.dumps(result, indent=2))
