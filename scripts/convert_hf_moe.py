#!/usr/bin/env python3
"""Convert HuggingFace MoE models (Qwen3-Coder-Next) to binary format.

Usage:
    python3 scripts/convert_hf_moe.py Qwen/Qwen3-Coder-Next --output qwen3-coder-next.moe.bin

Output format (all little-endian):
    Magic: "MOE1" (4 bytes)
    Header: JSON config (length-prefixed uint32 + UTF-8 bytes)
    Embedding: [vocab, dim] float32
    Per-layer:
      RMSAtt: [dim] float32
      Wq: [dim, dim] float32
      Wk: [kv_dim, dim] float32
      Wv: [kv_dim, dim] float32
      Wo: [dim, dim] float32
      RMSFFN: [dim] float32
      RouterWeight: [num_experts, dim] float32
      SharedExpert W1: [expert_hidden, dim] float32  (if has_shared_expert)
      SharedExpert W2: [dim, expert_hidden] float32
      SharedExpert W3: [expert_hidden, dim] float32
      For each expert 0..num_experts-1:
        W1: [expert_hidden, dim] float32
        W2: [dim, expert_hidden] float32
        W3: [expert_hidden, dim] float32
    RMSFinal: [dim] float32
"""

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np


def load_model(model_id: str):
    """Load model from HuggingFace."""
    try:
        from transformers import AutoConfig
        from safetensors import safe_open
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Install: pip install transformers safetensors huggingface_hub")
        sys.exit(1)

    config = AutoConfig.from_pretrained(model_id)
    model_path = Path(snapshot_download(model_id, allow_patterns=["*.safetensors", "*.json"]))

    state_dict = {}
    for sf_path in sorted(model_path.glob("*.safetensors")):
        with safe_open(str(sf_path), framework="numpy") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)

    return config, state_dict


def detect_moe_config(config):
    """Extract MoE configuration from HuggingFace config."""
    dim = config.hidden_size
    n_heads = config.num_attention_heads
    n_kv_heads = getattr(config, "num_key_value_heads", n_heads)
    n_layers = config.num_hidden_layers
    vocab = config.vocab_size
    seq = getattr(config, "max_position_embeddings", 262144)

    # MoE-specific
    num_experts = getattr(config, "num_experts", getattr(config, "num_local_experts", 0))
    num_active = getattr(config, "num_experts_per_tok",
                         getattr(config, "num_selected_experts", 10))
    expert_hidden = getattr(config, "moe_intermediate_size",
                           getattr(config, "expert_intermediate_size",
                                   config.intermediate_size))
    has_shared = getattr(config, "shared_expert_intermediate_size", 0) > 0

    # Layer types (for hybrid attention models like Qwen3-Coder-Next)
    layer_types = getattr(config, "layer_types", ["attention"] * n_layers)

    return {
        "dim": dim,
        "hidden": config.intermediate_size,
        "heads": n_heads,
        "kv_heads": n_kv_heads,
        "n_layers": n_layers,
        "vocab": vocab,
        "seq": seq,
        "num_experts": num_experts,
        "num_active_experts": num_active,
        "expert_hidden": expert_hidden,
        "has_shared_expert": has_shared,
        "layer_types": layer_types,
    }


def convert_moe(model_id: str, output_path: str):
    """Convert MoE model to binary format."""
    print(f"Loading {model_id}...")
    config, state_dict = load_model(model_id)
    moe_cfg = detect_moe_config(config)

    dim = moe_cfg["dim"]
    n_layers = moe_cfg["n_layers"]
    n_kv_heads = moe_cfg["kv_heads"]
    n_heads = moe_cfg["heads"]
    head_dim = dim // n_heads
    kv_dim = n_kv_heads * head_dim
    num_experts = moe_cfg["num_experts"]
    expert_hidden = moe_cfg["expert_hidden"]
    has_shared = moe_cfg["has_shared_expert"]

    print(f"MoE Config: {json.dumps(moe_cfg, indent=2, default=str)}")

    def get(name: str) -> np.ndarray:
        if name not in state_dict:
            available = [k for k in sorted(state_dict.keys()) if "layer" not in k or "layers.0" in k]
            raise KeyError(f"'{name}' not found. Sample keys: {available[:20]}")
        return state_dict[name].astype(np.float32)

    def try_get(name: str):
        if name in state_dict:
            return state_dict[name].astype(np.float32)
        return None

    with open(output_path, "wb") as f:
        # Magic
        f.write(b"MOE1")

        # JSON config header (length-prefixed)
        cfg_json = json.dumps(moe_cfg).encode("utf-8")
        f.write(struct.pack("<I", len(cfg_json)))
        f.write(cfg_json)

        # Embedding
        embed = get("model.embed_tokens.weight")
        print(f"Embedding: {embed.shape}")
        f.write(embed.tobytes())

        for i in range(n_layers):
            prefix = f"model.layers.{i}"
            print(f"  Layer {i}/{n_layers}...", end="\r")

            # RMSAtt
            f.write(get(f"{prefix}.input_layernorm.weight").tobytes())

            # Attention weights
            f.write(get(f"{prefix}.self_attn.q_proj.weight").tobytes())
            f.write(get(f"{prefix}.self_attn.k_proj.weight").tobytes())
            f.write(get(f"{prefix}.self_attn.v_proj.weight").tobytes())
            f.write(get(f"{prefix}.self_attn.o_proj.weight").tobytes())

            # RMSFFN
            f.write(get(f"{prefix}.post_attention_layernorm.weight").tobytes())

            # Router
            router = try_get(f"{prefix}.mlp.gate.weight")
            if router is None:
                router = try_get(f"{prefix}.mlp.router.weight")
            if router is None:
                raise KeyError(f"Router weight not found for layer {i}")
            assert router.shape == (num_experts, dim), f"Router shape {router.shape}"
            f.write(router.tobytes())

            # Shared expert (if present)
            if has_shared:
                for proj in ["gate_proj", "down_proj", "up_proj"]:
                    w = get(f"{prefix}.mlp.shared_expert.{proj}.weight")
                    f.write(w.tobytes())

            # Per-expert weights
            for e in range(num_experts):
                for proj in ["gate_proj", "down_proj", "up_proj"]:
                    w = try_get(f"{prefix}.mlp.experts.{e}.{proj}.weight")
                    if w is None:
                        # Some models use different naming
                        w = get(f"{prefix}.block_sparse_moe.experts.{e}.{proj}.weight")
                    f.write(w.tobytes())

        # RMSFinal
        f.write(get("model.norm.weight").tobytes())

    file_size = Path(output_path).stat().st_size
    print(f"\nWrote {output_path} ({file_size / 1e9:.2f} GB)")
    print(f"Total experts: {num_experts} × {n_layers} layers = {num_experts * n_layers} expert FFNs")
    print(f"Active per token: {moe_cfg['num_active_experts']}" +
          (f" + 1 shared" if has_shared else ""))


def main():
    parser = argparse.ArgumentParser(description="Convert HuggingFace MoE model")
    parser.add_argument("model_id", help="HF model ID (e.g., Qwen/Qwen3-Coder-Next)")
    parser.add_argument("--output", "-o", required=True, help="Output .moe.bin file")
    args = parser.parse_args()
    convert_moe(args.model_id, args.output)


if __name__ == "__main__":
    main()
