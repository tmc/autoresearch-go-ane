#!/usr/bin/env python3
"""Convert HuggingFace Qwen3/Llama models to .bin format for autoresearch-go-ane.

Usage:
    python3 scripts/convert_hf.py Qwen/Qwen3-4B --output qwen3-4b.bin
    python3 scripts/convert_hf.py Qwen/Qwen3-4B --output qwen3-4b.bin --verify

The output .bin format has:
    1. Llama2Config header (7 x int32 = 28 bytes)
    2. Embedding matrix [vocab, dim] as float32
    3. Per-layer RMSAtt weights [dim] as float32
    4. Per-layer Wq [dim, dim] as float32
    5. Per-layer Wk [dim, kvDim] as float32
    6. Per-layer Wv [dim, kvDim] as float32
    7. Per-layer Wo [dim, dim] as float32
    8. Per-layer RMSFFN weights [dim] as float32
    9. Per-layer W1 [hidden, dim] as float32
   10. Per-layer W2 [dim, hidden] as float32
   11. Per-layer W3 [hidden, dim] as float32
   12. RMSFinal [dim] as float32
"""

import argparse
import struct
import sys
from pathlib import Path

import numpy as np


def load_model(model_id: str):
    """Load model from HuggingFace, returning config and state_dict."""
    try:
        from transformers import AutoConfig
        from safetensors import safe_open
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Install requirements: pip install transformers safetensors huggingface_hub torch")
        sys.exit(1)

    config = AutoConfig.from_pretrained(model_id)

    # Download safetensors
    model_path = snapshot_download(model_id, allow_patterns=["*.safetensors", "*.json"])
    model_path = Path(model_path)

    # Load all safetensor files.
    # Use torch framework to handle bfloat16, then convert to float32 numpy.
    import torch
    state_dict = {}
    for sf_path in sorted(model_path.glob("*.safetensors")):
        with safe_open(str(sf_path), framework="pt") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key).to(torch.float32).numpy()

    return config, state_dict


def convert(model_id: str, output_path: str, verify: bool = False):
    """Convert HuggingFace model to .bin format."""
    print(f"Loading {model_id}...")
    config, state_dict = load_model(model_id)

    # Extract config values
    dim = config.hidden_size
    hidden = config.intermediate_size
    n_layers = config.num_hidden_layers
    n_heads = config.num_attention_heads
    n_kv_heads = getattr(config, "num_key_value_heads", n_heads)
    vocab_size = config.vocab_size
    seq_len = getattr(config, "max_position_embeddings", 4096)

    # head_dim may be explicitly set (Qwen3) or derived from dim/heads
    head_dim = getattr(config, "head_dim", dim // n_heads)
    q_dim = n_heads * head_dim      # Q projection output dim
    kv_dim = n_kv_heads * head_dim  # K/V projection output dim

    print(f"Config: dim={dim}, hidden={hidden}, layers={n_layers}, "
          f"heads={n_heads}, kv_heads={n_kv_heads}, vocab={vocab_size}, "
          f"seq={seq_len}, head_dim={head_dim}, q_dim={q_dim}, kv_dim={kv_dim}")

    # Map weight names
    def get(name: str) -> np.ndarray:
        if name not in state_dict:
            raise KeyError(f"Weight '{name}' not found. Available: {sorted(state_dict.keys())[:20]}...")
        return state_dict[name].astype(np.float32)

    # Check if lm_head is tied to embeddings
    embed = get("model.embed_tokens.weight")  # [vocab, dim]
    has_lm_head = "lm_head.weight" in state_dict
    shared_cl = not has_lm_head

    print(f"Embedding: {embed.shape}, shared_classifier={shared_cl}")

    # Write header (Llama2Config format)
    # struct: dim, hidden, n_layers, n_heads, n_kv_heads, vocab_size, seq_len
    # Note: positive vocab means shared classifier, negative means separate
    vocab_field = vocab_size if shared_cl else -vocab_size

    with open(output_path, "wb") as f:
        # Header
        f.write(struct.pack("<7i", dim, hidden, n_layers, n_heads, n_kv_heads,
                            vocab_field, seq_len))

        # Embedding
        f.write(embed.tobytes())

        # Per-layer weights in the order expected by LoadPretrained
        # 1. RMSAtt for all layers
        for i in range(n_layers):
            w = get(f"model.layers.{i}.input_layernorm.weight")
            f.write(w.tobytes())

        # 2. Wq for all layers [q_dim, dim]
        for i in range(n_layers):
            w = get(f"model.layers.{i}.self_attn.q_proj.weight")
            assert w.shape == (q_dim, dim), f"Wq shape mismatch: {w.shape} expected ({q_dim}, {dim})"
            f.write(w.tobytes())

        # 3. Wk for all layers [kv_dim, dim]
        for i in range(n_layers):
            w = get(f"model.layers.{i}.self_attn.k_proj.weight")
            assert w.shape == (kv_dim, dim), f"Wk shape mismatch: {w.shape} expected ({kv_dim}, {dim})"
            f.write(w.tobytes())

        # 4. Wv for all layers [kv_dim, dim]
        for i in range(n_layers):
            w = get(f"model.layers.{i}.self_attn.v_proj.weight")
            assert w.shape == (kv_dim, dim), f"Wv shape mismatch: {w.shape} expected ({kv_dim}, {dim})"
            f.write(w.tobytes())

        # 5. Wo for all layers [dim, q_dim] (projects attention output back to hidden dim)
        for i in range(n_layers):
            w = get(f"model.layers.{i}.self_attn.o_proj.weight")
            assert w.shape == (dim, q_dim), f"Wo shape mismatch: {w.shape} expected ({dim}, {q_dim})"
            f.write(w.tobytes())

        # 6. RMSFFN for all layers
        for i in range(n_layers):
            w = get(f"model.layers.{i}.post_attention_layernorm.weight")
            f.write(w.tobytes())

        # 7. W1 (gate_proj) for all layers [hidden, dim]
        for i in range(n_layers):
            w = get(f"model.layers.{i}.mlp.gate_proj.weight")
            assert w.shape == (hidden, dim), f"W1 shape mismatch: {w.shape} expected ({hidden}, {dim})"
            f.write(w.tobytes())

        # 8. W2 (down_proj) for all layers [dim, hidden]
        for i in range(n_layers):
            w = get(f"model.layers.{i}.mlp.down_proj.weight")
            assert w.shape == (dim, hidden), f"W2 shape mismatch: {w.shape} expected ({dim}, {hidden})"
            f.write(w.tobytes())

        # 9. W3 (up_proj) for all layers [hidden, dim]
        for i in range(n_layers):
            w = get(f"model.layers.{i}.mlp.up_proj.weight")
            assert w.shape == (hidden, dim), f"W3 shape mismatch: {w.shape} expected ({hidden}, {dim})"
            f.write(w.tobytes())

        # 10. RMSFinal
        w = get("model.norm.weight")
        f.write(w.tobytes())

    file_size = Path(output_path).stat().st_size
    print(f"Wrote {output_path} ({file_size / 1e9:.2f} GB)")

    if verify:
        verify_output(output_path, config, state_dict)


def verify_output(path: str, config, state_dict: dict):
    """Verify the output file can be read back correctly."""
    print("\nVerifying output...")

    dim = config.hidden_size
    n_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
    head_dim = dim // config.num_attention_heads
    kv_dim = n_kv_heads * head_dim

    with open(path, "rb") as f:
        # Read header
        header = struct.unpack("<7i", f.read(28))
        print(f"  Header: dim={header[0]}, hidden={header[1]}, layers={header[2]}, "
              f"heads={header[3]}, kv_heads={header[4]}, vocab={header[5]}, seq={header[6]}")

        assert header[0] == config.hidden_size
        assert header[1] == config.intermediate_size
        assert header[2] == config.num_hidden_layers

        # Read and verify embedding
        embed = np.frombuffer(f.read(config.vocab_size * dim * 4), dtype=np.float32)
        embed = embed.reshape(config.vocab_size, dim)
        ref = state_dict["model.embed_tokens.weight"].astype(np.float32)
        max_err = np.max(np.abs(embed - ref))
        print(f"  Embedding max error: {max_err:.2e}")
        assert max_err < 1e-6, f"Embedding mismatch: max_err={max_err}"

        # Verify first layer RMSAtt
        rms = np.frombuffer(f.read(dim * 4), dtype=np.float32)
        ref = state_dict["model.layers.0.input_layernorm.weight"].astype(np.float32)
        max_err = np.max(np.abs(rms - ref))
        print(f"  Layer 0 RMSAtt max error: {max_err:.2e}")
        assert max_err < 1e-6

    print("  Verification passed!")


def main():
    parser = argparse.ArgumentParser(description="Convert HuggingFace model to .bin format")
    parser.add_argument("model_id", help="HuggingFace model ID (e.g., Qwen/Qwen3-4B)")
    parser.add_argument("--output", "-o", required=True, help="Output .bin file path")
    parser.add_argument("--verify", action="store_true", help="Verify output after conversion")
    args = parser.parse_args()

    convert(args.model_id, args.output, args.verify)


if __name__ == "__main__":
    main()
