#!/usr/bin/env python3
"""Tokenizer wrapper for use with Go inference.

Usage:
    # Encode text to token IDs (one per line)
    echo "def fibonacci(n):" | python3 scripts/tokenize.py encode Qwen/Qwen3-4B

    # Decode token IDs (one per line) to text
    echo -e "755\n38837\n7" | python3 scripts/tokenize.py decode Qwen/Qwen3-4B

    # Interactive mode: encode lines from stdin
    python3 scripts/tokenize.py encode Qwen/Qwen3-4B --interactive
"""

import argparse
import sys


def get_tokenizer(model_id: str):
    """Load HuggingFace tokenizer."""
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("Install: pip install transformers", file=sys.stderr)
        sys.exit(1)
    return AutoTokenizer.from_pretrained(model_id)


def cmd_encode(args):
    tok = get_tokenizer(args.model_id)
    if args.interactive:
        print(f"Tokenizer loaded: {args.model_id} (vocab_size={tok.vocab_size})", file=sys.stderr)
        print("Enter text to encode (Ctrl+D to quit):", file=sys.stderr)
    for line in sys.stdin:
        line = line.rstrip("\n")
        ids = tok.encode(line, add_special_tokens=not args.no_special)
        print(",".join(str(i) for i in ids))
        if args.interactive:
            sys.stdout.flush()


def cmd_decode(args):
    tok = get_tokenizer(args.model_id)
    ids = []
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        if "," in line:
            ids.extend(int(x) for x in line.split(",") if x.strip())
        else:
            ids.append(int(line))
    text = tok.decode(ids, skip_special_tokens=args.skip_special)
    print(text, end="")


def cmd_info(args):
    tok = get_tokenizer(args.model_id)
    print(f"model: {args.model_id}")
    print(f"vocab_size: {tok.vocab_size}")
    print(f"bos_token_id: {tok.bos_token_id}")
    print(f"eos_token_id: {tok.eos_token_id}")
    print(f"pad_token_id: {tok.pad_token_id}")
    if hasattr(tok, "added_tokens_encoder"):
        print(f"special_tokens: {len(tok.added_tokens_encoder)}")


def main():
    parser = argparse.ArgumentParser(description="HuggingFace tokenizer wrapper")
    sub = parser.add_subparsers(dest="command", required=True)

    enc = sub.add_parser("encode", help="Encode text to token IDs")
    enc.add_argument("model_id", help="HF model ID")
    enc.add_argument("--no-special", action="store_true", help="Don't add special tokens")
    enc.add_argument("--interactive", action="store_true", help="Interactive mode")
    enc.set_defaults(func=cmd_encode)

    dec = sub.add_parser("decode", help="Decode token IDs to text")
    dec.add_argument("model_id", help="HF model ID")
    dec.add_argument("--skip-special", action="store_true", help="Skip special tokens")
    dec.set_defaults(func=cmd_decode)

    info = sub.add_parser("info", help="Print tokenizer info")
    info.add_argument("model_id", help="HF model ID")
    info.set_defaults(func=cmd_info)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
