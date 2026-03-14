#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

DATA_URL="https://huggingface.co/datasets/enio/TinyStories/resolve/main/tok32000/TinyStories_tok32000.tar.gz"
MODEL_URL="https://huggingface.co/karpathy/tinyllamas/resolve/main/stories110M.pt"

# Token data.
if [ ! -f tinystories_data00.bin ]; then
    echo "downloading TinyStories token data..."
    curl -L "$DATA_URL" | tar xz --include='tinystories_data00.bin' 2>/dev/null \
        || { curl -L -o tinystories.tar.gz "$DATA_URL" && tar xzf tinystories.tar.gz tinystories_data00.bin && rm tinystories.tar.gz; }
    echo "downloaded tinystories_data00.bin ($(du -h tinystories_data00.bin | cut -f1))"
else
    echo "tinystories_data00.bin already exists"
fi

# Model checkpoint (optional — random-init mode doesn't need it).
if [ ! -f stories110M.bin ]; then
    echo "downloading stories110M model checkpoint..."
    curl -L -o stories110M.bin "$MODEL_URL"
    echo "downloaded stories110M.bin ($(du -h stories110M.bin | cut -f1))"
else
    echo "stories110M.bin already exists"
fi

echo ""
echo "setup complete. run:"
echo "  go run . -data tinystories_data00.bin"
echo ""
echo "or run benchmarks:"
echo "  go test -bench . -benchtime 5x -v"
