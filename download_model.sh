#!/usr/bin/env bash
# Downloads the GPT-2 small (124M) files from Hugging Face into model/.
# Files that are already there are skipped.
set -euo pipefail

BASE_URL="https://huggingface.co/openai-community/gpt2/resolve/main"
DIR="$(dirname "$0")/model"
mkdir -p "$DIR"

for f in vocab.json merges.txt model.safetensors; do
    if [ -s "$DIR/$f" ]; then
        echo "have $f"
    else
        echo "downloading $f ..."
        curl -fL --progress-bar -o "$DIR/$f.part" "$BASE_URL/$f"
        mv "$DIR/$f.part" "$DIR/$f" # only keep it once fully downloaded
    fi
done
echo "done: model files are in $DIR"
