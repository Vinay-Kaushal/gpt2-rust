# gpt2-rust

GPT-2 (124M) text generation written from scratch in Rust, running on the CPU.

No ML frameworks, no tensor libraries: the safetensors weight file is parsed by hand,
the byte-level BPE tokenizer, every math op, the transformer forward pass, the KV cache
and the sampler are all written in plain Rust (~1000 lines).

![demo: generating 80 tokens at ~65 tokens/sec](docs/demo.gif)

## Quick start

Needs [Rust](https://rustup.rs) and `curl`.

```bash
git clone https://github.com/Vinay-Kaushal/gpt2-rust.git
cd gpt2-rust
./download_model.sh                 # GPT-2 small from Hugging Face, ~550 MB, into model/
cargo run --release -- "Once upon a time"
```

Always use `--release`: the debug build is about 50x slower (0.9 vs ~50 tokens/sec).

### Options

| Option | Meaning | Default |
|---|---|---|
| `PROMPT` | text to continue (in quotes) | `"The quick brown fox jumps over the lazy"` |
| `-n, --tokens N` | how many new tokens to make | 100 |
| `-k, --top-k K` | pick only from the K likeliest tokens | 40 |
| `-t, --temp T` | randomness: low = safe, high = wild | 0.8 |
| `-s, --seed S` | same seed + same options = same text | clock |
| `--greedy` | always pick the likeliest token (`-k 1`) | |
| `--bench` | speed test: greedy, 50 tokens, prints tokens/sec | |

## How it works

```
 "Hello world"
      │  tokenizer.rs   byte-level BPE: regex pre-split, bytes -> unicode, merge pairs by rank
      ▼
 [15496, 995]
      │  model.rs       word embedding (wte) + position embedding (wpe)
      ▼
 12 x transformer block ──────────────────────────────────────────────┐
      │   layernorm -> attention (12 heads, causal) -> + residual      │ k, v of every token
      │   layernorm -> MLP (768 -> 3072 -> GELU -> 768) -> + residual  │ kept in the KV cache
      ▼                                                                │
 final layernorm -> dot with every wte row -> 50257 scores (logits) ◄──┘
      │  sampler.rs     top-k + temperature + seeded random pick
      ▼
 next token -> printed, fed back in (only the new token is processed)
```

| File | What it does |
|---|---|
| `src/safetensors.rs` | reads the 8-byte header length, the JSON header, then the raw f32 tensors |
| `src/tokenizer.rs` | GPT-2's byte-level BPE encoder/decoder (`vocab.json`, `merges.txt`) |
| `src/ops.rs` | linear, layernorm, softmax, GELU, SIMD-friendly dot product, bf16 conversion |
| `src/model.rs` | loads and shape-checks all weights, forward pass with KV cache |
| `src/sampler.rs` | xorshift random numbers, top-k sampling with temperature |
| `src/main.rs` | command line, generation loop, streaming output |

Dependencies: `serde`/`serde_json` (the safetensors JSON header), `fancy-regex`
(GPT-2's pre-split pattern needs a lookahead), `rayon` (use all CPU cores).

## Correctness

- **Tokenizer** gives the same ids as OpenAI's original `encoder.py` on a set of
  tricky test inputs.
- **Forward pass** with f32 weights matches an independent NumPy implementation of
  GPT-2 on 3 prompts: every logit checked agrees to 3 decimals, and the top-5
  next-token probabilities are identical (e.g. `,` 5.57% · ` fox` 4.25% · ` brown` 3.38%).
- **bf16 weights** keep the same top-5 next tokens, with the same first choice:

| Prompt | f32 top-3 | bf16 top-3 |
|---|---|---|
| `The quick brown fox jumps over the lazy` | `,` 5.6% · ` fox` 4.3% · ` brown` 3.4% | `,` 5.0% · ` fox` 4.3% · ` brown` 4.1% |
| `The capital of France is` | ` the` 8.5% · ` now` 4.8% · ` a` 4.6% | ` the` 9.1% · ` a` 4.3% · ` now` 4.2% |
| `def fibonacci(n):` | `\n` 24.4% · ` return` 5.4% · ` #` 4.5% | `\n` 22.4% · ` #` 5.9% · ` return` 4.9% |

- `cargo test` runs 14 unit tests (math ops against hand-worked values, bf16 rounding,
  sampler behaviour).

## Performance

Speed of `--bench` (8-token prompt + 50 greedy tokens, including the prompt), on a
laptop Intel i5-1155G7 (4 cores, WSL2):

| Step | tokens/sec | Why it helped |
|---|---|---|
| Debug build | ~0.9 | |
| `--release` build | ~16 | compiler optimizations |
| KV cache | (included above) | each new token only processes itself, not the whole text again (7.9s -> 1.4s for 50 tokens) |
| 16-accumulator dot product | ~23 | one running total makes every `+` wait for the previous one; 16 independent totals let the CPU use SIMD |
| Logits on all cores (rayon) | ~33 | the 50257 vocab scores are independent, so each core computes its own share |
| bf16 weights | ~46 | half the bytes to pull from RAM per token (~500 MB -> ~250 MB) |
| Top-k without a full sort | ~50 | `select_nth_unstable` finds the top k without sorting all 50257 scores |

Numbers are medians of back-to-back runs; the laptop's speed drifts ~20% with heat.

**What did not help, and why:** splitting the linear layers across cores (by columns
or by rows with partial sums) was slower or no faster. Timing one layer alone showed a
single thread already reads weights at ~17 GB/s and all threads together top out near
~21 GB/s: the linear layers are limited by memory bandwidth, not arithmetic. That is
why reading fewer bytes (bf16) helped where more threads did not.

## Model

GPT-2 small: 124M parameters, 12 layers, 12 heads, 768 dims, 1024-token context,
50257-token vocabulary. Weights from
[openai-community/gpt2](https://huggingface.co/openai-community/gpt2) (safetensors).
