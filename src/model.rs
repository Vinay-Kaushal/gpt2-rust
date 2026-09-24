// Phase 5: the full GPT-2 forward pass. Phase 6: with a KV cache.
// token ids -> 12 blocks of (attention + MLP) -> a score for every word in the vocab.

use crate::ops;
use crate::safetensors::SafeTensors;
use std::error::Error;

// GPT-2 small's fixed sizes
const VOCAB: usize = 50257; // words in the dictionary
const N_LAYER: usize = 12; // number of blocks
const N_HEAD: usize = 12; // attention heads per block
const DIM: usize = 768; // numbers per token
const HEAD_DIM: usize = DIM / N_HEAD; // 64 numbers per head
const MLP_DIM: usize = 4 * DIM; // 3072
pub const MAX_POS: usize = 1024; // longest input GPT-2 can read

// All the weights of one block
struct Block {
    ln_1_w: Vec<f32>,
    ln_1_b: Vec<f32>,
    attn_w: Vec<f32>, // [768, 2304]: makes q, k, v
    attn_b: Vec<f32>,
    attn_proj_w: Vec<f32>, // [768, 768]
    attn_proj_b: Vec<f32>,
    ln_2_w: Vec<f32>,
    ln_2_b: Vec<f32>,
    fc_w: Vec<f32>, // [768, 3072]: MLP up
    fc_b: Vec<f32>,
    proj_w: Vec<f32>, // [3072, 768]: MLP down
    proj_b: Vec<f32>,
}

pub struct Gpt2 {
    wte: Vec<f32>, // [50257, 768] word table
    wpe: Vec<f32>, // [1024, 768]  position table
    blocks: Vec<Block>,
    ln_f_w: Vec<f32>,
    ln_f_b: Vec<f32>,
    pub vocab_size: usize,
}

impl Gpt2 {
    pub fn load(st: &SafeTensors) -> Result<Gpt2, Box<dyn Error>> {
        // small helper: get a tensor's numbers by name, and check its shape
        let get = |name: &str, shape: &[usize]| -> Result<Vec<f32>, Box<dyn Error>> {
            let t = st.tensor(name)?;
            if t.shape != shape {
                return Err(format!("{name}: expected shape {shape:?}, got {:?}", t.shape).into());
            }
            Ok(t.data)
        };

        let mut blocks = Vec::with_capacity(N_LAYER);
        for l in 0..N_LAYER {
            let p = format!("h.{l}.");
            blocks.push(Block {
                ln_1_w: get(&format!("{p}ln_1.weight"), &[DIM])?,
                ln_1_b: get(&format!("{p}ln_1.bias"), &[DIM])?,
                attn_w: get(&format!("{p}attn.c_attn.weight"), &[DIM, 3 * DIM])?,
                attn_b: get(&format!("{p}attn.c_attn.bias"), &[3 * DIM])?,
                attn_proj_w: get(&format!("{p}attn.c_proj.weight"), &[DIM, DIM])?,
                attn_proj_b: get(&format!("{p}attn.c_proj.bias"), &[DIM])?,
                ln_2_w: get(&format!("{p}ln_2.weight"), &[DIM])?,
                ln_2_b: get(&format!("{p}ln_2.bias"), &[DIM])?,
                fc_w: get(&format!("{p}mlp.c_fc.weight"), &[DIM, MLP_DIM])?,
                fc_b: get(&format!("{p}mlp.c_fc.bias"), &[MLP_DIM])?,
                proj_w: get(&format!("{p}mlp.c_proj.weight"), &[MLP_DIM, DIM])?,
                proj_b: get(&format!("{p}mlp.c_proj.bias"), &[DIM])?,
            });
        }

        Ok(Gpt2 {
            wte: get("wte.weight", &[VOCAB, DIM])?,
            wpe: get("wpe.weight", &[MAX_POS, DIM])?,
            blocks,
            ln_f_w: get("ln_f.weight", &[DIM])?,
            ln_f_b: get("ln_f.bias", &[DIM])?,
            vocab_size: VOCAB,
        })
    }

    // Runs only the NEW tokens `ids` through the model.
    // Earlier tokens are already in `cache` (their k and v), so they are not redone.
    // Returns one score (logit) per vocab word for the token after the last one.
    pub fn forward(&self, ids: &[u32], cache: &mut KvCache) -> Vec<f32> {
        let n = ids.len(); // how many new tokens
        let start = cache.len; // position of the first new token
        assert!(n > 0 && start + n <= MAX_POS, "need 1..=1024 tokens in total");

        // 1. embeddings: word numbers + position numbers, one row per new token
        let mut x = vec![0.0; n * DIM];
        for (t, &id) in ids.iter().enumerate() {
            let id = id as usize;
            let pos = start + t; // real position in the whole text
            for i in 0..DIM {
                x[t * DIM + i] = self.wte[id * DIM + i] + self.wpe[pos * DIM + i];
            }
        }

        // work boxes, made once and reused by every block
        let mut normed = vec![0.0; n * DIM];
        let mut qkv = vec![0.0; n * 3 * DIM];
        let mut attn_out = vec![0.0; n * DIM];
        let mut proj = vec![0.0; n * DIM];
        let mut up = vec![0.0; n * MLP_DIM];

        // 2. the 12 blocks
        for (l, b) in self.blocks.iter().enumerate() {
            // attention: tokens look at earlier tokens
            ops::layernorm(&mut normed, &x, &b.ln_1_w, &b.ln_1_b);
            ops::linear(&mut qkv, &normed, &b.attn_w, &b.attn_b, DIM, 3 * DIM);

            // save the new tokens' k and v into this block's cache
            for t in 0..n {
                let row = &qkv[t * 3 * DIM..(t + 1) * 3 * DIM];
                cache.k[l].extend_from_slice(&row[DIM..2 * DIM]);
                cache.v[l].extend_from_slice(&row[2 * DIM..3 * DIM]);
            }

            attention(&mut attn_out, &qkv, &cache.k[l], &cache.v[l], start, n);
            ops::linear(&mut proj, &attn_out, &b.attn_proj_w, &b.attn_proj_b, DIM, DIM);
            add(&mut x, &proj); // residual

            // MLP: each token thinks on its own
            ops::layernorm(&mut normed, &x, &b.ln_2_w, &b.ln_2_b);
            ops::linear(&mut up, &normed, &b.fc_w, &b.fc_b, DIM, MLP_DIM);
            ops::gelu(&mut up);
            ops::linear(&mut proj, &up, &b.proj_w, &b.proj_b, MLP_DIM, DIM);
            add(&mut x, &proj); // residual
        }
        cache.len += n; // the new tokens are now part of the cache

        // 3. final layernorm, only for the last token (it predicts what comes next)
        let last = &x[(n - 1) * DIM..n * DIM];
        let mut h = vec![0.0; DIM];
        ops::layernorm(&mut h, last, &self.ln_f_w, &self.ln_f_b);

        // 4. score every word: dot product of h with that word's wte row
        let mut logits = vec![0.0; self.vocab_size];
        for v in 0..self.vocab_size {
            let row = &self.wte[v * DIM..(v + 1) * DIM];
            let mut score = 0.0;
            for i in 0..DIM {
                score += h[i] * row[i];
            }
            logits[v] = score;
        }
        logits
    }
}

// The KV cache: every token's k and v, for every block.
// k[l] holds block l's keys, one row of 768 per token seen so far (same for v).
pub struct KvCache {
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    pub len: usize, // how many tokens are stored
}

impl KvCache {
    pub fn new() -> KvCache {
        KvCache {
            k: vec![Vec::new(); N_LAYER],
            v: vec![Vec::new(); N_LAYER],
            len: 0,
        }
    }
}

// x = x + y, number by number
fn add(x: &mut [f32], y: &[f32]) {
    assert_eq!(x.len(), y.len());
    for i in 0..x.len() {
        x[i] += y[i];
    }
}

// Causal self-attention for the new tokens only.
// qkv:  n rows of [q | k | v] for the new tokens (we only use the q part here).
// k, v: rows of 768 for ALL tokens so far (old ones from the cache + the new ones).
// start: position of the first new token, so new token t sits at position start + t.
// out:  n rows of 768 = the 12 heads' results placed side by side.
fn attention(out: &mut [f32], qkv: &[f32], k: &[f32], v: &[f32], start: usize, n: usize) {
    let scale = 1.0 / (HEAD_DIM as f32).sqrt(); // 1/8
    let mut scores = vec![0.0; start + n];

    for t in 0..n {
        // t = the new token that is asking; pos = where it sits in the whole text
        let pos = start + t;
        let row = &qkv[t * 3 * DIM..(t + 1) * 3 * DIM];
        for h in 0..N_HEAD {
            let q = &row[h * HEAD_DIM..(h + 1) * HEAD_DIM];

            // how well does each earlier token s (0..=pos) match my question?
            // later tokens are skipped entirely: that is the causal mask
            for s in 0..=pos {
                let k_start = s * DIM + h * HEAD_DIM;
                let ks = &k[k_start..k_start + HEAD_DIM];
                let mut dot = 0.0;
                for i in 0..HEAD_DIM {
                    dot += q[i] * ks[i];
                }
                scores[s] = dot * scale;
            }
            ops::softmax(&mut scores[0..=pos]);

            // blend the earlier tokens' values using those percentages
            let o_start = t * DIM + h * HEAD_DIM;
            let o = &mut out[o_start..o_start + HEAD_DIM];
            o.fill(0.0);
            for s in 0..=pos {
                let v_start = s * DIM + h * HEAD_DIM;
                let vs = &v[v_start..v_start + HEAD_DIM];
                for i in 0..HEAD_DIM {
                    o[i] += scores[s] * vs[i];
                }
            }
        }
    }
}
