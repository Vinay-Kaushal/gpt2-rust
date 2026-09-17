// Phase 5: the full GPT-2 forward pass.
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
const MAX_POS: usize = 1024; // longest input GPT-2 can read

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

    // Returns one score (logit) per vocab word: "how likely is this the next token?"
    pub fn forward(&self, ids: &[u32]) -> Vec<f32> {
        let n = ids.len();
        assert!(n > 0 && n <= MAX_POS, "need 1..=1024 tokens");

        // 1. embeddings: word numbers + position numbers, one row per token
        let mut x = vec![0.0; n * DIM];
        for (pos, &id) in ids.iter().enumerate() {
            let id = id as usize;
            for i in 0..DIM {
                x[pos * DIM + i] = self.wte[id * DIM + i] + self.wpe[pos * DIM + i];
            }
        }

        // work boxes, made once and reused by every block
        let mut normed = vec![0.0; n * DIM];
        let mut qkv = vec![0.0; n * 3 * DIM];
        let mut attn_out = vec![0.0; n * DIM];
        let mut proj = vec![0.0; n * DIM];
        let mut up = vec![0.0; n * MLP_DIM];

        // 2. the 12 blocks
        for b in &self.blocks {
            // attention: tokens look at earlier tokens
            ops::layernorm(&mut normed, &x, &b.ln_1_w, &b.ln_1_b);
            ops::linear(&mut qkv, &normed, &b.attn_w, &b.attn_b, DIM, 3 * DIM);
            attention(&mut attn_out, &qkv, n);
            ops::linear(&mut proj, &attn_out, &b.attn_proj_w, &b.attn_proj_b, DIM, DIM);
            add(&mut x, &proj); // residual

            // MLP: each token thinks on its own
            ops::layernorm(&mut normed, &x, &b.ln_2_w, &b.ln_2_b);
            ops::linear(&mut up, &normed, &b.fc_w, &b.fc_b, DIM, MLP_DIM);
            ops::gelu(&mut up);
            ops::linear(&mut proj, &up, &b.proj_w, &b.proj_b, MLP_DIM, DIM);
            add(&mut x, &proj); // residual
        }

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

// x = x + y, number by number
fn add(x: &mut [f32], y: &[f32]) {
    assert_eq!(x.len(), y.len());
    for i in 0..x.len() {
        x[i] += y[i];
    }
}

// Causal self-attention.
// qkv: n rows of [q (768) | k (768) | v (768)], each part split into 12 heads of 64.
// out: n rows of 768 = the 12 heads' results placed side by side.
fn attention(out: &mut [f32], qkv: &[f32], n: usize) {
    let scale = 1.0 / (HEAD_DIM as f32).sqrt(); // 1/8
    let mut scores = vec![0.0; n];

    for t in 0..n {
        // t = the token that is asking
        let row = &qkv[t * 3 * DIM..(t + 1) * 3 * DIM];
        for h in 0..N_HEAD {
            let q = &row[h * HEAD_DIM..(h + 1) * HEAD_DIM];

            // how well does each earlier token s (0..=t) match my question?
            // later tokens are skipped entirely: that is the causal mask
            for s in 0..=t {
                let k_start = s * 3 * DIM + DIM + h * HEAD_DIM;
                let k = &qkv[k_start..k_start + HEAD_DIM];
                let mut dot = 0.0;
                for i in 0..HEAD_DIM {
                    dot += q[i] * k[i];
                }
                scores[s] = dot * scale;
            }
            ops::softmax(&mut scores[0..=t]);

            // blend the earlier tokens' values using those percentages
            let o_start = t * DIM + h * HEAD_DIM;
            let o = &mut out[o_start..o_start + HEAD_DIM];
            o.fill(0.0);
            for s in 0..=t {
                let v_start = s * 3 * DIM + 2 * DIM + h * HEAD_DIM;
                let v = &qkv[v_start..v_start + HEAD_DIM];
                for i in 0..HEAD_DIM {
                    o[i] += scores[s] * v[i];
                }
            }
        }
    }
}
