mod model;
mod ops;
mod safetensors;
mod sampler;
mod tokenizer;

use model::{Gpt2, KvCache, MAX_POS};
use safetensors::SafeTensors;
use sampler::Rng;
use std::io::Write;
use tokenizer::Tokenizer;

const END_OF_TEXT: u32 = 50256; // GPT-2's "the document is over" token

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let tokenizer = Tokenizer::load("model/vocab.json", "model/merges.txt")?;

    // load the weights into the model; the raw file is dropped after this block
    let gpt2 = {
        let st = SafeTensors::load("model/model.safetensors")?;
        Gpt2::load(&st)?
    };

    let prompt = "The quick brown fox jumps over the lazy";
    let prompt_ids = tokenizer.encode(prompt)?;
    println!("prompt: {prompt:?}  ids: {prompt_ids:?}\n");

    // ---- 1. speed benchmark: greedy, so the text is the same every run ----
    // (only meaningful with `cargo run --release`)
    let n_new = 50;
    let start = std::time::Instant::now();
    let ids = generate(&gpt2, &tokenizer, &prompt_ids, n_new, 1, 1.0, &mut Rng::new(0), false)?;
    let secs = start.elapsed().as_secs_f64();
    println!("greedy, {} tokens in {secs:.2}s = {:.1} tokens/sec", ids.len(), ids.len() as f64 / secs);
    println!("  {:?}\n", tokenizer.decode(&ids)?);

    // ---- 2. top-k sampling: different text for every seed ----
    let seed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_nanos() as u64;
    let mut rng = Rng::new(seed);
    println!("top-k (k=40, temperature=0.8, seed={seed}):");
    print!("  {prompt}");
    generate(&gpt2, &tokenizer, &prompt_ids, 60, 40, 0.8, &mut rng, true)?;
    println!();

    Ok(())
}

// The generation loop: prompt once, then one new token per step, reusing the cache.
// Returns only the newly made ids. If `stream`, prints each token as it arrives.
fn generate(
    gpt2: &Gpt2,
    tokenizer: &Tokenizer,
    prompt: &[u32],
    max_new: usize,
    k: usize,
    temperature: f32,
    rng: &mut Rng,
    stream: bool,
) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    let mut cache = KvCache::new();
    let mut new_ids = Vec::new();

    // step 1: the whole prompt goes in and fills the cache
    let mut logits = gpt2.forward(prompt, &mut cache);

    for _ in 0..max_new {
        let id = sampler::sample_top_k(&logits, k, temperature, rng);
        if id == END_OF_TEXT {
            break;
        }
        new_ids.push(id);
        if stream {
            print!("{}", tokenizer.decode(&[id])?);
            std::io::stdout().flush()?; // show it now, don't wait for a newline
        }
        if cache.len == MAX_POS {
            break; // GPT-2 cannot read past 1024 tokens
        }
        // next steps: only the one new token goes in
        logits = gpt2.forward(&[id], &mut cache);
    }
    Ok(new_ids)
}
