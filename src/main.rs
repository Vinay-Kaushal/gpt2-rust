mod ops;
mod safetensors;
mod tokenizer;

use safetensors::SafeTensors;
use tokenizer::Tokenizer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ---- Phase 3: text -> token ids ----
    let tokenizer = Tokenizer::load("model/vocab.json", "model/merges.txt")?;
    let text = "Hello";
    let ids = tokenizer.encode(text)?;
    println!("text: {text:?} -> ids: {ids:?} -> {:?}", tokenizer.decode(&ids)?);

    // ---- Phase 4: run real GPT-2 weights through our math buttons ----
    let model = SafeTensors::load("model/model.safetensors")?;
    let wte = model.tensor("wte.weight")?; // [50257, 768] word table
    let wpe = model.tensor("wpe.weight")?; // [1024, 768]  position table
    let dim = wte.shape[1];

    // look up row for the token, plus row for position 0, add them
    let id = ids[0] as usize;
    let mut h = vec![0.0; dim];
    for i in 0..dim {
        h[i] = wte.data[id * dim + i] + wpe.data[i];
    }
    println!("embedding     (first 4 of {}): {:?}", h.len(), &h[0..4]);

    // button 2: layernorm with block 0's ln_1
    let gamma = model.tensor("h.0.ln_1.weight")?;
    let beta = model.tensor("h.0.ln_1.bias")?;
    let mut normed = vec![0.0; dim];
    ops::layernorm(&mut normed, &h, &gamma.data, &beta.data);
    println!("after ln_1    (first 4 of {}): {:?}", normed.len(), &normed[0..4]);

    // button 1: linear 768 -> 3072 with block 0's MLP weights
    let w = model.tensor("h.0.mlp.c_fc.weight")?; // [768, 3072]
    let b = model.tensor("h.0.mlp.c_fc.bias")?; // [3072]
    let (in_dim, out_dim) = (w.shape[0], w.shape[1]);
    let mut up = vec![0.0; out_dim];
    ops::linear(&mut up, &normed, &w.data, &b.data, in_dim, out_dim);
    println!("after c_fc    (first 4 of {}): {:?}", up.len(), &up[0..4]);

    // button 4: gelu
    ops::gelu(&mut up);
    println!("after gelu    (first 4 of {}): {:?}", up.len(), &up[0..4]);

    // button 3: softmax on a few scores
    let mut scores = vec![2.0, 1.0, 0.1];
    ops::softmax(&mut scores);
    let total: f32 = scores.iter().sum();
    println!("softmax [2, 1, 0.1] -> {scores:?} (sum {total})");

    Ok(())
}
