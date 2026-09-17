mod model;
mod ops;
mod safetensors;
mod tokenizer;

use model::Gpt2;
use safetensors::SafeTensors;
use tokenizer::Tokenizer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let tokenizer = Tokenizer::load("model/vocab.json", "model/merges.txt")?;

    // load the weights into the model; the raw file is dropped after this block
    let gpt2 = {
        let st = SafeTensors::load("model/model.safetensors")?;
        Gpt2::load(&st)?
    };

    // ---- Phase 5: text -> ids -> forward pass -> next-token guesses ----
    let text = "The quick brown fox jumps over the lazy";
    let ids = tokenizer.encode(text)?;
    println!("text: {text:?}");
    println!("ids:  {ids:?}");

    let start = std::time::Instant::now();
    let mut probs = gpt2.forward(&ids);
    println!("forward pass took {:.2?}", start.elapsed());
    println!("first 3 logits: {:?}", &probs[0..3]);

    // scores -> percentages
    ops::softmax(&mut probs);

    // sort word ids by probability, biggest first, and show the top 5
    let mut order: Vec<usize> = (0..probs.len()).collect();
    order.sort_by(|&a, &b| probs[b].total_cmp(&probs[a]));
    println!("top 5 next tokens:");
    for &id in &order[0..5] {
        let word = tokenizer.decode(&[id as u32])?;
        println!("  {id:>6} {word:?}  {:.2}%", probs[id] * 100.0);
    }

    Ok(())
}
