mod safetensors;
mod tokenizer;

use safetensors::SafeTensors;
use tokenizer::Tokenizer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ---- Phase 3: text -> token ids -> text ----
    let tokenizer = Tokenizer::load("model/vocab.json", "model/merges.txt")?;

    let text = "Hello world! GPT-2 splits text into tokens.";
    let ids = tokenizer.encode(text)?;
    println!("text:  {text}");
    println!("ids:   {:?}", ids);

    println!("each token:");
    for id in &ids {
        println!("  {id:>6} -> {:?}", tokenizer.decode(&[*id])?);
    }

    let back = tokenizer.decode(&ids)?;
    println!("decoded: {back}");
    println!("round trip ok: {}", back == text);

    // ---- bridge to Phase 5: token id -> its 768 numbers ----
    let model = SafeTensors::load("model/model.safetensors")?;
    println!("loaded {} tensors", model.tensors.len());
    let wte = model.tensor("wte.weight")?;
    let dim = wte.shape[1];
    let first = ids[0] as usize;
    let row = &wte.data[first * dim..first * dim + dim];
    println!(
        "embedding of token {first} (first 5 of {dim}): {:?}",
        &row[0..5]
    );

    Ok(())
}
