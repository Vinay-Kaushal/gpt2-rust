fn main() -> std::io::Result<()> {
    let merges = std::fs::read_to_string("model/merges.txt")?;
    let vocab = std::fs::read_to_string("model/vocab.json")?;
    let model_bytes = std::fs::read("model/model.safetensors")?;

    println!("merges.txt bytes : {}", merges.len());
    println!("merges line counts: {}", merges.lines().count());

    for line in merges.lines().take(5) {
        println!("{line}");
    }
    println!("vocab.json: {}", vocab.len());
    println!("model bytes length:{}", model_bytes.len());
    println!("model bytes in MB : {}", model_bytes.len() / 1_000_000);
    println!("first 8 bytes: {:?}", &model_bytes[0..8]);

    Ok(())
}
