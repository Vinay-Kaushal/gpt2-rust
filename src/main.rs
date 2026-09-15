mod safetensors;
use safetensors::SafeTensors;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = SafeTensors::load("model/model.safetensors")?;
    println!(
        "loaded {} tensors (all sizes verified)",
        model.tensors.len()
    );

    let wte = model.tensor("wte.weight")?;
    println!("wte.weight shape: {:?}", wte.shape);
    println!("wte.weight first 5 numbers: {:?}", &wte.data[0..5]);

    let ln_f = model.tensor("ln_f.weight")?;
    println!("ln_f.weight shape: {:?}", ln_f.shape);
    println!("ln_f.weight first 5 numbers: {:?}", &ln_f.data[0..5]);

    Ok(())
}
