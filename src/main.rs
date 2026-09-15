fn main() -> Result<(), Box<dyn std::error::Error>> {
    //load the whole file into memory as raw bytes
    let model_bytes = std::fs::read("model/model.safetensors")?;

    //the first 8 bytes are the header length

    let first_8: [u8; 8] = model_bytes[0..8].try_into()?;
    let header_len = u64::from_le_bytes(first_8);
    println!("header len {}", header_len);

    //cut the header out and read at as a text

    let header_len = header_len as usize;
    let header_end = 8 + header_len;
    let header_bytes = &model_bytes[8..header_end];
    let header = std::str::from_utf8(header_bytes)?;

    println!("first 300 chars of headers");
    println!("{}", &header[0..300]);

    Ok(())
}
