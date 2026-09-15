use serde::Deserialize;
use std::collections::HashMap;
use std::error::Error;

//one entry in the header where a tensor lives, and what shape it has
#[derive(Debug, Deserialize)]
pub struct TensorInfo {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data_offsets: [usize; 2],
}
// A Tensor loaded into memory its shape plus all it number in one flat list
pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

pub struct SafeTensors {
    bytes: Vec<u8>,
    data_start: usize,
    pub tensors: HashMap<String, TensorInfo>,
}

impl SafeTensors {
    pub fn load(path: &str) -> Result<SafeTensors, Box<dyn Error>> {
        let bytes = std::fs::read(path)?;

        //first 8 bytes = header length (little -endian u64)

        let first_8: [u8; 8] = bytes[0..8].try_into()?;
        let header_len = u64::from_le_bytes(first_8) as usize;
        let data_start = 8 + header_len;
        let header = std::str::from_utf8(&bytes[8..data_start])?;

        //parse loosely, then keep only real tensors (skip __metadata__)

        let raw: HashMap<String, serde_json::Value> = serde_json::from_str(header)?;
        let mut tensors = HashMap::new();

        for (name, value) in raw {
            if name == "__metadata__" {
                continue;
            }
            let info: TensorInfo = serde_json::from_value(value)?;

            //sanity check : byte range must match shape x 4 bytes
            let [start, end] = info.data_offsets;
            let count: usize = info.shape.iter().product();
            if info.dtype != "F32" || end - start != count * 4 {
                return Err(format!("tensor {name} has bad dtype or size").into());
            }
            tensors.insert(name, info);
        }

        Ok(SafeTensors {
            bytes,
            data_start,
            tensors,
        })
    }

    pub fn tensor(&self, name: &str) -> Result<Tensor, Box<dyn Error>> {
        let info = self
            .tensors
            .get(name)
            .ok_or(format!("tensor {name} not found"))?;
        let [start, end] = info.data_offsets;
        let raw = &self.bytes[self.data_start + start..self.data_start + end];

        // every 4 bytes = one f32
        let mut data = Vec::with_capacity(raw.len() / 4);
        for chunk in raw.chunks_exact(4) {
            let four: [u8; 4] = chunk.try_into()?;
            data.push(f32::from_le_bytes(four));
        }

        Ok(Tensor {
            shape: info.shape.clone(),
            data,
        })
    }
}
