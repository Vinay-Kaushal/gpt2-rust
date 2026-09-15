use fancy_regex::Regex;
use std::collections::HashMap;
use std::error::Error;

/// GPT-2's rule for pre-splitting text into words, numbers, punctuation and spaces.
const PATTERN: &str = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";

pub struct Tokenizer {
    // token text -> id (from vocab.json)
    encoder: HashMap<String, u32>,
    // id -> token text (the reverse)
    decoder: HashMap<u32, String>,
    // (left, right) pair -> priority (from merges.txt, lower = merge first)
    bpe_ranks: HashMap<(String, String), usize>,
    // byte -> visible char
    byte_encoder: [char; 256],
    // visible char -> byte (the reverse)
    byte_decoder: HashMap<char, u8>,
    // the pre-split regex, compiled once
    pattern: Regex,
}

/// Give every one of the 256 possible bytes its own visible character.
fn bytes_to_unicode() -> [char; 256] {
    let mut table = ['\0'; 256];
    let mut next_extra = 256;
    for b in 0..256u32 {
        let printable =
            (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b);
        let code = if printable {
            b
        } else {
            let extra = next_extra;
            next_extra += 1;
            extra
        };
        table[b as usize] = char::from_u32(code).unwrap();
    }
    table
}

impl Tokenizer {
    pub fn load(vocab_path: &str, merges_path: &str) -> Result<Tokenizer, Box<dyn Error>> {
        // vocab.json: {"!": 0, "\"": 1, ... }
        let vocab_text = std::fs::read_to_string(vocab_path)?;
        let encoder: HashMap<String, u32> = serde_json::from_str(&vocab_text)?;

        let mut decoder = HashMap::new();
        for (token, id) in &encoder {
            decoder.insert(*id, token.clone());
        }

        // merges.txt: skip "#version" line, line number = rank (lower = merge first)
        let merges_text = std::fs::read_to_string(merges_path)?;
        let mut bpe_ranks = HashMap::new();
        for (rank, line) in merges_text.lines().skip(1).enumerate() {
            if line.is_empty() {
                continue;
            }
            let (left, right) = line
                .split_once(' ')
                .ok_or(format!("bad merge line: {line}"))?;
            bpe_ranks.insert((left.to_string(), right.to_string()), rank);
        }

        let byte_encoder = bytes_to_unicode();
        let mut byte_decoder = HashMap::new();
        for (b, c) in byte_encoder.iter().enumerate() {
            byte_decoder.insert(*c, b as u8);
        }

        Ok(Tokenizer {
            encoder,
            decoder,
            bpe_ranks,
            byte_encoder,
            byte_decoder,
            pattern: Regex::new(PATTERN)?,
        })
    }

    /// Apply BPE merges to one pre-split word until no known pair is left.
    fn bpe(&self, word: &str) -> Vec<String> {
        let mut parts: Vec<String> = Vec::new();
        for c in word.chars() {
            parts.push(c.to_string());
        }

        loop {
            // find the adjacent pair with the lowest rank
            let mut best_rank = usize::MAX;
            let mut best_pos = None;
            for i in 0..parts.len().saturating_sub(1) {
                let pair = (parts[i].clone(), parts[i + 1].clone());
                let rank = self.bpe_ranks.get(&pair).copied().unwrap_or(usize::MAX);
                if rank < best_rank {
                    best_rank = rank;
                    best_pos = Some(i);
                }
            }

            // no mergeable pair left -> done
            let Some(i) = best_pos else {
                break;
            };

            // merge parts[i] and parts[i + 1] into one
            let right = parts.remove(i + 1);
            parts[i].push_str(&right);
        }

        parts
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>, Box<dyn Error>> {
        let mut ids = Vec::new();

        for found in self.pattern.find_iter(text) {
            let piece = found?.as_str();

            // bytes -> visible chars
            let mut mapped = String::new();
            for b in piece.bytes() {
                mapped.push(self.byte_encoder[b as usize]);
            }

            // merge, then look up each token's id
            for token in self.bpe(&mapped) {
                let id = self
                    .encoder
                    .get(&token)
                    .ok_or(format!("token {token} not in vocab"))?;
                ids.push(*id);
            }
        }

        Ok(ids)
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String, Box<dyn Error>> {
        let mut bytes = Vec::new();

        for id in ids {
            let token = self
                .decoder
                .get(id)
                .ok_or(format!("id {id} not in vocab"))?;
            for c in token.chars() {
                let b = self
                    .byte_decoder
                    .get(&c)
                    .ok_or(format!("char {c} not in byte table"))?;
                bytes.push(*b);
            }
        }

        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }
}
