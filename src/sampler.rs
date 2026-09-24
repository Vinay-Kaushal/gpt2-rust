// Phase 6: picking the next token from the model's scores.
// top-k: keep the k best tokens, sharpen/flatten with temperature, then roll a weighted die.

use crate::ops;

// A tiny random number generator (xorshift64).
// Same seed -> same sequence of "random" numbers, which makes runs repeatable.
pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Rng {
        // xorshift gets stuck forever on 0, so never start there
        Rng { state: if seed == 0 { 0x9E3779B97F4A7C15 } else { seed } }
    }

    // scramble the bits; each call gives a new, random-looking u64
    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    // a random number in [0, 1): take the top 24 bits and divide by 2^24
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}

// Pick one token id from `logits`.
// k = 1 means greedy (always the top token); temperature is then ignored.
pub fn sample_top_k(logits: &[f32], k: usize, temperature: f32, rng: &mut Rng) -> u32 {
    assert!(k >= 1 && k <= logits.len(), "k must be 1..=vocab size");
    assert!(temperature > 0.0, "temperature must be above 0");

    // 1. sort token ids by score, biggest first, and keep the top k
    let mut order: Vec<usize> = (0..logits.len()).collect();
    order.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));
    order.truncate(k);
    if k == 1 {
        return order[0] as u32;
    }

    // 2. temperature, then softmax -> k percentages that add up to 1
    let mut probs: Vec<f32> = order.iter().map(|&id| logits[id] / temperature).collect();
    ops::softmax(&mut probs);

    // 3. roll the die: walk along the percentages until we pass r
    let r = rng.next_f32();
    let mut total = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        total += p;
        if r < total {
            return order[i] as u32;
        }
    }
    order[k - 1] as u32 // rounding left total just under 1: take the last one
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn k1_is_greedy() {
        let mut rng = Rng::new(1);
        assert_eq!(sample_top_k(&[0.1, 3.0, -2.0, 2.9], 1, 1.0, &mut rng), 1);
    }

    #[test]
    fn only_top_k_are_picked() {
        let mut rng = Rng::new(42);
        let logits = [5.0, 4.0, -1.0, 0.0, 4.5];
        for _ in 0..1000 {
            let id = sample_top_k(&logits, 2, 1.0, &mut rng);
            assert!(id == 0 || id == 4, "picked {id}, which is not in the top 2");
        }
    }

    #[test]
    fn picks_follow_the_percentages() {
        // scores ln(3) and 0 -> softmax gives 75% and 25%
        let mut rng = Rng::new(7);
        let logits = [3.0f32.ln(), 0.0];
        let mut count0 = 0;
        for _ in 0..10000 {
            if sample_top_k(&logits, 2, 1.0, &mut rng) == 0 {
                count0 += 1;
            }
        }
        assert!((7200..7800).contains(&count0), "token 0 picked {count0}/10000 times");
    }

    #[test]
    fn same_seed_same_numbers() {
        let mut a = Rng::new(123);
        let mut b = Rng::new(123);
        for _ in 0..10 {
            let x = a.next_f32();
            assert_eq!(x, b.next_f32());
            assert!((0.0..1.0).contains(&x));
        }
    }
}
