// Phase 4: the 4 math "buttons" GPT-2 is built from.
// Every matrix is a flat list of numbers, stored row after row.

// bf16 ("brain float 16") = the top 16 bits of an f32: same sign and exponent,
// but only 7 of the 23 fraction bits. Half the bytes, so the big weight matrices
// come from RAM twice as fast; the math itself is still done in f32.
// A bf16 is kept in a plain u16 (Rust has no built-in bf16 type).

// f32 -> bf16, rounding to the nearest bf16 (ties go to the even one).
// Adding 0x7FFF (+1 if the kept part is odd) before cutting off the low
// 16 bits makes the cut round instead of always rounding down.
pub fn to_bf16(x: f32) -> u16 {
    let bits = x.to_bits();
    let round = 0x7FFF + ((bits >> 16) & 1);
    ((bits + round) >> 16) as u16
}

// bf16 -> f32: put the 16 bits back on top, fill the bottom with zeros.
// Just a shift, so the CPU can do 16 of these at once (SIMD).
pub fn from_bf16(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

// Button 1: weighted sums.
// For every token: output[o] = bias[o] + sum over i of (input[i] * weight[i][o])
// x:   n_tokens rows of in_dim numbers
// w:   in_dim rows of out_dim numbers (GPT-2 stores weights as [in, out])
// b:   out_dim numbers
// out: n_tokens rows of out_dim numbers (this function fills it)
pub fn linear(out: &mut [f32], x: &[f32], w: &[f32], b: &[f32], in_dim: usize, out_dim: usize) {
    let n_tokens = x.len() / in_dim;
    assert_eq!(x.len(), n_tokens * in_dim, "x is not whole rows");
    assert_eq!(w.len(), in_dim * out_dim, "weight has wrong size");
    assert_eq!(b.len(), out_dim, "bias has wrong size");
    assert_eq!(out.len(), n_tokens * out_dim, "out has wrong size");

    for t in 0..n_tokens {
        let x_row = &x[t * in_dim..(t + 1) * in_dim];
        let out_row = &mut out[t * out_dim..(t + 1) * out_dim];

        // start from the bias, then add input[i] * (row i of the weights)
        out_row.copy_from_slice(b);
        for i in 0..in_dim {
            let xi = x_row[i];
            let w_row = &w[i * out_dim..(i + 1) * out_dim];
            for o in 0..out_dim {
                out_row[o] += xi * w_row[o];
            }
        }
    }
}

// Dot product: sum of a[i] * b[i].
// One running total would make every + wait for the one before it.
// 16 separate totals don't wait on each other, so the CPU updates
// all 16 in one step (SIMD). At the end the 16 totals are added up.
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "dot of different lengths");
    let mut sums = [0.0f32; 16];

    // chunks_exact(16) hands out pieces of exactly 16 numbers;
    // whatever is left over at the end (fewer than 16) is the remainder
    let a_chunks = a.chunks_exact(16);
    let b_chunks = b.chunks_exact(16);
    let a_rest = a_chunks.remainder();
    let b_rest = b_chunks.remainder();

    for (ca, cb) in a_chunks.zip(b_chunks) {
        for j in 0..16 {
            sums[j] += ca[j] * cb[j];
        }
    }

    let mut total: f32 = sums.iter().sum();
    for (x, y) in a_rest.iter().zip(b_rest) {
        total += x * y;
    }
    total
}

// Button 2: layer normalization.
// For every token's row: shift to average 0, scale to spread 1,
// then apply the learned scale (gamma) and shift (beta).
pub fn layernorm(out: &mut [f32], x: &[f32], gamma: &[f32], beta: &[f32]) {
    let dim = gamma.len();
    assert_eq!(beta.len(), dim, "beta has wrong size");
    assert_eq!(out.len(), x.len(), "out has wrong size");
    assert_eq!(x.len() % dim, 0, "x is not whole rows");
    let n_tokens = x.len() / dim;

    for t in 0..n_tokens {
        let x_row = &x[t * dim..(t + 1) * dim];
        let out_row = &mut out[t * dim..(t + 1) * dim];

        let mut mean = 0.0;
        for i in 0..dim {
            mean += x_row[i];
        }
        mean /= dim as f32;

        let mut var = 0.0;
        for i in 0..dim {
            let d = x_row[i] - mean;
            var += d * d;
        }
        var /= dim as f32;

        let std = (var + 1e-5).sqrt(); // 1e-5 so we never divide by 0
        for i in 0..dim {
            out_row[i] = gamma[i] * (x_row[i] - mean) / std + beta[i];
        }
    }
}

// Button 3: softmax, in place.
// Turns scores into probabilities: all positive, all add up to 1.
pub fn softmax(x: &mut [f32]) {
    // subtract the biggest score first so exp() never overflows
    let mut max = f32::NEG_INFINITY;
    for i in 0..x.len() {
        if x[i] > max {
            max = x[i];
        }
    }

    let mut sum = 0.0;
    for i in 0..x.len() {
        x[i] = (x[i] - max).exp();
        sum += x[i];
    }

    for i in 0..x.len() {
        x[i] /= sum;
    }
}

// Button 4: GELU, in place.
// A smooth "keep positives, squash negatives" curve.
// This is the exact approximation GPT-2 was trained with.
const GELU_K: f32 = 0.044715; // weight of the v^3 term

pub fn gelu(x: &mut [f32]) {
    let c = (2.0 / std::f32::consts::PI).sqrt();
    for i in 0..x.len() {
        let v = x[i];
        x[i] = 0.5 * v * (1.0 + (c * (v + GELU_K * v * v * v)).tanh());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // floats are never compared with ==, only "close enough"
    fn assert_close(got: &[f32], want: &[f32]) {
        assert_eq!(got.len(), want.len());
        for i in 0..got.len() {
            assert!(
                (got[i] - want[i]).abs() < 1e-4,
                "index {i}: got {}, want {}",
                got[i],
                want[i]
            );
        }
    }

    #[test]
    fn bf16_round_trip() {
        // these fit in 7 fraction bits, so they survive exactly
        for x in [0.0, 1.0, -2.5, 0.15625, 3.0e38, -1.0e-30] {
            let back = from_bf16(to_bf16(x));
            assert!(back == x || (back - x).abs() <= x.abs() / 128.0, "{x} -> {back}");
        }
        // 1 + 2^-8 is exactly halfway between 1 and the next bf16: ties go to even (1.0)
        assert_eq!(from_bf16(to_bf16(1.00390625)), 1.0);
        // just above halfway rounds up
        assert_eq!(from_bf16(to_bf16(1.0040)), 1.0078125);
        // a weight like 0.1 lands within 1/256 of itself
        let back = from_bf16(to_bf16(0.1));
        assert!((back - 0.1).abs() < 0.1 / 256.0, "0.1 -> {back}");
    }

    #[test]
    fn linear_hand_example() {
        // 1 token with 2 inputs, 3 outputs
        let x = [1.0, 2.0];
        let w = [
            1.0, 0.0, 1.0, // row for input 0
            0.0, 1.0, 1.0, // row for input 1
        ];
        let b = [0.0, 0.0, 1.0];
        let mut out = [0.0; 3];
        linear(&mut out, &x, &w, &b, 2, 3);
        assert_close(&out, &[1.0, 2.0, 4.0]);
    }

    #[test]
    fn linear_two_tokens() {
        // each token is handled on its own
        let x = [1.0, 2.0, 3.0, 4.0];
        let w = [1.0, 0.0, 1.0, 0.0, 1.0, 1.0];
        let b = [0.0, 0.0, 1.0];
        let mut out = [0.0; 6];
        linear(&mut out, &x, &w, &b, 2, 3);
        assert_close(&out, &[1.0, 2.0, 4.0, 3.0, 4.0, 8.0]);
    }

    #[test]
    fn dot_matches_simple_loop() {
        // 37 numbers: two full chunks of 16 plus a remainder of 5
        let a: Vec<f32> = (0..37).map(|i| i as f32 * 0.5).collect();
        let b: Vec<f32> = (0..37).map(|i| 1.0 - i as f32 * 0.1).collect();
        let mut want = 0.0;
        for i in 0..37 {
            want += a[i] * b[i];
        }
        assert_close(&[dot(&a, &b)], &[want]);
        assert_close(&[dot(&[1.0, 2.0], &[3.0, 4.0])], &[11.0]); // shorter than 16
    }

    #[test]
    fn layernorm_hand_example() {
        let x = [1.0, 2.0, 3.0];
        let mut out = [0.0; 3];
        layernorm(&mut out, &x, &[1.0, 1.0, 1.0], &[0.0, 0.0, 0.0]);
        assert_close(&out, &[-1.2247357, 0.0, 1.2247357]);
    }

    #[test]
    fn softmax_hand_example() {
        let mut x = [1.0, 2.0, 3.0];
        softmax(&mut x);
        assert_close(&x, &[0.0900306, 0.2447285, 0.6652410]);
    }

    #[test]
    fn softmax_huge_scores_do_not_break() {
        let mut x = [1000.0, 1001.0, 1002.0];
        softmax(&mut x);
        assert_close(&x, &[0.0900306, 0.2447285, 0.6652410]);
    }

    #[test]
    fn softmax_minus_infinity_becomes_zero() {
        let mut x = [1.0, f32::NEG_INFINITY];
        softmax(&mut x);
        assert_close(&x, &[1.0, 0.0]);
    }

    #[test]
    fn gelu_hand_example() {
        let mut x = [0.0, 1.0, -1.0, -3.0, 2.0];
        gelu(&mut x);
        assert_close(&x, &[0.0, 0.8411920, -0.1588080, -0.0036374, 1.9545977]);
    }
}
