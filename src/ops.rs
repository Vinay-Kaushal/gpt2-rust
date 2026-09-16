// Phase 4: the 4 math "buttons" GPT-2 is built from.
// Every matrix is a flat list of numbers, stored row after row.

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
pub fn gelu(x: &mut [f32]) {
    let c = (2.0 / std::f32::consts::PI).sqrt();
    for i in 0..x.len() {
        let v = x[i];
        x[i] = 0.5 * v * (1.0 + (c * (v + 0.044158 * v * v * v)).tanh());
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
        assert_close(&x, &[0.0, 0.8410732, -0.1589268, -0.0037256, 1.9542811]);
    }
}
