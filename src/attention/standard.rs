//! Various forms of multi-head attention described in paper "Attention Is All You Need".

use tch::{nn, Tensor};

const A4D_TENSOR: &str = "Tensor must be 4-dimensional";
const A3D_TENSOR: &str = "Tensor must be 3-dimensional";

/// Ensures that each position (of a decoder's self-attention) cannot attend
/// to subsequent positions. Such connections in a QK matrix are represented by items
/// above the diagonal. So we assign -inf (or some large negative number)
/// to all invalid connections, and later softmax will turn them into zeros.
///
/// We need this to guarantee that decoder's predictions are based
/// on what has happened before the position, not after.
/// Argument `qkt` is a scaled dot-product of Q and K.T,
/// shaped as `(batch, num_heads, q_seq_len, k_seq_len)`.
fn causal_attention_mask(qkt: Tensor) -> Tensor {
    // Practically, q_seq_len and k_seq_len will always be the same
    let (_, _, q_seq_len, k_seq_len) = qkt.size4().unwrap();
    // Creates a boolean mask filled with `false` on and below the diagonal.
    // TODO: Can be cached using q_seq_len, k_seq_len and the device as a key
    let causal_mask = Tensor::ones(
        &[1, 1, q_seq_len, k_seq_len],
        (tch::Kind::Bool, qkt.device()),
    )
    .triu(1);
    // Applies the mask, replacing all above the diagonal with -inf
    qkt.masked_fill(&causal_mask, f64::NEG_INFINITY)
}

/// Calculates the output of the attention after the affine transformations
/// of the inputs were done. Concatenates the outputs of all heads,
/// without projecting them with the output projection.
/// Expects arguments and shapes:
///
/// * `q`: (batch_size, q_seq_len, num_heads, d_k)
/// * `v`: (batch_size, v_seq_len, num_heads, d_v)
/// * `k`: (batch_size, k_seq_len, num_heads, d_k)
/// * `attention_mask`: 1s or 0s for keys that should and should not be attended,
///    shaped as (batch_size, k_seq_len)
/// * `causal`: whether the causal mask must be applied to the QK product.
///
/// Returns a tensor shaped as `(batch_size, q_seq_len, num_heads * d_v)`
fn attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    attention_mask: Option<&Tensor>,
    causal: bool,
) -> Tensor {
    let (batch_size, q_seq_len, num_heads, _) = q.size4().expect(A4D_TENSOR);
    let (_, _, _, key_value_dim) = v.size4().expect(A4D_TENSOR);
    let sqrt_d_k = (key_value_dim as f64).sqrt();
    // Q * transposed(K) / sqrt(d_k)
    // The result will have shape of (batch, num_heads, q_seq_len, k_seq_len)
    let mut qkt_scaled = q.transpose(2, 1).matmul(
        // k is transposed into (batch, num_heads, d_k, seq_len)
        &k.permute(&[0, 2, 3, 1]),
    ) / sqrt_d_k;
    if let Some(attention_mask) = attention_mask {
        // setting keys we do not need to attend to a large negative, so that softmax
        // will turn them into zeroes
        qkt_scaled += -1e5 * (1.0 - attention_mask.view([batch_size, 1, 1, -1]));
    }
    if causal {
        qkt_scaled = causal_attention_mask(qkt_scaled)
    }
    let scaled_attention = qkt_scaled
        .softmax(-1, qkt_scaled.kind())
        .matmul(&v.transpose(2, 1));
    // "Concatenating" heads by rearranging dimensions
    // from (batch, num_heads, seq_len, key_value_dim)
    // and reshaping the result
    scaled_attention
        .transpose(2, 1)
        .reshape(&[-1, q_seq_len, num_heads * key_value_dim])
}

/// Multi-head attention described in paper "Attention Is All You Need"
/// (https://arxiv.org/pdf/1706.03762.pdf).
///
/// This implementation is designed and optimized for a case of self-attention,
/// when we have a single input tensor from which the keys, the values,
/// and the queries (`K`, `V`, and `Q`) are projected, all of the same shape.
/// This allows to clump all projection matrices into a single one and calculating
/// the projections faster, in one sweep.
#[derive(Debug)]
pub struct MultiHeadSelfAttention {
    pub causal: bool,
    pub num_heads: usize,
    pub key_query_value_dim: usize,
    pub output_dim: usize,
    qkv_weights: Tensor,
    output_weights: Tensor,
}

impl MultiHeadSelfAttention {
    /// Creates a new multi-head self-attention layer accepting input the shape of
    /// (batch-size, sequence-length, `input_dim`). This input will be projected into `num_heads`
    /// sets of queries, keys, and values with shapes of (sequence-length, `key_query_value_dim`).
    /// Each set is thus represent a sigle attention head, and the dimensions for keys and values
    /// are given for a single head too.
    ///
    /// The output's combined last dimension, if `None` is given, will be set to be the size of
    /// `key_query_value_dim` * `num_heads`.
    ///
    /// If `causal` is true, each position in the sequence will be limited in attending
    /// (in "seeing") only to itself and the precediing positions,
    /// which is useful in time series forecasting and language modelling.
    pub fn new(
        vs: nn::Path,
        causal: bool,
        input_dim: usize,
        num_heads: usize,
        key_query_value_dim: usize,
        output_dim: Option<usize>,
    ) -> Self {
        let actual_output_dim = output_dim.unwrap_or(key_query_value_dim * num_heads);
        // * 3 for q, k and v
        let qkv_dim = [
            input_dim as i64,
            (num_heads * key_query_value_dim * 3) as i64,
        ];
        // These weights are concatenated matrices W_q, W_k and W_v which
        // are, in turn, concatenated W matrices of keys, queries and values
        // for each of the heads. So, essentially it's a concatenation of
        // W_q1, W_q2,..., W_qh, W_k1, W_k2,..., W_kh, W_v1, W_v2,..., W_vh
        // for all h heads.
        let qkv_weights = vs.kaiming_uniform("qkv_weights", &qkv_dim);
        let output_shape = [
            (key_query_value_dim * num_heads) as i64,
            actual_output_dim as i64,
        ];
        let output_weights = vs.kaiming_uniform("output_weights", &output_shape);

        Self {
            causal,
            num_heads,
            key_query_value_dim,
            qkv_weights,
            output_weights,
            output_dim: actual_output_dim,
        }
    }

    /// Multi-head self-attention. Optimized for cases when attention receives
    /// a single input tensor and all Q, K, V projections for attention
    /// have the same dimensionality, allowing to calculate them all with
    /// a single (faster) matrix multiplication.
    /// Parameter `input` should be a tensor `[batch_size, seq_len, model_dim]`.
    /// Attention mask is an optional tensor of 1s and 0s shaped as `[batch_size, seq_len]`
    /// and marking positions that should be attended (1s) and padding that should not (0s).
    /// Returns a tensor the shape of `[batch_size, seq_len, output_dim]`
    pub fn forward(&self, input: &Tensor, attention_mask: Option<&Tensor>) -> Tensor {
        let (batch_size, seq_len, _) = input.size3().expect(A3D_TENSOR);

        let qkv = input.matmul(&self.qkv_weights).reshape(&[
            batch_size,
            seq_len,
            3,
            self.num_heads as i64,
            self.key_query_value_dim as i64,
        ]);

        let seam_dim_index = 2;
        // Splitting the keys, the values and the queries before further processing.
        // Each will have shape (batch_size, seq_len, 1, num_heads, d_k).
        // The redundant 1 dimension will be squeezed out later.
        let qkv_chunks = qkv.split(1, seam_dim_index); // vec![q, k, v]
        let attention_out = attention(
            &qkv_chunks[0].squeeze_dim(seam_dim_index),
            &qkv_chunks[1].squeeze_dim(seam_dim_index),
            &qkv_chunks[2].squeeze_dim(seam_dim_index),
            attention_mask,
            self.causal,
        );
        // Output projection
        attention_out.matmul(&self.output_weights)
    }
}

/// Multi-head attention described in paper "Attention Is All You Need"
/// (https://arxiv.org/pdf/1706.03762.pdf).
///
/// This implementation is designed to be fully abstract:
///
/// * it accepts 3 separate inputs: for the keys, the queries and the values
/// * Inputs for keys and values can have sequence length different from the sequence length
///   for queries.
///
/// This allows for construction of encoder-decoder models, where a decoder "interrogates"
/// through its queries whatever was produced by an encoder.
pub struct MultiHeadAttention {
    pub causal: bool,
    pub num_heads: usize,
    pub key_query_dim: usize,
    pub value_dim: usize,
    pub output_dim: usize,
    q_weights: Tensor,
    k_weights: Tensor,
    v_weights: Tensor,
    output_weights: Tensor,
}

impl MultiHeadAttention {
    /// Creates a new multi-head self-attention layer accepting 3 inputs the shapes of
    ///
    /// * (batch-size, sequence-length, `query_input_dim`)
    /// * (batch-size, sequence-length, `key_input_dim`)
    /// * (batch-size, sequence-length, `value_input_dim`)
    ///
    /// from which the queries, the keys, and the values will be projected by each of
    /// the `num_heads` attention heads.
    /// Each head will have its queries, keys, and values shaped as
    /// (sequence-length, `key_query_dim`) and (sequence-length, `value_dim`) respectingly.
    ///
    /// The output's last combined dimension, if `None` is given, will be set to be the size of
    /// `value_dim` * `num_heads`.
    ///
    /// If `causal` is true, each position in the sequence will be limited in attending
    /// (in "seeing") only to itself and the precediing positions,
    /// which is useful in time series forecasting and language modelling.
    pub fn new(
        vs: &nn::Path,
        causal: bool,
        query_input_dim: usize,
        key_input_dim: usize,
        value_input_dim: usize,
        num_heads: usize,
        key_query_dim: usize,
        value_dim: usize,
        output_dim: Option<usize>,
    ) -> Self {
        let actual_output_dim = output_dim.unwrap_or(value_dim * num_heads);
        // These weights are concatenated matrices W_q, W_k and W_v which
        // are, in turn, concatenated W matrices of keys, queries and values
        // for each of the heads. So, essentially it's a concatenation of
        // W_q1, W_q2,..., W_qh, W_k1, W_k2,..., W_kh, W_v1, W_v2,..., W_vh
        // for all h heads.
        let v_weights = vs.kaiming_uniform(
            "v_weights",
            &[value_input_dim as i64, (value_dim * num_heads) as i64],
        );
        let k_weights = vs.kaiming_uniform(
            "k_weights",
            &[key_input_dim as i64, (key_query_dim * num_heads) as i64],
        );
        let q_weights = vs.kaiming_uniform(
            "q_weights",
            &[query_input_dim as i64, (key_query_dim * num_heads) as i64],
        );

        let output_shape = [(value_dim * num_heads) as i64, actual_output_dim as i64];
        let output_weights = vs.kaiming_uniform("output_weights", &output_shape);

        Self {
            causal,
            num_heads,
            key_query_dim,
            value_dim,
            q_weights,
            k_weights,
            v_weights,
            output_weights,
            output_dim: actual_output_dim,
        }
    }

    /// Multi-head attention. Inputs must be of the following dimensions:
    ///
    /// * `query_input`: `[batch_size, query_seq_len, query_input_dim]`.
    /// * `key_input`: `[batch_size, key_value_seq_len, key_input_dim]`.
    /// * `value_input`: `[batch_size, key_value_seq_len, value_input_dim]`.
    /// * `attention_mask`: 1s or 0s for keys that should and should not be attended,
    ///    shaped as (batch_size, key_value_seq_len)
    ///
    /// Returns a tensor the shape of `[batch_size, value_seq_len, output_dim]`
    pub fn forward(
        &self,
        query_input: &Tensor,
        key_input: &Tensor,
        value_input: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Tensor {
        let (value_batch_size, value_seq_len, _) = value_input.size3().expect(A3D_TENSOR);
        let (key_batch_size, key_seq_len, _) = key_input.size3().expect(A3D_TENSOR);
        let (query_batch_size, query_seq_len, _) = key_input.size3().expect(A3D_TENSOR);
        assert_eq!(
            value_seq_len, key_seq_len,
            "Key and value sequences must not have different lengths. Currently {} and {}",
            key_seq_len, value_seq_len
        );
        assert_eq!(
            key_batch_size, value_batch_size,
            "Key and value sequences must have the same batch sizes. Currently {} and {}",
            key_batch_size, value_batch_size
        );
        assert_eq!(
            key_batch_size, query_batch_size,
            "Key and query sequences must have the same batch sizes. Currently {} and {}",
            key_batch_size, query_batch_size
        );
        let batch_size = value_batch_size;
        let kv_seq_len = value_seq_len;
        //  The first thing we need to do is to perform affine transformations
        //  of the inputs to get the Queries, the Keys and the Values.
        //  Each will have shape (batch_size, num_heads, sequence_len, dim)
        let v = value_input.matmul(&self.v_weights).reshape(&[
            batch_size,
            kv_seq_len,
            self.num_heads as i64,
            self.value_dim as i64,
        ]);
        let k = key_input.matmul(&self.k_weights).reshape(&[
            batch_size,
            kv_seq_len,
            self.num_heads as i64,
            self.key_query_dim as i64,
        ]);
        let q = query_input.matmul(&self.q_weights).reshape(&[
            batch_size,
            query_seq_len,
            self.num_heads as i64,
            self.key_query_dim as i64,
        ]);
        let attention_out = attention(&q, &k, &v, attention_mask, self.causal);
        attention_out.matmul(&self.output_weights)
    }
}

#[cfg(test)]
mod test {
    #[test]
    fn test_attention() {
        let batch_size = 1;
        let d_k = 5;
        let num_heads = 3;
        let seq_len = 2;
        let qkv_dim = [batch_size, seq_len, num_heads, d_k];

        let key = tch::Tensor::of_slice(&[
            -0.5917, 0.1766, 0.4846, 0.3123, 0.2224, 0.1224, 0.2628, -1.4355, -0.2553, 0.3784,
            1.6483, -0.2301, -0.7826, 1.9004, -0.6406, 0.2766, 0.9230, 0.2658, -0.1112, -0.8449,
            -0.5094, 0.9149, 1.2510, 0.7405, 2.0509, 0.1292, -0.1290, -0.5136, -0.3277, -1.2158,
        ])
        .reshape(&qkv_dim);
        let query = tch::Tensor::of_slice(&[
            -1.4838, 0.1216, -0.6428, 1.0730, 0.0514, 1.0894, -1.2976, 0.3055, 0.4674, -0.7904,
            0.5942, 1.0004, -1.0341, 0.8607, -0.4123, 0.4504, -1.3332, 0.0440, -0.8076, 1.1087,
            0.3849, 0.1982, -0.9366, -0.3024, 1.7482, -0.5707, 0.1702, 1.5397, 1.0245, -0.2351,
        ])
        .reshape(&qkv_dim);
        let value = tch::Tensor::of_slice(&[
            -1.7960, 0.3486, -1.2651, -0.2911, -0.3844, -0.5393, -0.9316, -0.1701, -1.0799,
            -0.5474, 0.6255, 0.2315, -1.1439, 1.2168, -0.0378, -0.3315, -0.7856, 1.9734, -0.6110,
            0.8073, 1.2904, -1.1004, 0.4783, -0.8432, 0.5684, -1.0430, -0.1173, -0.2158, 2.0082,
            -1.7302,
        ])
        .reshape(&qkv_dim);

        #[rustfmt::skip]
        let expect_result = tch::Tensor::of_slice(&[
            -1.31020674, -0.02762855, -0.1908484, -0.39721489, 0.01090203,
            0.0669134, -0.98752656, 0.0447269, -1.00147692, -0.17771485,
            0.24940618, 0.15287757, -0.93469852, 1.39518816, -0.41928108,
            -1.2939129, -0.04024752, -0.15481726, -0.40077406, 0.02416073,
            0.37404134, -1.01586082, 0.15356537, -0.96174517, 0.00957998,
            -0.05222779, 0.08982097, -0.76691518, 1.53825866, -0.72523573,
        ])
        .reshape(&[batch_size, seq_len, num_heads * d_k]);

        #[rustfmt::skip]
        let expect_causal_result = tch::Tensor::of_slice(&[
           -1.796, 0.3486, -1.2651,
            -0.2911, -0.3844, -0.5393,
            -0.9316, -0.1701, -1.0799,
            -0.5474, 0.6255, 0.2315,
            -1.1439, 1.2168, -0.0378,
            -1.2939129019362632, -0.04024751561890764, -0.15481726385837366,
            -0.4007740612294909, 0.024160733876787365, 0.3740413384758751,
            -1.015860817584701, 0.15356536802085446, -0.9617451687067607,
            0.009579977849582743, -0.05222779442962869, 0.08982097411024606,
            -0.766915183691856, 1.5382586613794478, -0.725235732270125
        ]).reshape(&[batch_size, seq_len, num_heads * d_k]);

        let result = super::attention(&query, &key, &value, None, false);
        let diff = (&expect_result - &result)
            .square()
            .sum(expect_result.kind());
        assert!(f64::from(&diff) < 1e-9);

        let causal_result = super::attention(&query, &key, &value, None, true);
        let diff = (&expect_causal_result - &causal_result)
            .square()
            .sum(expect_result.kind());
        assert!(f64::from(&diff) < 1e-9);
    }
}
