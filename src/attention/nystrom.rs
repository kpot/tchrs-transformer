//! An "efficient" self-attention presented in paper
//! "Nyströmformer: A Nyström-Based Algorithm for Approximating Self-Attention"
//! by Yunyang Xiong et al.
use tch::{
    nn::{self, Module},
    Tensor,
};

use super::general::{
    AttentionStrategy, GeneralMultiHeadAttention, GeneralMultiHeadSelfAttention, MhaConfig,
    SelfMhaConfig,
};

pub trait NystromMhaConfig {
    fn conv_kernel_size(&self) -> Option<usize>;
    fn num_landmarks(&self) -> usize;
}

#[derive(Debug, Clone)]
pub struct NystromAttentionConfig {
    /// Last dimension of the input from which the keys and the values will be created
    pub input_key_value_dim: usize,
    /// Last dimension of input from which the queries are constructed
    pub input_query_dim: usize,
    /// size of the output dimension all the heads combined are projected to
    pub output_dim: usize,
    /// number of attention heads
    pub num_heads: usize,
    /// size of the internal key/query vector in a single attention head
    pub head_key_query_dim: usize,
    /// size of the internal value vector from a single attention head.
    pub head_value_dim: usize,
    /// how many landmarks (key/query vectors for approximating self-attention) to use
    pub num_landmarks: usize,
    /// kernel size for
    /// "A skip connection of value V, implemented using a 1D depthwise convolution
    /// ... to help the training".
    /// Authors use conv_kernel_size=33 in their original implementation,
    /// although the optimal size likely depends on the task at hands.
    pub conv_kernel_size: Option<usize>,
}

impl NystromAttentionConfig {
    pub fn new(
        key_value_input_dim: usize,
        query_input_dim: usize,
        num_heads: usize,
        output_dim: usize,
    ) -> Self {
        const KEY_QUERY_DIM: usize = 64;
        Self {
            input_query_dim: query_input_dim,
            input_key_value_dim: key_value_input_dim,
            num_heads,
            head_key_query_dim: KEY_QUERY_DIM,
            head_value_dim: KEY_QUERY_DIM,
            output_dim,
            num_landmarks: 64,
            conv_kernel_size: None,
        }
    }
}

#[rustfmt::skip]
impl MhaConfig for NystromAttentionConfig {
    fn num_heads(&self) -> usize { self.num_heads }
    fn head_key_query_dim(&self) -> usize { self.head_key_query_dim }
    fn head_value_dim(&self) -> usize { self.head_value_dim }
    fn input_query_dim(&self) -> usize { self.input_query_dim }
    fn input_key_value_dim(&self) -> usize { self.input_key_value_dim }
    fn output_dim(&self) -> usize { self.output_dim }
}

#[rustfmt::skip]
impl NystromMhaConfig for NystromAttentionConfig {
    fn conv_kernel_size(&self) -> Option<usize> { self.conv_kernel_size }
    fn num_landmarks(&self) -> usize { self.num_landmarks }
}

#[derive(Clone, Debug)]
pub struct NystromSelfAttentionConfig {
    pub input_dim: usize,
    pub num_heads: usize,
    pub head_key_query_value_dim: usize,
    pub output_dim: usize,
    pub num_landmarks: usize,
    pub conv_kernel_size: Option<usize>,
}

#[rustfmt::skip]
impl SelfMhaConfig for NystromSelfAttentionConfig {
    fn input_dim(&self) -> usize { self.input_dim }
    fn num_heads(&self) -> usize { self.num_heads }
    fn head_key_query_value_dim(&self) -> usize { self.head_key_query_value_dim }
    fn output_dim(&self) -> usize { self.output_dim }
}

#[rustfmt::skip]
impl MhaConfig for NystromSelfAttentionConfig {
    fn num_heads(&self) -> usize { self.num_heads }
    fn head_key_query_dim(&self) -> usize { self.head_key_query_value_dim }
    fn head_value_dim(&self) -> usize { self.head_key_query_value_dim }
    fn input_query_dim(&self) -> usize { self.input_dim }
    fn input_key_value_dim(&self) -> usize { self.input_dim }
    fn output_dim(&self) -> usize { self.output_dim }
}

#[rustfmt::skip]
impl NystromMhaConfig for NystromSelfAttentionConfig {
    fn conv_kernel_size(&self) -> Option<usize> { self.conv_kernel_size }
    fn num_landmarks(&self) -> usize { self.num_landmarks }
}

/// Nystrom attention itself. Implementation is based
/// on [code](https://github.com/mlpen/Nystromformer) published
/// by the authors of the original paper.
#[derive(Debug)]
pub struct NystromAttentionStrategy<ConfigType> {
    config: ConfigType,
    pub conv_kernel: Option<nn::Conv<[i64; 2]>>,
}

impl<ConfigType> AttentionStrategy for NystromAttentionStrategy<ConfigType>
where
    ConfigType: NystromMhaConfig + MhaConfig + std::fmt::Debug + Clone,
{
    type Config = ConfigType;

    fn new(vs: nn::Path, config: &Self::Config) -> Self {
        let conv_kernel = config.conv_kernel_size().as_ref().map(|kernel_size| {
            let conv_config = nn::ConvConfigND::<[i64; 2]> {
                padding: [*kernel_size as i64 / 2, 0],
                stride: [1, 1],
                bias: false,
                groups: config.num_heads() as i64,
                ..Default::default()
            };
            nn::conv(
                &vs / "residual",
                config.num_heads() as i64,
                config.num_heads() as i64,
                [*kernel_size as i64, 1],
                conv_config,
            )
        });
        Self {
            conv_kernel,
            config: config.clone(),
        }
    }

    fn attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        attention_mask: Option<&Tensor>,
        _training: bool,
    ) -> Tensor {
        let (batch_size, seq_len, _, k_dim) = k.size4().expect("Keys must be a 4D tensor");
        let k_landmarks = form_landmarks(k, self.config.num_landmarks());
        let q_landmarks = form_landmarks(q, self.config.num_landmarks());
        // reshaping to facilitates the use of batch matmul down the line
        let q_t = q.transpose(2, 1); // to (batch_size, num_heads, seq_len, d_q)
        let k_t = k.permute(&[0, 2, 3, 1]); // (batch, num_heads, d_k, seq_len)
        let k_landmarks_t = k_landmarks.transpose(-2, -1); // (batch, num_heads, d_k, seq_len)
        let sqrt_d = (k_dim as f64).sqrt();

        let f_kernel = (q_t.matmul(&k_landmarks_t) / sqrt_d).softmax(-1, q.kind());
        let mut pre_b_kernel = q_landmarks.matmul(&k_t) / sqrt_d;
        if let Some(attention_mask) = attention_mask {
            pre_b_kernel += -1e5 * (1.0 - attention_mask.view([batch_size, 1, 1, -1]));
        }
        let b_kernel = pre_b_kernel.softmax(-1, q.kind());
        let a_kernel = (q_landmarks.matmul(&k_landmarks_t) / sqrt_d).softmax(-1, q.kind());
        let a_kernel_inv = fixed_iterative_inv(&a_kernel, 6, true);
        //  A full attention matrix would look like this:
        //  full_attention = (q.matmul(k_t) / sqrt_d).softmax(-1, q.kind())
        let v_t = v.transpose(2, 1); // to (batch_size, num_heads, seq_len, d_q)
        let mut scaled_attention = f_kernel
            .matmul(&a_kernel_inv)
            .matmul(&b_kernel.matmul(&v_t));
        if let Some(ref conv) = self.conv_kernel {
            // TODO: Add tests
            scaled_attention += conv.forward(&v_t.transpose(-2, -1));
        }
        // "Concatenating" heads by rearranging dimensions
        //  and reshaping the result to (batch, seq_len, num_heads * value_dim)
        scaled_attention.transpose(2, 1).reshape(&[
            -1,
            seq_len,
            (self.config.num_heads() * self.config.head_value_dim()) as i64,
        ])
    }
}

/// An "efficient" self-attention presented in paper
/// "Nyströmformer: A Nyström-Based Algorithm for Approximating Self-Attention"
/// by Yunyang Xiong et al.
///
/// This version can use two inputs:
/// First: from the encoder - it's used to project the keys and the values
/// Second: from the decoder - used to project the queries.
pub type NystromMultiHeadAttention =
    GeneralMultiHeadAttention<NystromAttentionStrategy<NystromAttentionConfig>>;
pub type NystromMultiHeadSelfAttention =
    GeneralMultiHeadSelfAttention<NystromAttentionStrategy<NystromSelfAttentionConfig>>;

/// Reduces a tensor of multi-head keys or queries of a sequence
/// into a similar tensor of a smaller sequence of "landmarks", or returns
/// the original tensor if its sequence length is already equal or shorter
/// than the `num_landmarks`.
/// Returns either the landmarks or the input tensor `t`
/// transposed to (batch, num_heads, seq_len, head_dim)
fn form_landmarks(t: &Tensor, num_landmarks: usize) -> Tensor {
    const CHUNK_DIM: Option<&[i64]> = Some(&[-2]);
    // (batch, num_heads, seq_len, head_dim)
    let t_cont_sequence = t.transpose(2, 1);
    let (batch_size, seq_len, num_heads, head_dim) = t
        .size4()
        .expect("A 4D tensor is expected for calculating landmarks");
    if seq_len <= num_landmarks as i64 {
        t_cont_sequence
    } else {
        // length of each landmark's chunk within the sequence, ideally (if divisible)
        let chunk_len = seq_len / num_landmarks as i64;
        if seq_len % num_landmarks as i64 == 0 {
            // Our sequence is perfectly divisible to num_landmarks each size of `chunk_len`
            t_cont_sequence
                .reshape(&[
                    batch_size,
                    num_heads,
                    num_landmarks as i64,
                    chunk_len,
                    head_dim,
                ])
                .mean_dim(CHUNK_DIM, false, t.kind())
        } else {
            // When a sequence is not evenly divisible into `num_landmarks`
            // the size of `chunk_len`, we can divide it into
            // some landmarks the size of `chunk_len`,
            // and then some landmarks the size of `chunk_len + 1`
            let num_normal_landmarks = (chunk_len + 1) * num_landmarks as i64 - seq_len;
            let num_larger_landmarks = num_landmarks as i64 - num_normal_landmarks;
            let divisible_sequence_len = num_normal_landmarks * chunk_len;
            let smaller_landmarks = t_cont_sequence
                .slice(2, 0, divisible_sequence_len, 1)
                .reshape(&[
                    batch_size,
                    num_heads,
                    num_normal_landmarks,
                    chunk_len,
                    head_dim,
                ])
                .mean_dim(CHUNK_DIM, false, t.kind());
            let larger_landmarks = t_cont_sequence
                .slice(2, divisible_sequence_len, seq_len, 1)
                .reshape(&[
                    batch_size,
                    num_heads,
                    num_larger_landmarks,
                    chunk_len + 1,
                    head_dim,
                ])
                .mean_dim(CHUNK_DIM, false, t.kind());
            Tensor::cat(&[smaller_landmarks, larger_landmarks], -2)
        }
    }
}

/// For a given square matrix `m`, calculates initial approximation for a pseudo-inverse
/// matrix as required by the iterative algorithm based on paper "A New Iterative Method
/// for Finding Approximate Inverses of Complex Matrices" by M. Kafaei Razavi et al.
/// https://www.hindawi.com/journals/aaa/2014/563787/
///
/// `from_softmax` indicates that the input matrix is a result of softmax operation
/// along the last axis. Which means that the sum of all elements per row is equal to 1.
/// This makes inf-norm of the matrix also equal to 1 and simplifies
/// inital matrix calculation.
fn initial_inv_matrix(m: &Tensor, from_softmax: bool) -> Tensor {
    let m_t = m.transpose(-2, -1);
    let m_abs = m.abs();
    const ROW_DIM: Option<&[i64]> = Some(&[-2]);
    const COL_DIM: Option<&[i64]> = Some(&[-1]);
    let dtype = m.kind();
    let m_norm1 = m_abs
        .sum_dim_intlist(ROW_DIM, true, dtype)
        .max_dim(-1, true)
        .0;
    if from_softmax {
        m_t / m_norm1
    } else {
        let m_norm_inf = m_abs
            .sum_dim_intlist(COL_DIM, true, dtype)
            .max_dim(-2, true)
            .0;
        m_t / (m_norm1 * m_norm_inf)
    }
}

/// Assuming we have a square `matrix` (or any tensor where the last 2 dimensions
/// are the same), this fuction calculates its inverse matrix using iterative algorithm,
/// while always performing a fixed number of steps.
///
/// Based on "A New Iterative Method for Finding Approximate Inverses
/// of Complex Matrices" by M. Kafaei Razavi et al.
/// https://www.hindawi.com/journals/aaa/2014/563787/
///
/// We assume that A is a square matrix.
fn fixed_iterative_inv(matrix: &Tensor, num_iterations: usize, from_softmax: bool) -> Tensor {
    let size = matrix.size();
    debug_assert!(
        size.len() >= 2 && size[size.len() - 1] == size[size.len() - 2],
        "A square matrix is expected"
    );
    let last_dim = size[size.len() - 1];
    let mut v = initial_inv_matrix(matrix, from_softmax);
    let i = Tensor::eye(last_dim, (matrix.kind(), matrix.device()));
    for _ in 0..num_iterations {
        let av = matrix.matmul(&v);
        v = 0.25 * v.matmul(&(13.0 * &i - av.matmul(&(15.0 * &i - av.matmul(&(7.0 * &i - &av))))));
    }
    v
}

#[cfg(test)]
mod tests {
    use tch::{nn, Tensor};

    use super::{fixed_iterative_inv, NystromAttentionConfig};

    /// Verifies that given several matrices, iterative pseudoinversion algorithm
    /// can get something close enough to actual iverse matrices.
    #[test]
    fn test_pinv() {
        #[rustfmt::skip]
        let A = Tensor::of_slice(&[
            0.6734755, -0.72500587, -0.80281806, -0.00983024,
            -0.3860941, -0.28099728, -0.5887611, 0.29009724,
            -0.6402378, 0.81629944, -0.1511004, -0.4696927,
            -0.86275196, -0.728971, -0.32663107, -0.95297,

             -0.42163205, -0.4410572, 0.08483005, 0.67091155,
             0.86466885, -0.23998046, 0.989794, 0.2590356,
             -0.22674155, 0.24751282, 0.01441097, -0.46450257,
             -0.635262, 0.3867767, 0.68246555, -0.75195813
        ]).reshape(&[2, 4, 4]);
        let I = Tensor::eye(4, (A.kind(), A.device()))
            .unsqueeze(0)
            .tile(&[2, 1, 1]);
        let A_inv = fixed_iterative_inv(&A, 6, false);
        assert!(f64::from((A.matmul(&A_inv) - I).abs().max()) < 1e-5);
    }
}
