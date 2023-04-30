use tch::{nn, Tensor};

/// Abstracts a particular method used to calculate attention.
/// Allows to quickly implement various "efficient attentions"
/// while reusing the code projecting inputs and outputs.
pub trait AttentionStrategy {
    type Config: std::fmt::Debug + Clone;

    fn new(vs: nn::Path, config: &Self::Config) -> Self;

    fn attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        attention_mask: Option<&Tensor>,
        training: bool,
    ) -> Tensor;
}

/// Abstract configuration for a layer implementing multi-head attention,
/// based on [`GeneratlMultiHeadAttention`].
pub trait MhaConfig {
    /// Number of attention heads
    fn num_heads(&self) -> usize;
    /// Dimensionality of keys and queries in each head
    fn head_key_query_dim(&self) -> usize;
    /// Dimensionality of values in each head
    fn head_value_dim(&self) -> usize;
    /// Dimensionality of the input for projecting queries
    fn input_query_dim(&self) -> usize;
    /// Dimensionality of the input for projecting keys and values
    fn input_key_value_dim(&self) -> usize;
    /// dimensionality of the output vector all heads combined will be projected to
    fn output_dim(&self) -> usize;
}

/// Config for a multi-head attention that supports causal masking, allowing each
/// position to attent only to itself and the previous positions.
pub trait CausalMhaConfig {
    fn is_causal(&self) -> bool;
}

/// A generalized multi-head attention described in paper "Attention Is All You Need"
/// (https://arxiv.org/pdf/1706.03762.pdf).
///
/// This implementation is designed to be abstract in that:
///
/// 1. It does not specify how exactly the attention should work. Which makes it possible
///    to combine it with various implementations of "efficient attentions".
/// 2. It accepts 2 separate inputs: one for queries and another for the keys/values.
///    Input for keys and values can have sequence length different from the sequence
///    length of queries.
///
///    This allows for construction of encoder-decoder models, in which a decoder
///    "interrogates" through its queries whatever values were produced by an encoder.
#[derive(Debug)]
pub struct GeneralMultiHeadAttention<S: AttentionStrategy> {
    pub strategy: S,
    pub config: S::Config,
    pub q_weights: Tensor,
    pub k_weights: Tensor,
    pub v_weights: Tensor,
    pub output_weights: Tensor,
}

impl<S> GeneralMultiHeadAttention<S>
where
    S: AttentionStrategy,
    S::Config: MhaConfig,
{
    pub fn new(vs: nn::Path, c: S::Config) -> Self {
        // These weights are concatenated matrices W_q, W_k and W_v which
        // are, in turn, concatenated W matrices of keys, queries and values
        // for each of the heads. So, essentially it's a concatenation of
        // W_q1, W_q2,..., W_qh, W_k1, W_k2,..., W_kh, W_v1, W_v2,..., W_vh
        // for all h heads.
        let value_dim = (c.head_value_dim() * c.num_heads()) as i64;
        let v_weights =
            vs.kaiming_uniform("v_weights", &[c.input_key_value_dim() as i64, value_dim]);
        let key_query_dim = (c.head_key_query_dim() * c.num_heads()) as i64;
        let k_weights = vs.kaiming_uniform(
            "k_weights",
            &[c.input_key_value_dim() as i64, key_query_dim],
        );
        let q_weights =
            vs.kaiming_uniform("q_weights", &[c.input_query_dim() as i64, key_query_dim]);
        let output_weights =
            vs.kaiming_uniform("output_weights", &[value_dim, c.output_dim() as i64]);
        let strategy = S::new(vs, &c);

        Self {
            strategy,
            q_weights,
            k_weights,
            v_weights,
            output_weights,
            config: c,
        }
    }

    /// Multi-head attention. Inputs must be of the following dimensions:
    ///
    /// * `query_input`: `[batch_size, query_seq_len, input_query_dim]`.
    /// * `key_value_input`: `[batch_size, key_value_seq_len, input_key_value_dim]`.
    /// * `value_input`: `[batch_size, key_value_seq_len, value_input_dim]`.
    /// * `attention_mask`: 1s or 0s for keys that should and should not be attended,
    ///    shaped as (batch_size, key_value_seq_len)
    ///
    /// Returns a tensor the shape of `[batch_size, value_seq_len, output_dim]`
    pub fn forward_t(
        &self,
        query_input: &Tensor,
        key_value_input: &Tensor,
        attention_mask: Option<&Tensor>,
        training: bool,
    ) -> Tensor {
        let (value_batch_size, value_seq_len, _) =
            key_value_input.size3().expect("Values must be a 3D tensor");
        let (query_batch_size, query_seq_len, _) =
            query_input.size3().expect("Queries must be a 3D tensor");
        assert_eq!(
            value_batch_size, query_batch_size,
            "Value and query sequences must have the same batch sizes. Currently {} and {}",
            value_batch_size, query_batch_size
        );
        let batch_size = value_batch_size;
        let kv_seq_len = value_seq_len;
        //  The first thing we need to do is to perform affine transformations
        //  of the inputs to get the Queries, the Keys and the Values.
        //  Each will have shape (batch_size, num_heads, sequence_len, dim)
        let v = key_value_input.matmul(&self.v_weights).reshape(&[
            batch_size,
            kv_seq_len,
            self.config.num_heads() as i64,
            self.config.head_value_dim() as i64,
        ]);
        let k = key_value_input.matmul(&self.k_weights).reshape(&[
            batch_size,
            kv_seq_len,
            self.config.num_heads() as i64,
            self.config.head_key_query_dim() as i64,
        ]);
        let q = query_input.matmul(&self.q_weights).reshape(&[
            batch_size,
            query_seq_len,
            self.config.num_heads() as i64,
            self.config.head_key_query_dim() as i64,
        ]);
        let attention_out = self
            .strategy
            .attention(&q, &k, &v, attention_mask, training);
        attention_out.matmul(&self.output_weights)
    }
}

/// Cofiguration for multi-head self-attention, implemented by [`GeneratlMultiHeadSelfAttention`].
pub trait SelfMhaConfig {
    /// Dimensionality of the input for projecting queries
    fn input_dim(&self) -> usize;
    /// Number of attention heads
    fn num_heads(&self) -> usize;
    /// Dimensionality of keys, queries and values in each head
    fn head_key_query_value_dim(&self) -> usize;
    /// dimensionality of the output vector all heads combined will be projected to
    fn output_dim(&self) -> usize;
}

/// A generalized multi-head self-attention described in paper
/// "Attention Is All You Need" (https://arxiv.org/pdf/1706.03762.pdf).
/// It doesn't specify how exactly the attention must be calculated, which
/// allows to plug-in various methods of doing so.
///
/// This implementation is designed and optimized for a case of self-attention,
/// when we have a single input tensor from which the keys, the values,
/// and the queries (`K`, `V`, and `Q`) are projected, all of the same shape.
/// This allows to clump all projection matrices into a single one and calculating
/// the projections faster, in one sweep.
#[derive(Debug)]
pub struct GeneralMultiHeadSelfAttention<S>
where
    S: AttentionStrategy,
{
    pub strategy: S,
    pub config: S::Config,
    pub qkv_weights: Tensor,
    pub output_weights: Tensor,
}

impl<S> GeneralMultiHeadSelfAttention<S>
where
    S: AttentionStrategy,
    S::Config: SelfMhaConfig,
{
    pub fn new(vs: nn::Path, config: S::Config) -> Self {
        // * 3 for q, k and v
        let qkv_dim = [
            config.input_dim() as i64,
            (config.num_heads() * config.head_key_query_value_dim() * 3) as i64,
        ];

        // These weights are concatenated matrices W_q, W_k and W_v which
        // are, in turn, concatenated W matrices of keys, queries and values
        // for each of the heads. So, essentially it's a concatenation of
        // W_q1, W_q2,..., W_qh, W_k1, W_k2,..., W_kh, W_v1, W_v2,..., W_vh
        // for all h heads.
        let qkv_weights = vs.kaiming_uniform("qkv_weights", &qkv_dim);
        let output_shape = [
            (config.head_key_query_value_dim() * config.num_heads()) as i64,
            config.output_dim() as i64,
        ];
        let output_weights = vs.kaiming_uniform("output_weights", &output_shape);
        let strategy = S::new(vs, &config);
        Self {
            config,
            strategy,
            qkv_weights,
            output_weights,
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
    pub fn forward_t(
        &self,
        input: &Tensor,
        attention_mask: Option<&Tensor>,
        training: bool,
    ) -> Tensor {
        let (batch_size, seq_len, _) = input.size3().expect("Input must be a 3D tensor");

        let qkv = input.matmul(&self.qkv_weights).reshape(&[
            batch_size,
            seq_len,
            3,
            self.config.num_heads() as i64,
            self.config.head_key_query_value_dim() as i64,
        ]);

        let seam_dim_index = 2;
        // Splitting the keys, the values and the queries before further processing.
        // Each will have shape (batch_size, seq_len, 1, num_heads, d_k).
        // The redundant 1 dimension will be squeezed out later.
        let qkv_chunks = qkv.split(1, seam_dim_index); // vec![q, k, v]
        let attention_out = self.strategy.attention(
            &qkv_chunks[0].squeeze_dim(seam_dim_index),
            &qkv_chunks[1].squeeze_dim(seam_dim_index),
            &qkv_chunks[2].squeeze_dim(seam_dim_index),
            attention_mask,
            training,
        );
        // Output projection
        attention_out.matmul(&self.output_weights)
    }
}
