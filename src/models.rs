use tch::nn::{self, LayerNormConfig};
use tch::nn::{Module, ModuleT};
use tch::Tensor;

use crate::attention::MultiHeadSelfAttention;

#[derive(Debug, Clone)]
pub struct AlbertConfig {
    /// Largest possible context the network will operate on. Must be bigger or equal than
    /// the input sequence's length. Defines the size of the positional embeddings.
    pub context_len: usize,
    pub embedding_dim: usize,
    pub vocab_size: usize,
    pub transformer_depth: usize,
    pub num_heads: usize,
    pub key_query_value_dim: usize,
    pub residual_dropout: f64,
}

impl Default for AlbertConfig {
    fn default() -> Self {
        Self {
            transformer_depth: 12,
            context_len: 1024,
            embedding_dim: 512,
            vocab_size: 16384,
            num_heads: 12,
            key_query_value_dim: 64,
            residual_dropout: 0.1,
        }
    }
}

/// Allows to reuse the same word embedding matrix both for the input and
/// the output layers of the network.
/// This is called Weight Tying and is proven to improve performance
/// of neural network language models, as well as decrease their number
/// of parameters (eliminating the need for a separate huge matrix
/// of output weights).

/// The module is supposed to be called with a shared `embedded_matrix`, and
/// an input which comes from previous layer (like LSTM or Transformer).
/// `embedding_matrix` is supposed to be the shape of `[vocabulary_size, embedding_dim]`.
///
/// Based on papers:
///
/// * [Using the Output Embedding to Improve Language Models](https://arxiv.org/abs/1608.05859) by Press et al.
/// * [Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling](https://arxiv.org/abs/1611.01462)
/// * [Improving language understanding with unsupervised learning](https://openai.com/research/language-unsupervised)
#[derive(Debug)]
pub struct TiedOutputEmbedding {
    input_projection: nn::Linear,
    layer_norm: nn::LayerNorm,
    output_bias: Tensor,
}

impl TiedOutputEmbedding {
    pub fn new(vs: nn::Path, input_dim: usize, embedding_dim: usize, vocab_size: usize) -> Self {
        // let projection = vs.kaiming_uniform("kernel", &[input_dim as i64, embedding_dim as i64]);
        let input_projection = nn::linear(
            &vs / "input_projection",
            input_dim as i64,
            embedding_dim as i64,
            Default::default(),
        );
        let layer_norm = nn::layer_norm(
            &vs / "layer_norm",
            vec![input_dim as i64],
            Default::default(),
        );
        let output_bias = vs.zeros("output_bias", &[vocab_size as i64]);
        Self {
            input_projection,
            layer_norm,
            output_bias,
        }
    }

    pub fn forward(&self, input_embedding_matrix: &Tensor, xs: &Tensor) -> Tensor {
        let (batch_size, seq_len, input_dim) = xs.size3().expect("The input must be a 3D tensor");
        let projected = self
            .input_projection
            .forward(xs)
            .gelu("none")
            .apply(&self.layer_norm)
            .view([-1, input_dim as i64]);
        // matching with the embedding
        // the embedding matrix is expected to be the shape of (vocabulary_size, embedding_dim)
        // output shaped as (batch_size, seq_len, vocab_size)
        (projected.matmul(&input_embedding_matrix.transpose(-2, -1))
            + self.output_bias.view([1, -1]))
        .reshape(&[batch_size, seq_len, -1])
    }
}

#[derive(Debug)]
pub struct AlbertTransformerBlock {
    norm1_layer: nn::LayerNorm,
    norm2_layer: nn::LayerNorm,
    transition1_layer: nn::Linear,
    transition2_layer: nn::Linear,
    attention: MultiHeadSelfAttention,
    dropout: f64,
}

impl AlbertTransformerBlock {
    pub fn new(p: nn::Path, config: AlbertConfig) -> Self {
        const TRANSITION_SCALER: usize = 4;
        let norm1_layer = nn::layer_norm(
            &p / "layer_norm1",
            vec![config.embedding_dim as i64],
            Default::default(),
        );
        let norm2_layer = nn::layer_norm(
            &p / "layer_norm2",
            vec![config.embedding_dim as i64],
            Default::default(),
        );
        let transition_dim = (TRANSITION_SCALER * config.embedding_dim) as i64;
        let transition1_layer = nn::linear(
            &p / "trans1",
            config.embedding_dim as i64,
            transition_dim,
            Default::default(),
        );
        let transition2_layer = nn::linear(
            &p / "trans2",
            transition_dim,
            config.embedding_dim as i64,
            Default::default(),
        );
        let attention = MultiHeadSelfAttention::new(
            &(p / "attention"),
            false,
            config.embedding_dim,
            config.num_heads,
            config.key_query_value_dim,
            Some(config.embedding_dim),
        );
        Self {
            norm1_layer,
            norm2_layer,
            transition1_layer,
            transition2_layer,
            attention,
            dropout: config.residual_dropout,
        }
    }
}

impl nn::ModuleT for AlbertTransformerBlock {
    fn forward_t(&self, input: &Tensor, train: bool) -> Tensor {
        let att = self.attention.forward(input);
        let post_residual1 = att.dropout(self.dropout, train) + input;
        let norm1_output = post_residual1.apply(&self.norm1_layer);
        let post_transitional = norm1_output
            .apply(&self.transition1_layer)
            .gelu("none")
            .apply(&self.transition2_layer);
        let post_residual2 = post_transitional.dropout(self.dropout, train) + norm1_output;
        let output = post_residual2.apply(&self.norm2_layer);
        output
    }
}

/// ALBERT is a model base on Tranformers that tries to keep the number of parameters
/// down by sharing weights across all its transformer blocks. Based on paper
/// [A Lite BERT for Self-supervised Learning of Language Representations](https://arxiv.org/abs/1909.11942)
/// by Zhenzhong Lan et al.
pub struct Albert {
    pub token_embedding: nn::Embedding,
    pub position_embedding: nn::Embedding,
    pub transformer_step: AlbertTransformerBlock,
    pub config: AlbertConfig,
    pub output_embedding: TiedOutputEmbedding,
    pub input_layer_norm: nn::LayerNorm,
}

impl Albert {
    pub fn new(vs: nn::Path, config: AlbertConfig) -> Self {
        let token_embedding = nn::embedding(
            &vs / "token_embedding",
            config.vocab_size as i64,
            config.embedding_dim as i64,
            Default::default(),
        );
        let position_embedding = nn::embedding(
            &vs / "position_embedding",
            config.context_len as i64,
            config.embedding_dim as i64,
            Default::default(),
        );
        let output_embedding = TiedOutputEmbedding::new(
            &vs / "tied_output_embedding",
            config.embedding_dim,
            config.embedding_dim,
            config.vocab_size,
        );
        let transformer_step = AlbertTransformerBlock::new(&vs / "transformer", config.clone());
        let input_layer_norm = nn::layer_norm(
            &vs / "input_layer_norm",
            vec![config.embedding_dim as i64],
            Default::default(),
        );
        Albert {
            token_embedding,
            position_embedding,
            transformer_step,
            output_embedding,
            config,
            input_layer_norm,
        }
    }

    /// Runs the model. `input` is an integer tansor the shape of `(batch_size, sequence_length)`,
    /// and is used to feed the token ids (their range must fit the given vocabulary size).
    /// `input_offsets` is an integer tensor too and has the shape of `(batch_size,)`.
    /// It stores the number of tokens from the context that preceed the current sequence.
    /// This allows to properly select the positional embeddings.
    pub fn forward_t(&self, input: &Tensor, input_offsets: &Tensor, training: bool) -> Tensor {
        let (batch_size, sequence_len) = input.size2().expect("The input must be a 2D tensor");
        let position_indices = Tensor::arange(sequence_len, (tch::Kind::Int, input.device()))
            .reshape(&[1, sequence_len])
            .tile(&[batch_size, 1])
            + input_offsets.reshape(&[batch_size, 1]);
        let tokens = self.token_embedding.forward(input);
        let positions = self.position_embedding.forward(&position_indices);
        let mut step_output = (tokens + positions)
            .apply(&self.input_layer_norm)
            .dropout(self.config.residual_dropout, training);
        for _depth in 0..self.config.transformer_depth {
            step_output = self.transformer_step.forward_t(&step_output, training);
        }
        let output = self
            .output_embedding
            .forward(&self.token_embedding.ws, &step_output);
        output
    }
}
