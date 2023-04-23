use tch::nn;
use tch::nn::Module;
use tch::Tensor;

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
            .view([-1, input_dim]);
        // matching with the embedding
        // the embedding matrix is expected to be the shape of (vocabulary_size, embedding_dim)
        // output shaped as (batch_size, seq_len, vocab_size)
        (projected.matmul(&input_embedding_matrix.transpose(-2, -1))
            + self.output_bias.view([1, -1]))
        .reshape(&[batch_size, seq_len, -1])
    }
}
