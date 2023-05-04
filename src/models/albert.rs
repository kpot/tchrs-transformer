//! ALBERT is a model base on Tranformers that tries to keep the number of parameters
//! down by sharing weights across all its transformer blocks. Based on paper
//! [A Lite BERT for Self-supervised Learning of Language Representations](https://arxiv.org/abs/1909.11942)
//! by Zhenzhong Lan et al.

use tch::nn::Module;
use tch::{nn, Tensor};

use crate::attention::general::{AttentionStrategy, SelfMhaConfig};
use crate::block::{AttentionWiring, SelfAttentingBlock, TransformerBlockConfig};
use crate::embeddings::TiedOutputEmbedding;
use crate::positional_encoding::transformer_coordinate_encoding;

/// How position information should be added to the model.
#[derive(Debug, Clone)]
pub enum PositionSourceConfig {
    /// Indicates that a separate embedding matrix should encode positional information.
    /// Contains largest possible context the network will operate on.
    /// Must be bigger or equal than the input sequence's length.
    /// Defines the size of the positional embeddings.
    Embedding(usize), // (context length)
    /// Use coordinate encoding, described in paper "Univerasl Transformers"
    /// (https://arxiv.org/pdf/1807.03819.pdf), section 2.1.
    Encoding,
}

#[derive(Debug, Clone)]
pub struct AlbertConfig {
    pub embedding_dim: usize,
    pub vocab_size: usize,
    pub transformer_depth: usize,
    pub num_heads: usize,
    pub key_query_value_dim: usize,
    pub dropout: f64,
    pub position_source: PositionSourceConfig,
}

impl Default for AlbertConfig {
    fn default() -> Self {
        Self {
            transformer_depth: 12,
            embedding_dim: 128,
            vocab_size: 16384,
            num_heads: 64,
            key_query_value_dim: 64,
            dropout: 0.1,
            position_source: PositionSourceConfig::Encoding,
        }
    }
}

impl AlbertConfig {
    fn block_config(&self) -> TransformerBlockConfig {
        TransformerBlockConfig {
            input_output_dim: self.embedding_dim,
            key_query_value_dim: self.key_query_value_dim,
            num_heads: self.num_heads,
            attention_wiring: AttentionWiring::PreLayerNorm,
            dropout: self.dropout,
            causal: false,
        }
    }
}

pub enum TransformerPositionSource {
    Embedding(nn::Embedding),
    Encoding,
}

/// ALBERT is a model base on Tranformers that tries to keep the number of parameters
/// down by sharing weights across all its transformer blocks. Based on paper
/// [A Lite BERT for Self-supervised Learning of Language Representations](https://arxiv.org/abs/1909.11942)
/// by Zhenzhong Lan et al.
pub struct Albert<S>
where
    S: AttentionStrategy,
{
    pub token_embedding: nn::Embedding,
    pub transformer_step: SelfAttentingBlock<S>,
    pub config: AlbertConfig,
    pub input_layer_norm: nn::LayerNorm,
    pub position_source: TransformerPositionSource,
    pub output_embedding: TiedOutputEmbedding,
}

impl<S> Albert<S>
where
    S: AttentionStrategy,
    S::Config: From<TransformerBlockConfig> + SelfMhaConfig,
{
    pub fn new(vs: nn::Path, config: AlbertConfig) -> Self {
        let token_embedding = nn::embedding(
            &vs / "token_embedding",
            config.vocab_size as i64,
            config.embedding_dim as i64,
            Default::default(),
        );
        let output_embedding = TiedOutputEmbedding::new(
            &vs / "tied_output_embedding",
            config.embedding_dim,
            config.embedding_dim,
            config.vocab_size,
        );
        let transformer_step =
            SelfAttentingBlock::<S>::new(&vs / "transformer0", config.block_config());
        let input_layer_norm = nn::layer_norm(
            &vs / "input_layer_norm",
            vec![config.embedding_dim as i64],
            Default::default(),
        );
        let position_source = match config.position_source {
            PositionSourceConfig::Embedding(context_length) => {
                TransformerPositionSource::Embedding(nn::embedding(
                    &vs / "position_embedding",
                    context_length as i64,
                    config.embedding_dim as i64,
                    Default::default(),
                ))
            }
            PositionSourceConfig::Encoding => TransformerPositionSource::Encoding,
        };
        Albert {
            token_embedding,
            transformer_step,
            config,
            input_layer_norm,
            position_source,
            output_embedding,
        }
    }

    /// Runs the model. `input` is an integer tansor the shape of `(batch_size, sequence_length)`,
    /// and is used to feed the token ids (their range must fit the given vocabulary size).
    /// `input_offsets` is an integer tensor too and has the shape of `(batch_size,)`.
    /// It stores the number of tokens from the context that preceed the current sequence.
    /// This allows to properly select the positional embeddings.
    pub fn forward_t(
        &self,
        input: &Tensor,
        input_offsets: &Tensor,
        attention_masks: Option<&Tensor>,
        training: bool,
    ) -> Tensor {
        let (batch_size, sequence_len) = input.size2().expect("The input must be a 2D tensor");
        let tokens = self.token_embedding.forward(input);
        let mut step_output = tokens;
        if let TransformerPositionSource::Embedding(ref position_embedding) = self.position_source {
            // Adding position only in the beginning
            let position_indices = Tensor::arange(sequence_len, (tch::Kind::Int, input.device()))
                .reshape(&[1, sequence_len])
                .tile(&[batch_size, 1])
                + input_offsets.reshape(&[batch_size, 1]);

            step_output += position_embedding.forward(&position_indices)
        }
        for depth in 0..self.config.transformer_depth {
            if let TransformerPositionSource::Encoding = self.position_source {
                // Position is added to each level of the Transformer,
                // like it was done in paper "Univerasl Transformers"
                // (https://arxiv.org/pdf/1807.03819.pdf), section 2.1.
                step_output += transformer_coordinate_encoding(
                    input_offsets,
                    self.config.embedding_dim,
                    sequence_len as usize,
                    depth + 1,
                    step_output.kind(),
                );
            }
            step_output = self
                .transformer_step
                .forward_t(&step_output, attention_masks, training);
        }
        self.output_embedding
            .forward(&self.token_embedding.ws, &step_output)
    }
}
