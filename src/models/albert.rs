//! ALBERT is a model base on Tranformers that tries to keep the number of parameters
//! down by sharing weights across all its transformer blocks. Based on paper
//! [A Lite BERT for Self-supervised Learning of Language Representations](https://arxiv.org/abs/1909.11942)
//! by Zhenzhong Lan et al.

use tch::nn::Module;
use tch::{nn, Tensor};

use crate::attention::standard::MultiHeadSelfAttention;
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
            transformer_depth: 5,
            embedding_dim: 512,
            vocab_size: 16384,
            num_heads: 12,
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
        }
    }
}

/// Two classical types of wiring attention in Transformers, as described in
/// paper ["Understanding the Difficulty of Training Transformers"](https://arxiv.org/abs/2004.08249)
#[derive(Clone, Debug)]
pub enum AttentionWiring {
    /// Pre-attention: Layer Normalization is done before the attention,
    /// and can be fully bypassed through residual conections.
    PreLayerNorm,
    /// Post-attention:  Layer Normalization is done after the attention,
    /// and cannot be bypassed through residual connections.
    PostLayerNorm,
}

#[derive(Debug, Clone)]
pub struct TransformerBlockConfig {
    /// Input and output last dimension size
    pub input_output_dim: usize,
    /// dimensionality of keys, queries and values in each attention head
    pub key_query_value_dim: usize,
    /// how many attention heads to use
    pub num_heads: usize,
    /// whether to use Pre-LN or Post-LN wiring
    pub attention_wiring: AttentionWiring,
    /// Dropout for all stages
    pub dropout: f64,
}

#[derive(Debug)]
pub struct TransformerBlock {
    norm1_layer: nn::LayerNorm,
    norm2_layer: nn::LayerNorm,
    transition1_layer: nn::Linear,
    transition2_layer: nn::Linear,
    attention: MultiHeadSelfAttention,
    config: TransformerBlockConfig,
}

impl TransformerBlock {
    pub fn new(p: nn::Path, config: TransformerBlockConfig) -> Self {
        const TRANSITION_SCALER: usize = 4;
        let attention = MultiHeadSelfAttention::new(
            &p / "attention",
            false,
            config.input_output_dim,
            config.num_heads,
            config.key_query_value_dim,
            Some(config.input_output_dim),
        );
        let norm1_layer = nn::layer_norm(
            &p / "layer_norm1",
            vec![config.input_output_dim as i64],
            Default::default(),
        );
        let norm2_layer = nn::layer_norm(
            &p / "layer_norm2",
            vec![config.input_output_dim as i64],
            Default::default(),
        );
        let transition_dim = (TRANSITION_SCALER * attention.output_dim) as i64;
        let transition1_layer = nn::linear(
            &p / "trans1",
            config.input_output_dim as i64,
            transition_dim,
            Default::default(),
        );
        let transition2_layer = nn::linear(
            &p / "trans2",
            transition_dim,
            config.input_output_dim as i64,
            Default::default(),
        );
        Self {
            config,
            norm1_layer,
            norm2_layer,
            transition1_layer,
            transition2_layer,
            attention,
        }
    }

    pub fn forward_t(
        &self,
        input: &Tensor,
        attention_mask: Option<&Tensor>,
        train: bool,
    ) -> Tensor {
        match self.config.attention_wiring {
            AttentionWiring::PreLayerNorm => self.pre_ln_forward_t(input, attention_mask, train),
            AttentionWiring::PostLayerNorm => self.post_ln_forward_t(input, attention_mask, train),
        }
    }

    fn post_ln_forward_t(
        &self,
        input: &Tensor,
        attention_mask: Option<&Tensor>,
        train: bool,
    ) -> Tensor {
        let att = self
            .attention
            .forward(input, attention_mask)
            .dropout(self.config.dropout, train);
        let post_residual1 = att + input;
        let norm1_output = post_residual1.apply(&self.norm1_layer);
        let post_transitional = norm1_output
            .apply(&self.transition1_layer)
            .gelu("none")
            .dropout(self.config.dropout, train)
            .apply(&self.transition2_layer)
            .dropout(self.config.dropout, train);
        let post_residual2 = post_transitional + norm1_output;
        post_residual2.apply(&self.norm2_layer)
    }

    fn pre_ln_forward_t(
        &self,
        input: &Tensor,
        attention_mask: Option<&Tensor>,
        train: bool,
    ) -> Tensor {
        let norm1_output = input.apply(&self.norm1_layer);
        let att = self
            .attention
            .forward(&norm1_output, attention_mask)
            .dropout(self.config.dropout, train);
        let post_residual1 = att + input;
        let post_transitional = post_residual1
            .apply(&self.norm2_layer)
            .apply(&self.transition1_layer)
            .gelu("none")
            .dropout(self.config.dropout, train)
            .apply(&self.transition2_layer)
            .dropout(self.config.dropout, train);
        post_transitional + post_residual1
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
pub struct Albert {
    pub token_embedding: nn::Embedding,
    pub transformer_step: TransformerBlock,
    pub config: AlbertConfig,
    pub input_layer_norm: nn::LayerNorm,
    pub position_source: TransformerPositionSource,
    pub output_embedding: TiedOutputEmbedding,
}

impl Albert {
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
        let transformer_step = TransformerBlock::new(&vs / "transformer0", config.block_config());
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
            step_output =
                self.transformer_step
                    .pre_ln_forward_t(&step_output, attention_masks, training);
        }
        self.output_embedding
            .forward(&self.token_embedding.ws, &step_output)
    }
}
