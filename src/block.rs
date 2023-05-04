//! Large layers/blocks of Transformer models they are mostly built from.
//! Each combines layer normalizations, dropouts, the attention itself.
use tch::{nn, Tensor};

use crate::attention::general::{AttentionStrategy, GeneralMultiHeadSelfAttention, SelfMhaConfig};

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
    /// Whether a causal form of attention must be used
    pub causal: bool,
}

#[derive(Debug)]
pub struct SelfAttentingBlock<S>
where
    S: AttentionStrategy,
{
    norm1_layer: nn::LayerNorm,
    norm2_layer: nn::LayerNorm,
    transition1_layer: nn::Linear,
    transition2_layer: nn::Linear,
    attention: GeneralMultiHeadSelfAttention<S>,
    config: TransformerBlockConfig,
}

impl<S> SelfAttentingBlock<S>
where
    S: AttentionStrategy,
    S::Config: From<TransformerBlockConfig> + SelfMhaConfig,
{
    pub fn new(p: nn::Path, config: TransformerBlockConfig) -> Self {
        const TRANSITION_SCALER: usize = 4;
        let attention =
            GeneralMultiHeadSelfAttention::<S>::new(&p / "attention", config.clone().into());
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
        let transition_dim = (TRANSITION_SCALER * attention.config.output_dim()) as i64;
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
            .forward_t(input, attention_mask, train)
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
            .forward_t(&norm1_output, attention_mask, train)
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
