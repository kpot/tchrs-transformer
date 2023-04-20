//! Instruments for training models using MLM (masked language modelling) task.
use rand::{thread_rng, Rng};
use tch::Tensor;

/// Responsible for creating a MLM version of a tokenized document where about 15% of the tokens
/// are replaced with a special `[mask]` or a random token, as described in papers
/// [ALBERT: A Lite BERT for Self-supervised Learning of Language Representations](https://arxiv.org/abs/1909.11942).
/// [SpanBERT: Improving Pre-training by Representing and Predicting Spans](https://arxiv.org/abs/1907.10529)
/// It is not, however constructing the SOP (sentence order prediction) task,
/// and it does not care about the boundaries of the words, so any span of tokens can be replaced.
pub struct DocumentMasker {
    masked_doc_buffer: Vec<u32>,
    mask_buffer: Vec<bool>,
    pub vocabulary_size: u32,
    pub mask_token_id: u32,
}

impl DocumentMasker {
    pub fn new(vocabulary_size: u32, mask_token_id: u32) -> Self {
        Self {
            vocabulary_size,
            mask_token_id,
            masked_doc_buffer: Default::default(),
            mask_buffer: Default::default(),
        }
    }

    /// Randomly replaces tokens in a tokenized document `doc` according to the rules
    /// described earlier.
    ///
    /// Returns two slices:
    /// 1. A modified `[mask]`ed document.
    /// 2. A `[bool]` slice of the same length with `true` in places where a replacement
    ///    was made.
    pub fn mask_document<'a>(&'a mut self, doc: &[u32]) -> (&'a [u32], &'a [bool]) {
        // Probabilities of selecting a 1-token span, 2-token span etc.
        // Calculated as p(n) = (1 / n) / sum(1 / n, for n in 1..K)
        const SPAN_LEN_PROBS: [f32; 3] = [0.54545455, 0.27272727, 0.18181818];
        self.masked_doc_buffer.clear();
        self.mask_buffer.clear();
        self.masked_doc_buffer.extend_from_slice(doc);
        self.mask_buffer.resize(doc.len(), false);

        let mut rng = rand::thread_rng();
        let mut choose_span_len = move || {
            let rand: f32 = rng.gen();
            for (i, p) in SPAN_LEN_PROBS.iter().enumerate().rev() {
                if rand < *p {
                    return i + 1;
                }
            }
            return 1;
        };

        let mut rng = rand::thread_rng();
        let document_span = rand::distributions::Uniform::new(0, doc.len());
        let vocabulary_span = rand::distributions::Uniform::new(0, self.vocabulary_size);
        let tokens_to_mask_budget = (doc.len() as f64 * 0.15) as usize;
        let mut masked_tokens = 0;
        while masked_tokens < tokens_to_mask_budget {
            let span_len = choose_span_len();
            let span_start = rng.sample(&document_span);
            let span_end = (span_start + span_len).min(doc.len());
            self.mask_buffer[span_start..span_end].fill(true);
            let strategy_threshold: f32 = rng.gen();
            if strategy_threshold < 0.8 {
                // 80% of the time we just replace the span with [mask]
                self.masked_doc_buffer[span_start..span_end].fill(self.mask_token_id);
            } else if strategy_threshold < 0.9 {
                // 10% of the time we replace the tokens with other randomly selected tokens
                self.masked_doc_buffer[span_start..span_end]
                    .fill_with(|| rng.sample(&vocabulary_span));
            } else {
                // and 10% of the time we keep the tokens the same
            }
            masked_tokens += span_end - span_start;
        }
        (&self.masked_doc_buffer, &self.mask_buffer)
    }
}

#[derive(Debug)]
pub struct SampledBatch {
    pub docs: Tensor,
    pub doc_offsets: Tensor,
    pub masked_docs: Tensor,
    pub replacement_masks: Tensor,
    pub padding_masks: Tensor,
}

impl SampledBatch {
    pub fn to_device(&self, device: tch::Device) -> Self {
        Self {
            docs: self.docs.to(device),
            doc_offsets: self.doc_offsets.to(device),
            masked_docs: self.masked_docs.to(device),
            replacement_masks: self.replacement_masks.to(device),
            padding_masks: self.padding_masks.to(device),
        }
    }
}

// Estimates epoch's size that 99% of the dataset appears during the epoch, with
// the smallest document appearing once.
// For instance, if we have a set of documents containing the following number of examples:
// `[2, 11, 5]` (18 total), then, if I we randomly pick 9 examples (18 / 2),
// one of them will likely be from the first (smallest) document. This way
fn estimate_samples_per_epoch(samples_per_doc: &[usize]) -> usize {
    const EPOCH_PERCENTILE: f32 = 0.99;
    let epoch_weight = samples_per_doc.iter().sum::<usize>() as f32 * EPOCH_PERCENTILE;
    let mut sorted_samples_per_doc: Vec<usize> = samples_per_doc.into();
    sorted_samples_per_doc.sort();
    let mut acc_weight = 0.0f32;
    let mut min_doc_size = 1.0f32;
    for &doc_size in sorted_samples_per_doc.iter().rev() {
        acc_weight += doc_size as f32;
        if acc_weight >= epoch_weight {
            min_doc_size = doc_size as f32;
            break;
        }
    }
    let epoch_size = samples_per_doc
        .iter()
        .map(|&doc_size| doc_size as f32 / min_doc_size)
        .sum::<f32>() as usize;
    epoch_size
}

/// Samples slices of documents from a collection, pads them with padding tokens,
/// randomly masks some pieces of texts and merges them into batches ready for training.
/// Epoch size is calculated so that 99% of the collection's content is sampled during each epoch.
pub struct DocCollectionSampler {
    pub samples_per_doc: Vec<usize>,
    pub cumsum_samples_per_doc: Vec<usize>,
    pub max_sequence_len: usize,
    pub batch_size: usize,
    pub epoch_size: usize,
    pub pad_token_id: u32,
    pub mask_token_id: u32,
    pub vocabulary_size: u32,
}

impl DocCollectionSampler {
    pub fn clone_for_docset(&self, doc_sizes: impl Iterator<Item = usize>) -> Self {
        Self::new(
            doc_sizes,
            self.pad_token_id,
            self.mask_token_id,
            self.vocabulary_size,
            self.max_sequence_len,
            self.batch_size,
        )
    }

    pub fn new(
        doc_sizes: impl Iterator<Item = usize>,
        pad_token_id: u32,
        mask_token_id: u32,
        vocabulary_size: u32,
        max_sequence_len: usize,
        batch_size: usize,
    ) -> Self {
        let samples_per_doc: Vec<usize> = doc_sizes
            .map(|doc_size| {
                if doc_size >= max_sequence_len {
                    doc_size - max_sequence_len + 1
                } else {
                    1
                }
            })
            .collect();
        let mut total_samples = 0;
        let cumsum_samples_per_doc: Vec<usize> = samples_per_doc
            .iter()
            .map(|num_samples| {
                total_samples += num_samples;
                total_samples
            })
            .collect();

        let epoch_size = if samples_per_doc.len() > 0 {
            estimate_samples_per_epoch(&samples_per_doc) / batch_size
        } else {
            0
        };
        Self {
            samples_per_doc,
            cumsum_samples_per_doc,
            max_sequence_len,
            batch_size,
            epoch_size,
            pad_token_id,
            vocabulary_size,
            mask_token_id,
        }
    }

    /// Returns an iterator of a single training epoch, which samples batches of texts from
    /// the documents so that each epoch would be covering the entire collection.
    ///
    /// Since the documents can be quite large and often do not fit into the maximum sequence
    /// length, they must be cut into pieces in order to be fed as traning examples.
    /// To ensure that such training examples are not repetitive, each example can contain
    /// any portion of the document.
    pub fn epoch_batches<'a, 'b>(
        &'a self,
        get_tokenized_doc: impl Fn(usize) -> &'b [u32] + 'a,
    ) -> impl Iterator<Item = SampledBatch> + 'a {
        let mut batches_left = self.epoch_size;
        let batch_shape = [self.batch_size as i64, self.max_sequence_len as i64];
        let mut masker = DocumentMasker::new(self.vocabulary_size, self.mask_token_id);
        // These buffers store fixed-sized documents sequentially
        let mut batch_docs: Vec<i64> = Vec::with_capacity(self.batch_size * self.max_sequence_len);
        let mut batch_masked_docs: Vec<i64> =
            Vec::with_capacity(self.batch_size * self.max_sequence_len);
        let mut batch_replacement_masks: Vec<bool> =
            Vec::with_capacity(self.batch_size * self.max_sequence_len);
        let mut batch_padding_masks: Vec<f32> =
            Vec::with_capacity(self.batch_size * self.max_sequence_len);
        let mut batch_doc_offsets: Vec<i64> = Vec::with_capacity(self.batch_size);
        let mut rng = rand::thread_rng();
        std::iter::from_fn(move || {
            if batches_left == 0 {
                return None;
            }
            batch_docs.clear();
            batch_masked_docs.clear();
            batch_doc_offsets.clear();
            batch_replacement_masks.clear();
            batch_padding_masks.clear();
            for batch_num in 1..self.batch_size + 1 {
                let (doc_id, slice_start) = self.randomly_select_example()?;
                let doc = get_tokenized_doc(doc_id);
                // 10% of the time force the slice to be shorter than possible to fill the rest
                // with <pad> tokens.
                let size_strategy_threshold: f32 = rng.gen();
                let max_slice_len = if size_strategy_threshold < 0.1 {
                    let portion: f64 = rng.gen();
                    // the size can be reduced down to 50% of the original
                    (0.5 * (self.max_sequence_len as f64) * (1.0 + portion)) as usize
                } else {
                    self.max_sequence_len
                };
                let doc_slice = &doc[slice_start..(slice_start + max_slice_len).min(doc.len())];
                let (masked_doc_slice, mask_slice) = masker.mask_document(&doc_slice);
                // The size all text/mask buffers must be filled to at the end of this iteration
                let step_sequence_len = batch_num * self.max_sequence_len;
                // filling with buffers
                batch_docs.extend(doc_slice.iter().copied().map(i64::from));
                batch_masked_docs.extend(masked_doc_slice.iter().copied().map(i64::from));
                batch_replacement_masks.extend_from_slice(mask_slice);
                batch_doc_offsets.push(slice_start as i64);
                batch_padding_masks.resize(
                    step_sequence_len - self.max_sequence_len + doc_slice.len(),
                    1.0,
                );
                // padding to self.max_sequence_len
                batch_docs.resize(step_sequence_len, self.pad_token_id as i64);
                batch_masked_docs.resize(step_sequence_len, self.pad_token_id as i64);
                batch_replacement_masks.resize(step_sequence_len, false);
                batch_padding_masks.resize(step_sequence_len, 0.0);
            }
            batches_left -= 1;
            Some(SampledBatch {
                docs: Tensor::of_slice(&batch_docs).reshape(&batch_shape),
                doc_offsets: Tensor::of_slice(&batch_doc_offsets)
                    .reshape(&[self.batch_size as i64]),
                masked_docs: Tensor::of_slice(&batch_masked_docs).reshape(&batch_shape),
                replacement_masks: Tensor::of_slice(&batch_replacement_masks).reshape(&batch_shape),
                padding_masks: Tensor::of_slice(&batch_padding_masks).reshape(&batch_shape),
            })
        })
    }

    /// Randomly selects a document to slice an example from along
    /// with the position of the beginning of the example within the document.
    /// Returns None if the dataset is empty.
    fn randomly_select_example(&self) -> Option<(usize, usize)> {
        let mut rng = thread_rng();
        let max_example_number = self.cumsum_samples_per_doc.last()?;
        // global document location
        let global_example_start = rng.gen_range(0..*max_example_number);
        let document_index = self
            .cumsum_samples_per_doc
            .partition_point(|&x| x <= global_example_start);
        let in_doc_example_start = if document_index > 0 {
            global_example_start - self.cumsum_samples_per_doc[document_index - 1]
        } else {
            global_example_start
        };
        Some((document_index, in_doc_example_start))
    }
}
