//! This example trains an ALBERT model to recover masked tokens in WikiText-2-raw dataset.
//! The ALBERT architecture is idescribed in paper
//! ["A Lite BERT for Self-supervised Learning of Language Representations"](https://arxiv.org/abs/1909.11942).
//!
//! For the sake of simplicity, this example uses only MLM task, which means it does not bother
//! with sentence order predictions/classification.

use std::io::Write;
use std::path;

use tch::nn::{self, OptimizerConfig};
use tch::Tensor;
use tokenizers::Tokenizer;

use tchrs_transformer::models::albert::Albert;
use tchrs_transformer::training::mlm::{DocCollectionSampler, SampledBatch};
use tchrs_transformer::training::schedulers::CosineLRSchedule;

const MAX_SEQUENCE_LEN: usize = 256;
const BATCH_SIZE: usize = 16;
const LEARNING_RATE: f64 = 5e-5;
const DATASET_FLAVOR: wikitext::WikiTextFlavor = wikitext::WikiTextFlavor::Raw2;
const CACHE_SUBDIR: &str = "cache";

mod tokenization;
mod wikitext;

fn tokens_from_tensor(t: &Tensor) -> Vec<u32> {
    Vec::<i64>::from(t.view([-1]))
        .iter()
        .map(|&i| i as u32)
        .collect()
}

/// Just for fun, runs the model on a piece of masked text to see its behaves during the training.
fn run_model(
    tokenizer: &Tokenizer,
    pad_token_id: u32,
    mask_token_id: u32,
    device: tch::Device,
    model: &Albert,
) {
    let example =
        "The era of the wooden steam ship - of - the - line was brief , because of new , \
         more powerful naval guns . In the 1820s and 1830s , warships began to mount \
         increasingly heavy guns , replacing 18- and 24 - pounder guns with 32 \
         - pounders on sailing ships - of - the - line and introducing 68 - \
         pounders on steamers . Then , the first shell guns firing explosive shells \
         were introduced following their development by the French Général \
         Henri - Joseph Paixhans , and by the 1840s were part of the standard armament \
         for naval powers including the French Navy , Royal Navy , Imperial Russian \
         Navy and United States Navy . It is often held that the power of explosive \
         shells to smash wooden hulls , as demonstrated by the Russian destruction \
         of an Ottoman squadron at the Battle of Sinop , spelled the end of the \
         wooden - hulled warship . The more practical threat to wooden ships was from \
         conventional cannon firing red - hot shot , which could lodge in the hull \
         of a wooden ship and cause a fire or ammunition explosion . \
         Some navies even experimented with hollow shot filled with molten metal \
         for extra incendiary power.";
    let encoded = tokenizer.encode(example, true).unwrap();
    let doc_tokens: Vec<u32> = encoded.get_ids().into();
    let sampler = DocCollectionSampler::new(
        [doc_tokens.len()].iter().copied(),
        pad_token_id,
        mask_token_id,
        tokenizer.get_vocab_size(true) as u32,
        MAX_SEQUENCE_LEN,
        1,
    );
    println!("Test run:\n=========");
    for batch in sampler.epoch_batches(|_| &doc_tokens) {
        let sample_tokens = tokens_from_tensor(&batch.masked_slices);
        println!(
            "Original (masked) document: {:?}",
            tokenizer.decode(sample_tokens, false)
        );
        let batch = batch.to_device(device);
        let model_output = model.forward_t(
            &batch.masked_slices,
            &batch.doc_offsets,
            Some(&batch.padding_masks),
            false,
        );
        let output_tokens = tokens_from_tensor(&model_output.argmax(Some(-1), false));
        println!(
            "Output document: {:?}",
            tokenizer.decode(output_tokens, false)
        );
    }
}

/// This function is a small hack allowing to load stored variables even if some of them
/// are missing from the file. This helps in tinkering with a "live" model.
fn load_var_store<T: AsRef<std::path::Path>>(
    vs: &mut nn::VarStore,
    path: T,
    ignore_missing_tensors: bool,
) -> Result<(), tch::TchError> {
    fn vs_named_tensors<T: AsRef<std::path::Path>>(
        vs: &mut nn::VarStore,
        path: T,
    ) -> Result<std::collections::HashMap<String, Tensor>, tch::TchError> {
        let named_tensors = match path.as_ref().extension().and_then(|x| x.to_str()) {
            Some("bin") | Some("pt") => Tensor::loadz_multi_with_device(&path, vs.device()),
            Some(_) | None => Tensor::load_multi_with_device(&path, vs.device()),
        };
        Ok(named_tensors?.into_iter().collect())
    }

    let named_tensors = vs_named_tensors(vs, &path)?;
    let mut variables = vs.variables_.lock().unwrap();
    for (name, var) in variables.named_variables.iter_mut() {
        match named_tensors.get(name) {
            Some(src) => tch::no_grad(|| var.f_copy_(src).map_err(|e| e.path_context(name)))?,
            None => {
                if !ignore_missing_tensors {
                    return Err(tch::TchError::TensorNameNotFound(
                        name.to_string(),
                        path.as_ref().to_string_lossy().into_owned(),
                    ));
                }
            }
        }
    }
    Ok(())
}

/// Runs the `model` of a `device` through a series of `batches`. If `optimizer` is provided,
/// performs back-propagation step after each batch and prints current loss and perplexity.
/// Returns loss and perplexity averaged over all batches.
fn run_epoch_on_batches(
    batches: impl Iterator<Item = SampledBatch>,
    epoch_size: usize,
    mut optimizer: Option<&mut nn::Optimizer>,
    device: tch::Device,
    model: &Albert,
    tokenizer_vocabulary: usize,
) -> (f32, f32) {
    let mut epoch_total_loss = 0.0;
    let mut epoch_total_repl_perplexity = 0.0;
    let mut epoch_batches = 0.0;
    for (batch_index, batch) in batches.enumerate() {
        let batch = batch.to_device(device);
        let model_output = model.forward_t(
            &batch.masked_slices,
            &batch.doc_offsets,
            Some(&batch.padding_masks),
            optimizer.is_some(),
        );
        // Calculating the loss
        let output_log_softmax = model_output.log_softmax(-1, tch::Kind::Float);
        let per_token_cross_entropy = output_log_softmax
            .reshape(&[-1, tokenizer_vocabulary as i64])
            .g_nll_loss::<Tensor>(
                &batch.doc_slices.reshape(&[-1]),
                None,
                tch::Reduction::None,
                -100,
            )
            .reshape_as(&batch.doc_slices);
        // Loss for masked tokens only
        let loss = per_token_cross_entropy
            .masked_select(&batch.replacement_masks)
            .mean(tch::Kind::Float);
        // Perplexity for masked tokens only
        let repl_masked_perplexity = loss.exp();
        // Backprop
        if let Some(ref mut optimizer) = optimizer {
            optimizer.backward_step(&loss);
        }
        if optimizer.is_some() {
            print!(
                "\rBatch {} of {}. Loss: {}. Perplexity of replacements: {}",
                batch_index + 1,
                epoch_size,
                f32::from(&loss),
                f32::from(&repl_masked_perplexity)
            );
            std::io::stdout().lock().flush().ok();
        }
        epoch_total_loss += f32::from(&loss);
        epoch_total_repl_perplexity += f32::from(&repl_masked_perplexity);
        epoch_batches += 1.0;
    }
    (
        epoch_total_loss / epoch_batches,
        epoch_total_repl_perplexity / epoch_batches,
    )
}

fn main() -> std::io::Result<()> {
    let cache_dir = path::PathBuf::from(CACHE_SUBDIR);
    let model_store_path = cache_dir.join("model.ot");
    let (tokenizer, tokenized_dataset) =
        wikitext::load_wikitext_dataset(&cache_dir, DATASET_FLAVOR, None)?;

    let pad_token_id = tokenizer
        .token_to_id("<pad>")
        .expect("Padding token must be present");
    let mask_token_id = tokenizer
        .token_to_id("<mask>")
        .expect("Masking token must be present");

    let tokenizer_vocabulary = tokenizer.get_vocab_size(true);

    let train_sampler = DocCollectionSampler::new(
        tokenized_dataset.train.iter().map(Vec::len),
        pad_token_id,
        mask_token_id,
        tokenizer_vocabulary as u32,
        MAX_SEQUENCE_LEN,
        BATCH_SIZE,
    );
    let validation_sampler =
        train_sampler.clone_for_docset(tokenized_dataset.validation.iter().map(Vec::len));

    let device = tch::Device::cuda_if_available();
    let mut var_store = nn::VarStore::new(device);
    let mut optimizer = nn::AdamW::default()
        .build(&var_store, LEARNING_RATE)
        .expect("Unable to build optimizer");
    let model = Albert::new(var_store.root() / "albert", Default::default());
    let mut epoch_num_var = var_store
        .root()
        .var("epoch", &[], tch::nn::Init::Const(1.0));
    if model_store_path.exists() {
        load_var_store(&mut var_store, &model_store_path, true).expect("Unable to load model");
    }

    let cosine_schedule = CosineLRSchedule {
        lr_high: LEARNING_RATE,
        lr_low: LEARNING_RATE / 30.0,
        warmup_period: 50,
        ..Default::default()
    };
    // loop {
    let mut min_avg_loss = f32::INFINITY;
    for _ in 1..10000 {
        run_model(&tokenizer, pad_token_id, mask_token_id, device, &model);
        let epoch_num = f64::from(&epoch_num_var) as usize;
        let new_lr = cosine_schedule.get_lr(epoch_num);
        println!("Epoch {}. Setting learning rate to {}", epoch_num, new_lr);
        optimizer.set_lr(new_lr);

        let (epoch_avg_loss, epoch_avg_repl_perplexity) = run_epoch_on_batches(
            train_sampler.epoch_batches(|doc_index| &tokenized_dataset.train[doc_index]),
            train_sampler.epoch_size,
            Some(&mut optimizer),
            device,
            &model,
            tokenizer_vocabulary,
        );
        println!(
            "\nEpoch {}. avg loss: {}, avg masked repl perplexity: {}",
            epoch_num, epoch_avg_loss, epoch_avg_repl_perplexity,
        );

        let (val_epoch_avg_loss, val_epoch_avg_repl_perplexity) = run_epoch_on_batches(
            validation_sampler.epoch_batches(|doc_index| &tokenized_dataset.validation[doc_index]),
            train_sampler.epoch_size,
            None,
            device,
            &model,
            tokenizer_vocabulary,
        );
        println!(
            "Validation set run. avg loss: {}, avg masked repl perplexity: {}",
            val_epoch_avg_loss, val_epoch_avg_repl_perplexity,
        );

        tch::no_grad(|| {
            epoch_num_var += 1.0;
        });
        if min_avg_loss > epoch_avg_loss {
            min_avg_loss = epoch_avg_loss;
            var_store
                .save(&model_store_path)
                .expect("Unable to save the model");
            println!("Model saved in {:?}", model_store_path);
        }
    }
    let tdata: Vec<f32> = (0..30).map(|i| i as f32).collect();
    let t = Tensor::of_slice(&tdata).reshape(&[2, 3, 5]);
    println!("Full Tensor: {}", t);
    let s = t.split(1, 1);
    println!("Split Tensor:\n{}\n\n{}\n\n{}", s[0], s[1], s[2]);
    Ok(())
}
