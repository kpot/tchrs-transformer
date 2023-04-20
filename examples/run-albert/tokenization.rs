use std::{fs, io, path};

use rayon::iter::{ParallelExtend, ParallelIterator};
use rayon::prelude::IntoParallelRefIterator;
use tokenizers::models::bpe::{BpeTrainerBuilder, BPE};
use tokenizers::normalizers::{Sequence, Strip, NFC};
use tokenizers::processors::byte_level::ByteLevel;
use tokenizers::{AddedToken, NormalizerWrapper, Tokenizer, TokenizerBuilder};

/// A bunch of documents that were converted into lists of token IDs by some tokenizer
pub type TokenizedDocs = Vec<Vec<u32>>;

pub fn train_tokenizer(
    vocab_size: usize,
    save_as: impl AsRef<path::Path> + Send,
    texts: impl Iterator<Item = impl AsRef<str> + Send> + Send,
) -> tokenizers::Result<Tokenizer> {
    let mut trainer = BpeTrainerBuilder::new()
        .show_progress(true)
        .vocab_size(vocab_size)
        .min_frequency(0)
        .special_tokens(vec![
            AddedToken::from(String::from("<pad>"), true),
            AddedToken::from(String::from("<mask>"), true),
            AddedToken::from(String::from("<unk>"), true),
        ])
        .build();

    let mut tokenizer = TokenizerBuilder::new()
        .with_model(BPE::default())
        .with_normalizer(Some(Sequence::new(vec![
            NormalizerWrapper::StripNormalizer(Strip::new(true, true)),
            NormalizerWrapper::NFC(NFC),
        ])))
        .with_pre_tokenizer(Some(ByteLevel::default()))
        .with_post_processor(Some(ByteLevel::default()))
        .with_decoder(Some(ByteLevel::default()))
        .build()?;
    tokenizer.train(&mut trainer, texts)?.save(save_as, false)?;
    Ok(tokenizer.into())
}

pub fn encode_documents_to_ids(
    tokenizer: &Tokenizer,
    docs: impl ParallelIterator<Item = impl AsRef<str>>,
) -> TokenizedDocs {
    let mut result: TokenizedDocs = match docs.opt_len() {
        Some(len) => Vec::with_capacity(len),
        None => Vec::new(),
    };
    result.par_extend(docs.filter_map(|doc| {
        let text = doc.as_ref();
        match tokenizer.encode(text, false) {
            Ok(encoding) => {
                let token_ids: Vec<u32> = encoding.get_ids().into();
                if token_ids.len() > 0 {
                    Some(token_ids)
                } else {
                    None
                }
            }
            Err(..) => None,
        }
    }));
    result
}

pub fn load_encoded_docs(path: impl AsRef<path::Path>) -> io::Result<TokenizedDocs> {
    fs::File::open(path.as_ref()).and_then(|cache_file| {
        println!("Loading tokenized docs from {:?}...", path.as_ref());
        rmp_serde::from_read::<fs::File, TokenizedDocs>(cache_file)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    })
}

pub fn encode_and_cache_texts(
    tokenizer: &Tokenizer,
    text_documents: &[String],
    path: impl AsRef<path::Path>,
) -> io::Result<TokenizedDocs> {
    println!(
        "Tokenizing {} documents for {:?}...",
        text_documents.len(),
        path.as_ref()
    );
    let result = encode_documents_to_ids(&tokenizer, text_documents.par_iter());
    println!("Caching tokenized docs into {:?}...", path.as_ref());
    let mut cache_file = fs::File::create(path.as_ref())?;
    rmp_serde::encode::write(&mut cache_file, &result).map_err(|e| {
        io::Error::new(
            io::ErrorKind::Other,
            format!("Unable to create cache file {:?}: {:?}", path.as_ref(), e),
        )
    })?;
    Ok(result)
}
