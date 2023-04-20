use std::io::{self, BufRead, Write};
use std::path;
use tokenizers::Tokenizer;

use crate::tokenization::{
    encode_and_cache_texts, load_encoded_docs, train_tokenizer, TokenizedDocs,
};
use tchrs_transformer::utils::download;

const VOCABULARY_SIZE: usize = 16384;

/// A very abstract dataset of something, containing sets for training, testing and validation.
/// For example, each set can be a vector of texts.
pub struct DataSet<T> {
    pub train: T,
    pub test: T,
    pub validation: T,
}

impl<T> DataSet<T> {
    pub fn try_map<U, F, E, N, K>(&self, field_keys: K, f: F) -> Result<DataSet<U>, E>
    where
        K: Into<DataSet<N>>,
        F: Fn(&T, N) -> Result<U, E>,
    {
        let keys = field_keys.into();
        Ok(DataSet::<U> {
            train: f(&self.train, keys.train)?,
            test: f(&self.test, keys.test)?,
            validation: f(&self.validation, keys.validation)?,
        })
    }

    pub fn try_from_fn<N, K, F, E>(field_keys: K, mut f: F) -> Result<Self, E>
    where
        K: Into<DataSet<N>>,
        F: FnMut(N) -> Result<T, E>,
    {
        let keys = field_keys.into();
        Ok(Self {
            train: f(keys.train)?,
            test: f(keys.test)?,
            validation: f(keys.validation)?,
        })
    }
}

impl<T> From<&[T; 3]> for DataSet<T>
where
    T: Copy,
{
    fn from(value: &[T; 3]) -> Self {
        Self {
            train: value[0],
            test: value[1],
            validation: value[2],
        }
    }
}

pub fn open_zip_archive(
    zip_file_path: impl AsRef<path::Path>,
) -> io::Result<zip::ZipArchive<std::fs::File>> {
    let zip_file = std::fs::File::open(zip_file_path.as_ref())?;
    Ok(zip::ZipArchive::new(zip_file)?)
}

fn wiki_preprocessor() -> impl Fn(&str) -> String {
    let wiki_special = regex::Regex::new(r"\s@(.)@\s").expect("A valid regex");
    move |input: &str| -> String {
        let result = wiki_special.replace_all(input, "$1");
        result.into_owned()
    }
}

fn read_wikitext_documents<'a>(
    archive: &'a mut zip::ZipArchive<impl std::io::Read + std::io::Seek>,
    nested_file_name: &str,
) -> io::Result<impl Iterator<Item = String> + 'a> {
    let nested_file = archive.by_name(nested_file_name)?;
    let mut buf_nested_reader = std::io::BufReader::new(nested_file);
    let mut line_buffer = String::new();

    let text_processor = wiki_preprocessor();
    let header_regex = regex::Regex::new(r"^\s=\s[^=]+?\s=\s+$").expect("A valid regex");

    Ok(std::iter::from_fn(move || -> Option<String> {
        let mut document = String::new();
        loop {
            if line_buffer.is_empty() {
                let file_exhausted = match buf_nested_reader.read_line(&mut line_buffer) {
                    Ok(bytes_read) if bytes_read > 0 => false,
                    _ => true,
                };
                if file_exhausted {
                    if document.is_empty() {
                        return None;
                    } else {
                        return Some(document);
                    }
                }
            }

            if header_regex.is_match(&line_buffer) && !document.is_empty() {
                // Next article's header is found, time to dump the document,
                // if any, while preserving the buffered line
                return Some(document);
            }

            let line = text_processor(&line_buffer);
            document.push_str(&line);
            // To allow reading of the next line
            line_buffer.clear();
        }
    }))
}

pub enum WikiTextFlavor {
    Raw2,
    Raw103,
}

/// Downloads WikiText dataset and parses it into a set of String documents.
/// This is done only when the dataset is not found in the cache directory.
fn load_wikitext_string_docs(
    cache_dir: impl AsRef<path::Path>,
    flavor_code: &WikiTextFlavor,
) -> io::Result<DataSet<Vec<String>>> {
    let (flavor, hash) = match flavor_code {
        WikiTextFlavor::Raw2 => ("wikitext-2-raw", "f407a2d53283fc4a49bcff21bc5f3770"),
        WikiTextFlavor::Raw103 => ("wikitext-103-raw", "0ca3512bd7a238be4a63ce7b434f8935"),
    };
    let wikitext_archive_path = cache_dir.as_ref().join(format!("{}.zip", flavor));
    download(
        format!(
            "https://s3.amazonaws.com/research.metamind.io/wikitext/{}-v1.zip",
            flavor,
        ),
        hash,
        &wikitext_archive_path,
    )?;
    println!("Reading raw documents from wikitext archive...");
    let mut wiki_archive = open_zip_archive(&wikitext_archive_path)?;
    DataSet::<Vec<String>>::try_from_fn(
        &["wiki.train.raw", "wiki.test.raw", "wiki.valid.raw"],
        |name| {
            read_wikitext_documents(&mut wiki_archive, &format!("{flavor}/{name}"))
                .map(Iterator::collect::<Vec<String>>)
        },
    )
}

/// Downloads WikiText dataset, trains a tokenizer on it, and then uses this tokenizer
/// to encode all the documents from the datasets, turning each of them into a vector
/// of tokens IDs.
/// Caches everything along the way to avoid doing the same work if run again.
pub fn load_wikitext_dataset(
    cache_dir: &path::Path,
    flavor_code: WikiTextFlavor,
    custom_tokenizer: Option<Tokenizer>,
) -> Result<(Tokenizer, DataSet<TokenizedDocs>), io::Error> {
    std::fs::create_dir_all(&cache_dir)?;
    let text_documents_cell = once_cell::unsync::OnceCell::new();
    let load_dataset = || load_wikitext_string_docs(&cache_dir, &flavor_code);
    let tokenizer_path = cache_dir.join("tokenizer.json");
    let tokenizer = match custom_tokenizer {
        Some(tokenizer) => tokenizer,
        None => if tokenizer_path.exists() {
            println!("Loading cached tokenizer...");
            io::stdout().flush().ok();
            Tokenizer::from_file(tokenizer_path)
        } else {
            println!("Training cached tokenizer...");
            io::stdout().flush().ok();
            let text_documents = text_documents_cell.get_or_try_init(&load_dataset)?;
            train_tokenizer(
                VOCABULARY_SIZE,
                tokenizer_path,
                text_documents
                    .train
                    .iter()
                    .chain(text_documents.test.iter())
                    .chain(text_documents.validation.iter()),
            )
        }
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?,
    };
    const TOKENIZED_DOCS_CACHES: [&str; 3] = [
        "train_tokenized.msgpack",
        "test_tokenized.msgpack",
        "valid_tokenized.msgpack",
    ];
    let tokenized_dataset = DataSet::<TokenizedDocs>::try_from_fn(&TOKENIZED_DOCS_CACHES, |name| {
        load_encoded_docs(cache_dir.join(name))
    })
    .or_else(|_| {
        let text_documents = text_documents_cell.get_or_try_init(&load_dataset)?;
        text_documents.try_map(&TOKENIZED_DOCS_CACHES, |docs, cache_name| {
            let cache_path = cache_dir.join(cache_name);
            encode_and_cache_texts(&tokenizer, docs, cache_path)
        })
    })?;
    Ok((tokenizer, tokenized_dataset))
}
