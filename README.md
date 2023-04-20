# Transformer library for tch-rs

A collection of tools for building and training Transformers
using [tch-rs](https://github.com/LaurentMazare/tch-rs)
(a Rust bindings for C++ PyTorch API).

The best way to explore it is to start from running an 
[example](./examples/run-albert/main.rs)

Assuming an NVIDIA GPU is available, after cloning this repository,
an example can be lauched with these shell commands:

```shell
export TORCH_CUDA_VERSION=cu118
cargo run --example run-albert --features=examples
```

Once launched, the example will 

* Download the WikiText dataset.
* Parse its training/validation/test documents and train a tokenizer on them.
* Tokenize the documents with the tokenizer.
* Launch training of an [ALBERT](https://arxiv.org/abs/1909.11942) model
  using MLM task.
* Keep the downloaded dataset, the trained tokenizer, and the model's weights
  inside a "cache" subdirectory, to facilitate restarts.
