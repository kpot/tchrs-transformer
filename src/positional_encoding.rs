/// Constant coordinate encoding, described in paper "Univerasl Transformers"
/// (https://arxiv.org/pdf/1807.03819.pdf), section 2.1.
/// It's a more generalized version of positional encoding proposed
/// in "Attention is all you need", in that it combines information about
/// each token's location with the "time step" of the encoder itself,
/// which is "the floor number" of a stacked attention block among
/// its siblings. Thus each block gets its distinct "flavor" of positional
/// information added to its input.
///
/// Returns positional vector `Pt` of the encoding that should be added
/// to the input before each Transformer block `t` (one of `T` number of blocks).
/// It has shape `[batch_size, sequence_lenght, input_dim]` and constructed
/// by the formula below:
///
///     P[i, k] = sin(i / 10000**(2*j/d)) + sin(t / 10000^(2*j/d)), if k = 2*j
///     P[i, k] = cos(i / 10000**(2*j/d)) + cos(t / 10000^(2*j/d)), if k = 2*j + 1
///
/// where `1 <= i <= sequence_length`, `1 <= k <= input_dim` and `1 <= t <= T`.
/// First "batch" dimension is omitted.
///
/// Parameter `index_offset` provides sequence offsets for each example from the
/// batch, since those examples are often sliced from larger pieces of text.
pub fn transformer_coordinate_encoding(
    input_offsets: &tch::Tensor,
    dim: usize,
    seq_len: usize,
    time_step: usize,
    dtype: tch::Kind,
) -> tch::Tensor {
    assert!(time_step > 0, "Time step cannot be smaller than 1");
    let batch_size = input_offsets
        .size1()
        .expect("Input offsets must be a 2D tensor");
    let device = input_offsets.device();
    let half_dim = dim as i64 / 2;
    // indices for even and odd parts of the resulting vectors
    let even_ids = tch::Tensor::arange_start(1, half_dim + 1, (dtype, device));
    let odd_ids = tch::Tensor::arange_start(1, dim as i64 - half_dim + 1, (dtype, device));
    // calculating 10000^(2*j/d) parts for even and odd resulting indices
    let fdim = dim as f64;
    let t1000 = tch::Tensor::scalar_tensor(1000.0, (dtype, device));
    let even_denominator = t1000.pow(&(2.0 * even_ids / fdim));
    let odd_denominator = t1000.pow(&(2.0 * odd_ids / fdim));
    // matrix of each of sequence position numbers, given the offsets
    let seq_steps = tch::Tensor::arange_start(1, (seq_len + 1) as i64, (dtype, device))
        .view([1, -1])
        .tile(&[batch_size, 1])
        + input_offsets.view([-1, 1]);
    // independent of sequence position time components, even and odd
    let even_time_component = (time_step as f64 / &even_denominator).sin();
    let odd_time_component = (time_step as f64 / &odd_denominator).cos();
    // sequence position-linked components, even and odd
    let even_pos_component =
        (seq_steps.view([batch_size, seq_len as i64, 1]) / even_denominator.view([1, 1, -1])).sin();
    let odd_pos_component =
        (seq_steps.view([batch_size, seq_len as i64, 1]) / odd_denominator.view([1, 1, -1])).cos();
    let even_added = even_pos_component + even_time_component.view([1, 1, -1]);
    let odd_added = odd_pos_component + odd_time_component.view([1, 1, -1]);
    // At this point we have encodings for even and odd dimensional
    // indices, and we need to interweave them. Unfortunately, if dim is odd
    // (dim % 2 != 0), we cannot easily do that. So we divide the odd
    // values into 2 pieces: one of the same size as the odd values
    // and a "leftover" vector (will be empty if dim is divisible by 2).
    let chunks = odd_added.split_sizes(&[half_dim, dim as i64 - 2 * half_dim], -1);
    let (odd_added_bulk, odd_added_leftover) = (&chunks[0], &chunks[1]);
    // interleaving even-indexed values with all (but one) odd-indexed values
    let bulk_combined = tch::Tensor::stack(&[odd_added_bulk, &even_added], -1).reshape(&[
        batch_size,
        seq_len as i64,
        -1,
    ]);
    // concatenating with the "leftover" odd-indexed values to form the final vector
    // the shape of (batch_size, sequence_len, dim)
    tch::Tensor::cat(&[&bulk_combined, odd_added_leftover], -1)
}
