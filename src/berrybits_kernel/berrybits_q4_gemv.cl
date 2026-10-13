__kernel void berrybits_q4_gemv(
    __global const uchar *packed_weights,
    __global const float *scales,
    __global const float *lut,
    __global const float *input,
    __global float *output,
    const uint rows,
    const uint cols,
    const uint blocks_per_row) {
    const uint row = get_global_id(0);
    if (row >= rows) return;

    const uint packed_row_bytes = (cols + 1u) / 2u;
    const uint row_packed_offset = row * packed_row_bytes;
    const uint row_scale_offset = row * blocks_per_row;
    float accumulators[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (uint block = 0; block < blocks_per_row; ++block) {
        const uint block_start = block * 32u;
        const uint block_end = min(block_start + 32u, cols);
        const float scale = scales[row_scale_offset + block];
        float scaled_lut[16];
        for (uint i = 0; i < 16u; ++i) scaled_lut[i] = lut[i] * scale;

        for (uint column = block_start; column < block_end; ++column) {
            const uchar packed = packed_weights[row_packed_offset + column / 2u];
            const uint code = (column & 1u) ? (packed >> 4) : (packed & 15u);
            accumulators[column & 3u] += scaled_lut[code] * input[column];
        }
    }

    output[row] = (accumulators[0] + accumulators[1]) +
                  (accumulators[2] + accumulators[3]);
}
