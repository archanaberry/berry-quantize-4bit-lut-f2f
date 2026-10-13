#include "berrybits_cuda.h"

#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

struct berrybits_cuda_state {
    uint32_t rows;
    uint32_t cols;
    uint32_t blocks_per_row;
    uint8_t *packed_weights;
    float *scales;
    float *lut;
    float *input;
    float *output;
};

__global__ static void berrybits_cuda_q4_gemv_kernel(
    const uint8_t *packed_weights,
    const float *scales,
    const float *lut,
    const float *input,
    float *output,
    uint32_t rows,
    uint32_t cols,
    uint32_t blocks_per_row) {
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;

    size_t packed_row_bytes = ((size_t)cols + 1u) / 2u;
    const uint8_t *packed = packed_weights + (size_t)row * packed_row_bytes;
    const float *row_scales = scales + (size_t)row * blocks_per_row;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;

    for (uint32_t column = 0; column < cols; ++column) {
        uint8_t byte = packed[column / 2u];
        uint8_t index = (column & 1u) ? (byte >> 4) : (byte & 0x0Fu);
        float weight = lut[index] * row_scales[column / BERRYBITS_Q4_BLOCK_SIZE];
        float product = weight * input[column];
        switch (column & 3u) {
            case 0: acc0 += product; break;
            case 1: acc1 += product; break;
            case 2: acc2 += product; break;
            default: acc3 += product; break;
        }
    }
    output[row] = (acc0 + acc1) + (acc2 + acc3);
}

static void berrybits_cuda_release_state(berrybits_cuda_state *state) {
    if (!state) return;
    if (state->packed_weights) cudaFree(state->packed_weights);
    if (state->scales) cudaFree(state->scales);
    if (state->lut) cudaFree(state->lut);
    if (state->input) cudaFree(state->input);
    if (state->output) cudaFree(state->output);
    free(state);
}

extern "C" int berrybits_cuda_available(void) {
    int device_count = 0;
    return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
}

extern "C" int berrybits_cuda_init(berrybits_cuda_context_t *context,
                                    const berrybits_q4_model_t *model) {
    if (!context || !model || !model->packed_weights || !model->scales ||
        !berrybits_cuda_available()) return 0;
    context->state = nullptr;

    berrybits_cuda_state *state = (berrybits_cuda_state *)calloc(1, sizeof(*state));
    if (!state) return 0;
    state->rows = model->rows;
    state->cols = model->cols;
    state->blocks_per_row = model->blocks_per_row;

    size_t packed_row_bytes = ((size_t)model->cols + 1u) / 2u;
    size_t packed_bytes = (size_t)model->rows * packed_row_bytes;
    size_t scale_count = (size_t)model->rows * model->blocks_per_row;
    cudaError_t status = cudaMalloc((void **)&state->packed_weights, packed_bytes);
    if (status == cudaSuccess) status = cudaMalloc((void **)&state->scales, scale_count * sizeof(float));
    if (status == cudaSuccess) status = cudaMalloc((void **)&state->lut, sizeof(model->lut));
    if (status == cudaSuccess) status = cudaMalloc((void **)&state->input, (size_t)model->cols * sizeof(float));
    if (status == cudaSuccess) status = cudaMalloc((void **)&state->output, (size_t)model->rows * sizeof(float));
    if (status == cudaSuccess) status = cudaMemcpy(state->packed_weights, model->packed_weights,
                                                    packed_bytes, cudaMemcpyHostToDevice);
    if (status == cudaSuccess) status = cudaMemcpy(state->scales, model->scales,
                                                    scale_count * sizeof(float), cudaMemcpyHostToDevice);
    if (status == cudaSuccess) status = cudaMemcpy(state->lut, model->lut,
                                                    sizeof(model->lut), cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
        berrybits_cuda_release_state(state);
        return 0;
    }
    context->state = state;
    return 1;
}

extern "C" int berrybits_cuda_gemv(berrybits_cuda_context_t *context,
                                    const float *input, float *output) {
    if (!context || !context->state || !input || !output) return 0;
    berrybits_cuda_state *state = (berrybits_cuda_state *)context->state;
    cudaError_t status = cudaMemcpy(state->input, input,
                                    (size_t)state->cols * sizeof(float),
                                    cudaMemcpyHostToDevice);
    if (status != cudaSuccess) return 0;

    const uint32_t threads = 128;
    const uint32_t blocks = (state->rows + threads - 1u) / threads;
    berrybits_cuda_q4_gemv_kernel<<<blocks, threads>>>(
        state->packed_weights, state->scales, state->lut, state->input,
        state->output, state->rows, state->cols, state->blocks_per_row);
    status = cudaGetLastError();
    if (status == cudaSuccess) {
        status = cudaMemcpy(output, state->output,
                            (size_t)state->rows * sizeof(float),
                            cudaMemcpyDeviceToHost);
    }
    return status == cudaSuccess;
}

extern "C" void berrybits_cuda_destroy(berrybits_cuda_context_t *context) {
    if (!context) return;
    berrybits_cuda_release_state((berrybits_cuda_state *)context->state);
    context->state = nullptr;
}
