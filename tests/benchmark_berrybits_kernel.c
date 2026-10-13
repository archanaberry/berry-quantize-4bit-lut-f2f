#define _POSIX_C_SOURCE 200809L

#include "../src/berrybits_kernel/berrybits_q4_model.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static const float codebook[16] = {
    -1.0000000f, -0.8750000f, -0.7500000f, -0.6250000f,
    -0.5000000f, -0.3750000f, -0.2500000f, -0.1250000f,
     0.0000000f,  0.1250000f,  0.2500000f,  0.3750000f,
     0.5000000f,  0.6250000f,  0.7500000f,  1.0000000f
};

static double seconds_now(void) {
    struct timespec value;
    clock_gettime(CLOCK_MONOTONIC, &value);
    return (double)value.tv_sec + (double)value.tv_nsec * 1e-9;
}

static void reference_quantize(const float *weights, size_t rows, size_t cols,
                               float *scales, uint8_t *packed) {
    size_t blocks_per_row = (cols + BERRYBITS_Q4_BLOCK_SIZE - 1u) /
                            BERRYBITS_Q4_BLOCK_SIZE;
    size_t packed_row_bytes = (cols + 1u) / 2u;
    memset(packed, 0, rows * packed_row_bytes);
    for (size_t row = 0; row < rows; ++row) {
        for (size_t block = 0; block < blocks_per_row; ++block) {
            size_t start = block * BERRYBITS_Q4_BLOCK_SIZE;
            size_t count = cols - start;
            if (count > BERRYBITS_Q4_BLOCK_SIZE) count = BERRYBITS_Q4_BLOCK_SIZE;
            float max_abs = 0.0f;
            for (size_t i = 0; i < count; ++i) {
                float magnitude = fabsf(weights[row * cols + start + i]);
                if (magnitude > max_abs) max_abs = magnitude;
            }
            float scale = max_abs > 0.0f ? max_abs : 1.0f;
            scales[row * blocks_per_row + block] = scale;
            for (size_t i = 0; i < count; ++i) {
                float normalized = weights[row * cols + start + i] / scale;
                size_t best = 0;
                float best_error = INFINITY;
                for (size_t index = 0; index < BERRYBITS_Q4_LUT_SIZE; ++index) {
                    float difference = normalized - codebook[index];
                    float error = difference * difference;
                    if (error < best_error) {
                        best_error = error;
                        best = index;
                    }
                }
                size_t column = start + i;
                size_t packed_index = row * packed_row_bytes + column / 2u;
                if ((column & 1u) == 0) packed[packed_index] = (uint8_t)best;
                else packed[packed_index] |= (uint8_t)(best << 4);
            }
        }
    }
}

static void reference_gemv(const berrybits_q4_model_t *model,
                           const float *input, float *output) {
    size_t packed_row_bytes = (model->cols + 1u) / 2u;
    for (size_t row = 0; row < model->rows; ++row) {
        double sum = 0.0;
        const uint8_t *packed = model->packed_weights + row * packed_row_bytes;
        for (size_t column = 0; column < model->cols; ++column) {
            uint8_t byte = packed[column / 2u];
            uint8_t index = (column & 1u) ? byte >> 4 : byte & 15u;
            size_t block = column / BERRYBITS_Q4_BLOCK_SIZE;
            float weight = codebook[index] *
                           model->scales[row * model->blocks_per_row + block];
            sum += (double)weight * input[column];
        }
        output[row] = (float)sum;
    }
}

int main(void) {
    const uint32_t rows = 128;
    const uint32_t cols = 1024;
    const size_t weight_count = (size_t)rows * cols;
    const size_t packed_count = (size_t)rows * ((cols + 1u) / 2u);
    const size_t blocks_per_row = (cols + BERRYBITS_Q4_BLOCK_SIZE - 1u) /
                                  BERRYBITS_Q4_BLOCK_SIZE;
    const size_t scale_count = rows * blocks_per_row;
    float *weights = (float *)malloc(weight_count * sizeof(float));
    float *input = (float *)malloc(cols * sizeof(float));
    float *reference_scales = (float *)malloc(scale_count * sizeof(float));
    uint8_t *reference_packed = (uint8_t *)calloc(packed_count, sizeof(uint8_t));
    float *output = (float *)malloc(rows * sizeof(float));
    float *reference_output = (float *)malloc(rows * sizeof(float));
    berrybits_q4_model_t model = {0};
    if (!weights || !input || !reference_scales || !reference_packed || !output ||
        !reference_output || !berrybits_q4_model_init(&model, rows, cols)) {
        fprintf(stderr, "benchmark allocation failed\n");
        return 1;
    }

    for (size_t i = 0; i < weight_count; ++i) {
        weights[i] = (float)((int)((i * 37u) % 2001u) - 1000) / 1000.0f;
    }
    for (size_t i = 0; i < cols; ++i) {
        input[i] = (float)((int)((i * 19u) % 101u) - 50) / 50.0f;
    }

    const int quantize_repeats = 8;
    double start = seconds_now();
    for (int repeat = 0; repeat < quantize_repeats; ++repeat) {
        if (!berrybits_q4_model_quantize(&model, weights)) return 2;
    }
    double optimized_quantize_seconds = seconds_now() - start;

    start = seconds_now();
    for (int repeat = 0; repeat < quantize_repeats; ++repeat) {
        reference_quantize(weights, rows, cols, reference_scales, reference_packed);
    }
    double reference_quantize_seconds = seconds_now() - start;

    reference_gemv(&model, input, reference_output);
    const int gemv_repeats = 1200;
    start = seconds_now();
    for (int repeat = 0; repeat < gemv_repeats; ++repeat) {
        berrybits_q4_model_gemv(&model, input, output);
    }
    double optimized_gemv_seconds = seconds_now() - start;

    start = seconds_now();
    for (int repeat = 0; repeat < gemv_repeats; ++repeat) {
        reference_gemv(&model, input, reference_output);
    }
    double reference_gemv_seconds = seconds_now() - start;

    float max_error = 0.0f;
    for (size_t row = 0; row < rows; ++row) {
        float error = fabsf(output[row] - reference_output[row]);
        if (error > max_error) max_error = error;
    }

    printf("host CPU benchmark, matrix=%ux%u, Q4 block=%u\n", rows, cols,
           BERRYBITS_Q4_BLOCK_SIZE);
    printf("quantize: optimized %.4fs, 16-code reference %.4fs, speedup %.2fx\n",
           optimized_quantize_seconds, reference_quantize_seconds,
           reference_quantize_seconds / optimized_quantize_seconds);
    printf("GEMV: block-scaled LUT %.4fs, per-weight-scale reference %.4fs, speedup %.2fx\n",
           optimized_gemv_seconds, reference_gemv_seconds,
           reference_gemv_seconds / optimized_gemv_seconds);
    printf("GEMV max absolute error: %.8g\n", max_error);

    berrybits_q4_model_destroy(&model);
    free(weights);
    free(input);
    free(reference_scales);
    free(reference_packed);
    free(output);
    free(reference_output);
    return max_error <= 2e-4f ? 0 : 3;
}
