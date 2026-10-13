#include "berrybits.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

static int test_mode(berrybits_mode_t mode) {
    enum { rows = 2, cols = 67, count = rows * cols };
    float weights[count];
    float decoded[count];
    float reloaded_values[count];
    float input[cols];
    float output[rows];
    float reloaded_output[rows];
    for (size_t i = 0; i < count; ++i) {
        int centered = (int)((i * 37u + 13u) % 101u) - 50;
        weights[i] = (float)centered / 53.0f;
    }
    for (size_t i = 0; i < cols; ++i) {
        int centered = (int)((i * 19u + 7u) % 61u) - 30;
        input[i] = (float)centered / 31.0f;
    }

    berrybits_tensor_t tensor = {0};
    if (!berrybits_tensor_init(&tensor, rows, cols, mode) ||
        !berrybits_tensor_quantize_f32(&tensor, weights) ||
        !berrybits_tensor_dequantize_f32(&tensor, decoded) ||
        !berrybits_tensor_gemv(&tensor, input, output)) {
        berrybits_tensor_destroy(&tensor);
        return 0;
    }

    for (size_t i = 0; i < count; ++i) {
        if (!isfinite(decoded[i])) {
            berrybits_tensor_destroy(&tensor);
            return 0;
        }
    }
    for (size_t row = 0; row < rows; ++row) {
        double reference = 0.0;
        for (size_t col = 0; col < cols; ++col) {
            reference += (double)decoded[row * cols + col] * input[col];
        }
        if (fabs((double)output[row] - reference) > 2e-4 * (1.0 + fabs(reference))) {
            berrybits_tensor_destroy(&tensor);
            return 0;
        }
    }

    char path[96];
    snprintf(path, sizeof(path), "build/test_mode_%u.blut", (unsigned)mode);
    if (!berrybits_tensor_save_blut(&tensor, path)) {
        berrybits_tensor_destroy(&tensor);
        return 0;
    }
    berrybits_tensor_t loaded = {0};
    int loaded_ok = berrybits_tensor_load_blut(&loaded, path);
    remove(path);
    if (!loaded_ok || loaded.mode != mode || loaded.rows != rows || loaded.cols != cols ||
        !berrybits_tensor_dequantize_f32(&loaded, reloaded_values) ||
        !berrybits_tensor_gemv(&loaded, input, reloaded_output) ||
        memcmp(decoded, reloaded_values, sizeof(decoded)) != 0 ||
        memcmp(output, reloaded_output, sizeof(output)) != 0) {
        berrybits_tensor_destroy(&tensor);
        berrybits_tensor_destroy(&loaded);
        return 0;
    }

    printf("%-10s bits=%u block=%u payload=%zu bytes round-trip passed\n",
           berrybits_mode_name(mode), tensor.bits_per_value, tensor.block_size,
           tensor.payload_bytes);
    berrybits_tensor_destroy(&tensor);
    berrybits_tensor_destroy(&loaded);
    return 1;
}

int main(void) {
    const berrybits_mode_t modes[] = {
        BERRYBITS_MODE_Q64, BERRYBITS_MODE_Q32, BERRYBITS_MODE_Q16,
        BERRYBITS_MODE_Q8, BERRYBITS_MODE_Q6, BERRYBITS_MODE_Q5,
        BERRYBITS_MODE_Q4_K_S, BERRYBITS_MODE_Q4_K_M, BERRYBITS_MODE_Q3,
        BERRYBITS_MODE_Q2, BERRYBITS_MODE_I2_S, BERRYBITS_MODE_Q1,
        BERRYBITS_MODE_TERNARY
    };
    for (size_t i = 0; i < sizeof(modes) / sizeof(modes[0]); ++i) {
        if (!test_mode(modes[i])) {
            fprintf(stderr, "BerryBits mode test failed: %s\n", berrybits_mode_name(modes[i]));
            return 1;
        }
    }

    double double_weights[4] = {0.1, -0.2, 0.3, -0.4};
    float converted[4];
    berrybits_tensor_t q64 = {0};
    if (!berrybits_tensor_init(&q64, 1, 4, BERRYBITS_MODE_Q64) ||
        !berrybits_tensor_quantize_f64(&q64, double_weights) ||
        !berrybits_tensor_dequantize_f32(&q64, converted)) {
        berrybits_tensor_destroy(&q64);
        return 2;
    }
    for (size_t i = 0; i < 4; ++i) {
        if (fabs((double)converted[i] - double_weights[i]) > 1e-7) {
            berrybits_tensor_destroy(&q64);
            return 3;
        }
    }
    berrybits_tensor_destroy(&q64);
    puts("All BerryBits modes passed.");
    return 0;
}
