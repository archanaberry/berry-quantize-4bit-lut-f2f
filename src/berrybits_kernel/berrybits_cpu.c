#include "berrybits_runtime_backend.h"

float berrybits_runtime_q4_backend_cpu_q4_dot(const uint8_t *packed, const float *x,
                                  size_t n, const float *lut,
                                  float scale, float zero) {
    if (!packed || !x || !lut || n == 0) return 0.0f;

    float scaled_lut[16];
    for (size_t i = 0; i < 16; ++i) {
        scaled_lut[i] = lut[i] * scale + zero;
    }

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    size_t i = 0;

    for (; i + 8 <= n; i += 8) {
        uint8_t byte0 = packed[i / 2];
        uint8_t byte1 = packed[i / 2 + 1];
        uint8_t byte2 = packed[i / 2 + 2];
        uint8_t byte3 = packed[i / 2 + 3];
        acc0 += scaled_lut[byte0 & 0x0Fu] * x[i];
        acc1 += scaled_lut[byte0 >> 4] * x[i + 1];
        acc2 += scaled_lut[byte1 & 0x0Fu] * x[i + 2];
        acc3 += scaled_lut[byte1 >> 4] * x[i + 3];
        acc0 += scaled_lut[byte2 & 0x0Fu] * x[i + 4];
        acc1 += scaled_lut[byte2 >> 4] * x[i + 5];
        acc2 += scaled_lut[byte3 & 0x0Fu] * x[i + 6];
        acc3 += scaled_lut[byte3 >> 4] * x[i + 7];
    }

    float tail = 0.0f;
    for (; i < n; ++i) {
        uint8_t byte = packed[i / 2];
        uint8_t index = (i & 1u) ? (byte >> 4) : (byte & 0x0Fu);
        tail += scaled_lut[index] * x[i];
    }

    return (acc0 + acc1) + (acc2 + acc3) + tail;
}
