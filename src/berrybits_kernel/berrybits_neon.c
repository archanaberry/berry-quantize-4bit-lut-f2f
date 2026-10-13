#include "berrybits_runtime_backend.h"

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>

float berrybits_runtime_q4_backend_neon_dot(const float *a, const float *b, size_t n) {
    if (!a || !b || n == 0) return 0.0f;
    float32x4_t acc = vdupq_n_f32(0.0f);
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(&a[i]);
        float32x4_t vb = vld1q_f32(&b[i]);
        acc = vmlaq_f32(acc, va, vb);
    }
    float out[4];
    vst1q_f32(out, acc);
    float sum = out[0] + out[1] + out[2] + out[3];
    for (; i < n; ++i) sum += a[i] * b[i];
    return sum;
}

float berrybits_runtime_q4_backend_neon_q4_dot(const uint8_t *packed, const float *x,
                                   size_t n, const float *lut,
                                   float scale, float zero) {
    if (!packed || !x || !lut || n == 0) return 0.0f;

    float scaled_lut[16];
    for (size_t i = 0; i < 16; ++i) scaled_lut[i] = lut[i] * scale + zero;

    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        size_t byte = i / 2;
        uint8_t b0 = packed[byte];
        uint8_t b1 = packed[byte + 1];
        uint8_t b2 = packed[byte + 2];
        uint8_t b3 = packed[byte + 3];
        float weights0[4] = {
            scaled_lut[b0 & 0x0Fu], scaled_lut[b0 >> 4],
            scaled_lut[b1 & 0x0Fu], scaled_lut[b1 >> 4]
        };
        float weights1[4] = {
            scaled_lut[b2 & 0x0Fu], scaled_lut[b2 >> 4],
            scaled_lut[b3 & 0x0Fu], scaled_lut[b3 >> 4]
        };
        acc0 = vmlaq_f32(acc0, vld1q_f32(weights0), vld1q_f32(x + i));
        acc1 = vmlaq_f32(acc1, vld1q_f32(weights1), vld1q_f32(x + i + 4));
    }

    float lanes0[4];
    float lanes1[4];
    vst1q_f32(lanes0, acc0);
    vst1q_f32(lanes1, acc1);
    float sum = lanes0[0] + lanes0[1] + lanes0[2] + lanes0[3] +
                lanes1[0] + lanes1[1] + lanes1[2] + lanes1[3];
    for (; i < n; ++i) {
        uint8_t byte = packed[i / 2];
        uint8_t index = (i & 1u) ? (byte >> 4) : (byte & 0x0Fu);
        sum += scaled_lut[index] * x[i];
    }
    return sum;
}
#else
float berrybits_runtime_q4_backend_neon_dot(const float *a, const float *b, size_t n) {
    (void)a; (void)b; (void)n;
    return 0.0f;
}

float berrybits_runtime_q4_backend_neon_q4_dot(const uint8_t *packed, const float *x,
                                   size_t n, const float *lut,
                                   float scale, float zero) {
    return berrybits_runtime_q4_backend_cpu_q4_dot(packed, x, n, lut, scale, zero);
}
#endif
