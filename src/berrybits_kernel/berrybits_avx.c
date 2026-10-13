#include "berrybits_runtime_backend.h"

#if (defined(__x86_64__) || defined(__i386__)) && defined(__AVX__)
#include <immintrin.h>

float berrybits_runtime_q4_backend_avx_dot(const float *a, const float *b, size_t n) {
    if (!a || !b || n == 0) return 0.0f;
    __m256 acc = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        acc = _mm256_add_ps(acc, _mm256_mul_ps(va, vb));
    }
    float out[8];
    _mm256_storeu_ps(out, acc);
    float sum = out[0] + out[1] + out[2] + out[3] + out[4] + out[5] + out[6] + out[7];
    for (; i < n; ++i) sum += a[i] * b[i];
    return sum;
}
#else
float berrybits_runtime_q4_backend_avx_dot(const float *a, const float *b, size_t n) {
    (void)a; (void)b; (void)n;
    return 0.0f;
}
#endif

float berrybits_runtime_q4_backend_avx2_q4_dot(const uint8_t *packed, const float *x,
                                   size_t n, const float *lut,
                                   float scale, float zero) {
#if (defined(__x86_64__) || defined(__i386__)) && defined(__AVX2__)
    if (!packed || !x || !lut || n == 0) return 0.0f;

    float scaled_lut[16];
    for (int i = 0; i < 16; ++i) scaled_lut[i] = lut[i] * scale + zero;

    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        size_t byte = i / 2;
        uint8_t b0 = packed[byte];
        uint8_t b1 = packed[byte + 1];
        uint8_t b2 = packed[byte + 2];
        uint8_t b3 = packed[byte + 3];
        uint8_t b4 = packed[byte + 4];
        uint8_t b5 = packed[byte + 5];
        uint8_t b6 = packed[byte + 6];
        uint8_t b7 = packed[byte + 7];
        __m256i indices0 = _mm256_setr_epi32(
            b0 & 15, b0 >> 4, b1 & 15, b1 >> 4,
            b2 & 15, b2 >> 4, b3 & 15, b3 >> 4);
        __m256i indices1 = _mm256_setr_epi32(
            b4 & 15, b4 >> 4, b5 & 15, b5 >> 4,
            b6 & 15, b6 >> 4, b7 & 15, b7 >> 4);
        __m256 values0 = _mm256_i32gather_ps(scaled_lut, indices0, 4);
        __m256 values1 = _mm256_i32gather_ps(scaled_lut, indices1, 4);
        __m256 input0 = _mm256_loadu_ps(x + i);
        __m256 input1 = _mm256_loadu_ps(x + i + 8);
#if defined(__FMA__)
        acc0 = _mm256_fmadd_ps(values0, input0, acc0);
        acc1 = _mm256_fmadd_ps(values1, input1, acc1);
#else
        acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(values0, input0));
        acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(values1, input1));
#endif
    }

    float lanes[8];
    _mm256_storeu_ps(lanes, _mm256_add_ps(acc0, acc1));
    float sum = lanes[0] + lanes[1] + lanes[2] + lanes[3] +
                lanes[4] + lanes[5] + lanes[6] + lanes[7];
    for (; i < n; ++i) {
        uint8_t byte = packed[i / 2];
        uint8_t index = (i & 1u) ? (byte >> 4) : (byte & 15u);
        sum += scaled_lut[index] * x[i];
    }
    return sum;
#else
    return berrybits_runtime_q4_backend_cpu_q4_dot(packed, x, n, lut, scale, zero);
#endif
}
