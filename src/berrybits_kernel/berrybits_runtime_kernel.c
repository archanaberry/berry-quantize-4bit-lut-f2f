#include "berrybits_runtime_kernel.h"

#include <stddef.h>
#include <stdint.h>

#if defined(__GNUC__) || defined(__clang__)
#define BERRYBITS_RUNTIME_Q4_PREFETCH(ptr) __builtin_prefetch((ptr), 0, 3)
#else
#define BERRYBITS_RUNTIME_Q4_PREFETCH(ptr) ((void)(ptr))
#endif

static const float berrybits_runtime_q4_default_lut[16] = {
    -1.0000f, -0.8750f, -0.7500f, -0.6250f,
    -0.5000f, -0.3750f, -0.2500f, -0.1250f,
     0.0000f,  0.1250f,  0.2500f,  0.3750f,
     0.5000f,  0.6250f,  0.7500f,  1.0000f
};

void berrybits_runtime_q4_kernel_prefetch(const void *ptr) {
    BERRYBITS_RUNTIME_Q4_PREFETCH(ptr);
}

void berrybits_runtime_q4_fused_gemv(const uint8_t *weights, const float *in, float *out, size_t rows, size_t cols, float scale, float zero, const float *lut) {
    if (!weights || !in || !out || rows == 0 || cols == 0) return;

    const float *active_lut = lut ? lut : berrybits_runtime_q4_default_lut;
    size_t packed_cols = (cols + 1u) / 2u;

    for (size_t r = 0; r < rows; ++r) {
        float acc = 0.0f;
        const uint8_t *row = weights + (r * packed_cols);

        for (size_t c = 0; c < cols; ++c) {
            size_t byte_pos = c / 2u;
            uint8_t raw = row[byte_pos];
            uint8_t idx = (c & 1u) ? (raw >> 4) : (raw & 0x0Fu);
            float w = active_lut[idx] * scale + zero;
            acc += w * in[c];
            BERRYBITS_RUNTIME_Q4_PREFETCH(&row[byte_pos + 1]);
        }

        out[r] = acc;
    }
}
