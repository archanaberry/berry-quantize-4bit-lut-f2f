#include "berrybits_runtime_backend.h"

#include <math.h>
#include <stdlib.h>

void berrybits_runtime_q4_optimizer_calibrate(float *teacher, float *student, size_t n, float scale, float zero) {
    if (!teacher || !student || n == 0) return;
    float max_abs = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float a = fabsf(teacher[i]);
        if (a > max_abs) max_abs = a;
    }

    float adaptive_scale = max_abs > 0.0f ? max_abs : (scale > 0.0f ? scale : 1.0f);
    float adaptive_zero = (zero == 0.0f) ? 0.0f : zero;

    for (size_t i = 0; i < n; ++i) {
        float v = teacher[i];
        float q = adaptive_scale > 0.0f ? (v / adaptive_scale) : 0.0f;
        if (q < -1.0f) q = -1.0f;
        if (q > 1.0f) q = 1.0f;
        float clipped = q * adaptive_scale + adaptive_zero;
        student[i] = v * 0.82f + clipped * 0.18f;
    }
}

float berrybits_runtime_q4_optimizer_teacher_loss(const float *teacher, const float *student, size_t n) {
    if (!teacher || !student || n == 0) return 0.0f;
    double acc = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double d = (double)teacher[i] - (double)student[i];
        acc += d * d;
    }
    return (float)sqrt(acc / (double)n);
}

void berrybits_runtime_q4_optimizer_model_plan(int backend_id, int *threads, int *tile_k, int *tile_m, int *tile_n) {
    if (!threads || !tile_k || !tile_m || !tile_n) return;

    *threads = 1;
    *tile_k = 32;
    *tile_m = 64;
    *tile_n = 128;

    switch (backend_id) {
        case 1: /* NEON */
            *threads = 4;
            *tile_k = 16;
            *tile_m = 32;
            *tile_n = 64;
            break;
        case 2: /* AVX2 */
            *threads = 8;
            *tile_k = 32;
            *tile_m = 64;
            *tile_n = 128;
            break;
        case 3: /* HVX */
            *threads = 8;
            *tile_k = 64;
            *tile_m = 64;
            *tile_n = 128;
            break;
        case 4: /* GPU */
            *threads = 16;
            *tile_k = 128;
            *tile_m = 128;
            *tile_n = 256;
            break;
        default:
            break;
    }
}
