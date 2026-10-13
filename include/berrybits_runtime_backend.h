#ifndef BERRYBITS_BACKEND_H
#define BERRYBITS_BACKEND_H

#include "berrybits_runtime.h"

#ifdef __cplusplus
extern "C" {
#endif

float berrybits_runtime_q4_backend_avx_dot(const float *a, const float *b, size_t n);
float berrybits_runtime_q4_backend_avx2_q4_dot(const uint8_t *packed, const float *x, size_t n, const float *lut, float scale, float zero);
float berrybits_runtime_q4_backend_neon_dot(const float *a, const float *b, size_t n);
float berrybits_runtime_q4_backend_neon_q4_dot(const uint8_t *packed, const float *x, size_t n, const float *lut, float scale, float zero);
float berrybits_runtime_q4_backend_hvx_dot(const float *a, const float *b, size_t n);
float berrybits_runtime_q4_backend_gpu_dot(const float *a, const float *b, size_t n);
float berrybits_runtime_q4_backend_cpu_q4_dot(const uint8_t *packed, const float *x, size_t n, const float *lut, float scale, float zero);

void berrybits_runtime_q4_optimizer_calibrate(float *teacher, float *student, size_t n, float scale, float zero);
float berrybits_runtime_q4_optimizer_teacher_loss(const float *teacher, const float *student, size_t n);
void berrybits_runtime_q4_optimizer_model_plan(int backend_id, int *threads, int *tile_k, int *tile_m, int *tile_n);

#ifdef __cplusplus
}
#endif

#endif
