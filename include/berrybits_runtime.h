#ifndef BERRYBITS_H
#define BERRYBITS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define BERRYBITS_RUNTIME_Q4_LUT_SIZE 16

typedef enum {
    BERRYBITS_RUNTIME_Q4_BACKEND_SCALAR = 0,
    BERRYBITS_RUNTIME_Q4_BACKEND_NEON = 1,
    BERRYBITS_RUNTIME_Q4_BACKEND_AVX2 = 2,
    BERRYBITS_RUNTIME_Q4_BACKEND_HVX = 3,
    BERRYBITS_RUNTIME_Q4_BACKEND_GPU = 4
} berrybits_runtime_q4_backend_t;

typedef struct {
    berrybits_runtime_q4_backend_t backend;
    int num_threads;
    int vector_width;
    int gpu_enabled;
} berrybits_runtime_q4_runtime_config_t;

typedef struct {
    int backend_id;
    int num_threads;
    int vector_width;
    int gpu_enabled;
    int tile_k;
    int tile_m;
    int tile_n;
} berrybits_runtime_q4_runtime_info_t;

typedef struct {
    int supported;
    int platform;
    int threads_per_block;
    int grid_x;
    int warp_size;
} berrybits_runtime_q4_gpu_plan_t;

void berrybits_runtime_q4_init_lut(float *lut);
void berrybits_runtime_q4_quantize_row(const float *input, uint8_t *packed, size_t n, const float *lut);
void berrybits_runtime_q4_dequantize_row(const uint8_t *packed, float *output, size_t n, const float *lut, float scale, float zero);
float berrybits_runtime_q4_dot_row(const uint8_t *packed, const float *x, size_t n, const float *lut, float scale, float zero);
void berrybits_runtime_q4_matmul_fused(const uint8_t *packed, const float *x, float *y, size_t rows, size_t cols, const float *lut, float scale, float zero);
void berrybits_runtime_q4_threaded_matmul(const uint8_t *packed, const float *x, float *y, size_t rows, size_t cols, const float *lut, float scale, float zero, int threads);
float berrybits_runtime_q4_teacher_error(const float *teacher, const float *student, size_t n);
float berrybits_runtime_q4_dot_asm(const float *a, const float *b, size_t n);
int berrybits_runtime_q4_backend_probe(void);
const char *berrybits_runtime_q4_backend_name(int backend);
berrybits_runtime_q4_runtime_config_t berrybits_runtime_q4_runtime_config(void);
berrybits_runtime_q4_runtime_info_t berrybits_runtime_q4_runtime_info(void);
void berrybits_runtime_q4_gpu_plan(berrybits_runtime_q4_gpu_plan_t *plan);
void berrybits_runtime_q4_fused_gemv(const uint8_t *weights, const float *in, float *out, size_t rows, size_t cols, float scale, float zero, const float *lut);
void berrybits_runtime_q4_model_register(const char *name, size_t params, size_t hidden, int backend_id, int threads, int tile_k, int tile_m, int tile_n);

#ifdef __cplusplus
}
#endif

#endif
