#include "berrybits_runtime.h"
#include "berrybits_runtime_backend.h"
#include "berrybits_runtime_kernel.h"
#include "berrybits_runtime_model.h"

#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static const float berrybits_runtime_q4_default_lut[BERRYBITS_RUNTIME_Q4_LUT_SIZE] = {
    -1.0000f, -0.8750f, -0.7500f, -0.6250f,
    -0.5000f, -0.3750f, -0.2500f, -0.1250f,
     0.0000f,  0.1250f,  0.2500f,  0.3750f,
     0.5000f,  0.6250f,  0.7500f,  1.0000f
};

typedef struct {
    const uint8_t *packed;
    const float *x;
    float *y;
    size_t rows;
    size_t cols;
    const float *lut;
    float scale;
    float zero;
    size_t start_row;
    size_t row_count;
} berrybits_runtime_q4_thread_task_t;

static inline float berrybits_runtime_q4_decode_index(uint8_t idx, const float *lut, float scale, float zero) {
    return lut[idx] * scale + zero;
}

static inline uint8_t berrybits_runtime_q4_encode_value(float value, float scale, float zero) {
    int idx = (int)lrintf((value - zero) / scale);
    if (idx < 0) idx = 0;
    if (idx > 15) idx = 15;
    return (uint8_t)idx;
}

static int berrybits_runtime_q4_cpu_thread_count(void) {
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    if (n <= 0) return 1;
    if (n > 32) return 32;
    return (int)n;
}

static float berrybits_runtime_q4_default_dot_row(const uint8_t *packed, const float *x, size_t n, const float *lut, float scale, float zero) {
    return berrybits_runtime_q4_backend_cpu_q4_dot(packed, x, n, lut, scale, zero);
}

static void *berrybits_runtime_q4_thread_worker(void *arg) {
    berrybits_runtime_q4_thread_task_t *task = (berrybits_runtime_q4_thread_task_t *)arg;
    for (size_t r = 0; r < task->row_count; ++r) {
        size_t row_index = task->start_row + r;
        const uint8_t *row_ptr = task->packed + (row_index * ((task->cols + 1u) / 2u));
        task->y[row_index] = berrybits_runtime_q4_dot_row(row_ptr, task->x, task->cols, task->lut, task->scale, task->zero);
    }
    return NULL;
}

void berrybits_runtime_q4_init_lut(float *lut) {
    if (lut == NULL) {
        return;
    }
    memcpy(lut, berrybits_runtime_q4_default_lut, sizeof(berrybits_runtime_q4_default_lut));
}

void berrybits_runtime_q4_quantize_row(const float *input, uint8_t *packed, size_t n, const float *lut) {
    (void)lut;
    if (input == NULL || packed == NULL || n == 0) {
        return;
    }

    float max_abs = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float a = fabsf(input[i]);
        if (a > max_abs) max_abs = a;
    }

    float scale = max_abs > 0.0f ? max_abs : 1.0f;
    float zero = 0.0f;

    memset(packed, 0, (n + 1) / 2);
    for (size_t i = 0; i < n; ++i) {
        float normalized = input[i] / (scale > 0.0f ? scale : 1.0f);
        float mapped = normalized + 1.0f;
        int idx = (int)lrintf(mapped * 7.5f);
        if (idx < 0) idx = 0;
        if (idx > 15) idx = 15;
        size_t byte_pos = i / 2;
        if ((i & 1u) == 0u) {
            packed[byte_pos] = (uint8_t)idx;
        } else {
            packed[byte_pos] |= (uint8_t)((uint8_t)idx << 4);
        }
    }
}

void berrybits_runtime_q4_dequantize_row(const uint8_t *packed, float *output, size_t n, const float *lut, float scale, float zero) {
    if (packed == NULL || output == NULL || n == 0) {
        return;
    }

    for (size_t i = 0; i < n; ++i) {
        size_t byte_pos = i / 2;
        uint8_t raw = packed[byte_pos];
        uint8_t idx = (i & 1u) == 0u ? (raw & 0x0Fu) : (raw >> 4);
        output[i] = berrybits_runtime_q4_decode_index(idx, lut, scale, zero);
    }
}

float berrybits_runtime_q4_dot_row(const uint8_t *packed, const float *x, size_t n, const float *lut, float scale, float zero) {
    if (packed == NULL || x == NULL || lut == NULL || n == 0) {
        return 0.0f;
    }

#if defined(__AVX2__)
    if (n >= 8192u) {
        return berrybits_runtime_q4_backend_avx2_q4_dot(packed, x, n, lut, scale, zero);
    }
#elif defined(__aarch64__) || defined(__ARM_NEON)
    return berrybits_runtime_q4_backend_neon_q4_dot(packed, x, n, lut, scale, zero);
#endif
    return berrybits_runtime_q4_default_dot_row(packed, x, n, lut, scale, zero);
}

void berrybits_runtime_q4_matmul_fused(const uint8_t *packed, const float *x, float *y, size_t rows, size_t cols, const float *lut, float scale, float zero) {
    if (packed == NULL || x == NULL || y == NULL || rows == 0 || cols == 0) {
        return;
    }

    for (size_t r = 0; r < rows; ++r) {
        const uint8_t *row_ptr = packed + (r * ((cols + 1u) / 2u));
        y[r] = berrybits_runtime_q4_dot_row(row_ptr, x, cols, lut, scale, zero);
    }
}

void berrybits_runtime_q4_threaded_matmul(const uint8_t *packed, const float *x, float *y, size_t rows, size_t cols, const float *lut, float scale, float zero, int threads) {
    if (packed == NULL || x == NULL || y == NULL || rows == 0 || cols == 0) {
        return;
    }

    int hw_threads = berrybits_runtime_q4_cpu_thread_count();
    if (threads <= 0) threads = hw_threads;
    if (threads > (int)rows) threads = (int)rows;
    if (threads <= 1 || rows < 2) {
        berrybits_runtime_q4_matmul_fused(packed, x, y, rows, cols, lut, scale, zero);
        return;
    }

    pthread_t *workers = (pthread_t *)malloc(sizeof(pthread_t) * (size_t)threads);
    berrybits_runtime_q4_thread_task_t *tasks = (berrybits_runtime_q4_thread_task_t *)malloc(sizeof(berrybits_runtime_q4_thread_task_t) * (size_t)threads);
    if (!workers || !tasks) {
        free(workers);
        free(tasks);
        berrybits_runtime_q4_matmul_fused(packed, x, y, rows, cols, lut, scale, zero);
        return;
    }

    size_t base = rows / (size_t)threads;
    size_t rem = rows % (size_t)threads;
    for (int t = 0; t < threads; ++t) {
        size_t start = t * base + ((size_t)t < rem ? (size_t)t : rem);
        size_t count = base + ((size_t)t < rem ? 1u : 0u);
        tasks[t].packed = packed;
        tasks[t].x = x;
        tasks[t].y = y;
        tasks[t].rows = rows;
        tasks[t].cols = cols;
        tasks[t].lut = lut;
        tasks[t].scale = scale;
        tasks[t].zero = zero;
        tasks[t].start_row = start;
        tasks[t].row_count = count;
        pthread_create(&workers[t], NULL, berrybits_runtime_q4_thread_worker, &tasks[t]);
    }

    for (int t = 0; t < threads; ++t) {
        pthread_join(workers[t], NULL);
    }

    free(workers);
    free(tasks);
}

float berrybits_runtime_q4_teacher_error(const float *teacher, const float *student, size_t n) {
    if (teacher == NULL || student == NULL || n == 0) {
        return 0.0f;
    }

    double acc = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double diff = (double)teacher[i] - (double)student[i];
        acc += diff * diff;
    }
    return (float)sqrt(acc / (double)n);
}

int berrybits_runtime_q4_backend_probe(void) {
#if defined(__hexagon__)
    return BERRYBITS_RUNTIME_Q4_BACKEND_HVX;
#elif defined(__aarch64__) || defined(__ARM_NEON)
    return BERRYBITS_RUNTIME_Q4_BACKEND_NEON;
#elif defined(__x86_64__) || defined(__i386__)
#if defined(__AVX2__) || defined(__AVX__)
    return BERRYBITS_RUNTIME_Q4_BACKEND_AVX2;
#else
    return BERRYBITS_RUNTIME_Q4_BACKEND_SCALAR;
#endif
#else
    return BERRYBITS_RUNTIME_Q4_BACKEND_SCALAR;
#endif
}

const char *berrybits_runtime_q4_backend_name(int backend) {
    switch (backend) {
        case BERRYBITS_RUNTIME_Q4_BACKEND_NEON:
            return "NEON";
        case BERRYBITS_RUNTIME_Q4_BACKEND_AVX2:
            return "AVX2";
        case BERRYBITS_RUNTIME_Q4_BACKEND_HVX:
            return "HVX";
        case BERRYBITS_RUNTIME_Q4_BACKEND_GPU:
            return "GPU";
        default:
            return "SCALAR";
    }
}

berrybits_runtime_q4_runtime_config_t berrybits_runtime_q4_runtime_config(void) {
    berrybits_runtime_q4_runtime_config_t cfg;
    cfg.backend = (berrybits_runtime_q4_backend_t)berrybits_runtime_q4_backend_probe();
    cfg.num_threads = berrybits_runtime_q4_cpu_thread_count();
    cfg.vector_width = 4;
    if (cfg.backend == BERRYBITS_RUNTIME_Q4_BACKEND_AVX2) cfg.vector_width = 8;
    if (cfg.backend == BERRYBITS_RUNTIME_Q4_BACKEND_HVX) cfg.vector_width = 128;
    cfg.gpu_enabled = 1;
    return cfg;
}

berrybits_runtime_q4_runtime_info_t berrybits_runtime_q4_runtime_info(void) {
    berrybits_runtime_q4_runtime_info_t info;
    int backend_id = berrybits_runtime_q4_backend_probe();
    int threads = 1, tile_k = 32, tile_m = 64, tile_n = 128;
    berrybits_runtime_q4_optimizer_model_plan(backend_id, &threads, &tile_k, &tile_m, &tile_n);

    info.backend_id = backend_id;
    info.num_threads = threads;
    info.vector_width = 4;
    info.gpu_enabled = 1;
    info.tile_k = tile_k;
    info.tile_m = tile_m;
    info.tile_n = tile_n;

    if (backend_id == BERRYBITS_RUNTIME_Q4_BACKEND_AVX2) info.vector_width = 8;
    if (backend_id == BERRYBITS_RUNTIME_Q4_BACKEND_HVX) info.vector_width = 128;
    return info;
}

void berrybits_runtime_q4_gpu_plan(berrybits_runtime_q4_gpu_plan_t *plan) {
    if (!plan) {
        return;
    }
    plan->supported = 1;
    plan->platform = 0;
    plan->threads_per_block = 128;
    plan->grid_x = 64;
    plan->warp_size = 32;

#if defined(__aarch64__) || defined(__arm__)
    plan->platform = 1;
    plan->threads_per_block = 256;
    plan->grid_x = 32;
    plan->warp_size = 32;
#elif defined(__x86_64__) || defined(__i386__)
    plan->platform = 2;
    plan->threads_per_block = 128;
    plan->grid_x = 64;
    plan->warp_size = 32;
#endif
}

