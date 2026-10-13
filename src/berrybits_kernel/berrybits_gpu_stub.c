#include "berrybits_runtime_backend.h"

float berrybits_runtime_q4_backend_gpu_dot(const float *a, const float *b, size_t n) {
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) sum += a[i] * b[i];
    return sum;
}
