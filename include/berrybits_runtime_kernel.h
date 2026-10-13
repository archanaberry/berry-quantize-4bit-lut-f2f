#ifndef BERRYBITS_KERNEL_H
#define BERRYBITS_KERNEL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void berrybits_runtime_q4_kernel_prefetch(const void *ptr);
void berrybits_runtime_q4_fused_gemv(const uint8_t *weights, const float *in, float *out, size_t rows, size_t cols, float scale, float zero, const float *lut);

#ifdef __cplusplus
}
#endif

#endif
