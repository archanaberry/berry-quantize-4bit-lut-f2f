#ifndef BERRYBITS_CUDA_H
#define BERRYBITS_CUDA_H

#include "berrybits_q4_model.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    void *state;
} berrybits_cuda_context_t;

int berrybits_cuda_available(void);
int berrybits_cuda_init(berrybits_cuda_context_t *context, const berrybits_q4_model_t *model);
int berrybits_cuda_gemv(berrybits_cuda_context_t *context, const float *input, float *output);
void berrybits_cuda_destroy(berrybits_cuda_context_t *context);

#ifdef __cplusplus
}
#endif

#endif
