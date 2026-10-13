#ifndef BERRYBITS_OPENCL_H
#define BERRYBITS_OPENCL_H

#include "berrybits_q4_model.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    void *state;
} berrybits_opencl_context_t;

int berrybits_opencl_available(void);
int berrybits_opencl_init(berrybits_opencl_context_t *context,
                          const berrybits_q4_model_t *model,
                          const char *kernel_source_path);
int berrybits_opencl_gemv(berrybits_opencl_context_t *context,
                          const float *input, float *output);
void berrybits_opencl_destroy(berrybits_opencl_context_t *context);

#ifdef __cplusplus
}
#endif

#endif
