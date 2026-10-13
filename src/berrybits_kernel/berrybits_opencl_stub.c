#include "berrybits_opencl.h"

int berrybits_opencl_available(void) {
    return 0;
}

int berrybits_opencl_init(berrybits_opencl_context_t *context,
                          const berrybits_q4_model_t *model,
                          const char *kernel_source_path) {
    (void)model;
    (void)kernel_source_path;
    if (context) context->state = 0;
    return 0;
}

int berrybits_opencl_gemv(berrybits_opencl_context_t *context,
                          const float *input, float *output) {
    (void)context;
    (void)input;
    (void)output;
    return 0;
}

void berrybits_opencl_destroy(berrybits_opencl_context_t *context) {
    if (context) context->state = 0;
}
