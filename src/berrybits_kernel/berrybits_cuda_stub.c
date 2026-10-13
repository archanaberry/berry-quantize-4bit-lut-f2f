#include "berrybits_cuda.h"

int berrybits_cuda_available(void) {
    return 0;
}

int berrybits_cuda_init(berrybits_cuda_context_t *context, const berrybits_q4_model_t *model) {
    (void)model;
    if (context) context->state = 0;
    return 0;
}

int berrybits_cuda_gemv(berrybits_cuda_context_t *context, const float *input, float *output) {
    (void)context;
    (void)input;
    (void)output;
    return 0;
}

void berrybits_cuda_destroy(berrybits_cuda_context_t *context) {
    if (context) context->state = 0;
}
