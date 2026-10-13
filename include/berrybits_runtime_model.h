#ifndef BERRYBITS_MODEL_H
#define BERRYBITS_MODEL_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    char name[64];
    size_t params;
    size_t hidden;
    int backend_id;
    int threads;
    int tile_k;
    int tile_m;
    int tile_n;
} berrybits_runtime_q4_model_desc_t;

void berrybits_runtime_q4_model_register(const char *name, size_t params, size_t hidden, int backend_id, int threads, int tile_k, int tile_m, int tile_n);
const berrybits_runtime_q4_model_desc_t *berrybits_runtime_q4_model_get_default(void);

#ifdef __cplusplus
}
#endif

#endif
