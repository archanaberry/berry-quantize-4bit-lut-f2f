#include "berrybits_runtime_model.h"

#include <stdio.h>
#include <string.h>

static berrybits_runtime_q4_model_desc_t g_default_model = {
    .name = "DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M_LUT_F",
    .params = 1500000000ULL,
    .hidden = 1536,
    .backend_id = 2,
    .threads = 8,
    .tile_k = 32,
    .tile_m = 64,
    .tile_n = 128,
};

void berrybits_runtime_q4_model_register(const char *name, size_t params, size_t hidden, int backend_id, int threads, int tile_k, int tile_m, int tile_n) {
    if (!name || !*name) return;

    memset(&g_default_model, 0, sizeof(g_default_model));
    snprintf(g_default_model.name, sizeof(g_default_model.name), "%s", name);
    g_default_model.params = params;
    g_default_model.hidden = hidden;
    g_default_model.backend_id = backend_id;
    g_default_model.threads = threads;
    g_default_model.tile_k = tile_k;
    g_default_model.tile_m = tile_m;
    g_default_model.tile_n = tile_n;
}

const berrybits_runtime_q4_model_desc_t *berrybits_runtime_q4_model_get_default(void) {
    return &g_default_model;
}
