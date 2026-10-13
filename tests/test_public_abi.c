#include "berrybits_q4_api.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    const float teacher_weights[8] = {
        0.1f, -0.2f, 0.3f, -0.4f,
        -0.4f, 0.3f, -0.2f, 0.1f
    };
    const float inputs[8] = {
        1.0f, 0.5f, -0.5f, 0.25f,
        -0.25f, 0.75f, 1.0f, -1.0f
    };
    berrybits_model_handle_t *model = berrybits_model_create(2, 4);
    if (!model) return 1;

    float rmse_before = 0.0f;
    float rmse_after = 0.0f;
    if (!berrybits_model_distill_linear(model, teacher_weights, NULL, inputs,
                                         2, 4, 0.02f,
                                         &rmse_before, &rmse_after) ||
        rmse_after > rmse_before + 1e-6f) {
        berrybits_model_destroy(model);
        return 2;
    }

    float output[2];
    if (berrybits_model_rows(model) != 2 || berrybits_model_cols(model) != 4 ||
        !berrybits_model_gemv(model, inputs, output)) {
        berrybits_model_destroy(model);
        return 3;
    }

    const char *path = "build/public_abi_test.b4lut";
    if (!berrybits_model_save(model, path)) {
        berrybits_model_destroy(model);
        return 4;
    }
    berrybits_model_destroy(model);

    berrybits_model_handle_t *loaded = berrybits_model_load(path);
    remove(path);
    if (!loaded || !berrybits_model_gemv(loaded, inputs, output) ||
        !isfinite(output[0]) || !isfinite(output[1])) {
        berrybits_model_destroy(loaded);
        return 5;
    }
    berrybits_model_destroy(loaded);
    puts("public C opaque-handle ABI passed");
    return 0;
}
