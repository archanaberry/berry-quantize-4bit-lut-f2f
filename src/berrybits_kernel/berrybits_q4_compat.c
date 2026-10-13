#include "berrybits_q4_model.h"
#include "../berrybits_training/berrybits_training.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

struct berrybits_model_handle {
    berrybits_q4_model_t model;
};

static int berrybits_ffi_size_multiply(size_t left, size_t right, size_t *result) {
    if (right != 0 && left > SIZE_MAX / right) return 0;
    *result = left * right;
    return 1;
}

berrybits_model_handle_t *berrybits_model_create(uint32_t rows, uint32_t cols) {
    berrybits_model_handle_t *handle =
        (berrybits_model_handle_t *)calloc(1, sizeof(*handle));
    if (!handle) return NULL;
    if (!berrybits_q4_model_init(&handle->model, rows, cols)) {
        free(handle);
        return NULL;
    }
    return handle;
}

berrybits_model_handle_t *berrybits_model_load(const char *path) {
    if (!path) return NULL;
    berrybits_model_handle_t *handle =
        (berrybits_model_handle_t *)calloc(1, sizeof(*handle));
    if (!handle) return NULL;
    if (!berrybits_q4_model_load(&handle->model, path)) {
        free(handle);
        return NULL;
    }
    return handle;
}

void berrybits_model_destroy(berrybits_model_handle_t *handle) {
    if (!handle) return;
    berrybits_q4_model_destroy(&handle->model);
    free(handle);
}

uint32_t berrybits_model_rows(const berrybits_model_handle_t *handle) {
    return handle ? handle->model.rows : 0;
}

uint32_t berrybits_model_cols(const berrybits_model_handle_t *handle) {
    return handle ? handle->model.cols : 0;
}

int berrybits_model_quantize(berrybits_model_handle_t *handle, const float *weights) {
    if (!handle) return 0;
    return berrybits_q4_model_quantize(&handle->model, weights);
}

int berrybits_model_gemv(berrybits_model_handle_t *handle,
                         const float *input, float *output) {
    if (!handle || !input || !output) return 0;
    berrybits_q4_model_gemv(&handle->model, input, output);
    return 1;
}

int berrybits_model_gemv_batch(berrybits_model_handle_t *handle,
                               const float *inputs,
                               size_t batch_size,
                               float *outputs) {
    size_t input_count;
    size_t output_count;
    if (!handle || !inputs || !outputs || batch_size == 0 ||
        !berrybits_ffi_size_multiply(batch_size, handle->model.cols, &input_count) ||
        !berrybits_ffi_size_multiply(batch_size, handle->model.rows, &output_count) ||
        input_count > SIZE_MAX / sizeof(float) || output_count > SIZE_MAX / sizeof(float)) {
        return 0;
    }
    for (size_t batch = 0; batch < batch_size; ++batch) {
        berrybits_q4_model_gemv(&handle->model,
                                inputs + batch * handle->model.cols,
                                outputs + batch * handle->model.rows);
    }
    return 1;
}

int berrybits_model_save(berrybits_model_handle_t *handle, const char *path) {
    if (!handle) return 0;
    return berrybits_q4_model_save(&handle->model, path);
}

int berrybits_model_distill_linear(
    berrybits_model_handle_t *handle,
    const float *teacher_weights,
    const float *teacher_bias,
    const float *calibration_inputs,
    size_t samples,
    uint32_t epochs,
    float learning_rate,
    float *rmse_before,
    float *rmse_after) {
    if (!handle || !teacher_weights || !calibration_inputs || samples == 0 || epochs == 0) return 0;

    size_t weight_count;
    size_t weight_bytes;
    size_t bias_bytes = 0;
    if (!berrybits_ffi_size_multiply(handle->model.rows, handle->model.cols, &weight_count) ||
        !berrybits_ffi_size_multiply(weight_count, sizeof(float), &weight_bytes) ||
        (teacher_bias && !berrybits_ffi_size_multiply(handle->model.rows, sizeof(float), &bias_bytes))) {
        return 0;
    }

    float *student_weights = (float *)malloc(weight_bytes);
    float *student_bias = teacher_bias
                              ? (float *)malloc(bias_bytes)
                              : NULL;
    if (!student_weights || (teacher_bias && !student_bias)) {
        free(student_weights);
        free(student_bias);
        return 0;
    }
    memcpy(student_weights, teacher_weights, weight_bytes);
    if (teacher_bias) {
        memcpy(student_bias, teacher_bias, bias_bytes);
    }

    berrybits_training_config_t config = {
        .epochs = epochs,
        .learning_rate = learning_rate
    };
    int result = berrybits_training_distill_linear(
        teacher_weights, teacher_bias, student_weights, student_bias,
        calibration_inputs, samples, handle->model.cols, handle->model.rows,
        &handle->model, &config, rmse_before, rmse_after);
    free(student_weights);
    free(student_bias);
    return result;
}
