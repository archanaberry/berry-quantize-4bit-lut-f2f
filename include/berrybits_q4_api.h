#ifndef BERRYBITS_H
#define BERRYBITS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define BERRYBITS_Q4_LUT_SIZE 16u
#define BERRYBITS_Q4_BLOCK_SIZE 32u
#define BERRYBITS_Q4_MODEL_VERSION 1u

typedef struct {
    uint32_t rows;
    uint32_t cols;
    uint32_t blocks_per_row;
    float lut[BERRYBITS_Q4_LUT_SIZE];
    float *scales;
    uint8_t *packed_weights;
} berrybits_q4_model_t;

typedef struct {
    uint32_t epochs;
    float learning_rate;
} berrybits_training_config_t;

int berrybits_q4_model_init(berrybits_q4_model_t *model, uint32_t rows, uint32_t cols);
void berrybits_q4_model_destroy(berrybits_q4_model_t *model);
int berrybits_q4_model_quantize(berrybits_q4_model_t *model, const float *weights);
void berrybits_q4_model_gemv(const berrybits_q4_model_t *model, const float *input, float *output);
int berrybits_q4_model_save(const berrybits_q4_model_t *model, const char *path);
int berrybits_q4_model_load(berrybits_q4_model_t *model, const char *path);

float berrybits_fp16_to_fp32(uint16_t value);
int berrybits_fp16_array_to_fp32(const uint16_t *source, float *destination, size_t count);
uint16_t berrybits_fp32_to_fp16(float value);
int berrybits_fp32_array_to_fp16(const float *source, uint16_t *destination, size_t count);

void berrybits_training_copy_teacher(const float *teacher_weights,
                                     const float *teacher_bias,
                                     float *student_master_weights,
                                     float *student_bias,
                                     size_t weight_count,
                                     size_t output_count);
int berrybits_training_distill_linear(
    const float *teacher_weights,
    const float *teacher_bias,
    float *student_master_weights,
    float *student_bias,
    const float *calibration_inputs,
    size_t samples,
    size_t input_count,
    size_t output_count,
    berrybits_q4_model_t *student_q4,
    const berrybits_training_config_t *config,
    float *rmse_before,
    float *rmse_after);

typedef struct berrybits_model_handle berrybits_model_handle_t;

berrybits_model_handle_t *berrybits_model_create(uint32_t rows, uint32_t cols);
berrybits_model_handle_t *berrybits_model_load(const char *path);
void berrybits_model_destroy(berrybits_model_handle_t *handle);
uint32_t berrybits_model_rows(const berrybits_model_handle_t *handle);
uint32_t berrybits_model_cols(const berrybits_model_handle_t *handle);
int berrybits_model_quantize(berrybits_model_handle_t *handle, const float *weights);
int berrybits_model_gemv(berrybits_model_handle_t *handle, const float *input, float *output);
int berrybits_model_gemv_batch(berrybits_model_handle_t *handle,
                               const float *inputs,
                               size_t batch_size,
                               float *outputs);
int berrybits_model_save(berrybits_model_handle_t *handle, const char *path);
int berrybits_model_distill_linear(
    berrybits_model_handle_t *handle,
    const float *teacher_weights,
    const float *teacher_bias,
    const float *calibration_inputs,
    size_t samples,
    uint32_t epochs,
    float learning_rate,
    float *rmse_before,
    float *rmse_after);

#ifdef __cplusplus
}
#endif

#endif
