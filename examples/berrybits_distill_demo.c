#include "berrybits_training.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define DEMO_INPUTS 32u
#define DEMO_OUTPUTS 8u
#define DEMO_TRAIN_SAMPLES 128u
#define DEMO_VALID_SAMPLES 32u
#define DEMO_TOTAL_SAMPLES (DEMO_TRAIN_SAMPLES + DEMO_VALID_SAMPLES)

static uint32_t next_random(uint32_t *state) {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    return *state;
}

static float dense_value(const float *weights, const float *input, size_t row) {
    const float *row_weights = weights + row * DEMO_INPUTS;
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;
    size_t column = 0;
    for (; column + 4 <= DEMO_INPUTS; column += 4) {
        sum0 += row_weights[column] * input[column];
        sum1 += row_weights[column + 1] * input[column + 1];
        sum2 += row_weights[column + 2] * input[column + 2];
        sum3 += row_weights[column + 3] * input[column + 3];
    }
    return (sum0 + sum1) + (sum2 + sum3);
}

static float dense_rmse(const float *weights, const float *inputs,
                        const float *targets, size_t samples) {
    double squared_error = 0.0;
    for (size_t sample = 0; sample < samples; ++sample) {
        const float *input = inputs + sample * DEMO_INPUTS;
        for (size_t row = 0; row < DEMO_OUTPUTS; ++row) {
            double difference = dense_value(weights, input, row) -
                                targets[sample * DEMO_OUTPUTS + row];
            squared_error += difference * difference;
        }
    }
    return (float)sqrt(squared_error / ((double)samples * DEMO_OUTPUTS));
}

static void dense_outputs(const float *weights, const float *inputs,
                          float *outputs, size_t samples) {
    for (size_t sample = 0; sample < samples; ++sample) {
        for (size_t row = 0; row < DEMO_OUTPUTS; ++row) {
            outputs[sample * DEMO_OUTPUTS + row] =
                dense_value(weights, inputs + sample * DEMO_INPUTS, row);
        }
    }
}

static float quantized_rmse(const berrybits_q4_model_t *model,
                            const float *inputs, const float *targets,
                            size_t samples) {
    float outputs[DEMO_OUTPUTS];
    double squared_error = 0.0;
    for (size_t sample = 0; sample < samples; ++sample) {
        berrybits_q4_model_gemv(model, inputs + sample * DEMO_INPUTS, outputs);
        for (size_t row = 0; row < DEMO_OUTPUTS; ++row) {
            double difference = outputs[row] - targets[sample * DEMO_OUTPUTS + row];
            squared_error += difference * difference;
        }
    }
    return (float)sqrt(squared_error / ((double)samples * DEMO_OUTPUTS));
}

static void train_teacher_from_zero(const float *inputs, const float *targets,
                                    float *weights) {
    float gradients[DEMO_OUTPUTS * DEMO_INPUTS];
    memset(weights, 0, sizeof(float) * DEMO_OUTPUTS * DEMO_INPUTS);

    for (size_t epoch = 0; epoch < 500; ++epoch) {
        memset(gradients, 0, sizeof(gradients));
        for (size_t sample = 0; sample < DEMO_TRAIN_SAMPLES; ++sample) {
            const float *input = inputs + sample * DEMO_INPUTS;
            for (size_t row = 0; row < DEMO_OUTPUTS; ++row) {
                float difference = dense_value(weights, input, row) -
                                   targets[sample * DEMO_OUTPUTS + row];
                float factor = 2.0f * difference /
                               ((float)DEMO_TRAIN_SAMPLES * DEMO_OUTPUTS);
                for (size_t column = 0; column < DEMO_INPUTS; ++column) {
                    gradients[row * DEMO_INPUTS + column] += factor * input[column];
                }
            }
        }
        float learning_rate = 0.45f;
        for (size_t i = 0; i < DEMO_OUTPUTS * DEMO_INPUTS; ++i) {
            weights[i] -= learning_rate * gradients[i];
        }
    }
}

static int distill_and_save(const char *path, const float *teacher_weights,
                            const float *train_inputs, const float *valid_inputs,
                            const float *teacher_valid_targets,
                            float *student_master, berrybits_q4_model_t *student,
                            float *calibration_before, float *calibration_after,
                            float *validation_rmse) {
    berrybits_training_copy_teacher(teacher_weights, NULL, student_master, NULL,
                                    DEMO_OUTPUTS * DEMO_INPUTS, DEMO_OUTPUTS);
    berrybits_training_config_t config = {
        .epochs = 100,
        .learning_rate = 0.04f
    };
    if (!berrybits_training_distill_linear(
            teacher_weights, NULL, student_master, NULL, train_inputs,
            DEMO_TRAIN_SAMPLES, DEMO_INPUTS, DEMO_OUTPUTS, student, &config,
            calibration_before, calibration_after)) return 0;
    *validation_rmse = quantized_rmse(student, valid_inputs, teacher_valid_targets,
                                     DEMO_VALID_SAMPLES);
    return berrybits_q4_model_save(student, path);
}

static long model_file_size(const char *path) {
    FILE *file = fopen(path, "rb");
    if (!file) return -1;
    if (fseek(file, 0, SEEK_END) != 0) {
        fclose(file);
        return -1;
    }
    long size = ftell(file);
    fclose(file);
    return size;
}

int main(void) {
    float true_weights[DEMO_OUTPUTS * DEMO_INPUTS];
    float fp32_teacher[DEMO_OUTPUTS * DEMO_INPUTS] = {0};
    float fp16_teacher[DEMO_OUTPUTS * DEMO_INPUTS];
    float train_inputs[DEMO_TRAIN_SAMPLES * DEMO_INPUTS];
    float valid_inputs[DEMO_VALID_SAMPLES * DEMO_INPUTS];
    float targets[DEMO_TOTAL_SAMPLES * DEMO_OUTPUTS];
    float fp32_teacher_valid[DEMO_VALID_SAMPLES * DEMO_OUTPUTS];
    float fp16_teacher_valid[DEMO_VALID_SAMPLES * DEMO_OUTPUTS];
    uint16_t half_weights[DEMO_OUTPUTS * DEMO_INPUTS];
    float fp32_calibration_before, fp32_calibration_after, fp32_valid_rmse;
    float fp16_calibration_before, fp16_calibration_after, fp16_valid_rmse;
    float fp32_student_truth_rmse, fp16_student_truth_rmse;

    for (size_t i = 0; i < DEMO_OUTPUTS * DEMO_INPUTS; ++i) {
        int centered = (int)((i * 17u + 11u) % 31u) - 15;
        true_weights[i] = (float)centered * 0.012f;
    }

    uint32_t random_state = 0xB4F04u;
    for (size_t sample = 0; sample < DEMO_TOTAL_SAMPLES; ++sample) {
        float *input = sample < DEMO_TRAIN_SAMPLES
                           ? train_inputs + sample * DEMO_INPUTS
                           : valid_inputs + (sample - DEMO_TRAIN_SAMPLES) * DEMO_INPUTS;
        for (size_t column = 0; column < DEMO_INPUTS; ++column) {
            uint32_t random_bits = next_random(&random_state);
            input[column] = (float)(random_bits & 0xFFFFu) / 32767.5f - 1.0f;
        }
        for (size_t row = 0; row < DEMO_OUTPUTS; ++row) {
            targets[sample * DEMO_OUTPUTS + row] = dense_value(true_weights, input, row);
        }
    }

    float initial_teacher_rmse = dense_rmse(fp32_teacher, train_inputs, targets,
                                             DEMO_TRAIN_SAMPLES);
    train_teacher_from_zero(train_inputs, targets, fp32_teacher);
    float trained_teacher_rmse = dense_rmse(fp32_teacher, train_inputs, targets,
                                             DEMO_TRAIN_SAMPLES);
    float fp32_valid_teacher_rmse = dense_rmse(
        fp32_teacher, valid_inputs,
        targets + DEMO_TRAIN_SAMPLES * DEMO_OUTPUTS, DEMO_VALID_SAMPLES);

    if (!berrybits_fp32_array_to_fp16(fp32_teacher, half_weights,
                                      DEMO_OUTPUTS * DEMO_INPUTS) ||
        !berrybits_fp16_array_to_fp32(half_weights, fp16_teacher,
                                      DEMO_OUTPUTS * DEMO_INPUTS)) {
        fprintf(stderr, "FP16 teacher conversion failed\n");
        return 1;
    }
    float fp16_valid_teacher_rmse = dense_rmse(
        fp16_teacher, valid_inputs,
        targets + DEMO_TRAIN_SAMPLES * DEMO_OUTPUTS, DEMO_VALID_SAMPLES);
    dense_outputs(fp32_teacher, valid_inputs, fp32_teacher_valid, DEMO_VALID_SAMPLES);
    dense_outputs(fp16_teacher, valid_inputs, fp16_teacher_valid, DEMO_VALID_SAMPLES);

    berrybits_q4_model_t fp32_student = {0};
    berrybits_q4_model_t fp16_student = {0};
    float fp32_student_master[DEMO_OUTPUTS * DEMO_INPUTS];
    float fp16_student_master[DEMO_OUTPUTS * DEMO_INPUTS];
    if (!berrybits_q4_model_init(&fp32_student, DEMO_OUTPUTS, DEMO_INPUTS) ||
        !berrybits_q4_model_init(&fp16_student, DEMO_OUTPUTS, DEMO_INPUTS)) {
        fprintf(stderr, "Q4 student allocation failed\n");
        berrybits_q4_model_destroy(&fp32_student);
        berrybits_q4_model_destroy(&fp16_student);
        return 1;
    }

    if (!distill_and_save("build/demo_fp32_teacher_q4.b4lut", fp32_teacher,
                          train_inputs, valid_inputs,
                          fp32_teacher_valid,
                          fp32_student_master, &fp32_student,
                          &fp32_calibration_before, &fp32_calibration_after,
                          &fp32_valid_rmse) ||
        !distill_and_save("build/demo_fp16_teacher_q4.b4lut", fp16_teacher,
                          train_inputs, valid_inputs,
                          fp16_teacher_valid,
                          fp16_student_master, &fp16_student,
                          &fp16_calibration_before, &fp16_calibration_after,
                          &fp16_valid_rmse)) {
        fprintf(stderr, "Q4 distillation or model save failed\n");
        berrybits_q4_model_destroy(&fp32_student);
        berrybits_q4_model_destroy(&fp16_student);
        return 1;
    }

    const float *valid_ground_truth = targets + DEMO_TRAIN_SAMPLES * DEMO_OUTPUTS;
    fp32_student_truth_rmse = quantized_rmse(&fp32_student, valid_inputs,
                                              valid_ground_truth, DEMO_VALID_SAMPLES);
    fp16_student_truth_rmse = quantized_rmse(&fp16_student, valid_inputs,
                                              valid_ground_truth, DEMO_VALID_SAMPLES);

    printf("Tiny teacher: %u inputs, %u outputs, %u train + %u held-out samples\n",
           DEMO_INPUTS, DEMO_OUTPUTS, DEMO_TRAIN_SAMPLES, DEMO_VALID_SAMPLES);
    printf("FP32 teacher train RMSE from zero: %.6f -> %.6f\n",
           initial_teacher_rmse, trained_teacher_rmse);
    printf("FP32 teacher held-out RMSE vs target: %.6f\n", fp32_valid_teacher_rmse);
    printf("FP16 snapshot held-out RMSE vs target: %.6f\n", fp16_valid_teacher_rmse);
    printf("FP32 teacher -> Q4 calibration RMSE: %.6f -> %.6f; held-out vs teacher: %.6f\n",
           fp32_calibration_before, fp32_calibration_after, fp32_valid_rmse);
    printf("FP16 teacher -> Q4 calibration RMSE: %.6f -> %.6f; held-out vs teacher: %.6f\n",
           fp16_calibration_before, fp16_calibration_after, fp16_valid_rmse);
        printf("Both Q4 students on identical held-out targets: FP32 teacher %.6f; FP16 teacher %.6f\n",
            fp32_student_truth_rmse, fp16_student_truth_rmse);
    printf("Saved build/demo_fp32_teacher_q4.b4lut (%ld bytes)\n",
           model_file_size("build/demo_fp32_teacher_q4.b4lut"));
    printf("Saved build/demo_fp16_teacher_q4.b4lut (%ld bytes)\n",
           model_file_size("build/demo_fp16_teacher_q4.b4lut"));

    berrybits_q4_model_destroy(&fp32_student);
    berrybits_q4_model_destroy(&fp16_student);
    return 0;
}
