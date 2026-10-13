#include "berrybits_training.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static int berrybits_training_multiply(size_t left, size_t right, size_t *result) {
    if (right != 0 && left > SIZE_MAX / right) return 0;
    *result = left * right;
    return 1;
}

static void berrybits_dense_forward(const float *weights, const float *bias,
                                    const float *input, float *output,
                                    size_t input_count, size_t output_count) {
    for (size_t row = 0; row < output_count; ++row) {
        const float *row_weights = weights + row * input_count;
        float sum0 = 0.0f;
        float sum1 = 0.0f;
        float sum2 = 0.0f;
        float sum3 = 0.0f;
        size_t column = 0;
        for (; column + 4 <= input_count; column += 4) {
            sum0 += row_weights[column] * input[column];
            sum1 += row_weights[column + 1] * input[column + 1];
            sum2 += row_weights[column + 2] * input[column + 2];
            sum3 += row_weights[column + 3] * input[column + 3];
        }
        float value = (sum0 + sum1) + (sum2 + sum3);
        for (; column < input_count; ++column) {
            value += row_weights[column] * input[column];
        }
        output[row] = value + (bias ? bias[row] : 0.0f);
    }
}

static double berrybits_student_squared_error(
    const berrybits_q4_model_t *student_q4,
    const float *student_bias,
    const float *calibration_inputs,
    const float *teacher_outputs,
    size_t samples,
    size_t input_count,
    size_t output_count,
    float *student_output) {
    double squared_error = 0.0;
    for (size_t sample = 0; sample < samples; ++sample) {
        berrybits_q4_model_gemv(student_q4,
                                calibration_inputs + sample * input_count,
                                student_output);
        for (size_t row = 0; row < output_count; ++row) {
            double difference = (double)(student_output[row] +
                                         (student_bias ? student_bias[row] : 0.0f)) -
                                teacher_outputs[sample * output_count + row];
            squared_error += difference * difference;
        }
    }
    return squared_error;
}

static int berrybits_finite_array(const float *values, size_t count) {
    if (!values) return 0;
    for (size_t i = 0; i < count; ++i) {
        if (!isfinite(values[i])) return 0;
    }
    return 1;
}

void berrybits_training_copy_teacher(const float *teacher_weights,
                                     const float *teacher_bias,
                                     float *student_master_weights,
                                     float *student_bias,
                                     size_t weight_count,
                                     size_t output_count) {
    if (!teacher_weights || !student_master_weights || weight_count == 0) return;
    memmove(student_master_weights, teacher_weights, weight_count * sizeof(float));
    if (teacher_bias && student_bias && output_count > 0) {
        memmove(student_bias, teacher_bias, output_count * sizeof(float));
    }
}

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
    float *rmse_after) {
    if (!teacher_weights || !student_master_weights || !calibration_inputs ||
        !student_q4 || !config || samples == 0 || input_count == 0 || output_count == 0 ||
        config->epochs == 0 || !isfinite(config->learning_rate) || config->learning_rate <= 0.0f ||
        student_q4->rows != output_count || student_q4->cols != input_count ||
        ((teacher_bias == NULL) != (student_bias == NULL))) {
        return 0;
    }

    size_t weight_count, sample_input_count, sample_output_count;
    if (!berrybits_training_multiply(input_count, output_count, &weight_count) ||
        !berrybits_training_multiply(samples, input_count, &sample_input_count) ||
        !berrybits_training_multiply(samples, output_count, &sample_output_count)) {
        return 0;
    }
    if (!berrybits_finite_array(teacher_weights, weight_count) ||
        !berrybits_finite_array(student_master_weights, weight_count) ||
        !berrybits_finite_array(calibration_inputs, sample_input_count) ||
        (teacher_bias && (!berrybits_finite_array(teacher_bias, output_count) ||
                          !berrybits_finite_array(student_bias, output_count)))) {
        return 0;
    }

    float *teacher_outputs = (float *)malloc(sample_output_count * sizeof(float));
    float *differences = (float *)malloc(sample_output_count * sizeof(float));
    float *student_output = (float *)malloc(output_count * sizeof(float));
    float *weight_backup = (float *)malloc(weight_count * sizeof(float));
    float *bias_backup = student_bias ? (float *)malloc(output_count * sizeof(float)) : NULL;
    if (!teacher_outputs || !differences || !student_output || !weight_backup ||
        (student_bias && !bias_backup)) {
        free(teacher_outputs);
        free(differences);
        free(student_output);
        free(weight_backup);
        free(bias_backup);
        return 0;
    }

    for (size_t sample = 0; sample < samples; ++sample) {
        berrybits_dense_forward(teacher_weights, teacher_bias,
                                calibration_inputs + sample * input_count,
                                teacher_outputs + sample * output_count,
                                input_count, output_count);
    }

    double initial_squared_error = 0.0;
    double final_squared_error = 0.0;
    const double normalization = 1.0 / ((double)samples * (double)output_count);

    for (uint32_t epoch = 0; epoch < config->epochs; ++epoch) {
        if (!berrybits_q4_model_quantize(student_q4, student_master_weights)) {
            free(teacher_outputs);
            free(differences);
            free(student_output);
            free(weight_backup);
            free(bias_backup);
            return 0;
        }

        double squared_error = 0.0;
        for (size_t sample = 0; sample < samples; ++sample) {
            const float *input = calibration_inputs + sample * input_count;
            berrybits_q4_model_gemv(student_q4, input, student_output);
            for (size_t row = 0; row < output_count; ++row) {
                size_t index = sample * output_count + row;
                float prediction = student_output[row] + (student_bias ? student_bias[row] : 0.0f);
                float difference = prediction - teacher_outputs[index];
                differences[index] = difference;
                squared_error += (double)difference * difference;
            }
        }
        if (epoch == 0) initial_squared_error = squared_error;
        final_squared_error = squared_error;

        memcpy(weight_backup, student_master_weights, weight_count * sizeof(float));
        if (student_bias) memcpy(bias_backup, student_bias, output_count * sizeof(float));

        int accepted = 0;
        float trial_rate = config->learning_rate;
        for (int attempt = 0; attempt < 12; ++attempt, trial_rate *= 0.5f) {
            for (size_t row = 0; row < output_count; ++row) {
                float *row_weights = student_master_weights + row * input_count;
                const float *original_weights = weight_backup + row * input_count;
                for (size_t column = 0; column < input_count; ++column) {
                    double gradient = 0.0;
                    for (size_t sample = 0; sample < samples; ++sample) {
                        gradient += (double)differences[sample * output_count + row] *
                                    calibration_inputs[sample * input_count + column];
                    }
                    row_weights[column] = original_weights[column] - trial_rate *
                                          (float)(2.0 * gradient * normalization);
                }
                if (student_bias) {
                    double gradient = 0.0;
                    for (size_t sample = 0; sample < samples; ++sample) {
                        gradient += differences[sample * output_count + row];
                    }
                    student_bias[row] = bias_backup[row] - trial_rate *
                                        (float)(2.0 * gradient * normalization);
                }
            }

            if (!berrybits_q4_model_quantize(student_q4, student_master_weights)) {
                free(teacher_outputs);
                free(differences);
                free(student_output);
                free(weight_backup);
                free(bias_backup);
                return 0;
            }
            double candidate_error = berrybits_student_squared_error(
                student_q4, student_bias, calibration_inputs, teacher_outputs,
                samples, input_count, output_count, student_output);
            if (candidate_error <= squared_error) {
                final_squared_error = candidate_error;
                accepted = 1;
                break;
            }
        }
        if (!accepted) {
            memcpy(student_master_weights, weight_backup, weight_count * sizeof(float));
            if (student_bias) memcpy(student_bias, bias_backup, output_count * sizeof(float));
            if (!berrybits_q4_model_quantize(student_q4, student_master_weights)) {
                free(teacher_outputs);
                free(differences);
                free(student_output);
                free(weight_backup);
                free(bias_backup);
                return 0;
            }
            final_squared_error = squared_error;
            break;
        }
    }

    if (!berrybits_q4_model_quantize(student_q4, student_master_weights)) {
        free(teacher_outputs);
        free(differences);
        free(student_output);
        free(weight_backup);
        free(bias_backup);
        return 0;
    }

    final_squared_error = 0.0;
    for (size_t sample = 0; sample < samples; ++sample) {
        berrybits_q4_model_gemv(student_q4,
                                calibration_inputs + sample * input_count,
                                student_output);
        for (size_t row = 0; row < output_count; ++row) {
            float prediction = student_output[row] + (student_bias ? student_bias[row] : 0.0f);
            double difference = (double)prediction - teacher_outputs[sample * output_count + row];
            final_squared_error += difference * difference;
        }
    }

    if (rmse_before) *rmse_before = (float)sqrt(initial_squared_error * normalization);
    if (rmse_after) *rmse_after = (float)sqrt(final_squared_error * normalization);
    free(teacher_outputs);
    free(differences);
    free(student_output);
    free(weight_backup);
    free(bias_backup);
    return 1;
}
