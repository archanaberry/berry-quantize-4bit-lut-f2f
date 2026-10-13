#include "berrybits_training.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void) {
    enum { rows = 2, cols = 9, samples = 16 };
    float teacher_weights[rows * cols];
    float student_weights[rows * cols];
    float teacher_bias[rows] = {0.03f, -0.05f};
    float student_bias[rows];
    float inputs[samples * cols];

    for (size_t i = 0; i < rows * cols; ++i) {
        teacher_weights[i] = sinf((float)i * 0.37f) * 0.9f;
    }
    for (size_t sample = 0; sample < samples; ++sample) {
        for (size_t column = 0; column < cols; ++column) {
            int value = (int)((sample * (column + 3u)) % 17u) - 8;
            inputs[sample * cols + column] = (float)value / 8.0f;
        }
    }

    berrybits_q4_model_t student_model = {0};
    if (!berrybits_q4_model_init(&student_model, rows, cols)) return 1;
    berrybits_training_copy_teacher(teacher_weights, teacher_bias,
                                    student_weights, student_bias,
                                    rows * cols, rows);

    berrybits_training_config_t config = {
        .epochs = 400,
        .learning_rate = 0.08f
    };
    float rmse_before = 0.0f;
    float rmse_after = 0.0f;
    if (!berrybits_training_distill_linear(
            teacher_weights, teacher_bias, student_weights, student_bias,
            inputs, samples, cols, rows, &student_model, &config,
            &rmse_before, &rmse_after)) {
        berrybits_q4_model_destroy(&student_model);
        return 2;
    }
    if (!isfinite(rmse_before) || !isfinite(rmse_after) || rmse_after > rmse_before + 1e-5f) {
        fprintf(stderr, "distillation did not improve: %.8f -> %.8f\n", rmse_before, rmse_after);
        berrybits_q4_model_destroy(&student_model);
        return 3;
    }

    const char *model_path = "build/test_berrybits_model.b4lut";
    if (!berrybits_q4_model_save(&student_model, model_path)) {
        berrybits_q4_model_destroy(&student_model);
        return 4;
    }
    berrybits_q4_model_t loaded_model = {0};
    if (!berrybits_q4_model_load(&loaded_model, model_path)) {
        berrybits_q4_model_destroy(&student_model);
        remove(model_path);
        return 5;
    }

    float original_output[rows];
    float loaded_output[rows];
    berrybits_q4_model_gemv(&student_model, inputs, original_output);
    berrybits_q4_model_gemv(&loaded_model, inputs, loaded_output);
    if (memcmp(student_model.lut, loaded_model.lut, sizeof(student_model.lut)) != 0 ||
        memcmp(original_output, loaded_output, sizeof(original_output)) != 0) {
        berrybits_q4_model_destroy(&student_model);
        berrybits_q4_model_destroy(&loaded_model);
        remove(model_path);
        return 6;
    }
    for (size_t row = 0; row < rows; ++row) {
        double reference = 0.0;
        for (size_t column = 0; column < cols; ++column) {
            size_t packed_index = row * ((cols + 1u) / 2u) + column / 2u;
            uint8_t packed_byte = student_model.packed_weights[packed_index];
            uint8_t code = (column & 1u) ? (packed_byte >> 4) : (packed_byte & 0x0Fu);
            size_t block = column / BERRYBITS_Q4_BLOCK_SIZE;
            float scale = student_model.scales[row * student_model.blocks_per_row + block];
            reference += student_model.lut[code] * scale * inputs[column];
        }
        if (fabsf(original_output[row] - (float)reference) > 1e-5f) {
            berrybits_q4_model_destroy(&student_model);
            berrybits_q4_model_destroy(&loaded_model);
            remove(model_path);
            return 9;
        }
    }

    if (berrybits_fp16_to_fp32(0x3c00u) != 1.0f ||
        berrybits_fp16_to_fp32(0xc000u) != -2.0f ||
        fabsf(berrybits_fp16_to_fp32(0x0001u) - 5.96046448e-8f) > 1e-12f) {
        berrybits_q4_model_destroy(&student_model);
        berrybits_q4_model_destroy(&loaded_model);
        remove(model_path);
        return 7;
    }
    uint16_t half_values[3] = {0x3c00u, 0xc000u, 0x0001u};
    float converted_values[3];
    if (!berrybits_fp16_array_to_fp32(half_values, converted_values, 3) ||
        converted_values[0] != 1.0f || converted_values[1] != -2.0f ||
        fabsf(converted_values[2] - 5.96046448e-8f) > 1e-12f) {
        berrybits_q4_model_destroy(&student_model);
        berrybits_q4_model_destroy(&loaded_model);
        remove(model_path);
        return 8;
    }
    float single_precision_values[4] = {1.0f, -2.0f, 5.96046448e-8f, 1.00048828125f};
    uint16_t converted_half_values[4];
    if (!berrybits_fp32_array_to_fp16(single_precision_values, converted_half_values, 4) ||
        converted_half_values[0] != 0x3c00u || converted_half_values[1] != 0xc000u ||
        converted_half_values[2] != 0x0001u || converted_half_values[3] != 0x3c00u) {
        berrybits_q4_model_destroy(&student_model);
        berrybits_q4_model_destroy(&loaded_model);
        remove(model_path);
        return 10;
    }

    printf("linear teacher/student RMSE: %.6f -> %.6f\n", rmse_before, rmse_after);
    printf("embedded LUT save/load and FP16 conversion passed\n");
    berrybits_q4_model_destroy(&student_model);
    berrybits_q4_model_destroy(&loaded_model);
    remove(model_path);
    return 0;
}
