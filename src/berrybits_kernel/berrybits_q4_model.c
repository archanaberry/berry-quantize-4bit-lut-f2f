#include "berrybits_q4_model.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const float berrybits_default_lut[BERRYBITS_Q4_LUT_SIZE] = {
    -1.0000000f, -0.8750000f, -0.7500000f, -0.6250000f,
    -0.5000000f, -0.3750000f, -0.2500000f, -0.1250000f,
     0.0000000f,  0.1250000f,  0.2500000f,  0.3750000f,
     0.5000000f,  0.6250000f,  0.7500000f,  1.0000000f
};

static size_t berrybits_packed_row_bytes(uint32_t cols) {
    return ((size_t)cols + 1u) / 2u;
}

static int berrybits_size_multiply(size_t left, size_t right, size_t *result) {
    if (right != 0 && left > SIZE_MAX / right) return 0;
    *result = left * right;
    return 1;
}

static int berrybits_lut_is_sorted(const float *lut) {
    for (size_t i = 0; i < BERRYBITS_Q4_LUT_SIZE; ++i) {
        if (!isfinite(lut[i]) || (i > 0 && lut[i] < lut[i - 1])) return 0;
    }
    return 1;
}

static uint8_t berrybits_lut_nearest_index(const float *lut, float value) {
    size_t low = 0;
    size_t high = BERRYBITS_Q4_LUT_SIZE;
    while (low < high) {
        size_t middle = low + (high - low) / 2;
        if (lut[middle] < value) low = middle + 1;
        else high = middle;
    }
    if (low == 0) return 0;
    if (low == BERRYBITS_Q4_LUT_SIZE) return BERRYBITS_Q4_LUT_SIZE - 1u;

    float lower_error = value - lut[low - 1];
    float upper_error = lut[low] - value;
    return (uint8_t)(lower_error <= upper_error ? low - 1 : low);
}

static int berrybits_write_u32(FILE *file, uint32_t value) {
    uint8_t bytes[4] = {
        (uint8_t)value,
        (uint8_t)(value >> 8),
        (uint8_t)(value >> 16),
        (uint8_t)(value >> 24)
    };
    return fwrite(bytes, sizeof(bytes), 1, file) == 1;
}

static int berrybits_read_u32(FILE *file, uint32_t *value) {
    uint8_t bytes[4];
    if (fread(bytes, sizeof(bytes), 1, file) != 1) return 0;
    *value = (uint32_t)bytes[0] | ((uint32_t)bytes[1] << 8) |
             ((uint32_t)bytes[2] << 16) | ((uint32_t)bytes[3] << 24);
    return 1;
}

static int berrybits_write_f32(FILE *file, float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return berrybits_write_u32(file, bits);
}

static int berrybits_read_f32(FILE *file, float *value) {
    uint32_t bits;
    if (!berrybits_read_u32(file, &bits)) return 0;
    memcpy(value, &bits, sizeof(bits));
    return 1;
}

int berrybits_q4_model_init(berrybits_q4_model_t *model, uint32_t rows, uint32_t cols) {
    if (!model || rows == 0 || cols == 0) return 0;

    size_t blocks_per_row = ((size_t)cols + BERRYBITS_Q4_BLOCK_SIZE - 1u) /
                            BERRYBITS_Q4_BLOCK_SIZE;
    size_t scale_count;
    size_t weight_count;
    if (blocks_per_row > UINT32_MAX ||
        !berrybits_size_multiply(rows, blocks_per_row, &scale_count) ||
        !berrybits_size_multiply(rows, berrybits_packed_row_bytes(cols), &weight_count)) {
        return 0;
    }

    float *scales = (float *)calloc(scale_count, sizeof(float));
    uint8_t *packed_weights = (uint8_t *)calloc(weight_count, sizeof(uint8_t));
    if (!scales || !packed_weights) {
        free(scales);
        free(packed_weights);
        return 0;
    }

    memset(model, 0, sizeof(*model));
    model->rows = rows;
    model->cols = cols;
    model->blocks_per_row = (uint32_t)blocks_per_row;
    memcpy(model->lut, berrybits_default_lut, sizeof(model->lut));
    model->scales = scales;
    model->packed_weights = packed_weights;
    return 1;
}

void berrybits_q4_model_destroy(berrybits_q4_model_t *model) {
    if (!model) return;
    free(model->scales);
    free(model->packed_weights);
    memset(model, 0, sizeof(*model));
}

int berrybits_q4_model_quantize(berrybits_q4_model_t *model, const float *weights) {
    if (!model || !weights || !model->scales || !model->packed_weights ||
        model->rows == 0 || model->cols == 0 || !berrybits_lut_is_sorted(model->lut)) return 0;

    size_t packed_row_bytes = berrybits_packed_row_bytes(model->cols);
    memset(model->packed_weights, 0, (size_t)model->rows * packed_row_bytes);

    for (uint32_t row = 0; row < model->rows; ++row) {
        for (uint32_t block = 0; block < model->blocks_per_row; ++block) {
            size_t block_start = (size_t)block * BERRYBITS_Q4_BLOCK_SIZE;
            size_t row_start = (size_t)row * model->cols;
            size_t count = model->cols - block_start;
            if (count > BERRYBITS_Q4_BLOCK_SIZE) count = BERRYBITS_Q4_BLOCK_SIZE;

            float max_abs = 0.0f;
            for (size_t i = 0; i < count; ++i) {
                float value = weights[row_start + block_start + i];
                if (!isfinite(value)) return 0;
                float magnitude = fabsf(value);
                if (magnitude > max_abs) max_abs = magnitude;
            }

            float scale = max_abs > 0.0f ? max_abs : 1.0f;
            model->scales[(size_t)row * model->blocks_per_row + block] = scale;
            for (size_t i = 0; i < count; ++i) {
                float normalized = weights[row_start + block_start + i] / scale;
                uint8_t best_index = berrybits_lut_nearest_index(model->lut, normalized);

                size_t column = block_start + i;
                size_t packed_index = (size_t)row * packed_row_bytes + column / 2u;
                if ((column & 1u) == 0u) {
                    model->packed_weights[packed_index] = best_index;
                } else {
                    model->packed_weights[packed_index] |= (uint8_t)(best_index << 4);
                }
            }
        }
    }
    return 1;
}

void berrybits_q4_model_gemv(const berrybits_q4_model_t *model, const float *input, float *output) {
    if (!model || !input || !output || !model->packed_weights || !model->scales ||
        model->rows == 0 || model->cols == 0) return;

    size_t packed_row_bytes = berrybits_packed_row_bytes(model->cols);
    for (uint32_t row = 0; row < model->rows; ++row) {
        const uint8_t *packed = model->packed_weights + (size_t)row * packed_row_bytes;
        float accumulators[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (uint32_t block = 0; block < model->blocks_per_row; ++block) {
            uint32_t block_start = block * BERRYBITS_Q4_BLOCK_SIZE;
            uint32_t block_end = block_start + BERRYBITS_Q4_BLOCK_SIZE;
            if (block_end > model->cols) block_end = model->cols;
            float scale = model->scales[(size_t)row * model->blocks_per_row + block];
            float scaled_lut[BERRYBITS_Q4_LUT_SIZE];
            for (size_t index = 0; index < BERRYBITS_Q4_LUT_SIZE; ++index) {
                scaled_lut[index] = model->lut[index] * scale;
            }

            uint32_t column = block_start;
            for (; column + 8 <= block_end; column += 8) {
                size_t byte_index = column / 2u;
                uint8_t byte0 = packed[byte_index];
                uint8_t byte1 = packed[byte_index + 1];
                uint8_t byte2 = packed[byte_index + 2];
                uint8_t byte3 = packed[byte_index + 3];
                accumulators[0] += scaled_lut[byte0 & 0x0Fu] * input[column];
                accumulators[1] += scaled_lut[byte0 >> 4] * input[column + 1];
                accumulators[2] += scaled_lut[byte1 & 0x0Fu] * input[column + 2];
                accumulators[3] += scaled_lut[byte1 >> 4] * input[column + 3];
                accumulators[0] += scaled_lut[byte2 & 0x0Fu] * input[column + 4];
                accumulators[1] += scaled_lut[byte2 >> 4] * input[column + 5];
                accumulators[2] += scaled_lut[byte3 & 0x0Fu] * input[column + 6];
                accumulators[3] += scaled_lut[byte3 >> 4] * input[column + 7];
            }
            for (; column < block_end; ++column) {
                uint8_t byte = packed[column / 2u];
                uint8_t index = (column & 1u) ? (byte >> 4) : (byte & 0x0Fu);
                accumulators[column & 3u] += scaled_lut[index] * input[column];
            }
        }
        output[row] = (accumulators[0] + accumulators[1]) +
                      (accumulators[2] + accumulators[3]);
    }
}

int berrybits_q4_model_save(const berrybits_q4_model_t *model, const char *path) {
    if (!model || !path || !model->scales || !model->packed_weights ||
        model->rows == 0 || model->cols == 0) return 0;

    FILE *file = fopen(path, "wb");
    if (!file) return 0;
    static const uint8_t magic[8] = {'B', '4', 'L', 'U', 'T', 'F', '1', 0};
    int ok = fwrite(magic, sizeof(magic), 1, file) == 1 &&
             berrybits_write_u32(file, BERRYBITS_Q4_MODEL_VERSION) &&
             berrybits_write_u32(file, model->rows) &&
             berrybits_write_u32(file, model->cols) &&
             berrybits_write_u32(file, model->blocks_per_row) &&
             berrybits_write_u32(file, BERRYBITS_Q4_BLOCK_SIZE) &&
             berrybits_write_u32(file, BERRYBITS_Q4_LUT_SIZE);

    for (size_t i = 0; ok && i < BERRYBITS_Q4_LUT_SIZE; ++i) {
        ok = berrybits_write_f32(file, model->lut[i]);
    }

    size_t scale_count = (size_t)model->rows * model->blocks_per_row;
    size_t weight_count = (size_t)model->rows * berrybits_packed_row_bytes(model->cols);
    for (size_t i = 0; ok && i < scale_count; ++i) {
        ok = berrybits_write_f32(file, model->scales[i]);
    }
    if (ok) ok = fwrite(model->packed_weights, weight_count, 1, file) == 1;
    if (fclose(file) != 0) ok = 0;
    return ok;
}

int berrybits_q4_model_load(berrybits_q4_model_t *model, const char *path) {
    if (!model || !path) return 0;
    FILE *file = fopen(path, "rb");
    if (!file) return 0;

    uint8_t magic[8];
    uint32_t version, rows, cols, blocks_per_row, block_size, lut_size;
    int ok = fread(magic, sizeof(magic), 1, file) == 1 &&
             memcmp(magic, "B4LUTF1", 7) == 0 &&
             berrybits_read_u32(file, &version) &&
             berrybits_read_u32(file, &rows) &&
             berrybits_read_u32(file, &cols) &&
             berrybits_read_u32(file, &blocks_per_row) &&
             berrybits_read_u32(file, &block_size) &&
             berrybits_read_u32(file, &lut_size);
    if (!ok || version != BERRYBITS_Q4_MODEL_VERSION || rows == 0 || cols == 0 ||
        block_size != BERRYBITS_Q4_BLOCK_SIZE || lut_size != BERRYBITS_Q4_LUT_SIZE ||
        blocks_per_row != ((size_t)cols + BERRYBITS_Q4_BLOCK_SIZE - 1u) / BERRYBITS_Q4_BLOCK_SIZE) {
        fclose(file);
        return 0;
    }

    berrybits_q4_model_t loaded = {0};
    if (!berrybits_q4_model_init(&loaded, rows, cols)) {
        fclose(file);
        return 0;
    }
    for (size_t i = 0; ok && i < BERRYBITS_Q4_LUT_SIZE; ++i) {
        ok = berrybits_read_f32(file, &loaded.lut[i]) && isfinite(loaded.lut[i]);
    }
    size_t scale_count = (size_t)rows * loaded.blocks_per_row;
    size_t weight_count = (size_t)rows * berrybits_packed_row_bytes(cols);
    for (size_t i = 0; ok && i < scale_count; ++i) {
        ok = berrybits_read_f32(file, &loaded.scales[i]) &&
             isfinite(loaded.scales[i]) && loaded.scales[i] > 0.0f;
    }
    if (ok) ok = fread(loaded.packed_weights, weight_count, 1, file) == 1;
    if (fclose(file) != 0) ok = 0;
    if (!ok) {
        berrybits_q4_model_destroy(&loaded);
        return 0;
    }

    *model = loaded;
    return 1;
}

float berrybits_fp16_to_fp32(uint16_t value) {
    uint32_t sign = (uint32_t)(value & 0x8000u) << 16;
    uint32_t exponent = (value >> 10) & 0x1Fu;
    uint32_t mantissa = value & 0x03FFu;
    uint32_t bits;

    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            int adjusted_exponent = -14;
            while ((mantissa & 0x0400u) == 0) {
                mantissa <<= 1;
                --adjusted_exponent;
            }
            mantissa &= 0x03FFu;
            bits = sign | ((uint32_t)(adjusted_exponent + 127) << 23) | (mantissa << 13);
        }
    } else if (exponent == 0x1Fu) {
        bits = sign | 0x7F800000u | (mantissa << 13);
    } else {
        bits = sign | ((exponent + 112u) << 23) | (mantissa << 13);
    }

    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

int berrybits_fp16_array_to_fp32(const uint16_t *source, float *destination, size_t count) {
    if (!source || !destination || count == 0) return 0;
    for (size_t i = 0; i < count; ++i) {
        destination[i] = berrybits_fp16_to_fp32(source[i]);
    }
    return 1;
}

uint16_t berrybits_fp32_to_fp16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    uint16_t sign = (uint16_t)((bits >> 16) & 0x8000u);
    uint32_t exponent = (bits >> 23) & 0xFFu;
    uint32_t mantissa = bits & 0x7FFFFFu;

    if (exponent == 0xFFu) {
        if (mantissa == 0) return (uint16_t)(sign | 0x7C00u);
        uint16_t payload = (uint16_t)(mantissa >> 13);
        if (payload == 0) payload = 1;
        return (uint16_t)(sign | 0x7C00u | payload);
    }

    int half_exponent = (int)exponent - 127 + 15;
    if (half_exponent >= 31) return (uint16_t)(sign | 0x7C00u);
    if (half_exponent <= 0) {
        if (half_exponent < -10) return sign;
        mantissa |= 0x800000u;
        uint32_t shift = (uint32_t)(14 - half_exponent);
        uint32_t rounded = mantissa >> shift;
        uint32_t remainder = mantissa & ((1u << shift) - 1u);
        uint32_t halfway = 1u << (shift - 1u);
        if (remainder > halfway || (remainder == halfway && (rounded & 1u))) ++rounded;
        return (uint16_t)(sign | rounded);
    }

    uint32_t rounded_mantissa = mantissa >> 13;
    uint32_t remainder = mantissa & 0x1FFFu;
    if (remainder > 0x1000u || (remainder == 0x1000u && (rounded_mantissa & 1u))) {
        ++rounded_mantissa;
        if (rounded_mantissa == 0x400u) {
            rounded_mantissa = 0;
            ++half_exponent;
            if (half_exponent >= 31) return (uint16_t)(sign | 0x7C00u);
        }
    }
    return (uint16_t)(sign | ((uint16_t)half_exponent << 10) | (uint16_t)rounded_mantissa);
}

int berrybits_fp32_array_to_fp16(const float *source, uint16_t *destination, size_t count) {
    if (!source || !destination || count == 0) return 0;
    for (size_t i = 0; i < count; ++i) {
        destination[i] = berrybits_fp32_to_fp16(source[i]);
    }
    return 1;
}
