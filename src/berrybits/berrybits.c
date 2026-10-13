#include "berrybits.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BERRYBITS_RAW_F64 1u
#define BERRYBITS_RAW_F32 2u
#define BERRYBITS_RAW_F16 3u
#define BERRYBITS_MODE_COUNT 13u

static int berrybits_size_mul(size_t left, size_t right, size_t *result) {
    if (right != 0 && left > SIZE_MAX / right) return 0;
    *result = left * right;
    return 1;
}

static uint32_t berrybits_raw_kind(berrybits_mode_t mode) {
    switch (mode) {
        case BERRYBITS_MODE_Q64: return BERRYBITS_RAW_F64;
        case BERRYBITS_MODE_Q32: return BERRYBITS_RAW_F32;
        case BERRYBITS_MODE_Q16: return BERRYBITS_RAW_F16;
        default: return 0;
    }
}

const char *berrybits_mode_name(berrybits_mode_t mode) {
    switch (mode) {
        case BERRYBITS_MODE_Q64: return "Q64_LUT_F2F";
        case BERRYBITS_MODE_Q32: return "Q32_LUT_F2F";
        case BERRYBITS_MODE_Q16: return "Q16_LUT_F2F";
        case BERRYBITS_MODE_Q8: return "Q8_LUT_F2F";
        case BERRYBITS_MODE_Q6: return "Q6_LUT_F2F";
        case BERRYBITS_MODE_Q5: return "Q5_LUT_F2F";
        case BERRYBITS_MODE_Q4_K_S: return "Q4_K_S_LUT_F2F";
        case BERRYBITS_MODE_Q4_K_M: return "Q4_K_M_LUT_F2F";
        case BERRYBITS_MODE_Q3: return "Q3_LUT_F2F";
        case BERRYBITS_MODE_Q2: return "Q2_LUT_F2F";
        case BERRYBITS_MODE_I2_S: return "I2_S_LUT_F2F";
        case BERRYBITS_MODE_Q1: return "Q1_LUT_F2F";
        case BERRYBITS_MODE_TERNARY: return "TERNARY_LUT_F2F";
        default: return "UNKNOWN";
    }
}

int berrybits_mode_info(berrybits_mode_t mode, uint32_t *bits_per_value,
                        uint32_t *default_block_size) {
    if (!bits_per_value || !default_block_size) return 0;
    switch (mode) {
        case BERRYBITS_MODE_Q64: *bits_per_value = 64; *default_block_size = 1; return 1;
        case BERRYBITS_MODE_Q32: *bits_per_value = 32; *default_block_size = 1; return 1;
        case BERRYBITS_MODE_Q16: *bits_per_value = 16; *default_block_size = 1; return 1;
        case BERRYBITS_MODE_Q8: *bits_per_value = 8; *default_block_size = 64; return 1;
        case BERRYBITS_MODE_Q6: *bits_per_value = 6; *default_block_size = 64; return 1;
        case BERRYBITS_MODE_Q5: *bits_per_value = 5; *default_block_size = 64; return 1;
        case BERRYBITS_MODE_Q4_K_S: *bits_per_value = 4; *default_block_size = 256; return 1;
        case BERRYBITS_MODE_Q4_K_M: *bits_per_value = 4; *default_block_size = 32; return 1;
        case BERRYBITS_MODE_Q3: *bits_per_value = 3; *default_block_size = 64; return 1;
        case BERRYBITS_MODE_Q2: *bits_per_value = 2; *default_block_size = 64; return 1;
        case BERRYBITS_MODE_I2_S: *bits_per_value = 2; *default_block_size = 256; return 1;
        case BERRYBITS_MODE_Q1: *bits_per_value = 1; *default_block_size = 256; return 1;
        case BERRYBITS_MODE_TERNARY: *bits_per_value = 2; *default_block_size = 256; return 1;
        default: return 0;
    }
}

static uint32_t berrybits_lut_entries(berrybits_mode_t mode, uint32_t bits) {
    if (mode == BERRYBITS_MODE_I2_S) return 4;
    if (mode == BERRYBITS_MODE_TERNARY) return 3;
    if (berrybits_raw_kind(mode)) return 0;
    return 1u << bits;
}

static void berrybits_init_lut(berrybits_tensor_t *tensor) {
    if (tensor->mode == BERRYBITS_MODE_I2_S) {
        tensor->lut[0] = -1.0f;
        tensor->lut[1] = -0.25f;
        tensor->lut[2] = 0.25f;
        tensor->lut[3] = 1.0f;
    } else if (tensor->mode == BERRYBITS_MODE_TERNARY) {
        tensor->lut[0] = -1.0f;
        tensor->lut[1] = 0.0f;
        tensor->lut[2] = 1.0f;
    } else if (tensor->lut_entries > 1) {
        for (uint32_t index = 0; index < tensor->lut_entries; ++index) {
            tensor->lut[index] = (float)index / (float)(tensor->lut_entries - 1u);
        }
    }
}

static int berrybits_is_custom_codebook(berrybits_mode_t mode) {
    return mode == BERRYBITS_MODE_I2_S || mode == BERRYBITS_MODE_TERNARY;
}

static uint64_t berrybits_round_nonnegative(double value) {
    double lower = floor(value);
    double fraction = value - lower;
    if (fraction > 0.5 || (fraction == 0.5 && fmod(lower, 2.0) != 0.0)) lower += 1.0;
    return (uint64_t)lower;
}

static int berrybits_store_code(uint8_t *payload, size_t index,
                                uint32_t bits, uint32_t code) {
    size_t first_bit;
    if (!berrybits_size_mul(index, bits, &first_bit)) return 0;
    for (uint32_t bit = 0; bit < bits; ++bit) {
        size_t target = first_bit + bit;
        uint8_t mask = (uint8_t)(1u << (target & 7u));
        if ((code >> bit) & 1u) payload[target >> 3] |= mask;
        else payload[target >> 3] &= (uint8_t)~mask;
    }
    return 1;
}

static int berrybits_load_code(const uint8_t *payload, size_t index,
                               uint32_t bits, uint32_t *code) {
    size_t first_bit;
    if (!payload || !code || !berrybits_size_mul(index, bits, &first_bit)) return 0;
    uint32_t value = 0;
    for (uint32_t bit = 0; bit < bits; ++bit) {
        size_t source = first_bit + bit;
        value |= (uint32_t)((payload[source >> 3] >> (source & 7u)) & 1u) << bit;
    }
    *code = value;
    return 1;
}

static uint16_t berrybits_f32_to_f16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    uint16_t sign = (uint16_t)((bits >> 16) & 0x8000u);
    uint32_t exponent = (bits >> 23) & 0xFFu;
    uint32_t mantissa = bits & 0x7FFFFFu;
    if (exponent == 0xFFu) {
        uint16_t payload = (uint16_t)(mantissa >> 13);
        if (mantissa && payload == 0) payload = 1;
        return (uint16_t)(sign | 0x7C00u | payload);
    }
    int half_exponent = (int)exponent - 112;
    if (half_exponent >= 31) return (uint16_t)(sign | 0x7C00u);
    if (half_exponent <= 0) {
        if (half_exponent < -10) return sign;
        mantissa |= 0x800000u;
        uint32_t shift = (uint32_t)(14 - half_exponent);
        uint32_t rounded = (uint32_t)berrybits_round_nonnegative(
            (double)mantissa / (double)(1u << shift));
        return (uint16_t)(sign | rounded);
    }
    uint32_t rounded = (uint32_t)berrybits_round_nonnegative((double)mantissa / 8192.0);
    if (rounded == 0x400u) {
        rounded = 0;
        ++half_exponent;
        if (half_exponent >= 31) return (uint16_t)(sign | 0x7C00u);
    }
    return (uint16_t)(sign | ((uint16_t)half_exponent << 10) | (uint16_t)rounded);
}

static float berrybits_f16_to_f32(uint16_t value) {
    uint32_t sign = (uint32_t)(value & 0x8000u) << 16;
    uint32_t exponent = (value >> 10) & 0x1Fu;
    uint32_t mantissa = value & 0x03FFu;
    uint32_t bits;
    if (exponent == 0) {
        if (!mantissa) bits = sign;
        else {
            int adjusted = -14;
            while ((mantissa & 0x0400u) == 0) { mantissa <<= 1; --adjusted; }
            mantissa &= 0x03FFu;
            bits = sign | ((uint32_t)(adjusted + 127) << 23) | (mantissa << 13);
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

static void berrybits_store_u16(uint8_t *destination, uint16_t value) {
    destination[0] = (uint8_t)value;
    destination[1] = (uint8_t)(value >> 8);
}

static uint16_t berrybits_load_u16(const uint8_t *source) {
    return (uint16_t)source[0] | ((uint16_t)source[1] << 8);
}

static void berrybits_store_u32(uint8_t *destination, uint32_t value) {
    for (uint32_t i = 0; i < 4; ++i) destination[i] = (uint8_t)(value >> (i * 8));
}

static uint32_t berrybits_load_u32(const uint8_t *source) {
    uint32_t value = 0;
    for (uint32_t i = 0; i < 4; ++i) value |= (uint32_t)source[i] << (i * 8);
    return value;
}

static void berrybits_store_u64(uint8_t *destination, uint64_t value) {
    for (uint32_t i = 0; i < 8; ++i) destination[i] = (uint8_t)(value >> (i * 8));
}

static uint64_t berrybits_load_u64(const uint8_t *source) {
    uint64_t value = 0;
    for (uint32_t i = 0; i < 8; ++i) value |= (uint64_t)source[i] << (i * 8);
    return value;
}

static void berrybits_store_f32(uint8_t *destination, float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    berrybits_store_u32(destination, bits);
}

static float berrybits_load_f32(const uint8_t *source) {
    uint32_t bits = berrybits_load_u32(source);
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

int berrybits_tensor_init(berrybits_tensor_t *tensor, uint32_t rows,
                          uint32_t cols, berrybits_mode_t mode) {
    if (!tensor || !rows || !cols) return 0;
    uint32_t bits, default_block;
    if (!berrybits_mode_info(mode, &bits, &default_block)) return 0;

    size_t value_count;
    if (!berrybits_size_mul(rows, cols, &value_count)) return 0;
    uint32_t raw_kind = berrybits_raw_kind(mode);
    size_t payload_bytes;
    size_t scale_count = 0;
    uint32_t blocks_per_row = 0;
    uint32_t lut_entries = berrybits_lut_entries(mode, bits);
    if (raw_kind) {
        if (!berrybits_size_mul(value_count, bits / 8u, &payload_bytes)) return 0;
    } else {
        blocks_per_row = (uint32_t)(((size_t)cols + default_block - 1u) / default_block);
        size_t total_bits;
        if (!berrybits_size_mul(value_count, bits, &total_bits) || total_bits > SIZE_MAX - 7u) return 0;
        payload_bytes = (total_bits + 7u) / 8u;
        if (!berrybits_size_mul(rows, blocks_per_row, &scale_count)) return 0;
    }

    uint8_t *payload = (uint8_t *)calloc(payload_bytes, 1);
    float *scales = scale_count ? (float *)calloc(scale_count, sizeof(float)) : NULL;
    float *offsets = scale_count ? (float *)calloc(scale_count, sizeof(float)) : NULL;
    if (!payload || (scale_count && (!scales || !offsets))) {
        free(payload);
        free(scales);
        free(offsets);
        return 0;
    }

    memset(tensor, 0, sizeof(*tensor));
    tensor->rows = rows;
    tensor->cols = cols;
    tensor->block_size = raw_kind ? 1u : default_block;
    tensor->blocks_per_row = blocks_per_row;
    tensor->bits_per_value = bits;
    tensor->lut_entries = lut_entries;
    tensor->mode = mode;
    tensor->scales = scales;
    tensor->offsets = offsets;
    tensor->payload = payload;
    tensor->payload_bytes = payload_bytes;
    berrybits_init_lut(tensor);
    return 1;
}

void berrybits_tensor_destroy(berrybits_tensor_t *tensor) {
    if (!tensor) return;
    free(tensor->scales);
    free(tensor->offsets);
    free(tensor->payload);
    memset(tensor, 0, sizeof(*tensor));
}

static int berrybits_quantize(berrybits_tensor_t *tensor, const float *weights) {
    if (!tensor || !weights || !tensor->payload) return 0;
    size_t value_count;
    if (!berrybits_size_mul(tensor->rows, tensor->cols, &value_count)) return 0;
    uint32_t raw_kind = berrybits_raw_kind(tensor->mode);
    if (raw_kind) {
        for (size_t i = 0; i < value_count; ++i) {
            float value = weights[i];
            if (!isfinite(value)) return 0;
            if (raw_kind == BERRYBITS_RAW_F64) {
                double widened = value;
                uint64_t bits;
                memcpy(&bits, &widened, sizeof(bits));
                berrybits_store_u64(tensor->payload + i * 8u, bits);
            } else if (raw_kind == BERRYBITS_RAW_F32) {
                berrybits_store_f32(tensor->payload + i * 4u, value);
            } else {
                berrybits_store_u16(tensor->payload + i * 2u, berrybits_f32_to_f16(value));
            }
        }
        return 1;
    }

    memset(tensor->payload, 0, tensor->payload_bytes);
    const uint32_t maximum_code = (1u << tensor->bits_per_value) - 1u;
    for (uint32_t row = 0; row < tensor->rows; ++row) {
        for (uint32_t block = 0; block < tensor->blocks_per_row; ++block) {
            size_t start = (size_t)row * tensor->cols + (size_t)block * tensor->block_size;
            size_t count = tensor->cols - (size_t)block * tensor->block_size;
            if (count > tensor->block_size) count = tensor->block_size;
            float minimum = weights[start];
            float maximum = weights[start];
            float max_abs = fabsf(weights[start]);
            for (size_t i = 0; i < count; ++i) {
                float value = weights[start + i];
                if (!isfinite(value)) return 0;
                if (value < minimum) minimum = value;
                if (value > maximum) maximum = value;
                if (fabsf(value) > max_abs) max_abs = fabsf(value);
            }

            size_t scale_index = (size_t)row * tensor->blocks_per_row + block;
            int nonlinear = berrybits_is_custom_codebook(tensor->mode);
            float scale = nonlinear ? (max_abs > 0.0f ? max_abs : 1.0f)
                                    : (maximum > minimum ? maximum - minimum : 1.0f);
            float offset = nonlinear ? 0.0f : minimum;
            tensor->scales[scale_index] = scale;
            tensor->offsets[scale_index] = offset;

            for (size_t i = 0; i < count; ++i) {
                float normalized = nonlinear ? weights[start + i] / scale
                                             : (weights[start + i] - offset) / scale;
                uint32_t best_code = 0;
                if (!nonlinear) {
                    uint64_t rounded = berrybits_round_nonnegative(
                        (double)normalized * maximum_code);
                    if (rounded > maximum_code) rounded = maximum_code;
                    best_code = (uint32_t)rounded;
                } else {
                    float best_error = INFINITY;
                    for (uint32_t code = 0; code < tensor->lut_entries; ++code) {
                        float difference = normalized - tensor->lut[code];
                        float error = difference * difference;
                        if (error < best_error) {
                            best_error = error;
                            best_code = code;
                        }
                    }
                }
                if (!berrybits_store_code(tensor->payload, start + i,
                                          tensor->bits_per_value, best_code)) return 0;
            }
        }
    }
    return 1;
}

int berrybits_tensor_quantize_f32(berrybits_tensor_t *tensor,
                                  const float *weights) {
    return berrybits_quantize(tensor, weights);
}

int berrybits_tensor_quantize_f64(berrybits_tensor_t *tensor,
                                  const double *weights) {
    if (!tensor || !weights) return 0;
    if (tensor->mode == BERRYBITS_MODE_Q64) {
        size_t count;
        size_t expected_bytes;
        if (!berrybits_size_mul(tensor->rows, tensor->cols, &count) ||
            !berrybits_size_mul(count, sizeof(double), &expected_bytes) ||
            !tensor->payload || tensor->payload_bytes != expected_bytes) return 0;
        for (size_t i = 0; i < count; ++i) {
            if (!isfinite(weights[i])) return 0;
            uint64_t bits;
            memcpy(&bits, &weights[i], sizeof(bits));
            berrybits_store_u64(tensor->payload + i * 8u, bits);
        }
        return 1;
    }

    size_t count;
    if (!berrybits_size_mul(tensor->rows, tensor->cols, &count) ||
        count > SIZE_MAX / sizeof(float)) return 0;
    float *converted = (float *)malloc(count * sizeof(float));
    if (!converted) return 0;
    for (size_t i = 0; i < count; ++i) {
        if (!isfinite(weights[i])) { free(converted); return 0; }
        converted[i] = (float)weights[i];
    }
    int result = berrybits_quantize(tensor, converted);
    free(converted);
    return result;
}

static float berrybits_value(const berrybits_tensor_t *tensor, size_t index) {
    uint32_t raw_kind = berrybits_raw_kind(tensor->mode);
    if (raw_kind == BERRYBITS_RAW_F64) {
        uint64_t bits = berrybits_load_u64(tensor->payload + index * 8u);
        double value;
        memcpy(&value, &bits, sizeof(value));
        return (float)value;
    }
    if (raw_kind == BERRYBITS_RAW_F32) return berrybits_load_f32(tensor->payload + index * 4u);
    if (raw_kind == BERRYBITS_RAW_F16) return berrybits_f16_to_f32(berrybits_load_u16(tensor->payload + index * 2u));

    uint32_t code;
    if (!berrybits_load_code(tensor->payload, index, tensor->bits_per_value, &code)) return 0.0f;
    uint32_t row = (uint32_t)(index / tensor->cols);
    uint32_t col = (uint32_t)(index % tensor->cols);
    uint32_t block = col / tensor->block_size;
    size_t scale_index = (size_t)row * tensor->blocks_per_row + block;
    return tensor->offsets[scale_index] + tensor->lut[code] * tensor->scales[scale_index];
}

int berrybits_tensor_dequantize_f32(const berrybits_tensor_t *tensor,
                                    float *weights) {
    if (!tensor || !weights || !tensor->payload) return 0;
    size_t count;
    if (!berrybits_size_mul(tensor->rows, tensor->cols, &count)) return 0;
    for (size_t i = 0; i < count; ++i) weights[i] = berrybits_value(tensor, i);
    return 1;
}

int berrybits_tensor_gemv(const berrybits_tensor_t *tensor,
                          const float *input, float *output) {
    if (!tensor || !input || !output || !tensor->payload) return 0;
    for (uint32_t row = 0; row < tensor->rows; ++row) {
        double sum = 0.0;
        size_t start = (size_t)row * tensor->cols;
        for (uint32_t col = 0; col < tensor->cols; ++col) {
            sum += (double)berrybits_value(tensor, start + col) * input[col];
        }
        output[row] = (float)sum;
    }
    return 1;
}

static int berrybits_write_u32(FILE *file, uint32_t value) {
    uint8_t bytes[4];
    berrybits_store_u32(bytes, value);
    return fwrite(bytes, sizeof(bytes), 1, file) == 1;
}

static int berrybits_read_u32(FILE *file, uint32_t *value) {
    uint8_t bytes[4];
    if (fread(bytes, sizeof(bytes), 1, file) != 1) return 0;
    *value = berrybits_load_u32(bytes);
    return 1;
}

static int berrybits_write_u64(FILE *file, uint64_t value) {
    uint8_t bytes[8];
    berrybits_store_u64(bytes, value);
    return fwrite(bytes, sizeof(bytes), 1, file) == 1;
}

static int berrybits_read_u64(FILE *file, uint64_t *value) {
    uint8_t bytes[8];
    if (fread(bytes, sizeof(bytes), 1, file) != 1) return 0;
    *value = berrybits_load_u64(bytes);
    return 1;
}

int berrybits_tensor_save_blut(const berrybits_tensor_t *tensor,
                               const char *path) {
    if (!tensor || !path || !tensor->payload) return 0;
    FILE *file = fopen(path, "wb");
    if (!file) return 0;
    static const uint8_t magic[8] = {'B', 'L', 'U', 'T', 1, 0, 0, 0};
    int ok = fwrite(magic, sizeof(magic), 1, file) == 1 &&
             berrybits_write_u32(file, BERRYBITS_BLUT_VERSION) &&
             berrybits_write_u32(file, (uint32_t)tensor->mode) &&
             berrybits_write_u32(file, tensor->rows) &&
             berrybits_write_u32(file, tensor->cols) &&
             berrybits_write_u32(file, tensor->block_size) &&
             berrybits_write_u32(file, tensor->blocks_per_row) &&
             berrybits_write_u32(file, tensor->bits_per_value) &&
             berrybits_write_u32(file, tensor->lut_entries) &&
             berrybits_write_u64(file, tensor->payload_bytes);

    for (uint32_t i = 0; ok && i < tensor->lut_entries; ++i) {
        uint32_t bits;
        memcpy(&bits, &tensor->lut[i], sizeof(bits));
        ok = berrybits_write_u32(file, bits);
    }
    size_t scale_count = (size_t)tensor->rows * tensor->blocks_per_row;
    for (size_t i = 0; ok && i < scale_count; ++i) {
        uint32_t bits;
        memcpy(&bits, &tensor->scales[i], sizeof(bits));
        ok = berrybits_write_u32(file, bits);
    }
    for (size_t i = 0; ok && i < scale_count; ++i) {
        uint32_t bits;
        memcpy(&bits, &tensor->offsets[i], sizeof(bits));
        ok = berrybits_write_u32(file, bits);
    }
    if (ok) ok = fwrite(tensor->payload, tensor->payload_bytes, 1, file) == 1;
    if (fclose(file) != 0) ok = 0;
    return ok;
}

int berrybits_tensor_load_blut(berrybits_tensor_t *tensor,
                               const char *path) {
    if (!tensor || !path) return 0;
    FILE *file = fopen(path, "rb");
    if (!file) return 0;
    uint8_t magic[8];
    uint32_t version, mode_value, rows, cols, block_size, blocks_per_row, bits, lut_entries;
    uint64_t payload_bytes;
    int ok = fread(magic, sizeof(magic), 1, file) == 1 &&
             memcmp(magic, "BLUT\1\0\0\0", sizeof(magic)) == 0 &&
             berrybits_read_u32(file, &version) &&
             berrybits_read_u32(file, &mode_value) &&
             berrybits_read_u32(file, &rows) &&
             berrybits_read_u32(file, &cols) &&
             berrybits_read_u32(file, &block_size) &&
             berrybits_read_u32(file, &blocks_per_row) &&
             berrybits_read_u32(file, &bits) &&
             berrybits_read_u32(file, &lut_entries) &&
             berrybits_read_u64(file, &payload_bytes);
    if (!ok || version != BERRYBITS_BLUT_VERSION || !rows || !cols ||
        payload_bytes > SIZE_MAX) {
        fclose(file);
        return 0;
    }

    berrybits_tensor_t loaded = {0};
    if (!berrybits_tensor_init(&loaded, rows, cols, (berrybits_mode_t)mode_value) ||
        loaded.block_size != block_size || loaded.blocks_per_row != blocks_per_row ||
        loaded.bits_per_value != bits || loaded.lut_entries != lut_entries ||
        loaded.payload_bytes != (size_t)payload_bytes) {
        berrybits_tensor_destroy(&loaded);
        fclose(file);
        return 0;
    }

    for (uint32_t i = 0; ok && i < lut_entries; ++i) {
        uint32_t raw;
        ok = berrybits_read_u32(file, &raw);
        if (ok) memcpy(&loaded.lut[i], &raw, sizeof(raw));
        if (ok && !isfinite(loaded.lut[i])) ok = 0;
    }
    size_t scale_count = (size_t)rows * blocks_per_row;
    for (size_t i = 0; ok && i < scale_count; ++i) {
        uint32_t raw;
        ok = berrybits_read_u32(file, &raw);
        if (ok) memcpy(&loaded.scales[i], &raw, sizeof(raw));
        if (ok && (!isfinite(loaded.scales[i]) || loaded.scales[i] < 0.0f)) ok = 0;
    }
    for (size_t i = 0; ok && i < scale_count; ++i) {
        uint32_t raw;
        ok = berrybits_read_u32(file, &raw);
        if (ok) memcpy(&loaded.offsets[i], &raw, sizeof(raw));
        if (ok && !isfinite(loaded.offsets[i])) ok = 0;
    }
    if (ok) ok = fread(loaded.payload, loaded.payload_bytes, 1, file) == 1;
    if (fclose(file) != 0) ok = 0;
    if (!ok) {
        berrybits_tensor_destroy(&loaded);
        return 0;
    }
    *tensor = loaded;
    return 1;
}

struct berrybits_handle {
    berrybits_tensor_t tensor;
};

berrybits_handle_t *berrybits_create(uint32_t rows, uint32_t cols,
                                     berrybits_mode_t mode) {
    berrybits_handle_t *handle = (berrybits_handle_t *)calloc(1, sizeof(*handle));
    if (!handle) return NULL;
    if (!berrybits_tensor_init(&handle->tensor, rows, cols, mode)) {
        free(handle);
        return NULL;
    }
    return handle;
}

berrybits_handle_t *berrybits_load(const char *path) {
    if (!path) return NULL;
    berrybits_handle_t *handle = (berrybits_handle_t *)calloc(1, sizeof(*handle));
    if (!handle) return NULL;
    if (!berrybits_tensor_load_blut(&handle->tensor, path)) {
        free(handle);
        return NULL;
    }
    return handle;
}

void berrybits_destroy(berrybits_handle_t *handle) {
    if (!handle) return;
    berrybits_tensor_destroy(&handle->tensor);
    free(handle);
}

uint32_t berrybits_rows(const berrybits_handle_t *handle) {
    return handle ? handle->tensor.rows : 0;
}

uint32_t berrybits_cols(const berrybits_handle_t *handle) {
    return handle ? handle->tensor.cols : 0;
}

berrybits_mode_t berrybits_mode(const berrybits_handle_t *handle) {
    return handle ? handle->tensor.mode : 0;
}

int berrybits_quantize_f32(berrybits_handle_t *handle, const float *weights) {
    return handle ? berrybits_tensor_quantize_f32(&handle->tensor, weights) : 0;
}

int berrybits_quantize_f64(berrybits_handle_t *handle, const double *weights) {
    return handle ? berrybits_tensor_quantize_f64(&handle->tensor, weights) : 0;
}

int berrybits_dequantize_f32(const berrybits_handle_t *handle, float *weights) {
    return handle ? berrybits_tensor_dequantize_f32(&handle->tensor, weights) : 0;
}

int berrybits_gemv(const berrybits_handle_t *handle, const float *input,
                   float *output) {
    return handle ? berrybits_tensor_gemv(&handle->tensor, input, output) : 0;
}

int berrybits_save(berrybits_handle_t *handle, const char *path) {
    return handle ? berrybits_tensor_save_blut(&handle->tensor, path) : 0;
}
