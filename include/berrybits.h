#ifndef BERRYBITS_H
#define BERRYBITS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define BERRYBITS_BLUT_VERSION 1u
#define BERRYBITS_MAX_LUT_ENTRIES 256u

typedef enum {
    BERRYBITS_MODE_Q64 = 1,
    BERRYBITS_MODE_Q32 = 2,
    BERRYBITS_MODE_Q16 = 3,
    BERRYBITS_MODE_Q8 = 8,
    BERRYBITS_MODE_Q6 = 9,
    BERRYBITS_MODE_Q5 = 10,
    BERRYBITS_MODE_Q4_K_S = 11,
    BERRYBITS_MODE_Q4_K_M = 12,
    BERRYBITS_MODE_Q3 = 13,
    BERRYBITS_MODE_Q2 = 14,
    BERRYBITS_MODE_I2_S = 15,
    BERRYBITS_MODE_Q1 = 16,
    BERRYBITS_MODE_TERNARY = 17
} berrybits_mode_t;

typedef struct {
    uint32_t rows;
    uint32_t cols;
    uint32_t block_size;
    uint32_t blocks_per_row;
    uint32_t bits_per_value;
    uint32_t lut_entries;
    berrybits_mode_t mode;
    float lut[BERRYBITS_MAX_LUT_ENTRIES];
    float *scales;
    float *offsets;
    uint8_t *payload;
    size_t payload_bytes;
} berrybits_tensor_t;

const char *berrybits_mode_name(berrybits_mode_t mode);
int berrybits_mode_info(berrybits_mode_t mode, uint32_t *bits_per_value,
                        uint32_t *default_block_size);
int berrybits_tensor_init(berrybits_tensor_t *tensor, uint32_t rows,
                          uint32_t cols, berrybits_mode_t mode);
void berrybits_tensor_destroy(berrybits_tensor_t *tensor);
int berrybits_tensor_quantize_f32(berrybits_tensor_t *tensor,
                                  const float *weights);
int berrybits_tensor_quantize_f64(berrybits_tensor_t *tensor,
                                  const double *weights);
int berrybits_tensor_dequantize_f32(const berrybits_tensor_t *tensor,
                                    float *weights);
int berrybits_tensor_gemv(const berrybits_tensor_t *tensor,
                          const float *input, float *output);
int berrybits_tensor_save_blut(const berrybits_tensor_t *tensor,
                               const char *path);
int berrybits_tensor_load_blut(berrybits_tensor_t *tensor,
                               const char *path);

typedef struct berrybits_handle berrybits_handle_t;

berrybits_handle_t *berrybits_create(uint32_t rows, uint32_t cols,
                                     berrybits_mode_t mode);
berrybits_handle_t *berrybits_load(const char *path);
void berrybits_destroy(berrybits_handle_t *handle);
uint32_t berrybits_rows(const berrybits_handle_t *handle);
uint32_t berrybits_cols(const berrybits_handle_t *handle);
berrybits_mode_t berrybits_mode(const berrybits_handle_t *handle);
int berrybits_quantize_f32(berrybits_handle_t *handle, const float *weights);
int berrybits_quantize_f64(berrybits_handle_t *handle, const double *weights);
int berrybits_dequantize_f32(const berrybits_handle_t *handle, float *weights);
int berrybits_gemv(const berrybits_handle_t *handle, const float *input,
                   float *output);
int berrybits_save(berrybits_handle_t *handle, const char *path);

typedef struct berrybits_handle berrybits_handle_t;

berrybits_handle_t *berrybits_create(uint32_t rows, uint32_t cols,
                                     berrybits_mode_t mode);
berrybits_handle_t *berrybits_load(const char *path);
void berrybits_destroy(berrybits_handle_t *handle);
uint32_t berrybits_rows(const berrybits_handle_t *handle);
uint32_t berrybits_cols(const berrybits_handle_t *handle);
berrybits_mode_t berrybits_mode(const berrybits_handle_t *handle);
int berrybits_quantize_f32(berrybits_handle_t *handle, const float *weights);
int berrybits_quantize_f64(berrybits_handle_t *handle, const double *weights);
int berrybits_dequantize_f32(const berrybits_handle_t *handle, float *weights);
int berrybits_gemv(const berrybits_handle_t *handle, const float *input,
                   float *output);
int berrybits_save(berrybits_handle_t *handle, const char *path);

#ifdef __cplusplus
}
#endif

#endif
