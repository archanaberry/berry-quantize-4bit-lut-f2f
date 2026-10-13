module berrybits;

extern fn void* berrybits_create(uint32 rows, uint32 cols, int mode);
extern fn void* berrybits_load(char* path);
extern fn void berrybits_destroy(void* handle);
extern fn uint32 berrybits_rows(void* handle);
extern fn uint32 berrybits_cols(void* handle);
extern fn int berrybits_mode(void* handle);
extern fn int berrybits_quantize_f32(void* handle, float* weights);
extern fn int berrybits_quantize_f64(void* handle, double* weights);
extern fn int berrybits_dequantize_f32(void* handle, float* weights);
extern fn int berrybits_gemv(void* handle, float* input, float* output);
extern fn int berrybits_save(void* handle, char* path);

const int MODE_Q64 = 1;
const int MODE_Q32 = 2;
const int MODE_Q16 = 3;
const int MODE_Q8 = 8;
const int MODE_Q6 = 9;
const int MODE_Q5 = 10;
const int MODE_Q4_K_S = 11;
const int MODE_Q4_K_M = 12;
const int MODE_Q3 = 13;
const int MODE_Q2 = 14;
const int MODE_I2_S = 15;
const int MODE_Q1 = 16;
const int MODE_TERNARY = 17;
