module berrybits;

// Link against libberrybits; this module binds the opaque native C ABI.
extern fn void* berrybits_model_create(uint32 rows, uint32 cols);
extern fn void* berrybits_model_load(char* path);
extern fn void berrybits_model_destroy(void* handle);
extern fn uint32 berrybits_model_rows(void* handle);
extern fn uint32 berrybits_model_cols(void* handle);
extern fn int berrybits_model_quantize(void* handle, float* weights);
extern fn int berrybits_model_gemv(void* handle, float* input, float* output);
extern fn int berrybits_model_save(void* handle, char* path);
extern fn int berrybits_model_distill_linear(
    void* handle,
    float* teacher_weights,
    float* teacher_bias,
    float* calibration_inputs,
    usz samples,
    uint32 epochs,
    float learning_rate,
    float* rmse_before,
    float* rmse_after
);
