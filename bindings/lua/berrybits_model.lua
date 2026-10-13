local ffi = require("ffi")

ffi.cdef[[
typedef struct berrybits_model_handle berrybits_model_handle_t;
berrybits_model_handle_t *berrybits_model_create(uint32_t rows, uint32_t cols);
berrybits_model_handle_t *berrybits_model_load(const char *path);
void berrybits_model_destroy(berrybits_model_handle_t *handle);
uint32_t berrybits_model_rows(const berrybits_model_handle_t *handle);
uint32_t berrybits_model_cols(const berrybits_model_handle_t *handle);
int berrybits_model_quantize(berrybits_model_handle_t *handle, const float *weights);
int berrybits_model_gemv(berrybits_model_handle_t *handle, const float *input, float *output);
int berrybits_model_save(berrybits_model_handle_t *handle, const char *path);
int berrybits_model_distill_linear(berrybits_model_handle_t *handle,
    const float *teacher_weights, const float *teacher_bias,
    const float *calibration_inputs, size_t samples, uint32_t epochs,
    float learning_rate, float *rmse_before, float *rmse_after);
]]

local library_path = os.getenv("BERRYBITS_LIBRARY") or "build/libberrybits.so"
local native = ffi.load(library_path)
local M = {}
local Model = {}
Model.__index = Model

local function float_array(values)
    local array = ffi.new("float[?]", #values)
    for index, value in ipairs(values) do
        array[index - 1] = value
    end
    return array
end

local function wrap(handle)
    assert(handle ~= nil, "Berryfour native model operation failed")
    return setmetatable({handle = ffi.gc(handle, native.berrybits_model_destroy)}, Model)
end

function M.new(rows, cols)
    return wrap(native.berrybits_model_create(rows, cols))
end

function M.load(path)
    return wrap(native.berrybits_model_load(path))
end

function Model:quantize(weights)
    assert(#weights == self:rows() * self:cols(), "weights shape mismatch")
    assert(native.berrybits_model_quantize(self.handle, float_array(weights)) ~= 0,
        "native Q4 quantization failed")
end

function Model:gemv(input)
    assert(#input == self:cols(), "input shape mismatch")
    local output = ffi.new("float[?]", self:rows())
    assert(native.berrybits_model_gemv(self.handle, float_array(input), output) ~= 0,
        "native Q4 GEMV failed")
    local result = {}
    for index = 0, self:rows() - 1 do result[index + 1] = output[index] end
    return result
end

function Model:distill_linear(weights, inputs, samples, epochs, learning_rate)
    local before = ffi.new("float[1]")
    local after = ffi.new("float[1]")
    assert(native.berrybits_model_distill_linear(
        self.handle, float_array(weights), nil, float_array(inputs), samples,
        epochs or 100, learning_rate or 0.04, before, after) ~= 0,
        "native teacher/student distillation failed")
    return tonumber(before[0]), tonumber(after[0])
end

function Model:save(path)
    assert(native.berrybits_model_save(self.handle, path) ~= 0,
        "native model save failed")
end

function Model:rows()
    return tonumber(native.berrybits_model_rows(self.handle))
end

function Model:cols()
    return tonumber(native.berrybits_model_cols(self.handle))
end

function Model:close()
    if self.handle ~= nil then
        ffi.gc(self.handle, nil)
        native.berrybits_model_destroy(self.handle)
        self.handle = nil
    end
end

M.Model = Model
return M
