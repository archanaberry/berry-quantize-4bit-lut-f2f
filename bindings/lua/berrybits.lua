local ffi = require("ffi")

ffi.cdef[[
typedef struct berrybits_handle berrybits_handle_t;
berrybits_handle_t *berrybits_create(uint32_t rows, uint32_t cols, int mode);
berrybits_handle_t *berrybits_load(const char *path);
void berrybits_destroy(berrybits_handle_t *handle);
uint32_t berrybits_rows(const berrybits_handle_t *handle);
uint32_t berrybits_cols(const berrybits_handle_t *handle);
int berrybits_mode(const berrybits_handle_t *handle);
int berrybits_quantize_f32(berrybits_handle_t *handle, const float *weights);
int berrybits_quantize_f64(berrybits_handle_t *handle, const double *weights);
int berrybits_dequantize_f32(const berrybits_handle_t *handle, float *weights);
int berrybits_gemv(const berrybits_handle_t *handle, const float *input, float *output);
int berrybits_save(berrybits_handle_t *handle, const char *path);
]]

local MODES = {
    Q64 = 1, Q32 = 2, Q16 = 3, Q8 = 8, Q6 = 9, Q5 = 10,
    Q4_K_S = 11, Q4_K_M = 12, Q3 = 13, Q2 = 14, I2_S = 15,
    Q1 = 16, TERNARY = 17
}
local names = {}
for name, value in pairs(MODES) do names[value] = name end

local native = ffi.load(os.getenv("BERRYBITS_LIBRARY") or "build/libberrybits.so")
local M = {modes = MODES}
local Tensor = {}
Tensor.__index = Tensor

local function array_of(ctype, values)
    local result = ffi.new(ctype .. "[?]", #values)
    for index, value in ipairs(values) do result[index - 1] = value end
    return result
end

local function wrap(handle)
    assert(handle ~= nil, "BerryBits native operation failed")
    return setmetatable({handle = ffi.gc(handle, native.berrybits_destroy)}, Tensor)
end

function M.new(rows, cols, mode)
    local mode_id = type(mode) == "string" and MODES[string.upper(mode)] or mode or MODES.Q4_K_M
    assert(mode_id, "unknown BerryBits mode")
    return wrap(native.berrybits_create(rows, cols, mode_id))
end

function M.load(path)
    return wrap(native.berrybits_load(path))
end

function Tensor:quantize(values)
    assert(#values == self:rows() * self:cols(), "weight shape mismatch")
    local ctype, fn = "float", native.berrybits_quantize_f32
    if names[native.berrybits_mode(self.handle)] == "Q64" then
        ctype, fn = "double", native.berrybits_quantize_f64
    end
    assert(fn(self.handle, array_of(ctype, values)) ~= 0, "BerryBits quantization failed")
end

function Tensor:dequantize()
    local length = self:rows() * self:cols()
    local output = ffi.new("float[?]", length)
    assert(native.berrybits_dequantize_f32(self.handle, output) ~= 0, "BerryBits dequantize failed")
    local result = {}
    for index = 0, length - 1 do result[index + 1] = tonumber(output[index]) end
    return result
end

function Tensor:gemv(input)
    assert(#input == self:cols(), "input shape mismatch")
    local output = ffi.new("float[?]", self:rows())
    assert(native.berrybits_gemv(self.handle, array_of("float", input), output) ~= 0,
        "BerryBits GEMV failed")
    local result = {}
    for index = 0, self:rows() - 1 do result[index + 1] = tonumber(output[index]) end
    return result
end

function Tensor:save(path)
    assert(native.berrybits_save(self.handle, path) ~= 0, "BerryBits save failed")
end

function Tensor:rows() return tonumber(native.berrybits_rows(self.handle)) end
function Tensor:cols() return tonumber(native.berrybits_cols(self.handle)) end
function Tensor:mode() return names[native.berrybits_mode(self.handle)] end
function Tensor:close()
    if self.handle ~= nil then
        ffi.gc(self.handle, nil)
        native.berrybits_destroy(self.handle)
        self.handle = nil
    end
end

M.Tensor = Tensor
return M
