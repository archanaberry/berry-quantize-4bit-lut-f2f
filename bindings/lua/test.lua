local berrybits = require("berrybits")
local model = berrybits.new(2, 4)
local weights = {0.1, -0.2, 0.3, -0.4, -0.4, 0.3, -0.2, 0.1}
local inputs = {1, 0.5, -0.5, 0.25, -0.25, 0.75, 1, -1}
local before, after = model:distill_linear(weights, inputs, 2, 4, 0.02)
assert(after <= before + 1e-6)
assert(#model:gemv({1, 0.5, -0.5, 0.25}) == 2)
model:save("build/lua_api_test.b4lut")
model:close()
local loaded = berrybits.load("build/lua_api_test.b4lut")
assert(#loaded:gemv({1, 0.5, -0.5, 0.25}) == 2)
loaded:close()
os.remove("build/lua_api_test.b4lut")
print("LuaJIT FFI Berryfour quantize/train/GEMV/save/load passed")
