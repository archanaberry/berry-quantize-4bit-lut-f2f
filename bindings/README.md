# Berryfour language bindings

All bindings call the opaque-handle API in `include/berrybits_q4_api.h`; model storage, quantization, training, and inference remain in the native C library.

- C: include `berrybits_q4_api.h` and link `build/libberrybits.so`.
- Python: import `BerryBitsModel` from `berrybits`; it uses `ctypes`.
- Node.js: `cd bindings/js && npm install && npm test`; it uses Koffi for direct FFI.
- LuaJIT: add `bindings/lua/?.lua` to `LUA_PATH`, then `require("berrybits")`; it uses `ffi`.
- C3: use `bindings/c3/berrybits.c3`; it declares the native ABI with `extern fn`.

Set `BERRYBITS_LIBRARY` to override the default library path. The C3 compiler and LuaJIT are not installed in the development VM; those bindings require validation on systems with those runtimes.
