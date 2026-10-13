# BerryBits

BerryBits is the new extensible native tensor-quantization core and `.blut` format. The original `berrybits` and `berrybits` packages remain compatibility APIs while the new modes and loaders are validated.

## CPU backends

The default build is portable and includes an ISA-independent packed-Q4 CPU kernel in `src/berrybits_cpu.c`. It decodes packed nibbles directly, uses multiple accumulators, and does not allocate a dequantization buffer per dot product. AVX, NEON, and other architecture-specific sources remain separate.

```sh
make clean && make
python3 test_q4_quality.py
```

For a build tuned to the current machine, use `make clean && make NATIVE=1`. A native build may require CPU instructions that are unavailable on other machines; use the default build for a portable binary.

## Kernel and training modules

- `include/berrybits.h` and `src/berrybits/berrybits.c` define the BerryBits C ABI and versioned `.blut` tensor container.
- `make` builds both `build/libberrybits.so` and the legacy `build/libberrybits.so`.
- `make test` round-trips Q64/Q32/Q16, Q8/Q6/Q5, custom K_S/K_M presets, Q3/Q2/I2_S/Q1, and ternary through `.blut` and GEMV.

- `src/berrybits_kernel/` contains the Q4 packed-weight GEMV, embedded 16-entry LUT model format, CPU fallback, AVX, NEON, HVX, and CUDA sources.
- `src/berrybits_training/` contains teacher/student linear-layer distillation and the existing optimizer helpers.
- `make test` checks training loss, model save/load including its LUT, and FP16-to-FP32 input conversion.
- `make demo-distill` trains a tiny linear FP32 teacher from zero, creates an FP16 snapshot, distills both into separate Berry Q4 LUT students, and compares both on the same held-out set.
- `make model-evaluate MODEL_EVAL_PYTHON=/path/to/venv/bin/python` runs the pinned TinyStories-33M FP32/FP16/native-Q4 pilot and exports per-layer `.b4lut` files plus manifests under `build/tinystories_q4_f2f/`.
- `make benchmark` compares block-scaled LUT GEMV and quantization against simple references on a synthetic `128x1024` CPU workload.
- `make clean && make CUDA=1` selects the CUDA GEMV backend and requires `nvcc` plus the CUDA runtime. Default builds use a stub that reports CUDA unavailable.
- `make clean && make OPENCL=1` enables the dynamically loaded OpenCL GPU adapter; call `berrybits_opencl_init` with the model and the path to `berrybits_q4_gemv.cl`.
- `src/berrybits_kernel/berrybits_q4_gemv.comp` and `berrybits_q4_gemv_gl.comp` are Vulkan and OpenGL compute shaders; `make vulkan-shader` and `make opengl-shader` compile them when `glslangValidator` is installed.
- `make opencl-kernel-check` syntax-checks the OpenCL kernel with Clang.

## Language APIs

`include/berrybits_q4_api.h` is the public C ABI. It also provides opaque model handles, so language bindings do not need to mirror C struct padding. The same create/load, quantize, GEMV, distill, save, and destroy calls are available through Python (`BerryBitsModel`), LuaJIT FFI, Node.js/Koffi, and C3 `extern fn` declarations in `bindings/`.

- Python: `from berrybits import BerryBitsModel`
- Node.js: `cd bindings/js && npm install && npm test`
- LuaJIT: add `bindings/lua/?.lua` to `LUA_PATH` and require `berrybits`
- C3: import the declarations from `bindings/c3/berrybits.c3` and link `build/libberrybits.so`

The VM has Python and Node.js, so those bindings can be tested here. LuaJIT and C3 runtimes are not installed, so their FFI/extern source examples are provided but not executed here.

See [docs/model-evaluation.md](docs/model-evaluation.md) for the recommended sub-100M FP32/FP16 teacher, fair Q4 evaluation protocol, current measured demo results, and LUT sidecar/load failure guidance.

The included trainer is a C reference implementation for a single linear layer with calibration inputs; it is not yet a full transformer/LLM trainer. It uses FP32 compute and a straight-through-style update with loss backtracking. Four-bit weights are lossy, so the trainer measures and reduces calibration-output error but cannot promise 100% equivalence to the FP32/FP16 teacher. The small distillation demo uses only 32 inputs, 8 outputs, 128 training examples, and 32 held-out examples; it does not download or load a multi-gigabyte model. CUDA, OpenCL, Vulkan, and OpenGL GPU execution could not be tested in the current environment because the corresponding runtime/SDKs and accessible GPU device are absent. Vulkan/OpenGL shader files are provided, but their host dispatch adapters are not yet integrated. Further device-specific targets include ROCm/HIP, Apple Metal, and WebGPU.
