CC := gcc
CFLAGS ?= -O3 -fPIC -Iinclude -pthread
LDFLAGS := -shared -pthread

ifeq ($(NATIVE),1)
CFLAGS += -march=native
endif

BUILD_DIR := build
LIB_OUT := $(BUILD_DIR)/libberrybits.so
SRC := src/berrybits_runtime.c src/berrybits/berrybits.c src/berrybits_kernel/berrybits_cpu.c src/berrybits_kernel/berrybits_runtime_asm.S src/berrybits_training/berrybits_optimizer.c src/berrybits_kernel/berrybits_avx.c src/berrybits_kernel/berrybits_neon.c src/berrybits_kernel/berrybits_hvx.c src/berrybits_kernel/berrybits_runtime_kernel.c src/berrybits_kernel/berrybits_runtime_model.c src/berrybits_kernel/berrybits_q4_model.c src/berrybits_kernel/berrybits_q4_compat.c src/berrybits_training/berrybits_training.c
TEST_BIN := $(BUILD_DIR)/test_berrybits_training
BITS_TEST_BIN := $(BUILD_DIR)/test_berrybits_modes
PUBLIC_API_TEST_BIN := $(BUILD_DIR)/test_public_abi
GPU_TEST_BIN := $(BUILD_DIR)/test_gpu_api
BENCH_BIN := $(BUILD_DIR)/benchmark_berrybits_kernel
DEMO_BIN := $(BUILD_DIR)/berrybits_distill_demo
CLANG ?= clang
GLSLANG ?= glslangValidator
PYTHON ?= python3
MODEL_EVAL_PYTHON ?= /tmp/berrybits-model-eval/bin/python
MODEL_EVAL_CACHE ?= /tmp/berrybits-hf-cache

ifeq ($(CUDA),1)
NVCC ?= nvcc
CUDA_OBJECT := $(BUILD_DIR)/berrybits_cuda.o
CUDA_LIBS := -lcudart -lstdc++
else
SRC += src/berrybits_kernel/berrybits_cuda_stub.c
endif

ifeq ($(OPENCL),1)
OPENCL_IMPL := src/berrybits_kernel/berrybits_opencl.c
OPENCL_LIBS := -ldl
else
OPENCL_IMPL := src/berrybits_kernel/berrybits_opencl_stub.c
endif
SRC += $(OPENCL_IMPL)

all: $(LIB_OUT)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(LIB_OUT): $(SRC) $(CUDA_OBJECT) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $(SRC) $(CUDA_OBJECT) $(CUDA_LIBS) -lm

$(CUDA_OBJECT): src/berrybits_kernel/berrybits_cuda.cu src/berrybits_kernel/berrybits_cuda.h src/berrybits_kernel/berrybits_q4_model.h | $(BUILD_DIR)
	$(NVCC) -O3 -Xcompiler -fPIC -Isrc/berrybits_kernel -c -o $@ $<

clean:
	rm -rf $(BUILD_DIR)

$(TEST_BIN): tests/test_berrybits_training.c src/berrybits_kernel/berrybits_q4_model.c src/berrybits_training/berrybits_training.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -Isrc/berrybits_kernel -Isrc/berrybits_training -o $@ $^ -lm

test: $(TEST_BIN) $(BITS_TEST_BIN) $(LIB_OUT) $(PUBLIC_API_TEST_BIN) $(GPU_TEST_BIN)
	./$(TEST_BIN)
	./$(BITS_TEST_BIN)
	./$(PUBLIC_API_TEST_BIN)
	./$(GPU_TEST_BIN)
	$(PYTHON) tests/test_python_api.py
	$(PYTHON) tests/test_berrybits_python.py

$(BITS_TEST_BIN): tests/test_berrybits_modes.c src/berrybits/berrybits.c include/berrybits.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -Iinclude -o $@ tests/test_berrybits_modes.c src/berrybits/berrybits.c -lm

$(PUBLIC_API_TEST_BIN): tests/test_public_abi.c $(LIB_OUT) | $(BUILD_DIR)
	$(CC) $(CFLAGS) -Iinclude -o $@ $< $(LIB_OUT) -Wl,-rpath,'$$ORIGIN' -lm

$(GPU_TEST_BIN): tests/test_gpu_api.c $(OPENCL_IMPL) | $(BUILD_DIR)
	$(CC) $(CFLAGS) -Isrc/berrybits_kernel -o $@ $^ $(OPENCL_LIBS)

$(BENCH_BIN): tests/benchmark_berrybits_kernel.c src/berrybits_kernel/berrybits_q4_model.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -Isrc/berrybits_kernel -o $@ $^ -lm

benchmark: $(BENCH_BIN)
	./$(BENCH_BIN)

$(DEMO_BIN): examples/berrybits_distill_demo.c src/berrybits_kernel/berrybits_q4_model.c src/berrybits_training/berrybits_training.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -Isrc/berrybits_kernel -Isrc/berrybits_training -o $@ $^ -lm

demo-distill: $(DEMO_BIN)
	./$(DEMO_BIN)

model-evaluate: all
	HF_HOME="$(MODEL_EVAL_CACHE)" PYTHONPATH=. $(MODEL_EVAL_PYTHON) scripts/evaluate_tinystories_q4.py

opencl-kernel-check:
	$(CLANG) -x cl -cl-std=CL1.2 -Wall -Wextra -Werror -fsyntax-only src/berrybits_kernel/berrybits_q4_gemv.cl

vulkan-shader: | $(BUILD_DIR)
	$(GLSLANG) -V src/berrybits_kernel/berrybits_q4_gemv.comp -o $(BUILD_DIR)/berrybits_q4_gemv.spv

opengl-shader: | $(BUILD_DIR)
	$(GLSLANG) -G src/berrybits_kernel/berrybits_q4_gemv_gl.comp -o $(BUILD_DIR)/berrybits_q4_gemv_gl.spv

.PHONY: all clean test benchmark demo-distill model-evaluate opencl-kernel-check vulkan-shader opengl-shader
