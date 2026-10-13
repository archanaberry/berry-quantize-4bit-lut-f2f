#define _POSIX_C_SOURCE 200809L

#include "berrybits_opencl.h"

#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BERRYBITS_CL_SUCCESS 0
#define BERRYBITS_CL_DEVICE_TYPE_GPU (1ULL << 2)
#define BERRYBITS_CL_MEM_WRITE_ONLY (1ULL << 1)
#define BERRYBITS_CL_MEM_READ_ONLY (1ULL << 2)
#define BERRYBITS_CL_TRUE 1u

typedef int32_t berrybits_cl_int;
typedef uint32_t berrybits_cl_uint;
typedef uint64_t berrybits_cl_bitfield;
typedef berrybits_cl_bitfield berrybits_cl_device_type;
typedef berrybits_cl_bitfield berrybits_cl_mem_flags;
typedef intptr_t berrybits_cl_context_properties;
typedef struct _cl_platform_id *berrybits_cl_platform_id;
typedef struct _cl_device_id *berrybits_cl_device_id;
typedef struct _cl_context *berrybits_cl_context;
typedef struct _cl_command_queue *berrybits_cl_command_queue;
typedef struct _cl_program *berrybits_cl_program;
typedef struct _cl_kernel *berrybits_cl_kernel;
typedef struct _cl_mem *berrybits_cl_mem;

struct berrybits_opencl_api {
    berrybits_cl_int (*get_platform_ids)(berrybits_cl_uint, berrybits_cl_platform_id *, berrybits_cl_uint *);
    berrybits_cl_int (*get_device_ids)(berrybits_cl_platform_id, berrybits_cl_device_type, berrybits_cl_uint, berrybits_cl_device_id *, berrybits_cl_uint *);
    berrybits_cl_context (*create_context)(const berrybits_cl_context_properties *, berrybits_cl_uint, const berrybits_cl_device_id *, void (*)(const char *, const void *, size_t, void *), void *, berrybits_cl_int *);
    berrybits_cl_command_queue (*create_command_queue)(berrybits_cl_context, berrybits_cl_device_id, berrybits_cl_bitfield, berrybits_cl_int *);
    berrybits_cl_program (*create_program_with_source)(berrybits_cl_context, berrybits_cl_uint, const char **, const size_t *, berrybits_cl_int *);
    berrybits_cl_int (*build_program)(berrybits_cl_program, berrybits_cl_uint, const berrybits_cl_device_id *, const char *, void (*)(berrybits_cl_program, void *), void *);
    berrybits_cl_kernel (*create_kernel)(berrybits_cl_program, const char *, berrybits_cl_int *);
    berrybits_cl_mem (*create_buffer)(berrybits_cl_context, berrybits_cl_mem_flags, size_t, void *, berrybits_cl_int *);
    berrybits_cl_int (*set_kernel_arg)(berrybits_cl_kernel, berrybits_cl_uint, size_t, const void *);
    berrybits_cl_int (*enqueue_write_buffer)(berrybits_cl_command_queue, berrybits_cl_mem, berrybits_cl_uint, size_t, size_t, const void *, berrybits_cl_uint, const void *, void *);
    berrybits_cl_int (*enqueue_read_buffer)(berrybits_cl_command_queue, berrybits_cl_mem, berrybits_cl_uint, size_t, size_t, void *, berrybits_cl_uint, const void *, void *);
    berrybits_cl_int (*enqueue_nd_range_kernel)(berrybits_cl_command_queue, berrybits_cl_kernel, berrybits_cl_uint, const size_t *, const size_t *, const size_t *, berrybits_cl_uint, const void *, void *);
    berrybits_cl_int (*finish)(berrybits_cl_command_queue);
    berrybits_cl_int (*release_mem_object)(berrybits_cl_mem);
    berrybits_cl_int (*release_kernel)(berrybits_cl_kernel);
    berrybits_cl_int (*release_program)(berrybits_cl_program);
    berrybits_cl_int (*release_command_queue)(berrybits_cl_command_queue);
    berrybits_cl_int (*release_context)(berrybits_cl_context);
};

struct berrybits_opencl_state {
    void *library;
    struct berrybits_opencl_api api;
    berrybits_cl_context context;
    berrybits_cl_command_queue queue;
    berrybits_cl_program program;
    berrybits_cl_kernel kernel;
    berrybits_cl_mem packed_weights;
    berrybits_cl_mem scales;
    berrybits_cl_mem lut;
    berrybits_cl_mem input;
    berrybits_cl_mem output;
    berrybits_cl_device_id device;
    berrybits_cl_uint rows;
    berrybits_cl_uint cols;
    berrybits_cl_uint blocks_per_row;
};

static int berrybits_opencl_load_symbol(void *library, const char *name,
                                        void *destination, size_t destination_size) {
    void *symbol = dlsym(library, name);
    if (!symbol || destination_size != sizeof(symbol)) return 0;
    memcpy(destination, &symbol, sizeof(symbol));
    return 1;
}

#define BERRYBITS_CL_LOAD(state, member, symbol) \
    do { \
        if (!berrybits_opencl_load_symbol((state)->library, symbol, \
                                          &(state)->api.member, sizeof((state)->api.member))) goto fail; \
    } while (0)

static void berrybits_opencl_release_state(struct berrybits_opencl_state *state) {
    if (!state) return;
    if (state->packed_weights) state->api.release_mem_object(state->packed_weights);
    if (state->scales) state->api.release_mem_object(state->scales);
    if (state->lut) state->api.release_mem_object(state->lut);
    if (state->input) state->api.release_mem_object(state->input);
    if (state->output) state->api.release_mem_object(state->output);
    if (state->kernel) state->api.release_kernel(state->kernel);
    if (state->program) state->api.release_program(state->program);
    if (state->queue) state->api.release_command_queue(state->queue);
    if (state->context) state->api.release_context(state->context);
    if (state->library) dlclose(state->library);
    free(state);
}

static char *berrybits_opencl_read_source(const char *path, size_t *source_size) {
    FILE *file = fopen(path, "rb");
    if (!file) return NULL;
    if (fseek(file, 0, SEEK_END) != 0) {
        fclose(file);
        return NULL;
    }
    long length = ftell(file);
    if (length <= 0 || fseek(file, 0, SEEK_SET) != 0) {
        fclose(file);
        return NULL;
    }
    char *source = (char *)malloc((size_t)length + 1u);
    if (!source) {
        fclose(file);
        return NULL;
    }
    size_t read_count = fread(source, 1, (size_t)length, file);
    fclose(file);
    if (read_count != (size_t)length) {
        free(source);
        return NULL;
    }
    source[read_count] = '\0';
    *source_size = read_count;
    return source;
}

static int berrybits_opencl_load_api(struct berrybits_opencl_state *state) {
    state->library = dlopen("libOpenCL.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!state->library) state->library = dlopen("libOpenCL.so", RTLD_NOW | RTLD_LOCAL);
    if (!state->library) return 0;

    BERRYBITS_CL_LOAD(state, get_platform_ids, "clGetPlatformIDs");
    BERRYBITS_CL_LOAD(state, get_device_ids, "clGetDeviceIDs");
    BERRYBITS_CL_LOAD(state, create_context, "clCreateContext");
    BERRYBITS_CL_LOAD(state, create_command_queue, "clCreateCommandQueue");
    BERRYBITS_CL_LOAD(state, create_program_with_source, "clCreateProgramWithSource");
    BERRYBITS_CL_LOAD(state, build_program, "clBuildProgram");
    BERRYBITS_CL_LOAD(state, create_kernel, "clCreateKernel");
    BERRYBITS_CL_LOAD(state, create_buffer, "clCreateBuffer");
    BERRYBITS_CL_LOAD(state, set_kernel_arg, "clSetKernelArg");
    BERRYBITS_CL_LOAD(state, enqueue_write_buffer, "clEnqueueWriteBuffer");
    BERRYBITS_CL_LOAD(state, enqueue_read_buffer, "clEnqueueReadBuffer");
    BERRYBITS_CL_LOAD(state, enqueue_nd_range_kernel, "clEnqueueNDRangeKernel");
    BERRYBITS_CL_LOAD(state, finish, "clFinish");
    BERRYBITS_CL_LOAD(state, release_mem_object, "clReleaseMemObject");
    BERRYBITS_CL_LOAD(state, release_kernel, "clReleaseKernel");
    BERRYBITS_CL_LOAD(state, release_program, "clReleaseProgram");
    BERRYBITS_CL_LOAD(state, release_command_queue, "clReleaseCommandQueue");
    BERRYBITS_CL_LOAD(state, release_context, "clReleaseContext");
    return 1;

fail:
    return 0;
}

static int berrybits_opencl_select_gpu(struct berrybits_opencl_state *state) {
    berrybits_cl_uint platform_count = 0;
    if (state->api.get_platform_ids(0, NULL, &platform_count) != BERRYBITS_CL_SUCCESS || platform_count == 0) return 0;
    berrybits_cl_platform_id *platforms = (berrybits_cl_platform_id *)calloc(platform_count, sizeof(*platforms));
    if (!platforms) return 0;
    berrybits_cl_int status = state->api.get_platform_ids(platform_count, platforms, NULL);
    if (status == BERRYBITS_CL_SUCCESS) {
        for (berrybits_cl_uint i = 0; i < platform_count; ++i) {
            status = state->api.get_device_ids(platforms[i], BERRYBITS_CL_DEVICE_TYPE_GPU,
                                               1, &state->device, NULL);
            if (status == BERRYBITS_CL_SUCCESS) break;
        }
    }
    free(platforms);
    return status == BERRYBITS_CL_SUCCESS;
}

int berrybits_opencl_available(void) {
    struct berrybits_opencl_state state = {0};
    if (!berrybits_opencl_load_api(&state) || !berrybits_opencl_select_gpu(&state)) {
        if (state.library) dlclose(state.library);
        return 0;
    }
    dlclose(state.library);
    return 1;
}

int berrybits_opencl_init(berrybits_opencl_context_t *context,
                          const berrybits_q4_model_t *model,
                          const char *kernel_source_path) {
    if (!context || !model || !kernel_source_path || !model->packed_weights ||
        !model->scales || model->rows == 0 || model->cols == 0) return 0;
    context->state = NULL;

    struct berrybits_opencl_state *state = (struct berrybits_opencl_state *)calloc(1, sizeof(*state));
    if (!state) return 0;
    char *source = NULL;
    if (!berrybits_opencl_load_api(state) || !berrybits_opencl_select_gpu(state)) goto fail;

    berrybits_cl_int status = BERRYBITS_CL_SUCCESS;
    state->context = state->api.create_context(NULL, 1, &state->device, NULL, NULL, &status);
    if (!state->context || status != BERRYBITS_CL_SUCCESS) goto fail;
    state->queue = state->api.create_command_queue(state->context, state->device, 0, &status);
    if (!state->queue || status != BERRYBITS_CL_SUCCESS) goto fail;

    size_t source_size = 0;
    source = berrybits_opencl_read_source(kernel_source_path, &source_size);
    if (!source) goto fail;
    const char *source_pointer = source;
    state->program = state->api.create_program_with_source(state->context, 1,
                                                            &source_pointer, &source_size, &status);
    free(source);
    source = NULL;
    if (!state->program || status != BERRYBITS_CL_SUCCESS ||
        state->api.build_program(state->program, 1, &state->device,
                                 "-cl-std=CL1.2", NULL, NULL) != BERRYBITS_CL_SUCCESS) goto fail;
    state->kernel = state->api.create_kernel(state->program, "berrybits_q4_gemv", &status);
    if (!state->kernel || status != BERRYBITS_CL_SUCCESS) goto fail;

    state->rows = model->rows;
    state->cols = model->cols;
    state->blocks_per_row = model->blocks_per_row;
    size_t packed_row_bytes = ((size_t)model->cols + 1u) / 2u;
    size_t packed_size = (size_t)model->rows * packed_row_bytes;
    size_t scale_size = (size_t)model->rows * model->blocks_per_row * sizeof(float);
    state->packed_weights = state->api.create_buffer(state->context, BERRYBITS_CL_MEM_READ_ONLY,
                                                     packed_size, NULL, &status);
    if (!state->packed_weights || status != BERRYBITS_CL_SUCCESS) goto fail;
    state->scales = state->api.create_buffer(state->context, BERRYBITS_CL_MEM_READ_ONLY,
                                             scale_size, NULL, &status);
    if (!state->scales || status != BERRYBITS_CL_SUCCESS) goto fail;
    state->lut = state->api.create_buffer(state->context, BERRYBITS_CL_MEM_READ_ONLY,
                                          sizeof(model->lut), NULL, &status);
    if (!state->lut || status != BERRYBITS_CL_SUCCESS) goto fail;
    state->input = state->api.create_buffer(state->context, BERRYBITS_CL_MEM_READ_ONLY,
                                            (size_t)model->cols * sizeof(float), NULL, &status);
    if (!state->input || status != BERRYBITS_CL_SUCCESS) goto fail;
    state->output = state->api.create_buffer(state->context, BERRYBITS_CL_MEM_WRITE_ONLY,
                                             (size_t)model->rows * sizeof(float), NULL, &status);
    if (!state->output || status != BERRYBITS_CL_SUCCESS) goto fail;

    if (state->api.enqueue_write_buffer(state->queue, state->packed_weights, BERRYBITS_CL_TRUE,
                                        0, packed_size, model->packed_weights, 0, NULL, NULL) != BERRYBITS_CL_SUCCESS ||
        state->api.enqueue_write_buffer(state->queue, state->scales, BERRYBITS_CL_TRUE,
                                        0, scale_size, model->scales, 0, NULL, NULL) != BERRYBITS_CL_SUCCESS ||
        state->api.enqueue_write_buffer(state->queue, state->lut, BERRYBITS_CL_TRUE,
                                        0, sizeof(model->lut), model->lut, 0, NULL, NULL) != BERRYBITS_CL_SUCCESS) goto fail;

    context->state = state;
    return 1;

fail:
    free(source);
    berrybits_opencl_release_state(state);
    return 0;
}

int berrybits_opencl_gemv(berrybits_opencl_context_t *context,
                          const float *input, float *output) {
    if (!context || !context->state || !input || !output) return 0;
    struct berrybits_opencl_state *state = (struct berrybits_opencl_state *)context->state;
    const berrybits_cl_uint arguments[] = {
        (berrybits_cl_uint)sizeof(state->packed_weights),
        (berrybits_cl_uint)sizeof(state->scales),
        (berrybits_cl_uint)sizeof(state->lut),
        (berrybits_cl_uint)sizeof(state->input),
        (berrybits_cl_uint)sizeof(state->output),
        (berrybits_cl_uint)sizeof(state->rows),
        (berrybits_cl_uint)sizeof(state->cols),
        (berrybits_cl_uint)sizeof(state->blocks_per_row)
    };
    const void *values[] = {
        &state->packed_weights, &state->scales, &state->lut, &state->input,
        &state->output, &state->rows, &state->cols, &state->blocks_per_row
    };
    for (berrybits_cl_uint i = 0; i < sizeof(arguments) / sizeof(arguments[0]); ++i) {
        if (state->api.set_kernel_arg(state->kernel, i, arguments[i], values[i]) != BERRYBITS_CL_SUCCESS) return 0;
    }

    size_t input_size = (size_t)state->cols * sizeof(float);
    if (state->api.enqueue_write_buffer(state->queue, state->input, BERRYBITS_CL_TRUE,
                                        0, input_size, input, 0, NULL, NULL) != BERRYBITS_CL_SUCCESS) return 0;
    size_t global_size = state->rows;
    if (state->api.enqueue_nd_range_kernel(state->queue, state->kernel, 1, NULL,
                                           &global_size, NULL, 0, NULL, NULL) != BERRYBITS_CL_SUCCESS ||
        state->api.enqueue_read_buffer(state->queue, state->output, BERRYBITS_CL_TRUE,
                                       0, (size_t)state->rows * sizeof(float),
                                       output, 0, NULL, NULL) != BERRYBITS_CL_SUCCESS) return 0;
    return state->api.finish(state->queue) == BERRYBITS_CL_SUCCESS;
}

void berrybits_opencl_destroy(berrybits_opencl_context_t *context) {
    if (!context) return;
    berrybits_opencl_release_state((struct berrybits_opencl_state *)context->state);
    context->state = NULL;
}
