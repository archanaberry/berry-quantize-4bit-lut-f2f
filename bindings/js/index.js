const fs = require('node:fs')
const path = require('node:path')
const koffi = require('koffi')

function defaultLibraryPath() {
  if (process.env.BERRYBITS_LIBRARY) return process.env.BERRYBITS_LIBRARY
  const root = path.resolve(__dirname, '../..')
  const candidates = process.platform === 'win32'
    ? ['build/berrybits.dll', 'build/libberrybits.dll']
    : process.platform === 'darwin'
      ? ['build/libberrybits.dylib', 'build/libberrybits.so']
      : ['build/libberrybits.so']
  for (const candidate of candidates) {
    const fullPath = path.join(root, candidate)
    if (fs.existsSync(fullPath)) return fullPath
  }
  throw new Error('Berryfour shared library not found; run `make` or set BERRYBITS_LIBRARY')
}

const native = koffi.load(defaultLibraryPath())
const createNative = native.func('void * berrybits_model_create(uint32_t rows, uint32_t cols)')
const loadNative = native.func('void * berrybits_model_load(const char *path)')
const destroyNative = native.func('void berrybits_model_destroy(void *handle)')
const rowsNative = native.func('uint32_t berrybits_model_rows(const void *handle)')
const colsNative = native.func('uint32_t berrybits_model_cols(const void *handle)')
const quantizeNative = native.func('int berrybits_model_quantize(void *handle, const float *weights)')
const gemvNative = native.func('int berrybits_model_gemv(void *handle, const float *input, _Out_ float *output)')
const saveNative = native.func('int berrybits_model_save(void *handle, const char *path)')
const distillNative = native.func(
  'int berrybits_model_distill_linear(void *handle, const float *teacher_weights, const float *teacher_bias, const float *calibration_inputs, size_t samples, uint32_t epochs, float learning_rate, _Out_ float *rmse_before, _Out_ float *rmse_after)'
)

class BerryBitsModel {
  constructor(rows, cols, handle = null) {
    this.handle = handle || createNative(rows, cols)
    if (!this.handle) throw new Error('native Berryfour model allocation failed')
  }

  static load(filePath) {
    const handle = loadNative(filePath)
    if (!handle) throw new Error(`could not load Berryfour model: ${filePath}`)
    return new BerryBitsModel(0, 0, handle)
  }

  get rows() { return rowsNative(this.handle) }
  get cols() { return colsNative(this.handle) }

  quantize(weights) {
    const values = Float32Array.from(weights)
    if (values.length !== this.rows * this.cols) throw new RangeError('weights shape mismatch')
    if (!quantizeNative(this.handle, values)) throw new Error('native Q4 quantization failed')
  }

  gemv(input) {
    const values = Float32Array.from(input)
    if (values.length !== this.cols) throw new RangeError('input shape mismatch')
    const output = new Float32Array(this.rows)
    if (!gemvNative(this.handle, values, output)) throw new Error('native Q4 GEMV failed')
    return Array.from(output)
  }

  distillLinear(teacherWeights, inputs, samples, { epochs = 100, learningRate = 0.04 } = {}) {
    const weights = Float32Array.from(teacherWeights)
    const calibration = Float32Array.from(inputs)
    if (weights.length !== this.rows * this.cols) throw new RangeError('teacher weight shape mismatch')
    if (calibration.length !== samples * this.cols) throw new RangeError('calibration input shape mismatch')
    const before = [0]
    const after = [0]
    if (!distillNative(this.handle, weights, null, calibration, samples,
      epochs, learningRate, before, after)) {
      throw new Error('native teacher/student distillation failed')
    }
    return { rmseBefore: before[0], rmseAfter: after[0] }
  }

  save(filePath) {
    if (!saveNative(this.handle, filePath)) throw new Error(`could not save Berryfour model: ${filePath}`)
  }

  close() {
    if (this.handle) {
      destroyNative(this.handle)
      this.handle = null
    }
  }
}

module.exports = { BerryBitsModel }
