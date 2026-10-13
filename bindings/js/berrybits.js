const fs = require('node:fs')
const path = require('node:path')
const koffi = require('koffi')

const MODES = Object.freeze({
  Q64: 1, Q32: 2, Q16: 3, Q8: 8, Q6: 9, Q5: 10,
  Q4_K_S: 11, Q4_K_M: 12, Q3: 13, Q2: 14, I2_S: 15, Q1: 16, TERNARY: 17
})

function nativePath() {
  if (process.env.BERRYBITS_LIBRARY) return process.env.BERRYBITS_LIBRARY
  const root = path.resolve(__dirname, '../..')
  const candidates = process.platform === 'win32'
    ? ['build/berrybits.dll', 'build/libberrybits.dll']
    : process.platform === 'darwin'
      ? ['build/libberrybits.dylib']
      : ['build/libberrybits.so']
  for (const candidate of candidates) {
    const file = path.join(root, candidate)
    if (fs.existsSync(file)) return file
  }
  throw new Error('libberrybits not found; run `make` or set BERRYBITS_LIBRARY')
}

const lib = koffi.load(nativePath())
const create = lib.func('void * berrybits_create(uint32_t rows, uint32_t cols, int mode)')
const load = lib.func('void * berrybits_load(const char *path)')
const destroy = lib.func('void berrybits_destroy(void *handle)')
const rowsOf = lib.func('uint32_t berrybits_rows(const void *handle)')
const colsOf = lib.func('uint32_t berrybits_cols(const void *handle)')
const modeOf = lib.func('int berrybits_mode(const void *handle)')
const quantizeF32 = lib.func('int berrybits_quantize_f32(void *handle, const float *weights)')
const quantizeF64 = lib.func('int berrybits_quantize_f64(void *handle, const double *weights)')
const dequantize = lib.func('int berrybits_dequantize_f32(const void *handle, _Out_ float *weights)')
const gemv = lib.func('int berrybits_gemv(const void *handle, const float *input, _Out_ float *output)')
const save = lib.func('int berrybits_save(void *handle, const char *path)')

class BerryBitsTensor {
  constructor(rows, cols, mode = 'Q4_K_M', handle = null) {
    const modeId = typeof mode === 'string' ? MODES[mode.toUpperCase()] : mode
    if (!modeId) throw new RangeError(`unsupported BerryBits mode: ${mode}`)
    this.handle = handle || create(rows, cols, modeId)
    if (!this.handle) throw new Error('BerryBits tensor allocation failed')
  }

  static load(filePath) {
    const handle = load(filePath)
    if (!handle) throw new Error(`failed to load .blut: ${filePath}`)
    return new BerryBitsTensor(0, 0, modeOf(handle), handle)
  }

  get rows() { return rowsOf(this.handle) }
  get cols() { return colsOf(this.handle) }
  get mode() { return Object.keys(MODES).find(name => MODES[name] === modeOf(this.handle)) }

  quantize(values) {
    if (values.length !== this.rows * this.cols) throw new RangeError('weight shape mismatch')
    const ok = this.mode === 'Q64'
      ? quantizeF64(this.handle, Float64Array.from(values))
      : quantizeF32(this.handle, Float32Array.from(values))
    if (!ok) throw new Error('BerryBits quantization failed')
  }

  dequantize() {
    const output = new Float32Array(this.rows * this.cols)
    if (!dequantize(this.handle, output)) throw new Error('BerryBits dequantization failed')
    return Array.from(output)
  }

  gemv(input) {
    if (input.length !== this.cols) throw new RangeError('input shape mismatch')
    const output = new Float32Array(this.rows)
    if (!gemv(this.handle, Float32Array.from(input), output)) throw new Error('BerryBits GEMV failed')
    return Array.from(output)
  }

  save(filePath) {
    if (!save(this.handle, filePath)) throw new Error(`failed to save .blut: ${filePath}`)
  }

  close() {
    if (this.handle) {
      destroy(this.handle)
      this.handle = null
    }
  }
}

module.exports = { BerryBitsTensor, MODES }
