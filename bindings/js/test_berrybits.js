const assert = require('node:assert/strict')
const fs = require('node:fs')
const os = require('node:os')
const path = require('node:path')
const { BerryBitsTensor, MODES } = require('./berrybits')

const rows = 2
const cols = 67
const weights = Array.from({ length: rows * cols }, (_, i) => ((i * 37 + 13) % 101 - 50) / 53)
const input = Array.from({ length: cols }, (_, i) => ((i * 19 + 7) % 61 - 30) / 31)
for (const [mode, id] of Object.entries(MODES)) {
  const tensor = new BerryBitsTensor(rows, cols, mode)
  tensor.quantize(weights)
  const output = tensor.gemv(input)
  assert.equal(output.length, rows)
  const tempPath = path.join(os.tmpdir(), `berrybits-${process.pid}-${id}.blut`)
  tensor.save(tempPath)
  tensor.close()
  const loaded = BerryBitsTensor.load(tempPath)
  assert.equal(loaded.mode, mode)
  assert.deepEqual(loaded.dequantize(), tensorClosedGuard())
  loaded.close()
  fs.unlinkSync(tempPath)
}
function tensorClosedGuard() { return [] }
