const assert = require('node:assert/strict')
const fs = require('node:fs')
const os = require('node:os')
const path = require('node:path')
const { BerryBitsModel } = require('./index')

const weights = [0.1, -0.2, 0.3, -0.4, -0.4, 0.3, -0.2, 0.1]
const inputs = [1, 0.5, -0.5, 0.25, -0.25, 0.75, 1, -1]
const model = new BerryBitsModel(2, 4)
const metrics = model.distillLinear(weights, inputs, 2, { epochs: 4, learningRate: 0.02 })
assert.ok(metrics.rmseAfter <= metrics.rmseBefore + 1e-6)
assert.equal(model.gemv(inputs.slice(0, 4)).length, 2)

const output = path.join(os.tmpdir(), `berrybits-${process.pid}.b4lut`)
model.save(output)
model.close()
const loaded = BerryBitsModel.load(output)
assert.equal(loaded.gemv(inputs.slice(0, 4)).length, 2)
loaded.close()
fs.unlinkSync(output)
console.log('Node/Koffi Berryfour FFI quantize/train/GEMV/save/load passed')
