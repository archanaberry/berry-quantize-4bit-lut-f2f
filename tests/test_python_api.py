#!/usr/bin/env python3
import math
import tempfile
from pathlib import Path

from berrybits import BerryBitsModel

weights = [[0.1, -0.2, 0.3, -0.4], [-0.4, 0.3, -0.2, 0.1]]
inputs = [[1.0, 0.5, -0.5, 0.25], [-0.25, 0.75, 1.0, -1.0], [0.5, -0.5, 0.25, 0.75]]
model = BerryBitsModel(2, 4)
metrics = model.distill_linear(weights, inputs, epochs=4, learning_rate=0.02)
assert metrics["rmse_after"] <= metrics["rmse_before"] + 1e-6
assert len(model.gemv(inputs[0])) == 2

with tempfile.TemporaryDirectory() as temp_dir:
    model_path = Path(temp_dir) / "python_api_test.b4lut"
    model.save(model_path)
    loaded = BerryBitsModel.load(model_path)
    assert len(loaded.gemv(inputs[0])) == 2
    loaded.close()
model.close()
print("Python Berryfour opaque-FFI quantize/train/GEMV/save/load passed")
