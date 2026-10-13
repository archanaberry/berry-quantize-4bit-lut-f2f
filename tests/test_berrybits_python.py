#!/usr/bin/env python3
import tempfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from berrybits import BerryBitsTensor, MODES

weights = [[((row * 67 + col) % 71 - 35) / 37.0 for col in range(67)] for row in range(2)]
vector = [((col * 13) % 43 - 21) / 23.0 for col in range(67)]

for mode in MODES:
    tensor = BerryBitsTensor(2, 67, mode)
    tensor.quantize(weights)
    decoded = tensor.dequantize()
    result = tensor.gemv(vector)
    assert len(decoded) == 134 and len(result) == 2
    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / f"{mode}.blut"
        tensor.save(path)
        loaded = BerryBitsTensor.load(path)
        assert loaded.mode == mode
        assert loaded.dequantize() == decoded
        assert loaded.gemv(vector) == result
        loaded.close()
    tensor.close()

q64 = BerryBitsTensor(1, 2, "Q64")
q64.quantize([0.125, -0.375])
assert q64.dequantize() == [0.125, -0.375]
q64.close()
print(f"Python BerryBits FFI passed all {len(MODES)} modes plus Q64 .blut round-trips")
