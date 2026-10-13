# BerryBits Python API

The `berrybits` package calls `libberrybits.so` through ctypes. Quantization modes are selected by name:

```python
from berrybits import BerryBitsTensor

with BerryBitsTensor(128, 256, "Q4_K_M") as tensor:
    tensor.quantize(flat_weights)
    output = tensor.gemv(input_vector)
    tensor.save("weights.blut")
```

The Q4_K and I2_S labels in this version name BerryBits-native presets; they are not byte-compatible with llama.cpp/GGUF quantization. See the root README for supported formats and current limitations.
