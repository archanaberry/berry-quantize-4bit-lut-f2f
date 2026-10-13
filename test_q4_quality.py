#!/usr/bin/env python3
import math

from berrybits import BerryBitsRuntime


def rmse(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)) / len(a))


def case(name, values, threshold):
    runtime = BerryBitsRuntime()
    q = runtime.quantize_row(values)
    d = runtime.dequantize_row(q)
    err = rmse(values, d)
    print(f"{name}: error={err:.6f} threshold={threshold}")
    assert err <= threshold, f"{name} exceeded threshold: {err} > {threshold}"


case("smooth", [(-1.0 + 0.08 * i) for i in range(32)], 0.15)
case("mixed", [(-1.0 if i % 3 == 0 else 1.0) * (0.45 + 0.03 * i) for i in range(32)], 0.15)
case("outlier", [0.0] * 24 + [3.0] + [0.0] * 7, 0.01)
case("random", [((i * 17) % 13 - 6) * 0.21 + 0.25 * ((i % 5) / 4.0) for i in range(32)], 0.15)

runtime = BerryBitsRuntime()
values = [((i % 11) - 5) * 0.17 for i in range(19)]
vector = [((i % 7) - 3) * 0.23 for i in range(len(values))]
payload = runtime.quantize_row(values)
decoded = runtime.dequantize_row(payload)
expected = sum(weight * value for weight, value in zip(decoded, vector))
actual = runtime.matmul_fused(payload, vector)
assert math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-6), (actual, expected)
print(f"generic CPU dot, odd tail: {actual:.6f}")
print("All quantization quality checks passed.")
