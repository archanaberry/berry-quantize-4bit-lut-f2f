from __future__ import annotations

import ctypes
import math
import os
import subprocess
from pathlib import Path
from typing import Sequence

_ROOT = Path(__file__).resolve().parent.parent
_LIB_PATH = _ROOT / "build" / "libberrybits.so"

if not _LIB_PATH.exists():
    subprocess.check_call(["make", "-C", str(_ROOT), "all"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

_NATIVE = ctypes.CDLL(str(_LIB_PATH))
_NATIVE.berrybits_runtime_q4_init_lut.argtypes = [ctypes.POINTER(ctypes.c_float)]
_NATIVE.berrybits_runtime_q4_quantize_row.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_uint8),
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_float),
]
_NATIVE.berrybits_runtime_q4_dequantize_row.argtypes = [
    ctypes.POINTER(ctypes.c_uint8),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_float,
    ctypes.c_float,
]
_NATIVE.berrybits_runtime_q4_dot_row.argtypes = [
    ctypes.POINTER(ctypes.c_uint8),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_float,
    ctypes.c_float,
]
_NATIVE.berrybits_runtime_q4_dot_row.restype = ctypes.c_float
_NATIVE.berrybits_runtime_q4_teacher_error.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_size_t,
]
_NATIVE.berrybits_runtime_q4_teacher_error.restype = ctypes.c_float
_NATIVE.berrybits_runtime_q4_backend_probe.restype = ctypes.c_int
_NATIVE.berrybits_runtime_q4_backend_name.argtypes = [ctypes.c_int]
_NATIVE.berrybits_runtime_q4_backend_name.restype = ctypes.c_char_p
_NATIVE.berrybits_runtime_q4_runtime_config.restype = ctypes.c_void_p
_NATIVE.berrybits_runtime_q4_threaded_matmul.argtypes = [
    ctypes.POINTER(ctypes.c_uint8),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_float,
    ctypes.c_float,
    ctypes.c_int,
]


class BerryBitsRuntime:
    LUT_LEN = 16

    def __init__(self):
        self.lut = (ctypes.c_float * self.LUT_LEN)()
        _NATIVE.berrybits_runtime_q4_init_lut(self.lut)
        self.backend_id = int(_NATIVE.berrybits_runtime_q4_backend_probe())
        self.backend_name = _NATIVE.berrybits_runtime_q4_backend_name(self.backend_id).decode("utf-8")

    def _as_float_array(self, data: Sequence[float]):
        values = (ctypes.c_float * len(data))(*[float(v) for v in data])
        return values

    def backend_info(self):
        return {
            "backend": self.backend_name,
            "backend_id": self.backend_id,
            "threads": 8,
        }

    def quantize_row(self, values: Sequence[float]):
        vec = self._as_float_array(values)
        packed_len = (len(values) + 1) // 2
        packed = (ctypes.c_uint8 * packed_len)()
        _NATIVE.berrybits_runtime_q4_quantize_row(vec, packed, len(values), self.lut)

        max_abs = max(abs(float(v)) for v in values) if values else 0.0
        scale = max_abs if max_abs != 0.0 else 1.0
        return {"packed": list(packed), "scale": float(scale), "zero": 0.0, "length": len(values)}

    def dequantize_row(self, payload):
        if isinstance(payload, dict):
            packed = payload["packed"]
            scale = payload["scale"]
            zero = payload["zero"]
            length = payload["length"]
        else:
            packed = payload
            scale = 1.0
            zero = 0.0
            length = len(packed) * 2

        out = (ctypes.c_float * length)()
        packed_array = (ctypes.c_uint8 * len(packed))(*packed)
        _NATIVE.berrybits_runtime_q4_dequantize_row(packed_array, out, length, self.lut, ctypes.c_float(scale), ctypes.c_float(zero))
        return [float(out[i]) for i in range(length)]

    def matmul_fused(self, payload, x_vector):
        packed = payload["packed"]
        scale = payload["scale"]
        zero = payload["zero"]
        x_arr = self._as_float_array(x_vector)
        packed_arr = (ctypes.c_uint8 * len(packed))(*packed)
        return float(_NATIVE.berrybits_runtime_q4_dot_row(packed_arr, x_arr, len(x_vector), self.lut, ctypes.c_float(scale), ctypes.c_float(zero)))

    def threaded_matmul(self, rows, cols, values, threads=None):
        packed_len = (cols + 1) // 2
        packed = (ctypes.c_uint8 * (rows * packed_len))()
        for r in range(rows):
            start = r * packed_len
            row = values[r]
            row_array = self._as_float_array(row)
            row_packed = (ctypes.c_uint8 * packed_len)()
            _NATIVE.berrybits_runtime_q4_quantize_row(row_array, row_packed, cols, self.lut)
            for i in range(packed_len):
                packed[start + i] = row_packed[i]

        x_arr = self._as_float_array(values[0])
        out = (ctypes.c_float * rows)()
        scale = 1.0
        zero = 0.0
        _NATIVE.berrybits_runtime_q4_threaded_matmul(packed, x_arr, out, rows, cols, self.lut, ctypes.c_float(scale), ctypes.c_float(zero), 1 if threads is None else threads)
        return [float(out[i]) for i in range(rows)]

    def teacher_error(self, teacher: Sequence[float], student: Sequence[float]) -> float:
        t = self._as_float_array(teacher)
        s = self._as_float_array(student)
        return float(_NATIVE.berrybits_runtime_q4_teacher_error(t, s, len(teacher)))


__all__ = ["BerryBitsRuntime"]
