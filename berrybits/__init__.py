from __future__ import annotations

import ctypes
import os
import subprocess
from pathlib import Path
from typing import Sequence

_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_LIBRARY = _ROOT / "build" / "libberrybits.so"

MODES = {
    "Q64_LUT_F2F": 1,
    "Q32_LUT_F2F": 2,
    "Q16_LUT_F2F": 3,
    "Q8_LUT_F2F": 8,
    "Q6_LUT_F2F": 9,
    "Q5_LUT_F2F": 10,
    "Q4_K_S_LUT_F2F": 11,
    "Q4_K_M_LUT_F2F": 12,
    "Q3_LUT_F2F": 13,
    "Q2_LUT_F2F": 14,
    "I2_S_LUT_F2F": 15,
    "Q1_LUT_F2F": 16,
    "TERNARY_LUT_F2F": 17,
}
_MODE_ALIASES = {name.removesuffix("_LUT_F2F"): value for name, value in MODES.items()}


def _load_native():
    library_path = Path(os.environ.get("BERRYBITS_LIBRARY", _DEFAULT_LIBRARY))
    if not library_path.exists() and "BERRYBITS_LIBRARY" not in os.environ:
        subprocess.check_call(["make", "-C", str(_ROOT), "build/libberrybits.so"])
    native = ctypes.CDLL(str(library_path))
    float_pointer = ctypes.POINTER(ctypes.c_float)
    double_pointer = ctypes.POINTER(ctypes.c_double)

    native.berrybits_create.argtypes = [ctypes.c_uint32, ctypes.c_uint32, ctypes.c_int]
    native.berrybits_create.restype = ctypes.c_void_p
    native.berrybits_load.argtypes = [ctypes.c_char_p]
    native.berrybits_load.restype = ctypes.c_void_p
    native.berrybits_destroy.argtypes = [ctypes.c_void_p]
    native.berrybits_destroy.restype = None
    native.berrybits_rows.argtypes = [ctypes.c_void_p]
    native.berrybits_rows.restype = ctypes.c_uint32
    native.berrybits_cols.argtypes = [ctypes.c_void_p]
    native.berrybits_cols.restype = ctypes.c_uint32
    native.berrybits_mode.argtypes = [ctypes.c_void_p]
    native.berrybits_mode.restype = ctypes.c_int
    native.berrybits_quantize_f32.argtypes = [ctypes.c_void_p, float_pointer]
    native.berrybits_quantize_f32.restype = ctypes.c_int
    native.berrybits_quantize_f64.argtypes = [ctypes.c_void_p, double_pointer]
    native.berrybits_quantize_f64.restype = ctypes.c_int
    native.berrybits_dequantize_f32.argtypes = [ctypes.c_void_p, float_pointer]
    native.berrybits_dequantize_f32.restype = ctypes.c_int
    native.berrybits_gemv.argtypes = [ctypes.c_void_p, float_pointer, float_pointer]
    native.berrybits_gemv.restype = ctypes.c_int
    native.berrybits_save.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    native.berrybits_save.restype = ctypes.c_int
    return native


_NATIVE = _load_native()
_FLOAT_PTR = ctypes.POINTER(ctypes.c_float)


def _flatten(values: Sequence[float] | Sequence[Sequence[float]]) -> list[float]:
    if not values:
        return []
    if isinstance(values[0], Sequence) and not isinstance(values[0], (str, bytes)):
        return [float(value) for row in values for value in row]
    return [float(value) for value in values]


class BerryBitsTensor:
    """Native BerryBits tensor with versioned `.blut` persistence."""

    def __init__(self, rows: int, cols: int, mode: str = "Q4_K_M_LUT_F2F"):
        if rows <= 0 or cols <= 0:
            raise ValueError("rows and cols must be positive")
        try:
            mode_name = mode.upper()
            mode_id = MODES.get(mode_name, _MODE_ALIASES.get(mode_name))
        except KeyError as exc:
            raise ValueError(f"unsupported BerryBits mode: {mode}") from exc
        self._handle = _NATIVE.berrybits_create(rows, cols, mode_id)
        if not self._handle:
            raise MemoryError("BerryBits native tensor allocation failed")
        self._closed = False

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> "BerryBitsTensor":
        instance = cls.__new__(cls)
        instance._handle = _NATIVE.berrybits_load(os.fsencode(path))
        if not instance._handle:
            raise ValueError(f"could not load BerryBits tensor: {path}")
        instance._closed = False
        return instance

    def _check_open(self):
        if self._closed:
            raise RuntimeError("BerryBits tensor is closed")

    @property
    def rows(self) -> int:
        self._check_open()
        return int(_NATIVE.berrybits_rows(self._handle))

    @property
    def cols(self) -> int:
        self._check_open()
        return int(_NATIVE.berrybits_cols(self._handle))

    @property
    def mode(self) -> str:
        self._check_open()
        mode_id = int(_NATIVE.berrybits_mode(self._handle))
        return next(name for name, value in MODES.items() if value == mode_id)

    def quantize(self, values: Sequence[float] | Sequence[Sequence[float]]) -> None:
        self._check_open()
        flat = _flatten(values)
        if len(flat) != self.rows * self.cols:
            raise ValueError("weights must contain rows * cols values")
        if self.mode == "Q64_LUT_F2F":
            native_values = (ctypes.c_double * len(flat))(*flat)
            ok = _NATIVE.berrybits_quantize_f64(self._handle, native_values)
        else:
            native_values = (ctypes.c_float * len(flat))(*flat)
            ok = _NATIVE.berrybits_quantize_f32(self._handle, native_values)
        if not ok:
            raise ValueError("BerryBits quantization failed")

    def dequantize(self) -> list[float]:
        self._check_open()
        output = (ctypes.c_float * (self.rows * self.cols))()
        if not _NATIVE.berrybits_dequantize_f32(self._handle, output):
            raise RuntimeError("BerryBits dequantization failed")
        return [float(value) for value in output]

    def gemv(self, input_values: Sequence[float]) -> list[float]:
        self._check_open()
        if len(input_values) != self.cols:
            raise ValueError("input length must equal cols")
        vector = (ctypes.c_float * self.cols)(*(float(value) for value in input_values))
        output = (ctypes.c_float * self.rows)()
        if not _NATIVE.berrybits_gemv(self._handle, vector, output):
            raise RuntimeError("BerryBits GEMV failed")
        return [float(value) for value in output]

    def save(self, path: str | os.PathLike[str]) -> None:
        self._check_open()
        if not _NATIVE.berrybits_save(self._handle, os.fsencode(path)):
            raise OSError(f"could not save BerryBits .blut tensor: {path}")

    def close(self) -> None:
        if not self._closed:
            _NATIVE.berrybits_destroy(self._handle)
            self._handle = None
            self._closed = True

    def __enter__(self) -> "BerryBitsTensor":
        self._check_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def __del__(self):
        if hasattr(self, "_closed") and not self._closed:
            self.close()


from .model import BerryBitsModel
from .runtime import BerryBitsRuntime

__all__ = ["BerryBitsModel", "BerryBitsRuntime", "BerryBitsTensor", "MODES"]
