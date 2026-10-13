from __future__ import annotations

import ctypes
import os
import subprocess
from pathlib import Path
from typing import Sequence

_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_LIBRARY = _ROOT / "build" / "libberrybits.so"


def _load_library():
    library_path = Path(os.environ.get("BERRYBITS_LIBRARY", _DEFAULT_LIBRARY))
    if not library_path.exists() and "BERRYBITS_LIBRARY" not in os.environ:
        subprocess.check_call(["make", "-C", str(_ROOT), "all"])
    library = ctypes.CDLL(str(library_path))
    float_pointer = ctypes.POINTER(ctypes.c_float)
    library.berrybits_model_create.argtypes = [ctypes.c_uint32, ctypes.c_uint32]
    library.berrybits_model_create.restype = ctypes.c_void_p
    library.berrybits_model_load.argtypes = [ctypes.c_char_p]
    library.berrybits_model_load.restype = ctypes.c_void_p
    library.berrybits_model_destroy.argtypes = [ctypes.c_void_p]
    library.berrybits_model_destroy.restype = None
    library.berrybits_model_rows.argtypes = [ctypes.c_void_p]
    library.berrybits_model_rows.restype = ctypes.c_uint32
    library.berrybits_model_cols.argtypes = [ctypes.c_void_p]
    library.berrybits_model_cols.restype = ctypes.c_uint32
    library.berrybits_model_quantize.argtypes = [ctypes.c_void_p, float_pointer]
    library.berrybits_model_quantize.restype = ctypes.c_int
    library.berrybits_model_gemv.argtypes = [ctypes.c_void_p, float_pointer, float_pointer]
    library.berrybits_model_gemv.restype = ctypes.c_int
    library.berrybits_model_gemv_batch.argtypes = [
        ctypes.c_void_p, float_pointer, ctypes.c_size_t, float_pointer,
    ]
    library.berrybits_model_gemv_batch.restype = ctypes.c_int
    library.berrybits_model_save.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    library.berrybits_model_save.restype = ctypes.c_int
    library.berrybits_model_distill_linear.argtypes = [
        ctypes.c_void_p, float_pointer, float_pointer, float_pointer,
        ctypes.c_size_t, ctypes.c_uint32, ctypes.c_float, float_pointer, float_pointer,
    ]
    library.berrybits_model_distill_linear.restype = ctypes.c_int
    return library


_NATIVE = _load_library()
_FLOAT_PTR = ctypes.POINTER(ctypes.c_float)


def _float_array(values: Sequence[float]):
    return (ctypes.c_float * len(values))(*(float(value) for value in values))


def _flatten(values: Sequence[float] | Sequence[Sequence[float]]):
    if not values:
        return []
    first = values[0]
    if isinstance(first, Sequence) and not isinstance(first, (str, bytes)):
        return [float(value) for row in values for value in row]
    return [float(value) for value in values]


class BerryBitsModel:
    """Native packed-Q4 LUT model with direct C ABI calls."""

    def __init__(self, rows: int, cols: int):
        if rows <= 0 or cols <= 0:
            raise ValueError("rows and cols must be positive")
        self._closed = False
        self._model = _NATIVE.berrybits_model_create(rows, cols)
        if not self._model:
            raise MemoryError("native Q4 model allocation failed")

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> "BerryBitsModel":
        model = cls.__new__(cls)
        model._closed = False
        model._model = _NATIVE.berrybits_model_load(os.fsencode(path))
        if not model._model:
            raise ValueError(f"could not load Berryfour model: {path}")
        return model

    @property
    def rows(self) -> int:
        self._ensure_open()
        return int(_NATIVE.berrybits_model_rows(self._model))

    @property
    def cols(self) -> int:
        self._ensure_open()
        return int(_NATIVE.berrybits_model_cols(self._model))

    def _ensure_open(self):
        if self._closed:
            raise RuntimeError("Berryfour model is closed")

    def quantize(self, weights: Sequence[float] | Sequence[Sequence[float]]) -> None:
        self._ensure_open()
        values = _flatten(weights)
        if len(values) != self.rows * self.cols:
            raise ValueError("weights must contain rows * cols values")
        array = _float_array(values)
        if not _NATIVE.berrybits_model_quantize(self._model, array):
            raise ValueError("native Q4 quantization failed; check for non-finite weights")

    def gemv(self, input_values: Sequence[float]) -> list[float]:
        self._ensure_open()
        if len(input_values) != self.cols:
            raise ValueError("input length must equal model cols")
        vector = _float_array(input_values)
        output = (ctypes.c_float * self.rows)()
        if not _NATIVE.berrybits_model_gemv(self._model, vector, output):
            raise RuntimeError("native Q4 GEMV failed")
        return [float(value) for value in output]

    def gemv_batch(self, input_values: Sequence[float], batch_size: int) -> list[float]:
        self._ensure_open()
        if batch_size <= 0 or len(input_values) != batch_size * self.cols:
            raise ValueError("input length must equal batch_size * cols")
        vector = _float_array(input_values)
        output = (ctypes.c_float * (batch_size * self.rows))()
        if not _NATIVE.berrybits_model_gemv_batch(
            self._model, vector, batch_size, output
        ):
            raise RuntimeError("native batched Q4 GEMV failed")
        return [float(value) for value in output]

    def _quantize_float32_address(self, address: int) -> None:
        self._ensure_open()
        pointer = ctypes.cast(ctypes.c_void_p(address), _FLOAT_PTR)
        if not _NATIVE.berrybits_model_quantize(self._model, pointer):
            raise ValueError("native tensor quantization failed")

    def _distill_float32_addresses(
        self,
        teacher_weights_address: int,
        calibration_inputs_address: int,
        samples: int,
        *,
        epochs: int = 100,
        learning_rate: float = 0.04,
    ) -> dict[str, float]:
        self._ensure_open()
        if samples <= 0 or epochs <= 0 or learning_rate <= 0:
            raise ValueError("samples, epochs, and learning_rate must be positive")
        teacher = ctypes.cast(ctypes.c_void_p(teacher_weights_address), _FLOAT_PTR)
        inputs = ctypes.cast(ctypes.c_void_p(calibration_inputs_address), _FLOAT_PTR)
        before = ctypes.c_float()
        after = ctypes.c_float()
        ok = _NATIVE.berrybits_model_distill_linear(
            self._model, teacher, _FLOAT_PTR(), inputs, samples, epochs,
            learning_rate, ctypes.byref(before), ctypes.byref(after),
        )
        if not ok:
            raise ValueError("native pointer-based linear distillation failed")
        return {"rmse_before": float(before.value), "rmse_after": float(after.value)}

    def _gemv_batch_addresses(
        self, inputs_address: int, batch_size: int, outputs_address: int
    ) -> None:
        self._ensure_open()
        inputs = ctypes.cast(ctypes.c_void_p(inputs_address), _FLOAT_PTR)
        outputs = ctypes.cast(ctypes.c_void_p(outputs_address), _FLOAT_PTR)
        if not _NATIVE.berrybits_model_gemv_batch(
            self._model, inputs, batch_size, outputs
        ):
            raise RuntimeError("native pointer-based batched GEMV failed")

    def save(self, path: str | os.PathLike[str]) -> None:
        self._ensure_open()
        if not _NATIVE.berrybits_model_save(self._model, os.fsencode(path)):
            raise OSError(f"could not save Berryfour model: {path}")

    def distill_linear(
        self,
        teacher_weights: Sequence[float] | Sequence[Sequence[float]],
        calibration_inputs: Sequence[float] | Sequence[Sequence[float]],
        *,
        epochs: int = 100,
        learning_rate: float = 0.04,
        teacher_bias: Sequence[float] | None = None,
    ) -> dict[str, float]:
        self._ensure_open()
        weights = _flatten(teacher_weights)
        inputs = _flatten(calibration_inputs)
        if len(weights) != self.rows * self.cols:
            raise ValueError("teacher weights must contain rows * cols values")
        if not inputs or len(inputs) % self.cols:
            raise ValueError("calibration inputs must be a non-empty multiple of cols")
        if epochs <= 0 or learning_rate <= 0:
            raise ValueError("epochs and learning_rate must be positive")
        if teacher_bias is not None and len(teacher_bias) != self.rows:
            raise ValueError("teacher_bias length must equal rows")

        teacher = _float_array(weights)
        input_array = _float_array(inputs)
        teacher_bias_array = _float_array(teacher_bias) if teacher_bias is not None else None
        teacher_bias_pointer = teacher_bias_array if teacher_bias_array is not None else _FLOAT_PTR()
        before = ctypes.c_float()
        after = ctypes.c_float()
        ok = _NATIVE.berrybits_model_distill_linear(
            self._model,
            teacher,
            teacher_bias_pointer,
            input_array,
            len(inputs) // self.cols,
            epochs,
            learning_rate,
            ctypes.byref(before),
            ctypes.byref(after),
        )
        if not ok:
            raise ValueError("native teacher/student distillation failed")
        return {"rmse_before": float(before.value), "rmse_after": float(after.value)}

    def close(self) -> None:
        if not self._closed:
            _NATIVE.berrybits_model_destroy(self._model)
            self._model = None
            self._closed = True

    def __enter__(self) -> "BerryBitsModel":
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def __del__(self):
        if hasattr(self, "_closed") and not self._closed:
            self.close()


__all__ = ["BerryBitsModel"]
