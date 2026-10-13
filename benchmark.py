#!/usr/bin/env python3
import random
import time

from berrybits import BerryBitsRuntime


def make_teacher_vector(length: int):
    random.seed(7)
    return [((i % 7) - 3) * 0.37 + random.random() * 0.25 for i in range(length)]


def main():
    runtime = BerryBitsRuntime()
    print(f"Backend: {runtime.backend_name}")

    rows = 32
    cols = 256
    teacher = make_teacher_vector(cols)
    student = runtime.quantize_row(teacher)
    unpacked = runtime.dequantize_row(student)
    loss = runtime.teacher_error(teacher, unpacked)
    print(f"RMSE teacher-vs-quantized: {loss:.6f}")

    x = make_teacher_vector(cols)
    start = time.perf_counter()
    for _ in range(3000):
        runtime.matmul_fused(student, x)
    elapsed = time.perf_counter() - start
    print(f"native matmul benchmark: {elapsed:.4f}s for 3000 calls")

    rows_matrix = 32
    row_values = [make_teacher_vector(cols) for _ in range(rows_matrix)]
    start = time.perf_counter()
    runtime.threaded_matmul(rows_matrix, cols, row_values, threads=4)
    elapsed2 = time.perf_counter() - start
    print(f"threaded matmul benchmark: {elapsed2:.4f}s for {rows_matrix} rows")
    print("Library ready: berrybits / berrybits")


if __name__ == "__main__":
    main()
