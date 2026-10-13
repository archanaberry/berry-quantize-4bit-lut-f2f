#!/usr/bin/env python3
"""Small end-to-end TinyStories FP32/FP16 versus native Berryfour Q4-LUT eval."""

from __future__ import annotations

import copy
import json
import math
import time
from collections import defaultdict
from pathlib import Path
import sys

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from berrybits import BerryBitsModel

MODEL_ID = "roneneldan/TinyStories-33M"
MODEL_REVISION = "2ad0a164221b7c4d21cac7c46aec74f6f98dbfc8"
TOKENIZER_ID = "EleutherAI/gpt-neo-125M"
TOKENIZER_REVISION = "21def0189f5705e2521767faed922f1f15e7d7db"
DATASET_ID = "roneneldan/TinyStories"
DATASET_REVISION = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
MAX_LENGTH = 48
MAX_CALIBRATION_TOKENS = 16
CALIBRATION_STORIES = 4
VALIDATION_STORIES = 8
Q4_EPOCHS = 1
Q4_LEARNING_RATE = 0.005


class NativeQ4Linear(torch.nn.Module):
    def __init__(self, source: torch.nn.Linear, calibration_inputs: torch.Tensor):
        super().__init__()
        self.in_features = source.in_features
        self.out_features = source.out_features
        self.native_model = BerryBitsModel(self.out_features, self.in_features)

        weights = source.weight.detach().to(torch.float32).contiguous()
        calibration = calibration_inputs.to(torch.float32).contiguous()
        metrics = self.native_model._distill_float32_addresses(
            weights.data_ptr(),
            calibration.data_ptr(),
            calibration.shape[0],
            epochs=Q4_EPOCHS,
            learning_rate=Q4_LEARNING_RATE,
        )
        self.calibration_rmse_before = metrics["rmse_before"]
        self.calibration_rmse_after = metrics["rmse_after"]
        bias = source.bias.detach().clone() if source.bias is not None else None
        self.register_buffer("bias", bias)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        original_shape = input_tensor.shape
        if original_shape[-1] != self.in_features:
            raise ValueError("Q4 linear input width mismatch")
        flat = input_tensor.reshape(-1, self.in_features).to(torch.float32).contiguous()
        output = torch.empty(
            (flat.shape[0], self.out_features), dtype=torch.float32, device="cpu"
        )
        self.native_model._gemv_batch_addresses(
            flat.data_ptr(), flat.shape[0], output.data_ptr()
        )
        if self.bias is not None:
            output.add_(self.bias.to(torch.float32))
        output = output.reshape(*original_shape[:-1], self.out_features)
        return output.to(input_tensor.dtype)

    def close(self) -> None:
        self.native_model.close()


def stream_texts(split: str, count: int) -> list[str]:
    dataset = load_dataset(
        DATASET_ID,
        split=split,
        streaming=True,
        revision=DATASET_REVISION,
    )
    result = []
    for row in dataset:
        text = row.get("text", "").strip()
        if text:
            result.append(text)
        if len(result) == count:
            break
    if len(result) != count:
        raise RuntimeError(f"requested {count} texts from {split}, received {len(result)}")
    return result


def tokenize_texts(tokenizer, texts: list[str]) -> list[torch.Tensor]:
    encoded = []
    for text in texts:
        ids = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
            add_special_tokens=True,
        )["input_ids"]
        if ids.shape[1] > 1:
            encoded.append(ids.contiguous())
    return encoded


def model_parameter_bytes(model: torch.nn.Module) -> int:
    return sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())


def evaluate(model: torch.nn.Module, sequences: list[torch.Tensor]) -> dict[str, float]:
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    started = time.perf_counter()
    with torch.inference_mode():
        for input_ids in sequences:
            logits = model(input_ids).logits[:, :-1, :].float().contiguous()
            labels = input_ids[:, 1:].contiguous()
            total_nll += torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                reduction="sum",
            ).item()
            total_tokens += labels.numel()
    elapsed = time.perf_counter() - started
    mean_nll = total_nll / max(total_tokens, 1)
    return {
        "perplexity": math.exp(mean_nll),
        "nll": mean_nll,
        "tokens": float(total_tokens),
        "seconds": elapsed,
        "tokens_per_second": total_tokens / max(elapsed, 1e-9),
    }


def compare_to_teacher(
    teacher: torch.nn.Module,
    student: torch.nn.Module,
    sequences: list[torch.Tensor],
) -> dict[str, float]:
    teacher.eval()
    student.eval()
    kl_sum = 0.0
    exact_top1 = 0
    total_positions = 0
    with torch.inference_mode():
        for input_ids in sequences:
            teacher_logits = teacher(input_ids).logits[:, :-1, :].float()
            student_logits = student(input_ids).logits[:, :-1, :].float()
            teacher_log_probs = torch.log_softmax(teacher_logits, dim=-1)
            student_log_probs = torch.log_softmax(student_logits, dim=-1)
            teacher_probs = teacher_log_probs.exp()
            kl_sum += torch.sum(
                teacher_probs * (teacher_log_probs - student_log_probs)
            ).item()
            exact_top1 += (teacher_logits.argmax(-1) == student_logits.argmax(-1)).sum().item()
            total_positions += teacher_logits.shape[0] * teacher_logits.shape[1]
    return {
        "teacher_kl": kl_sum / max(total_positions, 1),
        "next_token_agreement": exact_top1 / max(total_positions, 1),
    }


def capture_calibration_inputs(
    teacher: torch.nn.Module,
    sequences: list[torch.Tensor],
) -> dict[str, torch.Tensor]:
    linears = {
        name: module
        for name, module in teacher.named_modules()
        if isinstance(module, torch.nn.Linear) and name != "lm_head"
    }
    captured: dict[str, list[torch.Tensor]] = defaultdict(list)
    captured_counts = defaultdict(int)
    handles = []

    for name, module in linears.items():
        def capture(_module, arguments, key=name):
            activation = arguments[0].detach().reshape(-1, arguments[0].shape[-1])
            remaining = MAX_CALIBRATION_TOKENS - captured_counts[key]
            if remaining > 0:
                chunk = activation[:remaining].to(torch.float32).contiguous()
                if chunk.numel():
                    captured[key].append(chunk)
                    captured_counts[key] += chunk.shape[0]

        handles.append(module.register_forward_pre_hook(capture))

    teacher.eval()
    with torch.inference_mode():
        for input_ids in sequences:
            teacher(input_ids)

    for handle in handles:
        handle.remove()
    return {
        name: torch.cat(chunks, dim=0).contiguous()
        for name, chunks in captured.items()
        if chunks
    }


def replace_linears_with_q4(
    teacher: torch.nn.Module,
    student: torch.nn.Module,
    calibration_sequences: list[torch.Tensor],
) -> tuple[torch.nn.Module, list[tuple[str, NativeQ4Linear]]]:
    activations = capture_calibration_inputs(teacher, calibration_sequences)
    teacher_modules = dict(teacher.named_modules())
    student_modules = dict(student.named_modules())
    replaced = []

    names = [
        name for name, module in teacher_modules.items()
        if isinstance(module, torch.nn.Linear) and name != "lm_head"
    ]
    for name in names:
        source = teacher_modules[name]
        parent_name, _, child_name = name.rpartition(".")
        parent = student_modules[parent_name] if parent_name else student
        calibration = activations.get(name)
        if calibration is None or calibration.shape[0] == 0:
            raise RuntimeError(f"no calibration activations captured for {name}")
        q4_layer = NativeQ4Linear(source, calibration)
        setattr(parent, child_name, q4_layer)
        replaced.append((name, q4_layer))

    return student, replaced


def generate_sample(model, tokenizer, prompt: str) -> str:
    model.eval()
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    attention_mask = torch.ones_like(input_ids)
    with torch.inference_mode():
        output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=12,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)


def close_q4_layers(layers: list[tuple[str, NativeQ4Linear]]) -> None:
    for _, layer in layers:
        layer.close()


def q4_hybrid_parameter_bytes(
    model: torch.nn.Module, layers: list[tuple[str, NativeQ4Linear]]
) -> int:
    native_bytes = 0
    for _, layer in layers:
        native_bytes += (
            layer.out_features * ((layer.in_features + 1) // 2)
            + layer.out_features * ((layer.in_features + 31) // 32) * 4
            + 16 * 4
        )
        if layer.bias is not None:
            native_bytes += layer.bias.numel() * layer.bias.element_size()
    return model_parameter_bytes(model) + native_bytes


def save_q4_artifacts(
    precision: str,
    layers: list[tuple[str, NativeQ4Linear]],
) -> tuple[Path, int, int]:
    output_dir = ROOT / "build" / "tinystories_q4_f2f" / precision
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format": "Berryfour-TinyStories-layer-set-v1",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "teacher_precision": precision,
        "tokenizer_id": TOKENIZER_ID,
        "tokenizer_revision": TOKENIZER_REVISION,
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "block_size": 32,
        "lut_entries": 16,
        "calibration_stories": CALIBRATION_STORIES,
        "max_calibration_tokens_per_layer": MAX_CALIBRATION_TOKENS,
        "student_training_epochs": Q4_EPOCHS,
        "student_learning_rate": Q4_LEARNING_RATE,
        "note": "Only listed linear projections are Q4. Other model tensors remain in teacher precision.",
        "layers": [],
    }
    total_bytes = 0
    for name, layer in layers:
        filename = name.replace(".", "__") + ".b4lut"
        model_path = output_dir / filename
        layer.native_model.save(model_path)
        file_bytes = model_path.stat().st_size
        total_bytes += file_bytes
        manifest["layers"].append({
            "name": name,
            "file": filename,
            "rows": layer.out_features,
            "cols": layer.in_features,
            "bias_precision": precision,
            "calibration_rmse_before": layer.calibration_rmse_before,
            "calibration_rmse_after": layer.calibration_rmse_after,
            "file_bytes": file_bytes,
        })
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return output_dir, len(layers), total_bytes + manifest_path.stat().st_size


def main() -> int:
    torch.set_num_threads(2)
    torch.manual_seed(0)
    torch.backends.mkldnn.enabled = True

    calibration_texts = stream_texts("train", CALIBRATION_STORIES)
    validation_texts = stream_texts("validation", VALIDATION_STORIES)
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_ID,
        revision=TOKENIZER_REVISION,
    )
    calibration_sequences = tokenize_texts(tokenizer, calibration_texts)
    validation_sequences = tokenize_texts(tokenizer, validation_texts)

    fp32_teacher = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        dtype=torch.float32,
    ).cpu().eval()
    parameter_count = sum(parameter.numel() for parameter in fp32_teacher.parameters())
    if parameter_count >= 100_000_000:
        raise RuntimeError(f"model exceeds 100M parameters: {parameter_count}")
    fp16_teacher = copy.deepcopy(fp32_teacher).to(dtype=torch.float16).eval()

    print(f"Model: {MODEL_ID}@{MODEL_REVISION}")
    print(f"Parameters: {parameter_count:,}; architecture: {fp32_teacher.config.model_type}")
    print(f"Tokenizer: {TOKENIZER_ID}@{TOKENIZER_REVISION}")
    print(f"Calibration stories: {len(calibration_sequences)}; validation stories: {len(validation_sequences)}")
    print(f"FP32 parameter bytes: {model_parameter_bytes(fp32_teacher):,}")
    print(f"FP16 parameter bytes: {model_parameter_bytes(fp16_teacher):,}")

    fp32_baseline = evaluate(fp32_teacher, validation_sequences)
    fp16_baseline = evaluate(fp16_teacher, validation_sequences)
    print(f"FP32 baseline: PPL {fp32_baseline['perplexity']:.4f}; {fp32_baseline['tokens_per_second']:.2f} tok/s")
    print(f"FP16 baseline: PPL {fp16_baseline['perplexity']:.4f}; {fp16_baseline['tokens_per_second']:.2f} tok/s")

    prompt = "Once upon a time, a little rabbit"
    print("FP32 sample:", generate_sample(fp32_teacher, tokenizer, prompt))
    print("FP16 sample:", generate_sample(fp16_teacher, tokenizer, prompt))

    for precision, teacher in (("fp32", fp32_teacher), ("fp16", fp16_teacher)):
        student = copy.deepcopy(teacher).eval()
        student, q4_layers = replace_linears_with_q4(
            teacher, student, calibration_sequences
        )
        q4_result = evaluate(student, validation_sequences)
        alignment = compare_to_teacher(teacher, student, validation_sequences)
        packed_size = sum(
            layer.out_features * ((layer.in_features + 1) // 2)
            + layer.out_features * ((layer.in_features + 31) // 32) * 4
            + 16 * 4
            for _, layer in q4_layers
        )
        hybrid_size = q4_hybrid_parameter_bytes(student, q4_layers)
        calibration_before = sum(layer.calibration_rmse_before for _, layer in q4_layers) / len(q4_layers)
        calibration_after = sum(layer.calibration_rmse_after for _, layer in q4_layers) / len(q4_layers)
        artifact_dir, artifact_count, artifact_bytes = save_q4_artifacts(precision, q4_layers)
        print(
            f"{precision}->Q4-LUT: PPL {q4_result['perplexity']:.4f}; "
            f"KL {alignment['teacher_kl']:.6f}; top1 {alignment['next_token_agreement']:.2%}; "
            f"{q4_result['tokens_per_second']:.2f} tok/s; packed linear bytes {packed_size:,}; "
            f"estimated hybrid model bytes {hybrid_size:,}; "
            f"saved {artifact_count} layer files ({artifact_bytes:,} bytes total); "
            f"mean layer-calibration RMSE {calibration_before:.5f}->{calibration_after:.5f}"
        )
        print(f"Q4 layer artifacts: {artifact_dir}")
        print(f"{precision}->Q4 sample:", generate_sample(student, tokenizer, prompt))
        close_q4_layers(q4_layers)
        del student

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
