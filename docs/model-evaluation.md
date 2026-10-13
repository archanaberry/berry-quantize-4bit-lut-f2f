# Tiny-model evaluation plan

## Recommended teacher

Use [`roneneldan/TinyStories-33M`](https://huggingface.co/roneneldan/TinyStories-33M) as the first real model candidate. Its model card identifies a GPT-Neo causal LM trained on TinyStories, its config declares `torch_dtype: float32`, and it uses the GPT-Neo-125M tokenizer. The Hub currently publishes `pytorch_model.bin` (about 291 MB), not `model.safetensors`; after loading, verify and record the exact parameter count instead of treating the repository name as a formal count. The repository is MIT-licensed.

A secondary controlled research candidate is [`EleutherAI/pythia-70m-deduped`](https://huggingface.co/EleutherAI/pythia-70m-deduped). The model card documents the Pythia suite as a controlled checkpoint series and the 70M size. Its current Hub config declares FP16 and its SafeTensors checkpoint is about 166 MB. It is useful for reproducibility, but it is not an original FP32 checkpoint; promoting its FP16 values to FP32 does not restore precision that was discarded.

## Fair comparison

1. Pin the model and tokenizer repository revisions and record their commit hashes.
2. Load TinyStories' original FP32 checkpoint as the master teacher. Create the FP16 checkpoint only by rounding those exact same FP32 weights to IEEE FP16. Do not use separately trained FP32 and FP16 models.
3. Use the same tokenizer, fixed TinyStories train/validation split, context length, prompts, and inference settings for both precisions and Q4 students.
4. Distill two separate Q4 students from the FP32 teacher and the FP16 snapshot, using identical calibration samples, epochs, optimizer, and initialization policy.
5. Report held-out cross-entropy/perplexity, teacher-to-student KL divergence, next-token agreement, output/logit error, model size, load time, and tokens/second. Do not use training samples for the final rating.

## TinyStories pilot results

The evaluation harness is `scripts/evaluate_tinystories_q4.py`; run it with `make model-evaluate`. Its CPU-only dependencies are pinned separately in `requirements-model-eval.txt` so the native library stays lightweight. The script uses these fixed revisions:

- Teacher: `roneneldan/TinyStories-33M@2ad0a164221b7c4d21cac7c46aec74f6f98dbfc8`, loaded as FP32.
- Tokenizer: `EleutherAI/gpt-neo-125M@21def0189f5705e2521767faed922f1f15e7d7db`.
- Dataset: `roneneldan/TinyStories@f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`.

The checkpoint contains **68,514,048 parameters**. FP16 is made by converting that exact FP32 checkpoint, not by loading a separately trained model. The pilot calibrates on the first four streamed training stories and scores the same eight held-out validation stories at 48 tokens maximum. It converts 24 GPT-Neo linear projections to Berryfour Q4-LUT; embeddings, layer norms, biases, and the tied output head remain at teacher precision. Thus these are hybrid students, not whole-model 4-bit files.

To prepare the isolated CPU-only evaluator:

```sh
python3 -m venv /tmp/berrybits-model-eval
/tmp/berrybits-model-eval/bin/python -m pip install \
	--index-url https://download.pytorch.org/whl/cpu \
	--extra-index-url https://pypi.org/simple \
	-r requirements-model-eval.txt
make model-evaluate MODEL_EVAL_PYTHON=/tmp/berrybits-model-eval/bin/python
```

The run exports 24 per-linear `.b4lut` files and a `manifest.json` for each precision under `build/tinystories_q4_f2f/fp32/` and `build/tinystories_q4_f2f/fp16/`. Each `.b4lut` contains that linear layer's packed weights, per-block scales, and codebook LUT. The manifest maps files to GPT-Neo tensor names and records the teacher/data revisions. Biases, embeddings, norms, and the tied output head remain in the corresponding FP32/FP16 base checkpoint; the `.b4lut` layer set is not a standalone full-model checkpoint or GGUF file.

Measured on the current two-vCPU AMD EPYC VM:

| Model | Held-out PPL | Teacher KL | Next-token agreement | Throughput | Estimated parameters/storage |
| --- | ---: | ---: | ---: | ---: | ---: |
| FP32 teacher | 2.6546 | baseline | baseline | 318.69 tok/s | 274.1 MB |
| FP16 from same FP32 weights | 2.6548 | baseline | baseline | 88.44 tok/s | 137.0 MB |
| FP32 teacher -> Q4-LUT linear layers | 2.7450 | 0.051230 | 92.29% | 55.44 tok/s | 178.5 MB hybrid |
| FP16 teacher -> Q4-LUT linear layers | 2.7434 | 0.051413 | 91.22% | 41.66 tok/s | 98.1 MB hybrid |

The quantized projection payload is about 17.7 MB in either Q4 run, saved as the 24 `.b4lut` files. Greedy generation remained readable on the test prompt: the teachers produced “Once upon a time, a little rabbit named Benny was hopping through the forest. He saw a big”; both Q4 students produced “Once upon a time, a little rabbit named Benny was hopping around in the forest. He saw a”.

**Rating: promising for this small pilot, not production-validated.** Q4 PPL is about 3.4% above its teacher, and top-1 agreement is 91–92% on this tiny held-out subset. The current native path is slower than this VM's optimized PyTorch FP32/FP16 path, so it does not yet meet the “faster” goal. Token/s varies between runs; eight stories are far too few for a robust quality rating. Full-model package/load orchestration, large-scale perplexity, and a production tokenizer-aware native generation runtime are still future work.

The earlier `make demo-distill` still tests a synthetic **8-output by 32-input linear layer**, not the TinyStories model; do not mix its RMSE with the PPL table above.

## Sidecar LUT and loading failures

A global 16-entry FP32 codebook occupies 64 bytes. It should be loaded once at model initialization, validated (magic/version, codebook size/order, quantization format, group size, and model/tensor fingerprint), and kept resident for inference. Never reopen or parse the LUT for each tensor row or token.

A separate `.lut` file by itself does not make a GGUF or SafeTensors loader understand custom Q4 tensors. Those containers still need a matching tensor type/sidecar manifest and an inference backend that interprets the packed codes and per-group scales. The current `berrybits_model_load()` reads Berryfour's own `.b4lut` flat-matrix format; it does not load arbitrary `.gguf` or `.safetensors` files. Passing those files to it will fail, and interpreting their raw bytes as this custom format can produce nonsense output.

Likely causes of gibberish include a wrong tokenizer or tensor name/order, transposed matrices, FP16 bytes treated as FP32, nibble order mismatch, wrong group size/scale, mismatched LUT version, or applying the linear-layer test runtime as though it were a complete transformer. Validate one tensor first: FP32 -> packed Q4 -> dequantized FP32 error, then compare that layer's outputs on fixed inputs before testing full text generation.

References: [TinyStories model card](https://huggingface.co/roneneldan/TinyStories-33M), [TinyStories dataset](https://huggingface.co/datasets/roneneldan/TinyStories), [Pythia-70M model card](https://huggingface.co/EleutherAI/pythia-70m-deduped), [SafeTensors documentation](https://huggingface.co/docs/safetensors/index).
