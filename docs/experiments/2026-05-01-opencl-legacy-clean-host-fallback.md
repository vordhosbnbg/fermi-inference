# 2026-05-01 OpenCL Legacy Clean Run With Host Fallback

## Command

```bash
./build/llama.cpp-opencl-native/bin/llama-cli \
  --device GPUOpenCL \
  -m models/Qwen3-0.6B-Q4_0.gguf \
  -p "Answer in one sentence: what is OpenCL?" \
  -c 128 \
  -n 8 \
  -ngl 1 \
  -b 32 \
  -ub 1 \
  -nkvo \
  --single-turn \
  --reasoning off
```

## Relevant Output

```text
ggml_opencl: selected platform: 'NVIDIA CUDA'
ggml_opencl: device: 'GeForce GT 540M (OpenCL 1.1 CUDA)'
ggml_opencl: OpenCL driver: 390.157
ggml_opencl: vector subgroup broadcast support: false
ggml_opencl: device FP16 support: false
ggml_opencl: NVIDIA legacy mode enabled; only Q4_0 x F32 matmul is supported
ggml_opencl: legacy NVIDIA basic OpenCL C probe: true
ggml_opencl: legacy NVIDIA q4_0 half-storage probe: true
ggml_opencl: legacy NVIDIA Q4_0 x F32 matmul kernel: true
```

The model loaded, generated, and exited without an OpenCL teardown crash:

```text
OpenCL is a parallel computing framework that

[ Prompt: 36.8 t/s | Generation: 27.2 t/s ]
```

Memory breakdown:

```text
GPUOpenCL (GT 540M): model 0 MiB, context 0 MiB, compute 0 MiB
Host: total 432 MiB = model 403 MiB + context 28 MiB + compute 0 MiB
```

## Interpretation

This validates the OpenCL lifecycle fixes: device probing, kernel compilation,
model execution, and process shutdown all complete cleanly.

It does not validate useful GPU offload. Unlike the earlier teardown-crash run,
the memory breakdown shows no model weights resident on `GPUOpenCL`. The next
patch should focus on making the Q4_0 x F32 support predicate and kernel match
the tensor layouts seen by the llama.cpp model loader and scheduler, then check
for an OpenCL model-buffer allocation during load.
