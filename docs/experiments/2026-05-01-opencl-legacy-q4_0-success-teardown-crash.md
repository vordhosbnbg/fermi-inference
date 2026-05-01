# 2026-05-01 OpenCL Legacy Q4_0 Success With Teardown Crash

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
ggml_opencl: NVIDIA legacy mode enabled; only Q4_0 x F32 matmul is supported
ggml_opencl: legacy NVIDIA basic OpenCL C probe: true
ggml_opencl: legacy NVIDIA q4_0 half-storage probe: true
ggml_opencl: legacy NVIDIA Q4_0 x F32 matmul kernel: true
```

The model loaded and generated:

```text
OpenCL is a framework that provides a

[ Prompt: 28.2 t/s | Generation: 4.8 t/s ]
```

Memory breakdown:

```text
GPUOpenCL (GT 540M): model 83 MiB
Host: total 348 MiB = model 319 MiB + context 28 MiB + compute 0 MiB
```

The run then crashed during shutdown:

```text
ggml_opencl: clReleaseContext(ctx->context) error -34
```

## Interpretation

This is the first successful legacy Fermi OpenCL offload. At least one Q4_0
layer was resident on the GT 540M and generation completed before teardown.

The crash is a cleanup bug, not an inference failure. A follow-up patch releases
legacy kernel/program/queue resources explicitly and makes OpenCL context
release non-fatal during shutdown.
