# 2026-05-01 OpenCL Legacy Synchronize Crash

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

## Result

The legacy Q4_0 kernel compiled:

```text
ggml_opencl: legacy NVIDIA Q4_0 x F32 matmul kernel: true
```

The run then aborted during model/context setup:

```text
ggml_opencl: clEnqueueBarrierWithWaitList(backend_ctx->queue, 0, nullptr, &evt) error -36
```

## Interpretation

The NVIDIA 390xx OpenCL 1.1 device rejects the current backend synchronization
path. Legacy NVIDIA synchronization should use `clFinish`, which is available
on OpenCL 1.1, instead of `clEnqueueBarrierWithWaitList`.
