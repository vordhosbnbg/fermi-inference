# 2026-05-01 OpenCL Legacy Q4_0 First Run

## Command

```bash
./build/llama.cpp-opencl-native/bin/llama-cli \
  --device GPUOpenCL \
  -m models/Qwen3-0.6B-Q4_0.gguf \
  -p "Answer in one sentence: what is OpenCL?" \
  -c 128 \
  -n 8 \
  -ngl 1 \
  -b 1 \
  -ub 1 \
  -nkvo \
  --single-turn \
  --reasoning off
```

## Result

The run used the previous probe-only binary. It printed:

```text
ggml_opencl: NVIDIA legacy probe mode enabled; compute kernels are disabled for this device
warning: no usable GPU found, --gpu-layers option will be ignored
```

It then aborted with:

```text
llama-context.cpp:1599: GGML_ASSERT(n_tokens_all <= cparams.n_batch) failed
```

## Interpretation

The missing line below indicates the binary did not include the Q4_0 compute
patch yet:

```text
ggml_opencl: legacy NVIDIA Q4_0 x F32 matmul kernel: true
```

The assertion is caused by `-b 1`, not by OpenCL. The prompt contains more than
one token, so `n_batch` must be large enough to accept the prompt batch. Use
`-b 32 -ub 1` for the next small offload run.
