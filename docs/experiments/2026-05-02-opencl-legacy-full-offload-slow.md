# 2026-05-02 OpenCL Legacy Full Offload Is Slower Than CPU

## Command

```bash
./build/llama.cpp-opencl-native/bin/llama-cli \
  -fit off \
  --device GPUOpenCL \
  -m models/Qwen3-0.6B-Q4_0.gguf \
  -p "Answer in one sentence: what is OpenCL?" \
  -c 128 \
  -n 64 \
  -ngl 100 \
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
ggml_opencl: device FP16 support: false
ggml_opencl: NVIDIA legacy mode enabled; only Q4_0 x F32 matmul is supported
ggml_opencl: mem base addr align: 512
ggml_opencl: global mem size: 1985 MB
ggml_opencl: max mem alloc size: 496 MB
ggml_opencl: legacy NVIDIA basic OpenCL C probe: true
ggml_opencl: legacy NVIDIA q4_0 half-storage probe: true
ggml_opencl: legacy NVIDIA Q4_0 x F32 matmul kernel: true
```

The run completed and produced:

```text
OpenCL is a standard for writing software that can run on any platform that has a device with a compute unit, such as the GPU.

[ Prompt: 1.2 t/s | Generation: 0.8 t/s ]
```

Memory breakdown:

```text
GPUOpenCL (GT 540M): total 1985 MiB, free 1665 MiB, model 319 MiB, context 0 MiB, compute 0 MiB
Host: total 346 MiB = model 318 MiB + context 28 MiB + compute 0 MiB
```

## Interpretation

This is the first clean run proving broad model-weight offload through the
legacy NVIDIA OpenCL path. The OpenCL memory accounting patch also works: the
device now reports real global memory and tracked allocated memory instead of
`0 MiB` with negative unaccounted memory.

This is not a performance win. The legacy path only supports Q4_0 x F32
`MUL_MAT`; normalization, elementwise operations, sampling, and other graph
nodes still execute on CPU. With many layers offloaded, the scheduler likely
pays repeated CPU/GPU transfer and synchronization overhead around unsupported
ops. On this GT 540M plus 390xx driver stack, full offload drops generation to
about 0.8 tok/s.

## Next Step

Run an offload-count sweep with the same prompt and fixed parameters:

```text
-ngl 0
-ngl 1
-ngl 2
-ngl 4
-ngl 8
-ngl 16
-ngl 100
```

The likely useful result, if any, is a small offload count where one or a few
large Q4_0 matmuls move to the GPU without forcing excessive graph traffic back
and forth.
