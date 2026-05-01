# 2026-05-01 OpenCL Legacy Probe

## Command

```bash
./build/llama.cpp-opencl-native/bin/llama-cli --list-devices
```

## Result

```text
ggml_opencl: selected platform: 'NVIDIA CUDA'

ggml_opencl: device: 'GeForce GT 540M (OpenCL 1.1 CUDA)'
ggml_opencl: OpenCL driver: 390.157
ggml_opencl: vector subgroup broadcast support: false
ggml_opencl: device FP16 support: false
ggml_opencl: NVIDIA legacy probe mode enabled; compute kernels are disabled for this device
ggml_opencl: mem base addr align: 512
ggml_opencl: max mem alloc size: 496 MB
ggml_opencl: device max workgroup size: 1024
ggml_opencl: legacy NVIDIA basic OpenCL C probe: true
ggml_opencl: legacy NVIDIA q4_0 half-storage probe: true
ggml_opencl: default device: 'GeForce GT 540M (OpenCL 1.1 CUDA)'
Available devices:
  GPUOpenCL: GeForce GT 540M (0 MiB, 0 MiB free)
```

## Interpretation

The NVIDIA 390xx OpenCL compiler accepts basic OpenCL C 1.1 kernels and accepts
`half` storage with `vload_half` for Q4_0 block scales, even though the device
does not expose `cl_khr_fp16` arithmetic support.

The next checkpoint is a rebuild with the legacy Q4_0 x F32 matmul kernel
enabled.
