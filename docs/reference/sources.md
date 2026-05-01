# Sources

These references were used to review feasibility on May 1, 2026.

## NVIDIA

- Legacy CUDA GPU compute capability list:
  <https://developer.nvidia.com/cuda/gpus/legacy>
- CUDA 9.0 release notes:
  <https://docs.nvidia.com/cuda/archive/9.0/cuda-toolkit-release-notes/index.html>
- Current CUDA compute capability documentation:
  <https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html>

Key points:

- NVIDIA lists the GeForce GT 540M under compute capability `2.1`.
- CUDA 9.0 removed toolkit support for Fermi `sm_2.x`.
- `nvidia-smi --query-gpu=name,compute_cap` is the documented way to query
  compute capability when supported by the installed driver.

## llama.cpp

- OpenCL backend documentation:
  <https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/OPENCL.md>

Key points:

- The OpenCL backend is documented as Adreno-first.
- Linux support is documented, but verified hardware coverage is limited.
- `Q4_0` is the documented optimized quantization path.
- `Q4_K` support/optimization is not the safe first test target.
