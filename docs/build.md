# Build Notes

Builds should be run by the user in a normal local shell, not from a Codex
session. Optimized llama.cpp builds can saturate this laptop, especially with
LTO enabled.

## Optimized OpenCL Build

Working directory:

```bash
/home/vordhosbn/code/fermi-inference
```

Configure command:

```bash
cmake -S third_party/llama.cpp \
  -B build/llama.cpp-opencl-native \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=OFF \
  -DGGML_OPENCL=ON \
  -DGGML_OPENCL_PROFILING=OFF \
  -DGGML_OPENCL_EMBED_KERNELS=ON \
  -DGGML_OPENCL_USE_ADRENO_KERNELS=OFF \
  -DGGML_NATIVE=ON \
  -DGGML_LTO=ON \
  -DGGML_CCACHE=OFF \
  -DGGML_OPENMP=ON \
  -DGGML_LLAMAFILE=ON \
  -DGGML_CPU_REPACK=ON \
  -DGGML_BLAS=OFF \
  -DLLAMA_BUILD_TESTS=OFF \
  -DGGML_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF \
  -DGGML_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_SERVER=ON \
  -DLLAMA_BUILD_WEBUI=OFF \
  -DCMAKE_C_FLAGS_RELEASE='-O3 -DNDEBUG -march=native -mtune=native -fno-plt' \
  -DCMAKE_CXX_FLAGS_RELEASE='-O3 -DNDEBUG -march=native -mtune=native -fno-plt'
```

Build command:

```bash
nice -n 10 ionice -c 3 cmake --build build/llama.cpp-opencl-native --target llama-cli -j2
```

## Rationale

- `GGML_OPENCL=ON`: enables the experimental GPU path.
- `GGML_OPENCL_USE_ADRENO_KERNELS=OFF`: avoids Adreno-specific kernels on the
  NVIDIA Fermi GPU.
- `GGML_OPENCL_PROFILING=OFF`: avoids profiling overhead during inference.
- `GGML_OPENCL_EMBED_KERNELS=ON`: embeds OpenCL kernels into the executable.
- `GGML_NATIVE=ON`, `-march=native`, `-mtune=native`: optimize CPU code for the
  Sandy Bridge host CPU.
- `GGML_LTO=ON`: enables interprocedural/link-time optimization.
- `GGML_OPENMP=ON`: enables CPU threading support.
- `GGML_LLAMAFILE=ON`: enables llamafile CPU SGEMM code.
- `GGML_CPU_REPACK=ON`: keeps runtime CPU weight repacking enabled.
- `GGML_BLAS=OFF`: avoids linking reference BLAS, which is unlikely to beat
  llama.cpp native kernels on this host.
- `LLAMA_BUILD_SERVER=ON`: required by this upstream commit to build
  `llama-cli`.
- `LLAMA_BUILD_WEBUI=OFF`: avoids embedding/building the server web UI.

## Required Packages

The NVIDIA 390xx OpenCL package supplies the vendor ICD and runtime library, but
not the development headers required by CMake.

Observed split:

- `opencl-nvidia-390xx`: `/etc/OpenCL/vendors/nvidia.icd`,
  `/usr/lib/libnvidia-opencl.so.390.157`
- `opencl-headers`: `/usr/include/CL/cl.h` and related OpenCL headers
- `ocl-icd`: `/usr/lib/libOpenCL.so.1`

## Build Verification

Command:

```bash
./build/llama.cpp-opencl-native/bin/llama-cli --version
```

Observed:

```text
version: 8994 (aab68217b)
built with GNU 15.2.1 for Linux x86_64
```

The binary is OpenCL-linked and not CUDA-linked:

```text
GGML_CUDA=OFF
GGML_OPENCL=ON
libOpenCL.so.1
```

## OpenCL Device Probe

Command:

```bash
./build/llama.cpp-opencl-native/bin/llama-cli --list-devices
```

Observed from a host namespace before the Fermi fork patches:

```text
ggml_opencl: selected platform: 'NVIDIA CUDA'
ggml_opencl: device: 'GeForce GT 540M (OpenCL 1.1 CUDA)'
Unsupported GPU: GeForce GT 540M
ggml_opencl: drop unsupported device.
Available devices:
```

The `NVIDIA CUDA` string is the NVIDIA OpenCL platform name. The binary is not
initializing CUDA. Current upstream llama.cpp OpenCL rejects this GPU before
inference because it only accepts Adreno/Qualcomm and Intel devices, and then
requires OpenCL C 2.0 or newer.

The host OpenCL capability dump is recorded in
`docs/clinfo-opencl-nvidia-390xx-gt540m.txt`. It confirms OpenCL C 1.1, no
`cl_khr_fp16`, no subgroup extension, and a 496.3 MiB maximum single allocation.
See `docs/opencl-fermi-plan.md` for the Fermi bring-up strategy.

Observed after the Fermi fork patches:

```text
ggml_opencl: selected platform: 'NVIDIA CUDA'
ggml_opencl: device: 'GeForce GT 540M (OpenCL 1.1 CUDA)'
ggml_opencl: OpenCL driver: 390.157
ggml_opencl: device FP16 support: false
ggml_opencl: NVIDIA legacy mode enabled; only Q4_0 x F32 matmul is supported
ggml_opencl: global mem size: 1985 MB
ggml_opencl: legacy NVIDIA basic OpenCL C probe: true
ggml_opencl: legacy NVIDIA q4_0 half-storage probe: true
ggml_opencl: legacy NVIDIA Q4_0 x F32 matmul kernel: true
```
