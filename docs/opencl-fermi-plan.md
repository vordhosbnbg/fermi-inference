# Fermi OpenCL Bring-Up Plan

This plan targets the NVIDIA GeForce GT 540M through `opencl-nvidia-390xx`.
The goal is to make llama.cpp use the GPU for at least a small, measurable
slice of inference without destabilizing the host.

## Confirmed Device Limits

Captured in `docs/clinfo-opencl-nvidia-390xx-gt540m.txt`:

- Platform: `NVIDIA CUDA`
- Device: `GeForce GT 540M`
- Driver: `390.157`
- Device version: `OpenCL 1.1 CUDA`
- OpenCL C version: `OpenCL C 1.1`
- Global memory: `1.939 GiB`
- Max single allocation: `496.3 MiB`
- Local memory: `48 KiB`
- Max work group size: `1024`
- Image support: yes
- Half precision: not available
- Extensions: no `cl_khr_fp16`, no subgroup extension

These limits rule out the current upstream llama.cpp OpenCL backend as-is. It
accepts only Intel and Adreno/Qualcomm devices, requires OpenCL C 2.0 or newer,
checks OpenCL 2.x SVM capabilities, requires `cl_khr_fp16`, and uses subgroup
operations in many kernels.

## Repository Strategy

`third_party/llama.cpp` points at the fork:

```text
git@github.com:vordhosbnbg/llama.cpp.git
```

Keep upstream history available in the submodule. Do Fermi work on focused
branches in the fork, and only update the parent repository's submodule pointer
after a patch reaches a meaningful checkpoint.

Suggested branch name inside the submodule:

```bash
git -C third_party/llama.cpp switch -c fermi-opencl-legacy
```

## Implementation Track

Start from the pre-removal OpenCL/CLBlast backend, not the current backend.
Upstream removed the old backend at `b3086` and reintroduced the current
Adreno-oriented backend at `b4325`. The older backend is a better Fermi base
because it used OpenCL buffers, CLBlast, and subgroup-free kernels.

CLBlast is not currently installed on the host, so the first probe should not
make it a hard dependency. Use the old backend as source material for device
setup, buffer handling, and quantized matvec kernels. Add CLBlast later only if
we decide the general matrix-matrix path is worth testing and the user approves
the package dependency.

Initial probe:

1. Port only enough of the legacy OpenCL/CLBlast backend to current llama.cpp to
   register an OpenCL device and compile kernels.
2. First compile a tiny OpenCL C 1.1 kernel that exercises the half-storage
   layout used by GGUF quantized blocks. If the NVIDIA compiler rejects `half`
   without `cl_khr_fp16`, add explicit uint16-to-float conversion instead of
   using `half` in kernels.
3. Keep fp16 acceleration disabled; this GPU does not expose `cl_khr_fp16`.
4. Prefer Q4_0 matvec first, because the selected model is Q4_0 and the legacy
   backend already had quantized matvec kernels.
5. Keep KV cache and non-matrix ops on CPU at first.
6. Avoid image and OpenCL 2.x APIs unless `clinfo` proves a specific feature is
   available on this driver.

## Current Patch Checkpoint

The `fermi-opencl-legacy` branch in `third_party/llama.cpp` adds a
`NVIDIA_LEGACY` OpenCL family and the first narrow compute path:

- detects NVIDIA/GeForce/OpenCL CUDA platform devices;
- permits OpenCL C 1.1 for that family only;
- skips current Intel/Adreno kernel loading;
- compiles a required basic OpenCL C kernel;
- compiles an optional Q4_0 half-storage probe;
- compiles a legacy Q4_0 x F32 matmul kernel;
- keeps raw Q4_0 block buffers instead of the current OpenCL backend's
  struct-of-arrays conversion;
- advertises only Q4_0 x F32 `MUL_MAT` plus no-op view/reshape style ops.

The first probe succeeded on the GT 540M:

```text
ggml_opencl: legacy NVIDIA basic OpenCL C probe: true
ggml_opencl: legacy NVIDIA q4_0 half-storage probe: true
```

Probe test after rebuilding:

```bash
./build/llama.cpp-opencl-native/bin/llama-cli --list-devices
```

First runtime target:

```bash
./build/llama.cpp-opencl-native/bin/llama-cli \
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

The first Q4_0 runtime completed generation and then crashed during OpenCL
context teardown. The follow-up patch makes legacy NVIDIA cleanup explicit and
logs context-release failures as shutdown warnings instead of aborting after a
completed run.

The next run exposed another OpenCL 1.1 compatibility issue during model setup:
the driver rejected `clEnqueueBarrierWithWaitList` in backend synchronization.
Legacy NVIDIA synchronization now uses `clFinish` instead.

After that change, the same short prompt completed and exited cleanly, but the
memory breakdown showed all model weights on `Host` and `0 MiB` on
`GPUOpenCL`. This is a host fallback, not useful offload. The legacy Q4_0
kernel has been adjusted to use ggml byte strides for source and destination
tensors instead of requiring contiguous activation/output tensors, and it now
logs OpenCL buffer allocations of at least 1 MiB so the next run can confirm
whether the model loader is selecting the GPU buffer type.

The current checkpoint confirms real OpenCL offload. With `-fit off -ngl 100`,
`models/Qwen3-0.6B-Q4_0.gguf` places about 319 MiB of model weights on the GT
540M and completes generation. OpenCL device memory reporting has also been
patched: the backend now reports the device's `CL_DEVICE_GLOBAL_MEM_SIZE`
instead of `0 MiB`, and tracks llama.cpp OpenCL buffer allocations so the
memory breakdown no longer shows negative unaccounted memory.

This broad offload path is much slower than CPU-only inference. The observed
`-ngl 100` run produced about 0.8 generation tokens/sec. This is expected for
the current implementation because the legacy backend supports only Q4_0 x F32
`MUL_MAT`; unsupported graph nodes remain on CPU and can force synchronization
and transfer overhead between CPU and GPU. The next experiment should sweep low
offload counts (`-ngl 1`, `2`, `4`, `8`, `16`) and compare them against
CPU-only and full-offload results.

## Success Criteria

Minimum success:

- llama.cpp lists or initializes the GT 540M through OpenCL.
- OpenCL kernels compile on the NVIDIA 390xx compiler.

Meaningful success:

- `-ngl 1` completes with correct text.
- GPU memory usage changes during inference.
- CPU and OpenCL outputs are qualitatively consistent for short prompts.

Current status:

- OpenCL device detection: achieved.
- Kernel compilation on NVIDIA 390xx OpenCL C 1.1: achieved.
- Clean inference with nonzero OpenCL model memory: achieved.
- Performance improvement over CPU: not achieved.

Strong success:

- OpenCL offload improves generation speed over the CPU baseline.

## Known Risks

- The 496 MiB max allocation cap may prevent direct layer-buffer allocation
  unless tensors are split or only small allocations are moved.
- The lack of fp16 means many modern OpenCL kernels cannot be used directly.
- PCIe and kernel launch overhead may erase any speedup for tiny batch/token
  workloads.
- Porting the legacy backend may conflict with current ggml backend APIs and
  require a compatibility shim rather than a clean cherry-pick.
