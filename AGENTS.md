# Agent Rules

This repository documents and runs experiments on legacy NVIDIA Fermi hardware.
Treat the host system as fragile and recoverability as a hard requirement.

## Safety Boundaries

- Do not globally blacklist `nouveau`.
- Do not remove or rewrite the default systemd-boot entry.
- Do not make the NVIDIA GPU the primary display path for Plasma Wayland.
- Do not install, remove, or upgrade system packages unless the user explicitly asks.
- Do not write to `/boot`, `/etc`, `/usr`, or other system paths from repository scripts.
- Do not put model binaries, generated build trees, or large raw benchmark output in git.
- Do not assume CUDA is viable for this hardware on a current Arch stack.
- Do not run expensive builds, especially llama.cpp/CMake/Ninja builds, from a Codex session. Provide the command and working directory for the user to run in a normal local shell.

## Experiment Discipline

Every recorded run should include:

- date and timezone
- kernel version
- NVIDIA driver version
- OpenCL platform and device details
- llama.cpp commit or release
- model filename, quantization, and hash
- exact command line
- prompt, context size, generation length, and offload settings
- tokens/sec and whether output completed correctly
- relevant stderr/stdout or a link to raw local logs

Prefer documentation and reproducible manual commands over automation that changes system state.

## Current Fermi/OpenCL State

- The active llama.cpp work is in `third_party/llama.cpp` on branch
  `fermi-opencl-legacy`, backed by the fork
  `git@github.com:vordhosbnbg/llama.cpp.git`.
- The host uses `opencl-nvidia-390xx` with NVIDIA driver `390.157`; the GT 540M
  exposes OpenCL 1.1, no `cl_khr_fp16`, and no subgroup support.
- The fork has a narrow legacy NVIDIA OpenCL path. It detects the GT 540M,
  compiles OpenCL C 1.1 probes, and supports only Q4_0 x F32 `MUL_MAT` plus
  simple view/no-op style graph nodes.
- Broad offload is confirmed but slow: `-ngl 100` on
  `models/Qwen3-0.6B-Q4_0.gguf` places about 319 MiB of model weights on
  `GPUOpenCL`, but observed generation is about 0.8 tok/s because unsupported
  ops and transfers dominate.
- Use `-fit off` for controlled OpenCL experiments. Sweep low offload counts
  such as `-ngl 1`, `2`, `4`, `8`, and `16` before trying full offload.
- Memory reporting for the OpenCL device is patched in the fork. A healthy run
  should show about `1985 MiB` total GPU memory and nonzero model memory when
  offload is active.

## Build Handoff

When a rebuild is needed, give the user the working directory and command. Do
not run this from a Codex session.

Working directory:

```bash
/home/vordhosbn/code/fermi-inference
```

Preferred incremental build:

```bash
cmake --build build/llama.cpp-opencl-native --target llama-cli -j 1
```
