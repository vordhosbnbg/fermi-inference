# Fermi Inference Experiment

This repository documents an experiment to run modern local inference workloads
on an NVIDIA GeForce GT 540M, a Fermi-generation laptop GPU, while preserving a
normal Intel/Wayland desktop path on Arch Linux.

The goal is measurement, not migration. The expected useful outcome is a clear
answer about what still works, what fails, and whether OpenCL offload has any
practical value on this hardware.

## Current Status

- The NVIDIA 390xx proprietary driver builds and loads on the current test
  kernel.
- `nvidia-smi` can see the GeForce GT 540M when booted into the isolated NVIDIA
  test entry.
- OpenCL enumeration works through the NVIDIA 390xx OpenCL runtime.
- `clpeak` has compiled and executed kernels successfully.
- llama.cpp CPU inference is expected to work; OpenCL offload remains unproven.

## Repository Layout

- `FERMI_INFERENCE_EXPERIMENT.md`: original experiment note and observed state.
- `docs/hardware.md`: target host and hardware constraints.
- `docs/software-stack.md`: operating system, driver, OpenCL, and runtime notes.
- `docs/boot-isolation.md`: boot-entry strategy and recovery rules.
- `docs/experiment-protocol.md`: manual benchmark protocol.
- `docs/reference/sources.md`: upstream references used to evaluate feasibility.
- `results/README.md`: result logging format.
- `third_party/llama.cpp`: llama.cpp upstream as a git submodule.
- `AGENTS.md`: safety rules for coding agents and automation.
- `CONTRIBUTING.md`: documentation and experiment contribution rules.

## Hard Rules

- Keep the default Intel/Wayland boot path recoverable.
- Do not globally blacklist `nouveau`.
- Do not make the NVIDIA GPU primary for Plasma Wayland.
- Do not treat CUDA as a practical path for this experiment.
- Do not commit model files or large raw run logs.
- Keep third-party changes isolated in `third_party/llama.cpp`; switch the
  submodule URL to a fork before carrying long-lived patches.

## Expected Outcome

CPU inference with a small GGUF model is likely. OpenCL offload is worth testing
because the GPU can compile and run OpenCL kernels, but acceleration is unlikely
given the OpenCL C 1.1 device compiler, 2 GiB VRAM, about 496 MiB maximum OpenCL
allocation, and measured memory bandwidth around 23 GB/s.

## License

No license has been selected yet. Choose one before publishing or inviting
external reuse.
