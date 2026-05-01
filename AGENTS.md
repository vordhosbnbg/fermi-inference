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
