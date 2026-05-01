# Software Stack

## Operating System

- Distribution: Arch Linux
- Kernel under test: `6.19.14-zen1-1-zen`
- Bootloader: systemd-boot
- Desktop: Plasma Wayland on Intel iGPU

## NVIDIA Stack

Observed packages:

- `nvidia-390xx-dkms 390.157-21`
- `nvidia-390xx-utils 390.157-21`
- `opencl-nvidia-390xx 390.157-21`
- `clinfo 3.0.25.02.14-1`
- `ocl-icd 2.3.4-1`

Observed DKMS status:

```text
nvidia/390.157, 6.19.14-zen1-1-zen, x86_64: installed
```

## Proven Driver State

Observed NVIDIA modules:

```text
nvidia_drm
nvidia_modeset
nvidia_uvm
nvidia
```

Observed `nvidia-smi` state:

```text
NVIDIA-SMI 390.157
Driver Version: 390.157
GPU: GeForce GT 540M
Memory: 4 MiB / 1985 MiB
```

Observed OpenCL state:

```text
Platform Name: NVIDIA CUDA
Platform Version: OpenCL 1.2 CUDA 9.1.84
Device Name: GeForce GT 540M
Device Version: OpenCL 1.1 CUDA
Device OpenCL C Version: OpenCL C 1.1
Driver Version: 390.157
Compute Capability: 2.1
Global memory: 1.939 GiB
Max allocation: 496.3 MiB
```

## Runtime Direction

Use llama.cpp for inference experiments because it provides:

- a CPU baseline
- GGUF model support
- an OpenCL backend for the experimental GPU path

The OpenCL backend should be treated as a compatibility experiment on this GPU,
not as a supported or expected-fast path.

## Out of Scope

- Installing an ancient CUDA 8 stack on current Arch
- Making NVIDIA the desktop renderer
- Bumblebee or X11 rendering offload
- GUI-only NVIDIA settings packages unless needed for observation
