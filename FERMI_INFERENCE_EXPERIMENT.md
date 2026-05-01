# Fermi Inference Experiment

## Intent

Evaluate whether an old NVIDIA Fermi-generation laptop GPU can still be used for any useful local AI/inference work on a current Arch Linux software stack.

This is explicitly an experiment, not a migration plan. The goal is to discover what still works with modern kernel/userspace and the legacy proprietary NVIDIA driver, while keeping the normal Intel/Wayland desktop path recoverable.

## Target Hardware

Host: Dell XPS L502X laptop

Relevant GPU hardware:

- Intel 2nd Gen Core integrated graphics, primary display path
- NVIDIA GeForce GT 540M, Fermi generation
- NVIDIA PCI location: `0000:01:00.0`
- NVIDIA compute capability: `2.1`
- NVIDIA VRAM reported by OpenCL/NVIDIA-SMI: about 2 GiB

Other relevant system details:

- CPU: Intel Core i7-2670QM, 4 cores / 8 threads
- RAM: 7.6 GiB
- Root disk: Samsung SSD 850 PRO 512 GB

## Software Stack

Operating system:

- Arch Linux
- Kernel: `6.19.14-zen1-1-zen`
- Bootloader: systemd-boot
- Desktop session: Plasma Wayland on Intel iGPU

NVIDIA stack installed:

- `nvidia-390xx-dkms 390.157-21`
- `nvidia-390xx-utils 390.157-21`
- `opencl-nvidia-390xx 390.157-21`
- `clinfo 3.0.25.02.14-1`
- `ocl-icd 2.3.4-1`

DKMS status:

```text
nvidia/390.157, 6.19.14-zen1-1-zen, x86_64: installed
```

Optional GUI settings package:

- `nvidia-390xx-settings` depends on `gtk2`
- `gtk2` was no longer available as a prebuilt official package in the current Arch repo set
- A local `~/gitk2` gtk2 PKGBUILD was prepared to use a local `~/gtk` clone pinned to GTK `2.24.33`
- This package is not required for compute testing

## Boot Isolation

A separate systemd-boot entry was created:

```text
/boot/loader/entries/arch-zen-nvidia-test.conf
```

Entry contents:

```ini
title   Arch Linux Zen (NVIDIA proprietary test)
linux   /vmlinuz-linux-zen
initrd  /intel-ucode.img
initrd  /initramfs-linux-zen.img
options root=UUID=fe7e0e79-9a58-4ca6-8972-94fec0ae2bfa rw acpi_backlight=native modprobe.blacklist=nouveau nouveau.modeset=0 nvidia-drm.modeset=1
```

Default boot remains unchanged:

```ini
default arch-zen.conf
timeout 1
editor no
```

The test entry blocks `nouveau` only for that boot and allows the proprietary NVIDIA driver to bind the GT 540M. The normal boot remains the fallback.

## What Has Been Proven

The proprietary NVIDIA 390.157 kernel module builds and loads on the current Zen kernel.

Loaded modules observed:

```text
nvidia_drm
nvidia_modeset
nvidia_uvm
nvidia
```

The driver registers the GPU:

```text
Model:       GeForce GT 540M
GPU UUID:    GPU-d81b75c2-90fd-005c-071e-453ca5d1cdab
Bus Type:    PCIe
Bus Location: 0000:01:00.0
Device Minor: 0
```

`nvidia-smi` works from a host namespace with access to `/dev/nvidia*`:

```text
NVIDIA-SMI 390.157
Driver Version: 390.157
GPU: GeForce GT 540M
Memory: 4 MiB / 1985 MiB
Temperature: 54C
```

OpenCL enumeration works from a host namespace:

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
Max compute units: 2
Max clock: 1344 MHz
```

`clpeak` successfully compiled and executed kernels:

```text
Global memory bandwidth:
  float4: 23.53 GB/s

Single-precision compute:
  float2: 250.91 GFLOPS
  float4: 243.19 GFLOPS

Double-precision compute:
  double: 21.44 GFLOPS

Integer compute:
  int: about 85 GIOPS

Transfer bandwidth:
  enqueueWriteBuffer: 5.06 GB/s
  enqueueReadBuffer: 5.58 GB/s

Kernel launch latency:
  5.85 us
```

Conclusion so far: OpenCL compute is functional. The GT 540M is not merely detected; it can compile and execute OpenCL kernels.

## Known Constraints

Wayland rendering/offload:

- Rendering under Plasma Wayland is effectively off limits for this GPU/driver generation.
- `glxinfo` could not be made useful under Wayland.
- `DRI_PRIME=1` is a Mesa/open-driver path and is not expected to work with proprietary NVIDIA 390xx.
- Modern NVIDIA PRIME render offload variables are unlikely to work with 390xx/Fermi under Wayland.

CUDA:

- Modern CUDA is not viable.
- GT 540M is Fermi, compute capability 2.1.
- Fermi support was dropped from CUDA starting with CUDA 9.0.
- Installing an ancient CUDA 8-era stack on current Arch is outside the current experiment scope.

OpenCL:

- This is the best compute path.
- The stack exposes OpenCL platform version 1.2, but the device reports OpenCL C 1.1.
- Many modern ML runtimes will assume newer OpenCL, CUDA, Vulkan, or SYCL capabilities and may fail.
- Current llama.cpp OpenCL documentation describes the backend as Adreno-first, with limited verified Linux hardware coverage outside that path.
- For llama.cpp OpenCL, treat `Q4_0` as the safest first quantization target.

Memory:

- VRAM is about 2 GiB.
- Max OpenCL allocation is about 496 MiB.
- Any model must be very small and quantized.

Codex/sandbox note:

- Inside the Codex shell, unprivileged commands did not see `/dev/nvidia*`.
- Escalated/host-namespace commands did see:

```text
/dev/nvidia0
/dev/nvidiactl
/dev/nvidia-modeset
/dev/nvidia-uvm
/dev/nvidia-uvm-tools
```

- A normal local desktop shell should have direct access if device nodes are present and permissions are `0666`, as observed.

## Proposed Inference Experiment

Use `llama.cpp` as the experimental inference runtime because it can run small GGUF models on CPU and has an OpenCL backend. The CPU path provides a baseline; OpenCL offload is the experiment variable.

### 1. Build vendored llama.cpp With OpenCL

```bash
git submodule update --init --recursive third_party/llama.cpp
cmake -S third_party/llama.cpp -B build/llama.cpp-opencl -DGGML_OPENCL=ON
cmake --build build/llama.cpp-opencl -j"$(nproc)"
```

If OpenCL headers/libraries are missing, install the relevant Arch packages first. The OpenCL runtime itself is already installed via `opencl-nvidia-390xx` and `ocl-icd`.

### 2. Select a Small GGUF Model

Start with a tiny quantized model. Requirements:

- Around 1B parameters or smaller
- GGUF format
- Prefer pure `Q4_0` quantization
- Small context first, such as `-c 256` or `-c 512`

Candidate:

```bash
mkdir -p ~/models
cd ~/models
# Download a tiny pure Q4_0 GGUF model and record its source URL and SHA256.
```

Avoid starting with `Q4_K_M`; current llama.cpp OpenCL documentation lists `Q4_0` as the optimized path and `Q4_K` work as incomplete.

### 3. CPU Baseline

```bash
./build/llama.cpp-opencl/bin/llama-cli \
  -m ~/models/tinyllama-q4_0.gguf \
  -p "Explain OpenCL in one short paragraph." \
  -c 512 \
  -n 80
```

Record:

- Whether it runs
- Tokens per second
- Peak RAM usage if convenient

### 4. OpenCL Offload Attempt

Try the smallest offload first:

```bash
./build/llama.cpp-opencl/bin/llama-cli \
  -m ~/models/tinyllama-q4_0.gguf \
  -p "Explain OpenCL in one short paragraph." \
  -c 512 \
  -n 80 \
  -ngl 1
```

If that works, increase offload:

```bash
-ngl 2
-ngl 4
-ngl 8
-ngl 99
```

Record:

- Whether kernels compile
- Whether execution starts
- Any OpenCL compiler errors
- Any out-of-memory or max-allocation errors
- Tokens per second
- Whether `nvidia-smi` shows memory/activity during inference

### 5. Sanity Checks Before Running Inference

Run these in the NVIDIA test boot:

```bash
lsmod | grep -E 'nvidia|nouveau'
nvidia-smi
clinfo | grep -Ei 'platform|device|opencl|compute capability|global memory|max allocation'
```

Expected:

- `nvidia` modules present
- no `nouveau`
- `nvidia-smi` sees GT 540M
- `clinfo` sees one NVIDIA OpenCL platform and one GT 540M device

## Success Criteria

Minimum success:

- A small GGUF model runs on CPU.
- llama.cpp OpenCL build succeeds.
- OpenCL backend detects the NVIDIA OpenCL device.

Meaningful success:

- At least one transformer layer can be offloaded to OpenCL without crashing.
- GPU memory usage changes during inference.
- Output generation completes correctly.

Strong success:

- Partial or full OpenCL offload improves tokens/sec over CPU baseline.

Likely outcome:

- CPU inference works.
- OpenCL backend may fail due to old OpenCL C 1.1 compiler/runtime limits or memory allocation constraints.
- If OpenCL works, performance may still be worse than CPU because transformer inference is memory- and kernel-dispatch-heavy, and this GPU has only about 23 GB/s memory bandwidth and 2 GiB VRAM.

## Do Not Do Yet

- Do not globally blacklist `nouveau`; keep the per-boot test entry only.
- Do not make NVIDIA primary for Plasma Wayland.
- Do not spend time on Bumblebee/optirun for compute; OpenCL does not need X11/Bumblebee.
- Do not assume CUDA is viable on current Arch for this GPU.
- Do not remove the default boot entry.
