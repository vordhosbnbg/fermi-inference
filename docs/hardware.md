# Hardware

## Host

- Model: Dell XPS L502X laptop
- CPU: Intel Core i7-2670QM, 4 cores / 8 threads
- RAM: 7.6 GiB
- Root disk: Samsung SSD 850 PRO 512 GB

## Graphics

Primary desktop path:

- Intel 2nd Gen Core integrated graphics
- Plasma Wayland session

Experimental compute device:

- NVIDIA GeForce GT 540M
- Architecture: Fermi
- PCI location: `0000:01:00.0`
- CUDA compute capability: `2.1`
- VRAM reported by NVIDIA/OpenCL tooling: about 2 GiB
- OpenCL maximum allocation observed: about 496 MiB
- OpenCL compute units observed: 2

## Practical Constraints

This GPU predates the assumptions made by most current ML runtimes:

- no tensor cores
- no modern CUDA toolkit support
- old OpenCL device compiler
- small VRAM pool
- low memory bandwidth relative to transformer inference needs

The CPU is old but should remain the baseline path because llama.cpp CPU
inference is simpler, better supported, and easier to compare against.
