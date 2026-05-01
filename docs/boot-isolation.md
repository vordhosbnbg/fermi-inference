# Boot Isolation

The experiment uses a dedicated systemd-boot entry for proprietary NVIDIA
testing. The normal boot path should remain unchanged and should continue using
the Intel iGPU for Plasma Wayland.

## Test Entry

Path:

```text
/boot/loader/entries/arch-zen-nvidia-test.conf
```

Observed contents:

```ini
title   Arch Linux Zen (NVIDIA proprietary test)
linux   /vmlinuz-linux-zen
initrd  /intel-ucode.img
initrd  /initramfs-linux-zen.img
options root=UUID=fe7e0e79-9a58-4ca6-8972-94fec0ae2bfa rw acpi_backlight=native modprobe.blacklist=nouveau nouveau.modeset=0 nvidia-drm.modeset=1
```

## Default Entry

Observed loader default:

```ini
default arch-zen.conf
timeout 1
editor no
```

## Recovery Rules

- Keep the default entry as the fallback.
- Keep `nouveau` blocking scoped to the NVIDIA test entry.
- Do not add a global modprobe blacklist for `nouveau`.
- Do not make the NVIDIA GPU primary for the Wayland desktop.
- Prefer rebooting into the default entry over debugging display issues from the
  test boot.

## Preflight Checks

Run these after booting the NVIDIA test entry:

```bash
lsmod | grep -E 'nvidia|nouveau'
nvidia-smi
clinfo | grep -Ei 'platform|device|opencl|compute capability|global memory|max allocation'
```

Expected state:

- NVIDIA modules are loaded.
- `nouveau` is not loaded.
- `nvidia-smi` sees the GeForce GT 540M.
- `clinfo` reports the NVIDIA OpenCL platform and GT 540M device.
