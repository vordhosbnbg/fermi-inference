# 2026-05-03 OpenCL Legacy Decode Attention Mask Skip

## Run Metadata

- Date: 2026-05-03, Europe/Sofia
- Kernel: `6.19.14-zen1-1-zen`
- NVIDIA driver: `390.157`
- OpenCL platform: `NVIDIA CUDA`
- OpenCL device: `GeForce GT 540M (OpenCL 1.1 CUDA)`
- Model: `models/Qwen3-0.6B-Q4_0.gguf`
- Model SHA-256: `da2572f16c06133561ce56accaa822216f2391ef4d37fba427801cd6736417d4`
- Prompt: `Give me a detailed description of what is OpenCL?`
- Context / generation / batch / ubatch: `-c 128`, `-n 64`, `-b 32`, `-ub 1`
- KV offload: enabled
- Output layer: forced to CPU with `LLAMA_FERMI_OPENCL_OUTPUT_CPU=1`
- Trace/profile: `GGML_OPENCL_NVIDIA_LEGACY_TRACE=1`, `GGML_OPENCL_NVIDIA_LEGACY_PROFILE=1`
- Q4 row tile: `8`

Raw run directories:

- Baseline before mask skip: `results/runs/2026-05-03-010335-opencl-fermi-sweep`
- Mask-skip probe: `results/runs/2026-05-03-014316-opencl-fermi-sweep`
- Attention workgroup 32 probe: `results/runs/2026-05-03-015423-opencl-fermi-sweep`
- Max-offload mask-skip probe: `results/runs/2026-05-03-015807-opencl-fermi-sweep`

The mask-skip probe ran with local uncommitted changes on top of llama.cpp
commit `7f13edc1a`.

## Results

| Run | `-ngl` | Prompt t/s | Generation t/s | Kernel exec ms | `MUL_MAT` ms | `FLASH_ATTN_EXT` ms | Attention avg us |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | `2` | `30.7` | `16.8` | `1226.465` | `851.511` | `362.257` | `4025.078` |
| mask skip | `2` | `32.1` | `16.1` | `921.171` | `851.455` | `56.940` | `632.670` |
| baseline | `4` | `16.5` | `11.2` | `3995.837` | `2876.013` | `1086.712` | `4024.860` |
| mask skip | `4` | `19.4` | `12.0` | `3079.166` | `2875.943` | `170.054` | `629.829` |

## Interpretation

The decode attention kernel still launches as `q1_kv256_h16_hkv8`, but the
causal/KV mask contains `-inf` entries for invalid cache slots. The previous
kernel computed the K dot for all `256` slots and added the mask afterward.
The mask-skip probe checks for exact `-inf` before the dot product and skips
both the K dot and V accumulation for invalid slots.

This is a clear kernel-level win:

- `-ngl 2`: attention time drops from `362.257 ms` to `56.940 ms`, an `84.3%`
  reduction.
- `-ngl 4`: attention time drops from `1086.712 ms` to `170.054 ms`, an
  `84.4%` reduction.
- Q4_0 matmul time is effectively unchanged, which isolates the improvement to
  the attention path.

The `-ngl 2` generation tokens/sec did not improve in this single run, so that
point should be treated as timing noise at the application level. The profile
data is still decisive: total kernel execution fell by about `25%`, and finish
synchronization time fell by about `26%`.

## Decision

Keep the mask skip in the legacy decode attention kernel. It removes wasted
work on masked KV slots and makes attention a much smaller part of low-offload
runtime.

After this change, `MUL_MAT` is again the dominant operation:

- `-ngl 2`: `MUL_MAT` is `851.455 ms`; attention is `56.940 ms`.
- `-ngl 4`: `MUL_MAT` is `2875.943 ms`; attention is `170.054 ms`.

The next narrow attention probe is workgroup size. The default remains `64`,
but the fork now has a runtime compile knob:

```bash
GGML_OPENCL_NVIDIA_LEGACY_ATTN_WG=32
```

Use the sweep wrapper after rebuilding:

```bash
python scripts/run-opencl-fermi-sweep.py --ngl 2 4 --tokens 64 --batch 32 --ubatch 1 --profile --q4-row-tile 8 --attn-wg 32
```

If `32` does not materially improve the `ggml_legacy_flash_attn_decode...`
profile line, attention should stop being the active optimization target for
now.

## Workgroup 32 Probe

The `GGML_OPENCL_NVIDIA_LEGACY_ATTN_WG=32` probe did not materially improve the
attention profile:

| WG | `-ngl` | Generation t/s | Kernel exec ms | `FLASH_ATTN_EXT` ms | Attention avg us |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `64` | `2` | `16.1` | `921.171` | `56.940` | `632.670` |
| `32` | `2` | `16.1` | `920.766` | `56.391` | `626.567` |
| `64` | `4` | `12.0` | `3079.166` | `170.054` | `629.829` |
| `32` | `4` | `10.7` | `3078.368` | `168.739` | `624.958` |

The tiny attention-kernel reduction is not enough to justify changing the
default. Keep `64` as the default workgroup size.

## Max-Offload Diagnostic

Run one maximum-offload sweep point with the mask skip and row tile 8 to see the
new full-offload profile. This is not expected to be the best operating point,
but it can show whether the bottleneck at high offload remains Q4_0 matmul,
host-side synchronization, or unsupported/residual graph splits.

Recommended command after rebuilding:

```bash
python scripts/run-opencl-fermi-sweep.py --ngl 10000 --tokens 64 --batch 32 --ubatch 1 --profile --q4-row-tile 8
```

Use the default attention workgroup size; do not pass `--attn-wg 32`.

The max-offload run is complete:

| Run | `-ngl` | Generation t/s | Support rejects | Kernel exec ms | `MUL_MAT` ms | `FLASH_ATTN_EXT` ms | Attention avg us |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pre-mask skip | `10000` | `2.3` | `60` | `38711.131` | `28292.250` | `10133.689` | `4021.305` |
| mask skip | `10000` | `2.9` | `60` | `30159.238` | `28290.911` | `1582.804` | `628.097` |

The 60 support rejects are `CPY` rejects, not attention rejects:

```text
op=CPY supports=[accepted=0,rejected=60] compute=[nodes=0,failed=0,kernels=0]
op=FLASH_ATTN_EXT supports=[accepted=336,rejected=0] compute=[nodes=2520,failed=0,kernels=2520]
```

These same `CPY` rejects were present in the pre-mask-skip max-offload run.
The sweep parser now includes `CPY` in the op table so these rejects are visible
instead of only appearing in the final support count.

The max-offload profile confirms that attention is no longer the primary
problem. `MUL_MAT` accounts for `28290.911 ms` out of `30159.238 ms` of device
kernel execution, while attention is down to `1582.804 ms`. Full offload remains
diagnostically useful, but it is still a poor operating point on the GT 540M.
The follow-up Q4_0 cols1 matmul specialization is tracked in
`docs/experiments/2026-05-03-opencl-legacy-q4-cols1-specialization.md`.
