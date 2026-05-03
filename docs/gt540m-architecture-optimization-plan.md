# GT 540M Architecture and Optimization Plan

This note connects the measured Qwen3 `Q4_0` OpenCL profile to the actual GT
540M/Fermi execution model. The goal is to choose the next optimization path for
the `third_party/llama.cpp` legacy NVIDIA OpenCL fork, not to broaden model or
quantization scope.

## Hardware Facts

The GT 540M is a GF108/Fermi mobile GPU. Notebookcheck lists the GT 540M as
GF108-derived Fermi with 96 unified pipelines, 672 MHz core, 1344 MHz shader
clock, 900 MHz memory clock, and a 128-bit memory bus. Our local OpenCL capture
matches the important compute-side constraints: 2 compute units, OpenCL 1.1,
48 KiB local memory, no `cl_khr_fp16`, no subgroup extension, and a maximum
workgroup size of 1024.

For compute capability 2.1, NVIDIA documents each multiprocessor as having 48
CUDA cores, 8 SFUs, and 2 warp schedulers. The same CUDA guide also documents
the global occupancy limits for compute capability 2.x: 32-bit warp size, 8
resident blocks per SM, 48 resident warps per SM, 1536 resident threads per SM,
32K registers per SM, and up to 48 KiB shared memory per block.

The practical consequence is that this GPU is not just "a small GPU"; it is a
two-SM GPU where the per-SM block limit matters a lot. A kernel that launches
one 32-thread warp per workgroup can only place 8 active warps on an SM because
of the 8-block limit, even though the SM can hold up to 48 resident warps. That
is a hard latency-hiding problem.

Fermi global memory behavior also matters. NVIDIA documents 128-byte cache
lines for global loads cached in L1+L2 on compute capability 2.x, and notes
that L2-only mode uses 32-byte transactions. OpenCL does not give us CUDA load
cache modifiers here, so the safest assumption for the NVIDIA 390 OpenCL stack
is that awkward warp access patterns will over-fetch and depend heavily on the
L1/L2 behavior chosen by the driver compiler.

## Current Profile Checkpoint

Relevant recent runs:

- Generic row-tile-8 baseline after attention mask skip:
  `results/runs/2026-05-03-014316-opencl-fermi-sweep`
- `r8c1` low-offload probe:
  `results/runs/2026-05-03-023409-opencl-fermi-sweep`
- Max-offload probe that still used the old generic `r8` kernel:
  `results/runs/2026-05-03-021300-opencl-fermi-sweep`

The `r8c1` specialization is real and useful:

| Kernel path | `-ngl` | Generation t/s | Kernel exec ms | `MUL_MAT` ms | Attention ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| generic `r8` | `2` | `16.1` | `921.171` | `851.455` | `56.940` |
| `r8c1` | `2` | `18.7` | `539.608` | `470.043` | `56.825` |
| generic `r8` | `4` | `12.0` | `3079.166` | `2875.943` | `170.054` |
| `r8c1` | `4` | `15.4` | `1793.011` | `1589.732` | `170.138` |

Attention is no longer the main issue. In the `r8c1` run, Q4_0 matmul still
accounts for most device time:

- `-ngl 2`: `470.043 ms` of `539.608 ms`
- `-ngl 4`: `1589.732 ms` of `1793.011 ms`

The hottest shapes are decode GEMV shapes with `cols == 1`:

| Shape | `-ngl 2` count | `-ngl 2` avg us | `-ngl 4` count | `-ngl 4` avg us |
| --- | ---: | ---: | ---: | ---: |
| `k1024_rows3072_cols1` | `132` | `1375.795` | `492` | `1375.540` |
| `k1024_rows1024_cols1` | `180` | `465.974` | `540` | `466.397` |
| `k1024_rows2048_cols1` | `90` | `924.684` | `270` | `921.849` |
| `k3072_rows1024_cols1` | `66` | `1034.441` | `246` | `1029.813` |
| `k2048_rows1024_cols1` | `90` | `589.643` | `270` | `588.439` |

The timing is stable across offload depth, which suggests these are kernel-shape
costs rather than graph-level noise.

## Bottleneck Interpretation

The current `r8c1` kernel maps one workgroup to one 8-row output tile and one
decode column. The selected local size is tied to `K / 32`:

- `K=1024`: 32 workitems, one warp
- `K=2048`: 64 workitems, two warps
- `K=3072`: 128 workitems, four warps, with one mostly idle warp

This is a poor fit for GF108 occupancy:

| Shape | Current active warps per block | Max active warps per SM from 8-block limit | Share of 48-warp SM limit |
| --- | ---: | ---: | ---: |
| `K=1024` | `1` | `8` | `16.7%` |
| `K=2048` | `2` | `16` | `33.3%` |
| `K=3072` | `3` useful / `4` resident | `24` useful / `32` resident | `50.0%` useful |

That likely explains why the GPU remains slower than the CPU despite having many
ALUs. The hot `K=1024` kernels do not expose enough active warps per SM to hide
memory latency and instruction latency.

The weight access pattern is the second likely bottleneck. Raw GGUF `Q4_0`
stores each 32-value block as a 2-byte scale plus 16 quant bytes. In the current
kernel, adjacent workitems read adjacent Q4 blocks, so a warp walks memory with
an 18-byte per-thread stride for the quant bytes and separate row spans for the
eight tiled output rows. That is hostile to Fermi's 128-byte cached memory
transaction model and also leaves the compiler handling many byte-level loads.

Approximate effective throughput confirms that the kernel is not reaching
anything close to hardware memory bandwidth. For the `k1024_rows3072_cols1`
kernel, one call touches roughly 1.77 MiB of Q4_0 weight blocks plus about
1.50 MiB of activation data if each row tile reloads the activation vector. At
about 1.376 ms per call, that is only a few GiB/s of useful traffic before
counting extra over-fetch. The GT 540M DDR3 configuration is around 28.8 GB/s
on paper, so the gap points at occupancy, memory transaction shape, instruction
overhead, or driver compilation quality rather than pure DRAM bandwidth alone.

## Next Optimization Path

### 1. Implement a Warp-Packed `r8c1` Kernel

This is the highest-value next experiment because it directly attacks the
one-warp-per-block occupancy problem without changing tensor storage.

Create a new optional kernel for the same guarded case as `r8c1`:

- Q4_0 x F32
- row tile 8
- `cols == 1`
- no row tail
- contiguous activation/output
- `K` in the observed Qwen3 set: `1024`, `2048`, `3072`

Instead of one row tile per workgroup, put several independent row tiles in one
workgroup:

| `K` | Current mapping | Proposed mapping |
| ---: | --- | --- |
| `1024` | 1 warp computes 1 row tile | 4 warps compute 4 independent row tiles |
| `2048` | 2 warps compute 1 row tile | 4 warps compute 2 independent row tiles |
| `3072` | 3 useful warps compute 1 row tile inside a 128-thread block | 6 warps compute 2 independent row tiles, or 3 warps compute 1 row tile if 192-thread blocks misbehave |

This keeps the row tile at 8, avoids the register pressure that hurt row tile
16, and raises the useful active-warp ceiling:

- `K=1024`: from 8 useful warps/SM to up to 32 useful warps/SM
- `K=2048`: from 16 useful warps/SM to up to 32 useful warps/SM
- `K=3072`: from about 24 useful warps/SM to up to 48 useful warps/SM with the
  192-thread variant

Implementation notes:

- Keep it behind a runtime switch such as
  `GGML_OPENCL_NVIDIA_LEGACY_Q4_0_MUL_MAT_WARP_PACK=1`.
- The fork now has this opt-in switch and the sweep wrapper exposes it as
  `--q4-warp-pack`.
- The first implementation routed only auto-local-size `K=1024` shapes and
  measured positively in
  `results/runs/2026-05-03-111801-opencl-fermi-sweep`.
- The `K=2048` two-tile variant routed in
  `results/runs/2026-05-03-124724-opencl-fermi-sweep`, but regressed from the
  previous plain `r8c1` average of about `589 us` to about `603-608 us`, so it
  should stay disabled.
- The current implementation retargets the two-tile variant to auto-local-size
  `K=3072` shapes with a fixed 192-thread workgroup; do not combine the probe
  with `--q4-lws`.
- The K=3072 route measured positively in
  `results/runs/2026-05-03-130302-opencl-fermi-sweep`, but only by about
  `2-3%` per routed kernel launch. Keep it for now, but do not treat it as the
  main remaining win.
- The warp-sync reduction probe in
  `results/runs/2026-05-03-132214-opencl-fermi-sweep` did not improve total
  `MUL_MAT` and regressed the routed K=3072 kernel. Keep
  `GGML_OPENCL_NVIDIA_LEGACY_Q4_0_MUL_MAT_WARP_SYNC=1` as a diagnostic option
  only.
- Name profiled kernels with the row-groups-per-block, e.g.
  `ggml_legacy_mul_mat_q4_0_f32_r8c1wp4_lws128_k1024_rows3072_cols1`.
- The expected `K=3072` profile name is
  `ggml_legacy_mul_mat_q4_0_f32_r8c1wp2_lws192_k3072_rows1024_cols1`.
- Use the conservative barrier path for regular comparisons.
- Keep the existing `r8c1` kernel as the fallback.

Decision gate:

- `K=1024` passed: the top `k1024_rows3072_cols1` average fell by about 35%,
  and generation improved at both `-ngl 2` and `-ngl 4`.
- `K=2048` failed: keep the existing `r8c1_lws64` route for that shape.
- `K=3072` improved slightly but missed the original 10% gate.
- Warp-sync failed to move the top K=1024 kernels enough to justify using it.
- Stop this row-mapping/barrier path and move to storage-layout changes.

### 2. Add Fermi Q4_0 SoA Weight Conversion

If warp packing is positive or inconclusive, attack memory coalescing next.
The current legacy path intentionally keeps raw GGUF Q4_0 blocks. That was good
for bring-up, but the measured hot kernel is now mature enough to justify a
converted device layout.

The first storage-layout step should be conservative: split Q4_0 into scales
and quant bytes in the same tensor allocation, similar to the backend's existing
non-legacy Q4_0 conversion path:

```text
d: all fp16 scales
q: all 16-byte quant blocks
```

Then add an `r8c1` SoA kernel that reads quant bytes with aligned vector loads
instead of walking `struct block_q4_0` with an 18-byte stride.

Expected benefits:

- aligned 16-byte quant blocks instead of 18-byte structs
- simpler address arithmetic in the hot loop
- fewer scalar byte-load instructions
- better chance that Fermi coalesces warp loads into a small number of cache
  line requests

Keep this as a separate switch such as:

```bash
GGML_OPENCL_NVIDIA_LEGACY_Q4_0_SOA=1
```

Decision gate:

- Continue if the SoA path improves the `r8c1` or warp-packed kernel by at
  least 10% on the `K=1024` hot shapes.
- If SoA helps, consider a second-stage layout that is row-tile-major for the
  exact Qwen3 decode shapes.
- If SoA does not help, the main limit is more likely issue/latency/driver
  overhead than raw memory coalescing.

### 3. Revisit `K=3072` Local Sizing

The current power-of-two reduction makes `K=3072` use local size 128 for 96
Q4 blocks, leaving 32 lanes idle. A warp-packed kernel can avoid this by using
three useful warps per row tile.

Do not implement a broad arbitrary-size reduction first. Only support the known
`K=3072` case:

- 96-thread one-row-tile variant
- 192-thread two-row-tile variant

This is lower priority than `K=1024` because the largest total time is still in
the `K=1024` row-heavy FFN shapes, but it is a good follow-up once the reduction
structure is being touched.

### 4. Do Not Prioritize Attention or More Op Coverage Yet

Attention mask skipping already reduced the attention kernel by about 84%, and
the current low-offload `r8c1` profile has no support rejects. In the latest
valid low-offload run:

- `FLASH_ATTN_EXT` is about `56.8 ms` at `-ngl 2`
- `MUL_MAT` is about `470.0 ms` at `-ngl 2`
- `FLASH_ATTN_EXT` is about `170.1 ms` at `-ngl 4`
- `MUL_MAT` is about `1589.7 ms` at `-ngl 4`

Attention should become active again only after matmul has been cut enough that
attention is a comparable share of kernel time.

The `CPY` rejects seen in the max-offload run are useful to keep visible, but
they are not the next performance target. The max-offload point also needs to be
rerun with a binary that confirms the `r8c1` probe lines before it can guide
matmul work.

## Measurement Plan

After each kernel experiment, use the same controlled sweep:

```bash
python scripts/run-opencl-fermi-sweep.py --ngl 2 4 --tokens 64 --batch 32 --ubatch 1 --profile --q4-row-tile 8
```

For the warp-packed probe, add:

```bash
python scripts/run-opencl-fermi-sweep.py --ngl 2 4 --tokens 64 --batch 32 --ubatch 1 --profile --q4-row-tile 8 --q4-warp-pack
```

For diagnostic-only warp-synchronous reduction comparisons, add
`--q4-warp-sync`:

```bash
python scripts/run-opencl-fermi-sweep.py --ngl 2 4 10 --tokens 64 --batch 32 --ubatch 1 --profile --q4-row-tile 8 --q4-warp-pack --q4-warp-sync
```

Keep these settings:

- `LLAMA_FERMI_OPENCL_OUTPUT_CPU=1`
- `GGML_OPENCL_NVIDIA_LEGACY_TRACE=1`
- `GGML_OPENCL_NVIDIA_LEGACY_PROFILE=1`
- attention workgroup default `64`
- same prompt as the recent sweep

Compare at least:

- generation t/s
- `profile_MUL_MAT_exec_ms`
- top `k1024_rows3072_cols1` average microseconds
- top `k1024_rows1024_cols1` average microseconds
- top `k2048_rows1024_cols1` average microseconds
- top `k3072_rows1024_cols1` average microseconds
- support rejects
- kernel count

Then rerun max offload only after the low-offload result is positive and the log
contains:

```text
ggml_opencl: legacy NVIDIA Q4_0 matmul cols1 specialization: true
ggml_opencl: legacy NVIDIA Q4_0 x F32 row-tile r8 cols1 matmul kernel: true
```

## Recommended Order

1. Add a warp-packed `r8c1` kernel for `K=1024` only.
2. Sweep `-ngl 2 4`; decide using the hot `k1024` shapes.
3. Extend warp packing to `K=2048` and `K=3072` only if `K=1024` helps.
4. Add SoA Q4_0 conversion and an SoA `r8c1` kernel.
5. If SoA helps, combine SoA with warp packing.
6. Rerun max offload with confirmed `r8c1` or `r8c1` successor routing.
7. Re-evaluate whether attention is again visible after matmul drops.

This path is narrow, measurable, and aligned with the GT 540M's actual limits:
two SMs, an 8-block-per-SM limit, 32-thread warps, no subgroups, no fp16
arithmetic, and a memory path that punishes awkward byte-strided Q4 loads.

## Sources

- NVIDIA CUDA C Programming Guide 8.0, compute capability 2.x architecture and
  limits:
  <https://docs.nvidia.com/cuda/archive/8.0/cuda-c-programming-guide/>
- NVIDIA CUDA C Best Practices Guide 10.0, block-size and occupancy heuristics:
  <https://docs.nvidia.com/cuda/archive/10.0/cuda-c-best-practices-guide/index.html>
- NVIDIA Fermi Compute Architecture Whitepaper:
  <https://www.nvidia.com/content/PDF/fermi_white_papers/NVIDIA_Fermi_Compute_architecture_Whitepaper.pdf>
- Notebookcheck GT 540M hardware summary:
  <https://www.notebookcheck.net/NVIDIA-GeForce-GT-540M.41715.0.html>
- Local OpenCL capability capture:
  `docs/clinfo-opencl-nvidia-390xx-gt540m.txt`
- Current measured `r8c1` result:
  `docs/experiments/2026-05-03-opencl-legacy-q4-cols1-specialization.md`
