# 2026-05-03 OpenCL Legacy Q4_0 Cols1 Matmul Specialization

## Rationale

After row tile 8 and decode-attention mask skipping, the full-offload profile is
dominated by Q4_0 x F32 decode GEMV:

```text
MUL_MAT: 28290.911 ms
FLASH_ATTN_EXT: 1582.804 ms
```

The hot matmul kernels are all `cols1` decode shapes:

```text
ggml_legacy_mul_mat_q4_0_f32_r8_lws32_k1024_rows3072_cols1
ggml_legacy_mul_mat_q4_0_f32_r8_lws128_k3072_rows1024_cols1
ggml_legacy_mul_mat_q4_0_f32_r8_lws32_k1024_rows1024_cols1
ggml_legacy_mul_mat_q4_0_f32_r8_lws32_k1024_rows2048_cols1
ggml_legacy_mul_mat_q4_0_f32_r8_lws64_k2048_rows1024_cols1
```

The fork now has a guarded `r8c1` kernel for the narrow case:

- Q4_0 x F32
- row tile `8`
- `cols == 1`
- no row tail (`rows % 8 == 0`)
- contiguous activation and output row stride
- `K / 32 <= local_size`

All other shapes fall back to the existing generic row-tile kernels.

## Probe Commands

Rebuild first:

```bash
cmake --build build/llama.cpp-opencl-native --target llama-cli -j 1
```

Low-offload probe:

```bash
python scripts/run-opencl-fermi-sweep.py --ngl 2 4 --tokens 64 --batch 32 --ubatch 1 --profile --q4-row-tile 8
```

Max-offload diagnostic:

```bash
python scripts/run-opencl-fermi-sweep.py --ngl 10000 --tokens 64 --batch 32 --ubatch 1 --profile --q4-row-tile 8
```

If the specialization regresses, disable it while keeping row tile 8:

```bash
python scripts/run-opencl-fermi-sweep.py --ngl 2 4 --tokens 64 --batch 32 --ubatch 1 --profile --q4-row-tile 8 --no-q4-cols1
```

The profile should show `r8c1` in the hot kernel names when the specialization
is active.

## Run Metadata

Raw run directories:

- Generic row-tile-8 baseline after attention mask skip:
  `results/runs/2026-05-03-014316-opencl-fermi-sweep`
- Low-offload `r8c1` probe:
  `results/runs/2026-05-03-023409-opencl-fermi-sweep`
- Max-offload probe needing rerun:
  `results/runs/2026-05-03-021300-opencl-fermi-sweep`

The low-offload probe confirms the new path was active:

```text
ggml_opencl: legacy NVIDIA Q4_0 matmul cols1 specialization: true
ggml_opencl: legacy NVIDIA Q4_0 x F32 row-tile r8 cols1 matmul kernel: true
```

The max-offload probe requested `q4_cols1=true` in metadata, but the log does
not contain the `cols1 specialization` or `row-tile r8 cols1` probe lines, and
the profile still names generic `r8` kernels. Treat that run as an old-binary
or pre-rebuild control point, not as a valid max-offload `r8c1` result.

## Results

| Kernel path | `-ngl` | Prompt t/s | Generation t/s | Kernel exec ms | `MUL_MAT` ms | `FLASH_ATTN_EXT` ms | Finish sync ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| generic `r8` | `2` | `32.1` | `16.1` | `921.171` | `851.455` | `56.940` | `867.880` |
| `r8c1` | `2` | `35.8` | `18.7` | `539.608` | `470.043` | `56.825` | `491.058` |
| generic `r8` | `4` | `19.4` | `12.0` | `3079.166` | `2875.943` | `170.054` | `2900.642` |
| `r8c1` | `4` | `24.5` | `15.4` | `1793.011` | `1589.732` | `170.138` | `1633.780` |

Measured impact:

| `-ngl` | Generation change | Kernel time change | `MUL_MAT` time change | Finish sync change |
| ---: | ---: | ---: | ---: | ---: |
| `2` | `+16.1%` | `-41.4%` | `-44.8%` | `-43.4%` |
| `4` | `+28.3%` | `-41.8%` | `-44.7%` | `-43.7%` |

Attention time is essentially unchanged. The improvement is isolated to the
Q4_0 decode matmul path.

## Hot Kernels

For `-ngl 2`, the same decode shapes are present before and after the change,
but the per-kernel execution time drops sharply:

| Shape | Generic `r8` avg us | `r8c1` avg us | Change |
| --- | ---: | ---: | ---: |
| `k1024_rows3072_cols1` | `2471.080` | `1375.795` | `-44.3%` |
| `k1024_rows1024_cols1` | `832.744` | `465.974` | `-44.0%` |
| `k1024_rows2048_cols1` | `1656.089` | `924.684` | `-44.2%` |
| `k3072_rows1024_cols1` | `1877.501` | `1034.441` | `-44.9%` |
| `k2048_rows1024_cols1` | `1137.953` | `589.643` | `-48.2%` |

The `-ngl 4` profile shows the same pattern, with the top `k1024_rows3072`
shape moving from `2471.026 us` to `1375.540 us`.

## Interpretation

The generic row-tile kernel still handles a matrix-shaped right-hand side even
when decode uses a single column. The `r8c1` kernel removes that extra generality
from the hot path: it has one activation column, one output column, and can keep
the row-tile structure focused on reusing that activation vector across eight
output rows.

The result is large enough to keep this specialization. It does not solve the
overall Fermi performance problem by itself because Q4_0 matmul remains the
dominant operation:

- `-ngl 2`: `MUL_MAT` is `470.043 ms` out of `539.608 ms` device kernel time.
- `-ngl 4`: `MUL_MAT` is `1589.732 ms` out of `1793.011 ms` device kernel time.

## Decision

Keep the `r8c1` specialization enabled for row tile 8. The guard is narrow
enough that non-decode and non-contiguous shapes still fall back to the generic
row-tile kernels, and the `--no-q4-cols1` sweep flag remains available for
regression checks.

Before drawing conclusions for maximum offload, rerun the max-offload point
with a rebuilt binary and confirm both of these appear in the log:

```text
ggml_opencl: legacy NVIDIA Q4_0 matmul cols1 specialization: true
ggml_opencl: legacy NVIDIA Q4_0 x F32 row-tile r8 cols1 matmul kernel: true
```

The profile top kernels should include `r8c1`, for example:

```text
ggml_legacy_mul_mat_q4_0_f32_r8c1_lws32_k1024_rows3072_cols1
```

If max offload shows the same per-kernel reduction, the next matmul target is
shape-specific tuning of the `r8c1` local sizes and row mapping rather than more
operator coverage.

The first follow-up implementation is an opt-in warp-packed `r8c1` path for
auto-local-size `K=1024` shapes. Probe it with:

```bash
python scripts/run-opencl-fermi-sweep.py --ngl 2 4 --tokens 64 --batch 32 --ubatch 1 --profile --q4-row-tile 8 --q4-warp-pack
```

The profile should show `r8c1wp4_lws128_k1024...` kernels if that path routes.

## Warp-Pack K=1024 Result

Run:

```text
results/runs/2026-05-03-111801-opencl-fermi-sweep
```

Settings: `--q4-row-tile 8 --q4-warp-pack`, auto Q4 local size, attention WG
64, `LLAMA_FERMI_OPENCL_OUTPUT_CPU=1`, `-ngl 2 4`, `-n 64`.

The opt-in `K=1024` warp-pack path routed and improved the dominant matmul
shape:

| `-ngl` | gen t/s | `MUL_MAT` exec ms | top `K=1024 rows3072` exec ms | top avg us |
| ---: | ---: | ---: | ---: | ---: |
| `2` | `17.9` | `350.270` | `116.863` | `885.329` |
| `4` | `15.6` | `1186.747` | `440.464` | `895.251` |

Compared with the prior `r8c1` row-tile-8 run
`results/runs/2026-05-03-023409-opencl-fermi-sweep`, `MUL_MAT` dropped from
`470.043` to `350.270` ms at `-ngl 2` and from `1589.732` to `1186.747` ms at
`-ngl 4`. The hot `K=1024 rows3072` kernel dropped from about `1.38 ms` per
launch to about `0.89 ms` per launch.

The remaining top Q4 shapes are unchanged:

- `ggml_legacy_mul_mat_q4_0_f32_r8c1_lws64_k2048_rows1024_cols1`
- `ggml_legacy_mul_mat_q4_0_f32_r8c1_lws128_k3072_rows1024_cols1`

The follow-up `K=2048` two-tile warp-pack did route in
`results/runs/2026-05-03-124724-opencl-fermi-sweep`, but it missed the decision
gate:

| `-ngl` | routed `K=2048` avg us |
| ---: | ---: |
| `2` | `603.188` |
| `4` | `606.824` |
| `10` | `608.202` |

That is slower than the prior plain `r8c1_lws64_k2048_rows1024_cols1` baseline
of about `589 us`, so the K=2048 warp-pack route should stay disabled.

Next implementation: retarget the two-tile warp-pack mechanism to the remaining
hot `K=3072` shape with a 192-thread workgroup. The expected routed profile
name is:

```text
ggml_legacy_mul_mat_q4_0_f32_r8c1wp2_lws192_k3072_rows1024_cols1
```

## Warp-Pack K=3072 Result

Run:

```text
results/runs/2026-05-03-130302-opencl-fermi-sweep
```

The K=3072 route compiled and routed, but the device-profile gain was modest:

| `-ngl` | routed `K=3072` avg us | previous avg us | change |
| ---: | ---: | ---: | ---: |
| `2` | `1014.740` | `1034.907` | `-1.9%` |
| `4` | `1005.436` | `1029.793` | `-2.4%` |
| `10` | `1001.373` | `1027.905` | `-2.6%` |

Overall `MUL_MAT` time also moved slightly in the right direction, but the top
kernel remains `K=1024 rows3072`. The next focused experiment is an opt-in
warp-synchronous reduction variant for the warp-packed kernels. Probe it with:

```bash
python scripts/run-opencl-fermi-sweep.py --ngl 2 4 10 --tokens 64 --batch 32 --ubatch 1 --profile --q4-row-tile 8 --q4-warp-pack --q4-warp-sync
```

## Warp-Sync Reduction Result

Run:

```text
results/runs/2026-05-03-132214-opencl-fermi-sweep
```

Settings: `--q4-row-tile 8 --q4-warp-pack --q4-warp-sync`, auto Q4 local size,
attention WG 64, `LLAMA_FERMI_OPENCL_OUTPUT_CPU=1`, `-ngl 2 4 10`, `-n 64`.

The warp-synchronous reduction path routed, but it does not beat the
conservative warp-pack path overall:

| `-ngl` | gen t/s | `MUL_MAT` exec ms | K=1024 rows3072 avg us | K=3072 rows1024 avg us |
| ---: | ---: | ---: | ---: | ---: |
| `2` | `19.3` | `349.707` | `885.741` | `1037.026` |
| `4` | `16.1` | `1182.238` | `892.166` | `1025.400` |
| `10` | `11.4` | `3680.478` | `892.129` | `1022.440` |

Compared with `results/runs/2026-05-03-130302-opencl-fermi-sweep`,
warp-sync slightly helped some K=1024 sub-shapes but regressed the routed K=3072
kernel and left total `MUL_MAT` essentially flat to slightly worse. Keep
`GGML_OPENCL_NVIDIA_LEGACY_Q4_0_MUL_MAT_WARP_SYNC` as a diagnostic option, but
do not use it as the default benchmark path.

This closes the current row-mapping/barrier branch. The next plausible matmul
direction is Q4_0 storage-layout work rather than more local-size or barrier
tuning.
