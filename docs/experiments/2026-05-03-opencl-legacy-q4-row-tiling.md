# 2026-05-03 OpenCL Legacy Q4 Row Tiling

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

The raw run metadata records parent commit `2b5197b` and llama.cpp commit
`6649e5e27`, but the row-tile-8 run was made with the local uncommitted
row-tiling patch applied.

Raw run directories:

- `results/runs/2026-05-03-003952-opencl-fermi-sweep`: row tile 1
- `results/runs/2026-05-03-004317-opencl-fermi-sweep`: row tile 2
- `results/runs/2026-05-03-010242-opencl-fermi-sweep`: row tile 4 default
- `results/runs/2026-05-03-010335-opencl-fermi-sweep`: row tile 8 probe

## Results

| Row tile | `-ngl` | Prompt t/s | Generation t/s | Kernel exec ms | `MUL_MAT` ms | `FLASH_ATTN_EXT` ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `1` | `2` | `26.1` | `12.0` | `2287.699` | `1912.923` | `362.073` |
| `2` | `2` | `27.4` | `13.2` | `1766.850` | `1391.866` | `362.028` |
| `4` | `2` | `29.2` | `15.3` | `1340.911` | `966.181` | `362.028` |
| `8` | `2` | `30.7` | `16.8` | `1226.465` | `851.511` | `362.257` |
| `1` | `4` | `10.6` | `8.4` | `7583.582` | `6464.784` | `1086.738` |
| `2` | `4` | `11.8` | `8.5` | `5830.740` | `4710.426` | `1087.117` |
| `4` | `4` | `14.5` | `10.5` | `4387.959` | `3269.017` | `1086.122` |
| `8` | `4` | `16.5` | `11.2` | `3995.837` | `2876.013` | `1086.712` |

The CPU-only `-ngl 1` baseline in the row-tile-4 run was `20.5` generation
tokens/sec with no OpenCL kernels. The best GPU-offload point remains below
that baseline, but the gap is significantly smaller than before row tiling.

## Interpretation

Row tiling directly improves the legacy Q4_0 x F32 matmul kernel by reusing the
same activation vector across multiple output rows in one workgroup. The effect
is monotonic in the measured points:

| Comparison | `-ngl` | Generation change | Kernel time change | `MUL_MAT` time change |
| --- | ---: | ---: | ---: | ---: |
| row tile 4 vs 1 | `2` | `+27.5%` | `-41.4%` | `-49.5%` |
| row tile 8 vs 1 | `2` | `+40.0%` | `-46.4%` | `-55.5%` |
| row tile 8 vs 4 | `2` | `+9.8%` | `-8.5%` | `-11.9%` |
| row tile 4 vs 1 | `4` | `+25.0%` | `-42.1%` | `-49.4%` |
| row tile 8 vs 1 | `4` | `+33.3%` | `-47.3%` | `-55.5%` |
| row tile 8 vs 4 | `4` | `+6.7%` | `-8.9%` | `-12.0%` |

The row-tile-8 kernel compiled and executed on the GT 540M:

```text
ggml_opencl: legacy NVIDIA Q4_0 x F32 row-tile r8 matmul kernel: true
```

For `-ngl 2`, row tile 8 moves the top profiled kernel from Q4_0 matmul to
decode attention:

```text
profile_top_kernel=kernel_flash_attn_decode_f32_f16
profile_top_kernel_exec_ms=362.257
```

For `-ngl 4`, Q4_0 matmul remains the largest op family, but decode attention
is now a larger share of the total kernel time:

```text
MUL_MAT: 2876.013 ms
FLASH_ATTN_EXT: 1086.712 ms
```

## Decision

Keep row tile 4 as the default because it is already validated, robust, and
roughly halves Q4_0 matmul time. Keep row tile 8 as an explicit probe because it
adds another `6%` to `10%` generation improvement on the measured low-offload
points, but it is more register-heavy and should remain easy to disable.

The next performance target is decode attention and launch overhead, not
additional op coverage. The `-ngl 2` row-tile-8 run has zero support rejects,
small D2H traffic, and attention is now the single largest kernel.
