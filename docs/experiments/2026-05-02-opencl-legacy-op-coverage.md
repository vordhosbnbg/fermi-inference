# 2026-05-02 OpenCL Legacy Op Coverage and Low-Offload Sweep

## Run Metadata

- Date: 2026-05-02, Europe/Sofia
- Kernel: `6.19.14-zen1-1-zen`
- NVIDIA driver: `390.157`
- OpenCL platform: `NVIDIA CUDA`
- OpenCL device: `GeForce GT 540M (OpenCL 1.1 CUDA)`
- OpenCL C target: `OpenCL C 1.1`
- Device limits reported by run: `1985 MiB` global memory, `496 MiB` max allocation, `1024` max workgroup size
- FP16 support: false
- Subgroup/vector broadcast support: false
- llama.cpp build: `b9005-b57f9d327`
- llama.cpp fork commit: `b57f9d327`
- Model: `models/Qwen3-0.6B-Q4_0.gguf`
- Model SHA-256: `da2572f16c06133561ce56accaa822216f2391ef4d37fba427801cd6736417d4`

## Command

```bash
GGML_OPENCL_NVIDIA_LEGACY_TRACE=1 ./build/llama.cpp-opencl-native/bin/llama-cli \
  -fit off \
  --device GPUOpenCL \
  -m models/Qwen3-0.6B-Q4_0.gguf \
  -p "Answer in one sentence: what is OpenCL?" \
  -c 128 \
  -n 8 \
  -b 32 \
  -ub 1 \
  -nkvo \
  --single-turn \
  --reasoning off \
  -ngl 3
```

Settings of interest:

- prompt length target: one short sentence
- context size: `128`
- generation length: `8`
- prompt batch: `32`
- physical microbatch: `1`
- KV offload: disabled with `-nkvo`
- GPU layer count: `3` for the primary checkpoint; follow-up snippets compare
  `-ngl 2`, `3`, and `4`
- fit adjustment: disabled with `-fit off`

## Output

The run completed and exited cleanly. With `-n 8`, the response was truncated by
the token limit:

```text
OpenCL is a common programming language used
```

Throughput:

```text
[ Prompt: 9.2 t/s | Generation: 2.4 t/s ]
```

Memory breakdown:

```text
GPUOpenCL (GT 540M): total 1985 MiB, free 1884 MiB, self/model/context/compute = 100/100/0/0 MiB
Host: total 331 MiB = model 303 MiB + context 28 MiB + compute 0 MiB
```

## Trace Summary

Final legacy trace:

```text
graphs=99 nodes=1219 supports=[queries=4310,accepted=4308,rejected=2]
kernels=1219 buffers=[count=2,bytes=105844224] tensors=456
transfers=[h2d=199/106043564B,d2h=208/7158784B]
sync_other=1990 finishes=591
```

Per-op support and execution:

| Op | Supports | Compute nodes | Kernels | Notes |
| --- | ---: | ---: | ---: | --- |
| `ADD` | accepted `60`, rejected `0` | `119` | `119` | residual adds now run on OpenCL |
| `MUL` | accepted `1365`, rejected `0` | `251` | `251` | scale/broadcast multiplies now run on OpenCL |
| `RMS_NORM` | accepted `108`, rejected `0` | `251` | `251` | F32 contiguous-row RMS norm now runs on OpenCL |
| `MUL_MAT` | accepted `2379`, rejected `0` | `403` | `403` | Q4_0 x F32 matmul path |
| `GET_ROWS` | accepted `36`, rejected `0` | `20` | `20` | Q4_0 token embedding gather now runs on OpenCL |
| `ROPE` | accepted `48`, rejected `0` | `132` | `132` | F32 normal/NeoX RoPE now runs on OpenCL |
| `GLU` / `SWIGLU` | accepted `24`, rejected `0` | `43` | `43` | SwiGLU now runs on OpenCL |
| `FLASH_ATTN_EXT` | accepted `0`, rejected `2` | `0` | `0` | still intentionally unsupported |

## Low-Offload Sweep Start

Follow-up trace snippets were captured for the same command shape while changing
only the offloaded layer count to `-ngl 2`, `3`, and `4`. The snippets did not
repeat the command header, so this table inherits the same model, build, prompt,
context, batch, microbatch, `-nkvo`, and trace settings from the primary
checkpoint.

Summary:

| `-ngl` | Prompt t/s | Generation t/s | GPU model MiB | Host model MiB | Graphs | Kernels | Rejected supports | H2D transfers | D2H transfers | Finishes |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `2` | `14.7` | `2.6` | `91` | `311` | `66` | `559` | `1` | `155` / `96916652 B` | `109` / `6618112 B` | `426` |
| `3` | `10.4` | `2.4` | `100` | `303` | `99` | `1219` | `2` | `199` / `106043564 B` | `208` / `7158784 B` | `591` |
| `4` | `8.0` | `2.2` | `108` | `294` | `132` | `1879` | `3` | `243` / `115170476 B` | `307` / `7699456 B` | `756` |

The repeated `-ngl 3` sweep snippet reported `10.4` prompt tokens/sec, while
the primary full output above reported `9.2` prompt tokens/sec. The generation
rate and trace totals matched the same checkpoint shape.

Per-op compute node counts:

| `-ngl` | `ADD` | `MUL` | `RMS_NORM` | `MUL_MAT` | `GET_ROWS` | `ROPE` | `GLU` / `SWIGLU` | `FLASH_ATTN_EXT` rejected |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `2` | `53` | `119` | `119` | `172` | `20` | `66` | `10` | `1` |
| `3` | `119` | `251` | `251` | `403` | `20` | `132` | `43` | `2` |
| `4` | `185` | `383` | `383` | `634` | `20` | `198` | `76` | `3` |

The sweep shows a very regular per-layer cost after `-ngl 2`. Each additional
offloaded layer adds roughly:

- `660` OpenCL kernels
- `44` H2D transfers
- `99` D2H transfers
- `165` `clFinish` calls
- one additional unsupported `FLASH_ATTN_EXT` fallback
- about `8` to `9 MiB` of additional GPU model memory

Throughput moves in the wrong direction across these three points:

```text
-ngl 2: 2.6 tok/s
-ngl 3: 2.4 tok/s
-ngl 4: 2.2 tok/s
```

This makes `-ngl 2` the best of the measured low-offload points so far, but it
is still not a practical speedup. The added layers are executing their supported
non-attention ops on OpenCL, yet each layer also introduces another attention
fallback boundary and many more launch/readback/synchronization events.

## Attributed Transfer Rerun

After the transfer-attribution patch at llama.cpp fork commit `16dad336c`, the
same `-ngl 2`, `3`, and `4` points were rerun. The generation rates are
effectively unchanged, but the new summary explains the boundary costs.

Summary:

| `-ngl` | Prompt t/s | Generation t/s | D2H transfers | D2H bytes | `sync_other` waits | `sync_other` skipped | `clFinish` synchronize |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `2` | `10.9` | `2.6` | `109` | `6618112` | `0` | `1099` | `426` |
| `3` | `10.4` | `2.5` | `208` | `7158784` | `0` | `1990` | `591` |
| `4` | `8.0` | `2.3` | `307` | `7699456` | `0` | `2881` | `756` |

The old `sync_other` number is confirmed to be bookkeeping only for these runs:
there were no cross-device waits. All recorded `clFinish` calls were reported
as `reason=synchronize`, so the remaining synchronization cost comes from
backend/scheduler boundaries and blocking transfer paths, not from
multi-OpenCL-device ordering.

D2H transfer attribution:

| Source | Count / bytes at `-ngl 2` | Scaling | Interpretation |
| --- | ---: | --- | --- |
| `result_output` from `MUL_MAT` | `10` / `6077440` | constant | full-vocabulary logits readback from the GPU output layer |
| `Qcur-* (view) (permuted)` from `PERMUTE` | `33` / `270336` per offloaded repeating layer | per layer | Q readback for CPU attention fallback |
| `Kcur-* (view)` from `VIEW` | `33` / `135168` per offloaded repeating layer | per layer | K readback for CPU attention fallback |
| `Vcur-* (view)` from `VIEW` | `33` / `135168` per offloaded repeating layer | per layer | V readback for CPU attention fallback |

This matches the totals exactly:

```text
D2H count = 10 + 99 * offloaded_repeating_layers
D2H bytes = 6077440 + 540672 * offloaded_repeating_layers
```

For these low `-ngl` runs, llama.cpp counts the output layer as the first
offloaded layer. Therefore:

```text
-ngl 2 => output layer + blk.27
-ngl 3 => output layer + blk.26 + blk.27
-ngl 4 => output layer + blk.25 + blk.26 + blk.27
```

The H2D transfer summary also shows that `output.weight` is the largest model
upload by far:

```text
output.weight: 87515136 bytes
```

That output-layer placement creates the constant `result_output` readback:
`10` transfers of `607744` bytes each, corresponding to full F32 logits for
the model vocabulary.

## Output-on-CPU Rerun

The output-placement experiment at llama.cpp fork commit `5f50be4c7` added:

```text
LLAMA_FERMI_OPENCL_OUTPUT_CPU=1
```

With that switch enabled, `output.weight` stays on CPU while the selected
repeating layers still use `GPUOpenCL`.

Control and forced-output results:

| Run | Prompt t/s | Generation t/s | GPU model MiB | Host model MiB | Kernels | D2H transfers | D2H bytes | `FLASH_ATTN_EXT` rejected |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| plain `-ngl 1` | `23.7` | `2.9` | `83` | `319` | `40` | `10` | `6077440` | `0` |
| output CPU `-ngl 2` | `18.0` | `11.9` | `8` | `394` | `539` | `109` | `581632` | `1` |
| output CPU `-ngl 3` | `11.9` | `9.2` | `16` | `386` | `1199` | `208` | `1122304` | `2` |
| output CPU `-ngl 4` | `8.9` | `7.5` | `25` | `378` | `1859` | `307` | `1662976` | `3` |

The output-only control confirms that GPU output projection is a poor trade on
this hardware. Plain `-ngl 1` uploads `output.weight` (`87515136` bytes), then
reads back full-vocabulary `result_output` tensors:

```text
result_output: 10 transfers, 6077440 bytes
```

Forcing the output layer to CPU removes that full-vocabulary readback and
replaces it with a small final-norm readback:

```text
norm: 10 transfers, 40960 bytes
```

The remaining D2H traffic is now almost entirely attention fallback:

```text
D2H count = 10 + 99 * offloaded_repeating_layers
D2H bytes = 40960 + 540672 * offloaded_repeating_layers
```

This changes the performance conclusion. With output forced to CPU, low
repeating-layer offload is no longer obviously hopeless:

```text
output CPU -ngl 2: 11.9 tok/s
output CPU -ngl 3:  9.2 tok/s
output CPU -ngl 4:  7.5 tok/s
```

The decline across `-ngl 2`, `3`, and `4` is still regular and tracks one
additional CPU attention fallback per layer.

## Progression

The trace-guided implementation moved the fork through these checkpoints:

| Checkpoint | Rejected supports | Kernels | D2H transfers | Finishes | Main state |
| --- | ---: | ---: | ---: | ---: | --- |
| Q4_0 matmul only | `1955` | `403` | `403` | `1119` | every matmul was an isolated GPU island |
| Add F32 `ADD`/`MUL`/`RMS_NORM`/`SWIGLU` | `302` | `1067` | `274` | `723` | FFN/residual path mostly OpenCL-resident |
| Add F32 RoPE | `134` | `1199` | `274` | `756` | attention projections keep RoPE on OpenCL |
| Add `GET_ROWS` for F32 and Q4_0 | `2` | `1219` | `208` | `591` | all non-attention target ops accepted |
| Force output layer to CPU | `1` at `-ngl 2` | `539` | `109` | `449` | removes full-vocab GPU logits readback |

The `sync_other` counter is currently inflated as a diagnostic count: it is
incremented before the backend checks whether another OpenCL device exists.
It should not be read as 1990 real cross-device barriers.

The transfer-attribution patch changes this for newer runs. New trace output
reports `sync_other=[calls=<n>,waits=<n>,skipped=<n>]`, where `skipped` means no
other OpenCL device existed and no cross-device wait was queued. In the
attributed rerun above, all `sync_other` calls were skipped.

## Interpretation

The original bottleneck was broad CPU fallback around every OpenCL matmul. That
is now mostly addressed for this `-ngl 3`, `-nkvo`, short-context run. The
legacy NVIDIA path supports the repeated Qwen3 ops needed around offloaded
layers except attention itself.

The output-layer question is resolved for Fermi performance experiments: keep
`output.weight` on CPU. The full-vocabulary logits readback dominates the plain
low-`-ngl` runs, and removing it raises `-ngl 2` generation from `2.6 tok/s` to
`11.9 tok/s`.

The next useful work is attention fallback. With output on CPU, each additional
offloaded repeating layer still adds Q/K/V readbacks and a `FLASH_ATTN_EXT`
support rejection. That is why throughput falls from `11.9 tok/s` at `-ngl 2`
to `7.5 tok/s` at `-ngl 4`.

## Next Steps

1. Complete the stable benchmark sweep with the current fork:

   ```text
   -ngl 0, 1, 8, 16
   ```

   The `-ngl 2`, `3`, and `4` points are now recorded. Keep `-fit off`,
   `-c 128`, `-b 32`, `-ub 1`, `-nkvo`, and the same prompt for comparability.

2. Run an output-layer control experiment. The simplest existing control is
   `-ngl 1`, which offloads the output layer but no repeating layers. It should
   show the constant `result_output` readback without the per-layer Q/K/V
   attention fallback.

3. Use output-on-CPU as the default shape for Fermi performance experiments:

   ```bash
   LLAMA_FERMI_OPENCL_OUTPUT_CPU=1 \
   GGML_OPENCL_NVIDIA_LEGACY_TRACE=1 \
   ./build/llama.cpp-opencl-native/bin/llama-cli \
     -fit off \
     --device GPUOpenCL \
     -m models/Qwen3-0.6B-Q4_0.gguf \
     -p "Answer in one sentence: what is OpenCL?" \
     -c 128 \
     -n 8 \
     -b 32 \
     -ub 1 \
     -nkvo \
     --single-turn \
     --reasoning off \
     -ngl 3
   ```

   Repeat for `-ngl 2`, `3`, `4`, `8`, and `16` if the low points remain
   stable.

4. Investigate a narrow decode-oriented F32 attention path for Qwen3 shapes.
   The specific target is eliminating the per-layer D2H reads:

   ```text
   Qcur-* / Kcur-* / Vcur-*
   ```

5. Measure correctness before broadening support:

   - compare CPU-only and OpenCL logits for the same prompt
   - use short deterministic runs with fixed seed and greedy sampling
   - keep `-n` large enough to produce a complete sentence when checking output

6. Only after attention measurements, decide whether KV offload is worth
   re-enabling. Current runs intentionally use `-nkvo` to keep the experiment
   focused on model-weight and activation offload.
