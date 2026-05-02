# Fermi llama.cpp Fork Roadmap

This roadmap describes the remaining fork work needed to support the current
model target:

```text
models/Qwen3-0.6B-Q4_0.gguf
```

The goal is not to switch model families or quantization formats. The goal is
to make the existing GGUF `Q4_0` model run through the legacy NVIDIA OpenCL path
with fewer CPU fallbacks, then measure whether any of that work improves real
generation speed on the GT 540M.

## Baseline

Current fork state:

- `third_party/llama.cpp` branch: `fermi-opencl-legacy`
- OpenCL device family added: `NVIDIA_LEGACY`
- OpenCL C target: `OpenCL C 1.1`
- supported quantized weight path: raw GGUF `Q4_0` block storage
- supported matmul path: `Q4_0 x F32` `GGML_OP_MUL_MAT`
- supported non-attention Qwen3 helper ops: F32 `ADD`, `MUL`, `RMS_NORM`,
  normal/NeoX `ROPE`, `SWIGLU`, and F32/Q4_0 `GET_ROWS`
- supported graph plumbing: simple view/reshape/permutation no-op style nodes
- known working proof: `-fit off -ngl 100` puts about `319 MiB` of
  `Qwen3-0.6B-Q4_0.gguf` weights on `GPUOpenCL`
- known performance result: broad offload is much slower than CPU-only
  generation because unsupported ops and synchronization dominate

The current fork proves that the device can hold and use offloaded model
weights and can execute the repeated non-attention Qwen3 ops around the
offloaded layers. It does not yet keep attention on the GPU.

## Current Trace-Guided Checkpoint

The latest controlled run is documented in
`docs/experiments/2026-05-02-opencl-legacy-op-coverage.md`.

Run shape:

```text
build b9005-b57f9d327
model models/Qwen3-0.6B-Q4_0.gguf
-fit off --device GPUOpenCL -c 128 -n 8 -b 32 -ub 1 -nkvo -ngl 3
```

Result:

```text
[ Prompt: 9.2 t/s | Generation: 2.4 t/s ]
supports=[queries=4310,accepted=4308,rejected=2]
kernels=1219
transfers=[h2d=199/106043564B,d2h=208/7158784B]
finishes=591
```

Per-op outcome:

- `ADD`, `MUL`, `RMS_NORM`, `MUL_MAT`, `GET_ROWS`, `ROPE`, and `GLU/SWIGLU`
  are accepted and execute on OpenCL.
- The only remaining support rejection is `FLASH_ATTN_EXT`.
- Generation is still slower than CPU-only, so the project has moved from
  "add simple missing ops" to "measure attention/readback/synchronization
  costs."

First low-offload sweep points:

| `-ngl` | Prompt t/s | Generation t/s | GPU model MiB | Kernels | D2H transfers | Finishes | `FLASH_ATTN_EXT` rejections |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `2` | `14.7` | `2.6` | `91` | `559` | `109` | `426` | `1` |
| `3` | `10.4` | `2.4` | `100` | `1219` | `208` | `591` | `2` |
| `4` | `8.0` | `2.2` | `108` | `1879` | `307` | `756` | `3` |

The `-ngl 2` to `4` trend is negative. Each additional offloaded layer after
`-ngl 2` adds about `660` kernels, `99` D2H transfers, `165` finishes, and one
attention fallback. That makes transfer and attention attribution more valuable
than adding more simple F32 kernels.

The current fork patch adds that attribution. Legacy trace output now splits
`sync_other` into total calls, real waits, and skipped no-other-device checks,
and prints final summaries for:

- H2D/D2H transfer totals by tensor name, producing op, and tensor type
- H2D/D2H transfer totals by producing op
- `clFinish` counts by reason

Attributed reruns show:

- `sync_other` has `waits=0`; every call is skipped because only one OpenCL
  device is present.
- every `clFinish` in the final summary is `reason=synchronize`.
- the largest D2H byte source is the constant `result_output` readback from the
  GPU output projection: `10` transfers and `6077440` bytes.
- each offloaded repeating layer adds Q/K/V readbacks for CPU attention:
  `99` transfers and `540672` bytes.
- `output.weight` is the largest H2D upload at `87515136` bytes.

This separates the remaining problem into output-layer placement and
per-layer attention fallback.

The output-placement experiment resolves the first part. With
`LLAMA_FERMI_OPENCL_OUTPUT_CPU=1`, `output.weight` stays on CPU and low-offload
generation improves sharply:

| Run | Generation t/s | GPU model MiB | D2H bytes | Main state |
| --- | ---: | ---: | ---: | --- |
| plain `-ngl 1` | `2.9` | `83` | `6077440` | output-only GPU path reads back full logits |
| output CPU `-ngl 2` | `11.9` | `8` | `581632` | one repeating layer offloaded |
| output CPU `-ngl 3` | `9.2` | `16` | `1122304` | two repeating layers offloaded |
| output CPU `-ngl 4` | `7.5` | `25` | `1662976` | three repeating layers offloaded |

For Fermi performance experiments, keep the output layer on CPU and focus on
the per-layer attention fallback.

## Qwen3 Graph Surface

The relevant Qwen3 graph in llama.cpp is built in
`third_party/llama.cpp/src/models/qwen3.cpp`.

For this model, each layer logically includes:

- token embedding lookup from `tok_embd`
- attention input RMS norm
- Q, K, and V projections
- RMS norm on Q and K
- RoPE on Q and K
- KV cache write/update
- attention score computation
- attention mask and softmax
- attention value aggregation
- attention output projection
- residual add
- FFN input RMS norm
- FFN up and gate projections
- SiLU/SwiGLU-style activation and elementwise multiply
- FFN down projection
- residual add
- final output RMS norm
- output projection to logits

The current legacy path only accelerates some of the projection matmuls. The
rest either executes on CPU or forces transfers and synchronization between CPU
and GPU.

## Success Levels

Level 1, controlled partial offload:

- low `-ngl` runs complete with nonzero `GPUOpenCL` model memory
- generated text is qualitatively consistent with CPU output
- per-run logs identify which ops execute on OpenCL and which fall back

Level 2, layer-local OpenCL execution:

- one or more full transformer layers keep their major activations on OpenCL
- residual, norm, FFN activation, RoPE, and attention helper ops no longer force
  repeated host round trips inside the layer
- only expected boundary transfers remain

Level 3, practical full-model OpenCL path:

- all repeating-layer weights selected by `-ngl` execute through OpenCL-backed
  graph nodes
- KV cache for offloaded layers remains OpenCL-resident
- final logits may be copied back to CPU for sampling, but the decoder body does
  not depend on CPU fallbacks
- performance is compared against CPU-only using identical prompts and settings

Level 3 is a research target. It may still be slower than CPU on this hardware.

## Required Improvements

### 1. Maintain Placement and Transfer Instrumentation

The fork now has `GGML_OPENCL_NVIDIA_LEGACY_TRACE=1`, which gives direct
evidence about op support, kernel launches, transfers, and synchronization.
Keep this instrumentation active and extend it as the next bottlenecks become
more specific.

The legacy NVIDIA trace records, per graph node:

- op name and tensor name
- tensor shape, type, strides, and byte size
- selected backend
- whether the op was accepted by `supports_op`
- whether execution actually called an OpenCL kernel
- host-to-device and device-to-host transfer counts and byte totals
- `clFinish` and synchronization counts
- kernel launch count
- optional kernel timing when `GGML_OPENCL_PROFILING=ON`

The current trace also prints final aggregate lines:

```text
transfer-op-summary direction=d2h op=<op> count=<n> bytes=<n> avg=<n>
transfer-tensor-summary direction=d2h rank=<n> tensor=<name> op=<op> type=<type> count=<n> bytes=<n> min=<n> max=<n> avg=<n>
finish-summary reason=<reason> count=<n>
```

`sync_other` is reported as:

```text
sync_other=[calls=<n>,waits=<n>,skipped=<n>]
```

On the current single-OpenCL-device setup, `skipped` should account for the
old inflated `sync_other` count. Nonzero `waits` would mean the run actually
queued cross-device synchronization.

The trace is enabled with:

```text
GGML_OPENCL_NVIDIA_LEGACY_TRACE=1
```

Continue using this instrumentation to generate op inventories for:

```bash
./build/llama.cpp-opencl-native/bin/llama-cli \
  -fit off \
  --device GPUOpenCL \
  -m models/Qwen3-0.6B-Q4_0.gguf \
  -p "Answer in one sentence: what is OpenCL?" \
  -c 128 \
  -n 8 \
  -ngl 1 \
  -b 32 \
  -ub 1 \
  -nkvo \
  --single-turn \
  --reasoning off
```

At the current checkpoint, unsupported simple ops are no longer the main
bottleneck. The next trace run should use the new transfer summaries to
distinguish final logits reads from attention fallback reads.

### 2. Make Backend Claims Match Real Kernel Support

For `NVIDIA_LEGACY`, `ggml_opencl_supports_op` must stay conservative. It should
only return true for ops that can be executed correctly on OpenCL C 1.1 with the
actual tensor shapes produced by Qwen3.

For every newly advertised op:

- verify contiguous and strided layouts separately
- check input and output tensor types explicitly
- reject unsupported broadcast cases
- reject unsupported batch dimensions
- add trace output explaining rejections when legacy tracing is enabled

Do not reuse the current Intel/Adreno support matrix blindly. Some upstream
OpenCL kernels assume image objects, fp16 support, OpenCL 2.x behavior, or
subgroup-friendly layouts. The legacy path should prefer simple buffer kernels
that are obviously legal on OpenCL C 1.1.

### 3. Reduce Oversynchronization

The current legacy path uses `clFinish` for OpenCL 1.1 compatibility. That fixed
the `clEnqueueBarrierWithWaitList` failure, but broad use of `clFinish` can
destroy performance.

Needed work:

- separate correctness synchronization from routine sequencing
- use in-order command queue ordering where sufficient
- keep `clFinish` at backend boundaries and before required host reads
- count every `clFinish` in the trace
- avoid synchronizing with CPU backends before every node when no cross-backend
  dependency exists

OpenCL 1.1 events are available and can be used carefully, but the code should
not depend on OpenCL 2.x barrier APIs or SVM.

### 4. Harden Buffer and Tensor Residency

The full model cannot be practical if activations bounce through host memory.

Needed work:

- track whether each tensor is OpenCL-resident, host-resident, or copied
- make tensor allocation logs distinguish model, context, KV cache, and compute
  buffers
- keep F32 activations for offloaded layers in OpenCL buffers
- support views and offsets without materializing host copies
- keep the raw `Q4_0` block layout for model weights
- preserve alignment required by `CL_DEVICE_MEM_BASE_ADDR_ALIGN`
- enforce the `496 MiB` max allocation limit before calling `clCreateBuffer`
- avoid accidental flattening or struct-of-arrays conversion on the legacy path

The target is not merely nonzero model memory. The target is layer execution
where the intermediate tensors remain on the GPU until a real backend boundary.

### 5. Extend and Validate `Q4_0 x F32` Matmul

The existing Q4_0 kernel is the central useful kernel. It must be correct and
measured before adding more operations.

Required coverage:

- attention Q projection
- attention K projection
- attention V projection
- attention output projection
- FFN gate projection
- FFN up projection
- FFN down projection
- final output projection
- prompt-eval shape with `-b 32`
- single-token generation shape with `-ub 1`
- non-contiguous input/output strides where llama.cpp creates views

Potential improvements:

- tune workgroup size for the GT 540M instead of assuming the current `128`
- specialize matvec-style generation separately from prompt batch matmul
- reduce local-memory pressure; the device has only `48 KiB` local memory
- use vectorized reads only where alignment is proven
- add a small deterministic CPU-vs-OpenCL result comparison for representative
  Qwen3 projection shapes

Do this before implementing more quantization families. The current model is
already `Q4_0`; adding `Q4_K` or AWQ does not help this path.

### 6. Maintain Minimal F32 Elementwise Kernels

Status: implemented for the current Qwen3 run.

These kernels keep activations on OpenCL between large matmuls.

Currently covered:

- `GGML_OP_ADD` for residual connections
- `GGML_OP_MUL`
- `GGML_OP_GLU` with the exact SwiGLU pattern emitted by `build_ffn`
- no-op handling for `RESHAPE`, `VIEW`, `PERMUTE`, and `TRANSPOSE`

Still only add broad type/layout support when the Qwen3 graph requires it. The
legacy path should remain buffer-based, F32-first, and intentionally narrow.

### 7. Maintain RMS Norm Support

Status: implemented for F32 contiguous rows in the current Qwen3 run.

Qwen3 uses RMS norm in several places:

- attention input norm
- Q norm
- K norm
- FFN input norm
- final output norm

Completed work:

- implement F32 `GGML_OP_RMS_NORM` for contiguous rows
- support the following scale multiply as a separate `MUL`
- support Q/K norm shapes as well as residual-stream shapes in the traced run

Remaining work:

- compare OpenCL output against CPU within a documented tolerance
- consider fusing `RMS_NORM + MUL` only if trace timing proves launch overhead
  dominates

RMS norm is a high-value target because it occurs repeatedly and sits directly
between matmuls. Leaving it on CPU forces frequent boundary crossings.

### 8. Maintain RoPE Support for Qwen3 Shapes

Status: implemented for F32 normal/NeoX RoPE in the current Qwen3 run.

Qwen3 applies RoPE to Q and K after their per-head norms.

Completed work:

- implement F32 `GGML_OP_ROPE` for the non-vision, non-MRoPE path used here
- support the model's head dimension and position input layout
- handle prompt and single-token generation cases
- keep output in OpenCL buffers for attention

Do not attempt a generic RoPE implementation for every llama.cpp model variant
until the Qwen3 path is correct.

### 9. Move KV Cache for Offloaded Layers to OpenCL

Without GPU-resident KV cache, attention will continue to bounce across the host
boundary.

Needed work:

- allocate K and V cache buffers on OpenCL for offloaded layers
- implement the required cache write/update operations
- support `SET`, `SET_ROWS`, `CPY`, or the actual graph ops emitted for cache
  updates
- preserve llama.cpp cache layout and offsets
- verify that `-nkvo` and non-`-nkvo` modes are understood before changing the
  default experiment settings

Start with small contexts such as `-c 128` and `-c 256`. Larger context support
is secondary to correctness.

### 10. Implement a Simple OpenCL 1.1 Attention Path

The current upstream OpenCL attention support should not be assumed usable on
Fermi. The legacy path needs a simple buffer-based attention implementation.

Current first step: the fork is adding a legacy-only decode kernel for the
observed Qwen3 shape rather than enabling the generic upstream kernels. The
target shape is deliberately narrow:

- `FLASH_ATTN_EXT` with `n_q == 1`
- head dimension `128`
- Qwen3 `16` query heads and `8` KV heads
- F32 Q and output
- F16 K/V/mask as storage only, converted with `vload_half`
- no sinks, ALiBi, or logit softcap
- context ceiling `-c 128` for the first validation pass

Required pieces:

- F32 Q x K score computation
- mask application
- numerically stable row softmax
- softmax x V aggregation
- attention output projection through existing Q4_0 matmul

Implementation guidance:

- prefer simple, inspectable kernels over flash-attention-style fusion
- support small context and `-ub 1` generation first
- add prompt-eval support after generation correctness is proven
- keep all temporary score/probability buffers within the allocation limit
- document any context-size ceiling imposed by temporary buffers

This is likely the largest single chunk of work. It is also the point where the
project should reassess whether "more GPU" is still worth pursuing.

### 11. Reassess Output Layer Placement

Status: resolved for current Fermi performance experiments.

The attributed trace shows that low `-ngl` runs currently offload
`output.weight` before any repeating layer. That has two costs:

- `output.weight` uploads about `87.5 MiB` to the GT 540M.
- `result_output` reads back full F32 logits: `607744` bytes per sampled token
  batch in the measured run.

Before implementing a larger attention path, test whether keeping the output
layer on CPU improves low-offload generation. The fork now provides an explicit
environment switch for this experiment:

```text
LLAMA_FERMI_OPENCL_OUTPUT_CPU=1
```

This switch only applies when a `GPUOpenCL` device is active. It does not change
normal llama.cpp `-ngl` behavior unless set.

The first comparison should be:

```text
-ngl 1
-ngl 2
-ngl 3
-ngl 4
same values with LLAMA_FERMI_OPENCL_OUTPUT_CPU=1
```

With the switch enabled, the same low `-ngl` values keep the output layer on
CPU. For example, `-ngl 3` should offload the last two repeating layers but not
`output.weight`.

CPU output projection is faster overall than GPU output projection plus
full-logits readback in the measured low-`-ngl` runs. Keep the output layer on
CPU for Fermi performance experiments.

### 12. Add Final Logits Handling

The final output projection can remain OpenCL-backed, but sampling can stay on
CPU initially.

Needed work:

- keep final norm and output projection on OpenCL when their inputs are already
  OpenCL-resident
- read back only the logits needed by sampling
- avoid reading back full intermediate activations
- document the remaining CPU boundary as sampling/logits only

This is a practical compromise: the decoder body can be GPU-resident while the
small control-heavy sampling step remains CPU-side.

### 13. Add Correctness Tests Before Performance Claims

Each new operation should have a narrow correctness test before it becomes part
of the advertised support matrix.

Useful checks:

- CPU vs OpenCL output for one representative tensor shape
- prompt-eval smoke test
- single-token generation smoke test
- deterministic prompt with fixed seed where supported
- short run with `-n 8` before longer generations
- memory breakdown confirms expected `GPUOpenCL` residency
- generated text completes without teardown warnings or crashes

For risky changes, prefer adding a tracked experiment note under
`docs/experiments/` instead of only relying on raw logs.

## Recommended Implementation Order

Completed:

1. Instrument op placement, transfers, sync points, and kernel launches.
2. Add F32 residual and elementwise kernels.
3. Add RMS norm for the traced Qwen3 shapes.
4. Add Qwen3 F32 RoPE.
5. Add F32/Q4_0 `GET_ROWS` for token embedding gathers.
6. Keep `output.weight` on CPU for Fermi performance measurements.
7. Confirm output-on-CPU `-ngl 16` still scales with per-layer attention
   fallback: `15` `FLASH_ATTN_EXT` rejections, `1495` D2H transfers, and
   `2.3 tok/s` generation.

Next:

1. Complete the remaining low `-ngl` sweep with `-ngl 0`, `1`, and `8`.
2. Use `LLAMA_FERMI_OPENCL_OUTPUT_CPU=1` for Fermi performance measurements.
3. Test the narrow legacy decode attention kernel against the fixed `-c 128`,
   `-ub 1`, Qwen3 shape before broadening support.
4. Validate and tune the existing Q4_0 matmul kernel across all Qwen3 projection
   shapes.
5. Move KV cache for offloaded layers to OpenCL only after attention behavior
   is understood.
6. Expand prompt-eval coverage.
7. Reassess performance before broadening model or quantization support.

## Non-Goals

- Do not add CUDA support for this hardware.
- Do not switch to AWQ, GPTQ, or K-quants for this roadmap.
- Do not require CLBlast unless the user explicitly approves the dependency.
- Do not depend on OpenCL 2.x APIs, SVM, subgroups, or fp16 arithmetic.
- Do not run expensive llama.cpp builds from Codex sessions.
- Do not change bootloader, driver, package, or system configuration from repo
  scripts.

## Decision Point

After the current non-attention coverage checkpoint, run the same fixed prompt
and compare:

```text
-ngl 0
-ngl 1
-ngl 2
-ngl 3
-ngl 4
-ngl 8
-ngl 16
-ngl 100
```

If generation is still far below CPU-only and traces show large unavoidable
attention or cache costs, stop treating full OpenCL execution as a performance
goal unless a narrow attention kernel has a clear path to reducing readbacks.
At that point the fork remains valuable as a compatibility proof and an
experiment log, but not as a practical runtime.
