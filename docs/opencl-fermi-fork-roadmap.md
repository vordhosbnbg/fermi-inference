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
- supported compute path: `Q4_0 x F32` `GGML_OP_MUL_MAT`
- supported graph plumbing: simple view/reshape/permutation no-op style nodes
- known working proof: `-fit off -ngl 100` puts about `319 MiB` of
  `Qwen3-0.6B-Q4_0.gguf` weights on `GPUOpenCL`
- known performance result: broad offload is much slower than CPU-only
  generation because unsupported ops and synchronization dominate

The current fork proves that the device can hold and use offloaded model
weights. It does not yet keep a full Qwen3 decoder layer on the GPU.

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

### 1. Add Placement and Transfer Instrumentation

Before adding more kernels, the fork needs direct evidence about where time and
data movement go.

Add a legacy NVIDIA debug mode that records, per graph node:

- op name and tensor name
- tensor shape, type, strides, and byte size
- selected backend
- whether the op was accepted by `supports_op`
- whether execution actually called an OpenCL kernel
- host-to-device and device-to-host transfer counts and byte totals
- `clFinish` and synchronization counts
- kernel launch count
- optional kernel timing when `GGML_OPENCL_PROFILING=ON`

This should replace the current narrow `output.weight` diagnostics with a
general trace switch, for example an environment variable such as:

```text
GGML_OPENCL_NVIDIA_LEGACY_TRACE=1
```

Use this instrumentation to generate an op inventory for:

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

The output should make clear whether the next bottleneck is unsupported ops,
unnecessary transfers, oversynchronization, or the Q4_0 kernel itself.

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

### 6. Add Minimal F32 Elementwise Kernels

These are the next best candidates because they can keep activations on OpenCL
between large matmuls.

Prioritize:

- `GGML_OP_ADD` for residual connections
- `GGML_OP_MUL` and `GGML_OP_SCALE`
- `GGML_OP_UNARY` with `GGML_UNARY_OP_SILU`
- `GGML_OP_GLU` or the exact SwiGLU pattern emitted by `build_ffn`
- `GGML_OP_CPY`, `GGML_OP_DUP`, and `GGML_OP_CONT` for device-side layout fixes
- no-op handling for `RESHAPE`, `VIEW`, `PERMUTE`, and `TRANSPOSE`

These kernels should be buffer-based, F32-first, and intentionally narrow. Avoid
adding broad type support until the Qwen3 graph requires it.

### 7. Add RMS Norm Support

Qwen3 uses RMS norm in several places:

- attention input norm
- Q norm
- K norm
- FFN input norm
- final output norm

Needed work:

- implement F32 `GGML_OP_RMS_NORM` for contiguous rows
- implement the following scale multiply as either a separate `MUL` or a fused
  `RMS_NORM + MUL` kernel
- support Q/K norm shapes as well as residual-stream shapes
- compare OpenCL output against CPU within a documented tolerance

RMS norm is a high-value target because it occurs repeatedly and sits directly
between matmuls. Leaving it on CPU forces frequent boundary crossings.

### 8. Add RoPE Support for Qwen3 Shapes

Qwen3 applies RoPE to Q and K after their per-head norms.

Needed work:

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

### 11. Add Final Logits Handling

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

### 12. Add Correctness Tests Before Performance Claims

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

1. Instrument op placement, transfers, sync points, and kernel launches.
2. Run the low `-ngl` sweep to establish the current performance curve.
3. Validate and tune the existing Q4_0 matmul kernel across all Qwen3 projection
   shapes.
4. Add F32 residual and elementwise kernels.
5. Add RMS norm, preferably with scale multiply fused where useful.
6. Add Qwen3 RoPE.
7. Move KV cache for offloaded layers to OpenCL.
8. Add a simple attention path for small context and single-token generation.
9. Expand prompt-eval coverage.
10. Reassess performance before broadening model or quantization support.

## Non-Goals

- Do not add CUDA support for this hardware.
- Do not switch to AWQ, GPTQ, or K-quants for this roadmap.
- Do not require CLBlast unless the user explicitly approves the dependency.
- Do not depend on OpenCL 2.x APIs, SVM, subgroups, or fp16 arithmetic.
- Do not run expensive llama.cpp builds from Codex sessions.
- Do not change bootloader, driver, package, or system configuration from repo
  scripts.

## Decision Point

After instrumentation plus F32 elementwise/RMS norm support, run the same fixed
prompt and compare:

```text
-ngl 0
-ngl 1
-ngl 2
-ngl 4
-ngl 8
-ngl 16
-ngl 100
```

If generation is still far below CPU-only and traces show large unavoidable
attention or cache costs, stop treating full OpenCL execution as a performance
goal. At that point the fork remains valuable as a compatibility proof and an
experiment log, but not as a practical runtime.
