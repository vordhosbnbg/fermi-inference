# Model Selection

The model target should be chosen for the Fermi GPU experiment, not for a modern
CPU or modern GPU. The constraints are:

- about 2 GiB VRAM
- about 496 MiB maximum OpenCL allocation
- OpenCL C 1.1 device compiler
- current llama.cpp OpenCL compatibility is unproven and currently rejects the
  device before inference
- `Q4_0` is the safest first quantization family for llama.cpp OpenCL testing

## Primary Stretch Target

Use:

```text
ggml-org/Qwen3-0.6B-GGUF
Qwen3-0.6B-Q4_0.gguf
```

Download URL:

```text
https://huggingface.co/ggml-org/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q4_0.gguf
```

Observed remote metadata:

```text
Content-Length: 428970080
SHA256: da2572f16c06133561ce56accaa822216f2391ef4d37fba427801cd6736417d4
```

Why this model:

- Qwen3-0.6B is likely the most capable sub-1B model found with a simple public
  `Q4_0` GGUF.
- The file is about 409 MiB, below the observed 496 MiB OpenCL max-allocation
  limit.
- The model card is Apache-2.0 licensed.
- It is published by `ggml-org`, which reduces GGUF compatibility risk with
  current llama.cpp.

Risk:

- This is a stretch target, not a roomy target. It leaves limited headroom under
  the max-allocation limit if llama.cpp allocates one large OpenCL buffer for
  offloaded weights.
- Current upstream llama.cpp OpenCL rejects the GT 540M before inference due to
  device-family and OpenCL-version checks.

## Safer Fallback

Use:

```text
QuantFactory/SmolLM2-360M-Instruct-GGUF
Q4_0 variant
```

Why this fallback:

- about 229 MiB for `Q4_0`
- Apache-2.0
- instruction-tuned
- much more headroom under the max-allocation limit

Tradeoff:

- Likely less capable than Qwen3-0.6B overall.

## Rejected or Deferred

- Qwen3-0.6B `Q8_0`: too large for the Fermi comfort envelope.
- Qwen2.5-0.5B official `Q4_0`: public and capable, but about the same size as
  Qwen3 Q4_0 while being an older model family.
- NexaAI Qwen3 smaller 4-bit files: attractive sizes were listed on the model
  page, but direct resolver URLs returned `401`, so they were not selected.
