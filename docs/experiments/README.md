# Experiments

The first experiment is documented in the top-level
`FERMI_INFERENCE_EXPERIMENT.md` note. Keep follow-up experiment summaries here
when they become stable enough to track.

Use one file per experiment or decision point. Raw run output belongs under
`results/runs/` locally and is ignored by git.

Current checkpoint:

- The legacy NVIDIA OpenCL path detects the GT 540M through
  `opencl-nvidia-390xx`.
- Full model-weight offload with `-ngl 100` is confirmed for
  `Qwen3-0.6B-Q4_0`, with about 319 MiB resident on `GPUOpenCL`.
- Trace-guided `-ngl 3` work now covers the non-attention Qwen3 ops needed for
  the current legacy path: Q4_0 x F32 `MUL_MAT`, F32 `ADD`, `MUL`,
  `RMS_NORM`, `ROPE`, `SWIGLU`, and Q4_0/F32 `GET_ROWS`.
- The latest traced run rejects only `FLASH_ATTN_EXT`, but generation remains
  much slower than CPU-only inference. The next experiments should benchmark
  low offload counts and characterize remaining D2H transfers before attempting
  attention.
- The first low-offload sweep points show `-ngl 2`, `3`, and `4` at about
  `2.6`, `2.4`, and `2.2` generation tokens/sec respectively. Complete the
  remaining `-ngl 0`, `1`, `8`, and `16` points before making a final
  performance call.
- The current fork patch adds transfer and synchronization attribution to the
  legacy trace. After rebuilding, rerun `-ngl 2`, `3`, and `4` and inspect
  `transfer-tensor-summary`, `transfer-op-summary`, `finish-summary`, and the
  split `sync_other=[calls=...,waits=...,skipped=...]` field.
- The attributed rerun shows `sync_other` has no real waits. Remaining D2H is
  split between constant full-vocabulary `result_output` readback from the GPU
  output layer and per-layer Q/K/V readbacks for CPU attention fallback. The
  next control is `-ngl 1`, followed by the explicit
  `LLAMA_FERMI_OPENCL_OUTPUT_CPU=1` output-on-CPU experiment.
