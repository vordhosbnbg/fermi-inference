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
- Full offload is currently much slower than CPU-only inference. Prefer low
  `-ngl` sweeps for the next performance experiments.
