# Results

Raw run directories should be kept under:

```text
results/runs/YYYY-MM-DD-HHMM/
```

These directories are ignored by git because they may contain large logs. When a
run produces a reusable finding, summarize it in a tracked Markdown file under
`docs/experiments/`.

The current Fermi OpenCL sweep helper writes this shape automatically:

```bash
./scripts/run-opencl-fermi-sweep.py
```

By default it runs the no-`-nkvo`, output-on-CPU sweep for
`-ngl 2 3 4 8 16`, writes raw logs under `results/runs/.../logs/`, and writes
both `summary.md` and `summary.tsv`.

## Run Directory Shape

Suggested local files:

```text
metadata.env
system-state.txt
cpu-baseline.log
opencl-ngl-1.log
opencl-ngl-2.log
opencl-ngl-4.log
opencl-ngl-8.log
opencl-ngl-99.log
summary.md
```

## Metadata Template

```bash
RUN_DATE=
TIMEZONE=
HOST=
BOOT_ENTRY=
KERNEL=
NVIDIA_DRIVER=
OPENCL_PLATFORM=
OPENCL_DEVICE=
LLAMA_CPP_COMMIT=
MODEL_FILE=
MODEL_SHA256=
MODEL_QUANTIZATION=
PROMPT_FILE=
CONTEXT_SIZE=
GENERATE_TOKENS=
```

## Summary Template

```markdown
# Run Summary

Date:
Boot entry:
Kernel:
Driver:
llama.cpp commit:
Model:
Model SHA256:

## Commands

## CPU Baseline

- completed:
- prompt eval tokens/sec:
- generation tokens/sec:
- notes:

## OpenCL Offload

| `-ngl` | completed | generation tokens/sec | GPU memory changed | notes |
| --- | --- | --- | --- | --- |
| 1 |  |  |  |  |
| 2 |  |  |  |  |
| 4 |  |  |  |  |
| 8 |  |  |  |  |
| 99 |  |  |  |  |

## Decision
```
