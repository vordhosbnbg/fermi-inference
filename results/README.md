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

The helper captures `llama-cli` through a pseudo-terminal by default because
the human-facing prompt/generation tokens/sec line may not be printed when
stdout is a normal pipe.

To collect OpenCL command profiling in addition to the normal trace, rebuild
the llama.cpp fork after the profiling patch and run:

```bash
./scripts/run-opencl-fermi-sweep.py --profile
```

Profiling sets `GGML_OPENCL_NVIDIA_LEGACY_PROFILE=1`, enables OpenCL queue
profiling on the legacy NVIDIA path, and adds profile columns to `summary.tsv`
plus a `Profile Summary` section in `summary.md`. Start with a narrow sweep such
as `--ngl 2 4` because profiling records and reads one event per profiled
OpenCL command.

The legacy Q4_0 matmul local size can be tuned per run:

```bash
./scripts/run-opencl-fermi-sweep.py --profile --q4-lws 32 --ngl 2 4
```

Omit `--q4-lws` to use the automatic heuristic. The script records the selected
setting in metadata and parses the backend's local-size mode into `summary.tsv`.

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
