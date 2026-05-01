# Experiment Protocol

This protocol is intentionally manual. The repository should document and
compare runs without automating bootloader, package, or driver changes.

## 1. Record System State

Before inference runs, record:

```bash
date --iso-8601=seconds
uname -a
dkms status
nvidia-smi
clinfo | grep -Ei 'platform|device|opencl|compute capability|global memory|max allocation'
```

Also record:

- boot entry used
- NVIDIA driver package version
- OpenCL package versions
- llama.cpp submodule commit
- model filename and hash

The vendored runtime lives at `third_party/llama.cpp`. Build it out of tree so
the submodule remains clean unless an experiment intentionally patches it. See
`docs/build.md` for the optimized build configuration.

Do not run unconstrained builds on this laptop. Use the documented configure and
`-j2` build commands from `docs/build.md` in a normal local shell.

## 2. Select Model

Start with the model target documented in `docs/model-selection.md`.

Current primary stretch target:

```text
models/Qwen3-0.6B-Q4_0.gguf
```

Use a small context first, such as `-c 256` or `-c 512`.

## 3. CPU Baseline

Example shape:

```bash
./build/llama.cpp-opencl-native/bin/llama-cli \
  --device none \
  -ngl 0 \
  -m models/Qwen3-0.6B-Q4_0.gguf \
  -p "Explain OpenCL in one short paragraph." \
  -c 512 \
  -n 80 \
  --single-turn \
  --reasoning off
```

Record:

- whether the run completes
- prompt eval tokens/sec
- generation tokens/sec
- peak RAM if convenient
- output correctness at a basic smoke-test level

## 4. OpenCL Device Detection

Before offloading layers, verify that llama.cpp lists or initializes the OpenCL
backend. Record the exact command and output.

Current upstream llama.cpp is expected to reject the GT 540M. The active
bring-up plan is documented in `docs/opencl-fermi-plan.md`.

## 5. OpenCL Offload Matrix

Start small:

```bash
./build/llama.cpp-opencl-native/bin/llama-cli \
  -m models/Qwen3-0.6B-Q4_0.gguf \
  -p "Explain OpenCL in one short paragraph." \
  -c 512 \
  -n 80 \
  -ngl 1 \
  --single-turn \
  --reasoning off
```

Then try:

```text
-ngl 0
-ngl 1
-ngl 2
-ngl 4
-ngl 8
-ngl 16
-ngl 100
```

Use `-fit off` for controlled comparisons so llama.cpp does not adjust layer
offload or context choices between runs.

Stop increasing offload if any of these appear:

- OpenCL compiler failure
- out-of-memory error
- max-allocation error
- invalid output
- system instability

## 6. Result Classification

Minimum success:

- CPU inference completes.
- llama.cpp OpenCL build exists and starts.
- OpenCL backend detects the NVIDIA device.

Meaningful success:

- at least one transformer layer offloads without crashing
- GPU memory usage changes during inference
- output generation completes correctly

Current Fermi checkpoint:

- `-ngl 100` with `models/Qwen3-0.6B-Q4_0.gguf` offloads about 319 MiB of model
  weights to `GPUOpenCL` and completes generation.
- The same run is slow, around 0.8 generation tokens/sec.
- Treat broad offload as a correctness proof, not a performance setting.

Strong success:

- OpenCL offload improves generation tokens/sec over the CPU baseline

Expected result:

- CPU inference works.
- OpenCL offload may fail or may run slower than CPU.
