# Third-Party Code

## llama.cpp

`third_party/llama.cpp` is a git submodule pointing at upstream:

```text
https://github.com/ggml-org/llama.cpp.git
```

Initialize it with:

```bash
git submodule update --init --recursive third_party/llama.cpp
```

Build outputs should stay outside the submodule, for example:

```bash
cmake -S third_party/llama.cpp -B build/llama.cpp-opencl -DGGML_OPENCL=ON
cmake --build build/llama.cpp-opencl -j"$(nproc)"
```

If local patches become necessary, switch the submodule URL to a fork before
carrying long-lived changes. Short-lived compatibility experiments can be kept
as local submodule worktree changes while they are being evaluated.
