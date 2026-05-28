# Bonsai.c

English (this file). Japanese: [README.md](README.md).

This repository runs **[PrismML](https://prismml.com/)’s 1-bit Bonsai 8B** from a **GGUF file (`Bonsai-8B-Q1_0`)** using **a single C source**, **without relying on external libraries**.

### About 1-bit Bonsai 8B

Per [Announcing 1-bit Bonsai: The First Commercially Viable 1-bit LLMs](https://prismml.com/news/bonsai-8b) (PrismML, 2026), **1-bit Bonsai 8B** is a **true 1-bit model** end to end—embeddings, attention, MLP, and LM head are all 1-bit, with **no higher-precision “escape hatches”**—at roughly **8.2B parameters**. Weights are released under the **Apache License 2.0**. The post emphasizes **intelligence density**, deployability, and a compact footprint (about **1.15 GB**).

**This repo loads** the Hugging Face GGUF **`Bonsai-8B-Q1_0.gguf`** (Q1_0 quantization). The **three CPU variants** (`cpu`, `cpu-omp`, `cpu-blas`) are **for this file only**—linear weights must be **Q1_0**, norms **F32**; other GGUF quantizations exit with an error. Scope is **text prompt in, text out**; **image input is not supported**.

**This project does not link PyTorch, TensorFlow, JAX, ONNX Runtime, or other ML userland libraries.**

The reference implementation uses **standard C and `libm` only**, built from **`bonsai-8b/cpu/main.c`** into a **single-threaded CPU** binary (`bonsai-cpu`). For faster experimentation on multicore CPUs, **`bonsai-8b/cpu-omp/main.c`** builds **`bonsai-cpu-omp`**, parallelized with **OpenMP** (**standard C + `libm` + OpenMP runtime**).  
For practical throughput on **`Bonsai-8B-Q1_0`**, **`bonsai-8b/cpu-blas/`** builds **`bonsai-cpu-blas`** with **OpenMP + OpenBLAS** and **Q1_0×Q8_0 SIMD dot products** (llama.cpp **`ggml_vec_dot_q1_0_q8_0`** style) (**standard C + `libm` + OpenMP + OpenBLAS**).

### Why avoid ML libraries?

Typical LLM stacks hide **execution order, memory layout, alignment, and quantization packing** inside the framework.

This repo **makes the full pipeline visible in C**: GGUF read, weight restore, linear algebra, Transformer forward, sampling. The goal is to **inspect, validate, and change** inference—not to replace PyTorch.

- **Understandability**: follow the flow in the source and `doc/design.md`  
- **Minimal dependencies**: runs with a basic C toolchain  
- **Room to experiment**: try quantization and memory layouts individually  
- **Reference implementation**: a minimal example of **Bonsai 8B (GGUF)** decoder inference  

This is **not** aimed at peak performance or full feature parity.

## What you can run

You get a **single-thread CPU** reference build, an **OpenMP multicore** build, and an **OpenMP + OpenBLAS** optimized build—**three variants** in total.

| Mode | Source | Binary | Good for |
|---|---|---|---|
| CPU single-thread | `bonsai-8b/cpu/main.c` | `cpu/bonsai-cpu` | Learning the flow, minimal dependencies |
| CPU + OpenMP | `bonsai-8b/cpu-omp/main.c` | `cpu-omp/bonsai-cpu-omp` | Multicore smoke tests (reference) |
| CPU + OpenMP + OpenBLAS | `bonsai-8b/cpu-blas/main.c` | `cpu-blas/bonsai-cpu-blas` | Practical multicore throughput (**CPU recommended**) |

An 8B model on CPU stays **heavy**. Start with a small `-n` (e.g. `-n 1`) for smoke tests; for longer runs prefer **`cpu-blas`** (see [Reference benchmark](#reference-benchmark) below).

## Repository layout

```text
.
├── README.md
├── README.en.md
├── doc/
│   ├── ChangeLog
│   └── design.md
└── bonsai-8b/
    ├── Makefile              # make model only (GGUF fetch + checksum)
    ├── gguf.txt
    ├── Bonsai-8B-Q1_0.gguf.sha256sum
    ├── cpu/
    │   ├── Makefile
    │   └── main.c
    ├── cpu-omp/
    │   ├── Makefile
    │   └── main.c
    ├── cpu-blas/
    │   ├── Makefile
    │   └── main.c
    ├── gpu-cuda/          # appendix (end of this README, Q1_0×Q8_0)
    ├── gpu-cuda-nvfp4/    # appendix (end of this README, NVFP4 + CUTLASS)
    ├── gpu-rocm/          # appendix (end of this README)
    └── gpu-rocm-wmma/     # appendix (Prefill WMMA experiment)
```

The reference decoder lives under **`bonsai-8b/cpu/`**. The parallel variant is **`bonsai-8b/cpu-omp/`**; the CPU optimized build is **`bonsai-8b/cpu-blas/`**. GPU builds are optional appendices **`gpu-cuda`** (NVIDIA, Q1_0), **`gpu-cuda-nvfp4`** (NVIDIA, NVFP4 Tensor Core), **`gpu-rocm`** (AMD), and **`gpu-rocm-wmma`** (AMD, Prefill Attention WMMA experiment)—see the sections at the end of this file.

## Beginners: what happens during LLM inference?

1. **Read the GGUF**  
2. **Tokenize** the prompt  
3. **Run the Transformer** (one token at a time)  
4. **Sample** the next token (`-t`, `-k`, …)  
5. **Decode** tokens to text  

That pipeline is **not** inside PyTorch; you can follow it **in the C source**.

## Requirements

- Linux  
- `make`  
- A C compiler (`gcc`, `clang`, …)  
- `libm`  
- For **`cpu-omp`**, a compiler/toolchain with **OpenMP** (`libgomp` or `libomp`, commonly bundled with GCC/Clang)  
- For **`cpu-blas`**, the above plus **OpenBLAS** (e.g. `libopenblas-dev` on Debian/Ubuntu)  
- **`wget`** (for **`make model`** on **first** GGUF download)  
- **`Bonsai-8B-Q1_0.gguf`** from [prism-ml/Bonsai-8B-gguf](https://huggingface.co/prism-ml/Bonsai-8B-gguf) (run **`make model`** from **`bonsai-8b/`**)

```bash
sudo apt update
sudo apt install -y build-essential make
# For cpu-blas:
# sudo apt install -y libopenblas-dev
```

## Obtain the model file

**`bonsai-8b/Makefile`** exposes only the **`model`** target (fetch GGUF and verify SHA256; plain **`make`** does the same). Default **`MODEL=Bonsai-8B-Q1_0.gguf`**. For build/run, set **`MODEL`** / **`PROMPT`** in each subdirectory Makefile (e.g. `cd cpu && make run MODEL=/data/models/Bonsai-8B-Q1_0.gguf`).

The GGUF is **not** in the repo. Run **`make model`** to download from the URL in `bonsai-8b/gguf.txt` and place the file under `bonsai-8b/`. **If the file already exists, it is not re-downloaded; only SHA256 verification against `Bonsai-8B-Q1_0.gguf.sha256sum` runs** (removes the file on failure).

```bash
cd bonsai-8b
make model
```

Manual download:

```bash
cd bonsai-8b
url=$(sed 's|/blob/main/|/resolve/main/|' gguf.txt)
wget -O Bonsai-8B-Q1_0.gguf "$url"
sha256sum --check Bonsai-8B-Q1_0.gguf.sha256sum
```

Example layout:

```text
bonsai-8b/
├── Makefile
├── cpu/ … (`main.c` → `bonsai-cpu`)
├── cpu-omp/ … (`main.c` → `bonsai-cpu-omp`)
├── cpu-blas/ … (`main.c` → `bonsai-cpu-blas`)
└── Bonsai-8B-Q1_0.gguf
```

Integrity is checked by **`make model`** using **`Bonsai-8B-Q1_0.gguf.sha256sum`** (checksum runs even when the file already exists), which should match hashes on [Hugging Face](https://huggingface.co/prism-ml/Bonsai-8B-gguf/tree/main).

## Quick start

```bash
cd bonsai-8b
make model
cd cpu && make build
./bonsai-cpu ../Bonsai-8B-Q1_0.gguf -p "Hello" -n 1
```

(Optional) OpenMP build from `cpu-omp/`:

```bash
cd bonsai-8b/cpu-omp
make build
./bonsai-cpu-omp ../Bonsai-8B-Q1_0.gguf -p "Hello" -n 1
```

## Build & run (CPU)

**For `Bonsai-8B-Q1_0.gguf` only.** Linear weights use **Q1_0** fused row dots (**`dot_q1_0_row`**); embedding uses **`dequant_q1_0_blocks`**. Norm weights are **F32**.

### Build

```bash
cd bonsai-8b/cpu
make build
```

Produces **`bonsai-cpu`** in that directory.

### Run

```bash
cd bonsai-8b/cpu
./bonsai-cpu ../Bonsai-8B-Q1_0.gguf \
  -p "Give a one-sentence introduction of yourself." \
  -n 16
```

Using the local Makefile:

```bash
cd bonsai-8b/cpu
make run PROMPT="Give a one-sentence introduction of yourself."
```

Model elsewhere:

```bash
cd bonsai-8b/cpu
make run MODEL=/data/models/Bonsai-8B-Q1_0.gguf PROMPT="Hello"
```

## Build & run (CPU + OpenMP, `cpu-omp`)

Same model, CLI, and **Q1_0-only** scope as `cpu`; matmul, attention, SwiGLU, etc. are parallelized with **OpenMP**.

### Build

```bash
cd bonsai-8b/cpu-omp
make build
```

Produces **`bonsai-cpu-omp`** in that directory.

### Run

```bash
cd bonsai-8b/cpu-omp
./bonsai-cpu-omp ../Bonsai-8B-Q1_0.gguf -p "Hello" -n 16
```

Control thread count (default depends on the OpenMP runtime):

```bash
OMP_NUM_THREADS=8 ./bonsai-cpu-omp ../Bonsai-8B-Q1_0.gguf -p "Hello" -n 16
```

The local `Makefile` `run` target defaults to **`MODEL=../Bonsai-8B-Q1_0.gguf`**:

```bash
make run PROMPT="Give a one-sentence introduction of yourself."
```

## Build & run (CPU + OpenMP + OpenBLAS, `cpu-blas`, recommended)

Same model, CLI, and **Q1_0-only** scope as `cpu-omp`, with activations quantized to **Q8_0** and **`vec_dot_q1_0_q8_0`** (SIMD on AVX2, llama.cpp-style) plus **OpenBLAS** (batched `sgemv` for attention and F32 norm rows). OpenBLAS is pinned to **one thread**; parallelism comes from **OpenMP** (avoids nested threading).

### Build

```bash
sudo apt install -y libopenblas-dev   # if needed
cd bonsai-8b/cpu-blas
make build
```

Produces **`bonsai-cpu-blas`** (the build prints **`SIMD flags:`**—**`/proc/cpuinfo`** selects **`-mavx2`** and **`-mfma`** when available; override with `make build ARCH_FLAGS='-mavx2 -mfma'`).

### Run

```bash
cd bonsai-8b/cpu-blas
./bonsai-cpu-blas ../Bonsai-8B-Q1_0.gguf -p "Hello" -n 16
```

```bash
cd bonsai-8b/cpu-blas
make run PROMPT="Hello"
```

### Benchmark logging (`make log` / `make log.push`)

Run a long-prompt benchmark (~**130** tokens after ChatML + **128** decode tokens) and append results to **`cpu-blas/Makefile`** (same idea as **`gpu-rocm`** `log.push`). Column 2 is **`BENCH_SIMD`** (short label for **`ARCH_FLAGS`**, e.g. **`avx2+fma`**).

| Variable | Default | Meaning |
|---|---|---|
| `BENCH_PROMPT` | Long English text (in Makefile) | Benchmark prompt |
| `BENCH_N` | `128` | Max new tokens (`-n`) |
| `BENCH_SEED` | `42` | RNG seed (`-s`) |
| `BENCH_LOG_FILE` | `/tmp/benchmark.log` | key=value log path |

```bash
cd bonsai-8b/cpu-blas
make log.push          # build → benchmark → append one line to Makefile
make log               # print BENCH_LOG as a table
```

**Note:** **`make log.push` modifies `cpu-blas/Makefile`**. Check **`git diff`** before committing.

## Reference benchmark

**Reference numbers from development machines**—your results will vary with CPU, memory, compiler flags, and model placement. Re-run with the same GGUF and command to compare.

| Item | Value |
|---|---|
| CPU | AMD Ryzen 9 5950X (32 logical cores) |
| OS | Linux |
| Model | `Bonsai-8B-Q1_0.gguf` (pre-read, in RAM) |
| Command | `./<binary> Bonsai-8B-Q1_0.gguf -p "Hello" -n 16 -t 0` (`-n` = decode cap, `-t` = temperature) |
| Workload | prefill **18** tokens (`-p "Hello"` + chat template) + decode **16** tokens (`-n 16`) |
| Table metric | decode time and tok/s only (`Decode complete` on stderr; prefill excluded) |
| Environment | `cpu-omp` / `cpu-blas`: `OMP_NUM_THREADS=32`; `cpu-blas` also `OPENBLAS_NUM_THREADS=1` |
| Method | One warmup run per binary, then **best of 3** decode tok/s (GGUF pre-read in RAM) |

| Binary | Decode time | Decode throughput | Notes |
|---|---:|---:|---|
| `cpu/bonsai-cpu` | 66.8 s | **0.24 tok/s** | Single-threaded, `-O3` (`cpu/Makefile` defaults) |
| `cpu-omp/bonsai-cpu-omp` | 3.2 s | **4.94 tok/s** | `-O3 -fopenmp` (`cpu-omp/Makefile` defaults) |
| `cpu-blas/bonsai-cpu-blas` | 0.5 s | **30.79 tok/s** | `-O3 -fopenmp -ffast-math`, `ARCH_FLAGS` from cpuinfo (5950X: `-mavx2 -mfma`), OpenBLAS at 1 thread |

Under these conditions, **`cpu-blas` was about 6× faster than `cpu-omp`** and **about 128× faster than `cpu`** (`cpu-omp` was about 21× faster than `cpu`). Generated text (`Hello! I'm Bonsai, an AI assistant developed by PrismML.`) matched on all three CPU binaries.

### Long prompt (`cpu-blas` **`make log.push`**)

| Item | Value |
|---|---|
| Workload | **130** tokens after ChatML + **128** decode tokens (**`-n 128 -t 0 -s 42`**) |
| Metrics | **`prefill_tps` / `decode_tps` / `total_tps`** from **`/tmp/benchmark.log`** (or **`BENCH_LOG_FILE`**)—inference interval only |
| Reproduce | In **`bonsai-8b/cpu-blas/`**: **`make log.push`** → **`make log`** |
| SIMD column | Short label for **`ARCH_FLAGS`** on the benchmark host (**`BENCH_SIMD`**). **`avx`** vs **`avx2+fma`** are **different hosts**—compare like with like within the table |

| Timestamp | SIMD | Prefill tok/s | Decode tok/s | Total tok/s |
|---|---|---:|---:|---:|
| 2026-05-27 19:39 | **avx** | **1.96** | **1.99** | **1.98** |
| 2026-05-27 19:57 | **avx** | **1.97** | **2.00** | **1.99** |
| 2026-05-27 21:31 | **avx2+fma** | **26.34** | **25.85** | **26.09** |

Do **not** compare this table directly to the short-prompt table above (5950X, `-p "Hello" -n 16`).

## Common CLI options

| Option | Example | Meaning |
|---|---|---|
| `-p` | `-p "Hello"` | Prompt |
| `-n` | `-n 64` | Max new tokens |
| `-t` | `-t 0.7` | Temperature |
| `-k` | `-k 0.9` | Top-p |
| `-s` | `-s 1234` | RNG seed |
| `-l` | `-l 512` | Max sequence length |

```bash
./cpu/bonsai-cpu Bonsai-8B-Q1_0.gguf -p "Hello" -n 4
```

The OpenMP build accepts the same flags:

```bash
./cpu-omp/bonsai-cpu-omp Bonsai-8B-Q1_0.gguf -p "Hello" -n 4
```

The OpenBLAS build as well:

```bash
./cpu-blas/bonsai-cpu-blas Bonsai-8B-Q1_0.gguf -p "Hello" -n 4
```

## More deterministic output

```bash
./cpu/bonsai-cpu Bonsai-8B-Q1_0.gguf \
  -p "One sentence: what is GGUF?" \
  -n 32 \
  -t 0.2 \
  -s 42
```

## Clean

Run **`make clean`** in each subdirectory. The GGUF file is **not** deleted.

```bash
cd bonsai-8b/cpu && make clean
cd bonsai-8b/cpu-omp && make clean
cd bonsai-8b/cpu-blas && make clean
# Appendix GPU builds:
cd bonsai-8b/gpu-cuda && make clean
cd bonsai-8b/gpu-cuda-nvfp4 && make clean
cd bonsai-8b/gpu-rocm && make clean
cd bonsai-8b/gpu-rocm-wmma && make clean
```

## Troubleshooting

### `No such file or directory`

Wrong model path. If the GGUF is missing, run **`make model`** from `bonsai-8b/`.

```bash
ls -lh bonsai-8b/Bonsai-8B-Q1_0.gguf
cd bonsai-8b && make model
```

```bash
./cpu/bonsai-cpu /data/models/Bonsai-8B-Q1_0.gguf -p "Hello" -n 4
```

### CPU is slow

Expected for an 8B model on CPU. Try `-n 1` or `-n 4`, or use **`cpu-blas`** (requires `libopenblas-dev`). With **`cpu-omp`** only, tune **`OMP_NUM_THREADS`**:

```bash
./cpu/bonsai-cpu Bonsai-8B-Q1_0.gguf -p "Hello" -n 1
```

### OpenMP / `-fopenmp` failures

Install an OpenMP-capable toolchain and runtime (`libgomp` / `libomp` per distro), then rerun `make build` in **`cpu-omp`**.

### OpenBLAS not found (`cpu-blas`)

Install `libopenblas-dev` (or your distro’s equivalent) and rebuild in **`cpu-blas`**. If headers live off the default path, see comments in **`cpu-blas/Makefile`** and set `CPPFLAGS` with `-I`.

## Reading the codebase

1. `README.en.md` / `README.md`  
2. `doc/design.md`  
3. `bonsai-8b/cpu/main.c` — single-thread reference implementation  
4. `bonsai-8b/cpu-omp/main.c` — OpenMP parallel variant  
5. `bonsai-8b/cpu-blas/main.c` — OpenMP + OpenBLAS + Q1_0×Q8_0 SIMD dot products  

## Out of scope

- Training / fine-tuning  
- **AMD NPU (e.g. XDNA2)**, **Vulkan, Metal**, and similar (**`gpu-cuda`**, **`gpu-cuda-nvfp4`**, **`gpu-rocm`**, **`gpu-rocm-wmma`** are optional appendix builds; the main focus is the three CPU variants)  
- Batch inference tuning (GPU appendix prefill batching is for faster decode, not server batching)  
- Image input  
- Server or Web API packaging  
- **GGUF files other than `Bonsai-8B-Q1_0.gguf`** for the CPU variants (**Q1_0 linear weights + F32 norms** only; legacy Q4_K / IQ generic paths were removed)  
- Guaranteed numerical match with official implementations  

The goal is to **understand, experiment with, and adapt** **Bonsai-8B-Q1_0** (GGUF) text inference in **C**.

## More documentation

- `doc/design.md`  
- `doc/ChangeLog`  

If something fails, check **`bonsai-8b/Makefile`** (**`model` only**) and per-subdirectory Makefiles (`cpu/`, `cpu-blas/`, `gpu-cuda/`, `gpu-cuda-nvfp4/`, etc.), plus the model path you pass at runtime.

---

## NVIDIA CUDA implementation (`gpu-cuda`)

**`bonsai-8b/gpu-cuda/` is an appendix outside this repo’s goals** (single C source, minimal dependencies). It uses `main.c` + `kernels.cu` + `gpu.h` with **Q1_0×Q8_0 GEMV only** (no NVFP4 path). It requires **CUDA Toolkit, an NVIDIA driver, and a physical GPU**. Use **`make log` / `make log.push`** to record benchmark history in **`gpu-cuda/Makefile`**. After each run, **`BENCH_LOG_FILE`** (default **`/tmp/benchmark.log`**) also records **VRAM breakdown** (**`GpuVramProfile`** / **`gpu_model_vram_profile`**) as key=value fields in addition to tok/s. For Blackwell **NVFP4 Tensor Core**, see the separate appendix **`gpu-cuda-nvfp4/`**. The reference implementation to read first is **`cpu/main.c`**.

It is bundled only because the author **wanted to see how fast CUDA could go**. It does not complement the project’s purpose and is not an official feature for readers. It is easy to misread as part of the main project, so **`gpu-cuda/` is planned to move to a separate repository**. First-time readers can **ignore it**.

What follows is a technical note for anyone curious about GPU speed comparisons.

### Directory layout

```text
bonsai-8b/gpu-cuda/
├── Makefile
├── main.c
├── kernels.cu          # includes gpu_model_vram_profile
├── gpu.h               # GpuVramProfile / gpu_model_vram_profile
```

### Technical overview

Same GGUF and CLI as **`cpu-blas`**. The host side (`main.c`) handles GGUF mmap, the tokenizer, and sampling; the GPU side (`kernels.cu`) runs the forward pass.

#### VRAM layout (`gpu_model_create`)

Uploaded from host (mmap’d GGUF) via **H2D copy** at startup:

| Kind | Contents | Format |
|---|---|---|
| Weights | `token_embd`, per-layer `wq/wk/wv/wo/gate/up/down`, `output` | Q1_0 (g128) |
| Weights | `attn_norm`, `q_norm`, `k_norm`, `ffn_norm`, `output_norm` | F32 |
| KV cache | `kc`, `vc` | F32, `n_layers × max_seq × kv_dim` |
| Decode activations | `x`, `xb`, `xb2`, `q`, `k`, `v`, `hb`, `hb2`, `logits`, `q8` | F32 / Q8_0 |
| Prefill batch | `x_batch`, `xb_batch`, … `q8_batch`, etc. | pre-allocated for `-l` (`max_seq`) tokens |

Prefill buffer capacity is **`batch_cap = max_seq`** (CLI `-l`). If `n_tokens > max_seq`, the program exits with an error.

#### Generation loop (`generate` in `main.c`)

1. **Prefill** (`n_prompt > 1`): call **`gpu_forward_prefill` once** for all prompt tokens; copy only the **last-position logits** to CPU via `gpu_copy_logits`.
2. **Prefill** (`n_prompt == 1`): single call to **`gpu_forward`** as before.
3. **Decode**: sample a token, then call **`gpu_forward(token, pos)`** one token at a time; `pos` starts at `n_prompt` and increases.

Sampling (temperature, top-p, RNG) runs on the **CPU only**. The GPU returns a vocab-sized logits vector (D2H copy).

During prefill, the progress bar goes **0% → 100% when the batch finishes** (no per-token updates like the CPU builds, because `gpu_forward_prefill` is one batched kernel launch sequence).

#### Decode: `gpu_forward` (one token / one position)

For each decode step, layers `l = 0 … n_layers-1` run:

1. **Embedding** — `emb_q1_0_kernel`: dequant a Q1_0 row into F32 `x`.
2. **Pre-attention norm** — `rmsnorm_kernel` (F32 weights).
3. **Q/K/V projection** — `gpu_mm` (Q1_0 GEMV, below) → `q`, `k`, `v`.
4. **Q/K head norm** — `rmsnorm_head_kernel`.
5. **RoPE** — `rope_neox_kernel` (NeoX half-pair). cos/sin tables are built on the **CPU** (YaRN metadata) and copied H2D to `rope_cache`.
6. **KV write** — D2D copy of current `k`, `v` at `pos` into `kc`, `vc` (layout: `[layer][seq_pos][kv_dim]`).
7. **Attention** — `flash_attn_gqa_kernel` (below) → `xb`.
8. **Output projection + residual** — `gpu_mm` (`wo`) → `add_kernel`.
9. **Pre-FFN norm** → **gate/up projection** → **SwiGLU** → **down projection** → **residual**.
10. **Final norm + LM head** — `rmsnorm_kernel` → `gpu_mm` (`output`) → `logits`.

#### Prefill: `gpu_forward_prefill` (all prompt positions in parallel)

Prompt token IDs are copied H2D to `tokens_dev`. RoPE tables for positions `0 … n_tokens-1` are built on the **CPU in bulk** and uploaded to `rope_batch`.

| Step | Kernel | Parallelism |
|---|---|---|
| Embedding | `emb_q1_0_batch_kernel` | grid `(nb, n_tokens)` — token × Q1_0 block |
| RMSNorm | `rmsnorm_batch_kernel` | 1 block / token |
| Q/K/V/O, gate/up/down | `gpu_mm_batch` | parallel over `(token, output row)` |
| Q/K head norm | `rmsnorm_head_batch_kernel` | 1 block / `(token, head)` |
| RoPE | `rope_neox_batch_kernel` | 1 block / `(token, head)`, position-specific cos/sin |
| KV write | `kv_write_batch_kernel` | 1 block / token → fill `kc[0…n-1]`, `vc[0…n-1]` |
| Attention | `flash_attn_prefill_gqa_kernel` | 1 block / `(token, head)`, causal mask `npos = t + 1` |
| Residual, SwiGLU | `add_batch_kernel`, `swiglu_batch_kernel` | element-parallel |

FFN after attention also uses batch kernels; hidden state lives in `x_batch` as `[n_tokens, dim]`.

**LM head for the last token only:** logits are not computed for every position—only `x_batch[(n_tokens-1) * dim]` goes through `rmsnorm_kernel` → `gpu_mm` to produce `logits` (for the first decode token).

#### Q1_0 GEMV (`gpu_mm` / `gpu_mm_batch`)

Same approach as **`cpu-blas`**. Weights stay **Q1_0 in VRAM** (no upfront dequant). **`make run`** (Blackwell **`sm_120a`**) and **`make build`** (**`nvidia-smi` GPU auto-detect**) use this path. **RTX 5090-class GPUs need native `sm_120` / `sm_120a`** (PTX **`compute_86` JIT breaks inference**). For **NVFP4**, see the **`gpu-cuda-nvfp4`** section below.

1. Quantize the input vector (or each batch row) to Q8_0 (group size 32) with **`quantize_q8_0_kernel`**.
2. Run **`mm_q1_0_kernel`** / **`mm_q1_0_batch_kernel`**: `vec_dot_q1_0_q8_0` — dot product of a Q1_0 weight row and Q8_0 activations; 1-bit sign bits combined with Q8_0 int8 products, restored with FP16 scales.
3. Each Q1_0 block (128 elements) pairs with 4 Q8_0 blocks.

Decode: 256 threads/block parallel over output dimension `d`. Prefill: parallel over `n_tokens × d` output elements.

#### Flash Attention (GQA, online softmax)

The attention matrix `[n_heads, seq, seq]` is **never materialized**. K/V cache is scanned in **64-token (`FA_BR`) tiles** along the sequence; **online softmax** (running max `m` and sum `l`) updates the output.

- **Decode** — `flash_attn_gqa_kernel`: grid = `n_heads` blocks × `FA_HD` (128) threads. One query position (current token) × all heads. GQA: head `h` reads KV head `h / kv_mul`.
- **Prefill** — `flash_attn_prefill_gqa_kernel`: grid = `n_tokens × n_heads` blocks. Query at position `t` sees K/V at `0 … t` only (**causal mask**, `npos = t + 1`).

Per tile:

1. Cooperatively load K/V into **shared memory** (`k_tile[64][128]`, `v_tile[64][128]`).
2. `Q · K^T` → `scores[]` (shared).
3. Update online softmax `m`, `l` with tile max.
4. Accumulate softmax weights × V into `o_sh[]`.

Total shared memory is ~65 KB; **`cudaFuncAttributePreferredSharedMemoryCarveout = 100`** is set at launch (exceeds the default 48 KB limit).

#### RoPE

On the CPU (`build_rope_cache_host`), llama.cpp-style **NeoX half-pair** + **YaRN** metadata (`rope.scaling.*`, `context_length`, etc.) produce cos/sin tables. Prefill uploads `n_tokens` tables to `rope_batch`; decode uploads one position to `rope_cache`.

#### Constraints and known behavior

- Target GGUF: **`Bonsai-8B-Q1_0`** (Q1_0 g128 + F32 norms). Other quant formats are unsupported.
- If `head_dim > 128` (`FA_HD`), Flash Attention kernels become no-ops (Bonsai-8B uses `head_dim = 128`).
- KV cache is shared between prefill and decode—after prefill, decode continues at `pos = n_prompt` using the `kc`/`vc` filled during prefill.

### Requirements

- **CUDA Toolkit** (`nvcc`) and an **NVIDIA driver** (`libcudart` only; cuBLAS not required)

```bash
sudo apt install -y nvidia-cuda-toolkit   # or NVIDIA’s official CUDA Toolkit
```

### Build

| Goal | Command (`bonsai-8b/gpu-cuda/`) | Notes |
|---|---|---|
| Build with GPU auto-detect | `make build` | **`nvidia-smi`** picks **`CUDA_GENCODE`** (RTX 50 → **`sm_120`**, etc.) |
| Blackwell + Q1_0 (default) | `make` / `make run` | CUDA 13 + native **`sm_120a`** (first run may need **sudo**; later runs **skip apt**) |
| Show benchmark history | `make log` | |
| Run benchmark and append history | `make log.push` | e.g. `make log.push BENCH_N=64` |

```bash
cd bonsai-8b/gpu-cuda
make run              # default: blackwell (sm_120a, Q1_0) then inference (skips apt if CUDA 13 present)
# Long-prompt benchmark: make log.push → make log
```

Produces **`bonsai-gpu-cuda`**.

After changing **`CUDA_GENCODE`** or **`FA_BR`**, the Makefile uses **`.build_config.stamp`** to force **`kernels.o`** to rebuild. **`make clean`** removes the stamp as well.

**`make build`** auto-selects **`CUDA_GENCODE`** / **`FA_BR`** from **`nvidia-smi`** (build log shows `GPU_CCAP` / `CUDA_GENCODE`). Override example:

```bash
make build CUDA_GENCODE=arch=compute_90,code=sm_90
```

### Run

```bash
cd bonsai-8b/gpu-cuda
make run PROMPT="Hello"          # Blackwell sm_120a + Q1_0 (default)
```

CLI options (`-p`, `-n`, `-t`, `-k`, `-s`, `-l`) match the CPU builds.

```bash
./gpu-cuda/bonsai-gpu-cuda Bonsai-8B-Q1_0.gguf -p "Hello" -n 4
```

### Benchmark logging (`make log` / `make log.push`)

Same idea as **`gpu-rocm`** / **`gpu-cuda-nvfp4`**: run a **long prompt** (~**130** tokens after ChatML) + **128** decode tokens and append results to **`gpu-cuda/Makefile`**. Column 2 is **`GPU_LABEL`** (compute capability from **`nvidia-smi`**, e.g. **`sm_120`**).

| Variable | Default | Meaning |
|---|---|---|
| `BENCH_PROMPT` | Long English text (in Makefile) | Benchmark prompt |
| `BENCH_N` | `128` | Max generated tokens (`-n`) |
| `BENCH_SEED` | `42` | RNG seed (`-s`) |
| `BENCH_LOG_FILE` | `/tmp/benchmark.log` | key=value log path |

```bash
cd bonsai-8b/gpu-cuda
make log.push          # build → benchmark → append one line to Makefile
make log               # print BENCH_LOG as a table
```

**Note:** **`make log.push` modifies `gpu-cuda/Makefile`**. Check **`git diff`** before committing. Table **`total_tps`** is **inference only** (VRAM weight upload excluded).

**VRAM fields in `BENCH_LOG_FILE`** (written after each run): **`vram_total`**, **`vram_device_used`** / **`vram_device_total`** (**`cudaMemGetInfo`**, each in **bytes** and **`_mib`**), and a **`[vram_breakdown]`** section (Q1_0 embedding / F32 norm / Q1_0 linear weights / KV / decode activations / prefill batch). **`BENCH_LOG`** in the Makefile stores tok/s only; see **`BENCH_LOG_FILE`** for VRAM breakdown.

### Reference benchmark (GPU)

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 5090 (31 GiB VRAM) |
| OS | Linux |
| Model | `Bonsai-8B-Q1_0.gguf` (uploaded to VRAM at startup) |
| Command | **`make run`** (**`sm_120a`**, Q1_0) |
| Workload | Same as CPU table above (prefill 18 + decode 16 tokens) |
| Repro | One warmup per configuration, then representative of three runs (prefill / decode from stderr `Prefill complete` / `Decode complete` lines) |

#### Q1_0 GPU short prompt (Blackwell native, `make run`)

**CUDA 13**, **`sm_120a`**, **`FA_BR=32`**. `-p "Hello, how are you?"` (measured 2026-05-28).

| Binary | Prefill tok/s | Decode tok/s | Notes |
|---|---:|---:|---|
| `gpu-cuda/bonsai-gpu-cuda` | **~312** | **~47** | **`make run`** (**`sm_120a`**) |

#### Q1_0 GPU long prompt (`make log.push`)

| Item | Value |
|---|---|
| GPU | **sm_120** (RTX 5090; **`make log` GPU column**) |
| Workload | Long prompt (**130** tokens after ChatML) + **128** decode tokens (**`make log.push`** defaults: **`-n 128 -t 0 -s 42`**) |
| Metrics | **`prefill_tps` / `decode_tps` / `total_tps`** from **`/tmp/benchmark.log`** (or **`BENCH_LOG_FILE`**) — inference interval only |
| Reproduce | In **`bonsai-8b/gpu-cuda/`**: **`make log.push`** → **`make log`** |

| Timestamp | GPU | Prefill tok/s | Decode tok/s | Total tok/s | Notes |
|---|---|---:|---:|---:|---|
| 2026-05-28 17:35 | **sm_120** | **412.61** | **43.12** | **78.58** | 130+128 tokens (**`make log.push`**) |
| 2026-05-28 17:43 | **sm_120** | **411.64** | **43.16** | **78.62** | same (2nd run) |
| 2026-05-28 20:51 | **sm_120** | **413.24** | **43.17** | **78.66** | same (3rd run) |

**VRAM breakdown** (above **2026-05-28 20:51** run, 3rd measurement; **`BENCH_LOG_FILE`** **`[vram_breakdown]`**; default **`-l` / `max_seq`**):

| Item | bytes | MiB | Notes |
|---|---:|---:|---|
| **`vram_total`** (theoretical sum) | 1,408,171,860 | **1342.94** | Sum of categories below |
| **`vram_device_used`** | 2,226,388,992 | **2123.25** | **`cudaMemGetInfo`** (may include CUDA runtime overhead) |
| **`vram_device_total`** | 33,669,513,216 | **32109.75** | Total GPU VRAM |
| `vram_weights_q1_embd` | 87,361,344 | **83.31** | Q1_0 **`token_embd`** |
| `vram_weights_f32_norm` | 1,232,896 | **1.18** | F32 norm weights |
| `vram_weights_q1_linear` | 1,064,109,888 | **1014.81** | Q1_0 linear weights (**`wq`–`down`** + LM head) |
| `vram_kv_cache` | 150,994,944 | **144.00** | **`kc` / `vc`** (depends on **`-l`**) |
| `vram_decode_activations` | 792,788 | **0.76** | Single-token decode buffers |
| `vram_prefill_batch` | 103,680,000 | **98.88** | Prefill batch (**`batch_cap = max_seq`**) |

Linear weights (**~1015 MiB**) dominate theoretical VRAM. KV (**~144 MiB**) scales with **`-l`**. **`vram_device_used`** can exceed **`vram_total`** (driver / CUDA allocation).

**PTX `compute_86` JIT** is **not supported on RTX 5090** (garbled output).

Do **not** compare the tables above directly to the **short-prompt** or **NVFP4 long-prompt** tables (different prompt length, token counts, or code path).

With the same prompt and **`-t 0`**, **Q1_0 configurations** produce the same text as **`cpu-blas`** (`Hello! I'm Bonsai, an AI assistant developed by PrismML.`). **NVFP4** numbers are in the **`gpu-cuda-nvfp4`** section below.

### Troubleshooting (CUDA build)

**CUDA / `nvcc` not found:** Install CUDA Toolkit and an NVIDIA driver, then run **`make build`** or **`make run`** (Blackwell) in **`gpu-cuda`**. **RTX 50 series** need **CUDA 13** and native **`sm_120` / `sm_120a`** (do **not** use PTX **`compute_86`**). **`make build`** auto-selects **`CUDA_GENCODE`** via **`nvidia-smi`**.

**Garbled output / `flash_attn … unsupported toolchain`:** On Blackwell, avoid PTX JIT (`CUDA_GENCODE=arch=compute_86,code=compute_86`). Rebuild with **`make clean && make run`**.

**`make run` runs apt every time:** If CUDA 13 is already installed, later runs skip apt. Force reinstall: **`make blackwell FORCE_CUDA_APT=1`**.

If you see `[prompt length … exceeds max_seq …]`, increase **`-l`**.

**Clean:** run **`make clean`** in **`bonsai-8b/gpu-cuda`**, **`bonsai-8b/gpu-cuda-nvfp4`**, **`bonsai-8b/gpu-rocm`**, or **`bonsai-8b/gpu-rocm-wmma`**.

### Source files to read

6. `bonsai-8b/gpu-cuda/main.c` — host side (GGUF, tokenizer, `generate`, sampling, **VRAM benchmark log**)  
7. `bonsai-8b/gpu-cuda/kernels.cu` — forward pass, Q1_0 GEMV, Flash Attention, **`gpu_model_vram_profile`**  
8. `bonsai-8b/gpu-cuda/gpu.h` — C API (`gpu_forward_prefill`, etc., **`GpuVramProfile`**)  

---

## NVIDIA CUDA NVFP4 implementation (`gpu-cuda-nvfp4`)

**`bonsai-8b/gpu-cuda-nvfp4/`** is a separate appendix with the same GGUF, CLI, and prefill/decode split as **`gpu-cuda`**, but **linear layers** use **NVFP4 Tensor Core + CUTLASS**. Binary: **`bonsai-gpu-cuda-nvfp4`**. Requires **CUTLASS v4.5.1** (older v3.9.0 fails GEMM `initialize` on RTX 5090). Use **`make log` / `make log.push`** to record benchmark history in **`gpu-cuda-nvfp4/Makefile`**. After each run, **`BENCH_LOG_FILE`** (default **`/tmp/benchmark.log`**) also records **VRAM breakdown** (**`GpuVramProfile`** / **`gpu_model_vram_profile`**) as key=value fields in addition to tok/s. See **`doc/design.md`** (“Runtime behavior (`gpu-cuda-nvfp4`)”) for the full spec.

### Directory layout

```text
bonsai-8b/gpu-cuda-nvfp4/
├── Makefile
├── main.c
├── kernels.cu          # linear layers always gpu_mm_fp4 → fp4_bonsai_mm
├── gpu.h               # GpuVramProfile (NVFP4-specific fields)
├── fp4_bonsai.cu / fp4_bonsai.h
├── fp4_gemm.cu / fp4_gemm.h
└── third_party/cutlass/   # make cutlass (CUTLASS_TAG=v4.5.1)
```

### NVFP4 + CUTLASS (summary)

**NVFP4:** 4-bit floating format for Blackwell Tensor Cores. Values are **E2M1** (4 bit); every **16 elements** share one **UE4M3** scale (8 bit). GGUF stays **Q1_0**; at startup weights are **restored to BF16 → NVFP4 cache** for GEMM.

| File | Role |
|---|---|
| `fp4_gemm.cu` | CUTLASS **SM120 block-scaled NVFP4 GEMM**. Startup **`fp4_gemm_prealloc`**. **`fp4_gemm_init(M,N,K)`** before each GEMM. **`fp4_weight_cache_vram_bytes`** / **`fp4_gemm_vram_bytes`** |
| `fp4_bonsai.cu` | Q1_0 → NVFP4 weight cache, F32 ↔ BF16 ↔ GEMM. Activation padding uses separate **`M_act` / `M_pad`**. **`fp4_bonsai_vram_bytes`** |
| `kernels.cu` | Always **`fp4_bonsai_mm`** (no `#ifdef`). **`gpu_model_vram_profile`** |

**Layers:** `wq` through `down` and `output` (LM head). **`M`/`N`/`K` must be multiples of 128**. stderr should show `GPU: FP4 Tensor Core path enabled` at startup.

Full E2M1 tables and bit diagrams match the former **`gpu-cuda`** appendix (also in **`doc/design.md`**).

### Build and run

| Goal | Command (`bonsai-8b/gpu-cuda-nvfp4/`) |
|---|---|
| Fetch CUTLASS | `make cutlass` |
| Blackwell + NVFP4 (default) | `make` / `make run` |
| Show benchmark history | `make log` |
| Run benchmark and append history | `make log.push` (e.g. `make log.push BENCH_N=64`) |

```bash
cd bonsai-8b/gpu-cuda-nvfp4
make cutlass            # first time or after CUTLASS update
make run                # blackwell → sm_120a + NVFP4
./bonsai-gpu-cuda-nvfp4 ../Bonsai-8B-Q1_0.gguf -p "Hello"
```

After a CUTLASS upgrade: `rm -rf third_party/cutlass && make cutlass`.

### Benchmark logging (`make log` / `make log.push`)

Same idea as **`gpu-rocm`**: run a **long prompt** (~**130** tokens after ChatML) + **128** decode tokens and append results to **`gpu-cuda-nvfp4/Makefile`**. Column 2 is **`GPU_LABEL`** (compute capability from **`nvidia-smi`**, e.g. **`sm_120`**).

| Variable | Default | Meaning |
|---|---|---|
| `BENCH_PROMPT` | Long English text (in Makefile) | Benchmark prompt |
| `BENCH_N` | `128` | Max generated tokens (`-n`) |
| `BENCH_SEED` | `42` | RNG seed (`-s`) |
| `BENCH_LOG_FILE` | `/tmp/benchmark.log` | key=value log path |

```bash
cd bonsai-8b/gpu-cuda-nvfp4
make log.push          # build → benchmark → append one line to Makefile
make log               # print BENCH_LOG as a table
```

**Note:** **`make log.push` modifies `gpu-cuda-nvfp4/Makefile`**. Check **`git diff`** before committing. Table **`total_tps`** is **inference only** (VRAM weight upload excluded).

**VRAM fields in `BENCH_LOG_FILE`** (written after each run): **`vram_total`**, **`vram_device_used`** / **`vram_device_total`** (**`cudaMemGetInfo`**, each in **bytes** and **`_mib`**), and a **`[vram_breakdown]`** section (Q1_0 embedding / F32 norm / NVFP4 linear weights / KV / decode activations / prefill batch / FP4 GEMM scratch). **`BENCH_LOG`** in the Makefile stores tok/s only; see **`BENCH_LOG_FILE`** for VRAM breakdown.

### Reference benchmark (NVFP4, RTX 5090)

| Item | Value |
|---|---|
| GPU | **sm_120** (RTX 5090; **`make log` GPU column**) |
| OS | Linux |
| Model | `Bonsai-8B-Q1_0.gguf` (uploaded to VRAM at startup) |
| Workload | Long prompt (**130** tokens after ChatML) + **128** decode tokens (**`make log.push`** defaults: **`-n 128 -t 0 -s 42`**) |
| Metrics | **`prefill_tps` / `decode_tps` / `total_tps`** from **`/tmp/benchmark.log`** (or **`BENCH_LOG_FILE`**) — inference interval only |
| Reproduce | In **`bonsai-8b/gpu-cuda-nvfp4/`**: **`make log.push`** → **`make log`** |

| Timestamp | GPU | Prefill tok/s | Decode tok/s | Total tok/s | Notes |
|---|---|---:|---:|---:|---|
| 2026-05-28 18:48 | **sm_120** | **5751.92** | **64.12** | **127.80** | 130+128 tokens (**`make log.push`**) |
| 2026-05-28 20:30 | **sm_120** | **5762.22** | **64.14** | **127.84** | same (2nd run) |
| 2026-05-28 20:30 | **sm_120** | **5757.93** | **64.20** | **127.96** | same (3rd run) |

**VRAM breakdown** (above **2026-05-28 20:30** run, 3rd measurement; **`BENCH_LOG_FILE`** **`[vram_breakdown]`**; default **`-l` / `max_seq`**):

| Item | bytes | MiB | Notes |
|---|---:|---:|---|
| **`vram_total`** (theoretical sum) | 5,975,701,524 | **5698.87** | Sum of categories below |
| **`vram_device_used`** | 6,752,043,008 | **6439.25** | **`cudaMemGetInfo`** (may include CUDA runtime overhead) |
| **`vram_device_total`** | 33,669,513,216 | **32109.75** | Total GPU VRAM |
| `vram_weights_q1_embd` | 87,361,344 | **83.31** | Q1_0 **`token_embd`** |
| `vram_weights_f32_norm` | 1,232,896 | **1.18** | F32 norm weights |
| `vram_weights_fp4` | 4,256,464,896 | **4059.28** | NVFP4 linear weight cache (**`wq`–`down`** + LM head) |
| `vram_kv_cache` | 150,994,944 | **144.00** | **`kc` / `vc`** (depends on **`-l`**) |
| `vram_decode_activations` | 792,788 | **0.76** | Single-token decode buffers |
| `vram_prefill_batch` | 103,680,000 | **98.88** | Prefill batch (**`batch_cap = max_seq`**) |
| `vram_fp4_gemm_scratch` | 1,375,174,656 | **1311.47** | BF16 activations/output + CUTLASS workspace, etc. |

NVFP4 linear cache (**~4059 MiB**) and GEMM scratch (**~1311 MiB**) dominate theoretical VRAM. This is larger than the **Q1_0 path** ( **`gpu-cuda`** long-prompt table, theoretical **~1343 MiB**), but includes Tensor Core GEMM working memory. KV, embedding, and prefill batch are similar to the Q1_0 build.

Do **not** compare this table directly to the **Q1_0 GPU (`make run`, short prompt)** table (different prompt length and token counts).

### Troubleshooting (NVFP4)

**`fp4_gemm_run_cached: initialize: Error Internal`:** Re-fetch **CUTLASS v4.5.1** (`rm -rf third_party/cutlass && make cutlass`). Confirm **CUDA 13** and native **`sm_120a`**.

---

## AMD ROCm / HIP implementation (`gpu-rocm`)

**`bonsai-8b/gpu-rocm/` is also an appendix outside this repo’s main goals** (single C source, minimal dependencies). It uses `main.c` + `kernels.hip` + `gpu.h` (**C API based on `gpu-cuda/gpu.h` with `gpu_get_device_desc` added**) and requires **ROCm (`hipcc`)**, an **AMD GPU driver**, and a **physical GPU**. **hipBLAS is not required** (linear layers use custom Q1_0×Q8_0 HIP kernels). **There is no NVFP4 path** (same algorithm family as **`gpu-cuda` Q1_0**). Use **`make log` / `make log.push`** to record benchmark history in **`gpu-rocm/Makefile`**.

It exists so the author could **try the same forward on AMD GPUs**. First-time readers can **ignore it**. See **`doc/design.md`** (“Build and run (GPU ROCm)”, “Runtime behavior (`gpu-rocm`)”) for the full spec.

### Directory layout

```text
bonsai-8b/gpu-rocm/
├── Makefile
├── main.c
├── kernels.hip
└── gpu.h          # based on gpu-cuda (adds gpu_get_device_desc)
```

### Technical overview

Same GGUF and CLI as **`cpu-blas`** / **`gpu-cuda` (Q1_0 path)**: **batched prefill** + **sequential decode** via **`gpu_forward_prefill`** / **`gpu_forward`**. The host handles GGUF mmap, the tokenizer, and sampling; the device runs **Flash Attention** (K/V shared staging, **`FA_BR`** tiles) and **Q8_0 activations + Q1_0 GEMV**. VRAM layout matches the CUDA appendix in spirit (HIP / **`hipMalloc`** instead of CUDA).

On exit, a key=value benchmark log is written to **`BENCH_LOG_FILE`** (default **`/tmp/benchmark.log`**). stderr prefill/decode/total tok/s cover **`generate()` only**—**after** weight H2D in **`gpu_model_create`**.

### Requirements

- **ROCm** (default **`/opt/rocm`**, **`hipcc`** and **`rocminfo`**)
- **AMD GPU driver**
- Host: **`g++`** and **`libstdc++-dev`** (`hipcc` needs GCC’s C++ headers/libs)

No PyTorch, hipBLAS, etc.

### Build

| Goal | Command (`bonsai-8b/gpu-rocm/`) |
|---|---|
| Build and run (default) | `make` / `make run` |
| Build only | `make build` |
| Show detected ISA | `make detect-gpu-arch` |
| Show benchmark history | `make log` |
| Run benchmark and append history | `make log.push` (e.g. `make log.push BENCH_N=64`) |

**`GPU_ARCH`** (e.g. `gfx1100`) is auto-detected from **`rocminfo`**. If detection fails, set it explicitly: `make GPU_ARCH=gfx1100 build`. A successful build prints **Detected GPU arch**.

```bash
cd bonsai-8b/gpu-rocm
make run
```

**`FA_BR`**: defaults to **32** (gfx11/gfx12). On GPUs with more shared memory, `make FA_BR=64 build` may work.

### Run

```bash
cd bonsai-8b/gpu-rocm
./bonsai-gpu-rocm ../Bonsai-8B-Q1_0.gguf -p "Hello" -n 16
```

```bash
cd bonsai-8b/gpu-rocm
make run PROMPT="Hello"
```

CLI flags (`-p`, `-n`, `-t`, `-k`, `-s`, `-l`) match the CPU builds.

### Benchmark logging (`make log` / `make log.push`)

| Variable | Default | Meaning |
|---|---|---|
| `BENCH_PROMPT` | Long English text (in Makefile) | Prompt with **~130 tokens** after ChatML encoding |
| `BENCH_N` | `128` | Max new tokens (`-n`) |
| `BENCH_SEED` | `42` | RNG seed (`-s`) |
| `BENCH_LOG_FILE` | `/tmp/benchmark.log` | key=value log written after the run |

```bash
cd bonsai-8b/gpu-rocm
make log.push          # build → benchmark run → parse log → append one line to Makefile
make log               # print BENCH_LOG entries as a table
```

**`log.push` workflow:**

1. Run **`bonsai-gpu-rocm`** with **`BENCH_PROMPT`**, **`-n BENCH_N`**, **`-t 0`**, **`-s BENCH_SEED`**.
2. On exit, write **`prompt_tokens`**, **`gen_tokens`**, **`prefill_tps`**, **`decode_tps`**, **`total_tps`**, etc. to **`/tmp/benchmark.log`** (or **`BENCH_LOG_FILE`**).
3. **`log.push`** reads the log and appends one line  
   **`ISO8601|GPU_ARCH|hostname|prompt|gen|prefill|decode|total`**  
   before **`# BENCH_LOG_END`** in **`gpu-rocm/Makefile`**.

**Note:** **`make log.push` modifies `gpu-rocm/Makefile`**. Check **`git diff`** before committing. Table **`total_tps`** is **inference only** (weight VRAM upload is excluded).

### Reference benchmark (GPU ROCm)

| Item | Value |
|---|---|
| GPU | **gfx1201** / **gfx1100** etc. (ROCm, **`GPU_ARCH`**; see **GPU_ARCH** column in the table) |
| OS | Linux |
| Model | `Bonsai-8B-Q1_0.gguf` |
| Workload | Long prompt (**130** tokens after ChatML) + **128** decode tokens (**`make log.push`** defaults: **`-n 128 -t 0 -s 42`**) |
| Metrics (prefill / decode) | stderr **`Prefill complete` / `Decode complete`** |
| Metrics (total) | Benchmark log **`total_tps`** (inference window; weight H2D excluded) |
| Reproduce | In **`bonsai-8b/gpu-rocm/`**: **`make log.push`** then **`make log`** |

| Timestamp | GPU_ARCH | Prefill tok/s | Decode tok/s | Total tok/s | Notes |
|---|---|---:|---:|---:|---|
| 2026-05-27 17:21 | **gfx1201** | **175.03** | **41.89** | **67.92** | 130+128 tokens |
| 2026-05-27 17:29 | **gfx1201** | **174.18** | **42.06** | **68.08** | Same setup (2nd run) |
| 2026-05-27 19:15 | **gfx1201** | **174.76** | **41.95** | **67.98** | Same setup (3rd run) |
| 2026-05-27 21:40 | **gfx1100** | **206.40** | **46.22** | **75.90** | 130+128 tokens (different GPU/host than **gfx1201** rows) |

Do **not** compare these numbers directly to the **CPU table** (`-p "Hello" -n 16`) or the **CUDA appendix** (prefill 18 + decode 16)—prompt length and token counts differ. **GPU_ARCH** values (**gfx1201** vs **gfx1100**) denote different GPUs and hosts—compare like with like within the table. For short prompts, run `./bonsai-gpu-rocm ... -p "Hello" -n 16 -t 0` manually and read stderr **`--- throughput ---`**.

### Troubleshooting (ROCm build)

**Empty `GPU_ARCH` / build failure:** confirm the GPU appears in `rocminfo`, then try `make GPU_ARCH=gfx1100 build`. **C++ headers not found:** `sudo apt install -y g++ libstdc++-dev`. **ROCm path:** `make ROCM=/opt/rocm build`. **Model missing for `log.push`:** run **`make model`** from the parent **`bonsai-8b/`** directory first.

### Source files to read

10. `bonsai-8b/gpu-rocm/main.c` — host side (`generate`, **`write_benchmark_log`**)  
11. `bonsai-8b/gpu-rocm/kernels.hip` — HIP kernels, VRAM, **`gpu_get_device_desc`**  
12. `bonsai-8b/gpu-rocm/gpu.h` — C API (**`gpu_get_device_desc`** added)  

---

## AMD ROCm / HIP + rocWMMA (`gpu-rocm-wmma`)

**`bonsai-8b/gpu-rocm-wmma/` is also an appendix.** It derives from **`gpu-rocm`** and accelerates **Prefill Attention QK^T** only with **rocWMMA 16×16×16** (**`flash_attn_prefill_wmma_gqa_kernel`**). **Decode** and linear layers (Q1_0 GEMV) match **`gpu-rocm`**. **hipBLAS is not required.** **`make log` / `make log.push`** use the same format as **`gpu-rocm`**.

**PV (scores × V)** in prefill stays on an F32 scalar path (WMMA not used—matrix layout constraints). Prefill / decode / total performance is **GPU-dependent** (**gfx1201** may show lower prefill than **`gpu-rocm`**; **gfx1100** may show higher total). See the header comment in **`kernels.hip`** and **`doc/design.md`**.

### Directory layout

```text
bonsai-8b/gpu-rocm-wmma/
├── Makefile
├── main.c
├── kernels.hip    # Prefill WMMA + shared gpu-rocm kernels
└── gpu.h          # same as gpu-rocm
```

### Build and run

| Goal | Command (`bonsai-8b/gpu-rocm-wmma/`) |
|---|---|
| Build and run (default) | `make` / `make run` |
| Show benchmark history | `make log` |
| Run benchmark and append history | `make log.push` |

Same requirements as **`gpu-rocm`** (ROCm, **`hipcc`**, **`GPU_ARCH`**, **`g++` / `libstdc++-dev`**). Default **`FA_BR=32`** (WMMA transpose buffers increase LDS use).

```bash
cd bonsai-8b/gpu-rocm-wmma
make run
./bonsai-gpu-rocm-wmma ../Bonsai-8B-Q1_0.gguf -p "Hello" -n 16
```

### Reference benchmark (GPU ROCm WMMA)

| Item | Value |
|---|---|
| GPU | **gfx1201** / **gfx1100** etc. (**`GPU_ARCH`**; see **GPU_ARCH** column in the table) |
| Workload | Same as the **`gpu-rocm`** table above (130 + 128 tokens) |
| Reproduce | **`make log.push`** then **`make log`** |

| Timestamp | GPU_ARCH | Prefill tok/s | Decode tok/s | Total tok/s | Notes |
|---|---|---:|---:|---:|---|
| 2026-05-27 21:00 | **gfx1201** | **170.18** | **42.04** | **67.74** | Prefill QK^T via rocWMMA; prefill slightly below **`gpu-rocm`** (~175 tok/s) at same workload |
| 2026-05-27 21:44 | **gfx1100** | **199.00** | **49.01** | **79.02** | 130+128 tokens (different GPU/host than **gfx1201** rows) |
| 2026-05-27 21:44 | **gfx1100** | **199.90** | **49.29** | **79.45** | Same setup (2nd run) |

**GPU_ARCH** values denote different GPUs and hosts—compare like with like within the table and against the **`gpu-rocm`** table. On **gfx1100**, decode / total can exceed **`gpu-rocm`** (total **~76** tok/s).

### Source files to read

13. `bonsai-8b/gpu-rocm-wmma/kernels.hip` — **`flash_attn_prefill_wmma_gqa_kernel`** and WMMA notes  
14. `bonsai-8b/gpu-rocm-wmma/main.c` — host side (same shape as **`gpu-rocm`**)  
