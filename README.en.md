# Bonsai.c

English (this file). Japanese: [README.md](README.md).

This repository runs **[PrismML](https://prismml.com/)’s 1-bit Bonsai 8B** from a **GGUF file (`Bonsai-8B-Q1_0`)** using **a single C source**, **without relying on external libraries**.

### About 1-bit Bonsai 8B

Per [Announcing 1-bit Bonsai: The First Commercially Viable 1-bit LLMs](https://prismml.com/news/bonsai-8b) (PrismML, 2026), **1-bit Bonsai 8B** is a **true 1-bit model** end to end—embeddings, attention, MLP, and LM head are all 1-bit, with **no higher-precision “escape hatches”**—at roughly **8.2B parameters**. Weights are released under the **Apache License 2.0**. The post emphasizes **intelligence density**, deployability, and a compact footprint (about **1.15 GB**).

**This repo loads** the Hugging Face GGUF **`Bonsai-8B-Q1_0.gguf`** (Q1_0 quantization). Scope is **text prompt in, text out**; **image input is not supported**.

**This project does not link PyTorch, TensorFlow, JAX, ONNX Runtime, or other ML userland libraries.**

The reference implementation uses **standard C and `libm` only**, built from **`bonsai-8b/cpu/main.c`** into a **single-threaded CPU** binary (`bonsai-cpu`). For faster experimentation on multicore CPUs, **`bonsai-8b/cpu-omp/main.c`** builds **`bonsai-cpu-omp`**, parallelized with **OpenMP** (**standard C + `libm` + OpenMP runtime**).  
For practical throughput on **`Bonsai-8B-Q1_0`**, **`bonsai-8b/cpu-blas/`** builds **`bonsai-cpu-blas`** with **OpenMP + OpenBLAS** and a **fused Q1_0 dot-product kernel** (**standard C + `libm` + OpenMP + OpenBLAS**).

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
    ├── Makefile
    ├── gguf.txt
    ├── Bonsai-8B-Q1_0.gguf.sha256sum
    ├── cpu/
    │   ├── Makefile
    │   └── main.c
    ├── cpu-omp/
    │   ├── Makefile
    │   └── main.c
    └── cpu-blas/
        ├── Makefile
        └── main.c
```

The reference decoder lives under **`bonsai-8b/cpu/`**. The parallel variant is **`bonsai-8b/cpu-omp/`**; the CPU optimized build is **`bonsai-8b/cpu-blas/`**.

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
- **`wget`** (for `make model` to fetch the GGUF)  
- **`Bonsai-8B-Q1_0.gguf`** from [prism-ml/Bonsai-8B-gguf](https://huggingface.co/prism-ml/Bonsai-8B-gguf) (via `make model`)

```bash
sudo apt update
sudo apt install -y build-essential make
# For cpu-blas:
# sudo apt install -y libopenblas-dev
```

## Obtain the model file

`bonsai-8b/Makefile` defaults to **`MODEL=Bonsai-8B-Q1_0.gguf`**. Override with a path argument or `make run.cpu MODEL=...` if needed.

The GGUF is **not** in the repo. Run **`make model`** to download from the URL in `bonsai-8b/gguf.txt` and place the file under `bonsai-8b/`. It verifies SHA256 against **`Bonsai-8B-Q1_0.gguf.sha256sum`** after download (removes the file on failure).

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

Integrity is checked by **`make model`** using **`Bonsai-8B-Q1_0.gguf.sha256sum`**, which should match hashes on [Hugging Face](https://huggingface.co/prism-ml/Bonsai-8B-gguf/tree/main).

## Quick start

```bash
cd bonsai-8b
make model
make build.cpu
./cpu/bonsai-cpu Bonsai-8B-Q1_0.gguf -p "Hello" -n 1
```

(Optional) OpenMP build from `cpu-omp/`:

```bash
cd bonsai-8b/cpu-omp
make build
./bonsai-cpu-omp ../Bonsai-8B-Q1_0.gguf -p "Hello" -n 1
```

## Build & run (CPU)

### Build

```bash
cd bonsai-8b
make build.cpu
```

Produces **`cpu/bonsai-cpu`**.

### Run

```bash
./cpu/bonsai-cpu Bonsai-8B-Q1_0.gguf \
  -p "Give a one-sentence introduction of yourself." \
  -n 16
```

Using the Makefile:

```bash
make run.cpu PROMPT="Give a one-sentence introduction of yourself."
```

Model elsewhere:

```bash
make run.cpu MODEL=/data/models/Bonsai-8B-Q1_0.gguf PROMPT="Hello"
```

## Build & run (CPU + OpenMP, `cpu-omp`)

Same model and CLI as `cpu`; matmul, attention, SwiGLU, etc. are parallelized with **OpenMP**.

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

Same model and CLI as `cpu-omp`, with a **fused Q1_0 dot kernel** (no intermediate FP32 dequant buffer) and **OpenBLAS** (batched `sgemv` for attention and F32 rows). OpenBLAS is pinned to **one thread**; parallelism comes from **OpenMP** (avoids nested threading).

### Build

```bash
sudo apt install -y libopenblas-dev   # if needed
cd bonsai-8b/cpu-blas
make build
```

Produces **`bonsai-cpu-blas`**. From `bonsai-8b/`: `make build.cpu-blas` / `make run.cpu-blas`.

### Run

```bash
cd bonsai-8b/cpu-blas
./bonsai-cpu-blas ../Bonsai-8B-Q1_0.gguf -p "Hello" -n 16
```

```bash
cd bonsai-8b
make run.cpu-blas PROMPT="Hello"
```

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
| `cpu-blas/bonsai-cpu-blas` | 0.5 s | **30.79 tok/s** | `-O3 -fopenmp -march=native -ffast-math -mfma`, OpenBLAS at 1 thread |

Under these conditions, **`cpu-blas` was about 6× faster than `cpu-omp`** and **about 128× faster than `cpu`** (`cpu-omp` was about 21× faster than `cpu`). Generated text (`Hello! I'm Bonsai, an AI assistant developed by PrismML.`) matched on all three CPU binaries.

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

```bash
cd bonsai-8b
make clean
```

`make clean` at `bonsai-8b/` removes **`cpu/bonsai-cpu`** and **`cpu-blas/bonsai-cpu-blas`**. To remove **`cpu-omp/bonsai-cpu-omp`**, run `make clean` inside **`bonsai-8b/cpu-omp`**. The GGUF file is **not** deleted.

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
5. `bonsai-8b/cpu-blas/main.c` — OpenMP + OpenBLAS + fused Q1_0 kernel  

## Out of scope

- Training / fine-tuning  
- **ROCm/HIP, AMD NPU**, **Vulkan, Metal**, and **other GPU runtimes**  
- Batch inference tuning  
- Image input  
- Server or Web API packaging  
- Universal support for every GGUF quantization  
- Guaranteed numerical match with official implementations  

The goal is to **understand, experiment with, and adapt** **Bonsai-8B-Q1_0** (GGUF) text inference in **C**.

## More documentation

- `doc/design.md`  
- `doc/ChangeLog`  

If something fails, check **`bonsai-8b/Makefile`** (`model`, `build.cpu`, `build.cpu-blas`, `run.cpu-blas`, …), per-subdir Makefiles, and the model path you pass at runtime.

---

## NVIDIA CUDA implementation (`gpu-cuda`)

**`bonsai-8b/gpu-cuda/` is an appendix outside this repo’s goals** (single C source, minimal dependencies). It splits across `main.c` + `kernels.cu` + `turboquant.c` + `gpu.h` and requires **CUDA Toolkit, an NVIDIA driver, and a physical GPU**. The reference implementation to read first is **`cpu/main.c`**.

It is bundled only because the author **wanted to see how fast CUDA could go**. It does not complement the project’s purpose and is not an official feature for readers. It is easy to misread as part of the main project, so **`gpu-cuda/` is planned to move to a separate repository**. First-time readers can **ignore it**.

What follows is a technical note for anyone curious about GPU speed comparisons.

### Directory layout

```text
bonsai-8b/gpu-cuda/
├── Makefile
├── main.c
├── kernels.cu
├── turboquant.c
├── turboquant.h
└── gpu.h
```

### Technical overview

Same GGUF and CLI as **`cpu-blas`**. The host side (`main.c`) handles GGUF mmap, the tokenizer, and sampling; the GPU side (`kernels.cu`) runs the forward pass.

#### VRAM layout (`gpu_model_create`)

Uploaded from host (mmap’d GGUF) via **H2D copy** at startup:

| Kind | Contents | Format |
|---|---|---|
| Weights | `token_embd`, per-layer `wq/wk/wv/wo/gate/up/down`, `output` | Q1_0 (g128) |
| Weights | `attn_norm`, `q_norm`, `k_norm`, `ffn_norm`, `output_norm` | F32 |
| KV cache (default) | `kc`, `vc` | F32, `n_layers × max_seq × kv_dim` |
| KV cache (`--turboquant`) | `kc`, `vc` **and** `kc_pack`, `vc_pack` | **Attention uses F32 `kc`/`vc`**. `kc_pack`/`vc_pack` hold a **TurboQuant** compressed copy (PolarQuant 2-bit + QJL 1-bit per coordinate, 48 B/head, layout `[layer][kv_head][seq_pos]`). **Both F32 and pack buffers are allocated** (more VRAM than F32-only) |
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
6. **KV write** — D2D copy into F32 `kc`/`vc`. With **`--turboquant`**, **`kv_tq_compress_kernel`** also compresses into `kc_pack`/`vc_pack` (F32 copy is still performed).
7. **Attention** — `flash_attn_gqa_kernel` (below). Reads **F32 `kc`/`vc`** (same with `--turboquant`) → `xb`.
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
| KV write | `kv_write_batch_kernel` (F32) / with **`--turboquant`**, also **`kv_tq_compress_batch_kernel`** | 1 block / token → F32. With **`--turboquant`**, pack compression then **`kv_write_batch`** fills F32 too |
| Attention | `flash_attn_prefill_gqa_kernel` | 1 block / `(token, head)`, causal mask `npos = t + 1` (**F32 `kc`/`vc`**) |
| Residual, SwiGLU | `add_batch_kernel`, `swiglu_batch_kernel` | element-parallel |

FFN after attention also uses batch kernels; hidden state lives in `x_batch` as `[n_tokens, dim]`.

**LM head for the last token only:** logits are not computed for every position—only `x_batch[(n_tokens-1) * dim]` goes through `rmsnorm_kernel` → `gpu_mm` to produce `logits` (for the first decode token).

#### Q1_0 GEMV (`gpu_mm` / `gpu_mm_batch`)

Same approach as **`cpu-blas`**. Weights stay **Q1_0 in VRAM** (no upfront dequant).

1. Quantize the input vector (or each batch row) to Q8_0 (group size 32) with **`quantize_q8_0_kernel`**.
2. Run **`mm_q1_0_kernel`** / **`mm_q1_0_batch_kernel`**: `vec_dot_q1_0_q8_0` — dot product of a Q1_0 weight row and Q8_0 activations; 1-bit sign bits combined with Q8_0 int8 products, restored with FP16 scales.
3. Each Q1_0 block (128 elements) pairs with 4 Q8_0 blocks.

Decode: 256 threads/block parallel over output dimension `d`. Prefill: parallel over `n_tokens × d` output elements.

#### Flash Attention (GQA, online softmax)

The attention matrix `[n_heads, seq, seq]` is **never materialized**. K/V cache is scanned in **64-token (`FA_BR`) tiles** along the sequence; **online softmax** (running max `m` and sum `l`) updates the output. The **current forward path** loads K/V from **F32 `kc`/`vc`** into shared memory.

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
- **F32 `kc`/`vc`** is shared between prefill and decode—after prefill, decode continues at `pos = n_prompt` using the F32 KV filled during prefill.
- **TurboQuant** (Google Research, [arXiv:2504.19874](https://arxiv.org/abs/2504.19874)): Lloyd-Max scalar quantization (PolarQuant) after a random orthogonal rotation + 1-bit QJL residual. No training; applied online at inference. CLI: **`--turboquant`** adds pack compression (default is F32 KV only). Auto-disabled if **`head_dim != 128`**. **Attention still reads F32 KV** (TurboQuant attention branch in `kernels.cu` is not wired from forward). **No VRAM savings today** with `--turboquant` (F32 and pack buffers coexist).

### Requirements

- **CUDA Toolkit** (`nvcc`) and an **NVIDIA driver** (`libcudart` only; cuBLAS not required)

```bash
sudo apt install -y nvidia-cuda-toolkit   # or NVIDIA’s official CUDA Toolkit
```

### Build

```bash
cd bonsai-8b/gpu-cuda
make build
```

Produces **`bonsai-gpu-cuda`**. From `bonsai-8b/`: `make build.gpu-cuda` / `make run.gpu-cuda`.

Override **`CUDA_GENCODE`** for your GPU (default: PTX `compute_86` + driver JIT):

```bash
make build CUDA_GENCODE=arch=compute_90,code=sm_90
```

### Run

```bash
cd bonsai-8b/gpu-cuda
./bonsai-gpu-cuda ../Bonsai-8B-Q1_0.gguf -p "Hello" -n 16
```

```bash
cd bonsai-8b
make run.gpu-cuda PROMPT="Hello"
```

CLI options (`-p`, `-n`, `-t`, `-k`, `-s`, `-l`) match the CPU builds. **`gpu-cuda` only:** **`--turboquant`** (adds TurboQuant KV pack buffers; attention stays on F32 KV).

```bash
./gpu-cuda/bonsai-gpu-cuda Bonsai-8B-Q1_0.gguf -p "Hello" -n 4
```

### Reference benchmark (GPU)

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 5090 (31 GiB VRAM) |
| OS | Linux |
| Model | `Bonsai-8B-Q1_0.gguf` (uploaded to VRAM at startup) |
| Command | `./gpu-cuda/bonsai-gpu-cuda Bonsai-8B-Q1_0.gguf -p "Hello" -n 16 -t 0` |
| Workload | Same as CPU table above (prefill 18 + decode 16 tokens) |
| Build | `gpu-cuda/Makefile` defaults (PTX `compute_86`, `-use_fast_math`) |

| Binary | Prefill tok/s | Decode time | Decode throughput |
|---|---:|---:|---:|
| `gpu-cuda/bonsai-gpu-cuda` | **~294** | 0.32 s | **50.24 tok/s** |

With the same prompt and `-t 0`, output matched **`cpu-blas`** (`Hello! I'm Bonsai, an AI assistant developed by PrismML.`). `-ffast-math` / `-use_fast_math` can change FP reduction order vs CPU builds; output still matched in this run.

### Troubleshooting (CUDA build)

**CUDA / `nvcc` not found:** Install CUDA Toolkit and an NVIDIA driver, then rebuild in **`gpu-cuda`**. Older CUDA may not support `-arch=native`; the default builds PTX (`compute_86`) for driver JIT. Set **`CUDA_GENCODE`** for your GPU (see **`gpu-cuda/Makefile`**). If you see `[prompt length … exceeds max_seq …]`, increase **`-l`**.

**Clean:** `make clean` at the repo root also removes **`gpu-cuda/bonsai-gpu-cuda`**.

### Source files to read

6. `bonsai-8b/gpu-cuda/main.c` — host side (GGUF, tokenizer, batched prefill / sequential decode in `generate`, sampling)  
7. `bonsai-8b/gpu-cuda/kernels.cu` — CUDA kernels (`gpu_forward_prefill` / `gpu_forward`, Flash Attention, TurboQuant compression, etc.)  
8. `bonsai-8b/gpu-cuda/turboquant.c` — TurboQuant CPU reference (PolarQuant + QJL, table setup)  
9. `bonsai-8b/gpu-cuda/gpu.h` — C API (`gpu_forward_prefill`, etc., `GpuConfig.turboquant_kv`)
