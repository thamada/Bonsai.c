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

**`bonsai-8b/gpu-cuda/` is an appendix outside this repo’s goals** (single C source, minimal dependencies). It splits across `main.c` + `kernels.cu` + `gpu.h` (optionally `fp4_bonsai.cu` / `fp4_gemm.cu` + CUTLASS) and requires **CUDA Toolkit, an NVIDIA driver, and a physical GPU**. The reference implementation to read first is **`cpu/main.c`**.

It is bundled only because the author **wanted to see how fast CUDA could go**. It does not complement the project’s purpose and is not an official feature for readers. It is easy to misread as part of the main project, so **`gpu-cuda/` is planned to move to a separate repository**. First-time readers can **ignore it**.

What follows is a technical note for anyone curious about GPU speed comparisons.

### Directory layout

```text
bonsai-8b/gpu-cuda/
├── Makefile
├── main.c
├── kernels.cu
├── gpu.h
├── fp4_bonsai.cu / fp4_bonsai.h   # NVFP4 bridge (when BONSAI_FP4=1)
├── fp4_gemm.cu / fp4_gemm.h       # CUTLASS block-scaled NVFP4 GEMM
└── third_party/cutlass/           # fetched by make cutlass (FP4 builds)
```

### Technical overview

Same GGUF and CLI as **`cpu-blas`**. The host side (`main.c`) handles GGUF mmap, the tokenizer, and sampling; the GPU side (`kernels.cu`) runs the forward pass.

#### VRAM layout (`gpu_model_create`)

Uploaded from host (mmap’d GGUF) via **H2D copy** at startup:

| Kind | Contents | Format |
|---|---|---|
| Weights | `token_embd`, per-layer `wq/wk/wv/wo/gate/up/down`, `output` | Q1_0 (g128). With **`make run` (FP4)**, linear layers are converted at startup into a **GPU-resident NVFP4 + block-scale cache** (below) |
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

#### Q1_0 GEMV (`gpu_mm` / `gpu_mm_batch`) — default / `make run.no-fp4`

Same approach as **`cpu-blas`**. Weights stay **Q1_0 in VRAM** (no upfront dequant). With **`make run` (NVFP4)**, only linear layers switch to the path in the next section.

1. Quantize the input vector (or each batch row) to Q8_0 (group size 32) with **`quantize_q8_0_kernel`**.
2. Run **`mm_q1_0_kernel`** / **`mm_q1_0_batch_kernel`**: `vec_dot_q1_0_q8_0` — dot product of a Q1_0 weight row and Q8_0 activations; 1-bit sign bits combined with Q8_0 int8 products, restored with FP16 scales.
3. Each Q1_0 block (128 elements) pairs with 4 Q8_0 blocks.

Decode: 256 threads/block parallel over output dimension `d`. Prefill: parallel over `n_tokens × d` output elements.

#### NVFP4 + CUTLASS (Blackwell, `make run`)

Active only when built with **`BONSAI_FP4=1`** (`gpu-cuda/Makefile`: **`make run`** / **`make blackwell`**). **`main.c` does not call NVFP4 directly** — GGUF loading, `generate`, and sampling are unchanged; only **linear-layer GEMV/GEMM inside `kernels.cu`** switches to FP4 Tensor Core processing. Embedding, RMSNorm, RoPE, Flash Attention, SwiGLU, and the KV cache are unchanged.

| File | Role |
|---|---|
| `fp4_gemm.cu` | [CUTLASS](https://github.com/NVIDIA/cutlass) **SM120 block-scaled NVFP4 GEMM** (based on Example 79a). Quantization kernels from BF16 to **E2M1 (NVFP4) + UE4M3 scales**, and `fp4_gemm_run_cached` on Tensor Cores. |
| `fp4_bonsai.cu` | Bonsai bridge: **Q1_0 weights** on the host are restored to BF16, **quantized once at startup** into an NVFP4 cache; at inference time handles **F32 activations ↔ BF16 ↔ CUTLASS GEMM ↔ F32 output**. |
| `kernels.cu` | With `#ifdef BONSAI_FP4`, weight upload in `gpu_model_create` and dispatch in `gpu_mm` / `gpu_mm_batch` to the FP4 path (`gpu_mm_fp4` → `fp4_bonsai_mm`). |

**Where it is used (linear layers only):** per-layer **`wq` / `wk` / `wv` / `wo` / `gate` / `up` / `down`** and final **`output` (LM head)**. In decode, that is steps 3, 8, 9, and 10; in prefill, the `gpu_mm_batch` rows in the table above.

**How it works:**

1. **Build** — `make cutlass` fetches `third_party/cutlass` and links `fp4_gemm.o` / `fp4_bonsai.o`. GPU code must be native **`sm_120a`** (PTX JIT cannot use Tensor Core FP4).
2. **Startup (`gpu_model_create`)** — `fp4_bonsai_init` allocates CUTLASS workspace. Each layer’s Q1_0 tensor is converted via `fp4_bonsai_weight_from_q1_host` into a **GPU-resident FP4 weight cache** (packed NVFP4 + scales; `M`/`N`/`K` padded to **multiples of 128**).
3. **Inference (`gpu_forward` / `gpu_forward_prefill`)** — `gpu_mm` / `gpu_mm_batch` call `fp4_bonsai_mm`. Activations (F32, `M` rows × `n` cols) are uploaded as BF16, **quantized to NVFP4 on the fly**, and multiplied with **cached FP4 weights** via CUTLASS GEMM. Results are written back to F32 `o` / `logits` buffers; subsequent norm / attention kernels match the Q1_0 build.
4. **Check** — stderr at startup should show `GPU: FP4 Tensor Core path enabled (NVFP4, CUTLASS sm_120)` when the FP4 path is active.

With **`make run.no-fp4`**, this section does not apply; only the Q1_0 GEMV path above (Q8_0 activations + `mm_q1_0_*` kernels) is used.

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
| Generic GPU (PTX JIT) | `make build` / `make run.no-fp4` | Q1_0 + Q8_0 kernels (**no NVFP4**) |
| Blackwell + NVFP4 | `make` or `make run` | CUDA 13 + native **`sm_120a`** + CUTLASS (first run may need **sudo**) |

```bash
cd bonsai-8b/gpu-cuda
make build          # or for make run.no-fp4
# make run          # Blackwell + NVFP4 (default target)
```

Produces **`bonsai-gpu-cuda`**. From `bonsai-8b/`: `make build.gpu-cuda` / `make run.gpu-cuda` (PTX, no FP4) also work.

Override **`CUDA_GENCODE`** for your GPU (default for `make run.no-fp4`: PTX `compute_86` + driver JIT):

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

CLI options (`-p`, `-n`, `-t`, `-k`, `-s`, `-l`) match the CPU builds.

```bash
./gpu-cuda/bonsai-gpu-cuda Bonsai-8B-Q1_0.gguf -p "Hello" -n 4
```

### Reference benchmark (GPU)

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 5090 (31 GiB VRAM) |
| OS | Linux |
| Model | `Bonsai-8B-Q1_0.gguf` (uploaded to VRAM at startup) |
| Command | In `bonsai-8b/gpu-cuda`, `make run` or `make run.no-fp4` (each equivalent to `-p "Hello" -n 16 -t 0`) |
| Workload | Same as CPU table above (prefill 18 + decode 16 tokens) |
| Repro | One warmup per configuration, then representative of three runs (prefill / decode from stderr `Prefill complete` / `Decode complete` lines) |

#### FP4 Tensor Core enabled (`make run`)

Blackwell (RTX 50 series): **CUDA 13**, native **`sm_120a`**, **`BONSAI_FP4=1`** (NVFP4 + CUTLASS), **`FA_BR=32`**. From `gpu-cuda/`, `make run` (builds like `make blackwell` then runs; first-time CUDA 13 install may need **sudo**).

| Binary | Prefill tok/s | Decode time | Decode throughput | Notes |
|---|---:|---:|---:|---|
| `gpu-cuda/bonsai-gpu-cuda` | **~1365** | 0.18 s | **~90.4 tok/s** | stderr shows `GPU: FP4 Tensor Core path enabled` (measured 2026-05-21) |

#### FP4 disabled (`make run.no-fp4`)

PTX **`compute_86`** + driver JIT, Q1_0 GEMV (no FP4 path). From `gpu-cuda/`, `make run.no-fp4` (`make build` only).

| Binary | Prefill tok/s | Decode time | Decode throughput | Notes |
|---|---:|---:|---:|---|
| `gpu-cuda/bonsai-gpu-cuda` | **~293** | 0.34 s | **~47.0 tok/s** | Batch prefill, `-use_fast_math` (measured 2026-05-21) |

With the same prompt and `-t 0`, **both configurations** matched **`cpu-blas`** output (`Hello! I'm Bonsai, an AI assistant developed by PrismML.`). `-ffast-math` / `-use_fast_math` can change FP reduction order vs CPU builds; output still matched in these runs. With FP4 enabled, decode was **~1.9×** and prefill **~4.7×** faster (tok/s) than without FP4.

### Troubleshooting (CUDA build)

**CUDA / `nvcc` not found:** Install CUDA Toolkit and an NVIDIA driver, then rebuild in **`gpu-cuda`**. Older CUDA may not support `-arch=native`; the default builds PTX (`compute_86`) for driver JIT. Set **`CUDA_GENCODE`** for your GPU (see **`gpu-cuda/Makefile`**). If you see `[prompt length … exceeds max_seq …]`, increase **`-l`**.

**Clean:** `make clean` at the repo root also removes **`gpu-cuda/bonsai-gpu-cuda`**.

### Source files to read

6. `bonsai-8b/gpu-cuda/main.c` — host side (GGUF, tokenizer, `generate`, sampling); does not switch NVFP4  
7. `bonsai-8b/gpu-cuda/kernels.cu` — forward pass and `gpu_mm*` dispatch (to `fp4_bonsai` when FP4 is enabled)  
8. `bonsai-8b/gpu-cuda/fp4_bonsai.cu` / `fp4_gemm.cu` — Q1_0 ↔ NVFP4 bridge and CUTLASS GEMM (when `BONSAI_FP4=1`)  
9. `bonsai-8b/gpu-cuda/gpu.h` — C API (`gpu_forward_prefill`, etc.)  
