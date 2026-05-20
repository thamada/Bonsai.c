# Bonsai.c

English (this file). Japanese: [README.md](README.md).

This repository runs **[PrismML](https://prismml.com/)’s 1-bit Bonsai 8B** from a **GGUF file (`Bonsai-8B-Q1_0`)** using **a single C source**, **without relying on external libraries**.

### About 1-bit Bonsai 8B

Per [Announcing 1-bit Bonsai: The First Commercially Viable 1-bit LLMs](https://prismml.com/news/bonsai-8b) (PrismML, 2026), **1-bit Bonsai 8B** is a **true 1-bit model** end to end—embeddings, attention, MLP, and LM head are all 1-bit, with **no higher-precision “escape hatches”**—at roughly **8.2B parameters**. Weights are released under the **Apache License 2.0**. The post emphasizes **intelligence density**, deployability, and a compact footprint (about **1.15 GB**).

**This repo loads** the Hugging Face GGUF **`Bonsai-8B-Q1_0.gguf`** (Q1_0 quantization). Scope is **text prompt in, text out**; **image input is not supported**.

**This project does not link PyTorch, TensorFlow, JAX, ONNX Runtime, or other ML userland libraries.**

The primary path uses **standard C and `libm` only**, built from **`bonsai-8b/cpu/main.c`** into a **single-threaded CPU** binary (`bonsai-cpu`). For faster experimentation on multicore CPUs, **`bonsai-8b/cpu-omp/main.c`** builds **`bonsai-cpu-omp`**, parallelized with **OpenMP** (**standard C + `libm` + OpenMP runtime**).  
For practical throughput on **`Bonsai-8B-Q1_0`**, **`bonsai-8b/cpu-blas/`** builds **`bonsai-cpu-blas`** with **OpenMP + OpenBLAS** and a **fused Q1_0 dot-product kernel** (**standard C + `libm` + OpenMP + OpenBLAS**). There is no Python or `torch` dependency.

### Why avoid ML libraries?

Typical LLM stacks hide **execution order, memory layout, alignment, and quantization packing** inside the framework.

This repo **makes the full path visible in C**: GGUF read, weight restore, linear algebra, Transformer forward, sampling. The goal is to **inspect, validate, and change** the path—not to replace PyTorch.

That helps with **understandability**, **minimal dependencies**, **room to experiment** with quantization and layouts, and a **small reference** for how **Bonsai 8B (GGUF)** decoder inference can work.

This is **not** aimed at peak performance or full feature parity.

## What you can run

You get a **single-thread CPU** reference build, an **OpenMP multicore** build, and an **OpenMP + OpenBLAS** optimized build.

| Mode | Source | Binary | Good for |
|---|---|---|---|
| CPU single-thread | `bonsai-8b/cpu/main.c` | `cpu/bonsai-cpu` | Learning the flow, minimal dependencies |
| CPU + OpenMP | `bonsai-8b/cpu-omp/main.c` | `cpu-omp/bonsai-cpu-omp` | Multicore smoke tests (reference) |
| CPU + OpenMP + OpenBLAS | `bonsai-8b/cpu-blas/main.c` | `cpu-blas/bonsai-cpu-blas` | Practical multicore throughput (**recommended**) |

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

The reference decoder path lives under **`bonsai-8b/cpu/`**. The parallel variant is **`bonsai-8b/cpu-omp/`**; the optimized build is **`bonsai-8b/cpu-blas/`**.

## Beginners: what happens during LLM inference?

Roughly:

1. **Read the GGUF** — weights, vocabulary, hyperparameters.  
2. **Tokenize** — map the prompt string to token IDs.  
3. **Run the Transformer** one token at a time.  
4. **Sample** the next token (`-t`, `-k`, …).  
5. **Decode** tokens to text.

Here, that pipeline is **not** inside PyTorch; you can follow it **in the C source**.

## Requirements

- Linux  
- `make`  
- A C compiler (`gcc`, `clang`, …)  
- `libm`  
- For **`cpu-omp`**, a compiler/toolchain with **OpenMP** (`libgomp` or `libomp`, commonly bundled with GCC/Clang)  
- For **`cpu-blas`**, the above plus **OpenBLAS** (e.g. `libopenblas-dev` on Debian/Ubuntu)  
- **`wget`** (for `make model` to fetch the GGUF)  
- **`Bonsai-8B-Q1_0.gguf`** from [prism-ml/Bonsai-8B-gguf](https://huggingface.co/prism-ml/Bonsai-8B-gguf) (via `make model`)

On Ubuntu-like systems:

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

You should then have:

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

Example with the GGUF one level up:

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

**Reference numbers from one development machine**—your results will vary with CPU, memory, compiler flags, and whether the GGUF is already in RAM. Re-run with the same GGUF and command to compare.

| Item | Value |
|---|---|
| Date | 2026-05-19 |
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

Under these conditions, **`cpu-blas` was about 6× faster than `cpu-omp`** and **about 128× faster than `cpu`** (`cpu-omp` was about 21× faster than `cpu`). Generated text (`Hello! I'm Bonsai, an AI assistant developed by PrismML.`) matched on all three binaries. `-ffast-math` allows different FP reduction order than `cpu-omp`; output still matched in this run.

## Common CLI options

| Option | Example | Meaning |
|---|---|---|
| `-p` | `-p "Hello"` | Prompt |
| `-n` | `-n 64` | Max new tokens |
| `-t` | `-t 0.7` | Temperature |
| `-k` | `-k 0.9` | Top-p |
| `-s` | `-s 1234` | RNG seed |
| `-l` | `-l 512` | Max sequence length |

Example:

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

1. `README.en.md` / `README.md` — build and run  
2. `doc/design.md` — design and quantization  
3. `bonsai-8b/cpu/main.c` — GGUF load through one-token generation (single-thread baseline)  
4. `bonsai-8b/cpu-omp/main.c` — OpenMP parallel variant  
5. `bonsai-8b/cpu-blas/main.c` — OpenMP + OpenBLAS + fused Q1_0 kernel  

## Out of scope

- Training / fine-tuning  
- GPU / NPU stacks (CUDA, Vulkan compute, Metal, …) — no GPU build targets here  
- Batch inference tuning  
- Image input  
- Server or Web API packaging  
- Universal support for every GGUF quantization  
- Guaranteed numerical match with official implementations  

The goal is to **understand, experiment with, and adapt** **Bonsai-8B-Q1_0** (GGUF) text inference in **C**.

## More documentation

- Design: `doc/design.md`  
- Changelog: `doc/ChangeLog`  

If something fails, check **`bonsai-8b/Makefile`** (`model`, `build.cpu`, `build.cpu-blas`, `run.cpu-blas`, …), per-subdir Makefiles, and the model path you pass at runtime.
