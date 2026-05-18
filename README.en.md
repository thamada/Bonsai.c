# Bonsai.c

English (this file). Japanese: [README.md](README.md).

This repository runs **[PrismML](https://prismml.com/)’s 1-bit Bonsai 8B** from a **GGUF file (`Bonsai-8B-Q1_0`)** using **a single C source**, **without relying on external libraries**.

### About 1-bit Bonsai 8B

Per [Announcing 1-bit Bonsai: The First Commercially Viable 1-bit LLMs](https://prismml.com/news/bonsai-8b) (PrismML, 2026), **1-bit Bonsai 8B** is a **true 1-bit model** end to end—embeddings, attention, MLP, and LM head are all 1-bit, with **no higher-precision “escape hatches”**—at roughly **8.2B parameters**. Weights are released under the **Apache License 2.0**. The post emphasizes **intelligence density**, deployability, and a compact footprint (about **1.15 GB**).

**This repo loads** the Hugging Face GGUF **`Bonsai-8B-Q1_0.gguf`** (Q1_0 quantization). Scope is **text prompt in, text out**; **image input is not supported**.

**This project does not link PyTorch, TensorFlow, JAX, ONNX Runtime, or other ML userland libraries.** Inference uses **standard C and `libm`** only, built from **`bonsai-8b/cpu/main.c`** into a **single-threaded CPU** binary (`bonsai-cpu`). There is no Python or `torch` dependency.

### Why avoid ML libraries?

Typical LLM stacks hide **execution order, memory layout, alignment, and quantization packing** inside the framework.

This repo **makes the full path visible in C**: GGUF read, weight restore, linear algebra, Transformer forward, sampling. The goal is to **inspect, validate, and change** the path—not to replace PyTorch.

That helps with **understandability**, **minimal dependencies**, **room to experiment** with quantization and layouts, and a **small reference** for how **Bonsai 8B (GGUF)** decoder inference can work.

This is **not** aimed at peak performance or full feature parity.

## What you can run

Only the **CPU single-thread** build is provided.

| Mode | Source | Binary | Good for |
|---|---|---|---|
| CPU single-thread | `bonsai-8b/cpu/main.c` | `cpu/bonsai-cpu` | Learning the flow, minimal setup |

An 8B model on CPU can be **slow**. Start with a small `-n` (e.g. `-n 1`) for smoke tests.

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
    └── cpu/
        ├── Makefile
        └── main.c
```

Work mainly under `bonsai-8b/cpu/`.

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
- **`Bonsai-8B-Q1_0.gguf`** from [prism-ml/Bonsai-8B-gguf](https://huggingface.co/prism-ml/Bonsai-8B-gguf)

On Ubuntu-like systems:

```bash
sudo apt update
sudo apt install -y build-essential make
```

## Obtain the model file

`bonsai-8b/Makefile` defaults to **`MODEL=Bonsai-8B-Q1_0.gguf`**. Override with a path argument or `make run.cpu MODEL=...` if needed.

The GGUF is **not** in the repo. Download using the URL in `bonsai-8b/gguf.txt` and place the file under `bonsai-8b/`.

```bash
cd bonsai-8b
url=$(sed 's|/blob/main/|/resolve/main/|' gguf.txt)
wget -O Bonsai-8B-Q1_0.gguf "$url"
```

You should then have:

```text
bonsai-8b/
├── Makefile
├── cpu/ … (`main.c` → `cpu/bonsai-cpu`)
└── Bonsai-8B-Q1_0.gguf
```

Verify integrity using hashes published on [Hugging Face](https://huggingface.co/prism-ml/Bonsai-8B-gguf/tree/main) if needed.

## Quick start

```bash
cd bonsai-8b
make build.cpu
./cpu/bonsai-cpu Bonsai-8B-Q1_0.gguf -p "Hello" -n 1
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

Removes **`cpu/bonsai-cpu`**. The GGUF file is **not** deleted.

## Troubleshooting

### `No such file or directory`

Wrong model path.

```bash
ls -lh bonsai-8b/Bonsai-8B-Q1_0.gguf
```

```bash
./cpu/bonsai-cpu /data/models/Bonsai-8B-Q1_0.gguf -p "Hello" -n 4
```

### CPU is slow

Expected for an 8B model on a single CPU thread. Try `-n 1` or `-n 4`:

```bash
./cpu/bonsai-cpu Bonsai-8B-Q1_0.gguf -p "Hello" -n 1
```

## Reading the codebase

1. `README.en.md` / `README.md` — build and run  
2. `doc/design.md` — design and quantization  
3. `bonsai-8b/cpu/main.c` — load GGUF through one-token generation  

## Out of scope

- Training / fine-tuning  
- GPU / NPU / OpenMP builds (removed from this repo)  
- Batch inference tuning  
- Image input  
- Server or Web API packaging  
- Universal support for every GGUF quantization  
- Guaranteed numerical match with official implementations  

The goal is to **understand, experiment with, and adapt** **Bonsai-8B-Q1_0** (GGUF) text inference in **C**.

## More documentation

- Design: `doc/design.md`  
- Changelog: `doc/ChangeLog`  

If something fails, check `bonsai-8b/Makefile` targets and the model path you pass at runtime.
