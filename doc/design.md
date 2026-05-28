# 設計仕様書

> **注意**: 本ドキュメントは設計仕様書です。変更履歴は `doc/ChangeLog` を参照。以下は**現在の**システム設計である。

## 概要

### リポジトリの目的とスコープ

本リポジトリは **[PrismML](https://prismml.com/) の 1-bit Bonsai 8B** を **GGUF**（既定: **`Bonsai-8B-Q1_0.gguf`**、**Q1_0** 量子化）から、**C ソース**で **テキスト生成推論**する実装である。**プロジェクトの主眼**は **単一 `main.c`・依存最小**の **CPU 3 バリアント**（基準 **`bonsai-8b/cpu/main.c`**、並列 **`cpu-omp`**、実用 **`cpu-blas`**）である。

**PyTorch・TensorFlow・JAX・ONNX Runtime 等の ML ユーザランドにはリンクしない。** 基準経路のコアは **標準 C と `libm`** のみ。旧 **単一 `main-rocm*.c` / XDNA2 NPU** 経路は削除済み。**AMD Radeon** 向け **`bonsai-8b/gpu-rocm/`**（**HIP**・**`hipcc`**）を **付録**として新設する（**`gpu-cuda`** と同趣旨の Q1_0 + Q8_0 GEMV・Flash Attention・prefill バッチ）。

**CPU** については、**参照実装**として **`bonsai-8b/cpu/main.c`（単スレッド）** を保守し、**同ロジックを OpenMP で並列化した `bonsai-8b/cpu-omp/main.c`**（**`bonsai-cpu-omp`**）を検証用に、**OpenMP + OpenBLAS と Q1_0×Q8_0 SIMD 内積**（llama.cpp **`ggml_vec_dot_q1_0_q8_0`** 準拠）を用いた **`bonsai-8b/cpu-blas/main.c`**（**`bonsai-cpu-blas`**）を CPU 実用スループット向けに提供する。**`cpu-blas`** も **`make log` / `make log.push`** で長プロンプトベンチ履歴を **`cpu-blas/Makefile` に記録**（第 2 列は **`BENCH_SIMD`** = **`ARCH_FLAGS`** の短縮ラベル）。

**`bonsai-8b/gpu-cuda/`**（**`main.c`** + **`kernels.cu`** + **`gpu.h`** → **`bonsai-gpu-cuda`**）は **CUDA Runtime のみ（`libcudart`）** の **付録実装**（NVIDIA）。線形層は **Q1_0×Q8_0** のみ（**NVFP4 経路なし**）。README 上は **プロジェクト目的（単一 C・依存最小）の外**と明記し、**将来別リポジトリへ移す予定**。技術メモ・ビルド手順・GPU ベンチマークは README 末尾の **「NVIDIA CUDA 実装（`gpu-cuda`）について」** 付録に、仕様の静的説明は本書に記載。**Prefill** は **`gpu_forward_prefill`**（バッチ並列）、**Decode** は **`gpu_forward`**（1 トークンずつ）。Blackwell 向け **`make run`**（**`make blackwell`**）は **`sm_120a`** ネイティブ + **Q1_0×Q8_0**。

**`bonsai-8b/gpu-cuda-nvfp4/`**（上記と同構成 + **`fp4_bonsai.cu`** / **`fp4_gemm.cu`** + **CUTLASS v4.5.1** → **`bonsai-gpu-cuda-nvfp4`**）は **NVFP4 Tensor Core 専用の別付録**。線形層は起動時 **Q1_0 → NVFP4 キャッシュ** + **CUTLASS SM120 block-scaled GEMM**（本書「NVFP4 + CUTLASS（`gpu-cuda-nvfp4`）」）。**`make run`** 既定は **`sm_120a`** + CUDA 13（**`make blackwell`**）。

**`bonsai-8b/gpu-rocm/`**（**`main.c`** + **`kernels.hip`** + **`gpu.h`** → **`bonsai-gpu-rocm`**）は **ROCm HIP のみ（`libamdhip64`）** の **付録実装**（AMD）。**`gpu-cuda`** とほぼ同一の **`gpu.h` C API**（追加 **`gpu_get_device_desc`**）・GGUF・CLI・**`chat_encode`**・RoPE・prefill/decode 分割。**`make log` / `make log.push`** でベンチ履歴を **`Makefile` に記録**。**hipBLAS 不要**。**NVFP4 経路なし**。ビルドは **`hipcc`**、**`GPU_ARCH`** は **`rocminfo`** から **`gfx*`** を自動検出（上書き可）。

**`bonsai-8b/gpu-rocm-wmma/`**（**`main.c`** + **`kernels.hip`** + **`gpu.h`** → **`bonsai-gpu-rocm-wmma`**）は **`gpu-rocm`** の派生。**Prefill Attention** の **QK^T** のみ **rocWMMA 16×16×16**（**`flash_attn_prefill_wmma_gqa_kernel`**）で加速。**Decode**・線形層（Q1_0 GEMV）は **`gpu-rocm`** と同一。**hipBLAS 不要**。**`make log` / `make log.push`** は **`gpu-rocm`** と同形式。

目的は、推論経路（GGUF 読み取り・量子化復元・Transformer forward・サンプリング）を **C の明示的なコードパス**として追い、改変しやすくすることである。学習・商用 SLA・公式実装との数値一致はスコープ外。

#### ライブラリ非依存の意義

- **理解可能性**: バッファ配置・演算順・量子化レイアウトをソースで追える。
- **依存の単純化**: コンパイラと最小実行環境で再現できる。
- **参照実装**: Bonsai 8B（GGUF）デコーダの最小例として使える。

利用者向けの手順は **`README.md`** / **`README.en.md`** を参照（**本文は CPU 3 バリアント**。**`gpu-cuda`** / **`gpu-rocm`** は README 末尾付録。**`gpu-cuda-nvfp4`** の手順は本書「ビルドと実行（GPU CUDA）」を正とする。詳細仕様の補足は本書）。

### 実装バリアント（現状）

| ソース | 実行環境 | 概要 |
|--------|----------|------|
| `bonsai-8b/cpu/main.c` | **CPU、単スレッド** | **`Bonsai-8B-Q1_0.gguf` 専用**。GGUF mmap。線形重みは **Q1_0** のみ（**`expect_q1_0_weight`** で検証）。**Q1_0** GEMV は融合行内積（**`dot_q1_0_row`**、中間 FP32 展開なし）。Embedding は **`dequant_q1_0_blocks`**。Norm 等は F32。`libm` のみ。 |
| `bonsai-8b/cpu-omp/main.c` | **CPU、OpenMP マルチスレッド** | **`cpu`** と同一スコープ・GGUF・CLI・**`chat_encode`**・RoPE。**Q1_0** は融合行内積（**`mm_q1_0_rows`** を OpenMP 行並列）。Attention・SwiGLU 等も **OpenMP**。 |
| `bonsai-8b/cpu-blas/main.c` | **CPU、OpenMP + OpenBLAS** | **`cpu-omp`** と同一スコープ・GGUF・CLI・**`chat_encode`**・RoPE。**Q1_0** は活性化を **Q8_0** 化して **`vec_dot_q1_0_q8_0`**（AVX2 時は ggml-cpu x86 準拠 SIMD、**`-mfma`**）。Attention はヘッドあたり **2 回の `cblas_sgemv`**。Norm 等 F32 行は OpenMP 行帯 + serial **`sgemv`**。起動時 **`openblas_set_num_threads(1)`**。 |
| `bonsai-8b/gpu-cuda/main.c` + **`kernels.cu`** | **NVIDIA GPU、CUDA（付録）** | **`Bonsai-8B-Q1_0.gguf` 専用**（**`expect_q1_0`** で線形重みを検証）。**`main.c` に CPU 逆量子化コードは持たない**（Q1_0 重みは **`kernels.cu`** 側で mmap → **H2D**）。Prefill/decode 分割・Q8_0 活性化 + Q1_0 GEMV・Flash Attention。 |
| `bonsai-8b/gpu-cuda-nvfp4/main.c` + **`kernels.cu`** | **NVIDIA GPU、CUDA NVFP4（付録）** | **`gpu-cuda`** と同 API・GGUF・prefill/decode 分割。線形層のみ **NVFP4 + CUTLASS**（**`fp4_bonsai_mm`**）。Attention 等は Q1_0 版カーネルと同型。 |
| `bonsai-8b/gpu-rocm/main.c` + **`kernels.hip`** | **AMD GPU、ROCm/HIP（付録）** | **`gpu-cuda`** と同 API・アルゴリズム（Q8_0 活性化 + Q1_0 GEMV、Flash Attention、prefill バッチ）。**hipBLAS 不要**。 |
| `bonsai-8b/gpu-rocm-wmma/main.c` + **`kernels.hip`** | **AMD GPU、ROCm/HIP + rocWMMA（付録）** | **`gpu-rocm`** 派生。**Prefill** の **QK^T** のみ **rocWMMA 16×16×16**（**PV** は F32 スカラー）。**Decode**・線形層は **`gpu-rocm`** 同一。Prefill / total の優劣は **GPU 依存**（**gfx1201** では Prefill が低い例、**gfx1100** では total が上回る例あり）。 |

メタデータの照合は **GGUF 内の `qwen3.*` プレフィックス**（実装上のバイト列。`embedding_length` 等）と tokenizer 系キーで行う。実装コメントは「dense デコーダ」「Bonsai」と整合する。

### 参考ベンチマーク（開発環境）

環境依存の参考値。CPU・GPU・メモリ・ビルドフラグ・GGUF が RAM/VRAM に載っているかで大きく変わる。**CPU 表**の手順全文は **`README.md`** / **`README.en.md`** 本文の「参考ベンチマーク」。**GPU CUDA 表**（Q1_0 + NVFP4）は本書で管理し、Q1_0 の写しは README 付録 **「NVIDIA CUDA 実装（`gpu-cuda`）について」** にも載せる。**GPU ROCm 表**は本書（**`make log.push`**、長プロンプトワークロード。下記）。

#### CPU（2026-05-19 計測）

| 項目 | 値 |
|------|-----|
| CPU | AMD Ryzen 9 5950X（32 論理コア） |
| OS | Linux |
| モデル | `Bonsai-8B-Q1_0.gguf` (pre-read, in RAM) |
| コマンド | `./<binary> Bonsai-8B-Q1_0.gguf -p "Hello" -n 16 -t 0`（`-n` は decode 上限、`-t` は温度） |
| ワークロード | prefill **18** トークン（`-p "Hello"` + chat テンプレート）+ decode **16** トークン（`-n 16`） |
| 表の指標 | decode 時間・tok/s のみ（stderr の `Decode complete` 行。prefill は含めない） |
| 環境変数 | `cpu-omp` / `cpu-blas`: `OMP_NUM_THREADS=32`、`cpu-blas` のみ `OPENBLAS_NUM_THREADS=1` |
| 再現 | 各バイナリで 1 回ウォームアップ後、3 回計測の**最高** decode tok/s (GGUF pre-read in RAM) |

| バイナリ | decode 時間 | decode スループット | 備考 |
|----------|----------:|-----------------:|------|
| `bonsai-cpu` | 66.8 s | 0.24 tok/s | 単スレッド、`-O3`（`cpu/Makefile` 既定） |
| `bonsai-cpu-omp` | 3.2 s | 4.94 tok/s | `-O3 -fopenmp`（`cpu-omp/Makefile` 既定） |
| `bonsai-cpu-blas` | 0.5 s | 30.79 tok/s | `-O3 -fopenmp -ffast-math`、**`ARCH_FLAGS`** は **`/proc/cpuinfo` から自動選択**（5950X 計測時 **`-mavx2 -mfma`**）、OpenBLAS 1 スレッド |

この条件下では **`cpu-blas` が `cpu-omp` の約 6 倍**、**`cpu` の約 128 倍**の decode スループット（`cpu-omp` は `cpu` の約 21 倍）。生成テキスト（`Hello! I'm Bonsai, an AI assistant developed by PrismML.`）は 3 バイナリで一致。`cpu-blas` の Q1_0 は **Q8_0 活性化 + AVX2 内積**（llama.cpp 準拠）。

#### CPU 長プロンプト（2026-05-27 計測、`cpu-blas` **`make log.push`**）

| 項目 | 値 |
|------|-----|
| CPU | AVX（**`BENCH_SIMD=avx`**。計測ホストの cpuinfo 自動検出） |
| OS | Linux |
| モデル | `Bonsai-8B-Q1_0.gguf` |
| ワークロード | 長文プロンプト（ChatML 後 **130** トークン）+ decode **128** トークン（**`-n 128 -t 0 -s 42`**。`cpu-blas/Makefile` の **`BENCH_PROMPT`** / **`BENCH_N`** 既定） |
| 表の指標 | **`/tmp/benchmark.log`**（または **`BENCH_LOG_FILE`**）の **`prefill_tps` / `decode_tps` / `total_tps`** — **推論区間のみ**（モデル mmap 読み込みは含めない） |
| 再現 | `bonsai-8b/cpu-blas/` で **`make log.push`**（**`make log`** で履歴表示） |

| 計測日時 | SIMD | prefill tok/s | decode tok/s | total tok/s | 備考 |
|----------------|------|-------------:|-------------:|------------:|------|
| 2026-05-27 19:39 | **avx** | **1.96** | **1.99** | **1.98** | 130+128 トークン（AVX のみホスト） |
| 2026-05-27 19:57 | **avx** | **1.97** | **2.00** | **1.99** | 同上（2 回目） |
| 2026-05-27 21:31 | **avx2+fma** | **26.34** | **25.85** | **26.09** | 130+128 トークン（AVX2+FMA ホスト。**`BENCH_SIMD`** 列は **`ARCH_FLAGS`** 短縮ラベル） |

短プロンプト CPU 表（5950X・`-p "Hello" -n 16`）とは**直接比較しない**。**SIMD 列**（**`avx`** vs **`avx2+fma`**）は計測ホスト・**`ARCH_FLAGS`** が異なるため、長プロンプト表内でも**横並び比較は SIMD 条件を揃えてから**行う。

#### GPU CUDA（2026-05-21 計測）

| 項目 | 値 |
|------|-----|
| GPU | NVIDIA GeForce RTX 5090（31 GiB VRAM） |
| OS | Linux |
| モデル | `Bonsai-8B-Q1_0.gguf`（起動時 VRAM アップロード） |
| コマンド | **`gpu-cuda`**: **`make run`** / **`make build`**（Q1_0）。**`gpu-cuda-nvfp4`**: **`make run`**（NVFP4、**`make blackwell`**） |
| ワークロード | 上記 CPU 表と同じ（prefill 18 + decode 16） |
| 表の指標（prefill） | stderr の `Prefill complete` 行（**`gpu_forward_prefill`** バッチ、`-use_fast_math`） |
| 表の指標（decode） | decode 時間・tok/s（stderr の `Decode complete` 行） |
| 再現 | 各構成で 1 回ウォームアップ後、3 回計測の代表値 |
| Attention | **Flash Attention + K/V shared staging**（decode: **`flash_attn_gqa_kernel`** `<<<n_heads, FA_HD>>>` / prefill: **`flash_attn_prefill_gqa_kernel`** `<<<n_tokens×n_heads, FA_HD>>>`、因果マスク。**Ampere/Ada 等** は **`FA_BR=64`**（≈65 KB + carveout=100）。**compute 12.x（RTX 50 等）** は Makefile 自動で **`FA_BR=32`**（≈34 KB、静的 shared 上限 48 KB 以内）） |

**Q1_0 GPU（`bonsai-8b/gpu-cuda/`）** — CUDA 13、**`sm_120a`**（compute 12.x 既定）、**`FA_BR=32`**。`-p "Hello, how are you?"`（ChatML 後 **23** prefill + 生成 **~44**、**`-t 0.6`**、2026-05-28 実測）。

| バイナリ | prefill tok/s | decode tok/s | 備考 |
|----------|-------------:|-------------:|------|
| `bonsai-gpu-cuda` | **~312** | **~47** | **`make run`**（**`sm_120a`**）。`cpu-blas` と同趣旨の応答 |

**NVFP4 Tensor Core（`bonsai-8b/gpu-cuda-nvfp4/`）** — CUDA 13、**`sm_120a`** ネイティブ、**CUTLASS v4.5.1**（旧 v3.9.0 は RTX 5090 で GEMM `initialize` 失敗）。2026-05-21 手動ビルド参考値と 2026-05-28 **`make run`** 実測。

| バイナリ | prefill tok/s | decode tok/s | 備考 |
|----------|-------------:|-------------:|------|
| `bonsai-gpu-cuda-nvfp4` | **~1365**（2026-05-21） | **~90.4**（2026-05-21） | 短プロンプト・`-n 16` 相当 |
| `bonsai-gpu-cuda-nvfp4` | **~1767** | **~65** | **`make run`**、上記 Q1_0 プロンプト（2026-05-28） |

**PTX `compute_86` + ドライバ JIT（非推奨・Blackwell）** — RTX 5090（compute **12.0**）で PTX JIT すると Flash Attention が正しく動かず**推論が文字化け**（stderr: `flash_attn … unsupported toolchain`、異常に高い prefill tok/s）。**`make build CUDA_GENCODE=arch=compute_86,code=compute_86` は RTX 50 系では使わない**。2026-05-21 の PTX 表（prefill **~293** / decode **~47** tok/s）は **Ampere/Ada 向け JIT** の参考値。

同一プロンプト・`-t 0` で **Q1_0 構成は `cpu-blas` と同じ生成テキスト**（`Hello! I'm Bonsai, an AI assistant developed by PrismML.`）。2026-05-21 計測では NVFP4（**`gpu-cuda-nvfp4`**）が Q1_0（当時 **`BONSAI_FP4=1` ビルド**）より decode **約 1.9 倍**、prefill **約 4.7 倍**（いずれも tok/s）程度（PTX Q1_0 は decode で **`cpu-blas`（5950X 参考 30.79 tok/s）の約 1.5 倍**程度）。

#### GPU ROCm（2026-05-27 計測、`make log.push`）

| 項目 | 値 |
|------|-----|
| GPU | **gfx1201** / **gfx1100** 等（ROCm。**`rocminfo` → `GPU_ARCH`**。表の **GPU_ARCH** 列） |
| OS | Linux（計測ホスト名例: コンテナ ID） |
| モデル | `Bonsai-8B-Q1_0.gguf` |
| ワークロード | 長文プロンプト（ChatML 後 **130** トークン）+ decode **128** トークン（**`-n 128 -t 0 -s 42`**。`gpu-rocm/Makefile` の **`BENCH_PROMPT`** / **`BENCH_N`** 既定） |
| 表の指標（prefill / decode） | stderr の **`Prefill complete` / `Decode complete`** 行（**`generate()` 内**の prefill/decode 区間のみ） |
| 表の指標（total） | **`/tmp/benchmark.log`**（または **`BENCH_LOG_FILE`**）の **`total_tps`** — **prefill+decode の推論区間**（**重みの H2D（`gpu_model_create`）は含めない**） |
| 再現 | `bonsai-8b/gpu-rocm/` で **`make log.push`**（実行後 **`make log`** で履歴表示。結果は **`Makefile` 内の `BENCH_LOG +=` 行**に追記） |

| 計測日時 | GPU_ARCH | prefill tok/s | decode tok/s | total tok/s | 備考 |
|----------------|----------|-------------:|-------------:|------------:|------|
| 2026-05-27 17:21 | **gfx1201** | **175.03** | **41.89** | **67.92** | 130+128 トークン |
| 2026-05-27 17:29 | **gfx1201** | **174.18** | **42.06** | **68.08** | 同上（2 回目） |
| 2026-05-27 19:15 | **gfx1201** | **174.76** | **41.95** | **67.98** | 同上（3 回目） |
| 2026-05-27 21:40 | **gfx1100** | **206.40** | **46.22** | **75.90** | 130+128 トークン（**gfx1201** ホストとは別 GPU・別ホスト） |

短プロンプト（`-p "Hello" -n 16 -t 0`）の CPU/CUDA 表とは**直接比較しない**（トークン数・prefill 長が異なる）。**GPU_ARCH** 列（**gfx1201** vs **gfx1100**）は計測 GPU・ホストが異なるため、表内でも**横並び比較は GPU 条件を揃えてから**行う。短いプロンプトの参考計測は手動実行と stderr の throughput 行で行う。

#### GPU ROCm WMMA（2026-05-27 計測、`gpu-rocm-wmma` **`make log.push`**）

| 項目 | 値 |
|------|-----|
| GPU | **gfx1201** / **gfx1100** 等（**`GPU_ARCH`**。表の **GPU_ARCH** 列） |
| バイナリ | **`bonsai-gpu-rocm-wmma`** |
| ワークロード | 上記 GPU ROCm 表と同じ（130 + 128 トークン） |
| 再現 | `bonsai-8b/gpu-rocm-wmma/` で **`make log.push`** |

| 計測日時 | GPU_ARCH | prefill tok/s | decode tok/s | total tok/s | 備考 |
|----------------|----------|-------------:|-------------:|------------:|------|
| 2026-05-27 21:00 | **gfx1201** | **170.18** | **42.04** | **67.74** | Prefill QK^T=rocWMMA。同一条件の **`gpu-rocm`**（prefill **~175 tok/s**）より Prefill はやや低い（query 16 行ブロック・LDS 増で並列度が下がりうる。`kernels.hip` 先頭コメント参照） |
| 2026-05-27 21:44 | **gfx1100** | **199.00** | **49.01** | **79.02** | 130+128 トークン（**gfx1201** ホストとは別 GPU・別ホスト） |
| 2026-05-27 21:44 | **gfx1100** | **199.90** | **49.29** | **79.45** | 同上（2 回目） |

**GPU_ARCH** 列は計測 GPU・ホストが異なるため、表内でも **`gpu-rocm`** 表と同様に条件を揃えて比較する。**gfx1201** では Prefill が **`gpu-rocm`** より低い例があるが、**gfx1100** では decode / total が **`gpu-rocm`**（prefill **~206** / total **~76** tok/s）を上回る例もある（GPU・ワークロード依存）。

## ディレクトリとファイル構成

| パス | 役割 |
|------|------|
| `README.md` / `README.en.md` | **入口（日／英）**。**本文**: CPU 3 バリアントのビルド・実行・ベンチマーク・トラブルシュート（`make model` 含む）。**付録**（`---` 以降）: **`gpu-cuda`**（NVIDIA）、**`gpu-rocm`**（AMD）、**`gpu-rocm-wmma`**（AMD・Prefill WMMA）。**`cpu-blas`** の **`ARCH_FLAGS`**（cpuinfo 自動検出）・**`log`/`log.push`** も記載。 |
| `bonsai-8b/Makefile` | **`model` のみ**（GGUF 取得と SHA256 検証）。**`.DEFAULT_GOAL := model`**。ビルド・実行・**`clean`** は各サブディレクトリの Makefile を直接使用（**`cpu/`**・**`cpu-omp/`**・**`cpu-blas/`**・**`gpu-cuda/`**・**`gpu-cuda-nvfp4/`**・**`gpu-rocm/`**・**`gpu-rocm-wmma/`**）。 |
| `bonsai-8b/cpu/Makefile` | `bonsai-cpu` の生成。`MODEL` 既定は **`../Bonsai-8B-Q1_0.gguf`**。 |
| `bonsai-8b/cpu-omp/Makefile` | **`bonsai-cpu-omp`** の生成（`-fopenmp`）。`MODEL` 既定は **`../Bonsai-8B-Q1_0.gguf`**。 |
| `bonsai-8b/cpu-blas/Makefile` | **`bonsai-cpu-blas`** の生成（`-fopenmp`、OpenBLAS、**`-funroll-loops -ffast-math`**）。**`/proc/cpuinfo`** の `flags` から **AVX2 → `-mavx2`（FMA ありなら `-mfma`）**、AVX のみ → **`-mavx`**、それ以外 → **`-march=x86-64`** を **`ARCH_FLAGS`** に自動設定（上書き: **`make build ARCH_FLAGS='-mavx2 -mfma'`**）。ビルド時に **`SIMD flags:`** を表示。`pkg-config openblas` があれば `-I`/`-L` を自動付与。**`log`** / **`log.push`**（**`BENCH_SIMD`** 列付き **`BENCH_LOG`** 追記）。 |
| `bonsai-8b/cpu/main.c` | CPU 単スレッド推論の**参照ソース**（アルゴリズムの正として追う）。**`Bonsai-8B-Q1_0.gguf` 専用**（線形重み **Q1_0** + norm **F32**）。**単一 `main.c`**（標準 C + `libm` のみ）。 |
| `bonsai-8b/cpu-omp/main.c` | **`cpu/main.c` をベースに OpenMP を付与した派生**（挙動確認はまず `cpu` を正とする）。**単一 `main.c`**（+ OpenMP）。 |
| `bonsai-8b/cpu-blas/main.c` | **`cpu-omp` をベースに OpenBLAS・Q1_0×Q8_0 SIMD 内積・Attention `sgemv` 集約を付与した派生**（スループット試行の推奨経路）。**単一 `main.c`**（+ OpenMP + OpenBLAS）。終了時 **`write_benchmark_log`**（**`BENCH_LOG_FILE`** 既定 **`/tmp/benchmark.log`**）。 |
| `bonsai-8b/gpu-cuda/main.c` | **`cpu-blas` をベースに CPU 演算を GPU API 呼び出しへ置換したホスト側**（GGUF mmap・メタデータ・トークナイザ・サンプリング・進捗表示）。**CPU デ量子化コードは持たない**（**`load_weights`** で **`expect_q1_0`** 検証後、重み blob を **`gpu_model_create`** へ渡す）。**`gpu.h`** の C API 経由で **`kernels.cu`** を呼ぶ。 |
| `bonsai-8b/gpu-cuda/kernels.cu` | **CUDA カーネルと VRAM 管理**（**Q1_0×Q8_0 GEMV** のみ。単トークン／**バッチ** Norm・RoPE・SwiGLU、**Flash Attention** decode/prefill、**`gpu_forward`** / **`gpu_forward_prefill`**、**`flash_attn_init_once`** 等）。 |
| `bonsai-8b/gpu-cuda/gpu.h` | **`GpuModel`** / **`gpu_model_create`** / **`gpu_forward`** / **`gpu_forward_prefill`** / **`gpu_copy_logits`** 等の C API（**`extern "C"`**）。 |
| `bonsai-8b/gpu-cuda/Makefile` | **`bonsai-gpu-cuda`** の生成。**`.DEFAULT_GOAL := run`**（**`make blackwell`** → **`sm_120a`** + Q1_0）。**`build`**: **`nvidia-smi` 自動 `CUDA_GENCODE`**（**12.x → `sm_120` + `FA_BR=32`** 等、上書き可）。**`blackwell`**: 初回のみ apt（CUDA 11 削除 → **`cuda-toolkit-13`**）；**`$(CUDA13_NVCC)` が CUDA 13+ なら apt をスキップ**（**`FORCE_CUDA_APT=1`** で強制再実行）。**`.build_config.stamp`**（**`CUDA_GENCODE|fa=…`**）。**`nvcc` 実体を `PATH` 先頭**（`nvlink` 対策）。 |
| `bonsai-8b/gpu-cuda-nvfp4/main.c` | **`gpu-cuda/main.c` と同趣旨のホスト側**（GGUF・CLI・**`expect_q1_0`**）。 |
| `bonsai-8b/gpu-cuda-nvfp4/kernels.cu` | **`gpu-cuda/kernels.cu` をベースに線形層を常に NVFP4 経由**（**`gpu_mm_fp4`** / **`fp4_bonsai_mm`**）。Attention・Norm 等は Q1_0 版と同型。 |
| `bonsai-8b/gpu-cuda-nvfp4/fp4_bonsai.cu` / **`fp4_bonsai.h`** | **Q1_0 → NVFP4 重みキャッシュ**、F32 活性化 ↔ BF16 ↔ **`fp4_gemm_run_cached`** ブリッジ。起動時 **128³ サニティ GEMM**、活性化パディングは **`M_act`/`M_pad` 分離**（OOB 防止）。 |
| `bonsai-8b/gpu-cuda-nvfp4/fp4_gemm.cu` / **`fp4_gemm.h`** | CUTLASS **SM120 block-scaled NVFP4 GEMM**（Example 79a ベース）。各 GEMM 前 **`fp4_gemm_init`**、workspace は縮小しない。 |
| `bonsai-8b/gpu-cuda-nvfp4/gpu.h` | **`gpu-cuda/gpu.h` と同一**。 |
| `bonsai-8b/gpu-cuda-nvfp4/Makefile` | **`bonsai-gpu-cuda-nvfp4`** の生成。**`.DEFAULT_GOAL := run`**（**`make blackwell`** → **`sm_120a`** + NVFP4）。**`make cutlass`**（**`CUTLASS_TAG=v4.5.1`**）。**`blackwell`** / **`.build_config.stamp`** は **`gpu-cuda`** と同趣旨（**`fp4_gemm.o` / `fp4_bonsai.o`** もスタンプ依存）。 |
| `bonsai-8b/gpu-cuda-nvfp4/third_party/cutlass/` | **`make cutlass`** で取得（**v4.5.1** 必須。旧 v3.9.0 は RTX 5090 で GEMM `initialize` 失敗）。 |
| `bonsai-8b/gpu-rocm/main.c` | **`cpu-blas` / `gpu-cuda` と同趣旨のホスト側**（GGUF mmap・メタデータ・トークナイザ・サンプリング・進捗表示）。**CPU デ量子化コードは持たない**（Q1_0 重みは **`kernels.hip`** 側で mmap → **H2D**）。終了時 **`write_benchmark_log`**（**`BENCH_LOG_FILE`** 既定 **`/tmp/benchmark.log`**）。**`generate()`** 内の prefill/decode 計測は **重み H2D 後**のみ。 |
| `bonsai-8b/gpu-rocm/kernels.hip` | **HIP カーネルと VRAM 管理**（Q1_0×Q8_0 GEMV、Norm・RoPE・SwiGLU、**Flash Attention** decode/prefill、**`gpu_forward`** / **`gpu_forward_prefill`**、**`flash_attn_init_once`**、**`gpu_get_device_desc`**）。**`GpuBlockQ1_0` / `GpuBlockQ8_0`** は **`gpu.h`** 定義。 |
| `bonsai-8b/gpu-rocm/gpu.h` | **`gpu-cuda/gpu.h` をベースに `gpu_get_device_desc` を追加**（`GpuModel`・`gpu_forward*` 等は同一）。 |
| `bonsai-8b/gpu-rocm/Makefile` | **`bonsai-gpu-rocm`** の生成（**`-Wall -Wextra -Wno-unused-parameter`**、**`hipcc -x hip`**）。**`log`**（**`BENCH_LOG`** 履歴を表形式表示）、**`log.push`**（**`BENCH_PROMPT`**・**`BENCH_N`**・**`BENCH_SEED`** でベンチ実行→ログ解析→**`Makefile` に `BENCH_LOG +=` 追記**）。**`BENCH_LOG_FILE`** 既定 **`/tmp/benchmark.log`**。 |
| `bonsai-8b/gpu-rocm-wmma/main.c` | **`gpu-rocm/main.c` をベースにしたホスト側**（GGUF・CLI・**`write_benchmark_log`** 同一）。 |
| `bonsai-8b/gpu-rocm-wmma/kernels.hip` | **HIP カーネル**（**`gpu-rocm`** 共通部分 + **Prefill** **`flash_attn_prefill_wmma_gqa_kernel`**）。**rocWMMA**（**`#include <rocwmma/rocwmma.hpp>`**）で **QK^T** のみ WMMA。**PV** は F32 スカラー（行列レイアウト都合で WMMA 未採用。`kernels.hip` 先頭に開発知見）。**`blockDim=32`**（wave32）。 |
| `bonsai-8b/gpu-rocm-wmma/gpu.h` | **`gpu-rocm/gpu.h` と同一**（**`gpu_get_device_desc`** 含む）。 |
| `bonsai-8b/gpu-rocm-wmma/Makefile` | **`bonsai-gpu-rocm-wmma`** の生成（**`gpu-rocm/Makefile` と同趣旨** + リンク時 **Prefill Attention: rocWMMA** 表示）。**`FA_BR` 既定 32**（WMMA 転置バッファで LDS 増）。**`log` / `log.push`** 形式は **`gpu-rocm`** と同一（第 2 列は **`GPU_ARCH`**）。 |
| `bonsai-8b/gguf.txt` | 既定 GGUF の Hugging Face URL（`blob/main` 形式）。 |
| `bonsai-8b/Bonsai-8B-Q1_0.gguf.sha256sum` | 既定 GGUF の SHA256 チェックサム（`make model` の検証に使用）。 |
| `doc/design.md` | 本書。 |
| `doc/ChangeLog` | 変更履歴。 |
| `.gitignore` | ビルド生成物（**`bonsai-8b/cpu/bonsai-cpu`**、**`bonsai-8b/cpu-omp/bonsai-cpu-omp`**、**`bonsai-8b/cpu-blas/bonsai-cpu-blas`**、**`bonsai-8b/gpu-cuda/bonsai-gpu-cuda`**、**`bonsai-8b/gpu-cuda/*.o`**、**`bonsai-8b/gpu-cuda/.build_config.stamp`**、**`bonsai-8b/gpu-cuda-nvfp4/bonsai-gpu-cuda-nvfp4`**、**`bonsai-8b/gpu-cuda-nvfp4/*.o`**、**`bonsai-8b/gpu-cuda-nvfp4/.build_config.stamp`**、**`bonsai-8b/gpu-cuda-nvfp4/third_party/`**、**`bonsai-8b/gpu-rocm/bonsai-gpu-rocm`**、**`bonsai-8b/gpu-rocm/*.o`**、**`bonsai-8b/gpu-rocm-wmma/bonsai-gpu-rocm-wmma`**、**`bonsai-8b/gpu-rocm-wmma/*.o`** 等）、`*.gguf` 等。 |

### Make ターゲット（`bonsai-8b/Makefile`）

| ターゲット | 出力 | 動作 |
|------------|------|------|
| `model`（既定） | `$(MODEL)`（既定 **`Bonsai-8B-Q1_0.gguf`**） | **`$(MODEL)` が既にある** → 再ダウンロードせず **`sha256sum --check`** のみ。**ない** → **`gguf.txt`** の URL を `resolve/main` に変換して **`wget`** 後に検証。失敗時は破損ファイルを削除 |

**`bonsai-8b/Makefile`** 用変数:

| 変数 | 意味 | 既定例 |
|------|------|--------|
| `MODEL` | GGUF ファイル名（`bonsai-8b/` 直下） | `Bonsai-8B-Q1_0.gguf` |

### Make ターゲット（サブディレクトリ）

ビルド・実行・**`clean`** は各サブディレクトリの Makefile を使用する。

| 場所 | ターゲット | 出力 | ソース |
|------|------------|------|--------|
| `bonsai-8b/cpu/Makefile` | `build` / `run` / `clean` | `cpu/bonsai-cpu` | `cpu/main.c` |
| `bonsai-8b/cpu-omp/Makefile` | `build` / `run` / `clean` | `cpu-omp/bonsai-cpu-omp` | `cpu-omp/main.c` |
| `bonsai-8b/cpu-blas/Makefile` | `build` / `run` / `clean` / **`log`** / **`log.push`** | `cpu-blas/bonsai-cpu-blas` | `cpu-blas/main.c` |
| `bonsai-8b/gpu-cuda/Makefile` | **`run`**（既定）/ `build` / `clean` / **`blackwell`** | `gpu-cuda/bonsai-gpu-cuda` | `main.c` + `kernels.cu` |
| `bonsai-8b/gpu-cuda-nvfp4/Makefile` | **`run`**（既定）/ `build` / `clean` / **`blackwell`** / **`cutlass`** | `gpu-cuda-nvfp4/bonsai-gpu-cuda-nvfp4` | `main.c` + `kernels.cu` + `fp4_*.cu` |
| `bonsai-8b/gpu-rocm/Makefile` | **`run`**（既定）/ `build` / `clean` / **`detect-gpu-arch`** / **`log`** / **`log.push`** | `gpu-rocm/bonsai-gpu-rocm` | `main.c` + `kernels.hip`（**`hipcc -x hip`**） |
| `bonsai-8b/gpu-rocm-wmma/Makefile` | **`run`**（既定）/ `build` / `clean` / **`detect-gpu-arch`** / **`log`** / **`log.push`** | `gpu-rocm-wmma/bonsai-gpu-rocm-wmma` | `main.c` + `kernels.hip`（**rocWMMA** Prefill Attention） |

**各サブディレクトリ Makefile** には **`MODEL`** / **`PROMPT`** があり、`MODEL` の既定は **`../Bonsai-8B-Q1_0.gguf`**（親 **`bonsai-8b/`** の GGUF）。

```bash
cd bonsai-8b
make model              # または make（既定ターゲット）
make model MODEL=別名.gguf
```

CPU 単スレッド（`cpu/`）:

```bash
cd bonsai-8b/cpu
make build
make run PROMPT="試す文"
# または ./bonsai-cpu ../Bonsai-8B-Q1_0.gguf -p "試す文" -n 8
```

OpenMP 版（別ディレクトリ）:

```bash
cd bonsai-8b/cpu-omp
make build
./bonsai-cpu-omp ../Bonsai-8B-Q1_0.gguf -p "Hello" -n 4
# スレッド数: OMP_NUM_THREADS=8 など
```

OpenMP + OpenBLAS 版（推奨）:

```bash
cd bonsai-8b/cpu-blas
make build
make run PROMPT="Hello"
# 長プロンプトベンチ: make log.push → make log
```

GPU CUDA 版 Q1_0（NVIDIA、`gpu-cuda/`）:

```bash
cd bonsai-8b/gpu-cuda
make build              # nvidia-smi で CUDA_GENCODE / FA_BR を自動選択
make run                # Blackwell: blackwell（CUDA 13、sm_120a）→ 推論
# 上書き例: make build CUDA_GENCODE=arch=compute_90,code=sm_90
```

GPU CUDA 版 NVFP4（NVIDIA、`gpu-cuda-nvfp4/`。CUTLASS v4.5.1 要）:

```bash
cd bonsai-8b/gpu-cuda-nvfp4
make cutlass            # 初回または CUTLASS 更新時（rm -rf third_party/cutlass 後でも可）
make run                # blackwell → sm_120a + NVFP4
# ./bonsai-gpu-cuda-nvfp4 ../Bonsai-8B-Q1_0.gguf -p "Hello"
```

GPU ROCm 版（AMD）:

```bash
cd bonsai-8b/gpu-rocm
make run                # 既定ターゲット（ビルド＋実行）
# GPU_ARCH: rocminfo 自動。手動: make GPU_ARCH=gfx1100 build
```

GPU ROCm WMMA 版（AMD、Prefill Attention 実験）:

```bash
cd bonsai-8b/gpu-rocm-wmma
make run
# make log.push   # ベンチ履歴を Makefile に追記
```

## ビルドと実行（CPU）

単スレッド:

```bash
cd bonsai-8b/cpu
make build
./bonsai-cpu ../Bonsai-8B-Q1_0.gguf -p "Hello" -n 8
```

OpenMP:

```bash
cd bonsai-8b/cpu-omp
make build
./bonsai-cpu-omp ../Bonsai-8B-Q1_0.gguf -p "Hello" -n 8
```

OpenMP + OpenBLAS:

```bash
cd bonsai-8b/cpu-blas
make build          # SIMD flags: /proc/cpuinfo から自動（例: -mavx2 -mfma）
./bonsai-cpu-blas ../Bonsai-8B-Q1_0.gguf -p "Hello" -n 8
# 手動指定: make build ARCH_FLAGS='-mavx2 -mfma'
# 長プロンプトベンチ: make log.push → make log
```

**`cpu-blas` ベンチマークログ（`log.push`）**:

| 変数 | 既定 | 意味 |
|------|------|------|
| `BENCH_PROMPT` | 長文英文（Makefile 内） | ChatML 化後 **約 130 トークン** |
| `BENCH_N` | `128` | 最大生成トークン数 |
| `BENCH_SEED` | `42` | 乱数シード |
| `BENCH_LOG_FILE` | `/tmp/benchmark.log` | key=value ログ出力先 |
| `BENCH_SIMD` | cpuinfo 自動 | 履歴第 2 列（例: **`avx2+fma`**） |

**`BENCH_LOG` 形式**: **`ISO8601|BENCH_SIMD|hostname|prompt|gen|prefill|decode|total`**。**`total_tps`** は推論区間のみ（モデル mmap 読み込みは含めない）。**`make log.push` は `cpu-blas/Makefile` を書き換える**。

## ビルドと実行（GPU CUDA）

### Q1_0（`bonsai-8b/gpu-cuda/`）

| 目的 | コマンド |
|------|----------|
| GPU 自動検出でビルド | **`make build`** |
| Blackwell + Q1_0（`sm_120a`、既定） | **`make`** / **`make run`**（**`make blackwell`**） |

```bash
cd bonsai-8b/gpu-cuda
make run
./bonsai-gpu-cuda ../Bonsai-8B-Q1_0.gguf -p "Hello" -n 8
```

### NVFP4（`bonsai-8b/gpu-cuda-nvfp4/`）

| 目的 | コマンド |
|------|----------|
| CUTLASS 取得（v4.5.1） | **`make cutlass`** |
| Blackwell + NVFP4（`sm_120a`、既定） | **`make`** / **`make run`**（**`make blackwell`**） |

```bash
cd bonsai-8b/gpu-cuda-nvfp4
make cutlass
make run
./bonsai-gpu-cuda-nvfp4 ../Bonsai-8B-Q1_0.gguf -p "Hello" -n 8
```

**要件（共通）**: CUDA Toolkit（**`nvcc`**）、NVIDIA ドライバ、**`libcudart`**。cuBLAS は不要（Attention はカスタム CUDA カーネル）。Debian/Ubuntu の **`apt install nvidia-cuda-toolkit`** は **CUDA 11** 系であり、**Blackwell（compute capability 12.x）** および **NVFP4 Tensor Core** には **CUDA 13 以降**と **`sm_120a`** ネイティブコードが必要。

**Blackwell（RTX 50 系等）** では各ディレクトリで **`make run`**（内部で **`make blackwell`**）を使う。**初回**は要 **sudo** のことが多い。**CUDA 13 が既に入っている**（**`/usr/local/cuda/bin/nvcc`** が release 13+）場合は **`apt-get update` / `install` をスキップ**し、ビルドのみ行う（**`gpu-cuda-nvfp4`** は CUTLASS 取得が別途必要）。**`FORCE_CUDA_APT=1`** で apt 手順を強制。

初回（apt 実行時）の処理:

1. apt の **CUDA 11** 系パッケージ（**`nvidia-cuda-toolkit`** 等）を **`apt-get remove --purge`**
2. NVIDIA 公式リポジトリ（**`cuda-keyring`**）を追加し **`cuda-toolkit-13`** をインストール（**`/usr/local/cuda/bin/nvcc`**）。**`/usr/local/bin/nvlink`** シンボリックリンクも作成
3. **`BLACKWELL_GENCODE=arch=compute_120a,code=sm_120a`**、**`FA_BR=32`** で **`bonsai-gpu-cuda`** または **`bonsai-gpu-cuda-nvfp4`** をビルド

**GPU 自動検出**（**`make build`**）: **`nvidia-smi`** の compute capability から **`CUDA_GENCODE`** を選択（**12.x → `sm_120a`（nvfp4 既定）または `sm_120`**、**`FA_BR=32`**、CUDA 13 **`nvcc` 優先**）。**RTX 5090 で PTX `compute_86` を JIT すると推論が壊れる**ため、Blackwell ではネイティブコード必須。

**ビルド設定スタンプ**: **`CUDA_GENCODE`** / **`FA_BR`**（**`gpu-cuda-nvfp4`** は **`fp4_gemm.o` / `fp4_bonsai.o`** も）を変更したあと古いオブジェクトが残らないよう、**`.build_config.stamp`** を **`kernels.o`** 等の依存に含める。**`make clean`** で削除。

**Blackwell の Flash Attention 制約**: **sm_120** 系は静的 shared memory 上限 **48 KB**。PTX 既定 **`FA_BR=64`**（≈65 KB）は Blackwell ネイティブで **`ptxas`** 拒否のため **`make blackwell`** は **`FA_BR=32`**（≈34 KB）を自動指定。

**NVFP4 固有**: **CUTLASS v4.5.1**（**`CUTLASS_TAG`**）。旧 **v3.9.0** は RTX 5090（**sm_120a**）で **`fp4_gemm_run_cached: initialize: Error Internal`**。CUTLASS 更新後は **`rm -rf third_party/cutlass && make cutlass`**。

対応 OS: Ubuntu 20.04 / 22.04 / 24.04、Debian 12 / 13。Blackwell 向けは **`gpu-cuda/`** または **`gpu-cuda-nvfp4/`** で **`make blackwell`** / **`make run`**（既定）を使用。

## ビルドと実行（GPU ROCm）

| 目的 | コマンド（`bonsai-8b/gpu-rocm/`） |
|------|-----------------------------------|
| ビルド＋実行（既定） | **`make`** / **`make run`** |
| ビルドのみ | **`make build`** |
| GPU ISA 確認 | **`make detect-gpu-arch`** |
| ベンチマーク履歴表示 | **`make log`** |
| ベンチ実行＋履歴追記 | **`make log.push`**（上書き: **`BENCH_N=64`** 等） |

```bash
cd bonsai-8b/gpu-rocm
make run
./bonsai-gpu-rocm ../Bonsai-8B-Q1_0.gguf -p "Hello" -n 8
```

**要件**: **ROCm**（**`ROCM=/opt/rocm`** 既定、**`$(ROCM)/bin/hipcc`**・**`rocminfo`**）、AMD GPU ドライバ、**`libamdhip64`**。**hipBLAS・PyTorch 等は不要**。ホスト側は **`g++`** と **`libstdc++-dev`**（**`hipcc`** が GCC の C++ ヘッダ／`libstdc++` を参照するため。未検出時はビルド失敗）。**`GPU_ARCH`** 未検出時は **`make GPU_ARCH=gfx1100 build`** 等で明示（**`GPU_ARCH` 空のまま `make` はエラー**）。**`make build`** は **`-Wall -Wextra`** で警告なし（2026-05-27 時点）。

**`FA_BR`**: Makefile 既定 **32**（**`-DFA_BR=32`**）。shared 余裕のある GPU では **`make FA_BR=64 build`** 可（**`kernels.hip`** ソース既定は 64、ビルド時マクロが優先）。

**ベンチマークログ（`log.push`）**:

1. **`bonsai-gpu-rocm`** を **`BENCH_PROMPT`**（既定は約 128 トークン相当の英文）・**`-n BENCH_N`**（既定 **128**）・**`-t 0`**・**`-s BENCH_SEED`** で実行。
2. 終了時 **`write_benchmark_log`** が **`BENCH_LOG_FILE`**（既定 **`/tmp/benchmark.log`**）に key=value 形式で書き出す（**`prompt_tokens`**, **`gen_tokens`**, **`prefill_tps`**, **`decode_tps`**, **`total_tps`**, プロンプト全文・生成全文など）。
3. **`log.push`** がログを読み、**`ISO8601|GPU_ARCH|hostname|prompt|gen|prefill|decode|total`** の 1 行を **`gpu-rocm/Makefile`** の **`# BENCH_LOG_END`** 直前に **`sed -i`** で追記。
4. **`make log`** で **`BENCH_LOG`** 行を表表示。

**`BENCH_LOG` の `total_tps`** は **`generate()` 開始〜終了**の wall time に対する **`(n_prefill + n_decode) / total_sec`** であり、**`gpu_model_create` 以前の重みアップロード時間は含まない**（Makefile コメントと同旨）。

## ビルドと実行（GPU ROCm WMMA）

| 目的 | コマンド（`bonsai-8b/gpu-rocm-wmma/`） |
|------|----------------------------------------|
| ビルド＋実行（既定） | **`make`** / **`make run`** |
| ビルドのみ | **`make build`** |
| GPU ISA 確認 | **`make detect-gpu-arch`** |
| ベンチマーク履歴表示 | **`make log`** |
| ベンチ実行＋履歴追記 | **`make log.push`** |

```bash
cd bonsai-8b/gpu-rocm-wmma
make run
./bonsai-gpu-rocm-wmma ../Bonsai-8B-Q1_0.gguf -p "Hello" -n 8
```

**要件・ビルド変数**は **`gpu-rocm`** と同一（**ROCm**・**`hipcc`**・**`GPU_ARCH`**・**`g++` / `libstdc++-dev`**）。追加で **rocWMMA** ヘッダ（**`$(ROCM)/include/rocwmma`**）が必要。**`FA_BR` 既定 32**（WMMA 転置バッファで LDS 増。64 は未検証）。

**`gpu-rocm` との差分**:

| 項目 | **`gpu-rocm`** | **`gpu-rocm-wmma`** |
|------|----------------|---------------------|
| Prefill Attention QK^T | F32 スカラー（shared staging） | **rocWMMA 16×16×16** |
| Prefill Attention PV | F32 スカラー | F32 スカラー（WMMA 未採用） |
| Decode Attention | **`flash_attn_gqa_kernel`** | 同一 |
| 線形層 | Q1_0×Q8_0 GEMV | 同一 |
| 正確性 | — | WMMA QK + スカラー PV で **`gpu-rocm` と同等生成を確認** |
| 性能 | — | **GPU 依存**（**gfx1201** では Prefill が低い例、**gfx1100** では total が上回る例 — 上記 **GPU ROCm WMMA 表**） |

## 実行時の挙動（CPU）

3 バリアントは **RoPE**・**`chat_encode`** を共通とする。**Q1_0** の行列積は **`cpu`** / **`cpu-omp`** が融合行内積（**`dot_q1_0_row`**）、**`cpu-blas`** が **Q8_0 活性化 + `vec_dot_q1_0_q8_0`**（llama.cpp 準拠）と異なる。差分は並列化（OpenMP）、Q1_0 カーネル、Attention / F32 行への OpenBLAS 利用（**`cpu-blas`** のみ）である。

重みは **mmap した GGUF** 上の量子化 blob を参照する。**CPU 3 バリアント**は **`Bonsai-8B-Q1_0.gguf` 専用** — 線形重みテンソルは **`load_weights`** で **`expect_q1_0_weight`**（**`DT_Q1_0`**）を要求し、**`mm`** / **`emb_lookup`** も **Q1_0 以外はエラー終了**。**`cpu`** / **`cpu-omp`** の **Q1_0** GEMV は **`mm_q1_0_rows`** で符号ビットと FP32 活性の融合行内積（中間 FP32 ブロックなし）。**`cpu-blas`** の **Q1_0** は **`mm_q1_0_rows`** が活性化 **`x`** を一度 **`quantize_row_q8_0`** し、各行を **`vec_dot_q1_0_q8_0`** で計算（**`State.q8`** バッファを使い回し）。Embedding は **`dequant_q1_0_blocks`** で行復元。KV・活性は F32。サンプリングはホスト上の logits に対して行う。

**`-p` プロンプト**は **`chat_encode`** で ChatML 化する。GGUF **`tokenizer.chat_template`** の単一 user ターン + **`add_generation_prompt`** に合わせ、既定 system 文は挿入せず、assistant 開始は空 think ブロック付き（ソース内リテラル。PrismML **llama.cpp -cnv** と同趣旨）。

### 進捗表示とスループット計測（全バリアント共通）

**`generate()`** 内の static ヘルパ（各 **`main.c` にインライン**。共有ヘッダは使わない）で、次の順に出力する。

| タイミング | ストリーム | 内容 |
|------------|------------|------|
| prefill 中 | stderr | `\r` で更新するプログレスバー `Prefill [====...] N% (done/total)` |
| prefill 完了 | stderr | `Prefill complete: <n_prompt> tokens in <sec>s (<tok/s>)` |
| decode 中 | stdout | 生成トークンを逐次表示（`print_tok`） |
| decode 完了 | stderr | 生成テキストの直後に改行を入れ、`Decode complete: <gen> tokens in <sec>s (<tok/s>)` |
| 全体終了 | stdout | `--- <n_prompt> prompt tokens + <gen> generated tokens ---` と `--- <sec>s total ---` |
| 全体終了 | stderr | `--- throughput ---` の下に **prefill / decode / total** の 3 行（tok/s） |

スループットの定義:

- **prefill**: トークン数 **`n_prompt`** ÷ prefill 専用時間（最初のプロンプト forward 完了まで）
- **decode**: トークン数 **`gen`**（サンプリングで出力した数）÷ decode 専用時間（prefill 終了直後〜ループ終了）
- **total**: **`(n_prompt + gen)`** ÷ 全体 wall time（`generate` 開始〜終了）

参考ベンチマーク表: **CPU 表**は **decode 区間**（`Decode complete` 行）のみ。**GPU 表**は prefill 列が **`Prefill complete` 行**、decode 列が **`Decode complete` 行**。prefill / decode / total の 3 値はいずれも CLI の `--- throughput ---` でも確認できる。

**`gpu-cuda`** / **`gpu-rocm`** の prefill は **`gpu_forward_prefill`** による**一括バッチ**のため、プログレスバーは **0% 表示のあと完了時に一気に 100%** となる（CPU 3 バリアントのトークン逐次更新とは異なる）。prefill 時間・tok/s の定義は上記と同じ（バッチ全体の wall time ÷ **`n_prompt`**）。

**`cpu-omp`** は **`cpu`** と同アルゴリズムで、**`mm_q1_0_rows`** の行ループなどに **OpenMP parallel for** を入れたもの（細部は `cpu-omp/main.c` を参照）。

**`cpu-blas`** は次を追加する（細部は `cpu-blas/main.c` を参照）。

1. **起動時**: `openblas_set_num_threads(1)` — BLAS 内部スレッドと OpenMP のネスト並列を避ける。
2. **Q1_0 GEMV**（`mm_q1_0_rows`）: 活性化を **`quantize_row_q8_0`**（AVX/AVX2 時は SIMD、それ以外は参照実装）で **Q8_0** 化し、各行を **`vec_dot_q1_0_q8_0`** で内積（**`ggml_vec_dot_q1_0_q8_0`** / **`ggml-cpu/arch/x86/quants.c`** 準拠。AVX2 時は **`_mm256_maddubs_epi16`** 等と **`-mfma`**）。`d` 行まとめて同一量子化済み **`q8`** を使い回す。
3. **Attention**: ヘッドごとに K 行列（`(pos+1)×hd`、`lda=kv_dim`）と q の **`cblas_sgemv(NoTrans)`**、続けて V の **`cblas_sgemv(Trans)`** で出力ヘッドを得る（従来の `(pos+1)` 回の `sdot`/`saxpy` を集約）。
4. **F32 行**（`mm_f32`、norm 重み等）: OpenMP で行帯を分割し、帯内は serial `sgemv`。

## 実行時の挙動（gpu-cuda）

**`gpu-cuda`** は **`cpu-blas`** と同一の GGUF 読み込み・**`chat_encode`**・RoPE パラメータ・CLI を持つ。差分は forward の実行先・メモリ配置・**prefill / decode の経路分割**である。**`Bonsai-8B-Q1_0.gguf` 専用** — **`load_weights`** で **`expect_q1_0`** が線形重み（embedding・各層・LM head）の **`DT_Q1_0`** を要求する。

1. **起動時**: **`gpu_print_device_info()`** で GPU 名等を表示。**`load_weights`** で **`expect_q1_0`** 検証後、**`gpu_model_create`** 内で **`flash_attn_init_once()`** が **`flash_attn_gqa_kernel`** と **`flash_attn_prefill_gqa_kernel`** の双方に **`cudaFuncAttributePreferredSharedMemoryCarveout=100`** を設定（**`FA_BR=64`** 時 shared ≈ 65 KB/ブロック。Ampere/Ada + PTX JIT 向け）。GGUF を mmap したうえで **Q1_0 重み blob をそのまま** VRAM へ **H2D**（**`main.c` に CPU dequant 経路はない**）。norm・KV キャッシュもアップロード。加えて **prefill バッチ用**に **`x_batch` / `xb_batch` / `q_batch` …**（いずれも **`batch_cap = max_seq`** トークン分）と **`tokens_dev`**・**`q8_batch`**・**`rope_batch`** を **`cudaMalloc`**。**`GpuModel`**（**`gpu.h`**）がデバイス側ポインタを保持（推論中も mmap は保持し、終了時に **`munmap`**）。
2. **Prefill**（**`generate()`**、`n_prompt > 1`）: プロンプト ID 配列を **`gpu_forward_prefill(gm, tokens, n_tokens)`** に渡し、全プロンプト位置を**一度に** forward。各層で **`emb_q1_0_batch`**・**`rmsnorm_batch`** / **`rmsnorm_head_batch`**・**`rope_neox_batch`**・**`gpu_mm_batch`**（Q8_0 量子化 + Q1_0 GEMV のバッチ版）・**`kv_write_batch`**・**`flash_attn_prefill_gqa_kernel`** `<<<n_tokens × n_heads, FA_HD>>>`（位置 **`t`** は K/V の **`0..t`** のみ参照する因果 Attention）・**`swiglu_batch`** / **`add_batch`** を実行。最終 norm + LM head は**末尾トークン**（**`n_tokens - 1`**）のみ **`rmsnorm_kernel`** + **`gpu_mm`** で logits を得る。**`n_prompt == 1`** のときは従来どおり 1 トークン **`gpu_forward`**。
3. **Decode**: 末尾 logits からサンプリングした次トークンを、位置 **`n_prompt + gen_i`** で **1 トークンずつ `gpu_forward`**（teacher forcing なし）。各ステップで **`flash_attn_gqa_kernel`** `<<<n_heads, FA_HD>>>`（デコード用 Flash Attention + K/V staging）。
4. **線形層（`gpu_mm` / `gpu_mm_batch`）**: 単トークンは **`quantize_q8_0_kernel` + `mm_q1_0_kernel`**、prefill バッチは **`quantize_q8_0_batch_kernel` + `mm_q1_0_batch_kernel`**（**`cpu-blas`** 準拠）。**NVFP4** は **`gpu-cuda-nvfp4/`** を参照。
5. **Attention（共通）**: シーケンス方向 **`FA_BR`** タイル（**`-DFA_BR`** で上書き可。既定 **64**、Blackwell ビルド **32**）、K/V を **`k_tile` / `v_tile`**（shared）へ staging、**online softmax**（**`fa_sh_reduce_max` / `fa_sh_reduce_sum`**）、**`att` 非物質化**、GQA 対応。
6. **サンプリング**: prefill 直後および各 decode ステップ後に **`gpu_copy_logits`** で logits を D2H し、CPU で温度・top-p サンプリング（CPU バリアントと同ロジック）。

#### VRAM 配置（`gpu_model_create`、README 付録と同内容）

| 種別 | 内容 | 形式 |
|------|------|------|
| 重み | `token_embd`、各層 `wq/wk/wv/wo/gate/up/down`、`output` | Q1_0（g128） |
| 重み | `attn_norm`、`q_norm`、`k_norm`、`ffn_norm`、`output_norm` | F32 |
| KV キャッシュ | `kc`、`vc` | F32、`n_layers × max_seq × kv_dim` |
| Decode 活性化 | `x`、`xb`、`xb2`、`q`、`k`、`v`、`hb`、`hb2`、`logits`、`q8` | F32 / Q8_0 |
| Prefill バッチ | `x_batch`、`xb_batch`、… `q8_batch`、`tokens_dev`、`rope_batch` 等 | **`batch_cap = max_seq`**（**`-l`**）分を事前確保 |

#### Decode 1 ステップ（`gpu_forward`、レイヤー `l` あたり）

Embedding（`emb_q1_0_kernel`）→ Attention 前 RMSNorm → Q/K/V 投影（`gpu_mm`）→ Q/K head norm → RoPE（cos/sin は CPU 生成→H2D）→ KV 書き込み（D2D）→ **`flash_attn_gqa_kernel`** → `wo` + 残差 → FFN norm → gate/up → SwiGLU → down + 残差。最終層後: output norm → LM head → `logits`。

#### Prefill バッチ（`gpu_forward_prefill`、カーネル対応）

| 処理 | カーネル | 並列 |
|------|----------|------|
| Embedding | `emb_q1_0_batch_kernel` | トークン × Q1_0 ブロック |
| RMSNorm | `rmsnorm_batch_kernel` | 1 block / トークン |
| Q/K/V/O, gate/up/down | `gpu_mm_batch` | `(token, 出力行)` |
| Q/K head norm | `rmsnorm_head_batch_kernel` | `(token, head)` |
| RoPE | `rope_neox_batch_kernel` | `(token, head)` |
| KV 書き込み | `kv_write_batch_kernel` | 1 block / トークン |
| Attention | `flash_attn_prefill_gqa_kernel` | `(token, head)`、因果 `npos = t + 1` |
| 残差・SwiGLU | `add_batch_kernel`、`swiglu_batch_kernel` | 要素並列 |

LM head は **末尾トークン**（`n_tokens - 1`）のみ。Prefill 後の Decode は **`pos = n_prompt`** から、填充済み **`kc`/`vc`** を共有参照する。

進捗表示・スループット計測の**出力形式**は **「進捗表示とスループット計測（全バリアント共通）」** と同様（**`generate()`** 内の static ヘルパ）。prefill プログレスバーの更新タイミングは上記のとおり **`gpu-cuda` / `gpu-rocm` で CPU 3 バリアントと異なる**。

## 実行時の挙動（gpu-cuda-nvfp4）

**`gpu-cuda-nvfp4`** は **`gpu-cuda`** と同一の GGUF・**`chat_encode`**・prefill/decode 分割・Attention カーネル。**線形層のみ常に NVFP4 経路**（**`#ifdef` なし**）。

1. **起動時**: **`gpu-cuda`** と同様に Q1_0 重みを mmap → VRAM。**`fp4_bonsai_init`** で線形層を **NVFP4 キャッシュ**化、**`fp4_gemm_prealloc`** と **128³ サニティ GEMM**（**`fp4_gemm_run_cached`**）。stderr に `GPU: FP4 Tensor Core path enabled`。
2. **Prefill / Decode**: **`gpu-cuda`** と同型。線形層は **`fp4_bonsai_mm`** → **`fp4_gemm_run_cached`**（各 GEMM 前 **`fp4_gemm_init(M,N,K)`**。活性化は F32 → BF16 パッド（**`m >= M_act` は 0**）→ NVFP4）。
3. **VRAM 重み（線形層）**: Q1_0 blob に加え **NVFP4 + block scale の GPU キャッシュ**（起動時変換）。Embedding は Q1_0 のまま。

#### NVFP4 + CUTLASS（`gpu-cuda-nvfp4`）

**NVFP4（要約）:** Blackwell Tensor Core 向け **4 bit 浮動小数**。要素値は **E2M1**（4 bit）、**16 要素**（`SF_VEC_SIZE`）ごとに **UE4M3** スケール（8 bit）を 1 つ。おおよそ `実数 ≈ scale × E2M1`。

**E2M1（4 bit / 要素）** — 符号 1 + 指数 2 + 仮数 1（CUTLASS `float_e2m1_t`）。表現できる**絶対値は 8 段階のみ**（`fp4_gemm.cu` のルックアップ = NVIDIA 公式）:

| code (3 bit) | 絶対値 |
|:---:|:---:|
| 0〜7 | 0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0 |

bit 3 = 符号（0=非負、1=負）。**0.75 は E2M1 の離散値ではない**（丸め**境界** — ModelOpt の `e2m1_bounds` は `0.25, 0.75, 1.25, …` など中点）。2 要素 / byte（下位ニブル = 偶数 index）。

**UE4M3（8 bit / 16 要素）** — 符号なし、exp 4 bit（バイアス 7）+ mant 3 bit。ブロック max から `scale` を決め、正規化後に E2M1 へ量子化。

**実装分担** — **`main.c` は NVFP4 を直接呼ばない。**

| ファイル | 役割 |
|----------|------|
| `fp4_gemm.cu` | CUTLASS **SM120 block-scaled NVFP4 GEMM**（**v4.5.1**）。起動時 **`fp4_gemm_prealloc`**。**`fp4_gemm_run_cached`** は workspace 不足時のみ再確保（縮小しない）、`cudaDeviceSynchronize` 後に **`initialize`**。`beta=0` 時は epilogue の `C` を `nullptr` |
| `fp4_bonsai.cu` | Q1_0 → **起動時** NVFP4 重みキャッシュ、推論時 F32 ↔ BF16 ↔ GEMM。**`f32_to_bf16_pad_kernel`** は **`M_act`/`M_pad` 分離** |
| `kernels.cu` | 常に **`gpu_mm_fp4`** / **`fp4_bonsai_mm`** |

**対象レイヤ**: **`wq`〜`down`** と **`output`（LM head）** のみ。**`M`/`N`/`K` は 128 倍数**。GGUF は Q1_0 のまま、**BF16 復元 → NVFP4 キャッシュ**は起動時。ビット図・ビルド手順の全文は **README 付録「NVFP4 + CUTLASS」**（未更新時は本書を正とする）を参照。

## 実行時の挙動（gpu-rocm）

**`gpu-rocm`** は **`gpu-cuda` の Q1_0 経路**と実質同型（**NVFP4 なし**、線形層は **Q8_0 量子化 + Q1_0 GEMV** のみ）。

1. **起動時**: **`gpu_print_device_info()`**（**`hipGetDeviceProperties`**）で GPU 名・compute 版・VRAM を表示。**`gpu_model_create`** 内で **`flash_attn_init_once()`** が decode/prefill カーネルに **`hipFuncAttributePreferredSharedMemoryCarveout=100`** を設定。GGUF を mmap したうえで **Q1_0 重み blob をそのまま** **`hipMalloc`** / **`hipMemcpy(H2D)`** で VRAM 常駐（**`main.c` に CPU dequant 経路はない**）。KV・活性・prefill 用 **`x_batch` 等**（**`batch_cap = max_seq`**）も事前確保（**`gpu-cuda`** の VRAM 表と同構成）。
2. **Prefill / Decode**: **`gpu_forward_prefill`** → **`gpu_forward`**（**`gpu-cuda`** 節のカーネル対応表と同趣旨。API は **`hip`**）。
3. **線形層**: **`quantize_q8_0_*` + `mm_q1_0_*`** HIP カーネルのみ（FP4 分岐なし）。
4. **Attention**: **`FA_BR`** タイル（Makefile 既定 **32**）、K/V shared staging、online softmax、GQA。

**`gpu.h`** は **`gpu-cuda` をベースに `gpu_get_device_desc` を追加**（`extern "C"` API）。サンプリング・**`chat_encode`**・進捗表示（**`Prefill complete` / `Decode complete` / `--- throughput ---`**）は **`gpu-cuda`** と同様。終了時 **`BENCH_LOG_FILE`**（既定 **`/tmp/benchmark.log`**）へ key=value ログ（プロンプト・生成全文含む）。**`generate()`** の tok/s は **重み H2D 後**の推論区間のみ。

## 実行時の挙動（gpu-rocm-wmma）

**`gpu-rocm-wmma`** は **`gpu-rocm`** とホスト・Decode・線形層が同一。**Prefill Attention** のみ差し替え。

1. **Prefill Attention**: **`flash_attn_prefill_wmma_gqa_kernel`** — ヘッド × **16 行 query ブロック**（**`PREFILL_QB=WMMA_M=16`**）× **`FA_BR`** キータイル。各タイルで **Q/K を `__half` 転置バッファ**へ協調ロード → **`wmma_gemm_qk`**（rocWMMA **16×16×16**、f16 入力・f32 累積）→ online softmax → **F32 スカラー PV** 累積。**`blockDim=32`**（gfx1201 wave32）。
2. **PV を WMMA 化しない理由**: **`scores`** が **`FA_BR` 未満の有効列**を含む場合、rocWMMA の col-major **B** 要件と整合しない（詳細は **`kernels.hip` 先頭コメント**）。
3. **LDS**: q/k/v タイル + WMMA 転置バッファのため **`FA_BR=32` 推奨**（Makefile 既定）。**`FA_BR=64`** は LDS 上限に近づく。
4. **性能特性**: WMMA QK^T の利得と **query 16 行ブロックによる grid 並列度低下**のバランスは **GPU 依存**。**gfx1201** 実測（上記 WMMA 表）では Prefill tok/s が **`gpu-rocm`** より低い例がある。**gfx1100** では decode / total が **`gpu-rocm`** を上回る例もある。短プロンプトでは Prefill 並列度低下の影響が相対的に大きい。

## コマンドラインオプション

| オプション | 説明 | デフォルト（実装参照） |
|-----------|------|------------------------|
| `-p` | プロンプト | `Hello` |
| `-n` | 最大生成トークン数 | 実装既定値に従う |
| `-t` | Temperature | 実装既定値に従う |
| `-k` | Top-p | 実装既定値に従う |
| `-s` | 乱数シード | 実装既定値に従う |
| `-l` | 最大シーケンス長 | 実装既定値に従う |

## アーキテクチャ（`cpu/main.c` / `gpu-cuda/` / `gpu-cuda-nvfp4/` / `gpu-rocm/`）

**CPU 3 バリアント**は **1 ファイルの `main.c`** に実装を集約する（バリアント間でソースを `#include` しない）。**`gpu-cuda`** は **`main.c`（ホスト）** + **`kernels.cu`（デバイス、Q1_0 のみ）** + **`gpu.h`（API）**。**`gpu-cuda-nvfp4`** は同構成に **`fp4_bonsai.cu`** / **`fp4_gemm.cu`** + **CUTLASS v4.5.1** を追加し、**`kernels.cu`** は線形層を常に NVFP4 経由。**`gpu-rocm`** は **`main.c` + `kernels.hip` + `gpu.h`（`gpu_get_device_desc` 追加）** で **HIP** に置換した付録。**`cpu-omp/main.c`** はデータ構造・推論フェーズは **`cpu`** と同じで、ホットパスに **OpenMP** を挟んだ派生として読む。**`cpu-blas/main.c`** はさらに **OpenBLAS** と **Q1_0×Q8_0 SIMD 内積**でホットパスを置き換えた派生。**`gpu-cuda/main.c`** / **`gpu-cuda-nvfp4/main.c`** / **`gpu-rocm/main.c`** / **`gpu-rocm-wmma/main.c`** は **`cpu-blas`** から CPU forward を除き **`gpu_forward`** 呼び出しに置換した派生として読む。**`gpu-rocm-wmma/kernels.hip`** は **`gpu-rocm/kernels.hip`** の Prefill Attention カーネルのみ rocWMMA 版に差し替え。

### レイヤー構成（概略）

1. **GGUF / GGML dtype**: **CPU 3 バリアント**は **`ggml_dtype` の最小 subset**（**`DT_F32`**・**`DT_Q1_0`**。**`cpu-blas`** は活性化用に **`DT_Q8_0`** / **`BlockQ8_0`** も定義）と **`BlockQ1_0`** のみ。**Q4_K / IQ2_S / IQ3_S 等**の汎用逆量子化は **2026-05-27 に CPU 3 バリアントから削除**（**`Bonsai-8B-Q1_0.gguf` 専用**）。**`gpu-cuda/main.c`** / **`gpu-cuda-nvfp4/main.c`** / **`gpu-rocm/main.c`** はメタデータ用 **`ggml_dtype`**（**`DT_F32`**・**`DT_F16`**・**`DT_Q1_0`**）のみ（**CPU dequant ブロック型・IQ グリッド表は持たない**）。GPU 側量子化レイアウトの実体は **`gpu.h`** の **`GpuBlockQ1_0`** / **`GpuBlockQ8_0`** と **`kernels.hip`** / **`kernels.cu`**（**`gpu-cuda-nvfp4`** の Q1_0→NVFP4 変換は **`fp4_bonsai.cu`**）。
2. **モデル構造体**: `Config`、`TensorInfo`、トークナイザ、重み参照、`State`（KV・中間バッファ）。
3. **ロード**: mmap、metadata（`qwen3.*`）、tensor テーブル、tokenizer。
4. **推論**: CPU 3 バリアントは 1 トークンずつ forward（prefill も teacher forcing で逐次）。**`gpu-cuda`** / **`gpu-rocm`** は prefill を **`gpu_forward_prefill`** でバッチ、decode を **`gpu_forward`** で逐次。生成区間はサンプリング。

### GGUF メタデータ（抜粋）

実装が読む主なキーは **`qwen3.` + サフィックス**（ソース内 `gguf_key_dense_meta`）と `general.alignment`、`tokenizer.ggml.*` である。例:

- `qwen3.embedding_length` → `dim`
- `qwen3.feed_forward_length` → `hidden_dim`
- `qwen3.block_count` → `n_layers`
- `qwen3.attention.head_count` / `head_count_kv` / `key_length`
- `qwen3.attention.layer_norm_rms_epsilon`
- `qwen3.rope.freq_base` および **YARN 系 rope scaling** 各キー

（具体キー一覧は `parse_gguf` を参照。）

### 量子化と行列積

**対象 GGUF**: 既定 **`Bonsai-8B-Q1_0.gguf`**。線形重みは **Q1_0**、norm 重みは **F32**。**CPU 3 バリアント**は他量子化形式の GGUF を**未対応**（ロード時 **`expect_q1_0_weight`** または **`mm`** でエラー終了）。

**Q1_0** は **QK1_0=128** 要素を単位とする。

- **`cpu`**: **`dot_q1_0_row`** / **`mm_q1_0_rows`** で融合行内積（`ggml-quants.c` の dequantize + 内積と同等、中間 FP32 ブロックなし）。Embedding は **`dequant_q1_0_blocks`**。
- **`cpu-omp`**: 同上。行ループを **`#pragma omp parallel for`** で並列化。
- **`cpu-blas`**: **`quantize_row_q8_0`** + **`vec_dot_q1_0_q8_0`**（llama.cpp **`ggml_vec_dot_q1_0_q8_0`** 準拠。AVX2 で SIMD、非 AVX2 は generic 参照）。**`State`** に **`BlockQ8_0 *q8`** を確保（`max(dim, hidden_dim) / QK8_0` ブロック）。
- **`gpu-cuda`**: **Q8_0 活性化 + `vec_dot_q1_0_q8_0` CUDA カーネル**（単トークン + **`*_batch`** 版）。**RTX 50 系**は **`sm_120` / `sm_120a` ネイティブ**（**`make run`** または **`make build` の GPU 自動検出）。**PTX `compute_86` JIT は Blackwell 非対応**。
- **`gpu-cuda-nvfp4`**: 線形層は **NVFP4 E2M1（8 段階）+ UE4M3 ブロックスケール + CUTLASS v4.5.1 GEMM**（**`fp4_bonsai_mm`**）。Attention は **`gpu-cuda`** と同型（**`flash_attn_*`**、**`FA_BR`**、**`flash_attn_init_once`**）。
- **`gpu-rocm`**: **Q8_0 活性化 + Q1_0 HIP カーネル**（単トークン + **`*_batch`** 版）。Attention は **`gpu-cuda`** と同型（**`FA_BR`** 既定 **32**）。
- **`gpu-rocm-wmma`**: 線形層・Decode は **`gpu-rocm`** 同一。**Prefill Attention QK^T** のみ **rocWMMA 16×16×16**（**PV** は F32 スカラー）。

### RoPE

llama.cpp 系の NeoX 半分ペア配置に合わせ、YARN 等のメタがあれば `rope_scaling` 関連フィールドで処理。**`finalize_rope_hparams`**（全バリアント共通）では **llama-context.cpp** と同様、**`yarn_ext_factor != 0`** のとき **`yarn_attn_factor`** を確定する（詳細はソース内 `rope_apply` / `finalize_rope_hparams`）。

### トークナイザ・チャット

GPT-2 系 BPE と特殊トークン。**全バリアント共通**の **`chat_encode`** は GGUF **`tokenizer.chat_template`**（Qwen3 / Bonsai）に合わせ、**user** 1 ターン + assistant 生成プレフィックス（空 think ブロック付きリテラル）のみを組み立てる。既定 system 文は挿入しない。

### 生成ループとサンプリング

プロンプト区間（**prefill**）は CPU 3 バリアントで teacher forcing による逐次 forward、**`gpu-cuda`** / **`gpu-rocm`** では **`n_prompt > 1` のときバッチ prefill**（**`gpu_forward_prefill`**）。続く **decode** 区間は logits からサンプリングし 1 トークンずつ forward。温度・top-p 等の分岐は CPU 上で logits に対して実施。乱数は実装の xorshift 系 state を使用。進捗・区間別 tok/s は上記「進捗表示とスループット計測」を参照。

## モデル参照

- 既定ファイル名: **`bonsai-8b/Makefile` の `MODEL`**（既定 **`Bonsai-8B-Q1_0.gguf`**）。
- 前提: **`wget`**（初回ダウンロード時のみ。利用者向けの詳細手順・クイックスタート・トラブルシュートは **`README.md` / `README.en.md`** を参照）。
- 取得 URL: **`bonsai-8b/gguf.txt`**（Hugging Face の `blob/main` URL。`make model` は **`$(MODEL)` が無いとき** `resolve/main` に置換して `wget` する）。
- チェックサム: **`bonsai-8b/$(MODEL).sha256sum`**（既定 **`Bonsai-8B-Q1_0.gguf.sha256sum`**）。`make model` は常に **`sha256sum --check`** で照合する（**既存ファイルは再ダウンロードしない**）。成功・失敗いずれも**英語**のメッセージを表示する。失敗時は破損ファイルを削除する。
- 手動取得: README の手順どおり `gguf.txt` から URL を変換して `wget` し、**`sha256sum --check $(MODEL).sha256sum`** で検証してもよい。

## 制約・既知の制限

- **8B を CPU で動かすため重い**場合がある。単スレッド **`cpu`** は参考実装・検証向け（上記 CPU 参考計測 decode **0.24 tok/s**）。**`cpu-omp`** は decode **4.94 tok/s** 程度。実用的な CPU 試行は **`cpu-blas`**（参考 decode **30.79 tok/s**）を推奨。
- **CPU 3 バリアント**および **`gpu-cuda`** / **`gpu-rocm`** / **`gpu-rocm-wmma`** は **`Bonsai-8B-Q1_0.gguf` 専用**（線形重み **Q1_0** + norm **F32**）。**Q4_K / IQ2_S / IQ3_S 等**の汎用逆量子化パスは **2026-05-27（CPU）** / **2026-05-28（`gpu-cuda/main.c`）** に削除。他 GGUF は **`expect_q1_0`** / **`expect_q1_0_weight`** / **`mm`** でエラー終了。
- **`cpu-blas`** は **OpenBLAS**（`libopenblas-dev` 等）が必要。**AVX2** 非対応 CPU では Q1_0 内積が generic 参照実装にフォールバックする（Makefile は **AVX2 不可なら `-mavx` または `-march=x86-64`**）。**`-ffast-math`** と **FMA 対応時の `-mfma`** 使用のため、環境によっては **`cpu`** / **`cpu-omp`** と数値がわずかに異なり得る。長プロンプト **`make log.push`** では **`avx2+fma`** と **`avx`** で total tok/s が大きく異なりうる（上記 CPU 長プロンプト表）。旧 **`-march=native`** 固定は廃止（クロスビルド・非 x86 ホストでは **`ARCH_FLAGS`** を明示指定）。**`make log.push`** は **`cpu-blas/Makefile` を書き換える**（コミット前に差分確認）。
- **`gpu-cuda`**（付録・NVIDIA・Q1_0）: **CUDA Toolkit**・NVIDIA ドライバ・GPU 実機が必要。本書「実行時の挙動（gpu-cuda）」参照。**将来別リポジトリ移行予定**。RTX 5090 参考（2026-05-28 **`make run`**、**`sm_120a` Q1_0**）: prefill **~312 tok/s**、decode **~47 tok/s**（`-p "Hello, how are you?"`）。**PTX `compute_86` JIT は RTX 50 系で推論破損**。**`make run`** は **CUDA 13**・**`sm_120a`**（**`make blackwell`**）。**`make build`** は **`nvidia-smi` で `CUDA_GENCODE` 自動選択**。Blackwell は **`FA_BR=32`**。
- **`gpu-cuda-nvfp4`**（付録・NVIDIA・NVFP4）: 上記に加え **CUTLASS v4.5.1**（**`make cutlass`**）。**`bonsai-gpu-cuda-nvfp4`**。RTX 5090 参考: 2026-05-21 短プロンプト prefill **~1365** / decode **~90.4 tok/s**；2026-05-28 上記 Q1_0 プロンプトで prefill **~1767** / decode **~65 tok/s**。旧 CUTLASS v3.9.0 は **`initialize: Error Internal`**。**VRAM**・**`max_seq`**・**`head_dim > FA_HD`（128）** 制約は **`gpu-cuda`** と同趣旨。
- **`gpu-rocm`**（付録・AMD）: **ROCm**・**`hipcc`**・AMD GPU 実機が必要。本書「ビルドと実行（GPU ROCm）」「実行時の挙動（gpu-rocm）」参照。**hipBLAS 不要**。**NVFP4 非対応**。**`GPU_ARCH`** は **`rocminfo`** 自動（未検出時は手動指定）。**`g++` / `libstdc++-dev`** 必須。**VRAM**・**`max_seq`**・**`head_dim > FA_HD`** 制約は **`gpu-cuda`** と同趣旨。参考ベンチマーク: 上記 **GPU ROCm 表**（**gfx1201** / **gfx1100** 等・長プロンプト 130 + 生成 128。**GPU_ARCH** ごとにホストが異なる場合あり）。**`make log.push`** は **`Makefile` を書き換える**（コミット前に差分確認）。
- **`gpu-rocm-wmma`**（付録・AMD・実験）: **`gpu-rocm`** 要件に加え **rocWMMA**（ROCm 同梱ヘッダ）。Prefill Attention QK^T の WMMA 化のみ。**Prefill / decode / total の優劣は GPU 依存**（**gfx1201** では Prefill が **`gpu-rocm`** より低い例、**gfx1100** では total が上回る例 — 本書 **GPU ROCm WMMA 表**）。**`make log.push`** は **`Makefile` を書き換える**（コミット前に差分確認）。
- **画像・マルチモーダル入力は非対応**（テキストデコーダのみ）。
- **コンテキスト長**を大きくすると KV 用メモリが増える。
- 商用水平の性能・公式実装との一致は保証しない。

## 補足：ドキュメント間の役割

- **`README.md` / `README.en.md`**: **入口**。**本文**は CPU 3 バリアント。**付録**は **`gpu-cuda`**・**`gpu-cuda-nvfp4`**・**`gpu-rocm`**・**`gpu-rocm-wmma`**。ベンチマークの**正**は CPU 短プロンプト表を README 本文、CPU 長プロンプト（**`cpu-blas`**）・GPU CUDA（Q1_0 + NVFP4）・GPU ROCm・GPU ROCm WMMA を README 付録と本書。
- **`doc/design.md`（本書）**: **設計・仕様の静的説明**（CPU 3 バリアント + GPU 付録 4 種: **`gpu-cuda`** / **`gpu-cuda-nvfp4`** / **`gpu-rocm`** / **`gpu-rocm-wmma`**）。ベンチマークの**正**は上記 README と本書（各 Makefile の **`BENCH_LOG`** に写し可）。
- **`doc/ChangeLog`**: 履歴。

実装の**正**は **`bonsai-8b/cpu/main.c`**。**`cpu-omp`** / **`cpu-blas`** は CPU 派生。**`gpu-cuda`** / **`gpu-cuda-nvfp4`** / **`gpu-rocm`** / **`gpu-rocm-wmma`** は付録（CUDA は README 付録も参照）。**Prefill** **`gpu_forward_prefill`**、**Decode** **`gpu_forward`**。浮動小数の結合順などで数値差が出うる。

## 補足：`design.md` 更新時のチェックリスト

1. **`bonsai-8b/Makefile`** / **`cpu/Makefile`** / **`cpu-omp/Makefile`** / **`cpu-blas/Makefile`** / **`gpu-cuda/Makefile`** / **`gpu-cuda-nvfp4/Makefile`** / **`gpu-rocm/Makefile`** / **`gpu-rocm-wmma/Makefile`** と矛盾がないか。
2. **`README.md` と `README.en.md`** の**本文（CPU）**および**付録（`gpu-cuda` / `gpu-cuda-nvfp4` / `gpu-rocm` / `gpu-rocm-wmma`）**と整合するか。
3. 仕様変更は **`doc/ChangeLog`** に記録する。
