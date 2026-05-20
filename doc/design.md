# 設計仕様書

> **注意**: 本ドキュメントは設計仕様書です。変更履歴は `doc/ChangeLog` を参照。以下は**現在の**システム設計である。

## 概要

### リポジトリの目的とスコープ

本リポジトリは **[PrismML](https://prismml.com/) の 1-bit Bonsai 8B** を **GGUF**（既定: **`Bonsai-8B-Q1_0.gguf`**、**Q1_0** 量子化）から、**C ソース**で **テキスト生成推論**する実装である。**プロジェクトの主眼**は **単一 `main.c`・依存最小**の **CPU 3 バリアント**（基準 **`bonsai-8b/cpu/main.c`**、並列 **`cpu-omp`**、実用 **`cpu-blas`**）である。

**PyTorch・TensorFlow・JAX・ONNX Runtime 等の ML ユーザランドにはリンクしない。** 基準経路のコアは **標準 C と `libm`** のみ。**ROCm/HIP・AMD XDNA2 NPU** 向けの旧 `main.c` は本リポジトリから**削除済み**。

**CPU** については、**参照実装**として **`bonsai-8b/cpu/main.c`（単スレッド）** を保守し、**同ロジックを OpenMP で並列化した `bonsai-8b/cpu-omp/main.c`**（**`bonsai-cpu-omp`**）を検証用に、**OpenMP + OpenBLAS と Q1_0×Q8_0 SIMD 内積**（llama.cpp **`ggml_vec_dot_q1_0_q8_0`** 準拠）を用いた **`bonsai-8b/cpu-blas/main.c`**（**`bonsai-cpu-blas`**）を CPU 実用スループット向けに提供する。

**`bonsai-8b/gpu-cuda/`**（**`main.c`** + **`kernels.cu`** → **`bonsai-gpu-cuda`**）は **CUDA Runtime のみ（`libcudart`）** の **付録実装**である。README 上は **プロジェクト目的（単一 C・依存最小）の外**と明記し、**将来別リポジトリへ移す予定**。技術メモ・ビルド手順・GPU ベンチマークは README 末尾の **「NVIDIA CUDA 実装（`gpu-cuda`）について」** 付録に、仕様の静的説明は本書 **`doc/design.md`** に記載する。実装内容は **`cpu-blas`** と同一 GGUF・CLI・**`chat_encode`**・RoPE を前提に VRAM 常駐 forward。**Prefill** は **`gpu_forward_prefill`**（バッチ並列）、**Decode** は **`gpu_forward`**（1 トークンずつ）。

目的は、推論経路（GGUF 読み取り・量子化復元・Transformer forward・サンプリング）を **C の明示的なコードパス**として追い、改変しやすくすることである。学習・商用 SLA・公式実装との数値一致はスコープ外。

#### ライブラリ非依存の意義

- **理解可能性**: バッファ配置・演算順・量子化レイアウトをソースで追える。
- **依存の単純化**: コンパイラと最小実行環境で再現できる。
- **参照実装**: Bonsai 8B（GGUF）デコーダの最小例として使える。

利用者向けの手順は **`README.md`** / **`README.en.md`** を参照（**本文は CPU 3 バリアント**。**`gpu-cuda`** は README 末尾付録）。

### 実装バリアント（現状）

| ソース | 実行環境 | 概要 |
|--------|----------|------|
| `bonsai-8b/cpu/main.c` | **CPU、単スレッド** | GGUF mmap。**Q1_0** は融合行内積（**`dot_q1_0_row`**、中間 FP32 展開なし）。他量子化型はブロック単位で部分復元して GEMV。Norm 等は F32。`libm` のみ。 |
| `bonsai-8b/cpu-omp/main.c` | **CPU、OpenMP マルチスレッド** | **`cpu`** と同一 GGUF・CLI・**`chat_encode`**・RoPE。**Q1_0** は融合行内積（**`mm_q1_0_rows`** を OpenMP 行並列）。他量子化型はブロック dequant + OpenMP。Attention・SwiGLU 等も **OpenMP**。 |
| `bonsai-8b/cpu-blas/main.c` | **CPU、OpenMP + OpenBLAS** | **`cpu-omp`** と同一 GGUF・CLI・**`chat_encode`**・RoPE。**Q1_0** は活性化を **Q8_0** 化して **`vec_dot_q1_0_q8_0`**（AVX2 時は ggml-cpu x86 準拠 SIMD、**`-mfma`**）。Attention はヘッドあたり **2 回の `cblas_sgemv`**。F32 行は OpenMP 行帯 + serial **`sgemv`**。起動時 **`openblas_set_num_threads(1)`**。 |
| `bonsai-8b/gpu-cuda/main.c` + **`kernels.cu`** | **NVIDIA GPU、CUDA（付録）** | 上記 CPU 3 バリアントと**同趣旨**だが **CUDA 依存**。README 付録参照。 |

メタデータの照合は **GGUF 内の `qwen3.*` プレフィックス**（実装上のバイト列。`embedding_length` 等）と tokenizer 系キーで行う。実装コメントは「dense デコーダ」「Bonsai」と整合する。

### 参考ベンチマーク（開発環境）

環境依存の参考値。CPU・GPU・メモリ・ビルドフラグ・GGUF が RAM/VRAM に載っているかで大きく変わる。**CPU 表**の手順全文は **`README.md`** / **`README.en.md`** 本文の「参考ベンチマーク」。**GPU 表**は本書で管理し、数値の写しは README 付録 **「NVIDIA CUDA 実装（`gpu-cuda`）について」** にも載せる。

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
| `bonsai-cpu-blas` | 0.5 s | 30.79 tok/s | `-O3 -fopenmp -march=native -ffast-math -mfma`、OpenBLAS 1 スレッド |

この条件下では **`cpu-blas` が `cpu-omp` の約 6 倍**、**`cpu` の約 128 倍**の decode スループット（`cpu-omp` は `cpu` の約 21 倍）。生成テキスト（`Hello! I'm Bonsai, an AI assistant developed by PrismML.`）は 3 バイナリで一致。`cpu-blas` の Q1_0 は **Q8_0 活性化 + AVX2 内積**（llama.cpp 準拠）。

#### GPU CUDA（2026-05-21 計測）

| 項目 | 値 |
|------|-----|
| GPU | NVIDIA GeForce RTX 5090（31 GiB VRAM） |
| OS | Linux |
| モデル | `Bonsai-8B-Q1_0.gguf`（起動時 VRAM アップロード） |
| コマンド | `./gpu-cuda/bonsai-gpu-cuda Bonsai-8B-Q1_0.gguf -p "Hello" -n 16 -t 0` |
| ワークロード | 上記 CPU 表と同じ（prefill 18 + decode 16） |
| 表の指標（prefill） | stderr の `Prefill complete` 行（**`gpu_forward_prefill`** バッチ、`-use_fast_math`） |
| 表の指標（decode） | decode 時間・tok/s（stderr の `Decode complete` 行） |
| ビルド | `gpu-cuda/Makefile` 既定（**`-gencode arch=compute_86,code=compute_86`** PTX、ドライバ JIT） |
| Attention | **Flash Attention + K/V shared staging**（decode: **`flash_attn_gqa_kernel`** `<<<n_heads, FA_HD>>>` / prefill: **`flash_attn_prefill_gqa_kernel`** `<<<n_tokens×n_heads, FA_HD>>>`、因果マスク、shared ≈ 65 KB、carveout=100） |

| バイナリ | prefill tok/s | decode 時間 | decode スループット |
|----------|-------------:|----------:|-----------------:|
| `bonsai-gpu-cuda` | **~294** | 0.32 s | **50.24 tok/s** |

同一プロンプト・`-t 0` で **`cpu-blas` と同じ生成テキスト**（`Hello! I'm Bonsai, an AI assistant developed by PrismML.`）。この GPU 上では decode が **`cpu-blas`（5950X 参考 30.79 tok/s）の約 1.6 倍**程度。

## ディレクトリとファイル構成

| パス | 役割 |
|------|------|
| `README.md` / `README.en.md` | **入口（日／英）**。**本文**: CPU 3 バリアントのビルド・実行・ベンチマーク・トラブルシュート（`make model` 含む）。**付録**（`---` 以降）: **`gpu-cuda`** の位置づけ・技術メモ・ビルド／実行・GPU ベンチマーク（初見は CPU 本文のみで可）。 |
| `bonsai-8b/Makefile` | **`model`**（GGUF ダウンロード＋SHA256 検証）、**`build.cpu`** / **`run.cpu`**、**`build.cpu-blas`** / **`run.cpu-blas`**、**`build.gpu-cuda`** / **`run.gpu-cuda`**、**`blackwell`** / **`build.blackwell`** / **`run.blackwell`**（**`gpu-cuda`** へ委譲）、**`clean`**（`cpu`・`cpu-blas`・`gpu-cuda` を委譲）。**`cpu-omp` は未集約**（サブディレクトリの Makefile を直接使用）。 |
| `bonsai-8b/cpu/Makefile` | `bonsai-cpu` の生成。`MODEL` 既定は **`Bonsai-8B-Q1_0.gguf`**。 |
| `bonsai-8b/cpu-omp/Makefile` | **`bonsai-cpu-omp`** の生成（`-fopenmp`）。`MODEL` 既定は **`../Bonsai-8B-Q1_0.gguf`**。 |
| `bonsai-8b/cpu-blas/Makefile` | **`bonsai-cpu-blas`** の生成（`-fopenmp`、OpenBLAS、`-march=native -funroll-loops -ffast-math -mfma` 等）。`pkg-config openblas` があれば `-I`/`-L` を自動付与。 |
| `bonsai-8b/cpu/main.c` | CPU 単スレッド推論の**参照ソース**（アルゴリズムの正として追う）。**単一 `main.c`**（標準 C + `libm` のみ）。 |
| `bonsai-8b/cpu-omp/main.c` | **`cpu/main.c` をベースに OpenMP を付与した派生**（挙動確認はまず `cpu` を正とする）。**単一 `main.c`**（+ OpenMP）。 |
| `bonsai-8b/cpu-blas/main.c` | **`cpu-omp` をベースに OpenBLAS・Q1_0×Q8_0 SIMD 内積・Attention `sgemv` 集約を付与した派生**（スループット試行の推奨経路）。**単一 `main.c`**（+ OpenMP + OpenBLAS）。 |
| `bonsai-8b/gpu-cuda/main.c` | **`cpu-blas` をベースに CPU 演算を GPU API 呼び出しへ置換したホスト側**（GGUF 読み込み・トークナイザ・サンプリングは CPU）。**`gpu.h`** の C API 経由で **`kernels.cu`** を呼ぶ。 |
| `bonsai-8b/gpu-cuda/kernels.cu` | **CUDA カーネルと VRAM 管理**（Q1_0×Q8_0 GEMV、単トークン／**バッチ** Norm・RoPE・SwiGLU、**Flash Attention** decode/prefill、**`gpu_forward`** / **`gpu_forward_prefill`**、**`flash_attn_init_once`** 等）。 |
| `bonsai-8b/gpu-cuda/gpu.h` | **`GpuModel`** / **`gpu_model_create`** / **`gpu_forward`** / **`gpu_forward_prefill`** / **`gpu_copy_logits`** 等の C API（**`extern "C"`**）。 |
| `bonsai-8b/gpu-cuda/Makefile` | **`bonsai-gpu-cuda`** の生成（**`main.c`** → **`cc`**、**`kernels.cu`** → **`nvcc`**、リンクも **`nvcc`**。**`-lcudart`**。既定 **`CUDA_GENCODE=arch=compute_86,code=compute_86`** PTX、**`-use_fast_math`**）。**`blackwell`** / **`run.blackwell`**: apt **CUDA 11** 削除 → NVIDIA 公式 **`cuda-toolkit-13`** インストール → **`BLACKWELL_GENCODE=arch=compute_120,code=sm_120`** でビルド（要 **sudo**）。 |
| `bonsai-8b/gguf.txt` | 既定 GGUF の Hugging Face URL（`blob/main` 形式）。 |
| `bonsai-8b/Bonsai-8B-Q1_0.gguf.sha256sum` | 既定 GGUF の SHA256 チェックサム（`make model` の検証に使用）。 |
| `doc/design.md` | 本書。 |
| `doc/ChangeLog` | 変更履歴。 |
| `.gitignore` | ビルド生成物（**`bonsai-8b/cpu/bonsai-cpu`**、**`bonsai-8b/cpu-omp/bonsai-cpu-omp`**、**`bonsai-8b/cpu-blas/bonsai-cpu-blas`**、**`bonsai-8b/gpu-cuda/bonsai-gpu-cuda`**、**`bonsai-8b/gpu-cuda/*.o`** 等）、`*.gguf` 等。 |

### Make ターゲット（`bonsai-8b/Makefile`）

| ターゲット | 出力 | ソース |
|------------|------|--------|
| `model` | `$(MODEL)`（既定 **`Bonsai-8B-Q1_0.gguf`**） | `gguf.txt` の URL を `resolve/main` に変換して `wget` し、**`$(MODEL).sha256sum`** で `sha256sum --check` |
| `build.cpu` / `run.cpu` | `cpu/bonsai-cpu` | `cpu/main.c` |
| `build.cpu-blas` / `run.cpu-blas` | `cpu-blas/bonsai-cpu-blas` | `cpu-blas/main.c` |
| `build.gpu-cuda` / `run.gpu-cuda` | `gpu-cuda/bonsai-gpu-cuda` | `gpu-cuda/main.c` + `gpu-cuda/kernels.cu` |
| `blackwell` / `build.blackwell` / `run.blackwell` | `gpu-cuda/bonsai-gpu-cuda`（Blackwell 向け **`sm_120`**） | `gpu-cuda/Makefile` の **`blackwell`**（CUDA 13 セットアップ＋ビルド） |

**`cpu-omp/`** はルート Makefile からは呼ばず、次のようにサブディレクトリでビルドする。

| 場所 | ターゲット | 出力 | ソース |
|------|------------|------|--------|
| `bonsai-8b/cpu-omp/Makefile` | `build` / `run` / `clean` | `cpu-omp/bonsai-cpu-omp` | `cpu-omp/main.c` |
| `bonsai-8b/cpu-blas/Makefile` | `build` / `run` / `clean` | `cpu-blas/bonsai-cpu-blas` | `cpu-blas/main.c` |
| `bonsai-8b/gpu-cuda/Makefile` | `build` / `run` / `clean` / **`blackwell`** / **`run.blackwell`** | `gpu-cuda/bonsai-gpu-cuda` | `gpu-cuda/main.c` + `kernels.cu` |

**`bonsai-8b/Makefile`** 用変数:

| 変数 | 意味 | 既定例 |
|------|------|--------|
| `MODEL` | GGUF パス | `Bonsai-8B-Q1_0.gguf` |
| `PROMPT` | 実行用プロンプト文字列 | `Hello, how are you?` |

**`bonsai-8b/cpu-omp/Makefile`** にも `MODEL` / `PROMPT` があり、`MODEL` の既定は **`../Bonsai-8B-Q1_0.gguf`**（`cpu-omp` からの相対パス）。

```bash
cd bonsai-8b
make model
make build.cpu
make run.cpu PROMPT="試す文" MODEL=./Bonsai-8B-Q1_0.gguf
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
cd bonsai-8b
make build.cpu-blas
./cpu-blas/bonsai-cpu-blas Bonsai-8B-Q1_0.gguf -p "Hello" -n 16
# または cpu-blas/ で make build
```

GPU CUDA 版（NVIDIA）:

```bash
cd bonsai-8b
make build.gpu-cuda
./gpu-cuda/bonsai-gpu-cuda Bonsai-8B-Q1_0.gguf -p "Hello" -n 16
# または gpu-cuda/ で make build
# アーキテクチャ: gpu-cuda/Makefile の CUDA_GENCODE を上書き可能
```

Blackwell GPU 版（RTX 50 系等。CUDA 13 必須）:

```bash
cd bonsai-8b
make blackwell          # apt CUDA 11 削除 → CUDA 13 インストール → sm_120 ビルド（要 sudo）
make run.blackwell      # 上記ビルド後に推論実行
# または gpu-cuda/ で make blackwell / make run.blackwell
```

## ビルドと実行（CPU）

単スレッド:

```bash
cd bonsai-8b
make build.cpu
./cpu/bonsai-cpu Bonsai-8B-Q1_0.gguf -p "Hello" -n 8
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
make build
./bonsai-cpu-blas ../Bonsai-8B-Q1_0.gguf -p "Hello" -n 8
```

## ビルドと実行（GPU CUDA）

```bash
cd bonsai-8b/gpu-cuda
make build
./bonsai-gpu-cuda ../Bonsai-8B-Q1_0.gguf -p "Hello" -n 8
```

**要件**: CUDA Toolkit（**`nvcc`**）、NVIDIA ドライバ、**`libcudart`**。cuBLAS は不要（Attention はカスタム CUDA カーネル）。Debian/Ubuntu の **`apt install nvidia-cuda-toolkit`** は **CUDA 11** 系であり、**Blackwell（compute capability 12.x）** には **CUDA 13 以降**が必要。

**Blackwell（RTX 50 系等）** では **`make blackwell`** を使う（要 **sudo**）。処理内容:

1. apt の **CUDA 11** 系パッケージ（**`nvidia-cuda-toolkit`** 等）を **`apt-get remove --purge`**
2. NVIDIA 公式リポジトリ（**`cuda-keyring`**）を追加し **`cuda-toolkit-13`** をインストール（**`/usr/local/cuda/bin/nvcc`**）
3. **`BLACKWELL_GENCODE=arch=compute_120,code=sm_120`** で **`bonsai-gpu-cuda`** をビルド

対応 OS: Ubuntu 20.04 / 22.04 / 24.04、Debian 12 / 13。ルート **`bonsai-8b/Makefile`** の **`make blackwell`** / **`make run.blackwell`** も **`gpu-cuda/Makefile`** へ委譲する。

## 実行時の挙動（CPU）

3 バリアントは **RoPE**・**`chat_encode`** を共通とする。**Q1_0** の行列積は **`cpu`** / **`cpu-omp`** が融合行内積（**`dot_q1_0_row`**）、**`cpu-blas`** が **Q8_0 活性化 + `vec_dot_q1_0_q8_0`**（llama.cpp 準拠）と異なる。差分は並列化（OpenMP）、Q1_0 カーネル、Attention / F32 行への OpenBLAS 利用（**`cpu-blas`** のみ）である。

重みは **mmap した GGUF** 上の量子化 blob を参照する。**`cpu`** / **`cpu-omp`** の **Q1_0** は **`mm_q1_0_rows`** で符号ビットと FP32 活性の融合行内積（中間 FP32 ブロックなし）。**`cpu-blas`** の **Q1_0** は **`mm_q1_0_rows`** が活性化 **`x`** を一度 **`quantize_row_q8_0`** し、各行を **`vec_dot_q1_0_q8_0`** で計算（**`State.q8`** バッファを使い回し）。**Q1_0 以外**の量子化型は出力行／ブロックごとに部分復元して内積する（全テンソルの float 一括展開はしない）。KV・活性は主に float。サンプリングはホスト上の logits に対して行う。

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

**`gpu-cuda`** の prefill は **`gpu_forward_prefill`** による**一括バッチ**のため、プログレスバーは **0% 表示のあと完了時に一気に 100%** となる（CPU 3 バリアントのトークン逐次更新とは異なる）。prefill 時間・tok/s の定義は上記と同じ（バッチ全体の wall time ÷ **`n_prompt`**）。

**`cpu-omp`** は **`cpu`** と同アルゴリズムで、**`mm_q1_0_rows`** の行ループなどに **OpenMP parallel for** を入れたもの（細部は `cpu-omp/main.c` を参照）。

**`cpu-blas`** は次を追加する（細部は `cpu-blas/main.c` を参照）。

1. **起動時**: `openblas_set_num_threads(1)` — BLAS 内部スレッドと OpenMP のネスト並列を避ける。
2. **Q1_0 GEMV**（`mm_q1_0_rows`）: 活性化を **`quantize_row_q8_0`**（AVX/AVX2 時は SIMD、それ以外は参照実装）で **Q8_0** 化し、各行を **`vec_dot_q1_0_q8_0`** で内積（**`ggml_vec_dot_q1_0_q8_0`** / **`ggml-cpu/arch/x86/quants.c`** 準拠。AVX2 時は **`_mm256_maddubs_epi16`** 等と **`-mfma`**）。`d` 行まとめて同一量子化済み **`q8`** を使い回す。
3. **Attention**: ヘッドごとに K 行列（`(pos+1)×hd`、`lda=kv_dim`）と q の **`cblas_sgemv(NoTrans)`**、続けて V の **`cblas_sgemv(Trans)`** で出力ヘッドを得る（従来の `(pos+1)` 回の `sdot`/`saxpy` を集約）。
4. **F32 行**（`mm_f32`）: OpenMP で行帯を分割し、帯内は serial `sgemv`。
5. **その他量子化型・F16**: `cpu-omp` と同様の汎用パス（OpenMP 行並列 + ブロック dequant）。

## 実行時の挙動（gpu-cuda）

**`gpu-cuda`** は **`cpu-blas`** と同一の GGUF 読み込み・**`chat_encode`**・RoPE パラメータ・CLI を持つ。差分は forward の実行先・メモリ配置・**prefill / decode の経路分割**である。

1. **起動時**: **`gpu_print_device_info()`** で GPU 名等を表示。**`gpu_model_create`** 内で **`flash_attn_init_once()`** が **`flash_attn_gqa_kernel`** と **`flash_attn_prefill_gqa_kernel`** の双方に **`cudaFuncAttributePreferredSharedMemoryCarveout=100`** を設定（shared ≈ 65 KB/ブロック）。GGUF を mmap したうえで重み・norm・KV キャッシュを VRAM へアップロード。加えて **prefill バッチ用**に **`x_batch` / `xb_batch` / `q_batch` …**（いずれも **`batch_cap = max_seq`** トークン分）と **`tokens_dev`**・**`q8_batch`**・**`rope_batch`** を **`cudaMalloc`**。**`GpuModel`**（**`gpu.h`**）がデバイス側ポインタを保持（推論中も mmap は保持し、終了時に **`munmap`**）。
2. **Prefill**（**`generate()`**、`n_prompt > 1`）: プロンプト ID 配列を **`gpu_forward_prefill(gm, tokens, n_tokens)`** に渡し、全プロンプト位置を**一度に** forward。各層で **`emb_q1_0_batch`**・**`rmsnorm_batch`** / **`rmsnorm_head_batch`**・**`rope_neox_batch`**・**`gpu_mm_batch`**（Q8_0 量子化 + Q1_0 GEMV のバッチ版）・**`kv_write_batch`**・**`flash_attn_prefill_gqa_kernel`** `<<<n_tokens × n_heads, FA_HD>>>`（位置 **`t`** は K/V の **`0..t`** のみ参照する因果 Attention）・**`swiglu_batch`** / **`add_batch`** を実行。最終 norm + LM head は**末尾トークン**（**`n_tokens - 1`**）のみ **`rmsnorm_kernel`** + **`gpu_mm`** で logits を得る。**`n_prompt == 1`** のときは従来どおり 1 トークン **`gpu_forward`**。
3. **Decode**: 末尾 logits からサンプリングした次トークンを、位置 **`n_prompt + gen_i`** で **1 トークンずつ `gpu_forward`**（teacher forcing なし）。各ステップで **`flash_attn_gqa_kernel`** `<<<n_heads, FA_HD>>>`（デコード用 Flash Attention + K/V staging）。
4. **Q1_0 GEMV**: 単トークン経路は **`quantize_q8_0_kernel` + `mm_q1_0_kernel`**、prefill バッチ経路は **`quantize_q8_0_batch_kernel` + `mm_q1_0_batch_kernel`**（いずれも **`cpu-blas`** 準拠ロジック）。
5. **Attention（共通）**: シーケンス方向 **`FA_BR=64`** タイル、K/V を **`k_tile` / `v_tile`**（shared）へ staging、**online softmax**（**`fa_sh_reduce_max` / `fa_sh_reduce_sum`**）、**`att` 非物質化**、GQA 対応。
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

進捗表示・スループット計測の**出力形式**は **「進捗表示とスループット計測（全バリアント共通）」** と同様（**`generate()`** 内の static ヘルパ）。prefill プログレスバーの更新タイミングは上記のとおり **`gpu-cuda` のみ異なる**。

## コマンドラインオプション

| オプション | 説明 | デフォルト（実装参照） |
|-----------|------|------------------------|
| `-p` | プロンプト | `Hello` |
| `-n` | 最大生成トークン数 | 実装既定値に従う |
| `-t` | Temperature | 実装既定値に従う |
| `-k` | Top-p | 実装既定値に従う |
| `-s` | 乱数シード | 実装既定値に従う |
| `-l` | 最大シーケンス長 | 実装既定値に従う |

## アーキテクチャ（`cpu/main.c` / `gpu-cuda/`）

**CPU 3 バリアント**は **1 ファイルの `main.c`** に実装を集約する（バリアント間でソースを `#include` しない）。**`gpu-cuda`** は **`main.c`（ホスト）** + **`kernels.cu`（デバイス）** + **`gpu.h`（API）** の 3 ファイル構成。**`cpu-omp/main.c`** はデータ構造・推論フェーズは **`cpu`** と同じで、ホットパスに **OpenMP** を挟んだ派生として読む。**`cpu-blas/main.c`** はさらに **OpenBLAS** と **Q1_0×Q8_0 SIMD 内積**でホットパスを置き換えた派生。**`gpu-cuda/main.c`** は **`cpu-blas`** から CPU forward を除き **`gpu_forward`** 呼び出しに置換した派生として読む。

### レイヤー構成（概略）

1. **GGUF / GGML dtype**: 必要な tensor dtype（**Q1_0** 等）とブロック構造を定義。
2. **モデル構造体**: `Config`、`TensorInfo`、トークナイザ、重み参照、`State`（KV・中間バッファ）。
3. **ロード**: mmap、metadata（`qwen3.*`）、tensor テーブル、tokenizer。
4. **推論**: CPU 3 バリアントは 1 トークンずつ forward（prefill も teacher forcing で逐次）。**`gpu-cuda`** は prefill を **`gpu_forward_prefill`** でバッチ、decode を **`gpu_forward`** で逐次。生成区間はサンプリング。

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

**Q1_0** は **QK1_0=128** 要素を単位とする。

- **`cpu`**: **`dot_q1_0_row`** / **`mm_q1_0_rows`** で融合行内積（`ggml-quants.c` の dequantize + 内積と同等、中間 FP32 ブロックなし）。
- **`cpu-omp`**: 同上。行ループを **`#pragma omp parallel for`** で並列化。
- **`cpu-blas`**: **`quantize_row_q8_0`** + **`vec_dot_q1_0_q8_0`**（llama.cpp **`ggml_vec_dot_q1_0_q8_0`** 準拠。AVX2 で SIMD、非 AVX2 は generic 参照）。**`State`** に **`BlockQ8_0 *q8`** を確保（`max(dim, hidden_dim) / QK8_0` ブロック）。
- **`gpu-cuda`**: デバイス上で **Q8_0 活性化 + `vec_dot_q1_0_q8_0` CUDA カーネル**（単トークン + **`*_batch`** 版）。Attention は decode **`flash_attn_gqa_kernel`** / prefill **`flash_attn_prefill_gqa_kernel`**（K/V shared staging、起動時 **`flash_attn_init_once`**）。

### RoPE

llama.cpp 系の NeoX 半分ペア配置に合わせ、YARN 等のメタがあれば `rope_scaling` 関連フィールドで処理。**`finalize_rope_hparams`**（全バリアント共通）では **llama-context.cpp** と同様、**`yarn_ext_factor != 0`** のとき **`yarn_attn_factor`** を確定する（詳細はソース内 `rope_apply` / `finalize_rope_hparams`）。

### トークナイザ・チャット

GPT-2 系 BPE と特殊トークン。**全バリアント共通**の **`chat_encode`** は GGUF **`tokenizer.chat_template`**（Qwen3 / Bonsai）に合わせ、**user** 1 ターン + assistant 生成プレフィックス（空 think ブロック付きリテラル）のみを組み立てる。既定 system 文は挿入しない。

### 生成ループとサンプリング

プロンプト区間（**prefill**）は CPU 3 バリアントで teacher forcing による逐次 forward、**`gpu-cuda`** では **`n_prompt > 1` のときバッチ prefill**（**`gpu_forward_prefill`**）。続く **decode** 区間は logits からサンプリングし 1 トークンずつ forward。温度・top-p 等の分岐は CPU 上で logits に対して実施。乱数は実装の xorshift 系 state を使用。進捗・区間別 tok/s は上記「進捗表示とスループット計測」を参照。

## モデル参照

- 既定ファイル名: **`bonsai-8b/Makefile` の `MODEL`**（既定 **`Bonsai-8B-Q1_0.gguf`**）。
- 前提: **`wget`**（`make model`）。利用者向けの詳細手順・クイックスタート・トラブルシュートは **`README.md` / `README.en.md`** を参照。
- 取得 URL: **`bonsai-8b/gguf.txt`**（Hugging Face の `blob/main` URL。`make model` は `resolve/main` に置換して `wget` する）。
- チェックサム: **`bonsai-8b/$(MODEL).sha256sum`**（既定 **`Bonsai-8B-Q1_0.gguf.sha256sum`**）。`make model` はダウンロード後に **`sha256sum --check`** で照合し、成功・失敗いずれも**英語**のメッセージを表示する。失敗時は破損ファイルを削除する。
- 手動取得: README の手順どおり `gguf.txt` から URL を変換して `wget` し、**`sha256sum --check $(MODEL).sha256sum`** で検証してもよい。

## 制約・既知の制限

- **8B を CPU で動かすため重い**場合がある。単スレッド **`cpu`** は参考実装・検証向け（上記 CPU 参考計測 decode **0.24 tok/s**）。**`cpu-omp`** は decode **4.94 tok/s** 程度。実用的な CPU 試行は **`cpu-blas`**（参考 decode **30.79 tok/s**）を推奨。
- **`cpu-blas`** は **OpenBLAS**（`libopenblas-dev` 等）が必要。**AVX2** 非対応 CPU では Q1_0 内積が generic 参照実装にフォールバックする。`-ffast-math` / **`-mfma`** 使用のため、環境によっては **`cpu`** / **`cpu-omp`** と数値がわずかに異なり得る。
- **`gpu-cuda`**（付録）: **CUDA Toolkit**・NVIDIA ドライバ・GPU 実機が必要。README 付録および本書「実行時の挙動（gpu-cuda）」参照。**将来別リポジトリ移行予定**。RTX 5090 参考 prefill **~294 tok/s**（バッチ prefill）、decode **~50 tok/s**。既定 **`CUDA_GENCODE=arch=compute_86,code=compute_86`**（PTX + ドライバ JIT）。**Blackwell（RTX 50 系）** は **CUDA 13 以降**と **`sm_120`** ネイティブコードが必要 — apt の **`nvidia-cuda-toolkit`（CUDA 11）** では不可。**`make blackwell`**（要 **sudo**）で CUDA 11 削除・**`cuda-toolkit-13`** インストール・Blackwell 向けビルドを一括実行。**VRAM** に prefill バッチ（**`max_seq` 分**）含む。**Flash Attention** shared ≈ 65 KB/ブロック・**`n_tokens ≤ max_seq`**（**`-l`**）。**`head_dim > FA_HD`（128）** では Attention no-op。
- **画像・マルチモーダル入力は非対応**（テキストデコーダのみ）。
- **コンテキスト長**を大きくすると KV 用メモリが増える。
- 商用水平の性能・公式実装との一致は保証しない。

## 補足：ドキュメント間の役割

- **`README.md` / `README.en.md`**: **入口**。**本文**は CPU 3 バリアント（クイックスタート・ベンチマーク・トラブルシュート）。**付録**（`---` 以降）は **`gpu-cuda`** の位置づけ（目的外・将来別リポジトリ）と技術メモ・GPU ベンチマーク。初見は CPU 本文のみでよい。
- **`doc/design.md`（本書）**: **設計・仕様の静的説明**（CPU 3 バリアント + **`gpu-cuda`** 付録の実装詳細）。ベンチマークの**正**は CPU 表を README 本文、GPU 表を本書（README 付録に写し）。
- **`doc/ChangeLog`**: 履歴。

実装の**正**は **`bonsai-8b/cpu/main.c`**。**`cpu-omp`** / **`cpu-blas`** は CPU 派生。**`gpu-cuda`** は付録（README 付録と本書「実行時の挙動（gpu-cuda）」）。**Prefill** **`gpu_forward_prefill`**、**Decode** **`gpu_forward`**。浮動小数の結合順などで数値差が出うる。

## 補足：`design.md` 更新時のチェックリスト

1. **`bonsai-8b/Makefile`** / **`cpu/Makefile`** / **`cpu-omp/Makefile`** / **`cpu-blas/Makefile`** / **`gpu-cuda/Makefile`** と矛盾がないか。
2. **`README.md` と `README.en.md`** の**本文（CPU）**および**付録（`gpu-cuda`）**と整合するか。
3. 仕様変更は **`doc/ChangeLog`** に記録する。
