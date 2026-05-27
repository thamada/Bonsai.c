# Bonsai.c

英語版は [README.en.md](README.en.md) を参照してください。

本リポジトリは、[PrismML](https://prismml.com/) の **1-bit Bonsai 8B** を GGUF（`Bonsai-8B-Q1_0`）から、**ライブラリに依存せず単一の C言語ソースで直接動かす推論実装**です。

### 1-bit Bonsai 8B について

[Announcing 1-bit Bonsai: The First Commercially Viable 1-bit LLMs](https://prismml.com/news/bonsai-8b)（PrismML, 2026）では、**1-bit Bonsai 8B** が埋め込み・アテンション・MLP・言語モデルヘッドに至るまで **ネットワーク全体を 1-bit で設計**し、高位精度への「逃げ道」を置かない **真の 1-bit モデル**（約 82 億パラメータ）であることが説明されています。公開ウェイトは **Apache License 2.0** です。エッジからクラウドまで **知性の密度（intelligence density）** と実用的なスループット・省エネを両立させる、というビジョンと、モデルサイズの小ささ（記事では **約 1.15 GB**）が強調されています。

**本リポジトリが読み込む GGUF** は **`Bonsai-8B-Q1_0.gguf`**（Q1_0 量子化）です。**テキストのプロンプト入出力**に限定し、画像入力は扱いません。

**PyTorch・TensorFlow・JAX・ONNX Runtime など、機械学習向けのユーザランドライブラリ／ランタイムは一切リンクしていません。**  
推論の基準となる実装は **標準Cと `libm`** のみで、`bonsai-8b/cpu/main.c` から **CPU 単スレッド**の実行ファイル（`bonsai-cpu`）をビルドします。  
より速い検証向けに、同じ GGUF に対応した **OpenMP マルチスレッド**版を `bonsai-8b/cpu-omp/main.c` から **`bonsai-cpu-omp`** として別ビルドできます（ランタイムは **標準C + `libm` + OpenMP ランタイム**）。  
さらに **`bonsai-8b/cpu-blas/`** では **OpenMP + OpenBLAS** と **Q1_0 専用の融合内積カーネル**で **`bonsai-cpu-blas`** をビルドできます（**標準C + `libm` + OpenMP + OpenBLAS**）。

### なぜライブラリ非依存なのか

一般的な LLM推論では、**計算手順、メモリ配置、量子化レイアウト**といった低レベルな詳細がフレームワーク内部に隠れがちです。

本リポジトリでは **GGUF の読み取り、重みの復元、行列演算、Transformer の forward、サンプリングまでを C のコードパスとして明示**します。目的は PyTorch の代替ではなく、推論処理を**観察・検証・改造**しやすくすることです。

- **理解可能性**: ソースと `doc/design.md` から処理の流れを追える  
- **依存の単純化**: 基本的な C ツールチェーンで動かせる  
- **実験の自由度**: 量子化やメモリ表現などを個別に試しやすい  
- **参照実装**: **Bonsai 8B（GGUF）** のデコーダ推論の最小例として使える  

最高性能や機能網羅が目的ではありません。

## まず何ができるのか

**CPU 単スレッド**版、その **OpenMP** 並列版、**OpenMP + OpenBLAS** 最適化版の 3 通りがあります。

| 実行方法 | 使うファイル | 作られる実行ファイル | 向いている用途 |
|---|---|---|---|
| CPU 単スレッド | `bonsai-8b/cpu/main.c` | `cpu/bonsai-cpu` | 仕組みを追う、最小依存で動かす |
| CPU + OpenMP | `bonsai-8b/cpu-omp/main.c` | `cpu-omp/bonsai-cpu-omp` | マルチコアでの試運転（参照用） |
| CPU + OpenMP + OpenBLAS | `bonsai-8b/cpu-blas/main.c` | `cpu-blas/bonsai-cpu-blas` | マルチコアでの実用的なスループット（CPU 推奨） |

8B 級モデルの CPU 実行は依然として **重い**です。まずは `-n 1` など短い生成で動作確認し、本番的な試行は **`cpu-blas`** を推奨します（下記 [参考ベンチマーク](#参考ベンチマーク)）。

## ディレクトリ構成

```text
.
├── README.md
├── README.en.md
├── doc/
│   ├── ChangeLog
│   └── design.md
└── bonsai-8b/
    ├── Makefile              # make model のみ（GGUF 取得・checksum）
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
    ├── gpu-cuda/          # 付録（README 末尾）
    └── gpu-rocm/          # 付録（README 末尾）
```

推論コードは **`bonsai-8b/cpu/`**（参照・単スレッド）が基準です。並列版は **`bonsai-8b/cpu-omp/`**、CPU 最適化版は **`bonsai-8b/cpu-blas/`** です。GPU 向けは **`gpu-cuda`**（NVIDIA）と **`gpu-rocm`**（AMD）が付録としてあります（本文末尾）。

## 初心者向け: LLM推論で何が起きるか

1. **GGUF を読む**  
2. **プロンプトをトークン化する**  
3. **Transformer を実行する**（1 トークンずつ forward）  
4. **サンプリングする**（`-t`、`-k` など）  
5. **トークンを文字列に戻す**  

この流れを **PyTorch なし**で **C ソースから**追えます。

## 必要なもの

- Linux  
- `make`  
- Cコンパイラ（例: `gcc`, `clang`）  
- `libm`  
- **`cpu-omp`** をビルドするときは **OpenMP に対応したコンパイラ**（GCC / Clang と通常同梱の OpenMP ランタイム、`libgomp` または `libomp`）  
- **`cpu-blas`** をビルドするときは上記に加え **OpenBLAS**（例: Debian/Ubuntu では `libopenblas-dev`）  
- **`wget`**（`make model` で GGUF を**初回**取得するとき）  
- **Bonsai-8B-Q1_0.gguf**（[prism-ml/Bonsai-8B-gguf](https://huggingface.co/prism-ml/Bonsai-8B-gguf)。`bonsai-8b/` で `make model`）

```bash
sudo apt update
sudo apt install -y build-essential make
# cpu-blas を使う場合:
# sudo apt install -y libopenblas-dev
```

## モデルファイルを置く

`bonsai-8b/Makefile` は **`model` ターゲットのみ**（GGUF 取得と SHA256 検証。`make` だけでも同じ）。既定は **`MODEL=Bonsai-8B-Q1_0.gguf`** です。ビルド・実行時のモデルパスは各サブディレクトリの Makefile で指定します（例: `cd cpu && make run MODEL=/data/models/Bonsai-8B-Q1_0.gguf`）。

GGUF 本体はリポジトリに含めません。`bonsai-8b/gguf.txt` の URL から **`make model`** でダウンロードし、`bonsai-8b/` 直下に置きます。**既にファイルがある場合は再ダウンロードせず、`Bonsai-8B-Q1_0.gguf.sha256sum` による SHA256 検証のみ**行います（失敗時は破損ファイルを削除）。

```bash
cd bonsai-8b
make model
```

手動で取得する場合:

```bash
cd bonsai-8b
url=$(sed 's|/blob/main/|/resolve/main/|' gguf.txt)
wget -O Bonsai-8B-Q1_0.gguf "$url"
sha256sum --check Bonsai-8B-Q1_0.gguf.sha256sum
```

配置例:

```text
bonsai-8b/
├── Makefile
├── cpu/ … （`main.c` → `bonsai-cpu`）
├── cpu-omp/ … （`main.c` → `bonsai-cpu-omp`）
├── cpu-blas/ … （`main.c` → `bonsai-cpu-blas`）
└── Bonsai-8B-Q1_0.gguf
```

整合性は **`make model`** が **`Bonsai-8B-Q1_0.gguf.sha256sum`** で検証します（既存ファイルでも毎回 checksum を確認）。[Hugging Face](https://huggingface.co/prism-ml/Bonsai-8B-gguf/tree/main) 上のハッシュとも一致するはずです。

## いちばん簡単な実行手順

```bash
cd bonsai-8b
make model
cd cpu && make build
./bonsai-cpu ../Bonsai-8B-Q1_0.gguf -p "Hello" -n 1
```

任意で OpenMP 版（別ディレクトリでビルド）:

```bash
cd bonsai-8b/cpu-omp
make build
./bonsai-cpu-omp ../Bonsai-8B-Q1_0.gguf -p "Hello" -n 1
```

## CPU 単スレッド版

### ビルド

```bash
cd bonsai-8b/cpu
make build
```

成功すると **`bonsai-cpu`** がこのディレクトリにできます。

### 実行

```bash
cd bonsai-8b/cpu
./bonsai-cpu ../Bonsai-8B-Q1_0.gguf -p "日本語で短く自己紹介してください。" -n 16
```

`Makefile` の **`run`**:

```bash
cd bonsai-8b/cpu
make run PROMPT="日本語で短く自己紹介してください。"
```

別ディレクトリのモデル:

```bash
cd bonsai-8b/cpu
make run MODEL=/data/models/Bonsai-8B-Q1_0.gguf PROMPT="Hello"
```

## CPU OpenMP 版（`cpu-omp`）

`cpu/main.c` と同じモデル・CLI を前提にし、行列積・アテンション・SwiGLU などを **OpenMP** で並列化したビルドです。

### ビルド

```bash
cd bonsai-8b/cpu-omp
make build
```

成功すると **`bonsai-cpu-omp`** がこのディレクトリにできます。

### 実行

リポジトリ直下のモデルがある場合:

```bash
cd bonsai-8b/cpu-omp
./bonsai-cpu-omp ../Bonsai-8B-Q1_0.gguf -p "Hello" -n 16
```

スレッド数は環境変数で指定できます（既定は実行環境の OpenMP に依存）。

```bash
OMP_NUM_THREADS=8 ./bonsai-cpu-omp ../Bonsai-8B-Q1_0.gguf -p "Hello" -n 16
```

`Makefile` の `run` は既定で `MODEL=../Bonsai-8B-Q1_0.gguf` を渡します。

```bash
make run PROMPT="日本語で短く自己紹介してください。"
```

## CPU OpenMP + OpenBLAS 版（`cpu-blas`、推奨）

`cpu-omp` と同じモデル・CLI を前提に、**Q1_0 専用の融合内積**（中間 FP32 展開なし）と **OpenBLAS**（Attention の `sgemv` 集約、F32 行の `sgemv`）で高速化したビルドです。OpenBLAS は **1 スレッド固定**とし、並列度は **OpenMP** に任せます（ネスト並列の衝突を避けるため）。

### ビルド

```bash
sudo apt install -y libopenblas-dev   # 未導入の場合
cd bonsai-8b/cpu-blas
make build
```

成功すると **`bonsai-cpu-blas`** がこのディレクトリにできます（ビルド時に **`SIMD flags:`** が表示されます。**`/proc/cpuinfo`** から **`-mavx2`**（FMA ありなら **`-mfma`**）等を自動選択。上書き: `make build ARCH_FLAGS='-mavx2 -mfma'`）。

### 実行

```bash
cd bonsai-8b/cpu-blas
./bonsai-cpu-blas ../Bonsai-8B-Q1_0.gguf -p "Hello" -n 16
```

```bash
cd bonsai-8b/cpu-blas
make run PROMPT="Hello"
```

## 参考ベンチマーク

以下は **開発環境での参考値**です。CPU・メモリ・ビルドフラグ・GGUF の配置で大きく変わります。再現時は同じ GGUF・同じコマンドで計測してください。

| 項目 | 値 |
|---|---|
| CPU | AMD Ryzen 9 5950X（32 論理コア） |
| OS | Linux |
| モデル | `Bonsai-8B-Q1_0.gguf` (pre-read, in RAM) |
| コマンド | `./<binary> Bonsai-8B-Q1_0.gguf -p "Hello" -n 16 -t 0`（`-n` は decode 上限、`-t` は温度） |
| ワークロード | prefill **18** トークン（`-p "Hello"` + chat テンプレート）+ decode **16** トークン（`-n 16`） |
| 表の指標 | decode 時間・tok/s のみ（stderr の `Decode complete` 行。prefill は含めない） |
| 環境変数 | `cpu-omp` / `cpu-blas`: `OMP_NUM_THREADS=32`、`cpu-blas` のみ `OPENBLAS_NUM_THREADS=1` |
| 再現 | 各バイナリで 1 回ウォームアップ後、3 回計測の**最高** decode tok/s (GGUF pre-read in RAM) |

| バイナリ | decode 時間 | decode スループット | 備考 |
|---|---:|---:|---|
| `cpu/bonsai-cpu` | 66.8 s | **0.24 tok/s** | 単スレッド、`-O3`（`cpu/Makefile` 既定） |
| `cpu-omp/bonsai-cpu-omp` | 3.2 s | **4.94 tok/s** | `-O3 -fopenmp`（`cpu-omp/Makefile` 既定） |
| `cpu-blas/bonsai-cpu-blas` | 0.5 s | **30.79 tok/s** | `-O3 -fopenmp -ffast-math`、`ARCH_FLAGS` は cpuinfo 自動（5950X 時 `-mavx2 -mfma`）、OpenBLAS 1 スレッド |

この条件下では **`cpu-blas` が `cpu-omp` の約 6 倍**、**`cpu` の約 128 倍**の decode スループットでした（`cpu-omp` は `cpu` の約 21 倍）。生成テキスト（`Hello! I'm Bonsai, an AI assistant developed by PrismML.`）は 3 バイナリで一致しています。

## よく使うオプション

| オプション | 例 | 意味 |
|---|---|---|
| `-p` | `-p "Hello"` | プロンプト |
| `-n` | `-n 64` | 最大生成トークン数 |
| `-t` | `-t 0.7` | 温度 |
| `-k` | `-k 0.9` | Top-p |
| `-s` | `-s 1234` | 乱数シード |
| `-l` | `-l 512` | 最大シーケンス長 |

```bash
./cpu/bonsai-cpu Bonsai-8B-Q1_0.gguf -p "Hello" -n 4
```

OpenMP 版も同様のオプションです。

```bash
./cpu-omp/bonsai-cpu-omp Bonsai-8B-Q1_0.gguf -p "Hello" -n 4
```

OpenBLAS 版も同様です。

```bash
./cpu-blas/bonsai-cpu-blas Bonsai-8B-Q1_0.gguf -p "Hello" -n 4
```

## 生成を安定させたいとき

```bash
./cpu/bonsai-cpu Bonsai-8B-Q1_0.gguf \
  -p "1文で説明してください: GGUFとは？" \
  -n 32 \
  -t 0.2 \
  -s 42
```

## 片付け

各サブディレクトリで `make clean` します。GGUF は削除されません。

```bash
cd bonsai-8b/cpu && make clean
cd bonsai-8b/cpu-omp && make clean
cd bonsai-8b/cpu-blas && make clean
# 付録 GPU:
cd bonsai-8b/gpu-cuda && make clean
cd bonsai-8b/gpu-rocm && make clean
```

## よくあるトラブル

### `No such file or directory`

モデルパスを確認してください。未取得なら `bonsai-8b` で **`make model`** を実行してください。

```bash
ls -lh bonsai-8b/Bonsai-8B-Q1_0.gguf
cd bonsai-8b && make model
```

```bash
./cpu/bonsai-cpu /data/models/Bonsai-8B-Q1_0.gguf -p "Hello" -n 4
```

### CPU 版が遅い

正常です。8B を CPU で回す負荷は大きいです。`-n 1` や `-n 4` から試すか、**`cpu-blas`**（`libopenblas-dev` 要）を使ってください。`cpu-omp` のみの場合は **`OMP_NUM_THREADS`** を調整してください。

```bash
./cpu/bonsai-cpu Bonsai-8B-Q1_0.gguf -p "Hello" -n 1
```

### OpenMP がリンクできない／`-fopenmp` が通らない

コンパイラと OpenMP の開発パッケージ（ディストリビューションにより `libgomp`、`libomp` など）を入れたうえで、再度 `cpu-omp` の `make build` を試してください。

### OpenBLAS が見つからない（`cpu-blas`）

`libopenblas-dev`（またはディストリビューション相当）をインストールし、`cpu-blas` で `make build` を再実行してください。ヘッダが非標準パスの場合は `cpu-blas/Makefile` のコメントを参照し `CPPFLAGS` で `-I` を指定します。

## 実装を読みたい人へ

1. `README.md` / `README.en.md`  
2. `doc/design.md`  
3. `bonsai-8b/cpu/main.c` — 単スレッドの基準実装  
4. `bonsai-8b/cpu-omp/main.c` — OpenMP 並列版  
5. `bonsai-8b/cpu-blas/main.c` — OpenMP + OpenBLAS + Q1_0 融合カーネル  

## このリポジトリで扱わないもの

- 学習・ファインチューニング  
- **AMD NPU（XDNA2 等）** や **Vulkan / Metal** など（**`gpu-cuda`** / **`gpu-rocm`** は付録の参考実装。本文の主眼は CPU 3 バリアント）  
- バッチ推論の最適化（GPU 付録の prefill バッチは decode 高速化用）  
- 画像入力  
- サーバ化・Web API 化  
- すべての GGUF 量子化形式への汎用対応  
- 公式実装との完全な数値一致保証  

目的は、**Bonsai-8B-Q1_0**（GGUF）のテキスト推論を **C** で理解し、実験し、必要に応じて改造できるようにすることです。

## 詳細ドキュメント

- `doc/design.md`  
- `doc/ChangeLog`  

困ったときは、`bonsai-8b/Makefile`（**`model` のみ**）と各サブディレクトリの Makefile（`cpu/`、`cpu-blas/`、`gpu-cuda/` 等）、および実行時のモデルパスを確認してください。

---

## NVIDIA CUDA 実装（`gpu-cuda`）について

**`bonsai-8b/gpu-cuda/` は本リポジトリの目的（単一 C ソース・依存最小）から外れた付録**です。`main.c` + `kernels.cu` + `gpu.h`（任意で `fp4_bonsai.cu` / `fp4_gemm.cu` + CUTLASS）に分かれ、**CUDA Toolkit・NVIDIA ドライバ・GPU 実機**が必要です。参照すべき最小実装は **`cpu/main.c`** です。

同梱している理由は、筆者が **CUDA でどこまで高速化できるか** を試したくなっただけです。本プロジェクトの目的を補うものでも、読者向けの正式な機能でもありません。混乱を招きやすいため、**将来的には別リポジトリへ移す予定**です。初めて読む方は無視して構いません。

以下は、GPU 上の速度比較に興味がある場合の技術メモです。

### ディレクトリ構成

```text
bonsai-8b/gpu-cuda/
├── Makefile
├── main.c
├── kernels.cu
├── gpu.h
├── fp4_bonsai.cu / fp4_bonsai.h   # NVFP4 ブリッジ（BONSAI_FP4=1 時）
├── fp4_gemm.cu / fp4_gemm.h       # CUTLASS block-scaled NVFP4 GEMM
└── third_party/cutlass/           # make cutlass で取得（FP4 ビルド時）
```

### 技術的な概要

`cpu-blas` と同じ GGUF・CLI を前提とします。ホスト側（`main.c`）は GGUF の mmap 読み取り、トークナイザ、サンプリングを担当し、GPU 側（`kernels.cu`）は forward 本体を担当します。

#### VRAM 上のデータ配置（`gpu_model_create`）

起動時にホスト（mmap 上の GGUF）から **H2D コピー**で VRAM に載せるもの:

| 種別 | 内容 | 形式 |
|---|---|---|
| 重み | `token_embd`, 各層 `wq/wk/wv/wo/gate/up/down`, `output` | Q1_0（g128）。**`make run`（FP4）** では線形層のみ起動時に **NVFP4 + block scale の GPU キャッシュ**へ変換（下記） |
| 重み | `attn_norm`, `q_norm`, `k_norm`, `ffn_norm`, `output_norm` | F32 |
| KV キャッシュ | `kc`, `vc` | F32、`n_layers × max_seq × kv_dim` |
| Decode 用活性化 | `x`, `xb`, `xb2`, `q`, `k`, `v`, `hb`, `hb2`, `logits`, `q8` | F32 / Q8_0 |
| Prefill 用バッチ | `x_batch`, `xb_batch`, … `q8_batch` 等 | `-l`（`max_seq`）トークン分を事前確保 |

Prefill バッファの容量は **`batch_cap = max_seq`**（CLI の `-l`）です。`n_tokens > max_seq` の場合はエラーで停止します。

#### 生成ループ（`main.c` の `generate`）

1. **Prefill**（`n_prompt > 1`）: `gpu_forward_prefill` を **1 回**呼び出し、全プロンプトトークンをまとめて forward。末尾位置の logits だけ `gpu_copy_logits` で CPU へ。
2. **Prefill**（`n_prompt == 1`）: 従来どおり `gpu_forward` を 1 回。
3. **Decode**: サンプリングで得たトークンを 1 個ずつ `gpu_forward(token, pos)` に渡す。`pos` は `n_prompt` から増加。

サンプリング（温度・Top-p・乱数）は **CPU のみ**。GPU から返るのは vocab サイズの logits ベクトル（D2H コピー）だけです。

Prefill 中のプログレスバーは **0% → バッチ完了で 100%** です（`gpu_forward_prefill` が 1 カーネル起動群のため、CPU 版のようなトークン逐次更新はありません）。

#### Decode: `gpu_forward`（1 トークン / 1 位置）

各 decode ステップで、レイヤー `l = 0 … n_layers-1` に対し次を実行します。

1. **Embedding** — `emb_q1_0_kernel`: Q1_0 行を F32 へ dequant して `x` に載せる。
2. **Attention 前 norm** — `rmsnorm_kernel`（F32 重み）。
3. **Q/K/V 投影** — `gpu_mm`（Q1_0 GEMV、後述）で `q`, `k`, `v` を計算。
4. **Q/K head norm** — `rmsnorm_head_kernel`。
5. **RoPE** — `rope_neox_kernel`（NeoX 半分ペア）。cos/sin テーブルは **CPU で YaRN メタを解釈して生成**し、`rope_cache` へ H2D。
6. **KV 書き込み** — 現在位置 `pos` の `k`, `v` を `kc`, `vc` へ D2D コピー（レイアウト: `[layer][seq_pos][kv_dim]`）。
7. **Attention** — `flash_attn_gqa_kernel`（後述）。出力は `xb`。
8. **出力投影 + 残差** — `gpu_mm`（`wo`）→ `add_kernel`。
9. **FFN 前 norm** → **gate/up 投影** → **SwiGLU** → **down 投影** → **残差**。
10. **最終 norm + LM head** — `rmsnorm_kernel` → `gpu_mm`（`output`）→ `logits`。

#### Prefill: `gpu_forward_prefill`（全プロンプト位置を並列）

プロンプトトークン ID を `tokens_dev` へ H2D コピーし、位置 0 … `n_tokens-1` 分の RoPE テーブルを **CPU で一括生成**して `rope_batch` へ載せます。

| 処理 | カーネル | 並列の仕方 |
|---|---|---|
| Embedding | `emb_q1_0_batch_kernel` | grid `(nb, n_tokens)` — トークン × Q1_0 ブロック |
| RMSNorm | `rmsnorm_batch_kernel` | 1 block / トークン |
| Q/K/V/O, gate/up/down | `gpu_mm_batch` | `(token, 出力行)` ごとに並列 |
| Q/K head norm | `rmsnorm_head_batch_kernel` | 1 block / `(token, head)` |
| RoPE | `rope_neox_batch_kernel` | 1 block / `(token, head)`、位置別 cos/sin |
| KV 書き込み | `kv_write_batch_kernel` | 1 block / トークン → `kc[0…n-1]`, `vc[0…n-1]` を一括填充 |
| Attention | `flash_attn_prefill_gqa_kernel` | 1 block / `(token, head)`、因果マスク `npos = t + 1` |
| 残差・SwiGLU | `add_batch_kernel`, `swiglu_batch_kernel` | 要素並列 |

Attention 以降の FFN もすべてバッチ版カーネルで、`x_batch` に `[n_tokens, dim]` 形状の隠れ状態を保持します。

**LM head は末尾トークンのみ:** 全位置の logits は計算せず、`x_batch[(n_tokens-1) * dim]` だけを `rmsnorm_kernel` → `gpu_mm` して `logits` を得ます（最初の decode トークン予測用）。

#### Q1_0 GEMV（`gpu_mm` / `gpu_mm_batch`）— 既定・`make run.no-fp4`

`cpu-blas` と同じ方針です。重みは **Q1_0 のまま VRAM 常駐**（事前 dequant しない）。**`make run`（NVFP4）** では線形層だけ次節の処理に切り替わります。

1. 入力ベクトル（またはバッチ各行）を **`quantize_q8_0_kernel`** で Q8_0（group size 32）に量子化。
2. **`mm_q1_0_kernel`** / **`mm_q1_0_batch_kernel`** で `vec_dot_q1_0_q8_0` を実行 — Q1_0 重み行と Q8_0 活性化の内積。1-bit 重みの符号ビットと Q8_0 の int8 積を組み合わせ、FP16 スケールで復元。
3. 1 Q1_0 ブロック（128 要素）あたり Q8_0 ブロック 4 個と対応。

Decode 版は出力次元 `d` 方向に 256 スレッド/block で並列。Prefill 版は `(n_tokens × d)` 個の出力要素を並列計算します。

#### NVFP4 + CUTLASS（Blackwell、`make run`）

**NVFP4 とは（要約）:** NVIDIA Blackwell 向けの **4 bit 浮動小数**形式で、行列積は **ブロック単位のスケール（block-scaled）** と組み合わせて Tensor Core で実行する。1 要素あたり **4 bit の E2M1** に値を丸め、近傍 **16 要素**（`SF_VEC_SIZE`）ごとに **8 bit の UE4M3** スケールを 1 つ持つ。復元のイメージは `実数 ≈ scale × E2M1の値` である。

**E2M1（1 要素・4 bit）** — 符号 1 bit + 指数 2 bit + 仮数 1 bit（CUTLASS の `float_e2m1_t` と同型）。**表現できる絶対値は 8 段階のみ**（NVIDIA 公式・本実装のルックアップテーブル共通）:

| code (3 bit) | 絶対値 |
|:---:|:---:|
| 0 | 0 |
| 1 | 0.5 |
| 2 | 1.0 |
| 3 | 1.5 |
| 4 | 2.0 |
| 5 | 3.0 |
| 6 | 4.0 |
| 7 | 6.0 |

bit 3 が符号（0=非負、1=負）。**0.75 は上記 8 段階のいずれでもない** — 量子化の**しきい値**（隣接値の中点）として使われる。[NVIDIA Model-Optimizer](https://github.com/NVIDIA/Model-Optimizer/blob/main/modelopt/torch/quantization/qtensor/nvfp4_tensor.py) では `0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5` などが境界。`0.75` のような実数は、ブロックスケール `scale` を掛けたうえで最も近い段階に丸められる。

ビット割り当て（`fp4_gemm.cu` の `d_float_to_fp4`）:

```text
  bit:   3    2 1 0
        +---+-------+
        | S | code  |   S: 符号 (0=非負, 1=負) ／ code: 上表の 3 bit インデックス
        +---+-------+
```

2 要素を **1 byte** にパック（下位ニブルが偶数インデックス）:

```text
  byte:  7 6 5 4 | 3 2 1 0
        +---------+---------+
        | fp4[i+1]| fp4[i]  |
        +---------+---------+
```

**UE4M3（ブロックスケール・8 bit / 16 要素）** — 符号なし、指数 4 bit（バイアス 7）+ 仮数 3 bit。ブロック内の最大絶対値からスケールを決め、各 E2M1 をそのスケールで正規化してから量子化する（CUTLASS の block-scaled NVFP4 レイアウト）。

```text
  bit:   7 6 5 4   3 2 1 0
        +---------+-------+
        |  exp(4) | mant(3)|   値の範囲は実装コメント上おおよそ [~0.002, 480]
        +---------+-------+
```

本リポジトリの GGUF は **Q1_0（1 bit 重み + FP16 スケール）** のまま配布し、**起動時に BF16 へ復元 → NVFP4 キャッシュ**へ変換してから GEMM に渡す（GGUF ファイル自体は NVFP4 ではない）。

**`BONSAI_FP4=1`** でビルドしたときのみ有効（`gpu-cuda/Makefile` の **`make run`** / **`make blackwell`**）。**`main.c` は NVFP4 を直接呼びません** — GGUF 読み込み・`generate`・サンプリングは従来どおりで、**`kernels.cu` 内の線形層 GEMV/GEMM だけ**が FP4 Tensor Core 処理に切り替わります。Embedding・RMSNorm・RoPE・Flash Attention・SwiGLU・KV キャッシュは変更ありません。

| ファイル | 役割 |
|---|---|
| `fp4_gemm.cu` | [CUTLASS](https://github.com/NVIDIA/cutlass) の **SM120 block-scaled NVFP4 GEMM**（Example 79a ベース）。BF16 行列の **E2M1（NVFP4）+ UE4M3 スケール** への量子化カーネルと、Tensor Core 上の `fp4_gemm_run_cached`。 |
| `fp4_bonsai.cu` | Bonsai 向けブリッジ: ホスト上の **Q1_0 重み**を BF16 に復元してから **起動時に一度だけ** NVFP4 キャッシュ化し、推論時は **F32 活性化 ↔ BF16 ↔ CUTLASS GEMM ↔ F32 出力** を担当。 |
| `kernels.cu` | `#ifdef BONSAI_FP4` で `gpu_model_create` の重みアップロードと、`gpu_mm` / `gpu_mm_batch` のディスパッチを FP4 版（`gpu_mm_fp4` → `fp4_bonsai_mm`）に切り替え。 |

**どこで使うか（線形層のみ）:** 各層の **`wq` / `wk` / `wv` / `wo` / `gate` / `up` / `down`** および最終 **`output`（LM head）**。Decode のステップ 3・8・9・10、Prefill の `gpu_mm_batch` 行が該当します。

**どのように使うか:**

1. **ビルド** — `make cutlass` で `third_party/cutlass` を取得し、`fp4_gemm.o` / `fp4_bonsai.o` をリンク。GPU コードは **`sm_120a`** ネイティブ（PTX JIT では Tensor Core FP4 が使えない）。
2. **起動時（`gpu_model_create`）** — `fp4_bonsai_init` で CUTLASS 用ワークスペースを確保。各層の Q1_0 テンソルを `fp4_bonsai_weight_from_q1_host` で **GPU 常駐の FP4 重みキャッシュ**（パック済み NVFP4 + スケール）へ変換（`M`/`N`/`K` は **128 倍数**へパディング）。
3. **推論時（`gpu_forward` / `gpu_forward_prefill`）** — `gpu_mm` / `gpu_mm_batch` が `fp4_bonsai_mm` を呼ぶ。活性化（F32、`M` 行 × `n` 列）は都度 BF16 へ載せて **NVFP4 化**し、**キャッシュ済み FP4 重み**と CUTLASS GEMM。結果を F32 の `o` / `logits` 用バッファへ戻したあと、以降の norm / Attention 等は Q1_0 版と同じカーネル。
4. **確認** — 起動時 stderr に `GPU: FP4 Tensor Core path enabled (NVFP4, CUTLASS sm_120)` が出れば FP4 処理が有効。

**`make run.no-fp4`** では本節は無効で、上記「Q1_0 GEMV」（Q8_0 活性化 + `mm_q1_0_*` カーネル）のみです。

#### Flash Attention（GQA、online softmax）

`att` 行列 `[n_heads, seq, seq]` は **materialize しません**。K/V キャッシュをシーケンス方向に **64 トークン（`FA_BR`）タイル**で走査し、**online softmax**（running max `m` と sum `l`）で出力を更新します。

- **Decode** — `flash_attn_gqa_kernel`: grid = `n_heads` blocks × `FA_HD`（128）threads。1 クエリ位置（現在トークン）× 全ヘッド。GQA: ヘッド `h` は KV ヘッド `h / kv_mul` を参照。
- **Prefill** — `flash_attn_prefill_gqa_kernel`: grid = `n_tokens × n_heads` blocks。位置 `t` のクエリは **因果マスク**により K/V の `0 … t` のみ参照（`npos = t + 1`）。

各タイル内の処理:

1. K/V を **shared memory**（`k_tile[64][128]`, `v_tile[64][128]`）へ協調ロード。
2. `Q · K^T` → `scores[]`（shared）。
3. タイル内 max で online softmax の `m`, `l` を更新。
4. softmax 重み × V を `o_sh[]` に累積。

shared memory 合計は約 65 KB のため、起動前に **`cudaFuncAttributePreferredSharedMemoryCarveout = 100`** を設定しています（48 KB 既定を超えるため）。

#### RoPE

CPU 側（`build_rope_cache_host`）で llama.cpp 準拠の **NeoX 半分ペア** + **YaRN** メタ（`rope.scaling.*`, `context_length` 等）を解釈し、cos/sin テーブルを生成。Prefill 時は位置ごとに `n_tokens` 分を `rope_batch` へ、Decode 時は 1 位置分を `rope_cache` へ H2D します。

#### 制約・既知の挙動

- 対象 GGUF: **`Bonsai-8B-Q1_0`**（Q1_0 g128 + F32 norm）。他量子化形式は未対応。
- `head_dim > 128`（`FA_HD`）のモデルでは Flash Attention カーネルが no-op になります（Bonsai-8B では `head_dim = 128`）。
- Prefill と Decode で KV キャッシュは共有 — Prefill 後の Decode は `pos = n_prompt` から続き、Prefill で填充済みの `kc`/`vc` をそのまま参照します。

### 必要なもの

- **CUDA Toolkit**（**`nvcc`**）と **NVIDIA ドライバ**（**`libcudart`**。cuBLAS 不要）

```bash
sudo apt install -y nvidia-cuda-toolkit   # または NVIDIA 公式 CUDA Toolkit
```

### ビルド

| 目的 | コマンド（`bonsai-8b/gpu-cuda/`） | 備考 |
|---|---|---|
| 汎用 GPU（PTX JIT） | `make build` / `make run.no-fp4` | Q1_0 + Q8_0 カーネル（**NVFP4 なし**） |
| Blackwell + NVFP4 | `make` または `make run` | CUDA 13 導入 + **`sm_120a`** + CUTLASS（初回 **sudo** のことがある） |

```bash
cd bonsai-8b/gpu-cuda
make build          # または make run.no-fp4 用
# make run          # Blackwell + NVFP4（既定ターゲット）
```

成功すると **`bonsai-gpu-cuda`** がこのディレクトリにできます。

GPU アーキテクチャに合わせ **`CUDA_GENCODE`** を上書きできます（`make run.no-fp4` の既定は PTX `compute_86` + ドライバ JIT）:

```bash
make build CUDA_GENCODE=arch=compute_90,code=sm_90
```

### 実行

```bash
cd bonsai-8b/gpu-cuda
make run PROMPT="Hello"          # または make run.no-fp4 PROMPT="Hello"
```

CLI オプション（`-p`、`-n`、`-t`、`-k`、`-s`、`-l`）は CPU 版と同様です。

```bash
./gpu-cuda/bonsai-gpu-cuda Bonsai-8B-Q1_0.gguf -p "Hello" -n 4
```

### 参考ベンチマーク（GPU）

| 項目 | 値 |
|---|---|
| GPU | NVIDIA GeForce RTX 5090（31 GiB VRAM） |
| OS | Linux |
| モデル | `Bonsai-8B-Q1_0.gguf`（起動時 VRAM アップロード） |
| コマンド | `bonsai-8b/gpu-cuda` で `make run` または `make run.no-fp4`（いずれも `-p "Hello" -n 16 -t 0` 相当） |
| ワークロード | CPU 参考表と同じ（prefill 18 + decode 16 トークン） |
| 再現 | 各構成で 1 回ウォームアップ後、3 回計測の代表値（prefill / decode は stderr の `Prefill complete` / `Decode complete` 行） |

#### FP4 Tensor Core 有効（`make run`）

Blackwell（RTX 50 系）向け。**CUDA 13**・**`sm_120a`** ネイティブ・**`BONSAI_FP4=1`**（NVFP4 + CUTLASS）・**`FA_BR=32`**。`gpu-cuda/` で `make run`（内部で `make blackwell` 相当のビルド後に実行。初回は CUDA 13 導入で **sudo** が必要な場合あり）。

| バイナリ | prefill tok/s | decode 時間 | decode スループット | 備考 |
|---|---:|---:|---:|---|
| `gpu-cuda/bonsai-gpu-cuda` | **~1365** | 0.18 s | **~90.4 tok/s** | stderr に `GPU: FP4 Tensor Core path enabled`（2026-05-21 実測） |

#### FP4 無効（`make run.no-fp4`）

PTX **`compute_86`** + ドライバ JIT、Q1_0 GEMV（FP4 パスなし）。`gpu-cuda/` で `make run.no-fp4`（`make build` のみでリンク）。

| バイナリ | prefill tok/s | decode 時間 | decode スループット | 備考 |
|---|---:|---:|---:|---|
| `gpu-cuda/bonsai-gpu-cuda` | **~293** | 0.34 s | **~47.0 tok/s** | バッチ prefill・`-use_fast_math`（2026-05-21 実測） |

同一プロンプト・`-t 0` で、**いずれの構成も `cpu-blas` と同じ生成テキスト**（`Hello! I'm Bonsai, an AI assistant developed by PrismML.`）。`-ffast-math` / `-use_fast_math` により浮動小数の結合順は CPU 版と異なり得ますが、上記試行では同一出力でした。FP4 有効時は decode が **約 1.9 倍**、prefill が **約 4.7 倍**（いずれも tok/s）程度でした。

### よくあるトラブル（CUDA 版）

**CUDA / `nvcc` が見つからない:** CUDA Toolkit（**`nvcc`**）と NVIDIA ドライバをインストールし、`gpu-cuda` で `make build` を再実行してください。古い CUDA では `-arch=native` が使えないため、既定は PTX（`compute_86`）+ ドライバ JIT です。GPU に合わせ `CUDA_GENCODE` を指定してください（`gpu-cuda/Makefile` 参照）。プロンプトが `[prompt length … exceeds max_seq …]` で止まる場合は **`-l`** を大きくしてください。

**片付け:** `bonsai-8b/gpu-cuda` または `bonsai-8b/gpu-rocm` で `make clean` してください。

### ソースを読む場合

6. `bonsai-8b/gpu-cuda/main.c` — ホスト側（GGUF・トークナイザ・`generate`・サンプリング）。NVFP4 の切り替えは行わない  
7. `bonsai-8b/gpu-cuda/kernels.cu` — forward 本体・`gpu_mm*` のディスパッチ（FP4 時は `fp4_bonsai` へ）  
8. `bonsai-8b/gpu-cuda/fp4_bonsai.cu` / `fp4_gemm.cu` — Q1_0 ↔ NVFP4 ブリッジと CUTLASS GEMM（`BONSAI_FP4=1` 時）  
9. `bonsai-8b/gpu-cuda/gpu.h` — C API（`gpu_forward_prefill` 等）  

---

## AMD ROCm / HIP 実装（`gpu-rocm`）について

**`bonsai-8b/gpu-rocm/` も本リポジトリの目的（単一 C ソース・依存最小）から外れた付録**です。`main.c` + `kernels.hip` + `gpu.h`（**`gpu-cuda/gpu.h` をベースに `gpu_get_device_desc` を追加した C API**）に分かれ、**ROCm（`hipcc`）・AMD GPU ドライバ・実機**が必要です。**hipBLAS 不要**（線形層は Q1_0×Q8_0 のカスタム HIP カーネル）。**NVFP4 経路はありません**（アルゴリズムは **`gpu-cuda` の `make run.no-fp4`** と同系）。**`make log` / `make log.push`** でベンチマーク履歴を **`gpu-rocm/Makefile` に記録**できます。

筆者が **AMD GPU 上でも同じ forward を試す**ための実験用コードです。初めて読む方は無視して構いません。詳細仕様は **`doc/design.md`** の「ビルドと実行（GPU ROCm）」「実行時の挙動（gpu-rocm）」を参照してください。

### ディレクトリ構成

```text
bonsai-8b/gpu-rocm/
├── Makefile
├── main.c
├── kernels.hip
└── gpu.h          # gpu-cuda をベース（gpu_get_device_desc 追加）
```

### 技術的な概要

`cpu-blas` / **`gpu-cuda`（FP4 なし）** と同じ GGUF・CLI・**prefill バッチ + decode 逐次**（**`gpu_forward_prefill`** / **`gpu_forward`**）。ホストは GGUF mmap・トークナイザ・サンプリング、デバイスは **Flash Attention**（K/V shared staging、**`FA_BR`** タイル）と **Q8_0 活性化 + Q1_0 GEMV**。VRAM 配置の考え方は上記 CUDA 付録の表と同趣旨（API は **HIP** / **`hipMalloc`**）。

終了時に **`BENCH_LOG_FILE`**（既定 **`/tmp/benchmark.log`**）へ key=value 形式のベンチマークログを書きます。stderr の prefill/decode/total tok/s は **`generate()` 内**（**`gpu_model_create` による重み H2D の後**）のみを計測します。

### 必要なもの

- **ROCm**（既定 **`/opt/rocm`**、**`hipcc`**・**`rocminfo`**）
- **AMD GPU ドライバ**
- ホスト: **`g++`** と **`libstdc++-dev`**（`hipcc` が C++ 標準ライブラリを参照するため）

PyTorch・hipBLAS 等は不要です。

### ビルド

| 目的 | コマンド（`bonsai-8b/gpu-rocm/`） |
|---|---|
| ビルド＋実行（既定） | `make` / `make run` |
| ビルドのみ | `make build` |
| GPU ISA 確認 | `make detect-gpu-arch` |
| ベンチ履歴の表示 | `make log` |
| ベンチ実行＋履歴追記 | `make log.push`（例: `make log.push BENCH_N=64`） |

**`GPU_ARCH`**（例: `gfx1100`）は **`rocminfo`** から自動検出されます。未検出時は `make GPU_ARCH=gfx1100 build` のように明示してください。ビルド成功時に **Detected GPU arch** が表示されます。

```bash
cd bonsai-8b/gpu-rocm
make run
```

**`FA_BR`**: 既定 **32**（gfx11/gfx12 向け）。shared に余裕がある GPU では `make FA_BR=64 build` も可。

### 実行

```bash
cd bonsai-8b/gpu-rocm
./bonsai-gpu-rocm ../Bonsai-8B-Q1_0.gguf -p "Hello" -n 16
```

```bash
cd bonsai-8b/gpu-rocm
make run PROMPT="Hello"
```

CLI オプション（`-p`、`-n`、`-t`、`-k`、`-s`、`-l`）は CPU 版と同様です。

### ベンチマーク記録（`make log` / `make log.push`）

| 変数 | 既定 | 意味 |
|---|---|---|
| `BENCH_PROMPT` | 長文英文（Makefile 内） | ChatML 化後 **約 130 トークン**のプロンプト |
| `BENCH_N` | `128` | 最大生成トークン数（`-n`） |
| `BENCH_SEED` | `42` | 乱数シード（`-s`） |
| `BENCH_LOG_FILE` | `/tmp/benchmark.log` | 実行後に書き出す key=value ログ |

```bash
cd bonsai-8b/gpu-rocm
make log.push          # ビルド → ベンチ実行 → ログ解析 → Makefile に 1 行追記
make log               # 追記済み BENCH_LOG を表形式で表示
```

**`log.push` の流れ:**

1. **`BENCH_PROMPT`**・**`-n BENCH_N`**・**`-t 0`**・**`-s BENCH_SEED`** で `bonsai-gpu-rocm` を実行。
2. 終了時 **`/tmp/benchmark.log`**（または **`BENCH_LOG_FILE`**）に `prompt_tokens`・`gen_tokens`・`prefill_tps`・`decode_tps`・`total_tps` 等を書き出す。
3. **`log.push`** がログを読み、**`ISO8601|GPU_ARCH|hostname|prompt|gen|prefill|decode|total`** の 1 行を **`gpu-rocm/Makefile`** の **`# BENCH_LOG_END`** 直前に追記する。

**注意:** **`make log.push` は `gpu-rocm/Makefile` を書き換えます**。コミット前に `git diff` で差分を確認してください。表の **`total_tps`** は **推論区間のみ**（重みの VRAM アップロードは含みません）。

### 参考ベンチマーク（GPU ROCm）

| 項目 | 値 |
|---|---|
| GPU | **AMD gfx1201**（ROCm。**`GPU_ARCH`**） |
| OS | Linux |
| モデル | `Bonsai-8B-Q1_0.gguf` |
| ワークロード | 長文プロンプト（ChatML 後 **130** トークン）+ decode **128** トークン（**`make log.push`** 既定: **`-n 128 -t 0 -s 42`**） |
| 表の指標（prefill / decode） | stderr の **`Prefill complete` / `Decode complete`** |
| 表の指標（total） | ベンチログの **`total_tps`**（推論区間。重み H2D 除外） |
| 再現 | `bonsai-8b/gpu-rocm/` で **`make log.push`** → **`make log`** |

| 計測日時 | prefill tok/s | decode tok/s | total tok/s | 備考 |
|---|---:|---:|---:|---|
| 2026-05-27 17:21 | **175.03** | **41.89** | **67.92** | gfx1201、130+128 トークン |
| 2026-05-27 17:29 | **174.18** | **42.06** | **68.08** | 同上（2 回目） |

**CPU 表**（`-p "Hello" -n 16`）や **CUDA 付録**（prefill 18 + decode 16）とは**プロンプト長・トークン数が異なる**ため、数値をそのまま横並び比較しないでください。短いプロンプトでは手動で `./bonsai-gpu-rocm ... -p "Hello" -n 16 -t 0` を実行し、stderr の **`--- throughput ---`** を参照してください。

### よくあるトラブル（ROCm 版）

**`GPU_ARCH` が空 / ビルド失敗:** `rocminfo` で GPU が見えるか確認し、`make GPU_ARCH=gfx1100 build` を試してください。**C++ headers not found:** `sudo apt install -y g++ libstdc++-dev`。**ROCm のパス:** `make ROCM=/opt/rocm build` で上書きできます。**`log.push` でモデルがない:** 親ディレクトリで **`make model`** を先に実行してください。

### ソースを読む場合

10. `bonsai-8b/gpu-rocm/main.c` — ホスト側（`generate`・**`write_benchmark_log`**）  
11. `bonsai-8b/gpu-rocm/kernels.hip` — HIP カーネル・VRAM 管理・**`gpu_get_device_desc`**  
12. `bonsai-8b/gpu-rocm/gpu.h` — C API（**`gpu_get_device_desc`** 追加）  
