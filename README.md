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

推論コードは **`bonsai-8b/cpu/`**（参照・単スレッド）が基準です。並列版は **`bonsai-8b/cpu-omp/`**、CPU 最適化版は **`bonsai-8b/cpu-blas/`** です。

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
- **`wget`**（`make model` で GGUF を取得するとき）  
- **Bonsai-8B-Q1_0.gguf**（[prism-ml/Bonsai-8B-gguf](https://huggingface.co/prism-ml/Bonsai-8B-gguf)。`make model` で取得）

```bash
sudo apt update
sudo apt install -y build-essential make
# cpu-blas を使う場合:
# sudo apt install -y libopenblas-dev
```

## モデルファイルを置く

`bonsai-8b/Makefile` の既定は **`MODEL=Bonsai-8B-Q1_0.gguf`** です。別パスを使うときは実行時に渡すか `make run.cpu MODEL=...` で指定してください。

GGUF 本体はリポジトリに含めません。`bonsai-8b/gguf.txt` の URL から **`make model`** でダウンロードし、`bonsai-8b/` 直下に置きます。ダウンロード後は **`Bonsai-8B-Q1_0.gguf.sha256sum`** で SHA256 を自動検証します（失敗時は破損ファイルを削除）。

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

整合性は **`make model`** が **`Bonsai-8B-Q1_0.gguf.sha256sum`** で検証します。[Hugging Face](https://huggingface.co/prism-ml/Bonsai-8B-gguf/tree/main) 上のハッシュとも一致するはずです。

## いちばん簡単な実行手順

```bash
cd bonsai-8b
make model
make build.cpu
./cpu/bonsai-cpu Bonsai-8B-Q1_0.gguf -p "Hello" -n 1
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
cd bonsai-8b
make build.cpu
```

成功すると **`cpu/bonsai-cpu`** ができます。

### 実行

```bash
./cpu/bonsai-cpu Bonsai-8B-Q1_0.gguf -p "日本語で短く自己紹介してください。" -n 16
```

`Makefile` の **`run.cpu`**:

```bash
make run.cpu PROMPT="日本語で短く自己紹介してください。"
```

別ディレクトリのモデル:

```bash
make run.cpu MODEL=/data/models/Bonsai-8B-Q1_0.gguf PROMPT="Hello"
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

成功すると **`bonsai-cpu-blas`** がこのディレクトリにできます。`bonsai-8b/Makefile` からは `make build.cpu-blas` / `make run.cpu-blas` でもビルド・実行できます。

### 実行

```bash
cd bonsai-8b/cpu-blas
./bonsai-cpu-blas ../Bonsai-8B-Q1_0.gguf -p "Hello" -n 16
```

```bash
cd bonsai-8b
make run.cpu-blas PROMPT="Hello"
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
| `cpu-blas/bonsai-cpu-blas` | 0.5 s | **30.79 tok/s** | `-O3 -fopenmp -march=native -ffast-math -mfma`、OpenBLAS 1 スレッド |

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

```bash
cd bonsai-8b
make clean
```

ルートで `make clean` すると **`cpu/bonsai-cpu`** と **`cpu-blas/bonsai-cpu-blas`** が削除されます。**`cpu-omp/bonsai-cpu-omp`** を消すときは `bonsai-8b/cpu-omp` で `make clean` してください。GGUF は削除されません。

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
- **ROCm/HIP・AMD NPU** や **Vulkan / Metal** など **GPU 向けその他のランタイム**  
- バッチ推論の最適化  
- 画像入力  
- サーバ化・Web API 化  
- すべての GGUF 量子化形式への汎用対応  
- 公式実装との完全な数値一致保証  

目的は、**Bonsai-8B-Q1_0**（GGUF）のテキスト推論を **C** で理解し、実験し、必要に応じて改造できるようにすることです。

## 詳細ドキュメント

- `doc/design.md`  
- `doc/ChangeLog`  

困ったときは、`bonsai-8b/Makefile`（`model` / `build.cpu` / `build.cpu-blas` / `run.cpu-blas` など）と各サブディレクトリの Makefile、および実行時のモデルパスを確認してください。

---

## NVIDIA CUDA 実装（`gpu-cuda`）について

**`bonsai-8b/gpu-cuda/` は本リポジトリの目的（単一 C ソース・依存最小）から外れた付録**です。`main.c` + `kernels.cu` + `gpu.h` に分かれ、**CUDA Toolkit・NVIDIA ドライバ・GPU 実機**が必要です。参照すべき最小実装は **`cpu/main.c`** です。

同梱している理由は、筆者が **CUDA でどこまで高速化できるか** を試したくなっただけです。本プロジェクトの目的を補うものでも、読者向けの正式な機能でもありません。混乱を招きやすいため、**将来的には別リポジトリへ移す予定**です。初めて読む方は無視して構いません。

以下は、GPU 上の速度比較に興味がある場合の技術メモです。

### ディレクトリ構成

```text
bonsai-8b/gpu-cuda/
├── Makefile
├── main.c
├── kernels.cu
└── gpu.h
```

### 技術的な概要

`cpu-blas` と同じ GGUF・CLI を前提とします。ホスト側（`main.c`）は GGUF の mmap 読み取り、トークナイザ、サンプリングを担当し、GPU 側（`kernels.cu`）は forward 本体を担当します。

#### VRAM 上のデータ配置（`gpu_model_create`）

起動時にホスト（mmap 上の GGUF）から **H2D コピー**で VRAM に載せるもの:

| 種別 | 内容 | 形式 |
|---|---|---|
| 重み | `token_embd`, 各層 `wq/wk/wv/wo/gate/up/down`, `output` | Q1_0（g128） |
| 重み | `attn_norm`, `q_norm`, `k_norm`, `ffn_norm`, `output_norm` | F32 |
| KV キャッシュ | `kc_pack`, `vc_pack`（既定） | **TurboQuant**（PolarQuant 2-bit + QJL 1-bit / 座標、head_dim=128） |
| KV キャッシュ（`--no-tq`） | `kc`, `vc` | F32、`n_layers × max_seq × kv_dim` |
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
6. **KV 書き込み** — 既定では **TurboQuant** で `k`, `v` を圧縮して `kc_pack` / `vc_pack` へ格納（レイアウト: `[layer][kv_head][seq_pos]`、48 B/head）。`--no-tq` 時は F32 へ D2D コピー。
7. **Attention** — `flash_attn_gqa_kernel`（後述）。TurboQuant 時は圧縮 K から QJL 付き内積推定、V は PolarQuant 復元。出力は `xb`。
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

#### Q1_0 GEMV（`gpu_mm` / `gpu_mm_batch`）

`cpu-blas` と同じ方針です。重みは **Q1_0 のまま VRAM 常駐**（事前 dequant しない）。

1. 入力ベクトル（またはバッチ各行）を **`quantize_q8_0_kernel`** で Q8_0（group size 32）に量子化。
2. **`mm_q1_0_kernel`** / **`mm_q1_0_batch_kernel`** で `vec_dot_q1_0_q8_0` を実行 — Q1_0 重み行と Q8_0 活性化の内積。1-bit 重みの符号ビットと Q8_0 の int8 積を組み合わせ、FP16 スケールで復元。
3. 1 Q1_0 ブロック（128 要素）あたり Q8_0 ブロック 4 個と対応。

Decode 版は出力次元 `d` 方向に 256 スレッド/block で並列。Prefill 版は `(n_tokens × d)` 個の出力要素を並列計算します。

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
- Prefill と Decode で KV キャッシュは共有 — Prefill 後の Decode は `pos = n_prompt` から続き、Prefill で填充済みの `kc_pack`/`vc_pack`（または F32 `kc`/`vc`）をそのまま参照します。
- **TurboQuant**（Google Research, [arXiv:2504.19874](https://arxiv.org/abs/2504.19874)）: ランダム直交回転後の Lloyd-Max スカラー量子化（PolarQuant）+ 残差 1-bit QJL。学習不要・推論時オンライン適用。CLI: `--no-tq` で無効化。

### 必要なもの

- **CUDA Toolkit**（**`nvcc`**）と **NVIDIA ドライバ**（**`libcudart`**。cuBLAS 不要）

```bash
sudo apt install -y nvidia-cuda-toolkit   # または NVIDIA 公式 CUDA Toolkit
```

### ビルド

```bash
cd bonsai-8b/gpu-cuda
make build
```

成功すると **`bonsai-gpu-cuda`** がこのディレクトリにできます。`bonsai-8b/Makefile` からは `make build.gpu-cuda` / `make run.gpu-cuda` でもビルド・実行できます。

GPU アーキテクチャに合わせ **`CUDA_GENCODE`** を上書きできます（既定は PTX `compute_86` + ドライバ JIT）:

```bash
make build CUDA_GENCODE=arch=compute_90,code=sm_90
```

### 実行

```bash
cd bonsai-8b/gpu-cuda
./bonsai-gpu-cuda ../Bonsai-8B-Q1_0.gguf -p "Hello" -n 16
```

```bash
cd bonsai-8b
make run.gpu-cuda PROMPT="Hello"
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
| コマンド | `./gpu-cuda/bonsai-gpu-cuda Bonsai-8B-Q1_0.gguf -p "Hello" -n 16 -t 0` |
| ワークロード | CPU 参考表と同じ（prefill 18 + decode 16 トークン） |
| ビルド | `gpu-cuda/Makefile` 既定（PTX `compute_86`、`-use_fast_math`） |

| バイナリ | prefill tok/s | decode 時間 | decode スループット |
|---|---:|---:|---:|
| `gpu-cuda/bonsai-gpu-cuda` | **~294** | 0.32 s | **50.24 tok/s** |

同一プロンプト・`-t 0` で **`cpu-blas` と同じ生成テキスト**（`Hello! I'm Bonsai, an AI assistant developed by PrismML.`）。`-ffast-math` / `-use_fast_math` により浮動小数の結合順は CPU 版と異なり得ますが、上記試行では同一出力でした。

### よくあるトラブル（CUDA 版）

**CUDA / `nvcc` が見つからない:** CUDA Toolkit（**`nvcc`**）と NVIDIA ドライバをインストールし、`gpu-cuda` で `make build` を再実行してください。古い CUDA では `-arch=native` が使えないため、既定は PTX（`compute_86`）+ ドライバ JIT です。GPU に合わせ `CUDA_GENCODE` を指定してください（`gpu-cuda/Makefile` 参照）。プロンプトが `[prompt length … exceeds max_seq …]` で止まる場合は **`-l`** を大きくしてください。

**片付け:** ルートで `make clean` すると **`gpu-cuda/bonsai-gpu-cuda`** も削除されます。

### ソースを読む場合

6. `bonsai-8b/gpu-cuda/main.c` — ホスト側（GGUF・トークナイザ・prefill バッチ / decode 逐次の `generate`・サンプリング）  
7. `bonsai-8b/gpu-cuda/kernels.cu` — CUDA カーネル（`gpu_forward_prefill` / `gpu_forward`、Flash Attention 等）  
8. `bonsai-8b/gpu-cuda/gpu.h` — C API（`gpu_forward_prefill` 等）  
