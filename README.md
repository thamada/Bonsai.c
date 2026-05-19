# Bonsai.c

英語版は [README.en.md](README.en.md) を参照してください。

本リポジトリは、[PrismML](https://prismml.com/) の **1-bit Bonsai 8B** を GGUF（`Bonsai-8B-Q1_0`）から、**ライブラリに依存せず単一の C言語ソースで直接動かす推論実装**です。

### 1-bit Bonsai 8B について

[Announcing 1-bit Bonsai: The First Commercially Viable 1-bit LLMs](https://prismml.com/news/bonsai-8b)（PrismML, 2026）では、**1-bit Bonsai 8B** が埋め込み・アテンション・MLP・言語モデルヘッドに至るまで **ネットワーク全体を 1-bit で設計**し、高位精度への「逃げ道」を置かない **真の 1-bit モデル**（約 82 億パラメータ）であることが説明されています。公開ウェイトは **Apache License 2.0** です。エッジからクラウドまで **知性の密度（intelligence density）** と実用的なスループット・省エネを両立させる、というビジョンと、モデルサイズの小ささ（記事では **約 1.15 GB**）が強調されています。

**本リポジトリが読み込む GGUF** は **`Bonsai-8B-Q1_0.gguf`**（Q1_0 量子化）です。**テキストのプロンプト入出力**に限定し、画像入力は扱いません。

**PyTorch・TensorFlow・JAX・ONNX Runtime など、機械学習向けのユーザランドライブラリ／ランタイムは一切リンクしていません。**  
推論の基準経路は **標準Cと `libm`** のみで、`bonsai-8b/cpu/main.c` から **CPU 単スレッド**の実行ファイル（`bonsai-cpu`）をビルドします。  
より速い検証向けに、同じ GGUF に対応した **OpenMP マルチスレッド**版を `bonsai-8b/cpu-omp/main.c` から **`bonsai-cpu-omp`** として別ビルドできます（ランタイムは **標準C + `libm` + OpenMP ランタイム**）。  
さらに **`bonsai-8b/cpu-blas/`** では **OpenMP + OpenBLAS** と **Q1_0 専用の融合内積カーネル**で **`bonsai-cpu-blas`** をビルドできます（**標準C + `libm` + OpenMP + OpenBLAS**）。Python や `torch` には依存しません。

### なぜライブラリ非依存なのか

一般的な LLM推論では、**計算手順、メモリ配置、量子化レイアウト**といった低レベルな詳細がフレームワーク内部に隠れがちです。

本リポジトリでは **GGUF の読み取り、重みの復元、行列演算、Transformer の forward、サンプリングまでを C のコードパスとして明示**します。目的は PyTorch の代替ではなく、推論処理を**観察・検証・改造**しやすくすることです。

- **理解可能性**: ソースと `doc/design.md` から経路を追える  
- **依存の単純化**: 基本的な C ツールチェーンで動かせる  
- **実験の自由度**: 量子化やメモリ表現などを個別に試しやすい  
- **参照実装**: **Bonsai 8B（GGUF）** のデコーダ推論の最小例として使える  

最高性能や機能網羅が目的ではありません。

## まず何ができるのか

**CPU 単スレッド**版、その **OpenMP** 並列版、および **OpenMP + OpenBLAS** 最適化版の 3 通りがあります。

| 実行方法 | 使うファイル | 作られる実行ファイル | 向いている用途 |
|---|---|---|---|
| CPU 単スレッド | `bonsai-8b/cpu/main.c` | `cpu/bonsai-cpu` | 仕組みを追う、最小依存で動かす |
| CPU + OpenMP | `bonsai-8b/cpu-omp/main.c` | `cpu-omp/bonsai-cpu-omp` | マルチコアでの試運転（参照用） |
| CPU + OpenMP + OpenBLAS | `bonsai-8b/cpu-blas/main.c` | `cpu-blas/bonsai-cpu-blas` | マルチコアでの実用的なスループット（推奨） |

8B 級モデルの CPU 実行は依然として **重い**です。まずは `-n 1` など短い生成で動作確認し、本番試行は **`cpu-blas`** を推奨します（下記 [参考ベンチマーク](#参考ベンチマーク)）。

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

推論コードは **`bonsai-8b/cpu/`**（参照・単スレッド）が基準です。並列版は **`bonsai-8b/cpu-omp/`**、最適化版は **`bonsai-8b/cpu-blas/`** です。

## 初心者向け: LLM推論で何が起きるか

1. **GGUF を読む**  
2. **プロンプトをトークン化する**  
3. **Transformer を 1 トークンずつ実行する**  
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
- **Bonsai-8B-Q1_0.gguf**（[prism-ml/Bonsai-8B-gguf](https://huggingface.co/prism-ml/Bonsai-8B-gguf)）

```bash
sudo apt update
sudo apt install -y build-essential make
# cpu-blas を使う場合:
# sudo apt install -y libopenblas-dev
```

## モデルファイルを置く

`bonsai-8b/Makefile` の既定は **`MODEL=Bonsai-8B-Q1_0.gguf`** です。別パスを使うときは実行時に渡すか `make run.cpu MODEL=...` で指定してください。

モデルはリポジトリに含めません。`bonsai-8b/gguf.txt` の URL からダウンロードし、`bonsai-8b/` 直下に置きます。

```bash
cd bonsai-8b
url=$(sed 's|/blob/main/|/resolve/main/|' gguf.txt)
wget -O Bonsai-8B-Q1_0.gguf "$url"
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

整合性は [Hugging Face](https://huggingface.co/prism-ml/Bonsai-8B-gguf/tree/main) のハッシュと照合してください。

## いちばん簡単な実行手順

```bash
cd bonsai-8b
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

以下は **開発環境での参考値**です。CPU・メモリ・ビルドフラグ・ページキャッシュの状態で大きく変わります。再現時は同じ GGUF・同じコマンドで計測してください。

| 項目 | 値 |
|---|---|
| 日付 | 2026-05-19 |
| CPU | AMD Ryzen AI 5 340（12 論理コア） |
| OS | Linux |
| モデル | `Bonsai-8B-Q1_0.gguf`（ページキャッシュ温） |
| コマンド | `./<binary> Bonsai-8B-Q1_0.gguf -p "Hello" -n 16 -t 0` |
| 計測 | プロンプト 18 トークン + **生成 16 トークン**（バイナリ表示の集計） |
| 環境変数 | `cpu-omp` / `cpu-blas`: `OMP_NUM_THREADS=12`、`cpu-blas` のみ `OPENBLAS_NUM_THREADS=1` |
| 再現 | 各バイナリで 1 回ウォームアップ後、3 回計測の**最短**（GGUF は事前にページキャッシュ温） |

| バイナリ | 合計時間 | 生成スループット | 備考 |
|---|---:|---:|---|
| `cpu/bonsai-cpu` | 132.0 s | **0.12 tok/s** | 単スレッド、`-O3`（`cpu/Makefile` 既定） |
| `cpu-omp/bonsai-cpu-omp` | 21.5 s | **0.74 tok/s** | `-O3 -fopenmp`（`cpu-omp/Makefile` 既定） |
| `cpu-blas/bonsai-cpu-blas` | 2.6 s | **6.15 tok/s** | `-O3 -fopenmp -march=native -ffast-math -mfma`、OpenBLAS 1 スレッド |

この条件下では **`cpu-blas` が `cpu-omp` の約 8 倍**、**`cpu` の約 51 倍**の生成スループットでした（`cpu-omp` は `cpu` の約 6 倍）。生成テキスト（`Hello! I'm Bonsai, an AI assistant developed by PrismML.`）は 3 バイナリで一致しています。`-ffast-math` により浮動小数の結合順は `cpu-omp` と異なり得ますが、上記試行では同一出力でした。

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

モデルパスを確認してください。

```bash
ls -lh bonsai-8b/Bonsai-8B-Q1_0.gguf
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
3. `bonsai-8b/cpu/main.c` — 単スレッド基準経路  
4. `bonsai-8b/cpu-omp/main.c` — OpenMP 並列版  
5. `bonsai-8b/cpu-blas/main.c` — OpenMP + OpenBLAS + Q1_0 融合カーネル  

## このリポジトリで扱わないもの

- 学習・ファインチューニング  
- **GPU / NPU** や **CUDA / Vulkan / Metal** など GPU ランタイム向けビルド  
- バッチ推論の最適化  
- 画像入力  
- サーバ化・Web API 化  
- すべての GGUF 量子化形式への汎用対応  
- 公式実装との完全な数値一致保証  

目的は、**Bonsai-8B-Q1_0**（GGUF）のテキスト推論を **C** で理解し、実験し、必要に応じて改造できるようにすることです。

## 詳細ドキュメント

- `doc/design.md`  
- `doc/ChangeLog`  

困ったときは、`bonsai-8b/Makefile`（`build.cpu` / `build.cpu-blas` / `run.cpu-blas` など）と各サブディレクトリの Makefile、および実行時のモデルパスを確認してください。
