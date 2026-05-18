# 設計仕様書

> **注意**: 本ドキュメントは設計仕様書です。変更履歴は `doc/ChangeLog` を参照。以下は**現在の**システム設計である。

## 概要

### リポジトリの目的とスコープ

本リポジトリは **[PrismML](https://prismml.com/) の 1-bit Bonsai 8B** を **GGUF**（既定: **`Bonsai-8B-Q1_0.gguf`**、**Q1_0** 量子化）から、**C ソース**（基準は **`bonsai-8b/cpu/main.c`**、任意で OpenMP 版 **`bonsai-8b/cpu-omp/main.c`**）で **テキスト生成推論**する実装である。

**PyTorch・TensorFlow・JAX・ONNX Runtime 等の ML ユーザランドにはリンクしない。** 基準経路のコアは **標準 C と `libm`** のみ。**GPU（ROCm/HIP）・AMD XDNA2 NPU** 向けの別 `main.c` は本リポジトリから**削除済み**。

**CPU** については、**参照実装**として **`bonsai-8b/cpu/main.c`（単スレッド）** を保守し、検証・スループット改善用に **同ロジックを OpenMP で並列化した `bonsai-8b/cpu-omp/main.c`**（**`bonsai-cpu-omp`**、`libm` + **OpenMP ランタイム**）を別ディレクトリで提供する。

目的は、推論経路（GGUF 読み取り・量子化復元・Transformer forward・サンプリング）を **C の明示的なコードパス**として追い、改変しやすくすることである。学習・商用 SLA・公式実装との数値一致はスコープ外。

#### ライブラリ非依存の意義

- **理解可能性**: バッファ配置・演算順・量子化レイアウトをソースで追える。
- **依存の単純化**: コンパイラと最小実行環境で再現できる。
- **参照実装**: Bonsai 8B（GGUF）デコーダの最小例として使える。

利用者向けの手順は **`README.md`** / **`README.en.md`** を参照。

### 実装バリアント（現状）

| ソース | 実行環境 | 概要 |
|--------|----------|------|
| `bonsai-8b/cpu/main.c` | **CPU、単スレッド** | GGUF mmap。線形層・埋め込みは **Q1_0**（グループ **g128**）などをブロック単位で部分復元しつつ GEMV。Norm 等は F32。`libm` のみ。 |
| `bonsai-8b/cpu-omp/main.c` | **CPU、OpenMP マルチスレッド** | 上記と同一 GGUF・CLI。行列積・ヘッド単位アテンション・SwiGLU／残差など主要ループを **OpenMP** で並列化。`libm` と **OpenMP ランタイム**。 |

メタデータの照合は **GGUF 内の `qwen3.*` プレフィックス**（実装上のバイト列。`embedding_length` 等）と tokenizer 系キーで行う。実装コメントは「dense デコーダ」「Bonsai」と整合する。

## ディレクトリとファイル構成

| パス | 役割 |
|------|------|
| `README.md` / `README.en.md` | ビルド・実行・方針（日／英）。 |
| `bonsai-8b/Makefile` | **`build.cpu`** / **`run.cpu`** / **`clean`** を `cpu/Makefile` に委譲。**`cpu-omp` は未集約**（サブディレクトリの Makefile を直接使用）。 |
| `bonsai-8b/cpu/Makefile` | `bonsai-cpu` の生成。`MODEL` 既定は **`Bonsai-8B-Q1_0.gguf`**。 |
| `bonsai-8b/cpu-omp/Makefile` | **`bonsai-cpu-omp`** の生成（`-fopenmp`）。`MODEL` 既定は **`../Bonsai-8B-Q1_0.gguf`**。 |
| `bonsai-8b/cpu/main.c` | CPU 単スレッド推論の**参照ソース**（アルゴリズムの正として追う）。 |
| `bonsai-8b/cpu-omp/main.c` | **`cpu/main.c` をベースに OpenMP を付与した派生**（挙動確認はまず `cpu` を正とする）。 |
| `bonsai-8b/gguf.txt` | 既定 GGUF の Hugging Face URL。 |
| `bonsai-8b/hf-model.py` | （任意）`hf` CLI でダウンロード＋Hub LFS 検証。 |
| `doc/design.md` | 本書。 |
| `doc/ChangeLog` | 変更履歴。 |
| `.gitignore` | ビルド生成物（**`bonsai-8b/cpu/bonsai-cpu`**、**`bonsai-8b/cpu-omp/bonsai-cpu-omp`** 等）、`*.gguf` 等。 |

### Make ターゲット（`bonsai-8b/Makefile`）

| ターゲット | 出力 | ソース |
|------------|------|--------|
| `build.cpu` / `run.cpu` | `cpu/bonsai-cpu` | `cpu/main.c` |

**`cpu-omp/`** はルート Makefile からは呼ばず、次のようにサブディレクトリでビルドする。

| 場所 | ターゲット | 出力 | ソース |
|------|------------|------|--------|
| `bonsai-8b/cpu-omp/Makefile` | `build` / `run` / `clean` | `cpu-omp/bonsai-cpu-omp` | `cpu-omp/main.c` |

**`bonsai-8b/Makefile`** 用変数:

| 変数 | 意味 | 既定例 |
|------|------|--------|
| `MODEL` | GGUF パス | `Bonsai-8B-Q1_0.gguf` |
| `PROMPT` | 実行用プロンプト文字列 | `Hello, how are you?` |

**`bonsai-8b/cpu-omp/Makefile`** にも `MODEL` / `PROMPT` があり、`MODEL` の既定は **`../Bonsai-8B-Q1_0.gguf`**（`cpu-omp` からの相対パス）。

```bash
cd bonsai-8b
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

## 実行時の挙動（CPU）

重みは **mmap した GGUF** 上の量子化 blob を参照する。行列積は出力行／ブロックごとに **Q1_0** をスタック等で **float に部分復元**してから内積する（全テンソルの float 一括展開はしない）。KV・活性は主に float。サンプリングはホスト上の logits に対して行う。

**`cpu-omp`** は上記アルゴリズムは同じで、GEMV／RMSNorm の一部／マルチヘッドアテンション・活性結合などに **OpenMP parallel for** を入れたものである（細部は `cpu-omp/main.c` を参照）。

## コマンドラインオプション

| オプション | 説明 | デフォルト（実装参照） |
|-----------|------|------------------------|
| `-p` | プロンプト | `Hello` |
| `-n` | 最大生成トークン数 | 実装既定値に従う |
| `-t` | Temperature | 実装既定値に従う |
| `-k` | Top-p | 実装既定値に従う |
| `-s` | 乱数シード | 実装既定値に従う |
| `-l` | 最大シーケンス長 | 実装既定値に従う |

## アーキテクチャ（`cpu/main.c`）

**`cpu-omp/main.c`** もモジュール分割・データ構造・推論フェーズは同じで、ホットパスに **OpenMP** を挟んだ派生として読む。

### レイヤー構成（概略）

1. **GGUF / GGML dtype**: 必要な tensor dtype（**Q1_0** 等）とブロック構造を定義。
2. **モデル構造体**: `Config`、`TensorInfo`、トークナイザ、重み参照、`State`（KV・中間バッファ）。
3. **ロード**: mmap、metadata（`qwen3.*`）、tensor テーブル、tokenizer。
4. **推論**: 1 トークンずつ forward、プロンプト区間は teacher forcing、生成区間はサンプリング。

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

**Q1_0** は **QK1_0=128** 要素を単位とする。CPU パスは行／ブロック単位で `BlockQ1_0` を float に戻し、入力ベクトルとの内積に使用する。

### RoPE

llama.cpp 系の NeoX 半分ペア配置に合わせ、YARN 等のメタがあれば `rope_scaling` 関連フィールドで処理（詳細はソース内 `rope_apply` / `finalize_rope_scaling`）。

### トークナイザ・チャット

GPT-2 系 BPE と特殊トークン。ChatML 形式へのエンコードは実装の `chat_encode` に従う。

### 生成ループとサンプリング

温度・top-p・greedy 等の分岐は CPU 上で logits に対して実施。乱数は実装の xorshift 系 state を使用。

## モデル参照

- 既定ファイル名: **`bonsai-8b/Makefile` の `MODEL`**（既定 **`Bonsai-8B-Q1_0.gguf`**）。
- 取得 URL: **`bonsai-8b/gguf.txt`**。
- 手動検証: Hugging Face 上のハッシュと `sha256sum` 等で照合。

## 制約・既知の制限

- **8B を CPU で動かすため重い**場合がある。単スレッドは特に遅くなりがち（参考実装・検証向け）。**`cpu-omp`** でもコア数に依存し軽快になるとは限らない。
- **画像・マルチモーダル入力は非対応**（テキストデコーダのみ）。
- **コンテキスト長**を大きくすると KV 用メモリが増える。
- 商用水平の性能・公式実装との一致は保証しない。

## 補足：ドキュメント間の役割

- **`README.md` / `README.en.md`**: 入口（手順・方針）。
- **`doc/design.md`（本書）**: 設計・仕様の静的説明。
- **`doc/ChangeLog`**: 履歴。

実装の最終的な挙動は **`bonsai-8b/cpu/main.c`** のソースを正とする。**`cpu-omp`** は派生であり、浮動小数の結合順などで数値差が出うる。

## 補足：`design.md` 更新時のチェックリスト

1. **`bonsai-8b/Makefile`** / **`cpu/Makefile`** / **`cpu-omp/Makefile`** と矛盾がないか。
2. **`README.md` と `README.en.md`** の手順と整合するか。
3. 仕様変更は **`doc/ChangeLog`** に記録する。
