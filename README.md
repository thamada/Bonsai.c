# Bonsai.c

英語版は [README.en.md](README.en.md) を参照してください。

本リポジトリは、[PrismML](https://prismml.com/) の **1-bit Bonsai 8B** を GGUF（`Bonsai-8B-Q1_0`）から、**ライブラリに依存せず単一の C言語ソースで直接動かす推論実装**です。

### 1-bit Bonsai 8B について

[Announcing 1-bit Bonsai: The First Commercially Viable 1-bit LLMs](https://prismml.com/news/bonsai-8b)（PrismML, 2026）では、**1-bit Bonsai 8B** が埋め込み・アテンション・MLP・言語モデルヘッドに至るまで **ネットワーク全体を 1-bit で設計**し、高位精度への「逃げ道」を置かない **真の 1-bit モデル**（約 82 億パラメータ）であることが説明されています。公開ウェイトは **Apache License 2.0** です。エッジからクラウドまで **知性の密度（intelligence density）** と実用的なスループット・省エネを両立させる、というビジョンと、モデルサイズの小ささ（記事では **約 1.15 GB**）が強調されています。

**本リポジトリが読み込む GGUF** は **`Bonsai-8B-Q1_0.gguf`**（Q1_0 量子化）です。**テキストのプロンプト入出力**に限定し、画像入力は扱いません。

**PyTorch・TensorFlow・JAX・ONNX Runtime など、機械学習向けのユーザランドライブラリ／ランタイムは一切リンクしていません。** 推論は **標準Cと `libm`** のみで、`bonsai-8b/cpu/main.c` から **CPU 単スレッド**の実行ファイル（`bonsai-cpu`）をビルドします。Python や `torch` には依存しません。

### なぜライブラリ非依存なのか

一般的な LLM推論では、**計算手順、メモリ配置、量子化レイアウト**といった低レベルな詳細がフレームワーク内部に隠れがちです。

本リポジトリでは **GGUF の読み取り、重みの復元、行列演算、Transformer の forward、サンプリングまでを C のコードパスとして明示**します。目的は PyTorch の代替ではなく、推論処理を**観察・検証・改造**しやすくすることです。

- **理解可能性**: ソースと `doc/design.md` から経路を追える  
- **依存の単純化**: 基本的な C ツールチェーンで動かせる  
- **実験の自由度**: 量子化やメモリ表現などを個別に試しやすい  
- **参照実装**: **Bonsai 8B（GGUF）** のデコーダ推論の最小例として使える  

最高性能や機能網羅が目的ではありません。

## まず何ができるのか

現状、**CPU 単スレッド**のビルドのみを提供します。

| 実行方法 | 使うファイル | 作られる実行ファイル | 向いている用途 |
|---|---|---|---|
| CPU 単スレッド | `bonsai-8b/cpu/main.c` | `cpu/bonsai-cpu` | 仕組みを追う、最小構成で動かす |

8B 級モデルの CPU 単スレッド実行は **非常に重い**です。まずは `-n 1` など短い生成で動作確認してください。

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
    └── cpu/
        ├── Makefile
        └── main.c
```

主に触るのは **`bonsai-8b/cpu/`** です。

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
- **Bonsai-8B-Q1_0.gguf**（[prism-ml/Bonsai-8B-gguf](https://huggingface.co/prism-ml/Bonsai-8B-gguf)）

```bash
sudo apt update
sudo apt install -y build-essential make
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
├── cpu/ … （`main.c` → `cpu/bonsai-cpu`）
└── Bonsai-8B-Q1_0.gguf
```

整合性は [Hugging Face](https://huggingface.co/prism-ml/Bonsai-8B-gguf/tree/main) のハッシュと照合してください。

## いちばん簡単な実行手順

```bash
cd bonsai-8b
make build.cpu
./cpu/bonsai-cpu Bonsai-8B-Q1_0.gguf -p "Hello" -n 1
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

削除されるのは主に **`cpu/bonsai-cpu`** です。GGUF は削除されません。

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

正常です。8B を単スレッド CPU で回す負荷は大きいです。`-n 1` や `-n 4` から試してください。

```bash
./cpu/bonsai-cpu Bonsai-8B-Q1_0.gguf -p "Hello" -n 1
```

## 実装を読みたい人へ

1. `README.md` / `README.en.md`  
2. `doc/design.md`  
3. `bonsai-8b/cpu/main.c`  

## このリポジトリで扱わないもの

- 学習・ファインチューニング  
- **GPU / NPU / OpenMP** などの別実行経路（本リポジトリから削除済み）  
- バッチ推論の最適化  
- 画像入力  
- サーバ化・Web API 化  
- すべての GGUF 量子化形式への汎用対応  
- 公式実装との完全な数値一致保証  

目的は、**Bonsai-8B-Q1_0**（GGUF）のテキスト推論を **C** で理解し、実験し、必要に応じて改造できるようにすることです。

## 詳細ドキュメント

- `doc/design.md`  
- `doc/ChangeLog`  

困ったときは、`bonsai-8b/Makefile` のターゲットと、実行時に渡しているモデルパスを確認してください。
