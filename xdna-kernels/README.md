# XDNA GEMV 制御コード（`bf16-gemv-<n>x<d>.bin`）

`qwen3-xdna2` は **`XDNA_GEMV_DIR`** から **生の ctrlcode**（ERT が実行する txn / opcode 列）を読み込みます。

## このディレクトリに含まれるファイル

**64 バイトのプレースホルダ**です。先頭マジック `GQF3XDNA` で識別され、**実機 NPU にはディスパッチされません**（`load_gemv_kernel` が拒否し CPU 経路へフォールバック）。用途は次のとおりです。

- リポジトリ内で **期待ファイル名・ディレクトリレイアウト**を固定する
- `./qwen3-xdna2 model.gguf --xdna-status` で **`[STUB]`** 表示を確認する

実用的な NPU 加速には、**MLIR-AIE / AMD IRON / Peano** 等で各 `(n,d)` ごとにビルドした **本物の ctrlcode** に差し替えてください（[amdnpu.rst 概要](https://github.com/amd/xdna-driver/blob/main/src/driver/doc/amdnpu.rst)）。

## Qwen3-VL-8B テキスト経路で参照される形状（ファイル名）

| ファイル | 役割 |
|---------|------|
| `bf16-gemv-4096x4096.bin` | wq（Q）、wo（出力投影） |
| `bf16-gemv-4096x1024.bin` | wk / wv |
| `bf16-gemv-4096x12288.bin` | FFN gate / up |
| `bf16-gemv-12288x4096.bin` | FFN down |
| `bf16-gemv-4096x151936.bin` | lm_head |

モデルの `dim` / `hidden_dim` / `vocab_size` が異なる場合は、`--xdna-status` に表示される形状に合わせて追加の `.bin` が必要です。

## プレースホルダの再生成

リポジトリルートで:

```bash
python3 tools/gen-xdna-gemv-stubs.py
# または出力先を指定:
python3 tools/gen-xdna-gemv-stubs.py /path/to/dir
```

## 実行例

```bash
export XDNA_GEMV_DIR="$PWD/xdna-kernels"
./qwen3-xdna2 model.gguf --xdna-status
```

`/dev/accel` とファームウェアが正常な環境では、本物の ctrlcode に差し替えたうえで **`[ OK ]`** と NPU GEMV カウントが増えることを確認してください。
