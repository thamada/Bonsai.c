MODEL   ?= Bonsai-8B-Q1_0.gguf
PROMPT  ?= Hello, how are you?

# 各ビルドはサブディレクトリの Makefile の既定（CC / CFLAGS 等）を使う。
# 上書きは例: make -C cpu build CC=clang  または  make build.cpu CC=clang

.PHONY: \
	all \
	model \
	build.cpu run.cpu \
	build.cpu-blas run.cpu-blas \
	clean

.DEFAULT_GOAL := all

# -----------------------------------------------------------------------------
# GGUF モデル（gguf.txt の URL からダウンロード）
model:
	@if [ ! -f "$(MODEL).sha256sum" ]; then \
		echo "エラー: チェックサムファイル '$(MODEL).sha256sum' が見つかりません。"; \
		echo "リポジトリを最新版に更新するか、ファイルが存在するか確認してください。"; \
		exit 1; \
	fi; \
	url=$$(sed 's|/blob/main/|/resolve/main/|' gguf.txt); \
	wget -O "$(MODEL)" "$$url" && \
	if sha256sum --check "$(MODEL).sha256sum"; then \
		echo ""; \
		echo "チェックサム検証に成功しました。"; \
		echo ""; \
		echo "  ファイル: $(MODEL)"; \
		echo "  参照:     $(MODEL).sha256sum"; \
		echo ""; \
		echo "モデルのダウンロードと検証が完了しました。"; \
		echo "  make run.cpu"; \
		echo "などで推論を実行できます。"; \
	else \
		echo ""; \
		echo "エラー: ダウンロードしたモデルファイルのチェックサム検証に失敗しました。"; \
		echo ""; \
		echo "  ファイル: $(MODEL)"; \
		echo "  参照:     $(MODEL).sha256sum"; \
		echo ""; \
		echo "ダウンロード中にファイルが破損した可能性があります。"; \
		echo "破損したファイルを削除します。ネットワーク接続を確認してから"; \
		echo "  make model"; \
		echo "を再実行してください。"; \
		rm -f "$(MODEL)"; \
		exit 1; \
	fi

# -----------------------------------------------------------------------------
# CPU reference（ソースは cpu/main.c）
build.cpu:
	cd cpu && make build

run.cpu: build.cpu
	./cpu/bonsai-cpu "$(MODEL)" -p "$(PROMPT)"

# -----------------------------------------------------------------------------
# CPU + OpenMP + OpenBLAS（ソースは cpu-blas/main.c）
build.cpu-blas:
	cd cpu-blas && make build

run.cpu-blas: build.cpu-blas
	./cpu-blas/bonsai-cpu-blas "$(MODEL)" -p "$(PROMPT)"

# -----------------------------------------------------------------------------
all: build.cpu

# -----------------------------------------------------------------------------
clean:
	cd cpu && make clean
	cd cpu-blas && make clean
