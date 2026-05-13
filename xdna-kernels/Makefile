# SPDX-License-Identifier: Apache-2.0
# Download bf16-gemv-<n>x<d>.bin (real NPU ctrlcode), not GQF3XDNA stubs.
#
# 配布元 URL の末尾は必ず「/」で終わらせ、そこに次のファイル名がそのまま置いてあること。
#   bf16-gemv-4096x4096.bin
#   bf16-gemv-4096x1024.bin
#   bf16-gemv-4096x12288.bin
#   bf16-gemv-12288x4096.bin
#   bf16-gemv-4096x151936.bin
#
# 使用例:
#   make XDNA_GEMV_BIN_URL_BASE='https://example.com/path/to/gemv/'
#
# 環境変数でも可:
#   export XDNA_GEMV_BIN_URL_BASE='https://example.com/path/to/gemv/'
#   make

.DEFAULT_GOAL := all

# tools/gen-xdna-gemv-stubs.py の SHAPES と main-xdna2.c のテキスト経路を一致させること
GEMV_BINARIES := \
	bf16-gemv-4096x4096.bin \
	bf16-gemv-4096x1024.bin \
	bf16-gemv-4096x12288.bin \
	bf16-gemv-12288x4096.bin \
	bf16-gemv-4096x151936.bin

CURL  := $(shell command -v curl 2>/dev/null)
WGET  := $(shell command -v wget 2>/dev/null)

.PHONY: all help check-url

all: check-url $(GEMV_BINARIES)

help:
	@echo '本物の ctrlcode (.bin) をまとめてダウンロードします。'
	@echo ''
	@echo '  make XDNA_GEMV_BIN_URL_BASE=https://配布元のベースURL/'
	@echo ''
	@echo 'ベース URL は末尾が / で、上記 5 ファイルが PUT されている場所を指してください。'
	@echo 'リポジトリ単体には公開ミラーが無い場合があります（ツールチェイン生成物をホストする必要があることがあります）。'
	@echo ''
	@echo '依存: curl または wget'

check-url:
	@if [ -z "$(strip $(XDNA_GEMV_BIN_URL_BASE))" ]; then \
		echo >&2 'エラー: XDNA_GEMV_BIN_URL_BASE が未設定です。'; \
		echo >&2 '  make XDNA_GEMV_BIN_URL_BASE=https://…/ … のように指定してください。'; \
		echo >&2 '  詳細: make -f Makefile help'; \
		exit 1; \
	fi
	@if [ -z "$(CURL)" ] && [ -z "$(WGET)" ]; then \
		echo >&2 'エラー: curl または wget が見つかりません。'; \
		exit 1; \
	fi

$(GEMV_BINARIES): check-url
	@echo "Downloading $@ ..."
	@BASE="$(XDNA_GEMV_BIN_URL_BASE)"; \
	case "$$BASE" in */) ;; *) BASE="$$BASE/" ;; esac; \
	if [ -n "$(CURL)" ]; then \
		$(CURL) -fSL --retry 3 --retry-delay 2 \
			-o "$@" "$$BASE$@"; \
	elif [ -n "$(WGET)" ]; then \
		$(WGET) -O "$@" "$$BASE$@" || exit $$?; \
	fi
	@SZ=$$(wc -c < "$@" | tr -d ' '); \
	if [ "$$SZ" -le 128 ]; then \
		echo >&2 "警告: $@ のサイズが $$SZ バイトのみです。スタブかダウンロード失敗の可能性があります。"; \
	fi
