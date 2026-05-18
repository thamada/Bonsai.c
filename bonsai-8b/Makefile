MODEL   ?= Bonsai-8B-Q1_0.gguf
PROMPT  ?= Hello, how are you?

# 各ビルドはサブディレクトリの Makefile の既定（CC / CFLAGS 等）を使う。
# 上書きは例: make -C cpu build CC=clang  または  make build.cpu CC=clang

.PHONY: \
	all \
	build.cpu run.cpu \
	clean

.DEFAULT_GOAL := all

# -----------------------------------------------------------------------------
# CPU reference（ソースは cpu/main.c）
build.cpu:
	cd cpu && make build

run.cpu: build.cpu
	./cpu/bonsai-cpu "$(MODEL)" -p "$(PROMPT)"

# -----------------------------------------------------------------------------
all: build.cpu

# -----------------------------------------------------------------------------
clean:
	cd cpu && make clean
