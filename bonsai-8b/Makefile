MODEL   ?= Bonsai-8B-Q1_0.gguf
PROMPT  ?= Hello, how are you?

# 各ビルドはサブディレクトリの Makefile の既定（CC / CFLAGS 等）を使う。
# 上書きは例: make -C cpu build CC=clang  または  make build.cpu CC=clang

.PHONY: \
	all \
	build.cpu run.cpu \
	build.cpu-blas run.cpu-blas \
	clean

.DEFAULT_GOAL := all

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
