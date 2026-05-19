# OpenBLAS + OpenMP。ヘッダが見つからない場合は CPPFLAGS で追加してください。
# 例（Debian/Ubuntu の pthread ビルド）:
#   make build CPPFLAGS=-I/usr/include/x86_64-linux-gnu/openblas-pthread
# BLAS が独自スレッドを持つため OpenMP と二重並列化しないよう、実行時は例えば:
#   OPENBLAS_NUM_THREADS=1 または OMP_NUM_THREADS と調整してください。

CC      ?= cc
# -march=native: 実行ホストの SIMD 命令を有効化（Q1_0 内積カーネルの自動ベクトル化に重要）
# -funroll-loops: 8-bit ごとの内側ループを展開
# -ffast-math: FP 加算の結合則を許可（縮小総和を SIMD で並列化させる。LLM 推論用途では十分許容範囲）
CFLAGS  ?= -O3 -std=c11 -Wall -Wextra -Wno-unused-parameter -fopenmp \
           -march=native -funroll-loops -ffast-math -mfma
LDFLAGS ?= -fopenmp -lopenblas -lm

MODEL   ?= ../Bonsai-8B-Q1_0.gguf
PROMPT  ?= Hello, how are you?

OPENBLAS_CFLAGS := $(shell pkg-config --cflags openblas 2>/dev/null)
OPENBLAS_LIBS   := $(shell pkg-config --libs openblas 2>/dev/null)
ifneq ($(strip $(OPENBLAS_LIBS)),)
  CPPFLAGS += $(OPENBLAS_CFLAGS)
  LDFLAGS := $(filter-out -lopenblas,$(LDFLAGS))
  LDFLAGS += $(OPENBLAS_LIBS)
endif

.PHONY: build run clean

build:
	$(CC) $(CPPFLAGS) $(CFLAGS) -o bonsai-cpu-blas main.c $(LDFLAGS)

run: build
	./bonsai-cpu-blas "$(MODEL)" -p "$(PROMPT)"

clean:
	rm -f bonsai-cpu-blas
