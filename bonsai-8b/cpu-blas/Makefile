# OpenBLAS + OpenMP。ヘッダが見つからない場合は CPPFLAGS で追加してください。
# 例（Debian/Ubuntu の pthread ビルド）:
#   make build CPPFLAGS=-I/usr/include/x86_64-linux-gnu/openblas-pthread
# BLAS が独自スレッドを持つため OpenMP と二重並列化しないよう、実行時は例えば:
#   OPENBLAS_NUM_THREADS=1 または OMP_NUM_THREADS と調整してください。

CC      ?= cc
# /proc/cpuinfo から SIMD を検出。AVX2 不可なら AVX に落とす（-mfma は FMA 対応 CPU のみ）。
# 上書き例: make build ARCH_FLAGS='-mavx2 -mfma'
BASE_CFLAGS := -O3 -std=c11 -Wall -Wextra -Wno-unused-parameter -fopenmp \
               -funroll-loops -ffast-math

ifneq ($(wildcard /proc/cpuinfo),)
  CPU_FLAGS := $(shell grep -m1 '^flags' /proc/cpuinfo 2>/dev/null)
  HAVE_AVX2 := $(findstring avx2,$(CPU_FLAGS))
  HAVE_AVX  := $(findstring avx,$(CPU_FLAGS))
  HAVE_FMA  := $(findstring fma,$(CPU_FLAGS))
else
  HAVE_AVX2 :=
  HAVE_AVX  :=
  HAVE_FMA  :=
endif

ARCH_FLAGS ?=
ifeq ($(ARCH_FLAGS),)
  ifeq ($(HAVE_AVX2),avx2)
    ARCH_FLAGS := -mavx2
    ifneq ($(HAVE_FMA),)
      ARCH_FLAGS += -mfma
    endif
  else ifeq ($(HAVE_AVX),avx)
    ARCH_FLAGS := -mavx
  else
    ARCH_FLAGS := -march=x86-64
  endif
endif

CFLAGS  ?= $(BASE_CFLAGS) $(ARCH_FLAGS)
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
	@echo "SIMD flags: $(ARCH_FLAGS)"
	$(CC) $(CPPFLAGS) $(CFLAGS) -o bonsai-cpu-blas main.c $(LDFLAGS)

run: build
	./bonsai-cpu-blas "$(MODEL)" -p "$(PROMPT)"

clean:
	rm -f bonsai-cpu-blas
