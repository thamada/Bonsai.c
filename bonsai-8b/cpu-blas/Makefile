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

# Benchmark defaults for log.push (override: make log.push BENCH_N=64)
# BENCH_PROMPT is ~128 tokens (ChatML-encoded prompt length).
BENCH_PROMPT ?= Artificial intelligence has transformed many industries over the past decade. Machine learning models can now understand natural language, recognize images, and generate creative content with remarkable fluency. Large language models are trained on vast corpora of text and learn statistical patterns that enable them to predict the next token in a sequence. This capability powers applications ranging from virtual assistants and code completion tools to scientific research and education. However, deploying these models efficiently remains a challenge because they require substantial compute resources during both training and inference. Researchers explore quantization and specialized hardware to reduce latency while preserving quality.
BENCH_N      ?= 128
BENCH_SEED   ?= 42
BENCH_LOG_FILE ?= /tmp/benchmark.log

# Short SIMD label for benchmark history (override: make log.push BENCH_SIMD=avx2+fma)
BENCH_SIMD ?=
ifeq ($(BENCH_SIMD),)
  ifeq ($(HAVE_AVX2),avx2)
    BENCH_SIMD := avx2
    ifneq ($(HAVE_FMA),)
      BENCH_SIMD := avx2+fma
    endif
  else ifeq ($(HAVE_AVX),avx)
    BENCH_SIMD := avx
  else
    BENCH_SIMD := x86-64
  endif
endif

OPENBLAS_CFLAGS := $(shell pkg-config --cflags openblas 2>/dev/null)
OPENBLAS_LIBS   := $(shell pkg-config --libs openblas 2>/dev/null)
ifneq ($(strip $(OPENBLAS_LIBS)),)
  CPPFLAGS += $(OPENBLAS_CFLAGS)
  LDFLAGS := $(filter-out -lopenblas,$(LDFLAGS))
  LDFLAGS += $(OPENBLAS_LIBS)
endif

.PHONY: build run clean log log.push

build:
	@echo "SIMD flags: $(ARCH_FLAGS)"
	$(CC) $(CPPFLAGS) $(CFLAGS) -o bonsai-cpu-blas main.c $(LDFLAGS)

run: build
	./bonsai-cpu-blas "$(MODEL)" -p "$(PROMPT)"

clean:
	rm -f bonsai-cpu-blas

# -----------------------------------------------------------------------------
# Benchmark log (make log / make log.push)
# Format per entry (pipe-separated, no spaces):
#   ISO8601|BENCH_SIMD|hostname|prompt_tokens|gen_tokens|prefill_tps|decode_tps|total_tps
# Throughput is inference-only (prefill+decode); model load is excluded.
# BENCH_LOG_BEGIN — appended by make log.push
BENCH_LOG :=
BENCH_LOG += 2026-05-27T19:39:34|avx|5ae102a9321e|130|128|1.96|1.99|1.98
BENCH_LOG += 2026-05-27T19:57:16|avx|5ae102a9321e|130|128|1.97|2.00|1.99
# BENCH_LOG_END

log:
	@echo ""
	@echo "==============================================="
	@echo "  cpu-blas benchmark history"
	@echo "==============================================="
	@echo ""
	@count=$$(sed -n '/^BENCH_LOG += /p' Makefile | wc -l); \
	if [ "$$count" -eq 0 ]; then \
		echo "  (no entries yet — run: make log.push)"; \
	else \
		printf "  %-19s %10s %14s %8s %6s %10s %10s %10s\n" \
			"Date" "SIMD" "Host" "Prompt" "Gen" "Prefill/s" "Decode/s" "Total/s"; \
		printf "  %-19s %10s %14s %8s %6s %10s %10s %10s\n" \
			"-------------------" "----------" "--------------" "--------" "------" "----------" "----------" "----------"; \
		sed -n '/^BENCH_LOG += /s/^BENCH_LOG += //p' Makefile | \
		awk -F'|' '{ \
			date = $$1; sub(/[+].*$$/, "", date); \
			printf "  %-19s %10s %14s %8s %6s %10s %10s %10s\n", \
				date, $$2, $$3, $$4, $$5, $$6, $$7, $$8 }'; \
	fi
	@echo ""

log.push: build
	@if [ ! -f "$(MODEL)" ]; then \
		echo "error: model not found: $(MODEL)" >&2; \
		echo "  run: make -C .. model" >&2; \
		exit 1; \
	fi
	@echo "Running benchmark (prompt=\"$(BENCH_PROMPT)\", -n $(BENCH_N), -t 0, -s $(BENCH_SEED)) ..."
	@BENCH_LOG_FILE="$(BENCH_LOG_FILE)" ./bonsai-cpu-blas "$(MODEL)" -p "$(BENCH_PROMPT)" -n $(BENCH_N) -t 0 -s $(BENCH_SEED); \
	if [ ! -f "$(BENCH_LOG_FILE)" ]; then \
		echo "error: benchmark log not found: $(BENCH_LOG_FILE)" >&2; \
		exit 1; \
	fi; \
	np=$$(sed -n 's/^prompt_tokens=//p' "$(BENCH_LOG_FILE)" | tail -1); \
	ng=$$(sed -n 's/^gen_tokens=//p' "$(BENCH_LOG_FILE)" | tail -1); \
	pf=$$(sed -n 's/^prefill_tps=//p' "$(BENCH_LOG_FILE)" | tail -1); \
	dc=$$(sed -n 's/^decode_tps=//p' "$(BENCH_LOG_FILE)" | tail -1); \
	tt=$$(sed -n 's/^total_tps=//p' "$(BENCH_LOG_FILE)" | tail -1); \
	if [ -z "$$np" ] || [ -z "$$pf" ] || [ -z "$$dc" ] || [ -z "$$tt" ]; then \
		echo "error: failed to parse $(BENCH_LOG_FILE)" >&2; \
		cat "$(BENCH_LOG_FILE)" >&2; \
		exit 1; \
	fi; \
	entry="BENCH_LOG += $$(date +%Y-%m-%dT%H:%M:%S)|$(BENCH_SIMD)|$$(hostname -s)|$$np|$$ng|$$pf|$$dc|$$tt"; \
	sed -i "/^# BENCH_LOG_END/i $$entry" Makefile; \
	echo ""; \
	echo "Logged: $$np prompt + $$ng gen, prefill $${pf} / decode $${dc} / total $${tt} tok/s on $(BENCH_SIMD)"; \
	echo "  run: make log"
