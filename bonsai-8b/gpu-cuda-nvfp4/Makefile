# NVIDIA CUDA — NVFP4 + CUTLASS（Blackwell Tensor Core 線形層）。
# CUDA Toolkit 13 推奨。make blackwell / make run で CUDA 13 導入 + sm_120a ビルド。

CUDA13_NVCC ?= /usr/local/cuda/bin/nvcc

GPU_CCAP_NUM := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | awk -F. '{print $$1$$2}')
ifeq ($(GPU_CCAP_NUM),120)
DEFAULT_GENCODE := arch=compute_120a,code=sm_120a
DEFAULT_FA_BR   := 32
else ifeq ($(GPU_CCAP_NUM),121)
DEFAULT_GENCODE := arch=compute_120a,code=sm_120a
DEFAULT_FA_BR   := 32
else ifeq ($(GPU_CCAP_NUM),90)
DEFAULT_GENCODE := arch=compute_90,code=sm_90
else ifeq ($(GPU_CCAP_NUM),89)
DEFAULT_GENCODE := arch=compute_89,code=sm_89
else ifeq ($(GPU_CCAP_NUM),86)
DEFAULT_GENCODE := arch=compute_86,code=sm_86
else
DEFAULT_GENCODE := arch=compute_86,code=compute_86
endif

CUDA_GENCODE ?= $(DEFAULT_GENCODE)
FA_BR        ?= $(or $(DEFAULT_FA_BR),64)

ifneq ($(filter 120 121,$(GPU_CCAP_NUM)),)
ifneq ($(wildcard $(CUDA13_NVCC)),)
NVCC ?= $(CUDA13_NVCC)
endif
endif
NVCC ?= nvcc
CC      ?= cc

ifeq ($(findstring /,$(NVCC)),/)
CUDA_BIN := $(dir $(shell readlink -f $(NVCC)))
else
CUDA_BIN := $(dir $(shell readlink -f $$(command -v $(NVCC) 2>/dev/null || echo $(CUDA13_NVCC))))
endif
export PATH := $(CUDA_BIN):$(PATH)

NVCCFLAGS ?= -O3 -std=c++14 -Xcompiler -Wall,-Wextra,-Wno-unused-parameter \
             -use_fast_math --expt-relaxed-constexpr \
             -gencode $(CUDA_GENCODE)

CFLAGS  ?= -O3 -std=c11 -Wall -Wextra -Wno-unused-parameter
LDFLAGS ?= -lcudart -lm

BIN     ?= bonsai-gpu-cuda-nvfp4
MODEL   ?= ../Bonsai-8B-Q1_0.gguf
PROMPT  ?= Hello, how are you?

BENCH_PROMPT ?= Artificial intelligence has transformed many industries over the past decade. Machine learning models can now understand natural language, recognize images, and generate creative content with remarkable fluency. Large language models are trained on vast corpora of text and learn statistical patterns that enable them to predict the next token in a sequence. This capability powers applications ranging from virtual assistants and code completion tools to scientific research and education. However, deploying these models efficiently remains a challenge because they require substantial compute resources during both training and inference. Researchers explore quantization and specialized hardware to reduce latency while preserving quality.
BENCH_N      ?= 128
BENCH_SEED   ?= 42
BENCH_LOG_FILE ?= /tmp/benchmark.log

GPU_LABEL ?= sm_$(GPU_CCAP_NUM)

BLACKWELL_GENCODE ?= arch=compute_120a,code=sm_120a
BLACKWELL_FA_BR     ?= 32

CUTLASS_DIR ?= third_party/cutlass
CUTLASS_TAG ?= v4.5.1

FP4_CXXFLAGS = -std=c++17 --expt-relaxed-constexpr \
               -I$(CUTLASS_DIR)/include \
               -I$(CUTLASS_DIR)/tools/util/include \
               -DCUTLASS_ARCH_MMA_SM120_SUPPORTED

FP4_OBJS := fp4_gemm.o fp4_bonsai.o

BUILD_STAMP  := .build_config.stamp
BUILD_CONFIG := $(CUDA_GENCODE)|nvfp4|fa=$(or $(FA_BR),64)

FA_BR_FLAG :=
ifneq ($(FA_BR),)
FA_BR_FLAG := -DFA_BR=$(FA_BR)
endif

.PHONY: build run clean blackwell cutlass log log.push

.DEFAULT_GOAL := run

build: main.o kernels.o $(FP4_OBJS)
	@echo "=== build: GPU_CCAP=$(GPU_CCAP_NUM) CUDA_GENCODE=$(CUDA_GENCODE) FA_BR=$(or $(FA_BR),64) NVCC=$(NVCC) (NVFP4) ==="
	$(NVCC) $(NVCCFLAGS) -o $(BIN) main.o kernels.o $(FP4_OBJS) $(LDFLAGS)

main.o: main.c gpu.h
	$(CC) $(CFLAGS) -c main.c -o main.o

$(BUILD_STAMP): FORCE
	@printf '%s\n' '$(BUILD_CONFIG)' > $@.new
	@if ! cmp -s $@.new $@ 2>/dev/null; then mv $@.new $@; else rm -f $@.new; fi

FORCE:

kernels.o: kernels.cu gpu.h fp4_bonsai.h $(BUILD_STAMP)
	$(NVCC) $(NVCCFLAGS) $(FA_BR_FLAG) -c kernels.cu -o kernels.o

fp4_gemm.o: fp4_gemm.cu fp4_gemm.h $(BUILD_STAMP) | cutlass
	$(NVCC) $(NVCCFLAGS) $(FP4_CXXFLAGS) -c fp4_gemm.cu -o fp4_gemm.o

fp4_bonsai.o: fp4_bonsai.cu fp4_bonsai.h fp4_gemm.h gpu.h $(BUILD_STAMP) | cutlass
	$(NVCC) $(NVCCFLAGS) $(FP4_CXXFLAGS) -c fp4_bonsai.cu -o fp4_bonsai.o

cutlass:
	@if [ ! -f "$(CUTLASS_DIR)/include/cutlass/cutlass.h" ]; then \
		echo "=== Cloning CUTLASS $(CUTLASS_TAG) ==="; \
		mkdir -p third_party; \
		git clone --depth 1 --branch $(CUTLASS_TAG) \
			https://github.com/NVIDIA/cutlass.git $(CUTLASS_DIR); \
	fi

run: blackwell
	./$(BIN) "$(MODEL)" -p "$(PROMPT)"

clean:
	rm -f $(BIN) main.o kernels.o fp4_gemm.o fp4_bonsai.o .build_config.stamp

blackwell:
	@if [ "$$(id -u)" -ne 0 ]; then SUDO=sudo; else SUDO=; fi; \
	SKIP_APT=0; \
	if [ "$(FORCE_CUDA_APT)" != "1" ] && [ -x "$(CUDA13_NVCC)" ]; then \
		nvcc_ver=$$("$(CUDA13_NVCC)" --version | sed -n 's/.*release \([0-9][0-9]*\)\(\.[0-9]*\)*.*/\1/p' | head -1); \
		if [ "$$nvcc_ver" -ge 13 ] 2>/dev/null; then \
			echo "=== CUDA $$("$(CUDA13_NVCC)" --version | sed -n 's/.*release \([^,]*\).*/\1/p' | head -1) already installed — skipping apt ==="; \
			SKIP_APT=1; \
		fi; \
	fi; \
	if [ "$$SKIP_APT" = 0 ]; then \
		. /etc/os-release; \
		case "$${ID}-$${VERSION_ID}" in \
			ubuntu-22.04) CUDA_REPO=ubuntu2204 ;; \
			ubuntu-24.04) CUDA_REPO=ubuntu2404 ;; \
			ubuntu-20.04) CUDA_REPO=ubuntu2004 ;; \
			debian-12)    CUDA_REPO=debian12 ;; \
			debian-13)    CUDA_REPO=debian13 ;; \
			*) echo "Error: unsupported OS for CUDA 13 install ($${PRETTY_NAME:-$$ID $$VERSION_ID})."; \
			   echo "See https://developer.nvidia.com/cuda-downloads"; exit 1 ;; \
		esac; \
		echo "=== Removing apt CUDA Toolkit 11.x packages ==="; \
		CUDA11_PKGS="nvidia-cuda-toolkit nvidia-cuda-toolkit-doc nvidia-cuda-dev nvidia-cuda-gdb nvidia-profiler nvidia-visual-profiler libcupti-dev libcupti-doc libcupti11.5 libnvrtc11.2 libnvrtc-builtins11.5 libcudart11.0"; \
		to_remove=""; \
		for pkg in $$CUDA11_PKGS; do \
			if dpkg-query -W -f='$${Status}' "$$pkg" 2>/dev/null | grep -q "install ok installed"; then \
				to_remove="$$to_remove $$pkg"; \
			fi; \
		done; \
		if [ -n "$$to_remove" ]; then \
			$$SUDO apt-get remove --purge -y $$to_remove; \
			$$SUDO apt-get autoremove -y; \
		else \
			echo "No apt CUDA 11.x packages installed."; \
		fi; \
		echo "=== Installing CUDA Toolkit 13 from NVIDIA ($$CUDA_REPO) ==="; \
		KEYRING_DEB="cuda-keyring_1.1-1_all.deb"; \
		KEYRING_URL="https://developer.download.nvidia.com/compute/cuda/repos/$$CUDA_REPO/x86_64/$$KEYRING_DEB"; \
		if ! dpkg-query -W -f='$${Status}' cuda-keyring 2>/dev/null | grep -q "install ok installed"; then \
			wget -O "/tmp/$$KEYRING_DEB" "$$KEYRING_URL"; \
			$$SUDO dpkg -i "/tmp/$$KEYRING_DEB"; \
			rm -f "/tmp/$$KEYRING_DEB"; \
		fi; \
		$$SUDO apt-get update; \
		$$SUDO apt-get install -y cuda-toolkit-13 git; \
	fi; \
	if [ ! -x "$(CUDA13_NVCC)" ]; then \
		echo "Error: $(CUDA13_NVCC) not found. Run: make blackwell FORCE_CUDA_APT=1"; \
		exit 1; \
	fi; \
	nvcc_ver=$$("$(CUDA13_NVCC)" --version | sed -n 's/.*release \([0-9][0-9]*\)\(\.[0-9]*\)*.*/\1/p' | head -1); \
	if [ "$$nvcc_ver" -lt 13 ] 2>/dev/null; then \
		echo "Error: expected CUDA Toolkit 13+, got release $$nvcc_ver"; \
		exit 1; \
	fi; \
	if [ "$$SKIP_APT" = 0 ]; then \
		echo "=== CUDA $$("$(CUDA13_NVCC)" --version | sed -n 's/.*release \([^,]*\).*/\1/p' | head -1) ready ==="; \
		echo "=== Configuring CUDA PATH ==="; \
		CUDA_PROFILE="/etc/profile.d/cuda.sh"; \
		CUDA_BIN="/usr/local/cuda/bin"; \
		if [ ! -f "$$CUDA_PROFILE" ] || ! grep -q 'cuda/bin' "$$CUDA_PROFILE" 2>/dev/null; then \
			printf '%s\n' 'export PATH="/usr/local/cuda/bin$${PATH:+:$${PATH}}"' | $$SUDO tee "$$CUDA_PROFILE" > /dev/null; \
			$$SUDO chmod 644 "$$CUDA_PROFILE"; \
		fi; \
		$$SUDO ln -sf "$$CUDA_BIN/nvcc" /usr/local/bin/nvcc; \
		$$SUDO ln -sf "$$CUDA_BIN/nvlink" /usr/local/bin/nvlink; \
	fi; \
	if ! command -v nvcc >/dev/null 2>&1; then export PATH="/usr/local/cuda/bin:$$PATH"; fi; \
	echo "nvcc: $$(command -v nvcc)"; \
	$(MAKE) cutlass; \
	echo "=== Building Blackwell NVFP4 ($(BLACKWELL_GENCODE), FA_BR=$(BLACKWELL_FA_BR)) ==="; \
	$(MAKE) build CUDA_GENCODE="$(BLACKWELL_GENCODE)" NVCC="$(CUDA13_NVCC)" \
		FA_BR="$(BLACKWELL_FA_BR)"

log:
	@echo ""
	@echo "==============================================="
	@echo "  gpu-cuda-nvfp4 benchmark history"
	@echo "==============================================="
	@echo ""
	@count=$$(sed -n '/^BENCH_LOG += /p' Makefile | wc -l); \
	if [ "$$count" -eq 0 ]; then \
		echo "  (no entries yet — run: make log.push)"; \
	else \
		printf "  %-19s %8s %14s %8s %6s %10s %10s %10s\n" \
			"Date" "GPU" "Host" "Prompt" "Gen" "Prefill/s" "Decode/s" "Total/s"; \
		printf "  %-19s %8s %14s %8s %6s %10s %10s %10s\n" \
			"-------------------" "--------" "--------------" "--------" "------" "----------" "----------" "----------"; \
		sed -n '/^BENCH_LOG += /s/^BENCH_LOG += //p' Makefile | \
		awk -F'|' '{ \
			date = $$1; sub(/[+].*$$/, "", date); \
			printf "  %-19s %8s %14s %8s %6s %10s %10s %10s\n", \
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
	@BENCH_LOG_FILE="$(BENCH_LOG_FILE)" ./$(BIN) "$(MODEL)" -p "$(BENCH_PROMPT)" -n $(BENCH_N) -t 0 -s $(BENCH_SEED); \
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
	entry="BENCH_LOG += $$(date +%Y-%m-%dT%H:%M:%S)|$(GPU_LABEL)|$$(hostname -s)|$$np|$$ng|$$pf|$$dc|$$tt"; \
	sed -i "/^# BENCH_LOG_END/i $$entry" Makefile; \
	echo ""; \
	echo "Logged: $$np prompt + $$ng gen, prefill $${pf} / decode $${dc} / total $${tt} tok/s on $(GPU_LABEL)"; \
	echo "  run: make log"

# BENCH_LOG_BEGIN — appended by make log.push
BENCH_LOG :=
BENCH_LOG += 2026-05-28T18:48:15|sm_120|de9a61c86ea0|130|128|5751.92|64.12|127.80
BENCH_LOG += 2026-05-28T20:30:08|sm_120|de9a61c86ea0|130|128|5762.22|64.14|127.84
BENCH_LOG += 2026-05-28T20:30:48|sm_120|de9a61c86ea0|130|128|5757.93|64.20|127.96
# BENCH_LOG_END
