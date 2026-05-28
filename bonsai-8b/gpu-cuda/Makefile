# NVIDIA CUDA。CUDA Toolkit が必要（例: sudo apt install nvidia-cuda-toolkit）。
# 既定: 先頭 GPU の compute capability から -gencode を自動選択（nvidia-smi）。
#   RTX 50 / Blackwell (12.x) → sm_120 + FA_BR=32（PTX JIT の compute_86 は推論が壊れる）
#   上記以外のフォールバック → compute_86 PTX
# 手動例: make build CUDA_GENCODE=arch=compute_90,code=sm_90
#
# Blackwell FP4 フルセット: make blackwell
#   apt CUDA 11 削除 → CUDA 13 インストール → NVFP4 Tensor Core (CUTLASS) + sm_120a ビルド

CUDA13_NVCC ?= /usr/local/cuda/bin/nvcc

# nvidia-smi の "12.0" → 120
GPU_CCAP_NUM := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | awk -F. '{print $$1$$2}')
ifeq ($(GPU_CCAP_NUM),120)
DEFAULT_GENCODE := arch=compute_120,code=sm_120
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

# コマンドライン指定がなければ自動検出値を使う（make build CUDA_GENCODE=... で上書き可）
CUDA_GENCODE ?= $(DEFAULT_GENCODE)
FA_BR        ?= $(or $(DEFAULT_FA_BR),64)

# Blackwell 向けには CUDA 13 の nvcc を優先（存在すれば）
ifneq ($(filter 120 121,$(GPU_CCAP_NUM)),)
ifneq ($(wildcard $(CUDA13_NVCC)),)
NVCC ?= $(CUDA13_NVCC)
endif
endif
NVCC ?= nvcc
CC      ?= cc

# /usr/local/bin/nvcc のみ PATH にあるとリンク時の nvlink が見つからない
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

MODEL   ?= ../Bonsai-8B-Q1_0.gguf
PROMPT  ?= Hello, how are you?

BLACKWELL_GENCODE ?= arch=compute_120a,code=sm_120a
BLACKWELL_FA_BR     ?= 32

CUTLASS_DIR ?= third_party/cutlass
CUTLASS_TAG ?= v3.9.0

# Blackwell FP4 ビルド: BONSAI_FP4=1（make blackwell が自動設定）
BONSAI_FP4 ?= 0

FP4_CXXFLAGS = -std=c++17 --expt-relaxed-constexpr \
               -I$(CUTLASS_DIR)/include \
               -I$(CUTLASS_DIR)/tools/util/include \
               -DCUTLASS_ARCH_MMA_SM120_SUPPORTED

FP4_FLAG :=
FP4_OBJS :=
ifneq ($(BONSAI_FP4),0)
FP4_FLAG  := -DBONSAI_FP4=1
FP4_OBJS  := fp4_gemm.o fp4_bonsai.o
endif

# CUDA_GENCODE / BONSAI_FP4 / FA_BR が変わったら .o を必ず再コンパイル（古い kernels.o 混入防止）
BUILD_STAMP  := .build_config.stamp
BUILD_CONFIG := $(CUDA_GENCODE)|fp4=$(BONSAI_FP4)|fa=$(or $(FA_BR),64)

# Blackwell (sm_120) は静的 shared 上限 48 KB。FA_BR=32 で Flash Attention が収まる。
FA_BR_FLAG :=
ifneq ($(FA_BR),)
FA_BR_FLAG := -DFA_BR=$(FA_BR)
endif

.PHONY: build run run.no-fp4 clean blackwell cutlass

.DEFAULT_GOAL := run

build: main.o kernels.o $(FP4_OBJS)
	@echo "=== build: GPU_CCAP=$(GPU_CCAP_NUM) CUDA_GENCODE=$(CUDA_GENCODE) FA_BR=$(or $(FA_BR),64) NVCC=$(NVCC) ==="
	$(NVCC) $(NVCCFLAGS) -o bonsai-gpu-cuda main.o kernels.o $(FP4_OBJS) $(LDFLAGS)

main.o: main.c gpu.h
	$(CC) $(CFLAGS) -c main.c -o main.o

$(BUILD_STAMP): FORCE
	@printf '%s\n' '$(BUILD_CONFIG)' > $@.new
	@if ! cmp -s $@.new $@ 2>/dev/null; then mv $@.new $@; else rm -f $@.new; fi

FORCE:

kernels.o: kernels.cu gpu.h fp4_bonsai.h $(BUILD_STAMP)
	$(NVCC) $(NVCCFLAGS) $(FA_BR_FLAG) $(FP4_FLAG) -c kernels.cu -o kernels.o

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

run.no-fp4: build
	./bonsai-gpu-cuda "$(MODEL)" -p "$(PROMPT)"

clean:
	rm -f bonsai-gpu-cuda main.o kernels.o fp4_gemm.o fp4_bonsai.o .build_config.stamp

# 初回のみ apt で CUDA 13 を導入。2 回目以降は nvcc があれば apt をスキップしてビルドのみ。
# 強制再インストール: make blackwell FORCE_CUDA_APT=1
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
	echo "=== Building Blackwell ($(BLACKWELL_GENCODE), FA_BR=$(BLACKWELL_FA_BR), Q1_0 GPU) ==="; \
	$(MAKE) build CUDA_GENCODE="$(BLACKWELL_GENCODE)" NVCC="$(CUDA13_NVCC)" \
		FA_BR="$(BLACKWELL_FA_BR)" BONSAI_FP4=0

run: blackwell
	./bonsai-gpu-cuda "$(MODEL)" -p "$(PROMPT)"
