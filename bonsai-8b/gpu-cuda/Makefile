# NVIDIA CUDA。CUDA Toolkit が必要（例: sudo apt install nvidia-cuda-toolkit）。
# GPU アーキテクチャ（CUDA 11.x は -arch=native 非対応）。
# 既定は PTX (compute_86) を生成し、ドライバ JIT で新 GPU にも対応。
# 例: make build CUDA_GENCODE=arch=compute_90,code=sm_90
#
# Blackwell GPU（RTX 50 系等）: make blackwell
#   apt の CUDA 11 ツールキットを削除し、NVIDIA 公式リポジトリから CUDA 13 を入れてビルドする（要 sudo）。

CUDA_GENCODE ?= arch=compute_86,code=compute_86

NVCC    ?= nvcc
CC      ?= cc

NVCCFLAGS ?= -O3 -std=c++14 -Xcompiler -Wall,-Wextra,-Wno-unused-parameter \
             -use_fast_math --expt-relaxed-constexpr \
             -gencode $(CUDA_GENCODE)

CFLAGS  ?= -O3 -std=c11 -Wall -Wextra -Wno-unused-parameter
LDFLAGS ?= -lcudart -lm

MODEL   ?= ../Bonsai-8B-Q1_0.gguf
PROMPT  ?= Hello, how are you?

BLACKWELL_GENCODE ?= arch=compute_120,code=sm_120
CUDA13_NVCC         ?= /usr/local/cuda/bin/nvcc

.PHONY: build run clean blackwell run.blackwell

build: main.o kernels.o
	$(NVCC) $(NVCCFLAGS) -o bonsai-gpu-cuda main.o kernels.o $(LDFLAGS)

main.o: main.c gpu.h
	$(CC) $(CFLAGS) -c main.c -o main.o

kernels.o: kernels.cu gpu.h
	$(NVCC) $(NVCCFLAGS) -c kernels.cu -o kernels.o

run: build
	./bonsai-gpu-cuda "$(MODEL)" -p "$(PROMPT)"

clean:
	rm -f bonsai-gpu-cuda main.o kernels.o

# apt CUDA 11 を削除 → NVIDIA 公式 CUDA 13 をインストール → Blackwell 向けビルド
blackwell:
	@if [ "$$(id -u)" -ne 0 ]; then SUDO=sudo; else SUDO=; fi; \
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
	$$SUDO apt-get install -y cuda-toolkit-13; \
	if [ ! -x "$(CUDA13_NVCC)" ]; then \
		echo "Error: $(CUDA13_NVCC) not found after cuda-toolkit-13 install."; \
		echo "Check /usr/local/cuda/bin/nvcc or re-run: sudo apt install cuda-toolkit-13"; \
		exit 1; \
	fi; \
	nvcc_ver=$$("$(CUDA13_NVCC)" --version | sed -n 's/.*release \([0-9][0-9]*\)\(\.[0-9]*\)*.*/\1/p' | head -1); \
	if [ "$$nvcc_ver" -lt 13 ] 2>/dev/null; then \
		echo "Error: expected CUDA Toolkit 13+, got release $$nvcc_ver from $(CUDA13_NVCC)"; \
		exit 1; \
	fi; \
	echo "=== CUDA $$("$(CUDA13_NVCC)" --version | sed -n 's/.*release \([^,]*\).*/\1/p' | head -1) ready ==="; \
	echo "=== Building for Blackwell ($(BLACKWELL_GENCODE)) ==="; \
	$(MAKE) build CUDA_GENCODE="$(BLACKWELL_GENCODE)" NVCC="$(CUDA13_NVCC)"

run.blackwell: blackwell
	./bonsai-gpu-cuda "$(MODEL)" -p "$(PROMPT)"
