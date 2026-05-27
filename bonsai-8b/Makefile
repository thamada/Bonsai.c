MODEL   ?= Bonsai-8B-Q1_0.gguf
PROMPT  ?= Hello, how are you?

# 各ビルドはサブディレクトリの Makefile の既定（CC / CFLAGS 等）を使う。
# 上書きは例: make -C cpu build CC=clang  または  make build.cpu CC=clang

.PHONY: \
	all \
	model \
	build.cpu run.cpu \
	build.cpu-blas run.cpu-blas \
	build.gpu-cuda run.gpu-cuda \
	build.gpu-rocm run.gpu-rocm \
	blackwell build.blackwell run.blackwell \
	clean

.DEFAULT_GOAL := all

# -----------------------------------------------------------------------------
# GGUF モデル（gguf.txt の URL からダウンロード）
model:
	@if [ ! -f "$(MODEL).sha256sum" ]; then \
		echo "Error: checksum file '$(MODEL).sha256sum' not found."; \
		echo "Update the repository to the latest version, or verify the file exists."; \
		exit 1; \
	fi; \
	url=$$(sed 's|/blob/main/|/resolve/main/|' gguf.txt); \
	wget -O "$(MODEL)" "$$url" && \
	if sha256sum --check "$(MODEL).sha256sum"; then \
		echo ""; \
		echo "Checksum verification succeeded."; \
		echo ""; \
		echo "  File:     $(MODEL)"; \
		echo "  Expected: $(MODEL).sha256sum"; \
		echo ""; \
		echo "Model download and verification complete."; \
		echo "  make run.cpu"; \
		echo "or similar to run inference."; \
	else \
		echo ""; \
		echo "Error: checksum verification failed for the downloaded model file."; \
		echo ""; \
		echo "  File:     $(MODEL)"; \
		echo "  Expected: $(MODEL).sha256sum"; \
		echo ""; \
		echo "The file may have been corrupted during download."; \
		echo "Removing the corrupted file. Check your network connection, then run"; \
		echo "  make model"; \
		echo "again."; \
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
# GPU CUDA（ソースは gpu-cuda/main.c + kernels.cu。cuBLAS 不要）
build.gpu-cuda:
	cd gpu-cuda && make build

run.gpu-cuda: build.gpu-cuda
	./gpu-cuda/bonsai-gpu-cuda "$(MODEL)" -p "$(PROMPT)"

# -----------------------------------------------------------------------------
# Blackwell GPU（RTX 50 系等）: gpu-cuda/Makefile の blackwell を呼ぶ（要 sudo）
blackwell build.blackwell:
	cd gpu-cuda && $(MAKE) blackwell

run.blackwell: build.blackwell
	./gpu-cuda/bonsai-gpu-cuda "$(MODEL)" -p "$(PROMPT)"

# -----------------------------------------------------------------------------
# GPU ROCm/HIP（ソースは gpu-rocm/main.c + kernels.hip）
build.gpu-rocm:
	cd gpu-rocm && $(MAKE) build

run.gpu-rocm: build.gpu-rocm
	./gpu-rocm/bonsai-gpu-rocm "$(MODEL)" -p "$(PROMPT)"

# -----------------------------------------------------------------------------
all: build.cpu

# -----------------------------------------------------------------------------
clean:
	cd cpu && make clean
	cd cpu-blas && make clean
	cd gpu-cuda && make clean
	cd gpu-rocm && make clean
