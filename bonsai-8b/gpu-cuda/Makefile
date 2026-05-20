# NVIDIA CUDA。CUDA Toolkit が必要（例: sudo apt install nvidia-cuda-toolkit）。
# GPU アーキテクチャ（CUDA 11.x は -arch=native 非対応）。
# 既定は PTX (compute_86) を生成し、ドライバ JIT で新 GPU にも対応。
# 例: make build CUDA_GENCODE=arch=compute_90,code=sm_90

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

.PHONY: build run clean

build: main.o kernels.o turboquant.o
	$(NVCC) $(NVCCFLAGS) -o bonsai-gpu-cuda main.o kernels.o turboquant.o $(LDFLAGS)

main.o: main.c gpu.h turboquant.h
	$(CC) $(CFLAGS) -c main.c -o main.o

kernels.o: kernels.cu gpu.h turboquant.h
	$(NVCC) $(NVCCFLAGS) -c kernels.cu -o kernels.o

turboquant.o: turboquant.c turboquant.h
	$(CC) $(CFLAGS) -c turboquant.c -o turboquant.o

run: build
	./bonsai-gpu-cuda "$(MODEL)" -p "$(PROMPT)"

clean:
	rm -f bonsai-gpu-cuda main.o kernels.o turboquant.o
