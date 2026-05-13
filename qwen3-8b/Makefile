CC      ?= cc
CFLAGS  ?= -O3 -std=c11 -Wall -Wextra -Wno-unused-parameter
LDFLAGS ?= -lm

MODEL   ?= Qwen_Qwen3-VL-8B-Instruct-IQ2_M.gguf
PROMPT  ?= Hello, how are you?

# AMD Radeon GPU (ROCm / HIP)
ROCM     ?= /opt/rocm
HIPCC    ?= $(ROCM)/bin/hipcc
GPU_ARCH ?= gfx1201
# 既定値 gfx1201 は AMD Radeon AI PRO R9700

# XDNA2: <drm/drm.h> が標準外にあるときだけ `-I…` を渡す。
XDNA_INCS ?=

.PHONY: \
	build run \
	build.omp run.omp \
	build.rocm run.rocm \
	build.xdna2 run.xdna2 \
	build.xdna2.bfpx run.xdna2.bfpx \
	gen-xdna-kernels \
	clean

# -----------------------------------------------------------------------------
# CPU reference
build:
	$(CC) $(CFLAGS) -o qwen3-cpu main.c $(LDFLAGS)

run: build
	./qwen3-cpu $(MODEL) -p "$(PROMPT)"

# -----------------------------------------------------------------------------
# CPU + OpenMP
build.omp:
	$(CC) $(CFLAGS) -fopenmp -o qwen3-cpu-omp main-omp.c $(LDFLAGS)

run.omp: build.omp
	./qwen3-cpu-omp $(MODEL) -p "$(PROMPT)"

# -----------------------------------------------------------------------------
# AMD Radeon GPU (ROCm / HIP)
build.rocm:
	$(HIPCC) -O3 -std=c++17 -Wall -Wextra -Wno-unused-parameter \
		--offload-arch=$(GPU_ARCH) \
		-I$(ROCM)/include \
		-o qwen3-rocm -x hip main-rocm.c \
		-L$(ROCM)/lib -lamdhip64 -Wl,-rpath,$(ROCM)/lib

run.rocm: build.rocm
	./qwen3-rocm $(MODEL) -p "$(PROMPT)"

# -----------------------------------------------------------------------------
# AMD Ryzen AI XDNA2 NPU
# （amdxdna DRM ioctl）
# Kernel >= 6.10、/dev/accel/accel0。NPU なしでも OpenMP CPU GEMV にフォールバック。
build.xdna2:
	$(CC) $(CFLAGS) -fopenmp $(XDNA_INCS) \
		-o qwen3-xdna2 main-xdna2.c $(LDFLAGS)

run.xdna2: build.xdna2
	./qwen3-xdna2 $(MODEL) -p "$(PROMPT)"

# XDNA: regenerate bf16-gemv-<n>x<d>.bin stubs under ../xdna-kernels (not real NPU ctrlcode).
gen-xdna-kernels:
	cd .. && python3 tools/gen-xdna-gemv-stubs.py xdna-kernels

# -----------------------------------------------------------------------------
# AMD Ryzen AI XDNA2 NPU, low-host-RAM path
# 省ホスト RAM の XDNA2: 線形重み→ブロック FP16（BF16 スケール+int8）変換読み込み
build.xdna2.bfpx:
	$(CC) $(CFLAGS) -fopenmp $(XDNA_INCS) \
		-o qwen3-xdna2-bfpx main-xdna2-bfpx.c $(LDFLAGS)

run.xdna2.bfpx: build.xdna2.bfpx
	./qwen3-xdna2-bfpx $(MODEL) -p "$(PROMPT)"

# -----------------------------------------------------------------------------
clean:
	rm -f \
		qwen3-cpu \
		qwen3-cpu-omp \
		qwen3-rocm \
		qwen3-xdna2 \
		qwen3-xdna2-bfpx
