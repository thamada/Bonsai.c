# AMD Radeon GPU (ROCm / HIP) — Bonsai-8B Q1_0 inference
ROCM       ?= /opt/rocm
HIPCC      ?= $(ROCM)/bin/hipcc
ROCMININFO ?= $(ROCM)/bin/rocminfo

# Auto-detect GPU ISA from rocminfo (first GPU agent Name: gfx*). Override: make GPU_ARCH=gfx1100
GPU_ARCH   ?= $(shell $(ROCMININFO) 2>/dev/null | awk '/^  Name:/ { n=$$NF; if (n ~ /^gfx[0-9]+/) { print n; exit } }')

MODEL   ?= ../Bonsai-8B-Q1_0.gguf
PROMPT  ?= Hello, how are you?

# Flash Attention tile (shared memory). gfx11/gfx12 既定 32（64KB 上限）。大容量 shared の GPU は FA_BR=64 可。
FA_BR   ?= 32

.PHONY: build run clean detect-gpu-arch

.DEFAULT_GOAL := run

# -----------------------------------------------------------------------------
# GPU architecture (auto-detect via rocminfo, or override on command line)
# -----------------------------------------------------------------------------

detect-gpu-arch:
	@if [ -z "$(GPU_ARCH)" ]; then \
		echo "error: GPU_ARCH not detected. Ensure $(ROCMININFO) reports a GPU, or set GPU_ARCH manually (e.g. make GPU_ARCH=gfx1100)." >&2; \
		exit 1; \
	fi
	@echo ""
	@echo "==============================================="
	@echo "  Detected GPU arch: $(GPU_ARCH)"
	@echo "==============================================="
	@echo ""

# -----------------------------------------------------------------------------
# Build / run
# -----------------------------------------------------------------------------

CXX ?= g++

# hipcc (ROCm clang) does not search GCC libstdc++ headers/libs by default.
CXX_INC_DIRS := $(shell $(CXX) -E -x c++ - -v </dev/null 2>&1 | \
	awk '/#include <\.\.\.> search starts here:/,/^End of search list\./{gsub(/^ /,""); if ($$0 ~ /^\//) print $$0}')
ifeq ($(CXX_INC_DIRS),)
  GCC_VER := $(shell $(CXX) -dumpversion 2>/dev/null | cut -d. -f1)
  CXX_INC_DIRS := /usr/include/c++/$(GCC_VER) /usr/include/x86_64-linux-gnu/c++/$(GCC_VER)
endif
HIP_CXXFLAGS := $(foreach d,$(CXX_INC_DIRS),-isystem$(d))

STDCXX_LIBDIR ?= $(shell dirname "$$($(CXX) -print-file-name=libstdc++.a 2>/dev/null)" 2>/dev/null)
LDFLAGS_STDCXX := $(if $(STDCXX_LIBDIR),-L$(STDCXX_LIBDIR),-L/usr/lib/gcc/x86_64-linux-gnu/$(shell $(CXX) -dumpversion | cut -d. -f1)) -lstdc++

HIP_CFLAGS := -O3 -std=c++17 -Wall -Wextra -Wno-unused-parameter \
	--offload-arch=$(GPU_ARCH) -I$(ROCM)/include $(HIP_CXXFLAGS) -DFA_BR=$(FA_BR)
HIP_LDFLAGS := -L$(ROCM)/lib -lamdhip64 -Wl,-rpath,$(ROCM)/lib $(LDFLAGS_STDCXX) -lm

ifeq ($(GPU_ARCH),)
$(error GPU_ARCH not detected. Ensure $(ROCMININFO) reports a GPU, or: make GPU_ARCH=gfx1100 build)
endif

bonsai-gpu-rocm: main.o kernels.o
	@test -n "$(CXX_INC_DIRS)" || (echo "error: C++ headers not found. Install: apt install g++ libstdc++-dev" >&2; exit 1)
	@echo ""
	@echo "==============================================="
	@echo "  Detected GPU arch: $(GPU_ARCH)"
	@echo "==============================================="
	@echo ""
	$(HIPCC) $(HIP_CFLAGS) -o bonsai-gpu-rocm main.o kernels.o $(HIP_LDFLAGS)

build: bonsai-gpu-rocm

main.o: main.c gpu.h
	@test -n "$(CXX_INC_DIRS)" || (echo "error: C++ headers not found. Install: apt install g++ libstdc++-dev" >&2; exit 1)
	$(HIPCC) $(HIP_CFLAGS) -c -x hip main.c -o main.o

kernels.o: kernels.hip gpu.h
	@test -n "$(CXX_INC_DIRS)" || (echo "error: C++ headers not found. Install: apt install g++ libstdc++-dev" >&2; exit 1)
	$(HIPCC) $(HIP_CFLAGS) -c -x hip kernels.hip -o kernels.o

run: bonsai-gpu-rocm
	./bonsai-gpu-rocm "$(MODEL)" -p "$(PROMPT)"

clean:
	rm -f bonsai-gpu-rocm main.o kernels.o
