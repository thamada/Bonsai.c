CC      ?= cc
CFLAGS  ?= -O3 -std=c11 -Wall -Wextra -Wno-unused-parameter -fopenmp
LDFLAGS ?= -lm -fopenmp

MODEL   ?= ../Bonsai-8B-Q1_0.gguf
PROMPT  ?= Hello, how are you?

.PHONY: build run clean

build:
	$(CC) $(CFLAGS) -o bonsai-cpu-omp main.c $(LDFLAGS)

run: build
	./bonsai-cpu-omp "$(MODEL)" -p "$(PROMPT)"

clean:
	rm -f bonsai-cpu-omp
