CC      ?= cc
CFLAGS  ?= -O3 -std=c11 -Wall -Wextra -Wno-unused-parameter
LDFLAGS ?= -lm

MODEL   ?= ../Bonsai-8B-Q1_0.gguf
PROMPT  ?= Hello, how are you?

.PHONY: build run clean

build:
	$(CC) $(CFLAGS) -o bonsai-cpu main.c $(LDFLAGS)

run: build
	./bonsai-cpu "$(MODEL)" -p "$(PROMPT)"

clean:
	rm -f bonsai-cpu
