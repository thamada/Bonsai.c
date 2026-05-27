MODEL ?= Bonsai-8B-Q1_0.gguf

.PHONY: model

.DEFAULT_GOAL := model

# GGUF モデル（未ダウンロード時は gguf.txt の URL から取得、既存時は checksum のみ）
model:
	@if [ ! -f "$(MODEL).sha256sum" ]; then \
		echo "Error: checksum file '$(MODEL).sha256sum' not found."; \
		echo "Update the repository to the latest version, or verify the file exists."; \
		exit 1; \
	fi; \
	if [ -f "$(MODEL)" ]; then \
		echo "Model file already exists: $(MODEL)"; \
		echo "Skipping download, verifying checksum..."; \
		skipped=1; \
	else \
		url=$$(sed 's|/blob/main/|/resolve/main/|' gguf.txt); \
		wget -O "$(MODEL)" "$$url" || exit 1; \
		skipped=0; \
	fi; \
	if sha256sum --check "$(MODEL).sha256sum"; then \
		echo ""; \
		echo "Checksum verification succeeded."; \
		echo ""; \
		echo "  File:     $(MODEL)"; \
		echo "  Expected: $(MODEL).sha256sum"; \
		echo ""; \
		if [ "$$skipped" = 1 ]; then \
			echo "Model verification complete."; \
		else \
			echo "Model download and verification complete."; \
		fi; \
	else \
		echo ""; \
		echo "Error: checksum verification failed for '$(MODEL)'."; \
		echo ""; \
		echo "  File:     $(MODEL)"; \
		echo "  Expected: $(MODEL).sha256sum"; \
		echo ""; \
		echo "The file may be corrupted. Removing it. Check your network connection, then run"; \
		echo "  make model"; \
		echo "again."; \
		rm -f "$(MODEL)"; \
		exit 1; \
	fi
