RUSTC?=rustc
RUSTFLAGS=
SRC_DIR=src
RUST_SRC=${SRC_DIR}/ml.rs
BUILD_DIR=out

.PHONY: all
all: build examples

build: $(RUST_SRC)
	mkdir -p $(BUILD_DIR)
	$(RUSTC) $(RUSTFLAGS) --out-dir $(BUILD_DIR) --lib $(RUST_SRC)

test-compile: $(RUST_SRC)
	mkdir -p $(BUILD_DIR)
	$(RUSTC) --test --out-dir $(BUILD_DIR) $(RUST_SRC)

.PHONY: test
test: test-compile $(RUST_SRC)
	RUST_TEST_TASKS=1 $(BUILD_DIR)/ml

.PHONY: clean
clean:
	rm -rf $(BUILD_DIR)

.PHONY: examples
examples: build examples/kmeans.rs examples/linreg.rs
	$(RUSTC) -L $(BUILD_DIR) --out-dir $(BUILD_DIR) examples/kmeans.rs
	$(RUSTC) -L $(BUILD_DIR) --out-dir $(BUILD_DIR) examples/linreg.rs

