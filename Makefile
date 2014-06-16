RUSTC?=rustc
RUSTFLAGS=-L lib
SRC_DIR=src
RUST_SRC=${SRC_DIR}/lib.rs
BUILD_DIR=out
DOCS_DIR=doc

.PHONY: all
all: build examples docs

build: $(RUST_SRC)
	mkdir -p $(BUILD_DIR)
	$(RUSTC) $(RUSTFLAGS) --out-dir $(BUILD_DIR) --crate-type lib $(RUST_SRC)

test-compile: $(RUST_SRC)
	mkdir -p $(BUILD_DIR)
	$(RUSTC) $(RUSTFLAGS) --test --out-dir $(BUILD_DIR) $(RUST_SRC)

.PHONY: test
test: test-compile $(RUST_SRC)
	RUST_TEST_TASKS=1 $(BUILD_DIR)/ml

.PHONY: clean
clean:
	rm -rf $(BUILD_DIR) $(DOCS_DIR)

.PHONY: examples
examples: build examples/kmeans.rs examples/linreg.rs
	$(RUSTC) -L lib -L $(BUILD_DIR) --out-dir $(BUILD_DIR) examples/kmeans.rs
	$(RUSTC) -L lib -L $(BUILD_DIR) --out-dir $(BUILD_DIR) examples/linreg.rs
	$(RUSTC) -L lib -L $(BUILD_DIR) --out-dir $(BUILD_DIR) examples/logreg.rs

.PHONY: docs
docs:
	rustdoc -o $(DOCS_DIR) -L lib $(RUST_SRC)

