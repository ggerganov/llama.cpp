# Settings
# --------

BUILD_DIR := .build
DEPS_DIR  := deps
DEFN_DIR  := $(BUILD_DIR)/defn

K_SUBMODULE := $(DEPS_DIR)/k
ifneq (,$(wildcard deps/k/k-distribution/target/release/k/bin/*))
  K_RELEASE ?= $(abspath $(K_SUBMODULE)/k-distribution/target/release/k)
else
  K_RELEASE ?= $(dir $(shell which kompile))..
endif
K_BIN := $(K_RELEASE)/bin
K_LIB := $(K_RELEASE)/lib/kframework
export K_RELEASE

ifneq ($(RELEASE),)
    K_BUILD_TYPE := Release
else
    K_BUILD_TYPE := Debug
endif

PATH := $(K_BIN):$(PATH)
export PATH

PYK_PATH             := $(abspath $(K_SUBMODULE)/pyk/src/)
PYWASM_PATH          := ./deps/py-wasm

PYTHONPATH := $(PYK_PATH)
export PYTHONPATH

.PHONY: all clean deps                                                     \
        build build-llvm build-haskell                                     \
        test test-execution test-simple test-prove test-binary-parser      \
        test-conformance test-conformance-parse test-conformance-supported \
        media presentations reports

all: build

clean:
	rm -rf $(BUILD_DIR)

# Build Dependencies (K Submodule)
# --------------------------------

K_JAR := $(K_SUBMODULE)/k-distribution/target/release/k/lib/java/kernel-1.0-SNAPSHOT.jar

deps: $(K_JAR) $(TANGLER)

$(K_JAR):
	cd $(K_SUBMODULE) && mvn package -DskipTests -Dproject.build.type=$(K_BUILD_TYPE)

# Building Definition
# -------------------

KOMPILE_OPTS         :=
LLVM_KOMPILE_OPTS    :=
HASKELL_KOMPILE_OPTS :=

tangle_selector := k

SOURCE_FILES       := data         \
                      kwasm-lemmas \
                      numeric      \
                      test         \
                      wasm         \
                      wasm-text    \
                      wrc20
EXTRA_SOURCE_FILES :=
ALL_SOURCE_FILES   := $(patsubst %, %.md, $(SOURCE_FILES)) $(EXTRA_SOURCE_FILES)

build: build-llvm build-haskell

ifneq (,$(RELEASE))
    KOMPILE_OPTS += -O2
else
    KOMPILE_OPTS += --debug
endif

ifeq (,$(RELEASE))
    LLVM_KOMPILE_OPTS += -g
endif

KOMPILE_LLVM := kompile --backend llvm --md-selector "$(tangle_selector)" \
                $(KOMPILE_OPTS)                                           \
                $(addprefix -ccopt ,$(LLVM_KOMPILE_OPTS))

KOMPILE_HASKELL := kompile --backend haskell --md-selector "$(tangle_selector)" \
                   $(KOMPILE_OPTS)                                              \
                   $(HASKELL_KOMPILE_OPTS)

### LLVM

llvm_dir           := $(DEFN_DIR)/llvm
llvm_files         := $(ALL_SOURCE_FILES)
llvm_main_module   := WASM-TEST
llvm_syntax_module := $(llvm_main_module)-SYNTAX
llvm_main_file     := test
llvm_kompiled      := $(llvm_dir)/$(llvm_main_file)-kompiled/interpreter

build-llvm: $(llvm_kompiled)

$(llvm_kompiled): $(llvm_files)
	$(KOMPILE_LLVM) $(llvm_main_file).md      \
	    --directory $(llvm_dir) -I $(CURDIR)  \
	    --main-module $(llvm_main_module)     \
	    --syntax-module $(llvm_syntax_module)

### Haskell

haskell_dir           := $(DEFN_DIR)/haskell
haskell_files         := $(ALL_SOURCE_FILES)
haskell_main_module   := WASM-TEXT
haskell_syntax_module := $(haskell_main_module)-SYNTAX
haskell_main_file     := test
haskell_kompiled      := $(haskell_dir)/$(haskell_main_file)-kompiled/definition.kore

build-haskell: $(haskell_kompiled)

$(haskell_kompiled): $(haskell_files)
	$(KOMPILE_HASKELL) $(haskell_main_file).md   \
	    --directory $(haskell_dir) -I $(CURDIR)  \
	    --main-module $(haskell_main_module)     \
	    --syntax-module $(haskell_syntax_module)

# Testing
# -------

TEST  := ./kwasm
CHECK := git --no-pager diff --no-index --ignore-all-space -R

TEST_CONCRETE_BACKEND := llvm
TEST_SYMBOLIC_BACKEND := haskell

KPROVE_MODULE := KWASM-LEMMAS
KPROVE_OPTS   :=

tests/proofs/functions-spec.k.prove: KPROVE_MODULE = FUNCTIONS-LEMMAS
tests/proofs/functions-spec.k.prove: KPROVE_OPTS   += --concrete-rules WASM-DATA.wrap-Positive,WASM-DATA.setRange-Positive,WASM-DATA.getRange-Positive
tests/proofs/memory-spec.k.prove:    KPROVE_OPTS   += --concrete-rules WASM-DATA.wrap-Positive,WASM-DATA.setRange-Positive,WASM-DATA.getRange-Positive
tests/proofs/wrc20-spec.k.prove:     KPROVE_MODULE = WRC20-LEMMAS
tests/proofs/wrc20-spec.k.prove:     KPROVE_OPTS   += --concrete-rules WASM-DATA.wrap-Positive,WASM-DATA.setRange-Positive,WASM-DATA.getRange-Positive,WASM-DATA.get-Existing,WASM-DATA.set-Extend

test: test-execution test-prove test-binary-parser

# Generic Test Harnesses

tests/%.run: tests/% $(llvm_kompiled)
	$(TEST) run --backend $(TEST_CONCRETE_BACKEND) $< > tests/$*.$(TEST_CONCRETE_BACKEND)-out
	$(CHECK) tests/$*.$(TEST_CONCRETE_BACKEND)-out tests/success-$(TEST_CONCRETE_BACKEND).out
	rm -rf tests/$*.$(TEST_CONCRETE_BACKEND)-out

tests/%.run-term: tests/% $(llvm_kompiled)
	$(TEST) run --backend $(TEST_CONCRETE_BACKEND) $< > tests/$*.$(TEST_CONCRETE_BACKEND)-out
	grep --after-context=2 "<instrs>" tests/$*.$(TEST_CONCRETE_BACKEND)-out > tests/$*.$(TEST_CONCRETE_BACKEND)-out-term
	$(CHECK) tests/$*.$(TEST_CONCRETE_BACKEND)-out-term tests/success-k.out
	rm -rf tests/$*.$(TEST_CONCRETE_BACKEND)-out
	rm -rf tests/$*.$(TEST_CONCRETE_BACKEND)-out-term

tests/%.parse: tests/% $(llvm_kompiled)
	$(TEST) kast --backend $(TEST_CONCRETE_BACKEND) $< kast > $@-out
	rm -rf $@-out

tests/%.prove: tests/% $(haskell_kompiled)
	$(TEST) prove --backend $(TEST_SYMBOLIC_BACKEND) $< $(KPROVE_MODULE) --format-failures \
	$(KPROVE_OPTS)

### Execution Tests

test-execution: test-simple

simple_tests         := $(wildcard tests/simple/*.wast)
simple_tests_failing := $(shell cat tests/failing.simple)
simple_tests_passing := $(filter-out $(simple_tests_failing), $(simple_tests))

test-simple: $(simple_tests_passing:=.run)

### Conformance Tests

conformance_tests:=$(wildcard tests/wasm-tests/test/core/*.wast)
unsupported_conformance_tests:=$(patsubst %, tests/wasm-tests/test/core/%, $(shell cat tests/conformance/unsupported-$(TEST_CONCRETE_BACKEND).txt))
unparseable_conformance_tests:=$(patsubst %, tests/wasm-tests/test/core/%, $(shell cat tests/conformance/unparseable.txt))
parseable_conformance_tests:=$(filter-out $(unparseable_conformance_tests), $(conformance_tests))
supported_conformance_tests:=$(filter-out $(unsupported_conformance_tests), $(parseable_conformance_tests))

test-conformance-parse: $(parseable_conformance_tests:=.parse)
test-conformance-supported: $(supported_conformance_tests:=.run-term)

test-conformance: test-conformance-parse test-conformance-supported

### Proof Tests

proof_tests:=$(wildcard tests/proofs/*-spec.k)

test-prove: $(proof_tests:=.prove)

### Binary Parser Test

# TODO pyk is not globally installed. use the poetry-installed pyk
# until binary-parser is migrated to pykwasm
BINARY:=poetry -C pykwasm run python3 binary-parser/test.py

tests/binary/%.wasm: tests/binary/%.wat
	wat2wasm $< --output=$@

.PHONY: pykwasm-poetry-install
pykwasm-poetry-install:
	$(MAKE) -C pykwasm poetry-install

tests/%.wasm.bparse: tests/%.wasm pykwasm-poetry-install
	$(BINARY) $<

binary_parser_tests:=$(wildcard tests/binary/*.wat)

test-binary-parser: $(binary_parser_tests:.wat=.wasm.bparse) test-kwasm-ast

test-kwasm-ast: pykwasm-poetry-install
	poetry -C pykwasm run pytest binary-parser/test_kwasm_ast.py

# Analysis
# --------
json_build := $(haskell_dir)/$(haskell_main_file)-kompiled/parsed.json

$(json_build):
	$(MAKE) build-haskell -B KOMPILE_OPTS="--emit-json"

graph-imports: $(json_build)
	kpyk $(haskell_dir)/$(haskell_main_file)-kompiled graph-imports

# Presentation
# ------------

media: presentations reports

media/%.pdf: TO_FORMAT=beamer
presentations: $(patsubst %.md, %.pdf, $(wildcard media/*-presentation-*.md))

media/201903-report-chalmers.pdf: TO_FORMAT=latex
reports: media/201903-report-chalmers.pdf

media/%.pdf: media/%.md media/citations.md
	cat $^ | pandoc --from markdown-latex_macros --to $(TO_FORMAT) --filter pandoc-citeproc --output $@

media-clean:
	rm media/*.pdf
