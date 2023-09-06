# Define the default target now so that it is always the first target
BUILD_TARGETS = main quantize quantize-stats perplexity embedding vdot train-text-from-scratch convert-llama2c-to-ggml simple save-load-state server embd-input-test gguf llama-bench baby-llama beam-search speculative tests/test-c.o

# Binaries only useful for tests
TEST_TARGETS = tests/test-llama-grammar tests/test-grammar-parser tests/test-double-float tests/test-grad0 tests/test-opt tests/test-quantize-fns tests/test-quantize-perf tests/test-sampling tests/test-tokenizer-0-llama tests/test-tokenizer-0-falcon tests/test-tokenizer-1

# Code coverage output files
COV_TARGETS = *.gcno tests/*.gcno *.gcda tests/*.gcda *.gcov tests/*.gcov lcov-report gcovr-report

ifndef UNAME_S
UNAME_S := $(shell uname -s)
endif

ifndef UNAME_P
UNAME_P := $(shell uname -p)
endif

ifndef UNAME_M
UNAME_M := $(shell uname -m)
endif

# Mac OS + Arm can report x86_64
# ref: https://github.com/ggerganov/whisper.cpp/issues/66#issuecomment-1282546789
ifeq ($(UNAME_S),Darwin)
	ifndef LLAMA_NO_METAL
		LLAMA_METAL := 1
	endif

	ifneq ($(UNAME_P),arm)
		SYSCTL_M := $(shell sysctl -n hw.optional.arm64 2>/dev/null)
		ifeq ($(SYSCTL_M),1)
			# UNAME_P := arm
			# UNAME_M := arm64
			warn := $(warning Your arch is announced as x86_64, but it seems to actually be ARM64. Not fixing that can lead to bad performance. For more info see: https://github.com/ggerganov/whisper.cpp/issues/66\#issuecomment-1282546789)
		endif
	endif
endif

ifneq '' '$(or $(filter clean,$(MAKECMDGOALS)),$(LLAMA_METAL))'
BUILD_TARGETS += metal
endif

default: $(BUILD_TARGETS)

test:
	@echo "Running tests..."
	@for test_target in $(TEST_TARGETS); do \
		if [ "$$test_target" = "tests/test-tokenizer-0-llama" ]; then \
			./$$test_target $(CURDIR)/models/ggml-vocab-llama.gguf; \
		elif [ "$$test_target" = "tests/test-tokenizer-0-falcon" ]; then \
			continue; \
		elif [ "$$test_target" = "tests/test-tokenizer-1" ]; then \
			continue; \
		else \
			./$$test_target; \
		fi; \
	done
	@echo "All tests have been run."

all: $(BUILD_TARGETS) $(TEST_TARGETS)

coverage: ## Run code coverage
	gcov -pb tests/*.cpp

lcov-report: coverage ## Generate lcov report
	mkdir -p lcov-report
	lcov --capture --directory . --output-file lcov-report/coverage.info
	genhtml lcov-report/coverage.info --output-directory lcov-report

gcovr-report: coverage ## Generate gcovr report
	mkdir -p gcovr-report
	gcovr --root . --html --html-details --output gcovr-report/coverage.html

ifdef RISCV_CROSS_COMPILE
CC	:= riscv64-unknown-linux-gnu-gcc
CXX	:= riscv64-unknown-linux-gnu-g++
endif

CCV := $(shell $(CC) --version | head -n 1)
CXXV := $(shell $(CXX) --version | head -n 1)

#
# Compile flags
#

# keep standard at C11 and C++11
# -Ofast tends to produce faster code, but may not be available for some compilers.
ifdef LLAMA_FAST
OPT = -Ofast
else
OPT = -O3
endif
MK_CPPFLAGS = -I. -Icommon
MK_CFLAGS   = $(CPPFLAGS) $(OPT) -std=c11   -fPIC
MK_CXXFLAGS = $(CPPFLAGS) $(OPT) -std=c++11 -fPIC
MK_LDFLAGS  =

ifdef LLAMA_DEBUG
	MK_CFLAGS   += -O0 -g
	MK_CXXFLAGS += -O0 -g
	MK_LDFLAGS  += -g
else
	MK_CPPFLAGS += -DNDEBUG
endif

ifdef LLAMA_SERVER_VERBOSE
	MK_CPPFLAGS += -DSERVER_VERBOSE=$(LLAMA_SERVER_VERBOSE)
endif


ifdef LLAMA_CODE_COVERAGE
	MK_CXXFLAGS += -fprofile-arcs -ftest-coverage -dumpbase ''
endif

ifdef LLAMA_DISABLE_LOGS
	MK_CPPFLAGS += -DLOG_DISABLE_LOGS
endif # LLAMA_DISABLE_LOGS

# warnings
MK_CFLAGS    += -Wall -Wextra -Wpedantic -Wcast-qual -Wdouble-promotion -Wshadow -Wstrict-prototypes -Wpointer-arith \
				-Wmissing-prototypes -Werror=implicit-int -Wno-unused-function
MK_CXXFLAGS  += -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function -Wno-multichar

ifeq '' '$(findstring clang++,$(CXX))'
	# g++ only
	MK_CXXFLAGS += -Wno-format-truncation
endif

# OS specific
# TODO: support Windows
ifneq '' '$(filter $(UNAME_S),Linux Darwin FreeBSD NetBSD OpenBSD Haiku)'
	MK_CFLAGS   += -pthread
	MK_CXXFLAGS += -pthread
endif

# detect Windows
ifneq ($(findstring _NT,$(UNAME_S)),)
	_WIN32 := 1
endif

# library name prefix
ifneq ($(_WIN32),1)
	LIB_PRE := lib
endif

# Dynamic Shared Object extension
ifneq ($(_WIN32),1)
	DSO_EXT := .so
else
	DSO_EXT := .dll
endif

# Windows Sockets 2 (Winsock) for network-capable apps
ifeq ($(_WIN32),1)
	LWINSOCK2 := -lws2_32
endif

ifdef LLAMA_GPROF
	MK_CFLAGS   += -pg
	MK_CXXFLAGS += -pg
endif
ifdef LLAMA_PERF
	MK_CPPFLAGS += -DGGML_PERF
endif

# Architecture specific
# TODO: probably these flags need to be tweaked on some architectures
#       feel free to update the Makefile for your architecture and send a pull request or issue

ifndef RISCV

ifeq ($(UNAME_M),$(filter $(UNAME_M),x86_64 i686 amd64))
	# Use all CPU extensions that are available:
	MK_CFLAGS   += -march=native -mtune=native
	MK_CXXFLAGS += -march=native -mtune=native

	# Usage AVX-only
	#MK_CFLAGS   += -mfma -mf16c -mavx
	#MK_CXXFLAGS += -mfma -mf16c -mavx

	# Usage SSSE3-only (Not is SSE3!)
	#MK_CFLAGS   += -mssse3
	#MK_CXXFLAGS += -mssse3
endif

# The stack is only 16-byte aligned on Windows, so don't let gcc emit aligned moves.
# https://gcc.gnu.org/bugzilla/show_bug.cgi?id=54412
# https://github.com/ggerganov/llama.cpp/issues/2922
ifneq '' '$(findstring mingw,$(shell $(CC) -dumpmachine))'
	MK_CFLAGS   += -Xassembler -muse-unaligned-vector-move
	MK_CXXFLAGS += -Xassembler -muse-unaligned-vector-move
endif

ifneq ($(filter aarch64%,$(UNAME_M)),)
	# Apple M1, M2, etc.
	# Raspberry Pi 3, 4, Zero 2 (64-bit)
	MK_CFLAGS   += -mcpu=native
	MK_CXXFLAGS += -mcpu=native
endif

ifneq ($(filter armv6%,$(UNAME_M)),)
	# Raspberry Pi 1, Zero
	MK_CFLAGS   += -mfpu=neon-fp-armv8 -mfp16-format=ieee -mno-unaligned-access
	MK_CXXFLAGS += -mfpu=neon-fp-armv8 -mfp16-format=ieee -mno-unaligned-access
endif

ifneq ($(filter armv7%,$(UNAME_M)),)
	# Raspberry Pi 2
	MK_CFLAGS   += -mfpu=neon-fp-armv8 -mfp16-format=ieee -mno-unaligned-access -funsafe-math-optimizations
	MK_CXXFLAGS += -mfpu=neon-fp-armv8 -mfp16-format=ieee -mno-unaligned-access -funsafe-math-optimizations
endif

ifneq ($(filter armv8%,$(UNAME_M)),)
	# Raspberry Pi 3, 4, Zero 2 (32-bit)
	MK_CFLAGS   += -mfp16-format=ieee -mno-unaligned-access
	MK_CXXFLAGS += -mfp16-format=ieee -mno-unaligned-access
endif

ifneq ($(filter ppc64%,$(UNAME_M)),)
	POWER9_M := $(shell grep "POWER9" /proc/cpuinfo)
	ifneq (,$(findstring POWER9,$(POWER9_M)))
		MK_CFLAGS   += -mcpu=power9
		MK_CXXFLAGS += -mcpu=power9
	endif
endif

else
	MK_CFLAGS   += -march=rv64gcv -mabi=lp64d
	MK_CXXFLAGS += -march=rv64gcv -mabi=lp64d
endif

ifndef LLAMA_NO_K_QUANTS
	MK_CPPFLAGS += -DGGML_USE_K_QUANTS
	OBJS     += k_quants.o
ifdef LLAMA_QKK_64
	MK_CPPFLAGS += -DGGML_QKK_64
endif
endif

ifndef LLAMA_NO_ACCELERATE
	# Mac OS - include Accelerate framework.
	# `-framework Accelerate` works both with Apple Silicon and Mac Intel
	ifeq ($(UNAME_S),Darwin)
		MK_CPPFLAGS += -DGGML_USE_ACCELERATE
		MK_LDFLAGS  += -framework Accelerate
	endif
endif # LLAMA_NO_ACCELERATE

ifdef LLAMA_MPI
	MK_CPPFLAGS += -DGGML_USE_MPI
	MK_CFLAGS   += -Wno-cast-qual
	MK_CXXFLAGS += -Wno-cast-qual
	OBJS     += ggml-mpi.o
endif # LLAMA_MPI

ifdef LLAMA_OPENBLAS
	MK_CPPFLAGS += -DGGML_USE_OPENBLAS $(shell pkg-config --cflags-only-I openblas)
	MK_CFLAGS   += $(shell pkg-config --cflags-only-other openblas)
	MK_LDFLAGS  += $(shell pkg-config --libs openblas)
endif # LLAMA_OPENBLAS

ifdef LLAMA_BLIS
	MK_CPPFLAGS += -DGGML_USE_OPENBLAS -I/usr/local/include/blis -I/usr/include/blis
	MK_LDFLAGS  += -lblis -L/usr/local/lib
endif # LLAMA_BLIS

ifdef LLAMA_CUBLAS
	MK_CPPFLAGS  += -DGGML_USE_CUBLAS -I/usr/local/cuda/include -I/opt/cuda/include -I$(CUDA_PATH)/targets/x86_64-linux/include
	MK_LDFLAGS   += -lcublas -lculibos -lcudart -lcublasLt -lpthread -ldl -lrt -L/usr/local/cuda/lib64 -L/opt/cuda/lib64 -L$(CUDA_PATH)/targets/x86_64-linux/lib
	OBJS      += ggml-cuda.o
	NVCCFLAGS = --forward-unknown-to-host-compiler -use_fast_math
ifdef LLAMA_CUDA_NVCC
	NVCC = $(LLAMA_CUDA_NVCC)
else
	NVCC = nvcc
endif #LLAMA_CUDA_NVCC
ifdef CUDA_DOCKER_ARCH
	NVCCFLAGS += -Wno-deprecated-gpu-targets -arch=$(CUDA_DOCKER_ARCH)
else
	NVCCFLAGS += -arch=native
endif # CUDA_DOCKER_ARCH
ifdef LLAMA_CUDA_FORCE_DMMV
	NVCCFLAGS += -DGGML_CUDA_FORCE_DMMV
endif # LLAMA_CUDA_FORCE_DMMV
ifdef LLAMA_CUDA_DMMV_X
	NVCCFLAGS += -DGGML_CUDA_DMMV_X=$(LLAMA_CUDA_DMMV_X)
else
	NVCCFLAGS += -DGGML_CUDA_DMMV_X=32
endif # LLAMA_CUDA_DMMV_X
ifdef LLAMA_CUDA_MMV_Y
	NVCCFLAGS += -DGGML_CUDA_MMV_Y=$(LLAMA_CUDA_MMV_Y)
else ifdef LLAMA_CUDA_DMMV_Y
	NVCCFLAGS += -DGGML_CUDA_MMV_Y=$(LLAMA_CUDA_DMMV_Y) # for backwards compatibility
else
	NVCCFLAGS += -DGGML_CUDA_MMV_Y=1
endif # LLAMA_CUDA_MMV_Y
ifdef LLAMA_CUDA_F16
	NVCCFLAGS += -DGGML_CUDA_F16
endif # LLAMA_CUDA_F16
ifdef LLAMA_CUDA_DMMV_F16
	NVCCFLAGS += -DGGML_CUDA_F16
endif # LLAMA_CUDA_DMMV_F16
ifdef LLAMA_CUDA_KQUANTS_ITER
	NVCCFLAGS += -DK_QUANTS_PER_ITERATION=$(LLAMA_CUDA_KQUANTS_ITER)
else
	NVCCFLAGS += -DK_QUANTS_PER_ITERATION=2
endif
#ifdef LLAMA_CUDA_CUBLAS
#	NVCCFLAGS += -DGGML_CUDA_CUBLAS
#endif # LLAMA_CUDA_CUBLAS
ifdef LLAMA_CUDA_CCBIN
	NVCCFLAGS += -ccbin $(LLAMA_CUDA_CCBIN)
endif
ggml-cuda.o: ggml-cuda.cu ggml-cuda.h
	$(NVCC) $(NVCCFLAGS) $(subst -Ofast,-O3,$(CXXFLAGS)) -Wno-pedantic -c $< -o $@
endif # LLAMA_CUBLAS

ifdef LLAMA_CLBLAST

	MK_CPPFLAGS += -DGGML_USE_CLBLAST $(shell pkg-config --cflags-only-I clblast OpenCL)
	MK_CFLAGS   += $(shell pkg-config --cflags-only-other clblast OpenCL)
	MK_CXXFLAGS += $(shell pkg-config --cflags-only-other clblast OpenCL)

	# Mac provides OpenCL as a framework
	ifeq ($(UNAME_S),Darwin)
		MK_LDFLAGS += -lclblast -framework OpenCL
	else
		MK_LDFLAGS += $(shell pkg-config --libs clblast OpenCL)
	endif
	OBJS    += ggml-opencl.o

ggml-opencl.o: ggml-opencl.cpp ggml-opencl.h
	$(CXX) $(CXXFLAGS) -c $< -o $@
endif # LLAMA_CLBLAST

ifdef LLAMA_HIPBLAS
	ROCM_PATH	?= /opt/rocm
	HIPCC	    ?= $(ROCM_PATH)/bin/hipcc
	GPU_TARGETS ?= $(shell $(ROCM_PATH)/llvm/bin/amdgpu-arch)
	LLAMA_CUDA_DMMV_X       ?= 32
	LLAMA_CUDA_MMV_Y        ?= 1
	LLAMA_CUDA_KQUANTS_ITER ?= 2
	MK_CPPFLAGS += -DGGML_USE_HIPBLAS -DGGML_USE_CUBLAS
	MK_LDFLAGS  += -L$(ROCM_PATH)/lib -Wl,-rpath=$(ROCM_PATH)/lib
	MK_LDFLAGS	+= -lhipblas -lamdhip64 -lrocblas
	HIPFLAGS    += $(addprefix --offload-arch=,$(GPU_TARGETS))
	HIPFLAGS    += -DGGML_CUDA_DMMV_X=$(LLAMA_CUDA_DMMV_X)
	HIPFLAGS    += -DGGML_CUDA_MMV_Y=$(LLAMA_CUDA_MMV_Y)
	HIPFLAGS    += -DK_QUANTS_PER_ITERATION=$(LLAMA_CUDA_KQUANTS_ITER)
	HIPFLAGS    += -DCC_TURING=1000000000
ifdef LLAMA_CUDA_FORCE_DMMV
	HIPFLAGS 	+= -DGGML_CUDA_FORCE_DMMV
endif # LLAMA_CUDA_FORCE_DMMV
	OBJS        += ggml-cuda.o
ggml-cuda.o: ggml-cuda.cu ggml-cuda.h
	$(HIPCC) $(CXXFLAGS) $(HIPFLAGS) -x hip -c -o $@ $<
endif # LLAMA_HIPBLAS

ifdef LLAMA_METAL
	MK_CPPFLAGS += -DGGML_USE_METAL
	MK_LDFLAGS  += -framework Foundation -framework Metal -framework MetalKit
	OBJS		+= ggml-metal.o
ifdef LLAMA_METAL_NDEBUG
	MK_CPPFLAGS += -DGGML_METAL_NDEBUG
endif
endif # LLAMA_METAL

ifdef LLAMA_METAL
ggml-metal.o: ggml-metal.m ggml-metal.h
	$(CC) $(CFLAGS) -c $< -o $@
endif # LLAMA_METAL

ifdef LLAMA_MPI
ggml-mpi.o: ggml-mpi.c ggml-mpi.h
	$(CC) $(CFLAGS) -c $< -o $@
endif # LLAMA_MPI

ifndef LLAMA_NO_K_QUANTS
k_quants.o: k_quants.c k_quants.h
	$(CC) $(CFLAGS) -c $< -o $@
endif # LLAMA_NO_K_QUANTS

# combine build flags with cmdline overrides
override CPPFLAGS := $(MK_CPPFLAGS) $(CPPFLAGS)
override CFLAGS   := $(MK_CFLAGS) $(CFLAGS)
override CXXFLAGS := $(MK_CXXFLAGS) $(CXXFLAGS)
override LDFLAGS  := $(MK_LDFLAGS) $(LDFLAGS)

#
# Print build information
#

$(info I llama.cpp build info: )
$(info I UNAME_S:  $(UNAME_S))
$(info I UNAME_P:  $(UNAME_P))
$(info I UNAME_M:  $(UNAME_M))
$(info I CFLAGS:   $(CFLAGS))
$(info I CXXFLAGS: $(CXXFLAGS))
$(info I LDFLAGS:  $(LDFLAGS))
$(info I CC:       $(CCV))
$(info I CXX:      $(CXXV))
$(info )

#
# Build library
#

ggml.o: ggml.c ggml.h ggml-cuda.h
	$(CC)  $(CFLAGS)   -c $< -o $@

ggml-alloc.o: ggml-alloc.c ggml.h ggml-alloc.h
	$(CC)  $(CFLAGS)   -c $< -o $@

OBJS += ggml-alloc.o

llama.o: llama.cpp ggml.h ggml-alloc.h ggml-cuda.h ggml-metal.h llama.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

common.o: common/common.cpp common/common.h build-info.h common/log.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

console.o: common/console.cpp common/console.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

grammar-parser.o: common/grammar-parser.cpp common/grammar-parser.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

libllama.so: llama.o ggml.o $(OBJS)
	$(CXX) $(CXXFLAGS) -shared -fPIC -o $@ $^ $(LDFLAGS)

clean:
	rm -vrf *.o tests/*.o *.so *.dll benchmark-matmult build-info.h *.dot $(COV_TARGETS) $(BUILD_TARGETS) $(TEST_TARGETS)

#
# Examples
#

main: examples/main/main.cpp                                  build-info.h ggml.o llama.o common.o console.o grammar-parser.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(filter-out %.h,$^) -o $@ $(LDFLAGS)
	@echo
	@echo '====  Run ./main -h for help.  ===='
	@echo

simple: examples/simple/simple.cpp                            build-info.h ggml.o llama.o common.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(filter-out %.h,$^) -o $@ $(LDFLAGS)

quantize: examples/quantize/quantize.cpp                      build-info.h ggml.o llama.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(filter-out %.h,$^) -o $@ $(LDFLAGS)

quantize-stats: examples/quantize-stats/quantize-stats.cpp    build-info.h ggml.o llama.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(filter-out %.h,$^) -o $@ $(LDFLAGS)

perplexity: examples/perplexity/perplexity.cpp                build-info.h ggml.o llama.o common.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(filter-out %.h,$^) -o $@ $(LDFLAGS)

embedding: examples/embedding/embedding.cpp                   build-info.h ggml.o llama.o common.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(filter-out %.h,$^) -o $@ $(LDFLAGS)

save-load-state: examples/save-load-state/save-load-state.cpp build-info.h ggml.o llama.o common.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(filter-out %.h,$^) -o $@ $(LDFLAGS)

server: examples/server/server.cpp examples/server/httplib.h examples/server/json.hpp examples/server/index.html.hpp examples/server/index.js.hpp examples/server/completion.js.hpp build-info.h ggml.o llama.o common.o grammar-parser.o $(OBJS)
	$(CXX) $(CXXFLAGS) -Iexamples/server $(filter-out %.h,$(filter-out %.hpp,$^)) -o $@ $(LDFLAGS) $(LWINSOCK2)

$(LIB_PRE)embdinput$(DSO_EXT): examples/embd-input/embd-input.h examples/embd-input/embd-input-lib.cpp build-info.h ggml.o llama.o common.o $(OBJS)
	$(CXX) --shared $(CXXFLAGS) $(filter-out %.h,$(filter-out %.hpp,$^)) -o $@ $(LDFLAGS)


embd-input-test: $(LIB_PRE)embdinput$(DSO_EXT) examples/embd-input/embd-input-test.cpp build-info.h ggml.o llama.o common.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(filter-out %$(DSO_EXT),$(filter-out %.h,$(filter-out %.hpp,$^))) -o $@ $(LDFLAGS) -L. -lembdinput

gguf: examples/gguf/gguf.cpp ggml.o llama.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(filter-out %.h,$^) -o $@ $(LDFLAGS)

train-text-from-scratch: examples/train-text-from-scratch/train-text-from-scratch.cpp ggml.o llama.o common.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(filter-out %.h,$^) -o $@ $(LDFLAGS)

convert-llama2c-to-ggml: examples/convert-llama2c-to-ggml/convert-llama2c-to-ggml.cpp ggml.o llama.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(filter-out %.h,$^) -o $@ $(LDFLAGS)

llama-bench: examples/llama-bench/llama-bench.cpp build-info.h ggml.o llama.o common.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(filter-out %.h,$^) -o $@ $(LDFLAGS)

baby-llama: examples/baby-llama/baby-llama.cpp ggml.o llama.o common.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(filter-out %.h,$^) -o $@ $(LDFLAGS)

beam-search: examples/beam-search/beam-search.cpp build-info.h ggml.o llama.o common.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(filter-out %.h,$^) -o $@ $(LDFLAGS)

speculative: examples/speculative/speculative.cpp build-info.h ggml.o llama.o common.o grammar-parser.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(filter-out %.h,$^) -o $@ $(LDFLAGS)

ifdef LLAMA_METAL
metal: examples/metal/metal.cpp ggml.o $(OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)
endif

build-info.h: $(wildcard .git/index) scripts/build-info.sh
	@sh scripts/build-info.sh > $@.tmp
	@if ! cmp -s $@.tmp $@; then \
		mv $@.tmp $@; \
	else \
		rm $@.tmp; \
	fi

#
# Tests
#

tests: $(TEST_TARGETS)

benchmark-matmult: examples/benchmark/benchmark-matmult.cpp build-info.h ggml.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(filter-out %.h,$^) -o $@ $(LDFLAGS)
	./$@

vdot: pocs/vdot/vdot.cpp ggml.o $(OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

tests/test-llama-grammar: tests/test-llama-grammar.cpp build-info.h ggml.o common.o grammar-parser.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(filter-out %.h,$^) -o $@ $(LDFLAGS)

tests/test-grammar-parser: tests/test-grammar-parser.cpp build-info.h ggml.o llama.o common.o grammar-parser.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(filter-out %.h,$^) -o $@ $(LDFLAGS)

tests/test-double-float: tests/test-double-float.cpp build-info.h ggml.o llama.o common.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(filter-out %.h,$^) -o $@ $(LDFLAGS)

tests/test-grad0: tests/test-grad0.cpp build-info.h ggml.o llama.o common.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(filter-out %.h,$^) -o $@ $(LDFLAGS)

tests/test-opt: tests/test-opt.cpp build-info.h ggml.o llama.o common.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(filter-out %.h,$^) -o $@ $(LDFLAGS)

tests/test-quantize-fns: tests/test-quantize-fns.cpp build-info.h ggml.o llama.o common.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(filter-out %.h,$^) -o $@ $(LDFLAGS)

tests/test-quantize-perf: tests/test-quantize-perf.cpp build-info.h ggml.o llama.o common.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(filter-out %.h,$^) -o $@ $(LDFLAGS)

tests/test-sampling: tests/test-sampling.cpp build-info.h ggml.o llama.o common.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(filter-out %.h,$^) -o $@ $(LDFLAGS)

tests/test-tokenizer-0-falcon: tests/test-tokenizer-0-falcon.cpp build-info.h ggml.o llama.o common.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(filter-out %.h,$^) -o $@ $(LDFLAGS)

tests/test-tokenizer-0-llama: tests/test-tokenizer-0-llama.cpp build-info.h ggml.o llama.o common.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(filter-out %.h,$^) -o $@ $(LDFLAGS)

tests/test-tokenizer-1: tests/test-tokenizer-1.cpp build-info.h ggml.o llama.o common.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(filter-out %.h,$^) -o $@ $(LDFLAGS)

tests/test-c.o: tests/test-c.c llama.h
	$(CC) $(CFLAGS) -c $(filter-out %.h,$^) -o $@
